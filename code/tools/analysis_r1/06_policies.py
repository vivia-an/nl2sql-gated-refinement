"""Q3 + Q5: offline replay of selection / routing policies on the 1101 HR branching trees (and 3-system voting on HT-300).
Everything is a REPLAY on cached candidates: no new LLM calls. Candidates inside a tree are NOT independent samples (a revision is conditioned
on its parent SQL and critic feedback; T=0.7 siblings share the parent), so vote results are optimistic/pessimistic in unknown directions;
flagged in the report. Cost = sum of stored per-node calls (root = M1+gen+critic = 3, revision = rev+critic = 2); 'nocrit' = without critic calls."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
from execdata import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

Qs = load_trees(); miss = annotate(Qs)
dbs = np.array([q.db for q in Qs]); n = len(Qs)
P_IN37, P_OUT37, P_IN38, P_OUT38 = 2.0, 8.0, 12.0, 36.0


def node_yuan(nd):
    pt, ct = nd["prompt_tokens"], nd["completion_tokens"]
    base = (pt * P_IN37 + ct * P_OUT37) / 1e6
    if nd["tier"] == 1 and nd["action"] != "root":
        tot = nd["tok_actor"] + nd["tok_crit"] + nd.get("tok_m1", 0)
        f = nd["tok_actor"] / tot if tot else 0.0
        base += f * (pt * (P_IN38 - P_IN37) + ct * (P_OUT38 - P_OUT37)) / 1e6
    return base


def cost_of(samples):
    """samples: list of (node, multiplicity). returns calls_with_critic, calls_no_critic, yuan."""
    c = cn = y = 0.0
    for nd, m in samples:
        if nd["action"] == "root":
            c += 3; cn += 2; y += node_yuan(nd)
        else:
            c += 2 * m; cn += 1 * m; y += m * node_yuan(nd)
    return c, cn, y


def pick_vote(pool, root):
    """pool: list of (node, mult). plurality of result keys (errors never win unless everything is an error), tie -> best critic score, then root."""
    cnt = collections.defaultdict(float); best = {}
    for nd, m in pool:
        k = nd["rk"]
        cnt[k] += m
        if k not in best or (nd["score"], -nd["k"]) > (best[k]["score"], -best[k]["k"]):
            best[k] = nd
    keys = [k for k in cnt if k[0] != "E"] or list(cnt)
    k = max(keys, key=lambda k: (cnt[k], best[k]["score"], best[k]["id"] == 0))
    return best[k]


def pick_score(pool, root):
    return max((nd for nd, _ in pool), key=lambda nd: (nd["score"], -nd["k"]))


def d1(q, actions):
    out = []
    for c in q.depth1():
        if c["action"] in actions:
            out.append((c, c["weight"]))
    return out


results = {}


def evalp(name, fn):
    ex = np.zeros(n); calls = np.zeros(n); cnc = np.zeros(n); yuan = np.zeros(n); ab = np.zeros(n)
    for i, q in enumerate(Qs):
        final, samples, abstain = fn(q)
        ex[i] = 0 if abstain else final["ex"]; ab[i] = abstain
        c, cn, y = cost_of(samples); calls[i], cnc[i], yuan[i] = c, cn, y
    results[name] = {"ex_vec": ex, "calls": calls, "calls_nocrit": cnc, "yuan": yuan, "abstain": ab}
    return results[name]


root_s = lambda q: [(q.root, 1)]
# --- policies
evalp("root only", lambda q: (q.root, root_s(q), False))


def p0(theta, tmax=3, tie="low_k"):
    def f(q):
        r = q.p0(theta_acc=theta, t_max=tmax, tie_prefer=tie)
        return r["final"], [(x, 1) for x in r["nodes"]], r["abstain"]
    return f


evalp("P0 (theta 8.5, patent)", p0(8.5))
for th in (7.5, 9.0, 9.5, 10.0):
    evalp("P0 theta_acc %.1f" % th, p0(th))
evalp("P0 theta 10, tmax 1", p0(10.0, 1))
evalp("P0 theta 10, ties prefer revision", p0(10.0, 3, "high_k"))
evalp("P0 theta 8.5, no abstain", lambda q: (lambda r: (r["final"], [(x, 1) for x in r["nodes"]], False))(q.p0()))


def execgate(statuses, tmax=3):
    def f(q):
        if q.root["status"] not in statuses:
            return q.root, root_s(q), False
        r = q.p0(theta_acc=10.01, t_max=tmax, route_human=False)      # revise whatever the score, chain up to tmax, best by score
        # stop the chain as soon as the status is executable & nonempty: emulate by walking
        cur = q.root; nodes = [cur]
        while cur["status"] in statuses and cur["k"] < tmax:
            ch = q.first_child(cur["id"], "revise")
            if ch is None: break
            nodes.append(ch); cur = ch
        return cur, [(x, 1) for x in nodes], False
    return f


evalp("ExecGate: revise only if exec error/timeout (stop when executable)", execgate({"err_syntax", "err_identifier", "timeout"}))
evalp("ExecGate: revise if error/timeout/empty", execgate({"err_syntax", "err_identifier", "timeout", "ok_empty"}))


def agreegate(then="vote3", second_extra=True):
    """root + revise T0 (2 extra generation... 1 call w/o critic). agree -> accept. disagree -> add revise T.7 sample and vote of 3 (ties: critic score)."""
    def f(q):
        revs = q.children(0, "revise")
        if not revs:
            return q.root, root_s(q), False
        r0 = revs[0]
        if r0["rk"] == q.root["rk"]:
            return q.root, [(q.root, 1), (r0, 1)], False
        pool = [(q.root, 1)] + [(r0, 1)]
        if then == "vote3":
            # the T=0.7 sample: second revise child if distinct, else it equals r0 (weight 2 -> T.7 sample = r0's SQL)
            if len(revs) > 1:
                pool.append((revs[1], 1))
            else:
                pool.append((r0, 1))
            return pick_vote(pool, q.root), pool, False
        if then == "escalate":
            e = q.first_child(0, "escalate")
            if e is None: return pick_score(pool, q.root), pool, False
            return e, pool + [(e, 1)], False
        if then == "score":
            return pick_score(pool, q.root), pool, False
    return f


evalp("AgreeGate: root + 1 revision; agree -> accept else + T0.7 sample, vote3 (tie: critic)", agreegate("vote3"))
evalp("AgreeGate: disagree -> take critic-best of (root, rev)", agreegate("score"))
evalp("AgreeGate: disagree -> take escalate (qwen3.8-max reviser) result", agreegate("escalate"))

for nm, acts in (("A root+2 revise", ("revise",)), ("B root+2 revise+2 re_retrieve", ("revise", "re_retrieve")), ("C root+all depth-1 (incl. escalate)", ("revise", "re_retrieve", "escalate")),
                 ("D root+revise+escalate", ("revise", "escalate"))):
    evalp("Vote(exec-result plurality) pool " + nm, lambda q, acts=acts: (lambda pool: (pick_vote(pool, q.root), pool, False))([(q.root, 1)] + d1(q, acts)))
    evalp("Critic-argmax pool " + nm, lambda q, acts=acts: (lambda pool: (pick_score(pool, q.root), pool, False))([(q.root, 1)] + d1(q, acts)))
    evalp("Oracle-in-pool " + nm, lambda q, acts=acts: (lambda pool: (max((x for x, _ in pool), key=lambda x: x["ex"]), pool, False))([(q.root, 1)] + d1(q, acts)))
evalp("Oracle-in-pool whole tree", lambda q: (max(q.nodes, key=lambda x: x["ex"]), [(x, 1) for x in q.nodes], False))
# P0 for error roots, agreement gate for the rest
def combo(q):
    if q.root["status"] in ("err_syntax", "err_identifier", "timeout"):
        return execgate({"err_syntax", "err_identifier", "timeout"})(q)
    return agreegate("vote3")(q)
evalp("Combo: ExecGate on errors + AgreeGate(vote3) on the rest", combo)


# --- learned selector among depth-1 pool C (LODO)
def node_feats(q, nd, pool):
    cnt = collections.Counter(); tot = 0
    for x, m in pool:
        cnt[x["rk"]] += m; tot += m
    s = nd["state"]
    return [nd["syntax"], nd["semantic"], nd["efficiency"], nd["score"], float(nd["status"] == "ok_nonempty"), float(nd["status"] == "ok_empty"), float(nd["status"].startswith("err") or nd["status"] == "timeout"),
            cnt[nd["rk"]] / tot, float(nd["action"] == "root"), float(nd["action"] == "revise"), float(nd["action"] == "re_retrieve"), float(nd["tier"] == 1), s[10], s[11], s[12],
            float(nd["rk"] == q.root["rk"])]


def learned(pool_actions, model="lr"):
    X, Y, G, I = [], [], [], []
    for qi, q in enumerate(Qs):
        pool = [(q.root, 1)] + d1(q, pool_actions)
        for nd, m in pool:
            X.append(node_feats(q, nd, pool)); Y.append(nd["ex"]); G.append(q.db); I.append((qi, nd["id"]))
    X = np.array(X); Y = np.array(Y); G = np.array(G)
    P = np.zeros(len(Y))
    for d in np.unique(G):
        tr, te = G != d, G == d
        m = LogisticRegression(C=1.0, max_iter=2000) if model == "lr" else HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.05, random_state=0)
        if model == "lr":
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            m.fit((X[tr] - mu) / sd, Y[tr]); P[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
        else:
            m.fit(X[tr], Y[tr]); P[te] = m.predict_proba(X[te])[:, 1]
    best = {}
    for (qi, nid), p in zip(I, P):
        if qi not in best or p > best[qi][0] + 1e-12 or (abs(p - best[qi][0]) <= 1e-12 and nid == 0):
            best[qi] = (p, nid)
    return best


for nm, acts in (("A", ("revise",)), ("C", ("revise", "re_retrieve", "escalate"))):
    for model in ("lr", "hgb"):
        best = learned(acts, model)
        evalp("Learned selector (%s, LODO) pool %s" % (model, nm), lambda q, best=best, acts=acts: (lambda nid: (q.by_id[nid], [(q.root, 1)] + d1(q, acts), False))(best[Qs.index(q)][1]))

# --- summarise
base = results["P0 (theta 8.5, patent)"]["ex_vec"]; rootv = results["root only"]["ex_vec"]
nbf = dbs != "beer_factory"
out = []
for nm, r in results.items():
    ex = r["ex_vec"]
    e = cluster_boot(dbs, lambda idx: ex[idx].mean(), B=1000)
    dr = cluster_boot(dbs, lambda idx: (ex[idx] - rootv[idx]).mean(), B=1000)
    dp = cluster_boot(dbs, lambda idx: (ex[idx] - base[idx]).mean(), B=1000)
    out.append({"policy": nm, "EX": e, "d_vs_root": dr, "d_vs_P0": dp, "calls": float(r["calls"].mean()), "calls_nocrit": float(r["calls_nocrit"].mean()), "yuan_per_q": float(r["yuan"].mean()),
                "EX_excl_beer_factory": float(ex[nbf].mean()), "d_vs_P0_excl_beer_pp": 100 * float((ex[nbf] - base[nbf]).mean()), "d_vs_root_excl_beer_pp": 100 * float((ex[nbf] - rootv[nbf]).mean()),
                "wins_vs_P0": int(((ex == 1) & (base == 0)).sum()), "losses_vs_P0": int(((ex == 0) & (base == 1)).sum()), "abstain": float(r["abstain"].mean())})
dump("q3q5_policies.json", {"policies": out, "missing_exec": miss})
for o in out:
    print("%-95s EX %.3f [%.3f,%.3f] dRoot %+.1f dP0 %+.1f [%+.1f,%+.1f] calls %.2f nocrit %.2f yuan %.4f | exclBF dP0 %+.1f dRoot %+.1f | W/L vs P0 %d/%d" % (
        o["policy"], o["EX"][0], o["EX"][1], o["EX"][2], 100 * o["d_vs_root"][0], 100 * o["d_vs_P0"][0], 100 * o["d_vs_P0"][1], 100 * o["d_vs_P0"][2], o["calls"], o["calls_nocrit"], o["yuan_per_q"],
        o["d_vs_P0_excl_beer_pp"], o["d_vs_root_excl_beer_pp"], o["wins_vs_P0"], o["losses_vs_P0"]))
