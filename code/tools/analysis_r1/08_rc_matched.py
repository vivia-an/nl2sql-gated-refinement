"""Q2/Q5: risk-coverage at MATCHED coverage (ties broken randomly, averaged over 20 shuffles), db-clustered CI.
HR: routing of the ROOT answer by V3 score / execution agreement with depth-1 revisions (needs 5 extra gen calls) / LODO-LR combination.
HT: V3 score of full_fixed final vs result agreement across 3 systems (zs, zs_ours, full_fixed) vs both."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
from execdata import *
from sklearn.linear_model import LogisticRegression

COV = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]


def rc_at(sig, y, covs=COV, reps=20, seed=1):
    rng = np.random.RandomState(seed); n = len(y); out = {c: [] for c in covs}
    for _ in range(reps):
        o = np.lexsort((rng.rand(n), -sig))
        for c in covs:
            k = int(round(c * n)); out[c].append(y[o[:k]].mean())
    return {c: float(np.mean(v)) for c, v in out.items()}


def boot(dbs, sig, y, B=500, seed=SEED):
    rng = np.random.RandomState(seed); u = np.unique(dbs); mem = {d: np.where(dbs == d)[0] for d in u}
    acc = {c: [] for c in COV}
    for b in range(B):
        idx = np.concatenate([mem[d] for d in rng.choice(u, len(u))])
        r = rc_at(sig[idx], y[idx], reps=3, seed=b)
        for c in COV: acc[c].append(r[c])
    return {c: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for c, v in acc.items()}


def aurc(sig, y, reps=30, seed=2):
    rng = np.random.RandomState(seed); n = len(y); v = []
    for _ in range(reps):
        o = np.lexsort((rng.rand(n), -sig)); v.append((1 - np.cumsum(y[o]) / np.arange(1, n + 1)).mean())
    return float(np.mean(v))


res = {}
# ---- HR
Qs = load_trees(); annotate(Qs)
dbs = np.array([q.db for q in Qs]); y = np.array([q.root["ex"] for q in Qs])
def wfrac(q, nodes):
    tot = sum(n["weight"] for n in nodes); return sum(n["weight"] for n in nodes if n["rk"] == q.root["rk"]) / tot
v3 = np.array([q.root["score"] for q in Qs]); ag = np.array([wfrac(q, q.depth1()) for q in Qs]); agt = np.array([wfrac(q, q.nodes[1:]) for q in Qs])
ne = np.array([float(q.root["status"] == "ok_nonempty") for q in Qs])
F = np.column_stack([v3, ag, ne]); p = np.zeros(len(y))
for d in np.unique(dbs):
    tr, te = dbs != d, dbs == d
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
    m = LogisticRegression(C=1.0, max_iter=500).fit((F[tr] - mu) / sd, y[tr]); p[te] = m.predict_proba((F[te] - mu) / sd)[:, 1]
sigs = {"V3 score (root; +1 critic call)": v3, "agreement root vs 5 depth-1 nodes (+~9 calls)": ag, "agreement root vs whole tree (+45 calls; reference)": agt, "LR(V3, agree, nonempty) LODO": p, "oracle": y.astype(float)}
hr = {}
for nm, s in sigs.items():
    hr[nm] = {"EX_at_cov": rc_at(s, y), "ci": boot(dbs, s, y) if nm != "oracle" else None, "AURC": aurc(s, y)}
hr["random"] = {"EX_overall": float(y.mean()), "AURC": float(1 - y.mean())}
res["HR_root"] = hr
# excluding beer_factory (the quoting artefact)
m = dbs != "beer_factory"
res["HR_root_excl_beer"] = {nm: {"EX_at_cov": rc_at(s[m], y[m]), "AURC": aurc(s[m], y[m])} for nm, s in sigs.items()}
res["HR_root_excl_beer"]["random"] = {"EX_overall": float(y[m].mean())}
# ---- HT
zb = by_qid(jl(HT_ZS_B1 + "/preds.jsonl")); zo = by_qid(jl(HT_ZS + "/preds.jsonl")); ff = by_qid(jl(HT_RUN + "/preds.jsonl"))
qs = sorted(ff); d = np.array([ff[q]["db_id"] for q in qs]); yy = np.array([ff[q]["ex"] for q in qs])
K = lambda q, s: key(ff[q]["db_id"], s)
nd = np.array([len({K(q, zb[q]["final_sql"]), K(q, zo[q]["final_sql"]), K(q, ff[q]["final_sql"])}) for q in qs])
agr3 = 4.0 - nd           # 3 = all agree, 2 = two distinct, 1 = all differ
agr2 = np.array([float(K(q, zo[q]["final_sql"]) == K(q, ff[q]["final_sql"])) for q in qs])
v3f = np.array([ff[q]["critic"]["score"] for q in qs])
comb = agr3 * 100 + v3f
sg = {"V3 score of full_fixed final": v3f, "agreement across 3 systems (zs, zs_ours, full)": agr3 + 1e-6 * v3f * 0, "agreement(3) then V3 as tie-break": comb, "agreement zs_ours vs full only": agr2, "oracle": yy.astype(float)}
ht = {}
for nm, s in sg.items():
    ht[nm] = {"EX_at_cov": rc_at(s, yy), "ci": boot(d, s, yy) if nm != "oracle" else None, "AURC": aurc(s, yy)}
ht["random"] = {"EX_overall": float(yy.mean()), "AURC": float(1 - yy.mean())}
res["HT_final"] = ht
res["HT_AUROC"] = {nm: cluster_boot(d, lambda idx, s=s: auroc(s[idx], yy[idx]), B=1000) for nm, s in sg.items() if nm != "oracle"}
dump("q2_rc_matched.json", res)
for part in ("HR_root", "HR_root_excl_beer", "HT_final"):
    print(part)
    for nm, v in res[part].items():
        if "EX_at_cov" in v:
            print("  %-58s AURC %.3f | " % (nm, v["AURC"]) + " ".join("%.0f%%:%.3f" % (100 * c, e) for c, e in v["EX_at_cov"].items()))
        else:
            print("  ", nm, v)
print(res["HT_AUROC"])
