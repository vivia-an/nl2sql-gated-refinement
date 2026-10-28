"""Q1: oracle headroom, harm, per-db and depth profile on the 1101 HR branching trees + live HR run. No LLM."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
from trees import *

Qs = load_trees()
live = by_qid(jl(HR_RUN + "/preds.jsonl"))
cands = collections.defaultdict(list)
for c in jl(HR_RUN + "/cands.jsonl"):
    cands[c["qid"]].append(c)
dbs = np.array([q.db for q in Qs])
n = len(Qs)
res = {}


def ci(vec):
    vec = np.asarray(vec, float)
    return cluster_boot(dbs, lambda idx: vec[idx].mean())


rows = []
root_ex = np.array([q.root["ex"] for q in Qs])
sets = {
    "root only": lambda q: [q.root],
    "P0 chain (root + serial T=0 revisions, <=3)": lambda q: q.p0_chain_nodes(),
    "root + depth-1 revise (T=0 and T=0.7)": lambda q: [q.root] + q.children(0, "revise"),
    "root + depth-1 re_retrieve": lambda q: [q.root] + q.children(0, "re_retrieve"),
    "root + depth-1 escalate (qwen3.8-max reviser)": lambda q: [q.root] + q.children(0, "escalate"),
    "root + all depth-1 (<=5 nodes)": lambda q: [q.root] + q.depth1(),
    "root + all depth<=2": lambda q: [q.root] + [x for x in q.nodes if 1 <= x["k"] <= 2],
    "whole tree (<=23.5 nodes/q)": lambda q: q.nodes,
    "whole tree minus escalate-lineage (tier-1 model only)": lambda q: [x for x in q.nodes if x["tier"] == 0],
}
orc = {}
for name, fn in sets.items():
    v = np.array([q.oracle(fn(q)) for q in Qs])
    orc[name] = v
    e, lo, hi = ci(v)
    cost = np.mean([sum(x["calls"] for x in fn(q)) for q in Qs])
    rows.append({"set": name, "oracle_EX": e, "lo": lo, "hi": hi, "gain_pp_vs_root": 100 * (e - root_ex.mean()), "mean_calls_if_all_generated": cost,
                 "mean_nodes": float(np.mean([len(fn(q)) for q in Qs]))})
res["oracle_sets"] = rows

# depth profile: cumulative oracle by depth, first-correct depth among wrong roots
wrong = [q for q in Qs if q.root["ex"] == 0]
first = collections.Counter()
first_act = collections.Counter()
for q in wrong:
    cs = sorted([x for x in q.nodes if x["ex"] == 1], key=lambda x: (x["k"], x["id"]))
    if not cs:
        first["none"] += 1
    else:
        first[cs[0]["k"]] += 1
        first_act[(cs[0]["k"], cs[0]["action"], cs[0]["tier"])] += 1
res["wrong_roots"] = len(wrong)
res["first_correct_depth_among_wrong_roots"] = {str(k): v for k, v in first.items()}
res["first_correct_by_action"] = {str(k): v for k, v in first_act.items()}
# per depth: share of correct nodes among nodes of wrong-root questions (weight-agnostic)
dp = {}
for d in (1, 2, 3):
    xs = [x["ex"] for q in wrong for x in q.nodes if x["k"] == d]
    dp[d] = {"nodes": len(xs), "share_correct": float(np.mean(xs))}
res["share_correct_nodes_wrong_roots_by_depth"] = dp
# by action for wrong roots, depth-1 only: P(node correct)
byact = {}
for act in ("revise", "re_retrieve", "escalate"):
    xs = [(x["ex"], x["weight"]) for q in wrong for x in q.children(0, act)]
    byact[act] = {"nodes": len(xs), "weighted_share_correct": float(sum(e * w for e, w in xs) / max(1, sum(w for _, w in xs)))}
right = [q for q in Qs if q.root["ex"] == 1]
byact_r = {}
for act in ("revise", "re_retrieve", "escalate"):
    xs = [(x["ex"], x["weight"]) for q in right for x in q.children(0, act)]
    byact_r[act] = {"nodes": len(xs), "weighted_share_correct": float(sum(e * w for e, w in xs) / max(1, sum(w for _, w in xs)))}
res["depth1_by_action_wrong_root"] = byact
res["depth1_by_action_right_root"] = byact_r
# 'harm' potential: root right but every revise child wrong / some child wrong
res["right_roots"] = len(right)
res["right_root_with_any_depth1_wrong"] = float(np.mean([any(x["ex"] == 0 for x in q.depth1()) for q in right]))
res["right_root_forced_revise_T0_wrong"] = float(np.mean([(q.first_child(0, "revise") or {"ex": 1})["ex"] == 0 for q in right]))
res["wrong_root_forced_revise_T0_right"] = float(np.mean([(q.first_child(0, "revise") or {"ex": 0})["ex"] == 1 for q in wrong]))
res["wrong_root_forced_escalate_right"] = float(np.mean([(q.first_child(0, "escalate") or {"ex": 0})["ex"] == 1 for q in wrong]))
res["right_root_forced_escalate_wrong"] = float(np.mean([(q.first_child(0, "escalate") or {"ex": 1})["ex"] == 0 for q in right]))

# P0 replay vs live
rp = [q.p0() for q in Qs]
rp_ex = np.array([r["ex"] for r in rp]); rp_calls = np.array([r["calls"] for r in rp]); rp_ab = np.array([r["abstain"] for r in rp])
live_ex = np.array([live[q.qid]["ex"] for q in Qs]); live_calls = np.array([live[q.qid]["calls"] for q in Qs])
live_root = np.array([(cands[q.qid][0]["ex"]) for q in Qs])
res["replay_P0"] = {"EX_all": ci(rp_ex), "calls": float(rp_calls.mean()), "abstain_rate": float(rp_ab.mean()), "n_revised": int((np.array([len(r['nodes']) for r in rp]) > 1).sum())}
res["live_P0"] = {"EX_all": ci(live_ex), "calls": float(live_calls.mean()), "root_EX_live": ci(live_root), "abstain": int(sum(live[q.qid]["abstained"] for q in Qs))}
res["tree_root_EX"] = ci(root_ex)


def mcn(a, b):
    """a,b binary vectors (same items). returns (a only right, b only right)"""
    return int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())


def paired_ci(a, b):
    d = (np.asarray(a, float) - np.asarray(b, float))
    return cluster_boot(dbs, lambda idx: d[idx].mean())


res["live_final_vs_live_root"] = {"gain_root_wrong_final_right": int(((live_root == 0) & (live_ex == 1)).sum()), "harm_root_right_final_wrong": int(((live_root == 1) & (live_ex == 0)).sum()),
                                  "delta_EX": paired_ci(live_ex, live_root), "n_revised": int(sum(1 for q in Qs if len(cands[q.qid]) > 1))}
res["replay_P0_vs_tree_root"] = {"gain": int(((root_ex == 0) & (rp_ex == 1)).sum()), "harm": int(((root_ex == 1) & (rp_ex == 0)).sum()), "delta_EX": paired_ci(rp_ex, root_ex)}
# harm decomposition in the replay: abstain vs wrong revision
harm_ab = int(((root_ex == 1) & rp_ab).sum())
res["replay_P0_harm_by_abstain"] = harm_ab
# ideal gate (oracle critic): accept if root right, otherwise take the P0 chain best-correct
ideal = np.array([1 if q.root["ex"] == 1 else q.oracle(q.p0_chain_nodes()) for q in Qs])
res["ideal_gate_P0chain"] = {"EX": ci(ideal), "gain_pp_vs_replay_P0": 100 * (ideal.mean() - rp_ex.mean())}

# per db
perdb = []
for d in sorted(set(dbs)):
    ix = np.where(dbs == d)[0]
    perdb.append({"db": d, "n": len(ix), "root_EX": float(root_ex[ix].mean()), "P0_replay_EX": float(rp_ex[ix].mean()), "live_EX": float(live_ex[ix].mean()),
                  "oracle_chain": float(orc["P0 chain (root + serial T=0 revisions, <=3)"][ix].mean()), "oracle_depth1": float(orc["root + all depth-1 (<=5 nodes)"][ix].mean()),
                  "oracle_tree": float(orc["whole tree (<=23.5 nodes/q)"][ix].mean()),
                  "oracle_tier1_only": float(orc["whole tree minus escalate-lineage (tier-1 model only)"][ix].mean()),
                  "gain_replay": int(((root_ex[ix] == 0) & (rp_ex[ix] == 1)).sum()), "harm_replay": int(((root_ex[ix] == 1) & (rp_ex[ix] == 0)).sum()),
                  "root_exec_err": float(np.mean([Qs[i].root["status"].startswith("err") or Qs[i].root["status"] == "timeout" for i in ix])),
                  "root_empty": float(np.mean([Qs[i].root["status"] == "ok_empty" for i in ix])),
                  "root_score_ge_9.5": float(np.mean([Qs[i].root["score"] >= 9.5 for i in ix]))})
res["per_db"] = perdb

# where does P0 gain come from: root status of questions where replay fixes the answer
fix = [(q.root["status"]) for q, r in zip(Qs, rp) if q.root["ex"] == 0 and r["ex"] == 1]
res["P0_gains_by_root_status"] = dict(collections.Counter(fix))
allst = collections.Counter(q.root["status"] for q in Qs if q.root["ex"] == 0)
res["wrong_roots_by_status"] = dict(allst)
# leave-beer_factory-out sensitivity
mask = dbs != "beer_factory"
res["excl_beer_factory"] = {"root_EX": float(root_ex[mask].mean()), "P0_replay_EX": float(rp_ex[mask].mean()), "delta_pp": 100 * float(rp_ex[mask].mean() - root_ex[mask].mean()),
                            "oracle_chain": float(orc["P0 chain (root + serial T=0 revisions, <=3)"][mask].mean()), "oracle_tree": float(orc["whole tree (<=23.5 nodes/q)"][mask].mean())}
dump("q1_headroom.json", res)
print(json.dumps(res, indent=1, default=lambda o: float(o) if hasattr(o, "item") else str(o)))
