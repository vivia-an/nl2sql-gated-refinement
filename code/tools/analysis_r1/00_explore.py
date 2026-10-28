import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
hp = jl(HR_RUN + "/preds.jsonl"); hc = jl(HR_RUN + "/cands.jsonl"); tr = jl(TREES)
print("HR preds", len(hp), "cands", len(hc), "trees", len(tr))
P = by_qid(hp); T = {t["qid"]: t for t in tr}
print("pred keys with ex None:", sum(1 for r in hp if r["ex"] is None))
print(collections.Counter(r["error_class"] for r in hp))
print(collections.Counter(r["auto_class"] for r in hp))
print(collections.Counter(tuple(r["actions"]) for r in hp).most_common(8))
# root consistency
roots = {c["qid"]: c for c in hc if c["cand_idx"] == 0}
same_sql = sum(1 for q, c in roots.items() if T[q]["nodes"][0]["sql"].strip() == c["sql"].strip())
print("root sql same cands vs trees:", same_sql, "/", len(roots))
print("root ex: cands %.4f trees %.4f" % (np.mean([c["ex"] for c in roots.values()]), np.mean([T[q]["nodes"][0]["ex"] for q in roots])))
dif = [(q, roots[q]["ex"], T[q]["nodes"][0]["ex"]) for q in roots if roots[q]["ex"] != T[q]["nodes"][0]["ex"]]
print("root ex disagreements", len(dif), dif[:10])
print("keys cand", list(hc[0].keys()))
print(collections.Counter(len([c for c in hc if c["qid"] == q]) for q in list(P)[:200]))
# per db n and EX
for d in sorted(set(r["db_id"] for r in hp)):
    rr = [r for r in hp if r["db_id"] == d]
    print(d, len(rr), "EX %.3f" % np.mean([r["ex"] for r in rr]), "treeroot %.3f" % np.mean([T[r["qid"]]["nodes"][0]["ex"] for r in rr]))
ht = jl(HT_RUN + "/preds.jsonl"); htc = jl(HT_RUN + "/cands.jsonl")
print("HT preds", len(ht), "cands", len(htc), collections.Counter(r["error_class"] for r in ht))
print(collections.Counter(r["auto_class"] for r in ht))
print(ht[0].keys())
