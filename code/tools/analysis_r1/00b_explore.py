import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
hp = jl(HR_RUN + "/preds.jsonl"); hc = jl(HR_RUN + "/cands.jsonl"); tr = jl(TREES)
T = {t["qid"]: t for t in tr}; roots = {c["qid"]: c for c in hc if c["cand_idx"] == 0}
for d in sorted(set(r["db_id"] for r in hp)):
    qs = [q for q in roots if roots[q]["db_id"] == d]
    same = [q for q in qs if T[q]["nodes"][0]["sql"].strip() == roots[q]["sql"].strip()]
    exs = [(roots[q]["ex"], T[q]["nodes"][0]["ex"]) for q in same]
    print(d, len(qs), "same-sql", len(same), "ex agree among same", sum(a == b for a, b in exs), "live-root EX %.3f tree-root EX %.3f" % (np.mean([roots[q]["ex"] for q in qs]), np.mean([T[q]["nodes"][0]["ex"] for q in qs])),
          "tree exec_ok %.3f" % np.mean([T[q]["nodes"][0]["exec_ok"] is True for q in qs]), collections.Counter(T[q]["nodes"][0]["status"] for q in qs))
# beer_factory example where tree ex=0 and live ex=1 and same sql
for q in [q for q in roots if roots[q]["db_id"] == "beer_factory"][:6]:
    n = T[q]["nodes"][0]
    print(q, roots[q]["ex"], n["ex"], n["exec_ok"], n["status"], n["sql"][:100].replace("\n", " "), "|", roots[q]["sql"][:100].replace("\n", " "))
# ex None in tree?
print(collections.Counter(type(n["ex"]).__name__ for t in tr for n in t["nodes"]))
print(collections.Counter((n["ex"], n["exec_ok"]) for t in tr for n in t["nodes"]))
