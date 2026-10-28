"""Q2 addendum: within-database AUROC (removes between-db confounding: eff penalises big tables, big dbs are also the hard ones)."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
from trees import *
Qs = load_trees()
htc = [c for c in jl(HT_RUN + "/cands.jsonl") if c["cand_idx"] == 0]
def sets():
    hr = [{"db": q.db, "y": q.root["ex"], "score": q.root["score"], "sem": q.root["semantic"], "syn": q.root["syntax"], "eff": q.root["efficiency"], "nonempty": float(q.root["status"] == "ok_nonempty"),
           "execok": float(q.root["status"] in ("ok_nonempty", "ok_empty"))} for q in Qs]
    ht = [{"db": c["db_id"], "y": c["ex"], "score": c["score"], "sem": c["semantic"], "syn": c["syntax"], "eff": c["efficiency"], "nonempty": float(c["status"] == "ok_nonempty"),
           "execok": float(c["status"] in ("ok_nonempty", "ok_empty"))} for c in htc]
    return {"HR": hr, "HT": ht}
res = {}
for nm, rows in sets().items():
    out = {}
    dbs = np.array([r["db"] for r in rows]); y = np.array([r["y"] for r in rows])
    for sig in ("score", "sem", "syn", "eff", "nonempty", "execok"):
        s = np.array([r[sig] for r in rows], float)
        per = []
        for d in np.unique(dbs):
            m = dbs == d; a = auroc(s[m], y[m])
            if not math.isnan(a): per.append((a, int(m.sum())))
        w = sum(a * k for a, k in per) / sum(k for _, k in per)
        # exclude beer_factory
        per2 = [(auroc(s[dbs == d], y[dbs == d]), int((dbs == d).sum())) for d in np.unique(dbs) if d != "beer_factory" and not math.isnan(auroc(s[dbs == d], y[dbs == d]))]
        w2 = sum(a * k for a, k in per2) / max(1, sum(k for _, k in per2)) if per2 else None
        out[sig] = {"within_db_AUROC_weighted": float(w), "excl_beer_factory": w2, "per_db": {d: float(auroc(s[dbs == d], y[dbs == d])) for d in np.unique(dbs)}}
    res[nm] = out
dump("q2_withindb.json", res)
for nm, o in res.items():
    for s, v in o.items(): print(nm, s, "%.3f" % v["within_db_AUROC_weighted"], ("%.3f" % v["excl_beer_factory"]) if v["excl_beer_factory"] else "", {d: round(x, 2) for d, x in v["per_db"].items()})
