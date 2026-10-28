"""Q4: where could gains be larger - by db, SQL-complexity proxy (BIRD train has no difficulty label), schema size, evidence. HR (trees) and HT (live).
All proxies are derived from the gold SQL text / schema (analysis only, never given to the system)."""
import sys, re; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
from trees import *

train = json.load(open(TRAIN))
tables = {t["db_id"]: t for t in json.load(open(R + "/data/bird/train/train/train_tables.json"))}


def complexity(sql):
    s = sql.upper()
    n_join = len(re.findall(r"\bJOIN\b", s))
    nested = len(re.findall(r"\(\s*SELECT\b", s))
    grp = int("GROUP BY" in s); od = int("ORDER BY" in s); hv = int("HAVING" in s); cs = int("CASE" in s)
    agg = len(re.findall(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", s)); setop = len(re.findall(r"\b(UNION|INTERSECT|EXCEPT)\b", s))
    sc = n_join + 2 * nested + grp + hv + cs + min(agg, 3) + 2 * setop + (1 if len(s.split()) > 40 else 0)
    return sc


Qs = load_trees(); live = by_qid(jl(HR_RUN + "/preds.jsonl")); htl = jl(HT_RUN + "/preds.jsonl"); htz = by_qid(jl(HT_ZS + "/preds.jsonl")); htzb = by_qid(jl(HT_ZS_B1 + "/preds.jsonl"))
sp = json.load(open(SPLIT))


def qinfo(qid):
    _, db, idx = qid.split(":"); r = train[int(idx)]
    t = tables[db]
    return {"cx": complexity(r["SQL"]), "ev": int(bool((r.get("evidence") or "").strip())), "n_tab": len(t["table_names_original"]), "n_col": len(t["column_names_original"]), "qlen": len(r["question"].split())}


rows = []
for q in Qs:
    i = qinfo(q.qid); p0 = q.p0(); lv = live[q.qid]
    rows.append(dict(i, set="HR", db=q.db, root=q.root["ex"], p0=p0["ex"], live=lv["ex"], ctx_tokens=lv["ctx_tokens"], full=int(lv["ctx_full_schema"]),
                     orc_chain=q.oracle(q.p0_chain_nodes()), orc_tree=q.oracle(q.nodes), orc_tier1=q.oracle([x for x in q.nodes if x["tier"] == 0]),
                     esc1=(q.first_child(0, "escalate") or {"ex": 0})["ex"], rev1=(q.first_child(0, "revise") or {"ex": 0})["ex"], score=q.root["score"]))
for p in htl:
    i = qinfo(p["qid"]); z = htz[p["qid"]]; zb = htzb[p["qid"]]
    rows.append(dict(i, set="HT", db=p["db_id"], root=None, p0=p["ex"], live=p["ex"], zs_ours=z["ex"], zs=zb["ex"], ctx_tokens=p["ctx_tokens"], full=int(p["ctx_full_schema"])))
allcx = np.array([r["cx"] for r in rows])
q1, q2 = np.percentile(allcx, [33.3, 66.7])
for r in rows:
    r["cx_bin"] = "low" if r["cx"] <= q1 else ("mid" if r["cx"] <= q2 else "high")
ctxs = np.array([r["ctx_tokens"] for r in rows])
c1, c2 = np.percentile(ctxs, [33.3, 66.7])
for r in rows:
    r["ctx_bin"] = "small" if r["ctx_tokens"] <= c1 else ("mid" if r["ctx_tokens"] <= c2 else "large")
res = {"cx_cuts": [float(q1), float(q2)], "ctx_cuts": [float(c1), float(c2)]}


def agg(rs, keys):
    o = {"n": len(rs)}
    for k in keys:
        v = [r[k] for r in rs if r.get(k) is not None]
        o[k] = float(np.mean(v)) if v else None
    return o


for st in ("HR", "HT"):
    sub = [r for r in rows if r["set"] == st]
    keys = ["root", "p0", "live", "orc_chain", "orc_tree", "orc_tier1", "esc1", "rev1"] if st == "HR" else ["live", "zs_ours", "zs"]
    for dim in ("cx_bin", "ctx_bin", "ev", "full"):
        res["%s_by_%s" % (st, dim)] = {str(v): agg([r for r in sub if r[dim] == v], keys) for v in sorted(set(r[dim] for r in sub))}
    res["%s_by_db" % st] = {d: agg([r for r in sub if r["db"] == d], keys + ["cx", "ctx_tokens", "n_tab"]) for d in sorted(set(r["db"] for r in sub))}
# P0-vs-root delta excluding exec-error roots, by complexity (HR)
sub = [r for r in rows if r["set"] == "HR"]
for dim in ("cx_bin", "ctx_bin"):
    res["HR_delta_p0_vs_root_pp_by_" + dim] = {str(v): 100 * float(np.mean([r["p0"] - r["root"] for r in sub if r[dim] == v])) for v in sorted(set(r[dim] for r in sub))}
# correlation of per-question correctness with proxies (HR root)
def corr(a, b):
    a, b = np.array(a, float), np.array(b, float)
    return float(np.corrcoef(a, b)[0, 1])
res["HR_corr_root_ex"] = {"complexity": corr([r["root"] for r in sub], [r["cx"] for r in sub]), "ctx_tokens": corr([r["root"] for r in sub], [r["ctx_tokens"] for r in sub]),
                          "qlen": corr([r["root"] for r in sub], [r["qlen"] for r in sub]), "evidence": corr([r["root"] for r in sub], [r["ev"] for r in sub])}
# within-db AUROC of root score (control for db confounding)
dbs = np.array([r["db"] for r in sub]); y = np.array([r["root"] for r in sub]); sc = np.array([r["score"] for r in sub])
w = []; 
for d in np.unique(dbs):
    a = auroc(sc[dbs == d], y[dbs == d]); n = (dbs == d).sum()
    w.append((d, float(a), int(n), float(y[dbs == d].mean())))
res["HR_withindb_AUROC_V3_root"] = w
ok = [x for x in w if not math.isnan(x[1])]
res["HR_withindb_AUROC_V3_root_macro"] = float(np.mean([x[1] for x in ok]))
res["HR_withindb_AUROC_V3_root_weighted"] = float(sum(x[1] * x[2] for x in ok) / sum(x[2] for x in ok))
dump("q4_strata.json", res)
print(json.dumps(res, indent=1, default=lambda o: float(o) if hasattr(o, "item") else str(o)))
