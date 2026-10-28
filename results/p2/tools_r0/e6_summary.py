"""Aggregate E6 CV (pooled over folds weighted by #questions, mean over seeds, SD over folds) and finals; compare with baselines.json.
python e6_summary.py <E6 dir> <trees.jsonl> <out.json>"""
import sys, os, json, collections
import numpy as np
d, trees, out = sys.argv[1:4]
fold_n = collections.Counter()
for l in open(trees):
    t = json.loads(l)
    fold_n[t["fold"]] += 1
res = {"fold_sizes": dict(fold_n)}
base = json.load(open(d + "/base/baselines.json")) if os.path.exists(d + "/base/baselines.json") else {}
res["baselines"] = base
def agg(path, key):
    rows = [json.loads(l) for l in open(path)]
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        cells[(r["c_h"], r["lam"])][r["fold"]].append(r["val"])
    tab = []
    for (c, lam), byf in sorted(cells.items()):
        pooled = collections.defaultdict(float)
        per_fold = collections.defaultdict(list)
        W = 0
        nseeds = min(len(v) for v in byf.values())
        for f, vs in byf.items():
            w = fold_n[f]
            W += w
            for k in ("ex_all", "calls", "tokens_k", "deferral", "ret"):
                m = float(np.mean([v[k] for v in vs]))
                pooled[k] += w * m
                per_fold[k].append(m)
        row = {"c_h": c, "lam": lam, "n_folds": len(byf), "n_seeds": nseeds}
        for k in pooled:
            row[k] = round(pooled[k] / W, 4)
            row[k + "_sd_folds"] = round(float(np.std(per_fold[k])), 4)
        # seed spread of the pooled ex_all
        sp = []
        for s in range(nseeds):
            sp.append(sum(fold_n[f] * byf[f][s]["ex_all"] for f in byf) / W)
        row["ex_all_sd_seeds"] = round(float(np.std(sp)), 4)
        tab.append(row)
    return tab
for name in ("cv", "cv_a2c"):
    p = "%s/%s/cv.jsonl" % (d, name)
    if os.path.exists(p):
        res[name] = agg(p, name)
for name in ("final", "final_a5", "final_a2c"):
    p = "%s/%s/final.jsonl" % (d, name)
    if os.path.exists(p):
        rows = [json.loads(l) for l in open(p)]
        by = collections.defaultdict(list)
        for r in rows:
            by[(r["c_h"], r["lam"])].append(r)
        tab = []
        for (c, lam), rs in sorted(by.items()):
            rs = sorted(rs, key=lambda r: r["seed"])
            tr = [r["train_return_last20"] for r in rs]
            med = sorted(rs, key=lambda r: r["train_return_last20"])[len(rs) // 2]   # X2 deviation 6: median training return among the seeds
            tab.append({"c_h": c, "lam": lam, "n_seeds": len(rs), "train_ex_all_mean": round(float(np.mean([r["train_eval"]["ex_all"] for r in rs])), 4),
                        "train_calls_mean": round(float(np.mean([r["train_eval"]["calls"] for r in rs])), 3),
                        "train_deferral_mean": round(float(np.mean([r["train_eval"]["deferral"] for r in rs])), 4),
                        "median_seed": med["seed"], "median_seed_path": os.path.relpath(med["path"], d),
                        "median_seed_train_eval": {k: round(med["train_eval"][k], 4) for k in ("ex_all", "calls", "deferral", "tokens_k", "ret")}})
        res[name] = tab
bc = d + "/base_cells/baselines_all_cells.json"
if os.path.exists(bc):
    B = json.load(open(bc))
    res["baselines_all_cells"] = B
    for name in ("cv", "cv_a2c"):
        for r in res.get(name, []):
            b = B["ch%s_lam%s" % (r["c_h"], r["lam"])]
            P0 = b["P0_theta8.5"]
            best = max((v for k, v in b.items() if k.startswith("P0_theta")), key=lambda v: v["ret"])
            r["P0_8.5"] = {k: round(P0[k], 4) for k in ("ret", "ex_all", "calls")}
            r["P0_best_theta_ret"] = {k: round(best[k], 4) for k in ("ret", "ex_all", "calls")}
            r["d_ret_vs_P0_8.5"] = round(r["ret"] - P0["ret"], 4)
            r["d_ex_vs_P0_8.5"] = round(r["ex_all"] - P0["ex_all"], 4)
json.dump(res, open(out, "w"), indent=1)
P0 = base.get("P0_theta8.5")
print("P0(8.5) pooled:", P0 and {k: round(v, 3) for k, v in P0.items() if k != "actions"})
for name in ("cv", "cv_a2c"):
    for r in res.get(name, []):
        print(name, "c_h=%.2f lam=%.2f  EX_all=%.3f (sd_f %.3f, sd_s %.4f)  calls=%.2f  defer=%.3f  ret=%.3f  | dRet vs P0(8.5)=%+.3f dEX=%+.3f" % (r["c_h"], r["lam"], r["ex_all"], r["ex_all_sd_folds"], r["ex_all_sd_seeds"], r["calls"], r["deferral"], r["ret"], r.get("d_ret_vs_P0_8.5", float("nan")), r.get("d_ex_vs_P0_8.5", float("nan"))))
