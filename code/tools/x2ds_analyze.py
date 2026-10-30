#!/usr/bin/env python3
"""X2DS analysis (post hoc, non-Qwen backbone deepseek-v4.1-flash-1017 on the BIRD development set). New file; imports only analysis_pipeline.stats (read-only).
    python code/tools/x2ds_analyze.py --runs <dir with zs-dsv41flash-s0 fixed-... gated-... rootonly-...> --out <dir> [--B 10000] [--price-in 2.0 --price-out 8.0] [--cache-ratio 0.2]
Needs numpy. Scope: exploratory, post hoc; never enters a confirmatory family.
Per arm: n, EX, calls/q, tokens/q (prompt, completion, cached; cache-independent, as recorded per question), CNY per 1 000 questions (list price and with the implicit-cache
discount of exp/prices.yaml). Contrasts (percentage points, per-question paired): gated - zs, fixed - zs, gated - fixed, gated - rootonly, fixed - rootonly, with the database-clustered
bootstrap interval (shared draws, B resamples of the 11 databases), the question-level bootstrap interval, the cluster-robust t(G-1) interval (CR1), the exact sign-flip interval over the
size-weighted database means, their hull (the widest of the four; the paper's reading rule), the exact McNemar test and the discordant counts; calls ratio gated / fixed.
The four intervals are computed by r4_t0.four (the same code as the paper's other post hoc contrasts), so the cluster and item-level intervals equal those of the first version of this script."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
from analysis_pipeline import stats as S  # noqa: E402
sys.path.insert(0, HERE)
import r3_common as C  # noqa: E402
from r4_t0 import four  # noqa: E402

ARMS = ("zs", "zsretry", "fixed", "gated", "rootonly", "zsthink", "zsretryempty", "macsql", "gatedempty")
# W7 (post hoc): zsthink = single call with thinking on, zsretryempty = retry-only arm whose gate also fires on an empty result, macsql = the frozen
# MAC-SQL baseline (Plus: the locked run; Flash, 27B: run W7). W8 (post hoc): gatedempty = the pipeline with the empty gate (run W8).
# Contrasts whose arms are missing on a backbone are skipped.
VALUES = (0.01, 0.1, 1.0, 10.0)      # CNY per correct answer for the net-benefit probabilities
CONTRASTS = [("zsretry_minus_zs", "zsretry", "zs"), ("gated_minus_zsretry", "gated", "zsretry"), ("fixed_minus_zsretry", "fixed", "zsretry"), ("rootonly_minus_zsretry", "rootonly", "zsretry"),
             ("gated_minus_zs", "gated", "zs"), ("fixed_minus_zs", "fixed", "zs"), ("gated_minus_fixed", "gated", "fixed"),
             ("gated_minus_rootonly", "gated", "rootonly"), ("fixed_minus_rootonly", "fixed", "rootonly"), ("rootonly_minus_zs", "rootonly", "zs"),
             ("zsthink_minus_zs", "zsthink", "zs"), ("gated_minus_zsthink", "gated", "zsthink"), ("zsretryempty_minus_zs", "zsretryempty", "zs"),
             ("zsretryempty_minus_zsretry", "zsretryempty", "zsretry"), ("macsql_minus_zs", "macsql", "zs"), ("gated_minus_macsql", "gated", "macsql"),
             ("fixed_minus_gated", "fixed", "gated"), ("gated_minus_zsretryempty", "gated", "zsretryempty"), ("zsthink_minus_zsretryempty", "zsthink", "zsretryempty"),
             ("gatedempty_minus_zs", "gatedempty", "zs"), ("gatedempty_minus_gated", "gatedempty", "gated"), ("gatedempty_minus_zsretryempty", "gatedempty", "zsretryempty"),
             ("fixed_minus_gatedempty", "fixed", "gatedempty")]


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                rows[r["qid"]] = r
            except ValueError:
                pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None, help="directory with <arm>-<suffix>/preds.jsonl")
    ap.add_argument("--arm", action="append", default=[], help="NAME=path/to/preds.jsonl (overrides --runs for that arm; NAME in zs, zsretry, fixed, gated, rootonly)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--price-in", type=float, default=2.0)
    ap.add_argument("--price-out", type=float, default=8.0)
    ap.add_argument("--cache-ratio", type=float, default=0.2)
    ap.add_argument("--suffix", default="dsv41flash-s0")
    a = ap.parse_args()
    R = {}
    given = dict(x.split("=", 1) for x in a.arm)
    for arm in ARMS:
        p = given.get(arm) or (os.path.join(a.runs, "%s-%s" % (arm, a.suffix), "preds.jsonl") if a.runs else None)
        if p and os.path.exists(p):
            R[arm] = load(p)
    qids = sorted(set.intersection(*[set(r) for r in R.values()]))
    dbs = [R[next(iter(R))][q]["db_id"] for q in qids]
    out = {"sources": given, "n_common": len(qids), "arms": {}, "contrasts": {}, "B": a.B, "price_cny_per_mtok": {"in": a.price_in, "out": a.price_out}, "cache_ratio": a.cache_ratio}
    ex, cost_l, cost_d, sql = {}, {}, {}, {}
    for arm, rows in R.items():
        rs = [rows[q] for q in qids]
        e = np.array([1.0 if r.get("ex") == 1 else 0.0 for r in rs])
        ex[arm] = e
        sql[arm] = [(r.get("final_sql") or "").strip() for r in rs]
        n_null = sum(1 for r in rs if r.get("ex") is None)
        pt = np.array([r.get("prompt_tokens") or 0 for r in rs], float)
        ct = np.array([r.get("completion_tokens") or 0 for r in rs], float)
        cached = np.minimum(np.array([r.get("cached_tokens") or 0 for r in rs], float), pt)
        calls = np.array([r.get("calls") or 0 for r in rs], float)
        list_cny = (pt * a.price_in + ct * a.price_out) / 1e6
        disc_cny = ((pt - cached) * a.price_in + cached * a.price_in * a.cache_ratio + ct * a.price_out) / 1e6
        cost_l[arm], cost_d[arm] = list_cny, disc_cny
        lat = np.array([r.get("latency_s") or 0.0 for r in rs], float)
        errs = {}
        for r in rs:
            k = r.get("error_class")
            if k:
                errs[k] = errs.get(k, 0) + 1
        pe = {}
        for r in rs:
            m = (r.get("pred_error") or "")[:7]
            if m.startswith("timeout"):
                pe["timeout"] = pe.get("timeout", 0) + 1
        out["arms"][arm] = {"n": len(rs), "ex_pct": 100 * e.mean(), "ex_null": n_null, "calls_per_q": calls.mean(), "prompt_tok_per_q": pt.mean(), "completion_tok_per_q": ct.mean(),
                            "cached_tok_per_q": cached.mean(), "cny_per_1000q_list": 1000 * list_cny.mean(), "cny_per_1000q_cache_disc": 1000 * disc_cny.mean(),
                            "cny_per_1000q_list_offpeak_half": 500 * list_cny.mean(), "error_classes": errs, "pred_timeouts": pe.get("timeout", 0),
                            "exec_ok_pct": 100 * np.mean([1.0 if r.get("exec_ok") else 0.0 for r in rs]),
                            "lat_mean_s": float(lat.mean()), "lat_p95_s": float(np.percentile(lat, 95)),
                            "n_exec_near_limit": sum(1 for r in rs if (r.get("exec_s") or 0.0) >= 24.0),
                            "n_no_sql": sum(1 for r in rs if not (r.get("final_sql") or "").strip()),
                            "n_not_exec": sum(1 for r in rs if not r.get("exec_ok")),
                            "ct_max": max((r.get("completion_tokens") or 0) for r in rs)}
        out["arms"][arm]["_calls_vec_sum"] = float(calls.sum())
    boot = S.ClusterBoot(dbs, B=a.B)
    for cid, x, y in CONTRASTS:
        if x not in ex or y not in ex:
            continue
        v = ex[x] - ex[y]
        f4 = four(v, dbs, boot, a.B)
        b = int(((ex[x] == 1) & (ex[y] == 0)).sum())
        c = int(((ex[x] == 0) & (ex[y] == 1)).sum())
        draws = boot.means(v)                       # cluster-bootstrap draws of the mean difference (fraction), shared resample
        se = float(np.std(draws, ddof=1))
        same = np.array([sql[x][i] == sql[y][i] for i in range(len(qids))])
        vst = np.where(same, 0.0, v)                # W7: identical final SQL text scored identically (re-execution flips of one query set to no difference)
        fst = four(vst, dbs, boot, a.B)
        dl, dd = float(cost_l[x].mean() - cost_l[y].mean()), float(cost_d[x].mean() - cost_d[y].mean())
        out["contrasts"][cid] = {"est_pp": f4["est"], "cluster_ci_pp": list(f4["cl"]), "item_ci_pp": list(f4["item"]), "t_ci_pp": list(f4["t"]), "signflip_ci_pp": list(f4["sf"]),
                                 "wide_ci_pp": list(f4["wide"]), "reading_wide": C.reading(*f4["wide"]), "reading_cluster": C.reading(*f4["cl"]),
                                 "discordant_x_only": b, "discordant_y_only": c, "mcnemar_exact_p": S.mcnemar_exact(b, c),
                                 "p_pos_cluster": float(np.mean(draws > 0)), "se_cluster_pp": 100.0 * se, "mde80_pp": 100.0 * 2.8016 * se,
                                 "same_text_x_only": int(((v > 0) & same).sum()), "same_text_y_only": int(((v < 0) & same).sum()),
                                 "st_est_pp": fst["est"], "st_wide_ci_pp": list(fst["wide"]), "st_reading_wide": C.reading(*fst["wide"]),
                                 "st_x_only": int((vst > 0).sum()), "st_y_only": int((vst < 0).sum()),
                                 "dcost_list_cny_q": dl, "dcost_disc_cny_q": dd,
                                 "nb_p_list": {str(V): float(np.mean(V * draws - dl > 0)) for V in VALUES},
                                 "nb_p_disc": {str(V): float(np.mean(V * draws - dd > 0)) for V in VALUES}}
    if "gated" in R and "fixed" in R:
        cg = np.array([R["gated"][q].get("calls") or 0 for q in qids], float)
        cf = np.array([R["fixed"][q].get("calls") or 0 for q in qids], float)
        rr = boot.ratio(cg, cf)
        out["calls_ratio_gated_over_fixed"] = {"est": float(cg.sum() / cf.sum()), "cluster_ci": list(S.pctl_ci(rr))}
    if "zsretry" in R:
        rr = [R["zsretry"][q] for q in qids]
        rev = [r for r in rr if (r.get("iterations") or 0) > 0]
        zs_fail = sum(1 for q in qids if "zs" in R and not R["zs"][q].get("exec_ok"))
        out["retry"] = {"n_revised": len(rev), "share_revised_pct": 100.0 * len(rev) / len(rr), "revisions_total": int(sum(r.get("iterations") or 0 for r in rr)),
                        "revisions_per_revised": (sum(r.get("iterations") or 0 for r in rev) / float(len(rev))) if rev else 0.0,
                        "n_final_exec_ok_of_revised": sum(1 for r in rev if r.get("exec_ok")), "n_correct_of_revised": sum(1 for r in rev if r.get("ex") == 1),
                        "n_root_exec_fail_zs": zs_fail}
        if "zs" in R:
            same = [q for q in qids if (R["zsretry"][q].get("final_sql") or "").strip() == (R["zs"][q].get("final_sql") or "").strip()]
            out["retry"]["n_sql_identical_to_zs"] = len(same)
            out["retry"]["n_ex_up_same_sql"] = sum(1 for q in same if R["zsretry"][q].get("ex") == 1 and R["zs"][q].get("ex") != 1)
            out["retry"]["n_ex_down_same_sql"] = sum(1 for q in same if R["zsretry"][q].get("ex") != 1 and R["zs"][q].get("ex") == 1)
    if "zsretryempty" in R and "zs" in R:          # W7: the empty-gate arm, revised questions and their flips against the single call
        re_ = [q for q in qids if (R["zsretryempty"][q].get("iterations") or 0) > 0]
        out["retry_empty"] = {"n_revised": len(re_), "share_revised_pct": 100.0 * len(re_) / len(qids),
                              "n_up": sum(1 for q in re_ if R["zsretryempty"][q].get("ex") == 1 and R["zs"][q].get("ex") != 1),
                              "n_down": sum(1 for q in re_ if R["zsretryempty"][q].get("ex") != 1 and R["zs"][q].get("ex") == 1)}
    if "gatedempty" in R and "gated" in R:         # W8: questions where the empty gate changed the pipeline's answer, and their flips against the pipeline
        ch = [q for q in qids if (R["gatedempty"][q].get("final_sql") or "").strip() != (R["gated"][q].get("final_sql") or "").strip()]
        out["gated_empty"] = {"n_changed": len(ch), "share_changed_pct": 100.0 * len(ch) / len(qids),
                              "n_up": sum(1 for q in ch if R["gatedempty"][q].get("ex") == 1 and R["gated"][q].get("ex") != 1),
                              "n_down": sum(1 for q in ch if R["gatedempty"][q].get("ex") != 1 and R["gated"][q].get("ex") == 1)}
    for arm in out["arms"].values():
        arm.pop("_calls_vec_sum", None)
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "x2ds_summary.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
