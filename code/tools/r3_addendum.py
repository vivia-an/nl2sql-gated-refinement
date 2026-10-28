#!/usr/bin/env python3
"""R3 author addendum (internal note R3EXP; SAP section 26). Post hoc, descriptive, no model call. Uses the seed-0 runs of RV2 and results/posthoc2/rv2_summary.json
(costs, latency).

    worker machine:  PYTHONPATH=<code> python code/tools/r3_addendum.py --runs $R/runs --rv2 results/posthoc2/rv2_summary.json --rv2s results/posthoc2/rv2s_summary.json --out results/r3 [--B 10000]

inc       incremental cost per additional correct answer of each pipeline configuration versus (a) the Plus single call, (b) the root-only arm of its backbone,
          (c) the single call of its backbone; with the cache discount (exp/prices.yaml, hit ratio 0.2) and at list price; break-even = extra CNY per 1 000 questions per
          pp of EX. Defined only where the EX difference is positive; the interval of the ratio is [cost / upper end, cost / lower end] of the cluster interval of the EX
          difference and exists only where that interval excludes 0. A companion string says why a ratio is not interpretable.
mde       minimum detectable effect, 80% power, two-sided alpha 0.05, for paired EX contrasts: item level (z, SD of the observed paired difference over n = 1 534) and
          database level (t with G - 1 = 10 df, SE = SD of the cluster bootstrap distribution of the contrast).
critic    fixed minus gated on the questions with the same root SQL in both runs (the increment of the semantic Critic over the error-text-only loop): helped / hurt, net
          questions and pp, exact McNemar p, exact conditional (Clopper-Pearson) interval; the same figure implied by the helped / hurt counts of Table 4 (loop.<b>.eff_in).
stack     the pooled root-only-minus-single-call interval with sign-preserving 3-decimal bounds.
latency   ratios of mean latency per question (preds latency_s) to the single call of the backbone, from the RV2 cost keys."""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3_common as C  # noqa: E402
from r3_common import S  # noqa: E402
import rv2_analyze as RA  # noqa: E402
import rv2_keys as K  # noqa: E402
import r3_rest as RR  # noqa: E402

CONFIGS = {"plus": ("rootonly", "gated", "fixed", "route"), "flash": ("rootonly", "gated", "fixed"), "27b": ("rootonly", "gated", "fixed")}
MDE_CONTRASTS = (("stack", "rootonly", "zs"), ("loop_g", "gated", "rootonly"), ("loop_f", "fixed", "rootonly"), ("g_minus_f", "gated", "fixed"), ("g_minus_zs", "gated", "zs"))
Z975, Z80 = 1.959963984540054, 0.8416212335729143


def conv(o):
    if isinstance(o, dict):
        return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [conv(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def load_s0(runs_root):
    rows = {}
    for (b, a), rel in RA.RUNS.items():
        r = C.load_preds(runs_root, rel)
        ok, why = C.check_run(r)
        if not ok:
            raise SystemExit("seed-0 run %s %s: %s" % (b, a, why))
        rows[(b, a)] = r
    qids = sorted(set.intersection(*[set(k for k, x in r.items() if C.ex_of(x) is not None) for r in rows.values()]))
    ex = {k: np.array([C.ex_of(r[q]) for q in qids], float) for k, r in rows.items()}
    dbs = np.array([rows[("plus", "zs")][q]["db_id"] for q in qids])
    return rows, qids, ex, dbs


def part_inc(rv2, ex, dbs, boot, B):
    kk = rv2["keys"]
    cost = lambda b, c, mode: float(kk["cost.%s.%s.%s" % (b, c, mode)]["v"])
    out = {}
    refs = lambda b, c: [("plus_zs", ("plus", "zs"))] + ([("rootonly", (b, "rootonly"))] if c != "rootonly" else []) + ([("self_zs", (b, "zs"))] if b != "plus" else [])
    for b in C.BK:
        for c in CONFIGS[b]:
            for rname, rk in refs(b, c):
                if (b, c) == rk:
                    continue
                d = ex[(b, c)] - ex[rk]
                est, ci, ic = C.boot_ci(d, dbs, B=B, boot=boot)
                e = {"dex": est, "dex_ci": ci, "dex_item_ci": ic, "n": int(len(d))}
                for mode, nm in (("cny_q", "disc"), ("cny_q_nodisc", "nodisc")):
                    e["dcost_q_" + nm] = cost(b, c, mode) - cost(rk[0], rk[1], mode)
                reasons = []
                if est <= 0:
                    reasons.append("EX difference not positive (%.2f pp): the configuration is not more accurate than the reference" % est)
                if ci[0] <= 0 <= ci[1]:
                    reasons.append("cluster interval of the EX difference [%.2f, %.2f] pp contains 0" % ci)
                if ic[0] <= 0 <= ic[1]:
                    reasons.append("item-level interval [%.2f, %.2f] pp contains 0" % ic)
                if e["dcost_q_disc"] <= 0:
                    reasons.append("not more expensive than the reference (%.4f CNY per question)" % e["dcost_q_disc"])
                e["undefined_reason"] = "; ".join(reasons) if reasons else "none"
                e["ratio_defined"] = bool(est > 0 and e["dcost_q_disc"] > 0)
                e["ci_defined"] = bool(ci[0] > 0 and e["ratio_defined"])
                if e["ratio_defined"]:
                    for nm in ("disc", "nodisc"):
                        dq = e["dcost_q_" + nm]
                        e["cny_per_correct_" + nm] = dq / (est / 100.0)
                        e["cny_1000_per_pp_" + nm] = 1000.0 * dq / est
                        if e["ci_defined"]:
                            e["cny_per_correct_ci_" + nm] = (dq / (ci[1] / 100.0), dq / (ci[0] / 100.0))
                out["%s|%s|%s" % (b, c, rname)] = e
    return out


def part_mde(ex, dbs, boot):
    from scipy import stats as st
    n = len(dbs)
    G = boot.k
    tq = st.t.ppf(0.975, G - 1) + st.t.ppf(0.80, G - 1)
    out = {"n": n, "G": G, "t_sum": float(tq), "z_sum": Z975 + Z80, "cells": {}}
    vec = {}
    for cid, x, y in MDE_CONTRASTS:
        for b in C.BK:
            vec[(cid, b)] = ex[(b, x)] - ex[(b, y)]
        vec[(cid, "pooled3")] = np.mean([vec[(cid, b)] for b in C.BK], axis=0)
    for (cid, sc), d in vec.items():
        item = (Z975 + Z80) * float(np.std(d, ddof=1)) / math.sqrt(n) * 100.0
        se_cl = float(np.std(boot.means(d), ddof=1)) * 100.0
        e = {"disc_rate_pct": 100.0 * float((d != 0).mean()) if sc != "pooled3" else None, "sd_pp": 100.0 * float(np.std(d, ddof=1)),
             "mde_item_pp": item, "se_cluster_pp": se_cl, "mde_cluster_pp": float(tq) * se_cl, "est_pp": 100.0 * float(d.mean())}
        out["cells"]["%s|%s" % (cid, sc)] = e
    ii = [v["mde_item_pp"] for v in out["cells"].values()]
    cc = [v["mde_cluster_pp"] for v in out["cells"].values()]
    out["item_min"], out["item_max"], out["cluster_min"], out["cluster_max"] = min(ii), max(ii), min(cc), max(cc)
    return out


def part_critic(runs_root, rv2, B):
    out = {}
    kk = rv2["keys"]
    for b in C.BK:
        g = C.load_preds(runs_root, C.run_path(b, "gated", 0))
        f = C.load_preds(runs_root, C.run_path(b, "fixed", 0))
        gc = {r["qid"]: r for r in C.jl(os.path.join(runs_root, C.run_path(b, "gated", 0), "cands.jsonl")) if r.get("cand_idx") == 0}
        fc = {r["qid"]: r for r in C.jl(os.path.join(runs_root, C.run_path(b, "fixed", 0), "cands.jsonl")) if r.get("cand_idx") == 0}
        qs = sorted(set(g) & set(f))
        shared = [q for q in qs if (gc[q].get("sql") or "").strip() == (fc[q].get("sql") or "").strip()]
        gx = np.array([C.ex_of(g[q]) for q in shared], int)
        fx = np.array([C.ex_of(f[q]) for q in shared], int)
        helped, hurt = int(((fx == 1) & (gx == 0)).sum()), int(((fx == 0) & (gx == 1)).sum())
        m = helped + hurt
        mc = S.mcnemar_exact(helped, hurt)
        lo, hi = S.clopper_pearson(helped, m) if m else (0.0, 1.0)
        N = len(qs)
        e = {"n_questions": N, "n_shared_root": len(shared), "helped": helped, "hurt": hurt, "net_q": helped - hurt, "net_pp_all": 100.0 * (helped - hurt) / N,
             "net_ci_q": (m * (2 * lo - 1), m * (2 * hi - 1)), "net_ci_pp_all": (100.0 * m * (2 * lo - 1) / N, 100.0 * m * (2 * hi - 1) / N),
             "p_two": mc["p_two"], "p_one_greater": mc["p_one_greater"]}
        # implied by the helped / hurt counts of Table 4 (final answer against the root of the same run)
        bf, cf = int(kk["loop.%s.eff_in.fixed.b" % b]["v"]), int(kk["loop.%s.eff_in.fixed.c" % b]["v"])
        bg, cg = int(kk["loop.%s.eff_in.gated.b" % b]["v"]), int(kk["loop.%s.eff_in.gated.c" % b]["v"])
        e["table4"] = {"fixed_helped": bf, "fixed_hurt": cf, "gated_helped": bg, "gated_hurt": cg, "implied_net_q": (bf - cf) - (bg - cg),
                       "implied_net_pp": 100.0 * ((bf - cf) - (bg - cg)) / N}
        # the revisions that only the fixed loop makes (Critic-only revisions): questions where the gated loop did not revise and the fixed loop did
        n_fixed_only_rev = sum(1 for q in shared if (f[q].get("actions") or ["accept"]) != ["accept"] and (g[q].get("actions") or ["accept"]) == ["accept"])
        e["revised_by_fixed_only"] = n_fixed_only_rev
        out[b] = e
    return out


def part_stack(rv2s):
    """pooled root-only minus single call (RV2.loop.pooled.stack): all five small-cluster intervals with their exact ends (the printed one-decimal form of the widest
    interval, [0.0, +2.7], hides a lower end of -0.0487)."""
    kk = rv2s["keys"]
    g = lambda t: kk["small.loop.pooled.stack.%s" % t]["v"]
    out = {"est": g("est")}
    for t in ("cl_ci", "item_ci", "t_ci", "tdb_ci", "sf_ci", "wide_ci", "wideu_ci"):
        out[t] = g(t)
    out["wide_lo_exact"] = out["wide_ci"][0]
    out["wide_hi_exact"] = out["wide_ci"][1]
    out["wide_lo_text3"] = "%+.3f" % out["wide_ci"][0]
    out["wide_ci_text2"] = "[%+.2f, %+.2f]" % tuple(out["wide_ci"])
    return out


def part_latency(rv2):
    kk = rv2["keys"]
    out = {"source": "preds.jsonl latency_s (wall time of the question inside the run, includes retries and rate-limit waits); RV2 keys cost.<b>.<cfg>.lat_mean/lat_p50/lat_p95", "ratio": {}}
    for b in C.BK:
        z = float(kk["cost.%s.zs.lat_mean" % b]["v"])
        for c in ("rootonly", "gated", "fixed"):
            out["ratio"]["%s|%s" % (b, c)] = float(kk["cost.%s.%s.lat_mean" % (b, c)]["v"]) / z
    return out


def part_critinc(runs_root, dbs_ref, boot, B):
    """Critic's own increment inside the run: (fixed final - fixed root) - (gated final - gated root), per question, pp, cluster interval (RW3 'critinc')."""
    out = {}
    for b in C.BK:
        d = []
        qs = None
        parts = {}
        for arm in ("gated", "fixed"):
            rows = C.load_preds(runs_root, C.run_path(b, arm, 0))
            roots = {r["qid"]: r for r in C.jl(os.path.join(runs_root, C.run_path(b, arm, 0), "cands.jsonl")) if r.get("cand_idx") == 0}
            parts[arm] = (rows, roots)
        qs = sorted(set(parts["gated"][0]) & set(parts["fixed"][0]))
        vec = np.array([(C.ex_of(parts["fixed"][0][q]) - float(parts["fixed"][1][q].get("ex") or 0)) - (C.ex_of(parts["gated"][0][q]) - float(parts["gated"][1][q].get("ex") or 0)) for q in qs], float)
        dbs = np.array([parts["gated"][0][q]["db_id"] for q in qs])
        est, ci, ic = C.boot_ci(vec, dbs, B=B)
        out[b] = {"d": est, "ci": ci, "item_ci": ic, "net_q": int(round(vec.sum()))}
    return out


def part_mde_named(runs_root, ex, dbs):
    """MDE (80% power, two-sided 0.05, observed paired variance) of the four headline tests: P1' (pooled gated minus fixed), W1 (mean of Flash and 27B gated minus single call),
    P3 (AUROC difference, DeLong SE), P5' (rate difference on the failing roots)."""
    n = len(dbs)
    zs = Z975 + Z80
    d_p1 = np.mean([ex[(b, "gated")] - ex[(b, "fixed")] for b in C.BK], axis=0)
    d_w1 = np.mean([ex[(b, "gated")] - ex[(b, "zs")] for b in ("flash", "27b")], axis=0)
    out = {"P1p_pp": zs * float(np.std(d_p1, ddof=1)) / math.sqrt(n) * 100.0, "W1_pp": zs * float(np.std(d_w1, ddof=1)) / math.sqrt(n) * 100.0}
    pop = RR.p5_population(runs_root)
    d5 = np.array([r["gated"] - r["blind"] for r in pop], float)
    out["P5p_pp"] = zs * float(np.std(d5, ddof=1)) / math.sqrt(len(d5)) * 100.0
    out["P5p_n"] = int(len(d5))
    import fit_route_score as F
    from analysis_pipeline import primary
    run = C.load_preds(runs_root, "MIN/M1/gated-route-s0")
    rows = [run[q] for q in sorted(run)]
    coef = json.load(open(os.path.join(os.path.dirname(os.path.abspath(runs_root.rstrip("/"))), "exp/frozen/route_coef_hr.json")))
    v3, ca = primary.route_scores(rows, coef, 4)
    y = np.array([C.ex_of(r) if C.ex_of(r) is not None else -1 for r in rows])
    ok = np.isfinite(v3) & np.isfinite(ca) & (y >= 0)
    dl = S.delong_paired(ca[ok], v3[ok], y[ok].astype(int))
    out["P3_auroc"] = zs * float(dl["se"])
    out["P3_diff"] = float(dl["diff"])
    out["P3_n"] = int(ok.sum())
    return out


def part_mult(runs_root):
    import rv2s_keys as KS
    return {"n_dual": len(KS.DUAL_IDS), "n_loop": len(KS.LOOP_IDS), "n_confirmatory": len(KS.PSTAR_FAMILY), "family": list(KS.PSTAR_FAMILY)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--rv2", required=True)
    ap.add_argument("--rv2s", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=C.B_DEFAULT)
    a = ap.parse_args(argv)
    rv2 = json.load(open(a.rv2))
    rv2s = json.load(open(a.rv2s))
    rows, qids, ex, dbs = load_s0(a.runs)
    boot = S.ClusterBoot(dbs, B=a.B)
    res = {"n_qids": len(qids), "inc": part_inc(rv2, ex, dbs, boot, a.B), "mde": part_mde(ex, dbs, boot), "critic": part_critic(a.runs, rv2, a.B),
           "stack": part_stack(rv2s), "critinc": part_critinc(a.runs, dbs, boot, a.B), "mde_named": part_mde_named(a.runs, ex, dbs), "latency": part_latency(rv2), "mult": part_mult(a.runs)}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "addendum.json"), "w") as f:
        json.dump(conv(res), f, indent=1)
    inc = []
    for k, e in sorted(res["inc"].items()):
        b, c, r = k.split("|")
        inc.append([b, c, r, e["dex"], e["dex_ci"][0], e["dex_ci"][1], e["dex_item_ci"][0], e["dex_item_ci"][1], e["dcost_q_disc"], e["dcost_q_nodisc"],
                    e.get("cny_per_correct_disc"), e.get("cny_per_correct_nodisc"), e.get("cny_1000_per_pp_disc"), e.get("cny_1000_per_pp_nodisc"), e["undefined_reason"]])
    C.write_csv(os.path.join(a.out, "addendum_inc.csv"), ["backbone", "config", "reference", "dex_pp", "cl_lo", "cl_hi", "item_lo", "item_hi", "dcost_q_disc", "dcost_q_nodisc",
                                                          "cny_per_correct_disc", "cny_per_correct_nodisc", "cny_1000_per_pp_disc", "cny_1000_per_pp_nodisc", "undefined_reason"], inc)
    C.write_csv(os.path.join(a.out, "addendum_mde.csv"), ["contrast", "scope", "est_pp", "disc_rate_pct", "sd_pp", "mde_item_pp", "se_cluster_pp", "mde_cluster_pp"],
                [k.split("|") + [e["est_pp"], e["disc_rate_pct"], e["sd_pp"], e["mde_item_pp"], e["se_cluster_pp"], e["mde_cluster_pp"]] for k, e in sorted(res["mde"]["cells"].items())])
    C.write_csv(os.path.join(a.out, "addendum_critic.csv"), ["backbone", "shared_root_q", "helped", "hurt", "net_q", "net_pp_all", "net_ci_lo_pp", "net_ci_hi_pp", "p_two", "table4_implied_net_q", "table4_implied_net_pp", "revised_by_fixed_only"],
                [[b, e["n_shared_root"], e["helped"], e["hurt"], e["net_q"], e["net_pp_all"], e["net_ci_pp_all"][0], e["net_ci_pp_all"][1], e["p_two"], e["table4"]["implied_net_q"], e["table4"]["implied_net_pp"], e["revised_by_fixed_only"]]
                 for b, e in res["critic"].items()])
    print("[r3_addendum] inc cells %d, mde cells %d" % (len(res["inc"]), len(res["mde"]["cells"])))


if __name__ == "__main__":
    main()
