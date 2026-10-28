#!/usr/bin/env python3
"""R4 (round-4 statistics of Group A; internal review note revision list 4 A1-A8). Post hoc, descriptive, no model call, no API key. numpy + scipy; run on a worker machine (host python 3.6 has no numpy).

    worker machine:  PYTHONPATH=<code> python code/tools/r4_stats.py --runs $R/runs --rv2 results/posthoc2/rv2_summary.json --out results/r4 [--B 10000] [--part all|widest|critic|margin|regen|mde|p5|silent]

Parts (each writes one JSON file in --out; code/tools/r4_merge.py turns them into results/r4/rv4_summary.json, the file make_xmacros.py reads as RV4.*):

widest  (A2)  ONE widest-interval definition for every setting: the hull of the four intervals the Methods define for the locked run -- the database-clustered bootstrap interval,
              the item-level interval, the cluster-robust t(G-1) interval and the exact sign-flip interval (size-weighted database means) -- applied to the locked run (seed 0) AND to the
              replication (the run-averaged per-question differences of seeds 101 / 202 / 303). The between-run range of the replication stays a separate column and is not part of the
              hull. Output: widest.json (decomposition contrasts and the same-examples control, both settings, readings, the 'agrees' flag under the new and under the old definition)
              and regen.json (A4): share of questions whose root SQL differs between arms, and the net questions those questions carry (locked run and replication).
critic  (A1)  cost per extra correct answer of the Critic's own increment (fixed loop over gated loop, final minus own root in each run, per backbone), with and without the cache
              discount, cluster / item-level / widest interval of the increment, 'undefined' where the increment is not positive, break-even, extra correct answers per 1,000.
              Round 5 (A2): the same price on ONE basis, the final-answer difference (fixed final minus gated final, locked run), which is what a service pays for and gets; and the
              price of the retry on the same basis (gated final minus root-only final, separate runs). On Plus and 27B the two bases for the Critic coincide; on Flash they differ
              because the two runs regenerated the root query for 29 questions.
margin  (A3)  P1' margin ratio and share kept for the replication (final minus own root within each T = 0.7 run, averaged over the three runs, pooled over backbones), beside
              the locked-run value computed the same way. Round 5 (A3): the interval of both is the hull of FOUR intervals (clustered bootstrap, item-level bootstrap, cluster-robust
              t(G-1), exact sign flip). The ratio interval maps the interval of the fixed-loop effect through 1.5 / x; the share-kept interval is a Fieller-type test inversion
              (the set of R for which mean(gated - R * fixed) is not rejected), so that R = 1 lies in it exactly when the within-run difference (gated minus fixed) does.
mde     (A6)  minimum detectable effect (power 0.80, two-sided 0.05) at question level (z, SD of the paired differences / sqrt(n)) and at database level (t(G-1) sum x SD of the
              cluster-bootstrap distribution of the estimate) for the four named tests P1', W1, P3, P5'.
p5      (A7)  P5' with distinct questions: how many different questions the pooled failing roots are, how many fail on one, two, three backbones, and the test with one unit per question.
silent  (A7)  silent-error rate (executes without error and is wrong), its share of the wrong answers and the share of wrong roots the gate can reach (root failed to execute), per backbone.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3_common as C  # noqa: E402
from r3_common import S  # noqa: E402
import r3_seeds as RS  # noqa: E402
import r3_samex as SX  # noqa: E402
import r3_rest as RR  # noqa: E402
import small_cluster as SC  # noqa: E402

Z975, Z80 = 1.959963984540054, 0.8416212335729143
MARGIN = 1.5


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


def dump(out, name, obj):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, name), "w") as f:
        json.dump(conv(obj), f, indent=1)
    print("[r4_stats] wrote %s" % os.path.join(out, name))


# ------------------------------------------------------------------------------------------------------------ widest interval (A2)
def small4(vec, dbs, cl, item):
    """the two small-cluster intervals of the Methods and the hull of all four, in pp. vec: per-question difference in 0/1 units (any run-averaged vector)."""
    d = np.asarray(vec, float) * 100.0
    tb = SC.cr_t(d, dbs)
    _, n_g, m_g = SC.db_level(d, dbs)
    sf = SC.sf_ci(m_g, n_g / n_g.sum())
    t_iv = (float(tb["lo"]), float(tb["hi"]))
    sf = (float(sf[0]), float(sf[1])) if sf[0] is not None else (None, None)
    return {"t": t_iv, "sf": sf, "wide": SC.hull((float(cl[0]), float(cl[1])), (float(item[0]), float(item[1])), t_iv, sf)}


def widest_reading(iv):
    return C.reading(iv[0], iv[1])


_orig_contrast_stats = RS.contrast_stats


def contrast_stats4(d_by_seed, dbs, boot, B):
    """r3_seeds.contrast_stats plus the four-interval hull of A2 for the replication (run-averaged vector) and for the locked run."""
    out = _orig_contrast_stats(d_by_seed, dbs, boot, B)
    avg_vec = RS.rep_avg(d_by_seed)
    out["rep4"] = small4(avg_vec, dbs, out["ci"], out["item_ci"])
    if 0 in d_by_seed and out.get("s0"):
        s0 = out["s0"]
        out["s0_4"] = small4(d_by_seed[0], dbs, s0["ci"], s0["item_ci"])
        out["agrees4"] = widest_reading(out["rep4"]["wide"]) == widest_reading(out["s0_4"]["wide"])
        out["agrees_old"] = bool(out["agrees"]["wide"]) if out.get("agrees") else None
    return out


def part_widest(runs, B, out):
    RS.contrast_stats = contrast_stats4
    D, res = RS.run(runs, B)
    sx = SX.run(runs, B)
    RS.contrast_stats = _orig_contrast_stats
    slim = lambda c: {k: c[k] for k in ("est", "ci", "item_ci", "range", "t_ci", "wide", "rep4", "s0", "s0_4", "agrees", "agrees4", "agrees_old", "n_runs", "per") if k in c}
    obj = {"definition": "hull of the database-clustered bootstrap, item-level bootstrap, cluster-robust t(G-1) and exact sign-flip intervals (size-weighted database means); "
                         "replication = run-averaged per-question differences of seeds 101, 202, 303; the range over the runs is reported beside, not inside the hull",
           "n_qids": len(D.qids), "contrasts": {k: slim(v) for k, v in res["contrasts"].items()}, "samex": {k: slim(v) for k, v in sx["contrasts"].items()}}
    dump(out, "widest.json", obj)
    # regeneration audit with the net questions carried by the questions whose root differs (A4)
    reg = {}
    for b in C.BK:
        for s in C.SEEDS:
            if not all((b, a, s) in D.ex for a in ("gated", "fixed", "rootonly")):
                continue
            roots = {}
            for a in ("gated", "fixed", "rootonly"):
                p = os.path.join(runs, C.run_path(b, a, s), "cands.jsonl")
                roots[a] = {r["qid"]: r for r in C.jl(p) if r.get("cand_idx") == 0}
            e = {"n": len(D.qids)}
            for x, y in (("gated", "fixed"), ("gated", "rootonly"), ("fixed", "rootonly")):
                m = np.array([roots[x].get(q, {}).get("sql", "").strip() != roots[y].get(q, {}).get("sql", "").strip() for q in D.qids])
                dv = D.ex[(b, x, s)] - D.ex[(b, y, s)]
                e["%s_%s" % (x, y)] = {"n_diff_root": int(m.sum()), "share_diff_root": float(m.mean()), "net_q_all": float(dv.sum()), "net_q_diff_root": float(dv[m].sum()),
                                       "net_q_same_root": float(dv[~m].sum()), "n_same_root": int((~m).sum()),
                                       "est_all_pp": float(100.0 * dv.mean()), "est_same_root_pp": float(100.0 * dv[~m].mean()) if (~m).any() else None}
            reg["%s|%d" % (b, s)] = e
    dump(out, "regen.json", {"n_qids": len(D.qids), "cells": reg})


# ------------------------------------------------------------------------------------------------------------ Critic increment cost (A1)
def icer(dex_pp, dex_ci, dcost_q):
    """cost per extra correct answer = (extra CNY per question) / (EX difference as a share). Defined only for a positive point estimate and a positive extra cost; the
    interval is [dcost / upper end, dcost / lower end] and unbounded above when the lower end of the EX interval is not above zero; undefined when the upper end is not above
    zero. -> dict(defined, est, lo, hi (None = unbounded), reason)"""
    if not (dex_pp > 0):
        return {"defined": False, "est": None, "lo": None, "hi": None, "reason": "increment not positive (%.2f pp)" % dex_pp}
    if not (dcost_q > 0):
        return {"defined": False, "est": None, "lo": None, "hi": None, "reason": "not more expensive than the reference (%.4f CNY per question)" % dcost_q}
    est = dcost_q / (dex_pp / 100.0)
    if not (dex_ci[1] > 0):
        return {"defined": True, "est": est, "lo": None, "hi": None, "reason": "interval of the increment lies at or below zero: no price is attainable"}
    lo = dcost_q / (dex_ci[1] / 100.0)
    hi = (dcost_q / (dex_ci[0] / 100.0)) if dex_ci[0] > 0 else None
    return {"defined": True, "est": est, "lo": lo, "hi": hi, "reason": "none" if hi is not None else "interval of the increment includes zero: unbounded above"}


def critinc_vectors(runs, b):
    """per question: (fixed final - fixed root) - (gated final - gated root), seed 0, same code as r3_addendum.part_critinc."""
    parts = {}
    for arm in ("gated", "fixed"):
        rows = C.load_preds(runs, C.run_path(b, arm, 0))
        roots = {r["qid"]: r for r in C.jl(os.path.join(runs, C.run_path(b, arm, 0), "cands.jsonl")) if r.get("cand_idx") == 0}
        parts[arm] = (rows, roots)
    qs = sorted(set(parts["gated"][0]) & set(parts["fixed"][0]))
    vec = np.array([(C.ex_of(parts["fixed"][0][q]) - float(parts["fixed"][1][q].get("ex") or 0)) - (C.ex_of(parts["gated"][0][q]) - float(parts["gated"][1][q].get("ex") or 0)) for q in qs], float)
    dbs = np.array([parts["gated"][0][q]["db_id"] for q in qs])
    return vec, dbs


def final_vectors(runs, b, arm_a, arm_b):
    """per question: EX of the final answer of arm_a minus that of arm_b (locked run, seed 0), on the questions both runs scored."""
    pa = C.load_preds(runs, C.run_path(b, arm_a, 0))
    pb = C.load_preds(runs, C.run_path(b, arm_b, 0))
    qs = sorted(q for q in set(pa) & set(pb) if C.ex_of(pa[q]) is not None and C.ex_of(pb[q]) is not None)
    vec = np.array([C.ex_of(pa[q]) - C.ex_of(pb[q]) for q in qs], float)
    return vec, np.array([pa[q]["db_id"] for q in qs])


def price_block(vec, dbs, dq_disc, dq_nodisc, B):
    """difference of final answers (pp), its four intervals and the cost per extra correct answer for both price bases"""
    est, cl, it = C.boot_ci(vec, dbs, B=B)
    s4 = small4(vec, dbs, cl, it)
    e = {"n": int(len(vec)), "net_q": float(vec.sum()), "dex": est, "cl": cl, "item": it, "t": s4["t"], "sf": s4["sf"], "wide": s4["wide"]}
    for nm, dq in (("disc", dq_disc), ("nodisc", dq_nodisc)):
        e["dcost_q_" + nm] = dq
        e["dcost_1000_" + nm] = 1000.0 * dq
        for iv in ("cl", "item", "wide"):
            e["icer_%s_%s" % (nm, iv)] = icer(est, e[iv], dq)
    e["extra_correct_1000"] = 10.0 * est
    e["extra_correct_1000_ci"] = tuple(10.0 * x for x in cl)
    e["extra_correct_1000_wide"] = tuple(10.0 * x for x in s4["wide"])
    return e


def part_critic(runs, rv2_path, B, out):
    kk = json.load(open(rv2_path))["keys"]
    res = {}
    for b in C.BK:
        vec, dbs = critinc_vectors(runs, b)
        est, cl, it = C.boot_ci(vec, dbs, B=B)
        s4 = small4(vec, dbs, cl, it)
        cost = lambda c, mode: float(kk["cost.%s.%s.%s" % (b, c, mode)]["v"])
        e = {"n": int(len(vec)), "net_q": float(vec.sum()), "dex": est, "cl": cl, "item": it, "t": s4["t"], "sf": s4["sf"], "wide": s4["wide"]}
        for mode, nm in (("cny_q", "disc"), ("cny_q_nodisc", "nodisc")):
            dq = cost("fixed", mode) - cost("gated", mode)
            e["dcost_q_" + nm] = dq
            e["dcost_1000_" + nm] = 1000.0 * dq
            for iv in ("cl", "item", "wide"):
                e["icer_%s_%s" % (nm, iv)] = icer(est, e[iv], dq)
        e["extra_correct_1000"] = 10.0 * est
        e["extra_correct_1000_ci"] = tuple(10.0 * x for x in cl)
        e["extra_correct_1000_wide"] = tuple(10.0 * x for x in s4["wide"])
        # round 5 (A2): one basis for the price of a step = the difference of the FINAL answers of the two arms; the within-run increment above stays as a second quantity
        fv, fd = final_vectors(runs, b, "fixed", "gated")
        e["fa"] = price_block(fv, fd, cost("fixed", "cny_q") - cost("gated", "cny_q"), cost("fixed", "cny_q_nodisc") - cost("gated", "cny_q_nodisc"), B)
        rv, rd = final_vectors(runs, b, "gated", "rootonly")
        e["retry"] = price_block(rv, rd, cost("gated", "cny_q") - cost("rootonly", "cny_q"), cost("gated", "cny_q_nodisc") - cost("rootonly", "cny_q_nodisc"), B)
        res[b] = e
    dump(out, "critic_cost.json", res)


# ------------------------------------------------------------------------------------------------------------ margin ratio and share kept in the replication (A3)
def eff_in_vectors(runs, seeds):
    """per backbone and arm: mean over `seeds` of (final EX - root EX) of the same run, aligned on the common questions; plus dbs."""
    out, dbs, qids = {}, None, None
    for b in C.BK:
        for arm in ("gated", "fixed"):
            vs = []
            for s in seeds:
                rel = C.run_path(b, arm, s)
                rows = C.load_preds(runs, rel)
                roots = {r["qid"]: r for r in C.jl(os.path.join(runs, rel, "cands.jsonl")) if r.get("cand_idx") == 0}
                q = sorted(q for q in rows if q in roots and C.ex_of(rows[q]) is not None)
                vs.append({x: C.ex_of(rows[x]) - float(roots[x].get("ex") or 0) for x in q})
                if dbs is None:
                    dbs = {x: rows[x]["db_id"] for x in q}
            common = sorted(set.intersection(*[set(v) for v in vs]))
            qids = common if qids is None else sorted(set(qids) & set(common))
            out[(b, arm)] = vs
    qids = sorted(qids)
    vec = {k: np.mean([np.array([v[q] for q in qids], float) for v in vs], axis=0) for k, vs in out.items()}
    return vec, np.array([dbs[q] for q in qids])


INF = float("inf")


def hull_open(*ivs):
    """hull of intervals whose ends may be None (= not bounded on that side: -inf for a lower, +inf for an upper end)"""
    lo = min((-INF if x[0] is None else x[0]) for x in ivs)
    hi = max((INF if x[1] is None else x[1]) for x in ivs)
    return (None if lo == -INF else lo, None if hi == INF else hi)


def _db_means(x, dbs):
    u = sorted(set(dbs.tolist()))
    n_g = np.array([float((dbs == g).sum()) for g in u])
    return n_g, np.array([float(x[dbs == g].mean()) for g in u])


def _p_t(m_g, n_g):
    """two-sided p of H0: item-pooled mean = 0, CR1 variance and t(G-1) (same estimator as small_cluster.cr_t, from database-level means)"""
    G, n = len(m_g), float(n_g.sum())
    est = float((n_g * m_g).sum() / n)
    se = math.sqrt(G / (G - 1.0) * float(((n_g * (m_g - est)) ** 2).sum())) / n
    if not (se > 0):
        return 0.0 if est != 0 else 1.0
    return 2.0 * SC.t_sf(abs(est / se), G - 1)


def ratio_interval(g, f, dbs, kind, hi_max=6.0, step=0.002):
    """interval of R = mean(g) / mean(f) by test inversion: {R : H0 'mean(g - R f) = 0' is not rejected at 5 %}, with kind 't' (CR1 variance, t(G-1)) or 'sf' (exact sign flip over the
    database-level values, weights n_g / n). At R = 1 the test is the test of the difference g - f, so R = 1 lies in the interval exactly when that difference's interval holds 0.
    Search outward from the point estimate on a grid, then bisect the edge; the lower end is cut at 0, an upper end not reached below hi_max is None (unbounded)."""
    g, f = np.asarray(g, float), np.asarray(f, float)
    n_g, mg = _db_means(g, dbs)
    _, mf = _db_means(f, dbs)
    w = n_g / n_g.sum()

    def p(R):
        m = mg - R * mf
        return _p_t(m, n_g) if kind == "t" else SC.sf_p(m, w, 0.0)
    R0 = float((n_g * mg).sum() / (n_g * mf).sum())

    def edge(sign):
        a, x = R0, R0
        while True:
            x = x + sign * step
            if x <= 0:
                return 0.0
            if x > hi_max:
                return None
            if p(x) <= SC.ALPHA:
                break
            a = x
        b = x
        for _ in range(40):
            m_ = 0.5 * (a + b)
            if p(m_) > SC.ALPHA:
                a = m_
            else:
                b = m_
        return a
    return (edge(-1.0), edge(1.0))


def margin_block(vec, dbs, B):
    fx = np.mean([vec[(b, "fixed")] for b in C.BK], axis=0)
    gt = np.mean([vec[(b, "gated")] for b in C.BK], axis=0)
    boot = S.ClusterBoot(dbs, B=B)
    ef, eg = 100.0 * fx.mean(), 100.0 * gt.mean()
    cb_f, cb_g = 100.0 * boot.means(fx), 100.0 * boot.means(gt)
    ib_f, ib_g = 100.0 * S.item_boot_means(fx, B=B), 100.0 * S.item_boot_means(gt, B=B)
    ok = lambda f: np.where(f > 0, f, np.nan)
    out = {"n": int(len(dbs)), "eff_fixed": ef, "eff_gated": eg, "eff_fixed_ci": S.pctl_ci(cb_f), "eff_fixed_item_ci": S.pctl_ci(ib_f), "eff_gated_ci": S.pctl_ci(cb_g), "eff_gated_item_ci": S.pctl_ci(ib_g),
           "margin_over_eff": MARGIN / ef if ef > 0 else None, "share_kept": eg / ef if ef > 0 else None}
    r_cl, r_it = MARGIN / ok(cb_f), MARGIN / ok(ib_f)
    out["margin_over_eff_ci"] = S.pctl_ci(r_cl[np.isfinite(r_cl)])
    out["margin_over_eff_item_ci"] = S.pctl_ci(r_it[np.isfinite(r_it)])
    s_cl, s_it = cb_g / ok(cb_f), ib_g / ok(ib_f)
    out["share_kept_ci"] = S.pctl_ci(s_cl[np.isfinite(s_cl)])
    out["share_kept_item_ci"] = S.pctl_ci(s_it[np.isfinite(s_it)])
    out["share_defined_draws"] = float(np.mean(np.isfinite(s_cl)))
    # round 5 (A3): the two small-cluster intervals; the 'wide' entries are the hull of all four, the definition of the Methods
    f4 = small4(fx, dbs, tuple(out["eff_fixed_ci"]), tuple(out["eff_fixed_item_ci"]))
    out["eff_fixed_t"], out["eff_fixed_sf"] = f4["t"], f4["sf"]
    for tag in ("t", "sf"):
        lo, hi = f4[tag]
        out["margin_over_eff_" + tag] = (MARGIN / hi if hi > 0 else None, MARGIN / lo if lo > 0 else None)
    out["share_kept_t"] = ratio_interval(gt, fx, dbs, "t")
    out["share_kept_sf"] = ratio_interval(gt, fx, dbs, "sf")
    out["margin_over_eff_wide"] = hull_open(out["margin_over_eff_ci"], out["margin_over_eff_item_ci"], out["margin_over_eff_t"], out["margin_over_eff_sf"])
    out["share_kept_wide"] = hull_open(out["share_kept_ci"], out["share_kept_item_ci"], out["share_kept_t"], out["share_kept_sf"])
    # the within-run difference (gated effect minus fixed effect): its interval decides whether a share of 1 is excluded
    gap = gt - fx
    gcl, git = S.pctl_ci(100.0 * boot.means(gap)), S.pctl_ci(100.0 * S.item_boot_means(gap, B=B))
    g4 = small4(gap, dbs, tuple(gcl), tuple(git))
    out["gap"] = 100.0 * float(gap.mean())
    out["gap_ci"], out["gap_item_ci"], out["gap_t"], out["gap_sf"], out["gap_wide"] = gcl, git, g4["t"], g4["sf"], g4["wide"]
    return out


def part_margin(runs, B, out):
    res = {}
    v0, d0 = eff_in_vectors(runs, (0,))
    res["s0"] = margin_block(v0, d0, B)
    per = {}
    vr, dr = eff_in_vectors(runs, C.REP)
    res["rep"] = margin_block(vr, dr, B)
    for s in C.REP:
        vs, ds = eff_in_vectors(runs, (s,))
        per[s] = margin_block(vs, ds, B)
    res["rep_runs"] = {str(s): {"margin_over_eff": per[s]["margin_over_eff"], "share_kept": per[s]["share_kept"], "eff_fixed": per[s]["eff_fixed"], "eff_gated": per[s]["eff_gated"]} for s in per}
    mo = [per[s]["margin_over_eff"] for s in per if per[s]["margin_over_eff"] is not None]
    sk = [per[s]["share_kept"] for s in per if per[s]["share_kept"] is not None]
    res["rep_range"] = {"margin_over_eff": (min(mo), max(mo)) if mo else None, "share_kept": (min(sk), max(sk)) if sk else None}
    dump(out, "margin_rep.json", res)


# ------------------------------------------------------------------------------------------------------------ MDE at question and database level (A6)
def part_mde(runs, B, out):
    from scipy import stats as st
    rows = {}
    for (b, a), rel in C.S0.items():
        rows[(b, a)] = C.load_preds(runs, rel)
    qids = sorted(set.intersection(*[set(k for k, x in r.items() if C.ex_of(x) is not None) for r in rows.values()]))
    ex = {k: np.array([C.ex_of(r[q]) for q in qids], float) for k, r in rows.items()}
    dbs = np.array([rows[("plus", "zs")][q]["db_id"] for q in qids])
    boot = S.ClusterBoot(dbs, B=B)
    n, G = len(qids), boot.k
    zs = Z975 + Z80
    ts = st.t.ppf(0.975, G - 1) + st.t.ppf(0.80, G - 1)
    res = {"n": n, "G": G, "z_sum": zs, "t_sum": float(ts), "tests": {}}

    def row(name, d, dbs_, G_, unit):
        item = zs * float(np.std(d, ddof=1)) / math.sqrt(len(d)) * 100.0
        ts_ = st.t.ppf(0.975, G_ - 1) + st.t.ppf(0.80, G_ - 1)
        bt = boot if dbs_ is dbs else S.ClusterBoot(dbs_, B=B)
        se = float(np.std(bt.means(d), ddof=1)) * 100.0
        res["tests"][name] = {"unit": unit, "n": int(len(d)), "G": int(G_), "est_pp": 100.0 * float(d.mean()), "sd_pp": 100.0 * float(np.std(d, ddof=1)), "mde_item_pp": item,
                              "se_cluster_pp": se, "t_sum": float(ts_), "mde_cluster_pp": float(ts_) * se}
    row("P1p", np.mean([ex[(b, "gated")] - ex[(b, "fixed")] for b in C.BK], axis=0), dbs, G, "EX difference, pp")
    row("W1", np.mean([ex[(b, "gated")] - ex[(b, "zs")] for b in ("flash", "27b")], axis=0), dbs, G, "EX difference, pp")
    pop = RR.p5_population(runs)
    d5 = np.array([r["gated"] - r["blind"] for r in pop], float)
    db5 = np.array([r["db"] for r in pop])
    row("P5p", d5, db5, len(set(db5.tolist())), "repair-rate difference on failing roots, pp")
    # P3: AUROC difference, DeLong SE (question level) and the cluster bootstrap SD (database level)
    from analysis_pipeline import primary
    run = C.load_preds(runs, "MIN/M1/gated-route-s0")
    rr = [run[q] for q in sorted(run)]
    coef = json.load(open(os.path.join(os.path.dirname(os.path.abspath(runs.rstrip("/"))), "exp/frozen/route_coef_hr.json")))
    v3, ca = primary.route_scores(rr, coef, 4)
    y = np.array([C.ex_of(r) if C.ex_of(r) is not None else -1 for r in rr])
    ok = np.isfinite(v3) & np.isfinite(ca) & (y >= 0)
    v3, ca, y = v3[ok], ca[ok], y[ok].astype(int)
    d3 = np.array([r["db_id"] for r in rr])[ok]
    dl = S.delong_paired(ca, v3, y)
    b3 = S.ClusterBoot(d3, B=B)
    dauc = b3.stat(lambda idx: S.auroc(ca[idx], y[idx]) - S.auroc(v3[idx], y[idx]))
    se_cl = float(np.std(dauc, ddof=1))
    res["tests"]["P3"] = {"unit": "AUROC difference", "n": int(ok.sum()), "G": int(b3.k), "est": float(dl["diff"]), "mde_item": zs * float(dl["se"]), "se_item": float(dl["se"]),
                          "se_cluster": se_cl, "t_sum": float(st.t.ppf(0.975, b3.k - 1) + st.t.ppf(0.80, b3.k - 1)),
                          "mde_cluster": float((st.t.ppf(0.975, b3.k - 1) + st.t.ppf(0.80, b3.k - 1)) * se_cl)}
    dump(out, "mde_cluster.json", res)


# ------------------------------------------------------------------------------------------------------------ P5' units, silent errors (A7)
def part_p5(runs, B, out):
    pop = RR.p5_population(runs)
    qids = sorted(set(r["qid"] for r in pop))
    per_q = {q: [r for r in pop if r["qid"] == q] for q in qids}
    nb = np.array([len(v) for v in per_q.values()])
    res = {"n_roots": len(pop), "n_questions": len(qids), "fail_on": {str(k): int((nb == k).sum()) for k in (1, 2, 3)},
           "by_backbone": {b: int(sum(1 for r in pop if r["b"] == b)) for b in C.BK}}
    g = np.array([np.mean([r["gated"] for r in per_q[q]]) for q in qids])
    bl = np.array([np.mean([r["blind"] for r in per_q[q]]) for q in qids])
    dbs = np.array([per_q[q][0]["db"] for q in qids])
    d = g - bl
    boot = S.ClusterBoot(dbs, B=B)
    res["question_unit"] = {"n": len(qids), "n_dbs": int(boot.k), "rate_gated": 100.0 * float(g.mean()), "rate_blind": 100.0 * float(bl.mean()), "diff_pp": 100.0 * float(d.mean()),
                           "ci": tuple(100.0 * x for x in S.pctl_ci(boot.means(d))), "item_ci": tuple(100.0 * x for x in S.pctl_ci(S.item_boot_means(d, B=B))),
                           "b_questions": int((d > 0).sum()), "c_questions": int((d < 0).sum())}
    mc = S.mcnemar_exact(res["question_unit"]["b_questions"], res["question_unit"]["c_questions"])
    res["question_unit"]["p_two"] = mc["p_two"]
    # one backbone at a time
    for b in C.BK:
        rb = [r for r in pop if r["b"] == b]
        a_, b_ = np.array([r["gated"] for r in rb]), np.array([r["blind"] for r in rb])
        res["by_backbone_" + b] = {"n": len(rb), "diff_pp": 100.0 * float((a_ - b_).mean()), "repaired_gated": int(a_.sum()), "repaired_blind": int(b_.sum()),
                                   "b": int(((a_ == 1) & (b_ == 0)).sum()), "c": int(((a_ == 0) & (b_ == 1)).sum())}
    dump(out, "p5_units.json", res)


def part_silent(runs, B, out):
    res = {}
    for b in C.BK:
        for arm in ("zs", "rootonly", "gated", "fixed"):
            rows = C.load_preds(runs, C.run_path(b, arm, 0))
            q = sorted(k for k, r in rows.items() if C.ex_of(r) is not None)
            ex = np.array([C.ex_of(rows[k]) for k in q], float)
            okx = np.array([1.0 if rows[k].get("exec_ok") else 0.0 for k in q])
            dbs = np.array([rows[k]["db_id"] for k in q])
            silent = ((okx == 1) & (ex == 0)).astype(float)
            failed = ((okx == 0) & (ex == 0)).astype(float)
            n_wrong = int((ex == 0).sum())
            boot = S.ClusterBoot(dbs, B=B)
            res["%s|%s" % (b, arm)] = {"n": len(q), "n_wrong": n_wrong, "n_silent": int(silent.sum()), "n_exec_fail_wrong": int(failed.sum()),
                                       "silent_rate_pct": 100.0 * float(silent.mean()), "silent_rate_ci": tuple(100.0 * x for x in S.pctl_ci(boot.means(silent))),
                                       "silent_share_of_wrong_pct": 100.0 * float(silent.sum() / max(n_wrong, 1)), "reach_share_pct": 100.0 * float(failed.sum() / max(n_wrong, 1)),
                                       "ex_pct": 100.0 * float(ex.mean())}
    # pooled over the three backbones, root-only arm (the root is the answer): the share of wrong roots the gate can see
    for arm in ("rootonly", "gated"):
        w = sum(res["%s|%s" % (b, arm)]["n_wrong"] for b in C.BK)
        f = sum(res["%s|%s" % (b, arm)]["n_exec_fail_wrong"] for b in C.BK)
        s_ = sum(res["%s|%s" % (b, arm)]["n_silent"] for b in C.BK)
        res["pooled|" + arm] = {"n_wrong": w, "n_exec_fail_wrong": f, "n_silent": s_, "reach_share_pct": 100.0 * f / max(w, 1), "silent_share_of_wrong_pct": 100.0 * s_ / max(w, 1)}
    dump(out, "silent.json", res)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--rv2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=C.B_DEFAULT)
    ap.add_argument("--part", default="all")
    a = ap.parse_args(argv)
    parts = ("widest", "critic", "margin", "mde", "p5", "silent") if a.part == "all" else (a.part,)
    for p in parts:
        print("[r4_stats] part", p, flush=True)
        if p == "widest":
            part_widest(a.runs, a.B, a.out)
        elif p == "critic":
            part_critic(a.runs, a.rv2, a.B, a.out)
        elif p == "margin":
            part_margin(a.runs, a.B, a.out)
        elif p == "mde":
            part_mde(a.runs, a.B, a.out)
        elif p == "p5":
            part_p5(a.runs, a.B, a.out)
        elif p == "silent":
            part_silent(a.runs, a.B, a.out)
        else:
            raise SystemExit("unknown part " + p)


if __name__ == "__main__":
    main()
