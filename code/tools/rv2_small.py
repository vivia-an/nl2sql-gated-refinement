#!/usr/bin/env python3
"""S2 statistics add-on (post hoc, robustness; internal note S2, SAP section 25). No model or API call. Scope `posthoc-secondary`.

    worker machine:  PYTHONPATH=<code> python code/tools/rv2_small.py --runs $R/runs --data $R/data --results <host results copy> --exp <exp dir> --out <out dir> [--B 10000]

Re-computes the RV2 contrasts with the same code and the same seeds as `rv2_analyze.py` (it imports that module and wraps its estimators to keep the
per-item vectors), checks every estimate and cluster interval against the published `results/posthoc2/dual.csv`, and writes `rv2s_summary.json`
(grammar `rv2s_keys.py`) and `small.csv`. Nothing in `rv2_summary.json` or in the confirmatory results is touched.

  S1 small.*     t(G-1) and exact sign-flip readings for every dual contrast and loop effect; the confirmatory family P* re-tested
  S2 deff.*      design-effect audit (all contrasts, subsets, the four-pair table of setup.json re-computed)
  S3 margin.*    1.5 pp margin divided by the loop effect, with intervals
  S4 cost.nodisc.*  cost ratios with and without the cache discount; front; route per correct answer; cached-token audit
  S5 gate.*      gate-fire share and helped / hurt counts with a sign test
"""
import argparse
import collections
import csv
import io
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import rv2_analyze as A  # noqa: E402
import rv2s_keys as KS  # noqa: E402
import small_cluster as SC  # noqa: E402
from analysis_pipeline import runs as R, stats as S  # noqa: E402

BK = A.BK
OUT = collections.OrderedDict()
KINDS = dict(KS.expected_keys())
LOG = []
CAP = collections.OrderedDict()            # dual id -> pack dict (with _d/_dbs when the contrast is an item mean, _loo always)
NULLS = {"NI(1.5)": -1.5, "RATIO(0.80)": 0.8, "RATIO(1)": 1.0}


def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print("[rv2s] " + s, flush=True)


def put(key, v, ci=None, item_ci=None, n=None, meaning=""):
    if key not in KINDS:
        raise KeyError("key not in the grammar: " + key)
    if key in OUT:
        raise KeyError("duplicate key " + key)
    OUT[key] = {"v": A._f(v), "kind": KINDS[key], "ci": A._f(ci), "item_ci": A._f(item_ci), "n": A._f(n), "meaning": meaning}


# ------------------------------------------------------------------------------------------------ wrappers (keep per-item vectors, leave rv2_analyze unchanged)
def install_wrappers():
    o_pack, o_est_mean, o_put_dual = A.pack, A.est_mean, A.put_dual

    def pack_w(est, bt, ib, loo, g, n):
        e = o_pack(est, bt, ib, loo, g, n)
        e["_loo"] = [float(x) for x in loo] if loo else []
        return e

    def est_mean_w(d, dbs, B, scale=100.0):
        e = o_est_mean(d, dbs, B, scale)
        e["_d"] = scale * np.asarray(d, float)
        e["_dbs"] = np.asarray(dbs)
        return e

    def put_dual_w(i, e, mcn=None):
        CAP[i] = e
        o_put_dual(i, e, mcn)
    A.pack, A.est_mean, A.put_dual = pack_w, est_mean_w, put_dual_w


# ------------------------------------------------------------------------------------------------ S1
def classes_for(i):
    return A.rule_of(i)            # (rule name, fn(lo, hi, est))


def small_block(i, e, d_kind_unit=None):
    """compute and store small.<id>.* from a pack dict e. Additive (item-mean) contrasts carry _d/_dbs; everything else uses jackknife pseudo-values."""
    rname, fn = classes_for(i)
    null = NULLS.get(rname, 0.0)
    est = e["est"]
    G = e["g"]
    cl, it = tuple(e["ci"]), tuple(e["item_ci"])
    additive = "_d" in e
    if additive:
        d, dbs = e["_d"], e["_dbs"]
        u, n_g, m_g = SC.db_level(d, dbs)
        w = n_g / n_g.sum()
        tb = SC.cr_t(d, dbs, null)
        tdb = SC.t_db(d, dbs, null)
        sf_w = SC.sf_ci(m_g, w)
        sf_u = SC.sf_ci(m_g, None)
        p_sf = SC.sf_p(m_g, w, null, "two")
        p_sfu = SC.sf_p(m_g, None, null, "two")
        est_u = float(m_g.mean())
        method = "item-pooled mean; cluster-robust t(G-1); sign-flip over database means (weights n_g/n and 1/G)"
        sfv = (m_g, w)
        sfuv = (m_g, None)
        deff_cr = SC.deff_cr(d, dbs)
    else:
        psi = SC.pseudo(est, e["_loo"], G)
        se = float(np.std(psi, ddof=1)) / math.sqrt(len(psi))
        tb = SC._t_block(est, se, G - 1, null)
        tdb = tb
        sf_w = SC.sf_ci(psi, None)
        sf_u = sf_w
        p_sf = SC.sf_p(psi, None, null, "two")
        p_sfu = p_sf
        est_u = est
        method = "delete-one-database jackknife pseudo-values; t(G-1) on their mean; sign-flip over the pseudo-values (equal weights)"
        sfv = (psi, None)
        sfuv = sfv
        deff_cr = None
    t_iv = (tb["lo"], tb["hi"])
    tdb_iv = (tdb["lo"], tdb["hi"])
    ivs = {"cl": cl, "item": it, "t": t_iv, "tdb": tdb_iv, "sf": tuple(sf_w), "sfu": tuple(sf_u)}
    wide = SC.hull(*[ivs[k] for k in ("cl", "item", "t", "sf")])
    wideu = SC.hull(*[ivs[k] for k in ("cl", "item", "t", "tdb", "sf", "sfu")])
    old = SC.hull(cl, it)
    ivs["wide"], ivs["wideu"], ivs["old"] = wide, wideu, old
    pre = "small.%s." % i
    put(pre + "est", est, n=e["n"]); put(pre + "n", e["n"]); put(pre + "g", G)
    put(pre + "method", method); put(pre + "rule", rname); put(pre + "null", null)
    for k, iv in ivs.items():
        put(pre + k + "_ci", list(iv) if iv[0] is not None else "n/a")
    put(pre + "p_t", tb["p2"]); put(pre + "p_tdb", tdb["p2"]); put(pre + "p_sf", p_sf); put(pre + "p_sfu", p_sfu)
    cls = {}
    for k, iv in ivs.items():
        eu = est_u if k in ("tdb", "sfu") and additive else est
        cls[k] = fn(iv[0], iv[1], eu) if iv[0] is not None else "n/a"
        put(pre + "class_" + k, cls[k])
    put(pre + "reading_changes", "yes" if cls["wide"] != cls["old"] else "no")
    put(pre + "reading_changes_u", "yes" if cls["wideu"] != cls["old"] else "no")
    return {"id": i, "rule": rname, "est": est, "g": G, "n": e["n"], "deff": e.get("deff"), "deff_cr": deff_cr, "ivs": ivs, "cls": cls, "p_t": tb["p2"], "p_sf": p_sf,
            "p_sfu": p_sfu, "additive": additive, "sfv": sfv, "sfuv": sfuv, "tb": tb, "t_se_null": null}


def pstar_blocks(meta, W1, PR, SMALL):
    """re-test of the confirmatory family P* (one-sided p, doubled for Holm as in SAP 15.3) with exact sign-flip and t."""
    S_ = SC
    one = {}

    def add_pid(pid, p_boot, p_item, sf1, sfu1, t1, ok):
        put("small.pstar.%s.p_boot" % pid, p_boot)
        if p_item is not None:
            put("small.pstar.%s.p_item" % pid, p_item)
        put("small.pstar.%s.p_sf1" % pid, sf1); put("small.pstar.%s.p_sfu1" % pid, sfu1); put("small.pstar.%s.p_t1" % pid, t1)
        put("small.pstar.%s.ok_wide" % pid, "yes" if ok else "no")

    # P1' EX end point: H0 dEX <= -1.5 (additive)
    s = SMALL["P1p.pooled"]
    m_g, w = s["sfv"]
    ex_sf1 = S_.sf_p(m_g, w, -1.5, "gt"); ex_sfu1 = S_.sf_p(m_g, None, -1.5, "gt")
    d, dbs = CAP["P1p.pooled"]["_d"], CAP["P1p.pooled"]["_dbs"]
    ex_t1 = S_.cr_t(d, dbs, -1.5)["p_gt"]
    ex_ok = s["ivs"]["wide"][0] > -1.5
    add_pid("P1p_ex", W1["P1prime.p_a"], None, ex_sf1, ex_sfu1, ex_t1, ex_ok)
    # P1' calls ratio: H0 ratio >= 0.80 (pseudo-values)
    s = SMALL["P1p_ratio.pooled"]
    psi = SC.pseudo(CAP["P1p_ratio.pooled"]["est"], CAP["P1p_ratio.pooled"]["_loo"], CAP["P1p_ratio.pooled"]["g"])
    c_sf1 = S_.sf_p(psi, None, 0.8, "lt")
    se = float(np.std(psi, ddof=1)) / math.sqrt(len(psi))
    c_t1 = S_._t_block(CAP["P1p_ratio.pooled"]["est"], se, len(psi) - 1, 0.8)["p_lt"]
    c_ok = s["ivs"]["wide"][1] < 0.8
    add_pid("P1p_calls", W1["P1prime.p_b"], None, c_sf1, c_sf1, c_t1, c_ok)
    add_pid("P1p", W1["P1prime.p_1s"], W1["P1prime.p_item_1s"], max(ex_sf1, c_sf1), max(ex_sfu1, c_sf1), max(ex_t1, c_t1), ex_ok and c_ok)
    # W1 (pooled flash/27b): H0 <= 0
    s = SMALL["W1.pooled"]
    m_g, w = s["sfv"]
    d, dbs = CAP["W1.pooled"]["_d"], CAP["W1.pooled"]["_dbs"]
    w_sf1, w_sfu1, w_t1 = S_.sf_p(m_g, w, 0.0, "gt"), S_.sf_p(m_g, None, 0.0, "gt"), S_.cr_t(d, dbs, 0.0)["p_gt"]
    add_pid("W1", W1["W1.p_1s"], W1["W1.p_item_1s"], w_sf1, w_sfu1, w_t1, s["ivs"]["wide"][0] > 0)
    # P3 (AUROC difference): H0 <= 0 (pseudo-values)
    e = CAP["P3"]
    psi = SC.pseudo(e["est"], e["_loo"], e["g"])
    se = float(np.std(psi, ddof=1)) / math.sqrt(len(psi))
    p_sf1 = S_.sf_p(psi, None, 0.0, "gt"); p_t1 = S_._t_block(e["est"], se, len(psi) - 1, 0.0)["p_gt"]
    add_pid("P3", PR["P3.p_1s"], PR["P3.p_item_1s"], p_sf1, p_sf1, p_t1, SMALL["P3"]["ivs"]["wide"][0] > 0)
    # family: doubled one-sided p, Holm m = 4; P5' keeps its exact McNemar p (not a cluster test)
    p5 = W1["P5prime.p_2s"]
    okw = {"P1p": SMALL["P1p.pooled"]["ivs"]["wideu"][0] > -1.5 and SMALL["P1p_ratio.pooled"]["ivs"]["wideu"][1] < 0.8, "W1": SMALL["W1.pooled"]["ivs"]["wideu"][0] > 0,
           "P3": SMALL["P3"]["ivs"]["wideu"][0] > 0, "P5p": False}
    ok = {"P1p": ex_ok and c_ok, "W1": SMALL["W1.pooled"]["ivs"]["wide"][0] > 0, "P3": SMALL["P3"]["ivs"]["wide"][0] > 0, "P5p": False}
    p2 = {"boot": {"P1p": W1["P1prime.p_2s"], "W1": W1["W1.p_2s"], "P3": W1["Pstar.P3.p_2s"], "P5p": p5},
          "sf": {"P1p": min(1.0, 2 * max(ex_sf1, c_sf1)), "W1": min(1.0, 2 * w_sf1), "P3": min(1.0, 2 * p_sf1), "P5p": p5},
          "sfu": {"P1p": min(1.0, 2 * max(ex_sfu1, c_sf1)), "W1": min(1.0, 2 * w_sfu1), "P3": min(1.0, 2 * p_sf1), "P5p": p5},
          "t": {"P1p": min(1.0, 2 * max(ex_t1, c_t1)), "W1": min(1.0, 2 * w_t1), "P3": min(1.0, 2 * p_t1), "P5p": p5}}
    adj = {k: S.holm(v, m=4) for k, v in p2.items()}
    for fam in KS.PSTAR_FAMILY:
        put("small.pstar.%s.p2_boot" % fam, p2["boot"][fam]); put("small.pstar.%s.padj_boot" % fam, W1[{"P1p": "P1prime.p_adj", "W1": "W1.p_adj", "P3": "Pstar.P3.p_adj", "P5p": "P5prime.p_adj"}[fam]])
        put("small.pstar.%s.p2_sf" % fam, p2["sf"][fam]); put("small.pstar.%s.padj_sf" % fam, adj["sf"][fam])
        put("small.pstar.%s.p2_sfu" % fam, p2["sfu"][fam]); put("small.pstar.%s.padj_sfu" % fam, adj["sfu"][fam])
        put("small.pstar.%s.holds_sfu" % fam, "yes" if (adj["sfu"][fam] < 0.05 and okw[fam]) else "no")
        put("small.pstar.%s.p2_t" % fam, p2["t"][fam]); put("small.pstar.%s.padj_t" % fam, adj["t"][fam])
        put("small.pstar.%s.holds_sf" % fam, "yes" if (adj["sf"][fam] < 0.05 and ok[fam]) else "no")
        put("small.pstar.%s.holds_t" % fam, "yes" if (adj["t"][fam] < 0.05 and ok[fam]) else "no")
    put("small.pstar.sf_min_p1", SC.sf_min_p(11, "one"))
    put("small.pstar.def", "Family P* (Holm over P1', W1, P3, P5', m = 4, one-sided p doubled, SAP 15.3) re-tested with the database as the unit: p_sf1 = exact one-sided "
        "sign-flip p over the 11 database-level values (weights n_g/n, the headline estimand; p_sfu1 with equal weights), p_t1 = one-sided cluster-robust t(10) p; P1' EX end point "
        "against -1.5 pp, P1' calls ratio against 0.80, W1 and P3 against 0; the ratio and the AUROC difference use delete-one-database jackknife pseudo-values. P1' needs both "
        "end points (maximum). P5' keeps its exact McNemar p (not a cluster test). holds_sf and holds_t = Holm-adjusted p < 0.05 AND the hull of cl, item, t, sf satisfies the rule; holds_sfu = the same with the equal-weight sign-flip p and the hull of all six intervals. "
        "The smallest attainable one-sided p is 2^-11 = 0.00049 (0.00098 doubled), against 0.0001 for the percentile bootstrap.")


# ------------------------------------------------------------------------------------------------ S2
def deff_blocks(F, SMALL, B):
    items = [(i, s["deff"], s["g"], s["rule"]) for i, s in SMALL.items() if i in KS.DUAL_IDS and s["deff"] is not None]
    assert len(items) == 110, len(items)

    def stats(sel, name):
        vals = np.array([x[1] for x in sel], float)
        put("deff.%s.n" % name, len(vals)); put("deff.%s.min" % name, vals.min()); put("deff.%s.max" % name, vals.max()); put("deff.%s.median" % name, float(np.median(vals)))
        put("deff.%s.n_lt1" % name, int((vals < 1).sum())); put("deff.%s.n_gt1" % name, int((vals > 1).sum())); put("deff.%s.n_gt2" % name, int((vals > 2).sum()))
    is_ratio = lambda i: i.startswith("P1p_ratio") or i in ("XB3_ratio", "PF3b", "PF3b_q100")
    is_aux = lambda i: i in ("P3", "P3_aurc", "P3_acc60", "P5p") or "C3_" in i
    stats(items, "all")
    stats([x for x in items if not is_ratio(x[0]) and not is_aux(x[0])], "ex")
    stats([x for x in items if is_ratio(x[0])], "ratio")
    stats([x for x in items if x[2] == 11], "g11"); stats([x for x in items if x[2] == 19], "g19")
    cr = [(i, s["deff_cr"], s["g"], s["rule"]) for i, s in SMALL.items() if i in KS.DUAL_IDS and s["deff_cr"] is not None]
    stats(cr, "cr")
    vals = np.array([x[1] for x in items], float)
    mn, mx = items[int(vals.argmin())], items[int(vals.argmax())]
    put("deff.all.min_id", mn[0]); put("deff.all.max_id", mx[0])
    gt2 = sorted([x for x in items if x[1] > 2], key=lambda x: -x[1])
    put("deff.all.ids_gt2", "; ".join("%s %.2f" % (x[0], x[1]) for x in gt2) or "none")
    lt = sorted([x for x in items if x[1] < 0.5], key=lambda x: x[1])
    put("deff.all.ids_lt0.5", "; ".join("%s %.2f" % (x[0], x[1]) for x in lt) or "none"); put("deff.all.n_lt0.5", len(lt))
    put("deff.all.share_lt1", float((vals < 1).mean()))
    # the four-pair table of setup.json: gated vs fixed (plus), gated vs zs (plus, flash, 27b), B = 2000, np.var (ddof 0), exactly as misc.run_power_constants
    stored = json.load(open(os.path.join(F.args.results, "setup.json")))
    pairs = (("p1_plus", ("plus", "gated"), ("plus", "fixed"), "full_gated_vs_full_fixed.qwen37plus", "P1p.plus"),
             ("w_plus", ("plus", "gated"), ("plus", "zs"), "full_gated_vs_zs.qwen37plus", "W1.plus"),
             ("w_flash", ("flash", "gated"), ("flash", "zs"), "full_gated_vs_zs.qwen37flash", "W1.flash"),
             ("w_27b", ("27b", "gated"), ("27b", "zs"), "full_gated_vs_zs.qwen3527b", "W1.27b"))
    v2000 = []
    for tag, ka, kb, stag, did in pairs:
        m = F.pair_mask(ka, kb)
        d = (F.ex[ka][m] - F.ex[kb][m]).astype(float)
        boot = S.ClusterBoot(F.dbs[m], 2000)
        vc, vi = float(np.var(boot.means(d))), float(np.var(S.item_boot_means(d, 2000)))
        v2000.append(vc / vi)
        put("deff.setup4.%s.b2000" % tag, vc / vi)
        put("deff.setup4.%s.b10000" % tag, SMALL[did]["deff"])
        put("deff.setup4.%s.stored" % tag, stored["setup.design_effect_emp." + stag])
        if abs(vc / vi - stored["setup.design_effect_emp." + stag]) > 0.006:
            log("WARNING: setup4 %s recomputed %.4f vs stored %.4f" % (tag, vc / vi, stored["setup.design_effect_emp." + stag]))
    put("deff.setup4.min", min(v2000)); put("deff.setup4.max", max(v2000))
    put("deff.setup4.source", "results/setup.json setup.design_effect_emp.* (code/analysis_pipeline/misc.py run_power_constants): four pairs (gated vs fixed on plus; gated vs single call on plus, flash, 27b), "
        "BIRD dev, 2 000 bootstrap draws, np.var (ddof 0) of the database-clustered bootstrap means over that of the item bootstrap means. Re-computed here to the printed values.")
    put("deff.source_of_larger_values", "The supplement tables (RV2 dual rows, B = 10 000, ddof 1) cover all 110 contrasts, among them the calls-ratio rows (1.5 to 3.1), the pooled and corrected-reference rows and the Spider-Realistic rows. "
        "The sentence of Section 3.2.3 quotes only the four pairs of setup.json, with 2 000 draws, so 1.69 (flash, gated vs single call) is 1.77 with 10 000 draws; the two numbers estimate the same quantity with different Monte Carlo noise.")


# ------------------------------------------------------------------------------------------------ S3
def fmt_ratio_iv(lo, hi, margin):
    """interval of margin / effect from an interval of the effect (monotone transform); the upper end is unbounded when the effect interval reaches 0."""
    if hi is None or hi <= 0:
        return None, None, "n/a (the interval of the effect lies at or below 0)"
    lo_r = margin / hi
    if lo is None or lo <= 0:
        return lo_r, None, "[%.2f, unbounded]" % lo_r
    hi_r = margin / lo
    return lo_r, hi_r, "[%.2f, %.2f]" % (lo_r, hi_r)


def margin_blocks(res, SMALL):
    M = 1.5
    for b in K_PB():
        for d in ("in", "btw"):
            for arm in ("fixed", "gated"):
                e = res[(b, d, arm)]
                p = "margin.%s.%s.%s." % (b, d, arm)
                est = e["est"]
                put(p + "eff", est)
                put(p + "ratio", (M / est) if est > 0 else "n/a")
                sm = SMALL["loop.%s.eff_%s.%s" % (b, d, arm)]
                ivs = {"cl": sm["ivs"]["cl"], "item": sm["ivs"]["item"], "t": sm["ivs"]["t"], "sf": sm["ivs"]["sf"]}
                ivs["wide"] = SC.hull(ivs["cl"], ivs["item"], ivs["t"], ivs["sf"])
                for t, iv in ivs.items():
                    lo_r, hi_r, txt = fmt_ratio_iv(iv[0], iv[1], M)
                    put(p + t + "_text", txt); put(p + "lo_" + t, lo_r if lo_r is not None else "n/a")
                    put(p + "hi_" + t, hi_r if hi_r is not None else ("unbounded" if lo_r is not None else "n/a"))
                    if t == "wide":
                        put(p + "wide_contains_1", "n/a" if lo_r is None else ("yes" if (lo_r <= 1.0 and (hi_r is None or hi_r >= 1.0)) else "no"))
                put(p + "p_nonpos_boot", float((np.asarray(e["bt"]) <= 0).mean()))
    # sensitivity of P1' to a margin that is a fraction of the loop effect
    ex = SMALL["P1p.pooled"]
    eff = res[("pooled", "in", "fixed")]["est"]
    for tag, mg in (("half_eff", 0.5 * eff), ("abs05", 0.5)):
        p = "margin.sens.%s." % tag
        put(p + "margin", mg)
        met = {}
        for t in ("cl", "item", "t", "sf"):
            lo = ex["ivs"][t][0]
            put(p + "lo_" + t, lo); met[t] = lo > -mg
            put(p + "met_" + t, "yes" if met[t] else "no")
        wide = ex["ivs"]["wide"]
        put(p + "lo_wide", wide[0]); put(p + "met_wide", "yes" if wide[0] > -mg else "no")
    put("margin.def", "Ratio = 1.5 pp / (loop effect), effect = final minus root candidate of the same run ('in') or final minus the separate root-only run ('btw'), 'fixed' = fixed loop, 'gated' = gated loop, "
        "pooled = mean of the three backbones. Each interval of the effect is mapped through x -> 1.5 / x (monotone while the effect is positive); when the effect interval reaches 0 the upper end is "
        "unbounded and the text says so. Intervals: cl = database-clustered percentile bootstrap, item = item bootstrap, t = cluster-robust t(G-1), sf = exact sign-flip inversion, wide = hull of these four. A wide interval that contains 1 means the data do not say whether the margin is larger or smaller than the effect it protects. "
        "margin.sens.*: P1' EX difference (gated minus fixed, pooled) against a margin of half the pooled fixed-loop effect ('half_eff') and against 0.5 pp ('abs05'): lo_* = lower end of the interval of that type, met_* = lo > -margin.")


def K_PB():
    return KS.PB


# ------------------------------------------------------------------------------------------------ S4
def item_cost2(run, qids, prices, disc):
    ratio = float(prices.get("cache_hit_ratio", 0.2))
    out = []
    for q in qids:
        r = run.rows.get(q)
        pr = A.price_of(prices, r.get("model")) if r else None
        if r is None or pr is None:
            out.append(0.0)
            continue
        pt, ct = r.get("prompt_tokens", 0) or 0, r.get("completion_tokens", 0) or 0
        cached = min(r.get("cached_tokens", 0) or 0, pt) if disc else 0
        out.append(((pt - cached) * pr["in"] + cached * pr["in"] * ratio + ct * pr["out"]) / 1e6)
    return np.array(out)


def cost_blocks(F, B, prices, rvec):
    refs = {"plus_zs": ("plus", "zs"), "plus_gated": ("plus", "gated")}
    tot = {}
    pts = {m: {} for m in KS.MODES}
    for b in BK:
        for cfg in KS.K.COST_CFG[b]:
            k = (b, cfg)
            r = F.runs[k]
            t = A.run_costs(r, prices)
            tot[k] = t
            n = len(r.rows)
            ex = F.ex[k]
            exm = float(ex[ex >= 0].mean())
            pts["disc"][k] = (t["cny_disc"] / n, exm)
            pts["nodisc"][k] = (t["cny_nodisc"] / n, exm)
            crow = [x for x in r.usage if x.get("kind") not in ("embed", "rerank")]
            rows_c = sum(1 for x in crow if (x.get("cached_tokens", 0) or 0) > 0)
            put("cost.nodisc.cached.%s.%s.share" % k, (t["cached"] / t["pt"]) if t["pt"] else 0.0)
            put("cost.nodisc.cached.%s.%s.rows_cached" % k, rows_c); put("cost.nodisc.cached.%s.%s.rows" % k, len(crow))
    # ratios with intervals (per-item costs from preds, as the Pareto claims of RV2)
    for b in BK:
        for cfg in KS.K.COST_CFG[b]:
            for ref in KS.COST_REFS:
                if ref == "plus_zs" and (b, cfg) == ("plus", "zs"):
                    continue
                if ref == "plus_gated" and (b, cfg) == ("plus", "gated"):
                    continue
                if ref == "self_zs" and cfg == "zs":
                    continue
                kref = refs[ref] if ref in refs else (b, "zs")
                k = (b, cfg)
                m = F.pair_mask(kref, k)
                qs = [q for q, mm in zip(F.qids, m) if mm]
                for mode in KS.MODES:
                    disc = mode == "disc"
                    ca, cb = item_cost2(F.runs[kref], qs, prices, disc), item_cost2(F.runs[k], qs, prices, disc)
                    er = A.est_ratio(cb, ca, F.dbs[m], B)
                    put("cost.nodisc.ratio.%s.%s.vs_%s.%s" % (b, cfg, ref, mode), er["est"], ci=er["ci"], item_ci=er["item_ci"], n=er["n"])
    for cl in KS.K.PARETO_CLAIMS:
        _, rest = cl.split("_zs_vs_")
        bb, arm = rest.split("_")
        for mode, suffix in (("nodisc", "cost_ratio"), ("disc", "cost_ratio_disc")):
            src = OUT["cost.nodisc.ratio.%s.%s.vs_plus_zs.%s" % (bb, arm, mode)]
            put("cost.nodisc.%s.%s" % (cl, suffix), src["v"], ci=src["ci"], item_ci=src["item_ci"], n=src["n"])
            put("cost.nodisc.%s_%s_vs_plus_zs.%s" % (bb, arm, suffix), src["v"], ci=src["ci"], item_ci=src["item_ci"], n=src["n"])
    # front and dominance
    fronts = {}
    for mode in KS.MODES:
        names = list(pts[mode])
        nd = []
        for k in names:
            c, e = pts[mode][k]
            dom = [o for o in names if o != k and pts[mode][o][0] <= c + 1e-12 and pts[mode][o][1] >= e - 1e-12 and (pts[mode][o][0] < c - 1e-12 or pts[mode][o][1] > e + 1e-12)]
            if mode == "nodisc":
                put("cost.nodisc.%s.%s.dominated_by" % k, ", ".join("%s %s" % x for x in dom) if dom else "none")
            if not dom:
                nd.append(k)
        fronts[mode] = sorted(nd, key=lambda x: pts[mode][x][0])
        put("cost.nodisc.front.%s.set" % mode, "; ".join("%s %s" % x for x in fronts[mode])); put("cost.nodisc.front.%s.k" % mode, len(nd))
    pub = json.load(open(os.path.join(F.args.results, "posthoc2", "rv2_summary.json")))["keys"]["pareto.nondominated"]["v"]
    mine = "; ".join("%s %s" % x for x in fronts["disc"])
    log("front with discount equals RV2 pareto.nondominated:", mine == pub)
    if mine != pub:
        log("WARNING front mismatch: mine=%s published=%s" % (mine, pub))
    gone, new = [k for k in fronts["disc"] if k not in fronts["nodisc"]], [k for k in fronts["nodisc"] if k not in fronts["disc"]]
    put("cost.nodisc.front.changed", ("same members with and without the discount" if not gone and not new else
                                      "without the discount: leaves the front %s; enters the front %s" % (", ".join("%s %s" % x for x in gone) or "none", ", ".join("%s %s" % x for x in new) or "none")))
    for cl in KS.K.PARETO_CLAIMS:
        _, rest = cl.split("_zs_vs_")
        bb, arm = rest.split("_")
        a_, b_ = ("plus", "zs"), (bb, arm)
        cA, eA = pts["nodisc"][a_]; cB, eB = pts["nodisc"][b_]
        point = (cA <= cB + 1e-12) and (eA >= eB - 1e-12) and (cA < cB - 1e-12 or eA > eB + 1e-12)
        point_rev = (cB <= cA + 1e-12) and (eB >= eA - 1e-12) and (cB < cA - 1e-12 or eB > eA + 1e-12)
        put("cost.nodisc.claim.%s.dominates" % cl,
            ("yes at point estimates (single call cheaper without the discount and EX not lower; EX interval as in RV2.pareto.claim)" if point else
             ("no: without the discount the pipeline is cheaper than the single call (cost %.4f vs %.4f CNY per question) and EX %.3f vs %.3f%s" % (cB, cA, eB, eA, "; it dominates the single call at point estimates" if point_rev else ""))
             if cB < cA else "no (cost %.4f vs %.4f, EX %.3f vs %.3f)" % (cA, cB, eA, eB)))
    # route
    v3, ca, y, ok = rvec
    sc = ca[ok]; yy = y[ok]
    n = len(yy)
    cum = S._expected_cum_acc(sc, yy)
    k60 = max(1, int(round(0.6 * n)))
    acc60 = float(cum[k60 - 1])
    per_asked = acc60 * k60 / n
    runs_route = {"gated": ("plus", "gated"), "bonvote2": ("plus", "bonvote2"), "bonvote3": ("plus", "bonvote3"), "route": ("plus", "route")}
    cq = {}
    for x, k in runs_route.items():
        t = tot[k]
        nn = len(F.runs[k].rows)
        cq[x] = {"disc": t["cny_disc"] / nn, "nodisc": t["cny_nodisc"] / nn}
        for mode in KS.MODES:
            put("cost.nodisc.route.cny_q.%s.%s" % (x, mode), cq[x][mode])
    pc = {}
    for mode in KS.MODES:
        ex = {x: pts[mode][runs_route[x]][1] for x in ("gated", "bonvote2", "bonvote3")}
        pc["route_ca60", mode] = cq["route"][mode] / per_asked
        for x in ("gated", "bonvote2", "bonvote3"):
            pc[x, mode] = cq[x][mode] / ex[x]
        for x in ("route_ca60", "gated", "bonvote2", "bonvote3"):
            put("cost.nodisc.route.per_correct.%s.%s" % (x, mode), pc[x, mode])
        put("cost.nodisc.route.per_correct.route_vs_gated.%s" % mode, pc["route_ca60", mode] / pc["gated", mode])
    put("cost.nodisc.def", "disc = CNY per question with the cache discount as in RV2 (tokens that the API reported as cached are priced at 0.2 of the input price, the 0.2 is an assumption of exp/prices.yaml); "
        "nodisc = every prompt token at the full input price. ratio.<b>.<cfg>.vs_<ref>.<mode> = cost of (backbone b, configuration cfg) divided by the cost of the reference, same questions, "
        "ratio of sums with the database-clustered interval (.ci) and the item interval (.item_ci); refs: plus_zs = qwen3.7-plus single call, plus_gated = qwen3.7-plus gated loop, self_zs = the same "
        "backbone's single call. Per-item costs are rebuilt from the token fields of preds.jsonl (as the RV2 Pareto claims); route.* use the totals of the usage logs (as RV2.cost.*). "
        "front.* = Pareto front on point estimates (cost per question against EX). route.per_correct = CNY per correct answer: for route_ca60 the run's cost per question divided by the correct answers "
        "per question asked at 60 % coverage; for the other arms cost divided by EX.")
    put("cost.nodisc.cached.note", "Cached prompt tokens are recorded by the API for qwen3.7-plus in every arm (share of prompt tokens per arm in cost.nodisc.cached.plus.<cfg>.share), for qwen3.7-flash in "
        "the single-call arm and weakly in the fixed loop and the other pipeline arms, and for qwen3.5-27b in no arm. The discount therefore applies unequally across arms, and depends on run order and prefix reuse "
        "rather than on the method; the no-discount figures price every arm alike.")


# ------------------------------------------------------------------------------------------------ S5
def gate_blocks(F, args, B):
    put("gate.def", "A question is counted as fired when the loop produced at least one revision (a second candidate in cands.jsonl). Gated loop: the root fails to execute (error or timeout); "
        "fixed loop: the Critic score of the current candidate is below 8.5 or the root fails. share = fired / n, with the database-clustered interval (.ci) and the exact Clopper-Pearson interval (.item_ci). "
        "n = BIRD dev 1 534 questions (11 databases) and Spider-Realistic 508 questions (19 databases).")
    for b in BK:
        for arm in KS.GATE_ARMS:
            r = F.runs[(b, arm)]
            qs = [q for q in F.qids if r.root(q) is not None]
            fire = np.array([1.0 if len(r.cands[q]) > 1 else 0.0 for q in qs])
            dbs = np.array([r.rows[q].get("db_id") or q.split(":")[1] for q in qs])
            if arm == "gated":
                alt = np.array([1.0 if r.root(q).get("status") in A.EXEC_FAIL else 0.0 for q in qs])
                if not np.array_equal(alt, fire):
                    log("WARNING gated fire != root fails for", b, int(alt.sum()), int(fire.sum()))
            _gate_put("bird", b, arm, fire, dbs, B)
    import ext1_analyze as X
    sp = {}
    for b in X.BACKBONES:
        for arm in ("zs", "rootonly", "gated", "fixed"):
            d = os.path.join("EXT1", "%s-%s-s0" % (arm, b))
            sp[(b, arm)] = A.Run(args.runs, d)
    for b in X.BACKBONES:
        for arm in KS.GATE_ARMS:
            r = sp[(b, arm)]
            qs = sorted(q for q in r.rows if q in r.cands)
            fire = np.array([1.0 if len(r.cands[q]) > 1 else 0.0 for q in qs])
            dbs = np.array([r.rows[q]["db_id"] for q in qs])
            _gate_put("spider", b, arm, fire, dbs, B)
    # helped / hurt: BIRD from the RV2 loop keys (same definitions), Spider-Realistic final vs root-only run (test-suite EX)
    for b in BK:
        for arm in KS.GATE_ARMS:
            for d in ("in", "btw"):
                k = "loop.%s.eff_%s.%s" % (b, d, arm)
                bb, cc = A.OUT[k + ".b"]["v"], A.OUT[k + ".c"]["v"]
                _hh_put("bird", b, arm, d, bb, cc, A.OUT[k]["v"] / 100.0 * 1.0, None)
    for b in X.BACKBONES:
        ro = sp[(b, "rootonly")]
        for arm in KS.GATE_ARMS:
            r = sp[(b, arm)]
            qs = sorted(set(r.rows) & set(ro.rows))
            ea = np.array([X.ex_ts_value(r.rows[q]) for q in qs]); eb = np.array([X.ex_ts_value(ro.rows[q]) for q in qs])
            bb = int(((ea == 1) & (eb == 0)).sum()); cc = int(((ea == 0) & (eb == 1)).sum())
            _hh_put("spider", b, arm, "btw", bb, cc, (bb - cc) / float(len(qs)), len(qs))
    put("gate.hh.def", "helped (b) = the loop turned a wrong answer right, hurt (c) = right to wrong, paired on the same questions; net = (b - c) / n in percentage points. BIRD dev: 'in' = final minus the root candidate "
        "of the same run, 'btw' = final minus the separate root-only run (the keys of RV2.loop.*, repeated here). Spider-Realistic: final minus the root-only run, test-suite EX (the root candidate of a run "
        "has no test-suite score). p_two = exact two-sided sign test (binomial, p = 1/2, on the b + c discordant questions; identical to the exact McNemar test), p_one = exact one-sided p that helped exceeds hurt. "
        "A sign test treats the discordant questions as independent; it ignores the database clusters.")


def _gate_put(ds, b, arm, fire, dbs, B):
    p = "gate.%s.%s.%s." % (ds, b, arm)
    n, k = len(fire), int(fire.sum())
    boot = A.CB(dbs, B)
    ci = list(S.pctl_ci(100.0 * boot.means(fire)))
    ici = list(S.clopper_pearson(k, n))
    put(p + "n", n); put(p + "fired", k)
    put(p + "share", k / float(n), ci=[ci[0] / 100.0, ci[1] / 100.0], item_ci=ici, n=n)


def _hh_put(ds, b, arm, d, bb, cc, net, n):
    p = "gate.hh.%s.%s.%s.%s." % (ds, b, arm, d)
    mc = S.mcnemar_exact(int(bb), int(cc))
    put(p + "b", int(bb)); put(p + "c", int(cc)); put(p + "p_two", mc["p_two"]); put(p + "p_one", mc["p_one_greater"])
    put(p + "net", 100.0 * net)


# ------------------------------------------------------------------------------------------------ driver
def check_against_published(res_dir):
    rows = {r["id"]: r for r in csv.DictReader(open(os.path.join(res_dir, "posthoc2", "dual.csv")))}
    bad = []
    for i, e in CAP.items():
        r = rows[i]
        for a_, b_ in ((e["est"], float(r["est"])), (e["ci"][0], float(r["ci_lo"])), (e["ci"][1], float(r["ci_hi"])), (e["item_ci"][0], float(r["item_lo"])), (e["item_ci"][1], float(r["item_hi"]))):
            if abs(a_ - b_) > 1e-9 * (1 + abs(b_)):
                bad.append((i, a_, b_))
    return bad


class Args(object):
    pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=10000)
    a = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    B = a.B
    install_wrappers()
    prices = A.load_prices(os.path.join(a.exp, "prices.yaml"))
    coef = json.load(open(os.path.join(a.exp, "frozen", "route_coef_hr.json")))
    F = A.Frame(a.runs, B)
    F.args = a
    res, per, m_all = A.analysis_loop(F, B)
    A.analysis_dual_dev(F, B)
    A.analysis_p5(F, B)
    rvec = A.analysis_p3(F, B, coef)
    A.analysis_rescore(F, B, a, coef)
    A.analysis_ext1(a, B)
    A.analysis_pf(a, B)
    bad = check_against_published(a.results)
    log("check against results/posthoc2/dual.csv: %d dual rows, %d mismatches" % (len(CAP), len(bad)))
    if bad:
        raise SystemExit("recomputation differs from the published dual.csv: %r" % (bad[:3],))
    # S1
    SMALL = collections.OrderedDict()
    for i in KS.DUAL_IDS:
        SMALL[i] = small_block(i, CAP[i])
    for i in KS.LOOP_IDS:
        parts = i.split(".")
        key = (parts[1], parts[2][4:], parts[3]) if len(parts) == 4 else (parts[1], "stack")
        SMALL[i] = small_block(i, res[key])
    put("small.def", "Small-cluster readings (post hoc, robustness). est = headline estimate (item-pooled mean; for ratios, AUROC differences and P5' the statistic itself). "
        "t_ci = estimate +- t(G-1, 0.975) x SE with the cluster-robust SE  sqrt(G/(G-1) sum_g (sum_{i in g}(d_i - mean))^2)/n  (G = 11 or 19 databases); tdb_ci = t(G-1) interval of the unweighted mean of the "
        "database-level mean differences; sf_ci / sfu_ci = {theta0 : exact two-sided sign-flip p > 0.05} over the G database-level mean differences with weights n_g/n (sf) or 1/G (sfu), all 2^G sign patterns "
        "enumerated; for statistics that are not item means (ratios of sums, AUROC and AURC differences, P5') t and sign-flip use delete-one-database jackknife pseudo-values (tdb = t, sfu = sf). "
        "cl_ci and item_ci are the intervals of RV2.dual (database-clustered percentile and item bootstrap, B = 10 000); wide_ci = hull(cl, item, t, sf), wideu_ci = hull of all six, old_ci = hull(cl, item). class_* = reading word under the "
        "rule of the contrast (RV2.dual.<id>.rule) applied to that interval; reading_changes = yes when the word of wide_ci differs from that of old_ci (reading_changes_u: wideu_ci). p_* = two-sided p for H0: theta = null "
        "(0, the margin -1.5 for NI(1.5), 0.80 or 1 for the ratio rules). Smallest attainable two-sided sign-flip p: 2^(1-G) = 0.00098 (G = 11), 0.0000038 (G = 19).")
    pstar_blocks(None, json.load(open(os.path.join(a.results, "W1.json"))), json.load(open(os.path.join(a.results, "primary.json"))), SMALL)
    # S2, S3, S4, S5
    deff_blocks(F, SMALL, B)
    margin_blocks(res, SMALL)
    cost_blocks(F, B, prices, rvec)
    gate_blocks(F, a, B)
    missing = [k for k in KINDS if k not in OUT]
    extra = [k for k in OUT if k not in KINDS]
    log("keys: %d written, %d expected, %d missing, %d extra" % (len(OUT), len(KINDS), len(missing), len(extra)))
    meta = {"scope": "posthoc-secondary", "sap": "docs/SAP.md section 25 (post hoc note S2)", "B": B, "seed": A.SEED, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "api_calls": 0, "missing_keys": missing, "log": LOG, "n_dev": F.n, "recomputation_check": {"rows": len(CAP), "mismatches": len(bad)},
            "sign_flip": "exact enumeration of all 2^G sign patterns (G = 11: 2048; G = 19: 524288)", "price_date": "2026-10-26"}
    json.dump({"_meta": meta, "keys": OUT}, io.open(os.path.join(a.out, "rv2s_summary.json"), "w", encoding="utf-8"), indent=0, ensure_ascii=False)
    rows = []
    for i, s in SMALL.items():
        r = {"id": i, "rule": s["rule"], "n": s["n"], "g": s["g"], "est": s["est"], "deff": s["deff"], "deff_cr": s["deff_cr"], "p_t": s["p_t"], "p_sf": s["p_sf"], "p_sfu": s["p_sfu"],
             "additive": int(s["additive"])}
        for k, iv in s["ivs"].items():
            r[k + "_lo"], r[k + "_hi"] = iv
        for k, c in s["cls"].items():
            r["class_" + k] = c
        r["reading_changes"] = "yes" if s["cls"]["wide"] != s["cls"]["old"] else "no"
        r["reading_changes_u"] = "yes" if s["cls"]["wideu"] != s["cls"]["old"] else "no"
        rows.append(r)
    A.write_csv(os.path.join(a.out, "small.csv"), rows)
    log("done in %.0f s" % (time.time() - t0))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
