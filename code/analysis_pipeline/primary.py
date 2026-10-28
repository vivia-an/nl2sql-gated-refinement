"""Primary confirmatory family P (SAP section 13.3): P1-P5, Holm (13.5), verdicts (13.3 decision rule).

    python -m analysis_pipeline.primary --runs $R/runs --results results [--e4-roots DIR ...] [--B 10000]

Reads only runs/ (and exp/selection_decisions.json, exp/analysis_manifest.json if present); writes results/primary.json (flat keys).
A test whose runs are missing or incomplete is reported as `P<k>.status = "not run: ..."` with no numbers (never imputed); it stays in the
Holm family (p treated as 1) unless the SAP removes it (P3: route_ca not adopted; P5: n_R < 40).

Conventions (documented in docs/ANALYSIS_RUNBOOK.md): one-sided tests enter Holm as p_2s = min(1, 2 p_1s); p_1s = (#{boot stat on the wrong side of the
null} + 1) / (B + 1) from the db-clustered paired bootstrap; item-level checks: Tango score test (NI), DeLong (AUROC), exact McNemar. Holm is
computed twice, on the db-clustered p-values (`p_adj`) and on the item-level p-values (`p_adj_item`); `verdict` uses both as in SAP 13.3."""
import argparse
import os
import sys

import numpy as np

from . import e4 as E4, routing, runs as R, stats as S
from .util import Result, REPO, detect_scope, git_hash, set_default_scope, sha256_file

M1_MARGIN, RATIO_NULL, M2_MARGIN, N_R_MIN = 1.5, 0.80, 3.0, 40
THR = "thrombosis_prediction"
EXEC_FAIL = ("err_syntax", "err_identifier", "timeout")


# ------------------------------------------------------------------------------------------ building blocks
def _same_sign_share(dbs, d, delta):
    """share of databases whose own mean difference has the sign of the pooled one."""
    shares = []
    for db in sorted(set(dbs.tolist())):
        m = d[dbs == db].mean()
        shares.append(1.0 if (m * delta > 0) else 0.0)
    return float(np.mean(shares)) if shares else None


def ni_block(ex_new, ex_ref, dbs, B, margin_pp, boot=None):
    """Non-inferiority of `new` vs `ref` on EX (paired, db-clustered). Differences in percentage points."""
    ex_new, ex_ref = np.asarray(ex_new, int), np.asarray(ex_ref, int)
    boot = boot or S.ClusterBoot(dbs, B)
    d = ex_new - ex_ref
    delta = 100.0 * d.mean()
    bt = 100.0 * (boot.means(ex_new) - boot.means(ex_ref))
    lo, hi = S.pctl_ci(bt)
    ib = 100.0 * S.item_boot_means(d, B)
    ilo, ihi = S.pctl_ci(ib)
    n10 = int(((ex_new == 1) & (ex_ref == 0)).sum())
    n01 = int(((ex_new == 0) & (ex_ref == 1)).sum())
    tg = S.tango_ni(n10, n01, len(d), margin_pp / 100.0)
    mc = S.mcnemar_exact(n10, n01)
    loo = boot.loo(lambda idx: 100.0 * d[idx].mean())
    out = {"n": int(len(d)), "ex_new": float(ex_new.mean()), "ex_ref": float(ex_ref.mean()), "delta_pp": float(delta), "ci_lo": lo, "ci_hi": hi, "item_ci_lo": ilo, "item_ci_hi": ihi,
           "p_1s": S.p_le(bt, -margin_pp), "p_item_1s": tg["p_one"], "n10": n10, "n01": n01, "mcnemar_p": mc["p_two"],
           "loo_min": min(v for _, v in loo) if loo else None, "loo_max": max(v for _, v in loo) if loo else None,
           "same_sign_share": _same_sign_share(dbs, d, delta), "n_dbs": boot.k, "margin_pp": margin_pp}
    out["margin_p"] = {str(m): S.p_le(bt, -m) for m in (1.0, 1.5, 2.0, 3.0)}
    out["_boot"] = bt
    return out


def _put_ni(res, pre, o):
    res.ci(pre + ".delta_ex", o["delta_pp"], o["ci_lo"], o["ci_hi"])
    res.set(pre + ".delta_ex.item_ci_lo", o["item_ci_lo"]); res.set(pre + ".delta_ex.item_ci_hi", o["item_ci_hi"])
    for k in ("n", "n10", "n01", "n_dbs", "loo_min", "loo_max", "same_sign_share", "mcnemar_p", "ex_new", "ex_ref"):
        res.set("%s.%s" % (pre, k), o[k])
    res.set(pre + ".p_1s", o["p_1s"]); res.set(pre + ".p_item_1s", o["p_item_1s"])
    for m, p in o["margin_p"].items():
        res.set("%s.p_1s_margin%s" % (pre, m.replace(".", "_")), p)


# ------------------------------------------------------------------------------------------ P1
def p1_core(ex_g, ex_f, calls_g, calls_f, dbs, B):
    boot = S.ClusterBoot(dbs, B)
    o = ni_block(ex_g, ex_f, dbs, B, M1_MARGIN, boot)
    rb = boot.ratio(calls_g, calls_f)
    r = float(np.sum(calls_g) / max(np.sum(calls_f), 1e-12))
    rlo, rhi = S.pctl_ci(rb)
    rib = S.item_boot_ratio(calls_g, calls_f, B)
    ilo, ihi = S.pctl_ci(rib)
    o["ratio"] = {"mean": r, "ci_lo": rlo, "ci_hi": rhi, "ci_hi_1s95": float(np.percentile(rb, 95)), "item_ci_lo": ilo, "item_ci_hi": ihi,
                  "p_1s": S.p_ge(rb, RATIO_NULL), "p_item_1s": S.p_ge(rib, RATIO_NULL), "calls_gated": float(np.mean(calls_g)), "calls_fixed": float(np.mean(calls_f))}
    o["p_1s_iut"] = max(o["p_1s"], o["ratio"]["p_1s"])
    o["p_item_1s_iut"] = max(o["p_item_1s"], o["ratio"]["p_item_1s"])
    o["cl_ok"] = bool(o["ci_lo"] is not None and o["ci_lo"] > -M1_MARGIN and rlo is not None and rhi < RATIO_NULL)
    o["item_ok"] = bool(o["item_ci_lo"] > -M1_MARGIN and ihi < RATIO_NULL)
    o["favourable"] = bool(o["delta_pp"] > -M1_MARGIN and r < RATIO_NULL)
    return o


def test_p1(reg, res, B):
    g, f = reg.get("full_gated"), reg.get("full_fixed")
    miss = [n for n, r in (("full_gated", g), ("full_fixed", f)) if r is None]
    if miss:
        res.set("P1.status", "not run: missing or incomplete r0 run(s): " + ", ".join(miss))
        return None
    A = R.align({"g": g, "f": f})
    ex_g, ex_f = A["ex"]["g"], A["ex"]["f"]
    cg, cf = R.col(A["rows"]["g"], "calls"), R.col(A["rows"]["f"], "calls")
    o = p1_core(ex_g, ex_f, cg, cf, A["dbs"], B)
    res.set("P1.status", "ok")
    _put_ni(res, "P1.a", o)
    res.ci("P1.delta_ex", o["delta_pp"], o["ci_lo"], o["ci_hi"])
    res.set("P1.delta_ex.item_ci_lo", o["item_ci_lo"]); res.set("P1.delta_ex.item_ci_hi", o["item_ci_hi"])
    rr = o["ratio"]
    res.ci("P1.calls_ratio", rr["mean"], rr["ci_lo"], rr["ci_hi"])
    for k in ("ci_hi_1s95", "item_ci_lo", "item_ci_hi", "p_1s", "p_item_1s", "calls_gated", "calls_fixed"):
        res.set("P1.calls_ratio." + k, rr[k])
    res.ci("P1.calls_saved", 1.0 - rr["mean"], 1.0 - rr["ci_hi"], 1.0 - rr["ci_lo"])        # fraction of answer-path calls saved (STORYLINE R3 template)
    res.set("P1.p_1s", o["p_1s_iut"]); res.set("P1.p_item_1s", o["p_item_1s_iut"])
    res.set("P1.margin_pp", M1_MARGIN); res.set("P1.ratio_null", RATIO_NULL)
    res.set("P1.n", o["n"]); res.set("P1.dropped_ex_null", A["dropped"]["ex_null"])
    # descriptives reported with P1 (not tested)
    tok_g = R.col(A["rows"]["g"], "prompt_tokens") + R.col(A["rows"]["g"], "completion_tokens")
    tok_f = R.col(A["rows"]["f"], "prompt_tokens") + R.col(A["rows"]["f"], "completion_tokens")
    res.set("P1.tokens_per_q.gated", float(tok_g.mean())); res.set("P1.tokens_per_q.fixed", float(tok_f.mean()))
    res.set("P1.tokens_ratio", float(tok_g.sum() / max(tok_f.sum(), 1)))
    res.set("P1.tokens_per_correct.gated", float(tok_g.sum() / max(ex_g.sum(), 1))); res.set("P1.tokens_per_correct.fixed", float(tok_f.sum() / max(ex_f.sum(), 1)))
    res.set("P1.latency_mean_s.gated", float(R.col(A["rows"]["g"], "latency_s").mean())); res.set("P1.latency_mean_s.fixed", float(R.col(A["rows"]["f"], "latency_s").mean()))
    # sensitivity without thrombosis_prediction (SAP 8)
    keep = A["dbs"] != THR
    if keep.any() and (~keep).any():
        s = p1_core(ex_g[keep], ex_f[keep], cg[keep], cf[keep], A["dbs"][keep], B)
        res.ci("P1.sens_no_thr.delta_ex", s["delta_pp"], s["ci_lo"], s["ci_hi"])
        res.ci("P1.sens_no_thr.calls_ratio", s["ratio"]["mean"], s["ratio"]["ci_lo"], s["ratio"]["ci_hi"])
        res.set("P1.sens_no_thr.p_1s", s["p_1s_iut"]); res.set("P1.sens_no_thr.n", s["n"])
        res.set("P1.sens_no_thr.same_decision", bool(s["cl_ok"] == o["cl_ok"] and (s["delta_pp"] > 0) == (o["delta_pp"] > 0)))
    return o


# ------------------------------------------------------------------------------------------ P2
def pick_bstar(reg, full_run, decisions, res):
    """B*: among the baselines actually run, EX-maximal with calls <= 1.15 * calls(full) (ties: fewer calls). Pre-registered HR choice wins."""
    base = reg.baselines()
    subset = [l for l in base if base[l].subset]
    if subset:
        res.set("P2.baselines_subset_excluded", ", ".join(sorted(subset)))     # subset baselines are compared on their subset only (E1), never B* candidates
    base = {l: r for l, r in base.items() if not r.subset}
    if not base:
        return None, {}, "no baseline run is complete"
    A = R.align(dict({"full": full_run}, **base))
    calls_full = float(R.col(A["rows"]["full"], "calls").mean())
    cand = {}
    for lab in base:
        ex = A["ex"][lab]
        calls = float(R.col(A["rows"][lab], "calls").mean())
        cand[lab] = {"ex": float(ex.mean()), "calls": calls, "eligible": calls <= 1.15 * calls_full + 1e-9}
    res.set("P2.b_ref_calls", calls_full)
    res.set("P2.baselines_run", ", ".join(sorted(base)))
    res.set("P2.n_baselines_run", len(base))
    for lab, c in cand.items():
        res.set("P2.cand.%s.ex" % lab, c["ex"]); res.set("P2.cand.%s.calls" % lab, c["calls"]); res.set("P2.cand.%s.eligible" % lab, c["eligible"])
    pre = decisions.get("bstar")
    if pre and pre in base:
        res.set("P2.bstar_selected_on", "pre-registered choice (exp/selection_decisions.json)")
        return pre, A, None
    elig = [l for l in cand if cand[l]["eligible"]]
    if not elig:
        return None, A, "no baseline has calls <= 1.15 x calls(full) = %.2f" % (1.15 * calls_full)
    best = sorted(elig, key=lambda l: (-cand[l]["ex"], cand[l]["calls"]))[0]
    res.set("P2.bstar_selected_on", "%s (MINIMAL_PLAN: no HR baseline runs; selection on the test data makes the comparison conservative for `full`)" % reg.split)
    return best, A, None


def test_p2(reg, res, B, decisions, full_label):
    full = reg.get(full_label)
    if full is None:
        res.set("P2.status", "not run: %s r0 missing or incomplete" % full_label)
        return None
    bstar, A, why = pick_bstar(reg, full, decisions, res)
    if bstar is None:
        res.set("P2.status", "not run: " + (why or "no baseline"))
        return None
    ex_f, ex_b = A["ex"]["full"], A["ex"][bstar]
    o = ni_block(ex_f, ex_b, A["dbs"], B, M2_MARGIN)
    res.set("P2.status", "ok")
    res.set("P2.bstar", bstar)
    res.set("P2.full", full_label)
    res.set("P2.bstar_is_cheap", bool(R.SYSTEM_TO_BASELINE.get(bstar.split("@")[0], bstar.split("@")[0]) in ("zs", "dail")))
    res.set("P2.reading_rule", "B* is zero-shot or DAIL-style: P2 only says the loop does not cost accuracy relative to a cheaper alternative" if res.d["P2.bstar_is_cheap"]
            else "B* is a loop or voting baseline at a comparable budget")
    _put_ni(res, "P2", o)
    res.ci("P2.delta_ex", o["delta_pp"], o["ci_lo"], o["ci_hi"])
    res.set("P2.margin_pp", M2_MARGIN)
    res.set("P2.calls_full", float(R.col(A["rows"]["full"], "calls").mean())); res.set("P2.calls_bstar", float(R.col(A["rows"][bstar], "calls").mean()))
    o["cl_ok"] = bool(o["ci_lo"] > -M2_MARGIN)
    o["item_ok"] = bool(o["item_ci_lo"] > -M2_MARGIN)
    o["favourable"] = bool(o["delta_pp"] > -M2_MARGIN)
    keep = A["dbs"] != THR
    if keep.any() and (~keep).any():
        s = ni_block(ex_f[keep], ex_b[keep], A["dbs"][keep], B, M2_MARGIN)
        res.ci("P2.sens_no_thr.delta_ex", s["delta_pp"], s["ci_lo"], s["ci_hi"]); res.set("P2.sens_no_thr.p_1s", s["p_1s"])
        res.set("P2.sens_no_thr.same_decision", bool((s["ci_lo"] > -M2_MARGIN) == o["cl_ok"]))
    return o


# ------------------------------------------------------------------------------------------ P3
def route_scores(rows, coef=None, k=None):
    """-> (v3, route_ca, n_missing) arrays from preds rows (NaN where unavailable). route_ca = in-run `route_score` unless `coef` (fit_route_score json) is given."""
    v3, ca = [], []
    for r in rows:
        c = r.get("critic") or {}
        v3.append(float(c["score"]) if (c.get("semantic") is not None and c.get("score") is not None) else np.nan)
        if coef is not None:
            bits = r.get("cons_bits")
            cons = (float(np.mean(bits[:k])) if (k and bits) else r.get("consistency"))
            st = r.get("final_status")
            nonempty = (1.0 if st == "ok_nonempty" else 0.0) if st is not None else (1.0 if r.get("exec_ok") else None)
            ex_sc = c.get("syntax")
            feats = {"v3": v3[-1], "exec": ex_sc, "consistency": cons, "nonempty": nonempty}
            vals = [feats.get(n) for n in coef["features"]]
            if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
                ca.append(np.nan)
            else:
                z = coef["intercept"] + float(np.dot(coef["coef"], vals))
                ca.append(10.0 / (1.0 + np.exp(-z)))
        else:
            ca.append(float(r["route_score"]) if r.get("route_score") is not None else np.nan)
    return np.array(v3), np.array(ca)


def test_p3(reg, res, B, decisions, full_label, coef=None, k=None):
    if decisions.get("route_ca_adopted") is False:
        res.set("P3.status", "dropped: route_ca failed the pre-lock adoption gate (SAP 13.6 rule 4); family m reduced")
        return None
    run = reg.get("full_gated_route") or reg.get(full_label)
    if run is None:
        res.set("P3.status", "not run: no complete run with routing fields (full_gated_route / %s)" % full_label)
        return None
    rows = [run.rows[q] for q in sorted(run.rows)]
    v3, ca = route_scores(rows, coef, k)
    y = np.array([R.ex_value(r) if R.ex_value(r) is not None else -1 for r in rows])
    dbs = np.array([r.get("db_id") for r in rows])
    ok = np.isfinite(v3) & np.isfinite(ca) & (y >= 0)
    if ok.sum() < 50:
        res.set("P3.status", "not run: run %s has route_score/V3 for %d items only (needs route.score v3+consistency and a V3 score of the final answer)" % (run.run_id, int(ok.sum())))
        return None
    v3, ca, y, dbs = v3[ok], ca[ok], y[ok], dbs[ok]
    boot = S.ClusterBoot(dbs, B)
    pan, bs, pt = routing.panel({"route_ca": ca, "v3": v3}, y, dbs, boot, ref="v3", prefix="P3.")
    d_auc = bs["route_ca"][:, 0] - bs["v3"][:, 0]
    delta = float(pt["route_ca"][0] - pt["v3"][0])
    lo, hi = S.pctl_ci(d_auc)
    dl = S.delong_paired(ca, v3, y)
    ilo = (dl["diff"] - S.norm_ppf(0.975) * dl["se"]) if dl["se"] else None
    ihi = (dl["diff"] + S.norm_ppf(0.975) * dl["se"]) if dl["se"] else None
    loo = boot.loo(lambda idx: S.auroc(ca[idx], y[idx]) - S.auroc(v3[idx], y[idx]))
    res.set("P3.status", "ok")
    res.update(pan)
    res.ci("P3.dauroc", delta, lo, hi)
    res.set("P3.dauroc.item_ci_lo", ilo); res.set("P3.dauroc.item_ci_hi", ihi)
    res.set("P3.dauroc.delong_se", dl["se"]); res.set("P3.p_1s", S.p_le(d_auc, 0.0)); res.set("P3.p_item_1s", dl["p_one_greater"])
    res.set("P3.dauroc.loo_min", min(v for _, v in loo)); res.set("P3.dauroc.loo_max", max(v for _, v in loo))
    res.set("P3.dauroc.same_sign_share", float(np.mean([1.0 if v * delta > 0 else 0.0 for _, v in loo])))
    res.set("P3.n", int(ok.sum())); res.set("P3.n_dropped_no_score", int((~ok).sum()))
    res.set("P3.ex_rate", float(y.mean()))
    # co-reported readouts (S-F); wording gate of SAP 13.3: 'improves the risk-coverage trade-off' only if AURC moved favourably with CI excluding 0
    ak = "P3.route_ca_vs_v3.daurc"
    res.set("P3.aurc_favourable", bool(res.d.get(ak + ".ci_hi", 1) < 0))
    res.set("P3.acc60_favourable", bool(res.d.get("P3.route_ca_vs_v3.dacc_ex_cov60.ci_lo", -1) > 0))
    ns = [r.get("n_samples") for r in rows if r.get("n_samples") is not None]
    kk = float(np.mean(ns)) if ns else None
    if kk is not None:
        res.set("P3.extra_calls.route_ca", 1.0 + kk); res.set("P3.extra_calls.v3", 1.0); res.set("P3.k_samples", kk)
    res.set("P3.run", run.run_id)
    return {"delta": delta, "cl_ok": bool(lo is not None and lo > 0), "item_ok": bool(ilo is not None and ilo > 0), "favourable": delta > 0}


# ------------------------------------------------------------------------------------------ P4
def test_p4(res, e4_roots, B, pollution=40, noharm=2.0):
    if not e4_roots:
        res.set("P4.status", "not run: no E4 stream directory given (--e4-roots)")
        return None
    units, notes = E4.load_all(e4_roots)
    for n in notes:
        res.note("E4: " + n)
    pols = sorted(set(u["policy"] for u in units))
    comp = "append" if "append" in pols else ("no_evict" if "no_evict" in pols else None)
    if "ws_full" not in pols or comp is None:
        res.set("P4.status", "not run: need complete units for ws_full and append at p=%d (found policies: %s)" % (pollution, ", ".join(pols) or "none"))
        return None
    c = E4.policy_contrast(units, "ws_full", comp, pollution, B)
    if c is None or "logodds" not in c:
        res.set("P4.status", "not run: ws_full vs %s at p=%d has no units / GLMM failed (%s)" % (comp, pollution, (c or {}).get("glmm_error", "no data")))
        return None
    res.set("P4.status", "ok")
    res.set("P4.comparator", comp); res.set("P4.pollution", pollution)
    res.ci("P4.logodds", c["logodds"], c["logodds_ci_lo"], c["logodds_ci_hi"])
    res.ci("P4.or", c["odds_ratio"], c["or_ci_lo"], c["or_ci_hi"])
    res.ci("P4.diff_pp", c["diff_pp"], c["diff_pp_ci_lo"], c["diff_pp_ci_hi"])
    for k in ("p_lrt", "p_wald", "ex_a", "ex_b", "n_a", "n_b", "n_dbs", "b", "c", "p_mcnemar", "sigma_db", "sigma_qid", "sigma_perm"):
        res.set("P4." + k, c.get(k))
    res.set("P4.glmm_converged", c.get("glmm_converged"))
    res.set("P4.p_2s", c["p_lrt"]); res.set("P4.p_item_2s", c["p_mcnemar"])
    res.set("P4.mde_note", "planning MDE 6.9 pp at alpha 0.05 and 8.6 pp at 0.0083 (SAP 9); an effect below about 7 pp is reported as direction consistent, not confirmed")
    # p = 0 no-harm contrast, non-inferiority margin 2 pp (secondary)
    nh = E4.policy_contrast(units, "ws_full", comp, 0, B, margin_pp=noharm, fit=False)
    if nh:
        res.ci("P4.noharm_p0.diff_pp", nh["diff_pp"], nh["diff_pp_ci_lo"], nh["diff_pp_ci_hi"]); res.set("P4.noharm_p0.p_1s", nh.get("p_ni"))
    z_pos = c["logodds"] > 0
    return {"cl_ok": bool(c["logodds_ci_lo"] > 0), "item_ok": bool(c["p_mcnemar"] < 0.05 and c["diff_pp"] > 0), "favourable": bool(z_pos)}


# ------------------------------------------------------------------------------------------ P5
def _class_of(c):
    st = c.get("status") or ""
    err = str(c.get("error") or c.get("pred_error") or "").lower()
    if st == "timeout":
        return "timeout"
    if "ambiguous column" in err:
        return "ambiguous column"
    if st == "err_identifier" or "no such column" in err or "no such table" in err:
        return "unknown identifier"
    if st == "err_syntax" or "syntax error" in err:
        return "syntax"
    return "other"


def repair_population(run):
    """-> list of rows (preds) of items whose root candidate fails to execute, with their root class. Needs cands.jsonl."""
    roots = run.roots()
    if not roots:
        return None
    out = []
    for q, c in roots.items():
        if c.get("status") in EXEC_FAIL and q in run.rows:
            out.append((run.rows[q], _class_of(c)))
    return out


def test_p5(reg, res, B, decisions):
    gated = reg.get("full_gated")
    if gated is None:
        res.set("P5.status", "not run: full_gated r0 missing or incomplete")
        return None
    pop = repair_population(gated)
    if pop is None:
        res.set("P5.status", "not run: %s has no cands.jsonl (root candidates)" % gated.run_id)
        return None
    nR = len(pop)
    res.set("P5.n_R", nR)
    rep = np.array([1 if R.ex_value(r) == 1 else 0 for r, _ in pop])
    exe = np.array([1 if r.get("exec_ok") else 0 for r, _ in pop])
    dbs = np.array([r.get("db_id") for r, _ in pop])
    if nR:
        lo, hi = S.clopper_pearson(int(rep.sum()), nR)
        res.ci("P5.repair_rate", float(rep.mean()), lo, hi)
        elo, ehi = S.clopper_pearson(int(exe.sum()), nR)
        res.ci("P5.executable_rate", float(exe.mean()), elo, ehi)
        res.set("P5.n_dbs", int(len(set(dbs.tolist()))))
        for cl in ("syntax", "unknown identifier", "ambiguous column", "timeout", "other"):
            sel = np.array([c == cl for _, c in pop])
            key = "P5.by_class.%s" % cl.replace(" ", "_")
            res.set(key + ".n", int(sel.sum()))
            if sel.sum():
                w = S.wilson(int(rep[sel].sum()), int(sel.sum()))
                res.ci(key + ".repair_rate", float(rep[sel].mean()), w[0], w[1])
        top = max(set(dbs.tolist()), key=lambda d: (dbs == d).sum())
        res.set("P5.top_db", top); res.set("P5.top_db_share", float((dbs == top).mean()))
    legacy = reg.get("abl_q")
    if legacy is not None:
        lp = repair_population(legacy)
        if lp:
            lr = np.array([1 if R.ex_value(r) == 1 else 0 for r, _ in lp])
            llo, lhi = S.clopper_pearson(int(lr.sum()), len(lp))
            res.set("P5.legacy.n_R", len(lp)); res.ci("P5.legacy.repair_rate", float(lr.mean()), llo, lhi)
    blind = reg.get("full_gated_blind")
    in_family = nR >= N_R_MIN and blind is not None
    res.set("P5.in_family", bool(in_family))
    if blind is None:
        res.set("P5.status", ("not run: full_gated_blind r0 missing or incomplete (n_R = %d)" if nR >= N_R_MIN else "descriptive: n_R = %d < 40 (full_gated_blind r0 missing)") % nR)
        return None

    def _bx(q):
        r = blind.rows.get(q)
        v = R.ex_value(r) if r is not None else None
        return -1 if v is None else int(v)

    brep = np.array([_bx(r["qid"]) for r, _ in pop])
    okb = brep >= 0
    if okb.any():
        blo, bhi = S.clopper_pearson(int(brep[okb].sum()), int(okb.sum()))
        res.ci("P5.blind_rate", float(brep[okb].mean()), blo, bhi)
    b = int(((rep == 1) & (brep == 0) & okb).sum())
    c = int(((rep == 0) & (brep == 1) & okb).sum())
    mc = S.mcnemar_exact(b, c)
    res.set("P5.b", b); res.set("P5.c", c)
    res.set("P5.mcnemar.p_1s", mc["p_one_greater"]); res.set("P5.mcnemar.p_2s", mc["p_two"])
    cl_lo = None
    if okb.sum() >= 2 and len(set(dbs.tolist())) >= 2:
        boot = S.ClusterBoot(dbs[okb], B)
        dd = rep[okb] - brep[okb]
        bt = boot.means(dd)
        lo, hi = S.pctl_ci(bt)
        cl_lo = lo
        res.ci("P5.diff_rate_pp", 100 * float(dd.mean()), 100 * lo, 100 * hi)
        res.set("P5.p_1s_cluster", S.p_le(bt, 0.0))
    res.set("P5.status", "ok" if in_family else "descriptive: n_R = %d < %d (family m reduced)" % (nR, N_R_MIN))
    if in_family:
        res.set("P5.p_1s", mc["p_one_greater"])           # the pre-registered test is the exact one-sided McNemar on R
        res.set("P5.p_item_1s", mc["p_one_greater"])
    return {"cl_ok": bool(cl_lo is not None and cl_lo > 0), "item_ok": bool(mc["p_one_greater"] < 0.05), "favourable": b > c, "in_family": in_family}


# ------------------------------------------------------------------------------------------ driver
def run_primary(reg, e4_roots, decisions, B=10000, coef=None, route_k=None, out_path=None):
    res = Result("primary")
    res.set("P.scope", res.scope)                 # confirmatory | tuning-set | synthetic: anything but `confirmatory` is a dry run and never reaches the paper
    full_label = decisions.get("full") or "full_gated"
    res.set("P.full_label", full_label)
    res.set("P.full_label_source", "exp/selection_decisions.json" if decisions.get("full") else "default (no selection_decisions.json: full := full_gated)")
    info = {}
    info["P1"] = test_p1(reg, res, B)
    info["P2"] = test_p2(reg, res, B, decisions, full_label)
    info["P3"] = test_p3(reg, res, B, decisions, full_label, coef, route_k)
    info["P4"] = test_p4(res, e4_roots, B)
    info["P5"] = test_p5(reg, res, B, decisions)
    # ---- family and Holm
    members = ["P1", "P2", "P4"]
    dropped = []
    if decisions.get("route_ca_adopted") is False:
        dropped.append("P3 (route_ca not adopted)")
    else:
        members.append("P3")
    p5_in = bool(res.d.get("P5.in_family", False))
    # P5 stays a family member when its data are missing (n_R unknown) - only a known n_R < 40 removes it
    if "P5.n_R" in res.d and res.d["P5.n_R"] < N_R_MIN:
        dropped.append("P5 (n_R = %d < %d)" % (res.d["P5.n_R"], N_R_MIN))
    else:
        members.append("P5")
    members = sorted(members)
    m = decisions.get("m_family") or len(members)
    two = {}
    item = {}
    for t in members:
        p1 = res.d.get(t + ".p_1s")
        pi = res.d.get(t + ".p_item_1s")
        if t == "P4":
            two[t] = res.d.get("P4.p_2s"); item[t] = res.d.get("P4.p_item_2s")
        else:
            two[t] = None if p1 is None else min(1.0, 2.0 * p1)
            item[t] = None if pi is None else min(1.0, 2.0 * pi)
    adj = S.holm(two, m)
    adj_i = S.holm(item, m)
    res.set("P.family.m", m)
    res.set("P.family.members", ", ".join(members))
    res.set("P.family.dropped", ", ".join(dropped) if dropped else "none")
    res.set("P.family.label", "SUPERSEDED: family P of SAP 13.3 (P1-P5, single backbone); replaced by family P* of Amendment 3 (SAP 15.3), see Pstar.* in results/W1.json")
    for t in members:
        if two[t] is not None:
            res.set(t + ".p_2s", two[t]); res.set(t + ".p_adj", adj[t])
        if item[t] is not None:
            res.set(t + ".p_item_2s", item[t]); res.set(t + ".p_adj_item", adj_i[t])
        o = info.get(t)
        if o is not None and two[t] is not None:
            res.set(t + ".verdict", S.verdict(adj[t], o["cl_ok"], o["item_ok"], o["favourable"], adj_i[t]))
        elif res.d.get(t + ".status", "").startswith("not run"):
            res.set(t + ".verdict", "not run")
    for t in ("P3", "P5"):
        if t not in members:
            res.set(t + ".verdict", "descriptive only" if t == "P5" else "dropped before the lock")
    # provenance
    res.meta.update({"B": B, "split": reg.split, "expected_n": reg.expected_n, "sap_sha256": sha256_file(os.path.join(REPO, "docs", "SAP.md")),
                     "decisions": decisions, "registry_notes": reg.notes, "runs_root": reg.root,
                     "n_runs_found": len(reg.all)})
    res.meta["roles"] = {l: (reg.get(l).run_id if _safe_get(reg, l) is not None else None) for l in ("full_gated", "full_fixed", "full_gated_blind", "full_gated_route", "abl_q")}
    if out_path:
        res.write(out_path)
    return res, info


def _safe_get(reg, lab):
    try:
        return reg.get(lab)
    except R.RunError:
        return None


def load_coef(path):
    import json
    if not path:
        return None
    d = json.load(open(path))
    return {"features": d["features"], "coef": d["coef"], "intercept": d["intercept"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--results", default=os.path.join(REPO, "results"))
    ap.add_argument("--split", default="bird_dev")
    ap.add_argument("--model", default=R.PRIMARY_MODEL)
    ap.add_argument("--e4-roots", nargs="*", default=[])
    ap.add_argument("--manifest", default=os.path.join(REPO, "exp", "analysis_manifest.json"))
    ap.add_argument("--decisions", default=os.path.join(REPO, "exp", "selection_decisions.json"))
    ap.add_argument("--replay-manifest", default=os.path.join(REPO, "exp", "replay_manifest.json"), help="run directories (relative to --runs) that take part; 'none' = every run found. Default: exp/replay_manifest.json (the run set of the released result files)")
    ap.add_argument("--models-yaml", default=os.path.join(REPO, "exp", "models.yaml"))
    ap.add_argument("--route-coef", default=None, help="exp/frozen/route_coef_hr.json: recompute route_ca from the frozen coefficients")
    ap.add_argument("--route-k", type=int, default=None)
    ap.add_argument("--expected-n", type=int, default=None)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--scope", default=None, choices=["confirmatory", "tuning-set", "synthetic"])
    a = ap.parse_args(argv)
    set_default_scope(a.scope or detect_scope(a.runs, a.split))
    reg = R.Registry(a.runs, R.load_manifest(a.manifest), a.models_yaml, a.split, a.model, a.expected_n, R.load_replay_dirs(a.replay_manifest))
    res, _ = run_primary(reg, a.e4_roots, R.load_decisions(a.decisions), a.B, load_coef(a.route_coef), a.route_k, os.path.join(a.results, "primary.json"))
    for k in sorted(res.d):
        if k.endswith(".status") or k.endswith(".verdict"):
            print("%-14s %s" % (k, res.d[k]))
    print("wrote", os.path.join(a.results, "primary.json"), "(%d keys)" % len(res.d))


if __name__ == "__main__":
    main()
