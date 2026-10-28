"""E7 (routing / selective prediction, S-F), E5 (score decomposition on stored candidates) and the data behind the risk-coverage and
calibration figures.  Unit = final answer of the routing run (`full_gated_route`, else `full`)."""
import numpy as np

from . import primary, routing, runs as R, stats as S

COVS = routing.COVERAGES


def _scores_from_rows(rows, coef=None, k=None):
    v3, ca = primary.route_scores(rows, coef, k)
    cons = np.array([float(r["consistency"]) * 10.0 if r.get("consistency") is not None else np.nan for r in rows])
    ex_sc = np.array([float((r.get("critic") or {}).get("syntax")) if (r.get("critic") or {}).get("syntax") is not None else np.nan for r in rows])
    verb = np.array([float(r["verbalized_confidence"]) / 10.0 if r.get("verbalized_confidence") is not None else np.nan for r in rows])
    return {"route_ca": ca, "v3": v3, "consistency": cons, "exec": ex_sc, "verbalised": verb}


def curve_band(scores, y, boot, points=20, B=1000):
    """pointwise 95 % db-clustered band of the risk-coverage curve for each score: dict label -> (cov, lo, hi)."""
    y = np.asarray(y, int)
    B = min(B, boot.B)
    cov = np.linspace(1.0 / points, 1.0, points)
    out = {k: np.empty((B, points)) for k in scores}
    for b in range(B):
        idx = np.concatenate([boot.members[j] for j in boot.draws[b]])
        yy = y[idx]
        for k, s in scores.items():
            cum = S._expected_cum_acc(np.asarray(s)[idx], yy)
            n = len(cum)
            out[k][b] = [1.0 - cum[max(1, int(round(c * n))) - 1] for c in cov]
    return {k: ([float(c) for c in cov], np.percentile(v, 2.5, axis=0).tolist(), np.percentile(v, 97.5, axis=0).tolist()) for k, v in out.items()}


def run_e7(reg, res, B, decisions=None, coef=None, k=None, calib_preds=None, full_label="full_gated"):
    decisions = decisions or {}
    run = reg.get("full_gated_route") or reg.get(full_label)
    if run is None:
        res.set("E7.status", "not run: no complete routing run (full_gated_route / %s)" % full_label)
        return
    rows = [run.rows[q] for q in sorted(run.rows)]
    y = np.array([R.ex_value(r) if R.ex_value(r) is not None else -1 for r in rows])
    dbs = np.array([r.get("db_id") for r in rows])
    sc = _scores_from_rows(rows, coef, k)
    base_ok = (y >= 0) & np.isfinite(sc["v3"])
    use = {n: v for n, v in sc.items() if np.isfinite(v[base_ok]).all() and base_ok.sum() > 0}
    if "v3" not in use:
        res.set("E7.status", "not run: run %s has no V3 score on its final answers (needs m5.score_for_routing)" % run.run_id)
        return
    use["v3_binary85"] = (use["v3"] >= 8.5).astype(float) * 10.0          # patent operating threshold as a 1-bit score (proxy for the binary critic V1)
    y, dbs = y[base_ok], dbs[base_ok]
    use = {n: v[base_ok] for n, v in use.items()}
    boot = S.ClusterBoot(dbs, B)
    pan, bs, pt = routing.panel(use, y, dbs, boot, ref="v3", prefix="E7.")
    # E7 keys carry the model segment of CONTRACT 4 only where the paper expects it; keep both short and model-specific forms
    model = reg.primary
    for kk, v in list(pan.items()):
        parts = kk.split(".")
        if len(parts) >= 3 and not parts[1].endswith("vs_v3") and parts[1] in use:
            res.set("E7.%s.%s.%s" % (parts[1], model, ".".join(parts[2:])), v)
    res.update(pan)
    res.set("E7.status", "ok"); res.set("E7.run", run.run_id); res.set("E7.n", int(len(y))); res.set("E7.scores", ", ".join(use))
    for c_ in (40, 60, 80):                                                # coverage levels as plain numbers for the text (design constants, not estimates)
        res.set("E7.cov%d" % c_, c_); res.set("E7.deferral_cov%d" % c_, 100 - c_)
    res.set("E7.deferral", 40)
    res.set("E7.ex_rate", float(y.mean()))
    res.set("E7.aurc.score", pan.get("E7.v3.aurc.mean")); res.set("E7.aurc.consistency", pan.get("E7.consistency.aurc.mean"))
    # extra-call accounting
    ns = [r.get("n_samples") for r in rows if r.get("n_samples") is not None]
    if ns:
        res.set("E7.route_ca.extra_calls.mean", 1.0 + float(np.mean(ns))); res.set("E7.consistency.extra_calls.mean", float(np.mean(ns)))
    res.set("E7.v3.extra_calls.mean", 1.0); res.set("E7.exec.extra_calls.mean", 0.0)
    # curves (+ band for the two headline scores) and reliability data
    for name in use:
        cov, risk = S.risk_coverage_curve(use[name], y, 100)
        res.set("E7.curve.%s.coverage" % name, cov); res.set("E7.curve.%s.risk" % name, risk)
    band = curve_band({n: use[n] for n in ("route_ca", "v3") if n in use}, y, boot)
    for name, (cov, lo, hi) in band.items():
        res.set("E7.band.%s.coverage" % name, cov); res.set("E7.band.%s.risk_lo" % name, lo); res.set("E7.band.%s.risk_hi" % name, hi)
    # RS(c) at coverage 60 / 80 % and for the run's own abstention flags
    n = len(y)
    for name in use:
        cum = S._expected_cum_acc(use[name], y)
        for cov in (0.6, 0.8):
            m = max(1, int(round(cov * n)))
            corr = m * cum[m - 1]
            wrong = m - corr
            for c in (0, 1, 5, 10, n):
                res.set("E7.%s.rs_c%s_cov%d.mean" % (name, "N" if c == n else c, int(cov * 100)), float((corr - c * wrong) / n))
    ab = np.array([1 if r.get("abstained") else 0 for r in np.array(rows, dtype=object)[base_ok]])
    corr = int(((ab == 0) & (y == 1)).sum()); wrong = int(((ab == 0) & (y == 0)).sum())
    res.set("E7.own.coverage", float(1 - ab.mean())); res.set("E7.own.ex_accepted", float(y[ab == 0].mean()) if (ab == 0).any() else None)
    res.set("E7.own.ex_all", float(((ab == 0) & (y == 1)).sum() / float(n)))
    res.set("E7.own.ex_with_oracle_human", float((1 - ab.mean()) * (y[ab == 0].mean() if (ab == 0).any() else 0) + ab.mean()))
    for c in (0, 1, 5, 10, n):
        res.set("E7.own.rs_c%s.mean" % ("N" if c == n else c), float((corr - c * wrong) / n))
    # calibration: raw reliability bins; isotonic fitted on HR (calib_preds) applied here
    for name in ("route_ca", "v3"):
        if name not in use:
            continue
        p = np.clip(use[name] / 10.0, 0, 1)
        res.set("E7.rel.%s.bins" % name, [b for b in S.reliability_bins(p, y, 10)])
    if calib_preds:
        hr = [r for r in R.jl(calib_preds)]
        hv3, hca = primary.route_scores(hr, coef, k)
        hy = np.array([R.ex_value(r) if R.ex_value(r) is not None else -1 for r in hr])
        for name, hv in (("v3", hv3), ("route_ca", hca)):
            ok = np.isfinite(hv) & (hy >= 0)
            if name in use and ok.sum() > 50:
                iso = S.isotonic_fit(np.clip(hv[ok] / 10.0, 0, 1), hy[ok])
                pc = S.isotonic_apply(iso, np.clip(use[name] / 10.0, 0, 1))
                eb = boot.stat(lambda idx: S.ece_equal_mass(pc[idx], y[idx], 15), min(B, 1000))
                res.ci("E7.%s.ece_cal" % name, S.ece_equal_mass(pc, y, 15), *S.pctl_ci(eb))
                res.set("E7.%s.brier_cal.mean" % name, S.brier(pc, y))
                res.set("E7.rel.%s.bins_cal" % name, S.reliability_bins(pc, y, 10))
        res.set("E7.calibration", "isotonic fitted on %s, applied to %s" % (calib_preds, reg.split))
    else:
        res.set("E7.calibration", "raw only (no --calib-preds given)")
    # bon_vote self-consistency confidence on its own answers (separate unit: its own correctness)
    try:
        bv = reg.get("bon_vote")
    except R.RunError as e:                      # two bon_vote budgets (n=2, n=3) are complete: the SC5 side readout is skipped, it must not abort the whole E7 stage
        res.note("E7: sc_bon_vote skipped - %s" % e)
        bv = None
    if bv is not None:
        brows = [bv.rows[q] for q in sorted(bv.rows)]
        sc5, by = [], []
        for r in brows:
            ex_ = R.ex_value(r)
            v = (r.get("extra") or {}).get("vote") or {}
            nn = (r.get("extra") or {}).get("n")
            if ex_ is None or not nn:
                continue
            sc5.append(10.0 * v.get("top_cluster", 0) / float(nn)); by.append(ex_)
        if len(by) > 50:
            res.set("E7.sc_bon_vote.auroc.mean", S.auroc(sc5, by)); res.set("E7.sc_bon_vote.aurc.mean", S.aurc_expected(sc5, by))
            res.note("E7: SC5 (bon_vote top-cluster share) is scored on bon_vote's own answers, so it is not a paired comparison with route_ca")


def run_e5(reg, res, B, full_label="full_gated"):
    """AUROC of the Critic score and its sub-scores on (a) final answers, (b) all stored candidates; ECE/Brier raw."""
    run = reg.get("full_fixed") or reg.get(full_label)
    if run is None or not run.cands:
        res.set("E5.status", "not run: no run with cands.jsonl")
        return
    cands = [c for c in run.cands if c.get("ex") is not None]
    if len(cands) < 100:
        res.set("E5.status", "not run: fewer than 100 scored candidates")
        return
    y = np.array([int(c["ex"]) for c in cands])
    dbs = np.array([c.get("db_id") for c in cands])
    fields = {"score": "v3", "semantic": "semantic", "syntax": "syntax", "efficiency": "efficiency"}
    sc = {}
    for f, nm in fields.items():
        v = np.array([float(c[f]) if c.get(f) is not None else np.nan for c in cands])
        if np.isfinite(v).all():
            sc[nm] = v
    if "v3" not in sc:
        res.set("E5.status", "not run: candidates carry no V3 score")
        return
    boot = S.ClusterBoot(dbs, min(B, 2000))
    pan, bs, pt = routing.panel(sc, y, dbs, boot, ref="v3", prefix="E5.cands.")
    res.update(pan)
    res.set("E5.status", "ok"); res.set("E5.run", run.run_id); res.set("E5.n_cands", len(cands))
    res.set("E5.weighted.auroc", pan.get("E5.cands.v3.auroc.mean")); res.set("E5.weighted.ece", pan.get("E5.cands.v3.ece_raw.mean"))
    b = (sc["v3"] >= 8.5).astype(float)
    res.set("E5.binary.auroc", S.auroc(b, y))
    res.note("E5: binary critic V1 is approximated by V3 >= 8.5 (patent threshold); V1-V4 end-to-end variants were dropped by MINIMAL_PLAN")
