"""Derived RV4T0 keys (host, pure Python): quantities that are min / max over arms of keys already in results/r4_t0/rv4t0_summary.json.

The summary is written by code/tools/r4_t0.py from the raw rerun directories; the keys below need no raw data, only the per-arm keys, so they are added here
(idempotent) and the grammar is regenerated from the result:

  python3 code/tools/rv4t0_derive.py results/r4_t0/rv4t0_summary.json && python3 code/tools/rv4t0_keys.py --write-grammar results/r4_t0/rv4t0_summary.json

  nd.complete.n_loop_arms             loop arms (root-only, gated, fixed on three backbones) whose retrieved example set can differ from the locked run
  nd.complete.exemplars_differ_range  lowest and highest share (%) of questions with a different retrieved example set over those arms (the retrieval cache base of the
                                      rerun was not the cache of the locked run; see internal note R4T0 section 9b)

Round 7 (revision list 7 D1; no model call): ctr.<c>.<scope>.line_indep / line_direct / times_indep (reading line = twice the rerun-change scale; the
  independence-formula scale decides, the direct scale is the bound for arms that share roots), ctr.n_ge2sd, ctr.n_lt2sd_direct.

Round 8 (V8 M-1, M-2, L-2; no model call): ctr.<c>.<scope>.times_direct (estimate in units of the direct scale), ctr.<c>.<scope>.sd_indep3 (independence scale, 3 decimals),
  ctr.n_same_wide_zero_positive (contrasts with the same reading in both runs if an end at zero within 1e-6 were read as positive: the same 24 as ctr.n_same_wide).

Round 6 (revision list 6 C1, C6, C11, C12; no model call, no new data; every value is a function of keys already in the summary):
  ctr.n_nonnull_s0 / n_nonnull_kept / n_null_both   of the 25 contrasts: readings positive or negative in the locked run (widest interval), those that keep the reading in the
                                      rerun, and those that read "contains 0" in both runs
  ctr.n_lt2sd                         contrasts whose absolute locked-run estimate is below twice their independence-formula scale (sd_indep): the rule for "at or below the noise scale"
  nd.complete.ex_diff_mean            mean of the 12 arm-level EX changes (rerun minus locked run), pp
  nd.complete.n_ex_diff_le0           arms whose EX change is at or below 0
  nd.complete.ex_diff_mean_sd         SD of that mean if the 12 arms varied independently: sqrt(sum sd_net_i^2) / 12 (arms of one chain are positively correlated, so this is a floor)
  p1p.allow_s0_lo / allow_r1_lo       P1': lower end of the widest interval widened for generation variance, lo' = est - sqrt((est - lo)^2 + (1.96 s)^2), with s the pooled independence
                                      scale of the contrast (the SD of the net change between two identical runs; one run's SD is s / sqrt(2), so this is the conservative reading);
                                      allow_*_clearance = lo' minus the margin; allow_direct_* the same with the direct scale (per-question change of the contrast itself)
"""
import json
import math
import sys

BACKBONES = ("plus", "flash", "27b")
LOOP_ARMS = ("rootonly", "gated", "fixed")
ARMS = ("zs", "rootonly", "gated", "fixed")
CONTRASTS = ("stack", "loop_g", "loop_f", "g_minus_f", "g_minus_zs", "f_minus_zs")
SCOPES = ("plus", "flash", "27b", "pooled3")
Z95 = 1.96


def derive(summary):
    keys = summary["keys"]
    vals = []
    for b in BACKBONES:
        for a in LOOP_ARMS:
            k = "nd.%s.%s.exemplars_differ" % (b, a)
            if k in keys:
                vals.append(float(keys[k]["v"]))
    if not vals:
        raise ValueError("no nd.<b>.<arm>.exemplars_differ keys in the summary")
    keys["nd.complete.n_loop_arms"] = {"v": len(vals), "kind": "int",
                                       "meaning": "loop arms (root-only, gated, fixed; three backbones) with a retrieved example set to compare"}
    keys["nd.complete.exemplars_differ_range"] = {"v": [min(vals), max(vals)], "kind": "pctv2",
                                                  "meaning": "lowest and highest share (%) of questions with a different retrieved example set, loop arms"}
    if "ctr.n_total" in keys:          # a real summary (the toy summaries of the tests carry no contrast keys)
        _derive_round6(keys)
    return summary


def _put(keys, k, v, kind, meaning):
    keys[k] = {"v": v, "kind": kind, "meaning": meaning}


def _contrast_ids():
    ids = ["%s.%s" % (c, s) for c in CONTRASTS for s in SCOPES]
    ids.append("w1.weak2")
    return ids


def allowance(est, lo, scale, z=Z95):
    """Lower end of an interval widened in quadrature by z * scale (generation-variance allowance)."""
    return est - math.sqrt((est - lo) ** 2 + (z * scale) ** 2)


def _derive_round6(keys):
    ids = _contrast_ids()
    # readings of the 25 contrasts
    n_nonnull = n_kept = n_null_both = n_lt2 = 0
    for cid in ids:
        a = keys["ctr.%s.reading_s0_wide" % cid]["v"]
        b = keys["ctr.%s.reading_r1_wide" % cid]["v"]
        if a != "contains 0":
            n_nonnull += 1
            if a == b:
                n_kept += 1
        elif b == "contains 0":
            n_null_both += 1
        if abs(float(keys["ctr.%s.s0" % cid]["v"])) < 2 * float(keys["ctr.%s.sd_indep" % cid]["v"]):
            n_lt2 += 1
    _put(keys, "ctr.n_nonnull_s0", n_nonnull, "int", "contrasts whose widest reading in the locked run is positive or negative")
    _put(keys, "ctr.n_nonnull_kept", n_kept, "int", "of those, contrasts that keep the same reading in the rerun")
    _put(keys, "ctr.n_null_both", n_null_both, "int", "contrasts that read 'contains 0' under the widest interval in both runs")
    _put(keys, "ctr.n_lt2sd", n_lt2, "int", "contrasts whose absolute locked-run estimate is below twice their independence-formula scale (rule for 'at or below the noise scale')")
    # round 7 (revision list 7 D1): the reading line of a contrast is twice its rerun-change scale (independence formula decides; the direct scale is the bound for contrasts of arms that share roots)
    n_lt2_direct = 0
    for cid in ids:
        est = abs(float(keys["ctr.%s.s0" % cid]["v"]))
        si = float(keys["ctr.%s.sd_indep" % cid]["v"])
        sd = float(keys["ctr.%s.sd_direct" % cid]["v"])
        _put(keys, "ctr.%s.line_indep" % cid, 2 * si, "x2", "reading line of %s: twice the independence-formula rerun-change scale (pp)" % cid)
        _put(keys, "ctr.%s.line_direct" % cid, 2 * sd, "x2", "the same from the direct scale (pp)")
        _put(keys, "ctr.%s.times_indep" % cid, est / si, "x2", "absolute locked-run estimate of %s in units of its independence-formula scale" % cid)
        # round 8 (V8 M-2, L-2): the same in units of the direct scale; the independence scale to three decimals (the line is twice the unrounded scale)
        _put(keys, "ctr.%s.times_direct" % cid, est / sd, "x2", "absolute locked-run estimate of %s in units of its direct scale" % cid)
        _put(keys, "ctr.%s.sd_indep3" % cid, si, "x3", "independence-formula rerun-change scale of %s to three decimals (pp); the reading line is twice the unrounded value" % cid)
        n_lt2_direct += int(est < 2 * sd)
    # round 8 (V8 M-1): an interval end at zero up to the sign-flip tolerance counts as containing 0 (SAP SUP(0)); read as positive, which contrast is the exception changes, not how many agree
    def _word(w, tol=1e-6):
        lo, hi = float(w[0]), float(w[1])
        if abs(lo) < tol:
            lo = tol
        return "positive" if lo > 0 else ("negative" if hi < 0 else "contains 0")

    n_same_alt = sum(int(_word(keys["ctr.%s.s0_wide" % cid]["v"]) == _word(keys["ctr.%s.r1_wide" % cid]["v"])) for cid in ids)
    _put(keys, "ctr.n_same_wide_zero_positive", n_same_alt, "int", "contrasts with the same widest reading in the locked run and the rerun if an interval end at zero (within 1e-6) were read as positive instead of containing 0")
    _put(keys, "ctr.n_ge2sd", len(ids) - n_lt2, "int", "contrasts whose absolute locked-run estimate is at or above their reading line (twice the independence-formula scale)")
    _put(keys, "ctr.n_lt2sd_direct", n_lt2_direct, "int", "contrasts below twice their direct scale (the direct scale does not assume independent arms; not the deciding scale)")
    # arm-level EX changes
    d = []
    var = 0.0
    for b in BACKBONES:
        for a in ARMS:
            d.append(float(keys["nd.%s.%s.ex_diff" % (b, a)]["v"]))
            var += float(keys["nd.%s.%s.sd_net" % (b, a)]["v"]) ** 2
    n = len(d)
    _put(keys, "nd.complete.ex_diff_mean", sum(d) / n, "pp2", "mean over the %d arms of the EX change (rerun minus locked run), pp" % n)
    _put(keys, "nd.complete.n_ex_diff_le0", sum(1 for x in d if x <= 0), "int", "arms whose EX change is at or below 0")
    _put(keys, "nd.complete.ex_diff_mean_sd", math.sqrt(var) / n, "x2",
         "SD (pp) of the mean EX change if the arms varied independently, sqrt(sum sd_net^2) / n (arms of one chain are positively correlated)")
    # P1' with a generation-variance allowance
    cid = "g_minus_f.pooled3"
    est = {"s0": float(keys["ctr.%s.s0" % cid]["v"]), "r1": float(keys["ctr.%s.r1" % cid]["v"])}
    wide = {"s0": keys["ctr.%s.s0_wide" % cid]["v"], "r1": keys["ctr.%s.r1_wide" % cid]["v"]}
    margin = float(keys["p1p.margin"]["v"])
    for tag, sk in (("", "sd_indep"), ("direct_", "sd_direct")):
        s = float(keys["ctr.%s.%s" % (cid, sk)]["v"])
        for run in ("s0", "r1"):
            lo = allowance(est[run], float(wide[run][0]), s)
            _put(keys, "p1p.allow_%s%s_lo" % (tag, run), lo, "pp2",
                 "P1': lower end of the widest interval of %s widened in quadrature by 1.96 x the %s scale of the pooled contrast, pp" % (run, "independence" if not tag else "direct"))
            _put(keys, "p1p.allow_%s%s_clearance" % (tag, run), lo - margin, "pp2",
                 "that widened lower end minus the margin (pp); positive = margin kept")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    with open(sys.argv[1]) as f:
        s = json.load(f)
    derive(s)
    with open(sys.argv[1], "w") as f:
        json.dump(s, f, indent=1)
    print("derived keys added; %d keys" % len(s["keys"]))
