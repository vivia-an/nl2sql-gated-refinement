"""rv4t0_derive: derived min / max keys over the loop arms of the rerun summary."""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import rv4t0_derive  # noqa: E402

SUMMARY = os.path.join(HERE, "..", "..", "results", "r4_t0", "rv4t0_summary.json")


def _toy():
    keys = {}
    for i, b in enumerate(rv4t0_derive.BACKBONES):
        for j, a in enumerate(rv4t0_derive.LOOP_ARMS):
            keys["nd.%s.%s.exemplars_differ" % (b, a)] = {"v": 1.0 + i + 0.5 * j, "kind": "pct"}
    return {"keys": keys}


def test_range_and_count_over_nine_loop_arms():
    s = rv4t0_derive.derive(_toy())["keys"]
    assert s["nd.complete.n_loop_arms"]["v"] == 9
    assert s["nd.complete.exemplars_differ_range"]["v"] == [1.0, 4.0]


def test_idempotent():
    once = rv4t0_derive.derive(_toy())
    twice = rv4t0_derive.derive(copy.deepcopy(once))
    assert once == twice


def test_missing_inputs_raise():
    try:
        rv4t0_derive.derive({"keys": {}})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_stored_summary_matches_derivation():
    with open(SUMMARY) as f:
        stored = json.load(f)
    again = rv4t0_derive.derive(copy.deepcopy(stored))
    assert again["keys"]["nd.complete.n_loop_arms"] == stored["keys"]["nd.complete.n_loop_arms"]
    assert again["keys"]["nd.complete.exemplars_differ_range"]["v"] == stored["keys"]["nd.complete.exemplars_differ_range"]["v"]


# --- round 6 keys (revision list 6 C1, C6, C11, C12): recomputed here from the CSV files, not from the summary ---
import csv  # noqa: E402
import math  # noqa: E402

R4T0 = os.path.join(HERE, "..", "..", "results", "r4_t0")


def _rows(name):
    with open(os.path.join(R4T0, name)) as f:
        return list(csv.DictReader(f))


def test_round6_reading_counts_from_the_contrast_csv():
    with open(SUMMARY) as f:
        keys = json.load(f)["keys"]
    rows = _rows("contrasts_r1.csv")
    assert len(rows) == 25
    nonnull = [r for r in rows if r["reading_s0_wide"] != "contains 0"]
    kept = [r for r in nonnull if r["reading_r1_wide"] == r["reading_s0_wide"]]
    both_null = [r for r in rows if r["reading_s0_wide"] == "contains 0" and r["reading_r1_wide"] == "contains 0"]
    lt2 = [r for r in rows if abs(float(r["est_s0"])) < 2 * float(r["sd_indep_pp"])]
    assert keys["ctr.n_nonnull_s0"]["v"] == len(nonnull) == 11
    assert keys["ctr.n_nonnull_kept"]["v"] == len(kept) == 10
    assert keys["ctr.n_null_both"]["v"] == len(both_null) == 14
    assert keys["ctr.n_lt2sd"]["v"] == len(lt2)
    # the one change is the fixed loop against the single call on 27B
    changed = [r for r in nonnull if r["reading_r1_wide"] != r["reading_s0_wide"]]
    assert [(r["contrast"], r["scope"]) for r in changed] == [("f_minus_zs", "27b")]


def test_round6_mean_arm_change_from_the_arm_csv():
    with open(SUMMARY) as f:
        keys = json.load(f)["keys"]
    rows = _rows("arms_partial.csv")
    assert len(rows) == 12
    d = [float(r["ex_r1"]) - float(r["ex_s0"]) for r in rows]
    assert abs(keys["nd.complete.ex_diff_mean"]["v"] - sum(d) / 12) < 0.01       # CSV values are rounded to 3 decimals
    assert keys["nd.complete.n_ex_diff_le0"]["v"] == sum(1 for x in d if x <= 1e-9)
    sd = math.sqrt(sum(float(r["sd_net_pp"]) ** 2 for r in rows)) / 12
    assert abs(keys["nd.complete.ex_diff_mean_sd"]["v"] - sd) < 0.005


def test_round6_allowance_formula_and_p1p_values():
    assert abs(rv4t0_derive.allowance(-0.26, -1.06, 0.0) - (-1.06)) < 1e-12        # no generation variance: the interval is unchanged
    assert abs(rv4t0_derive.allowance(0.0, -3.0, 4.0 / 1.96) - (-5.0)) < 1e-9       # 3-4-5 in quadrature
    with open(SUMMARY) as f:
        keys = json.load(f)["keys"]
    for run, wide in (("s0", "ctr.g_minus_f.pooled3.s0_wide"), ("r1", "ctr.g_minus_f.pooled3.r1_wide")):
        lo = keys["p1p.allow_%s_lo" % run]["v"]
        assert lo < keys[wide]["v"][0]                                              # widening only lowers the end
        assert abs(keys["p1p.allow_%s_clearance" % run]["v"] - (lo - keys["p1p.margin"]["v"])) < 1e-12
        assert lo > keys["p1p.margin"]["v"]                                          # margin kept in both runs (the sentence in the paper depends on it)
        # the direct scale (arms share roots) is smaller, so its end lies between the interval end and the independence end
        assert keys[wide]["v"][0] > keys["p1p.allow_direct_%s_lo" % run]["v"] > lo


def test_round6_idempotent():
    with open(SUMMARY) as f:
        stored = json.load(f)
    again = rv4t0_derive.derive(copy.deepcopy(stored))
    assert again["keys"] == stored["keys"]


# --- round 7 keys (revision list 7 D1): reading line = twice the rerun-change scale; recomputed from the CSV, not from the summary ---
def test_round7_reading_line_keys_from_the_contrast_csv():
    with open(SUMMARY) as f:
        keys = json.load(f)["keys"]
    rows = _rows("contrasts_r1.csv")
    for r in rows:
        cid = "%s.%s" % (r["contrast"], r["scope"])
        assert abs(keys["ctr.%s.line_indep" % cid]["v"] - 2 * float(r["sd_indep_pp"])) < 0.002        # CSV values are rounded to 3 decimals
        assert abs(keys["ctr.%s.line_direct" % cid]["v"] - 2 * float(r["sd_direct_pp"])) < 0.002
        assert abs(keys["ctr.%s.times_indep" % cid]["v"] - abs(float(r["est_s0"])) / float(r["sd_indep_pp"])) < 0.002 * max(1.0, abs(keys["ctr.%s.times_indep" % cid]["v"]))
    lt = [r for r in rows if abs(float(r["est_s0"])) < 2 * float(r["sd_indep_pp"])]
    lt_d = [r for r in rows if abs(float(r["est_s0"])) < 2 * float(r["sd_direct_pp"])]
    assert keys["ctr.n_lt2sd"]["v"] == len(lt) == 12
    assert keys["ctr.n_ge2sd"]["v"] == 25 - len(lt) == 13
    assert keys["ctr.n_lt2sd_direct"]["v"] == len(lt_d) == 10
    below = sorted((r["contrast"], r["scope"]) for r in lt)
    assert ("loop_g", "flash") in below and ("loop_g", "pooled3") not in below          # Flash retry below the line, pooled retry just above it
    pooled = keys["ctr.loop_g.pooled3.times_indep"]["v"]
    assert 2.0 < pooled < 2.1
    assert keys["ctr.g_minus_zs.plus.line_indep"]["v"] > keys["ctr.g_minus_zs.plus.s0"]["v"] > 0.6          # +0.65 against a line of 1.20


# --- round 8 keys (V8 M-1, M-2, L-2): direct-scale multiples, three-decimal scale, and the convention check; recomputed from the CSV ---
def test_round8_keys_direct_multiple_scale_and_zero_convention():
    with open(SUMMARY) as f:
        keys = json.load(f)["keys"]
    rows = _rows("contrasts_r1.csv")
    for r in rows:
        cid = "%s.%s" % (r["contrast"], r["scope"])
        est, sd = abs(float(r["est_s0"])), float(r["sd_direct_pp"])
        assert abs(keys["ctr.%s.times_direct" % cid]["v"] - est / sd) < 0.003 * max(1.0, est / sd)
        assert abs(keys["ctr.%s.sd_indep3" % cid]["v"] - float(r["sd_indep_pp"])) < 0.0006
    # M-2: exactly two contrasts change class between the independence and the direct scale: the gated and the fixed loop over root-only on Flash; the pooled retry is above both
    cls = lambda r, col: abs(float(r["est_s0"])) < 2 * float(r[col])
    changed = sorted((r["contrast"], r["scope"]) for r in rows if cls(r, "sd_indep_pp") != cls(r, "sd_direct_pp"))
    assert changed == [("loop_f", "flash"), ("loop_g", "flash")]
    assert 1.7 < keys["ctr.loop_f.flash.times_indep"]["v"] < 2.0 < keys["ctr.loop_f.flash.times_direct"]["v"]
    assert keys["ctr.loop_g.pooled3.times_indep"]["v"] > 2 and keys["ctr.loop_g.pooled3.times_direct"]["v"] > 2
    # L-2: the line of the Plus gated loop against the single call is twice the unrounded scale, 1.19 (not twice the rounded 0.60)
    assert "%.3f" % keys["ctr.g_minus_zs.plus.sd_indep3"]["v"] == "0.597" and keys["ctr.g_minus_zs.plus.sd_indep3"]["kind"] == "x3"
    assert abs(keys["ctr.g_minus_zs.plus.line_indep"]["v"] - 1.1949) < 1e-3


def test_round8_zero_convention_changes_the_exception_not_the_rerun_count():
    with open(SUMMARY) as f:
        keys = json.load(f)["keys"]
    w = lambda cid, run: keys["ctr.%s.%s_wide" % (cid, run)]["v"]
    # the two contrasts whose rerun lower end is zero within the sign-flip tolerance
    zero_end = sorted(c for c in ("f_minus_zs.27b", "loop_g.plus") if abs(w(c, "r1")[0]) < 1e-6)
    assert zero_end == ["f_minus_zs.27b", "loop_g.plus"]
    pos = lambda lo: lo > 0
    # under the SAP rule (end at zero contains 0): the 27B contrast changes reading (locked positive, rerun contains 0); Plus keeps (contains 0 in both)
    assert pos(w("f_minus_zs.27b", "s0")[0]) and not pos(w("f_minus_zs.27b", "r1")[0]) and not pos(w("loop_g.plus", "s0")[0])
    # read as positive: 27B keeps, Plus changes -> still one exception
    assert keys["ctr.n_same_wide"]["v"] == keys["ctr.n_same_wide_zero_positive"]["v"] == 24
