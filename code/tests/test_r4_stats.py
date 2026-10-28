"""Tests of the pure functions of code/tools/r4_stats.py (round-4 post hoc statistics: single widest interval, cost per extra correct answer, margin ratio).
Needs numpy and scipy (worker machine, $R/venv); skipped on the host. Run with PYTHONPATH=code/tools:code."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(HERE, ".."))
try:
    import numpy as np
    import r4_stats as R4
except ImportError:         # host python 3.6 without numpy
    np = None


def _skip():
    if np is None:
        try:
            import pytest
            pytest.skip("numpy not available on this machine")
        except ImportError:
            return True
    return False


def test_icer_defined_cases():
    if _skip():
        return
    r = R4.icer(1.0, (0.5, 1.5), 0.01)                  # 1 pp gain, interval above zero, 0.01 CNY per question
    assert r["defined"] and abs(r["est"] - 1.0) < 1e-12 and abs(r["lo"] - 0.01 / 0.015) < 1e-12 and abs(r["hi"] - 0.01 / 0.005) < 1e-12
    r = R4.icer(1.0, (-0.5, 1.5), 0.01)                 # interval includes zero: unbounded above
    assert r["defined"] and r["hi"] is None and abs(r["lo"] - 0.01 / 0.015) < 1e-12 and "unbounded" in r["reason"]


def test_icer_undefined_cases():
    if _skip():
        return
    assert not R4.icer(0.0, (-1.0, 1.0), 0.01)["defined"]            # estimate not positive
    assert not R4.icer(-0.2, (-1.0, 1.0), 0.01)["defined"]
    assert not R4.icer(1.0, (0.5, 1.5), -0.001)["defined"]           # not more expensive
    r = R4.icer(1.0, (-1.5, -0.1), 0.01)                              # estimate positive but interval entirely below zero
    assert r["defined"] and r["lo"] is None


def test_small4_hull_contains_the_other_intervals():
    if _skip():
        return
    rng = np.random.RandomState(3)
    dbs = np.repeat(np.arange(11), 100)
    vec = rng.choice([-1.0, 0.0, 1.0], size=len(dbs), p=[0.05, 0.9, 0.05]) + 0.0
    cl, item = (-0.5, 0.6), (-0.3, 0.4)
    o = R4.small4(vec, dbs, cl, item)
    w = o["wide"]
    for iv in (cl, item, o["t"], o["sf"]):
        assert w[0] <= iv[0] + 1e-12 and w[1] >= iv[1] - 1e-12
    assert o["t"][0] < o["t"][1] and o["sf"][0] <= o["sf"][1]


def test_widest_reading_words():
    if _skip():
        return
    assert R4.widest_reading((0.1, 1.0)) == "positive" and R4.widest_reading((-1.0, -0.1)) == "negative" and R4.widest_reading((-0.1, 1.0)) == "contains 0"
    assert R4.widest_reading((0.0, 1.0)) == "contains 0"            # a lower end of exactly zero is not positive


def test_margin_block_ratio_and_share():
    if _skip():
        return
    rng = np.random.RandomState(5)
    n = 2200
    dbs = np.repeat(np.arange(11), n // 11)
    vec = {}
    for b in R4.C.BK:
        vec[(b, "fixed")] = (rng.rand(n) < 0.03).astype(float) * 1.0       # fixed loop effect about 3 pp
        vec[(b, "gated")] = np.where(vec[(b, "fixed")] > 0, (rng.rand(n) < 0.5).astype(float), 0.0)
    m = R4.margin_block(vec, dbs, B=400)
    assert abs(m["margin_over_eff"] - R4.MARGIN / m["eff_fixed"]) < 1e-12
    assert abs(m["share_kept"] - m["eff_gated"] / m["eff_fixed"]) < 1e-12
    assert 0.2 < m["share_kept"] < 0.8
    lo, hi = m["margin_over_eff_wide"]
    assert lo <= m["margin_over_eff_ci"][0] + 1e-9 and hi >= m["margin_over_eff_item_ci"][1] - 1e-9


def test_hull_open_none_means_unbounded():
    if _skip():
        return
    assert R4.hull_open((1.0, 2.0), (0.5, 1.5)) == (0.5, 2.0)
    assert R4.hull_open((1.0, 2.0), (0.5, None)) == (0.5, None)
    assert R4.hull_open((None, 2.0), (0.5, 3.0)) == (None, 3.0)


def _loop_data(seed, n=2200, g_gain=0.5):
    rng = np.random.RandomState(seed)
    dbs = np.repeat(np.arange(11), n // 11)
    f = (rng.rand(n) < 0.04).astype(float)                       # fixed loop effect about 4 pp
    g = np.where(f > 0, (rng.rand(n) < g_gain).astype(float), 0.0)
    return g, f, dbs


def test_ratio_interval_contains_the_point_estimate_and_both_methods_agree_on_one():
    if _skip():
        return
    g, f, dbs = _loop_data(11)
    est = float(g.mean() / f.mean())
    for kind in ("t", "sf"):
        lo, hi = R4.ratio_interval(g, f, dbs, kind)
        assert lo is not None and lo <= est + 1e-9 and (hi is None or hi >= est - 1e-9)
    # R = 1 is inside the interval exactly when the interval of the difference g - f holds 0 under the same method (round 5, A3)
    n_g, mg = R4._db_means(g, dbs)
    _, mf = R4._db_means(f, dbs)
    p1 = R4._p_t(mg - mf, n_g)
    lo, hi = R4.ratio_interval(g, f, dbs, "t")
    inside = (lo is None or lo <= 1.0) and (hi is None or hi >= 1.0)
    assert inside == (p1 >= 0.05)


def test_ratio_interval_is_unbounded_above_when_the_denominator_effect_is_noise():
    if _skip():
        return
    rng = np.random.RandomState(3)
    n = 2200
    dbs = np.repeat(np.arange(11), n // 11)
    f = (rng.rand(n) < 0.004).astype(float)
    g = (rng.rand(n) < 0.004).astype(float)
    lo, hi = R4.ratio_interval(g, f, dbs, "t")
    assert hi is None                                              # the fixed-loop effect cannot be told from zero


def test_margin_block_wide_is_the_four_interval_hull_and_has_the_gap():
    if _skip():
        return
    g, f, dbs = _loop_data(5)
    vec = {}
    for b in R4.C.BK:
        vec[(b, "fixed")] = f
        vec[(b, "gated")] = g
    m = R4.margin_block(vec, dbs, B=300)
    lo, hi = m["share_kept_wide"]
    assert lo <= m["share_kept_ci"][0] + 1e-9 and (hi is None or hi >= m["share_kept_item_ci"][1] - 1e-9)
    glo, ghi = m["gap_wide"]
    assert glo <= m["gap_ci"][0] + 1e-9 and ghi >= m["gap_item_ci"][1] - 1e-9
    assert abs(m["gap"] - (m["eff_gated"] - m["eff_fixed"])) < 1e-9


if __name__ == "__main__":
    bad = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                bad += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if bad else 0)
