"""Tests of the driver functions of code/tools/rv2_small.py (S2 statistics add-on; post hoc). Synthetic inputs only. Needs numpy, scipy, pyyaml
(worker machine: cd code && python -m pytest tests/test_rv2_small.py)."""
import os
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("yaml")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, ".."))
import rv2_small as M  # noqa: E402

A = M.A
M.install_wrappers()


def _fresh():
    M.OUT.clear()
    M.CAP.clear()


def _synthetic(G=11, per=30, shift=1.2, seed=0):
    rng = np.random.RandomState(seed)
    dbs = np.repeat(np.array(["db%02d" % i for i in range(G)]), per)
    d = rng.choice([-1.0, 0.0, 1.0], size=G * per, p=[0.1, 0.7, 0.2]) * 1.0
    return d, dbs


def test_fmt_ratio_iv_cases():
    assert M.fmt_ratio_iv(0.5, 1.9, 1.5) == (pytest.approx(1.5 / 1.9), pytest.approx(3.0), "[0.79, 3.00]")
    lo, hi, txt = M.fmt_ratio_iv(-0.2, 0.8, 1.5)
    assert hi is None and txt == "[1.88, unbounded]" and lo == pytest.approx(1.875)
    assert M.fmt_ratio_iv(-0.9, -0.1, 1.5)[2].startswith("n/a")
    assert M.fmt_ratio_iv(None, None, 1.5)[2].startswith("n/a")


def test_small_block_additive_writes_every_key_and_orders_the_intervals():
    _fresh()
    d, dbs = _synthetic()
    e = A.est_mean(d, dbs, 500)
    assert "_d" in e and "_loo" in e and len(e["_loo"]) == 11
    s = M.small_block("W1.pooled", e)
    for k, _ in M.KS.expected_keys():
        pass
    wrote = {k for k in M.OUT if k.startswith("small.W1.pooled.")}
    want = {k for k, _ in M.KS.expected_keys() if k.startswith("small.W1.pooled.")}
    assert wrote == want
    iv = s["ivs"]
    assert iv["wide"][0] <= min(iv["cl"][0], iv["item"][0], iv["t"][0], iv["sf"][0]) + 1e-12
    assert iv["wideu"][0] <= iv["wide"][0] + 1e-12 and iv["wideu"][1] >= iv["wide"][1] - 1e-12
    assert iv["old"] == (min(iv["cl"][0], iv["item"][0]), max(iv["cl"][1], iv["item"][1]))
    assert s["additive"] and s["deff_cr"] is not None and s["p_sf"] >= 2.0 ** -10 - 1e-12


def test_small_block_jackknife_for_ratio_statistics():
    _fresh()
    d, dbs = _synthetic(seed=3)
    num = np.abs(d)
    den = np.ones_like(d) * 2.0
    e = A.est_ratio(num, den, dbs, 500)
    assert "_d" not in e and len(e["_loo"]) == 11
    s = M.small_block("P1p_ratio.pooled", e)
    assert not s["additive"] and s["deff_cr"] is None
    assert s["ivs"]["tdb"] == s["ivs"]["t"] and s["ivs"]["sfu"] == s["ivs"]["sf"]
    lo, hi = s["ivs"]["t"]
    assert lo < e["est"] < hi


def test_item_cost_discount_only_changes_cached_tokens():
    class R:
        pass
    r = R()
    r.rows = {"q1": {"model": "qwen3.7-plus", "prompt_tokens": 1000, "completion_tokens": 100, "cached_tokens": 600},
              "q2": {"model": "qwen3.7-plus", "prompt_tokens": 500, "completion_tokens": 50, "cached_tokens": 0}}
    prices = {"cache_hit_ratio": 0.2, "models": {"qwen3.7-plus": {"in": 2.0, "out": 8.0}}}
    nod = M.item_cost2(r, ["q1", "q2"], prices, False)
    dis = M.item_cost2(r, ["q1", "q2"], prices, True)
    assert nod[0] == pytest.approx((1000 * 2.0 + 100 * 8.0) / 1e6)
    assert dis[0] == pytest.approx((400 * 2.0 + 600 * 2.0 * 0.2 + 100 * 8.0) / 1e6)
    assert dis[1] == pytest.approx(nod[1])
    assert (dis <= nod).all()


def test_grammar_key_guard():
    _fresh()
    with pytest.raises(KeyError):
        M.put("small.not.a.key", 1.0)
    M.put("small.def", "x")
    with pytest.raises(KeyError):
        M.put("small.def", "y")
    _fresh()
