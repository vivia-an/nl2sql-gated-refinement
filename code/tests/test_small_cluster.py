"""Tests of code/tools/small_cluster.py, rv2s_keys.py and the published results/posthoc2/rv2s_summary.json (statistics add-on S2, post hoc; scope posthoc-secondary).
Synthetic inputs only: no LLM call, no worker machine data. Needs numpy and scipy (worker machine: cd code && python -m pytest tests/test_small_cluster.py)."""
import itertools
import json
import math
import os
import sys

import pytest

np = pytest.importorskip("numpy")
sps = pytest.importorskip("scipy.stats")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import rv2_keys as K  # noqa: E402
import rv2s_keys as KS  # noqa: E402
import small_cluster as SC  # noqa: E402

RESULTS = os.environ.get("RV2_RESULTS", os.path.join(HERE, "..", "..", "results", "posthoc2"))


def brute_flip(v, w, theta0, alt):
    x = np.asarray(w) * (np.asarray(v) - theta0)
    obs = x.sum()
    T = np.array([sum(s * xi for s, xi in zip(sg, x)) for sg in itertools.product((1, -1), repeat=len(x))])
    if alt == "two":
        return float(np.mean(np.abs(T) >= abs(obs) - 1e-12))
    if alt == "gt":
        return float(np.mean(T >= obs - 1e-12))
    return float(np.mean(T <= obs + 1e-12))


def test_sign_matrix_has_all_patterns_and_observed_first():
    for G in (1, 3, 6):
        M = SC.sign_matrix(G)
        assert M.shape == (2 ** G, G)
        assert len({tuple(r) for r in M.tolist()}) == 2 ** G
        assert (M[0] == 1).all()


def test_sf_p_three_values_by_hand():
    v = [1.0, 2.0, 3.0]
    assert SC.sf_p(v, None, 0.0, "two") == pytest.approx(2 / 8)
    assert SC.sf_p(v, None, 0.0, "gt") == pytest.approx(1 / 8)
    assert SC.sf_p(v, None, 0.0, "lt") == pytest.approx(1.0)


@pytest.mark.parametrize("alt", ["two", "gt", "lt"])
def test_sf_p_matches_brute_force_with_weights_and_shift(alt):
    rng = np.random.RandomState(3)
    v = rng.normal(0.4, 1.0, 7)
    w = rng.uniform(0.5, 2.0, 7)
    w = w / w.sum()
    for th in (-0.3, 0.0, 0.7):
        assert SC.sf_p(v, w, th, alt) == pytest.approx(brute_flip(v, w, th, alt), abs=1e-12)


def test_smallest_attainable_p_is_two_to_minus_G():
    v = np.linspace(1.0, 2.0, 11)
    assert SC.sf_p(v, None, 0.0, "gt") == pytest.approx(1 / 2048)
    assert SC.sf_p(v, None, 0.0, "two") == pytest.approx(2 / 2048)
    assert SC.sf_min_p(11) == pytest.approx(2 / 2048) and SC.sf_min_p(11, "one") == pytest.approx(1 / 2048)
    assert SC.sf_p(np.linspace(1, 2, 19), None, 0.0, "two") == pytest.approx(2 / 2 ** 19)


def test_sf_ci_is_the_set_of_not_rejected_values():
    rng = np.random.RandomState(5)
    v = rng.normal(1.0, 1.5, 9)
    w = rng.uniform(1, 3, 9)
    w = w / w.sum()
    lo, hi = SC.sf_ci(w * 0 + v, w)
    est = float((w * v).sum())
    assert lo < est < hi
    grid = np.linspace(est - 8, est + 8, 4001)
    acc = [g for g in grid if SC.sf_p(v, w, g, "two") > 0.05]
    assert min(acc) == pytest.approx(lo, abs=0.01) and max(acc) == pytest.approx(hi, abs=0.01)
    # boundary behaviour: just inside is not rejected, just outside is
    assert SC.sf_p(v, w, lo + 1e-6, "two") > 0.05 and SC.sf_p(v, w, lo - 1e-3, "two") <= 0.05
    assert SC.sf_p(v, w, hi - 1e-6, "two") > 0.05 and SC.sf_p(v, w, hi + 1e-3, "two") <= 0.05


def test_sf_ci_equivariance_and_small_G():
    rng = np.random.RandomState(8)
    v = rng.normal(0, 1, 8)
    lo, hi = SC.sf_ci(v)
    lo2, hi2 = SC.sf_ci(3.0 * v + 5.0)
    assert lo2 == pytest.approx(3 * lo + 5, abs=1e-6) and hi2 == pytest.approx(3 * hi + 5, abs=1e-6)
    assert SC.sf_ci([1.0, 2.0, 3.0, 4.0, 5.0]) == (None, None)            # 2^(1-5) = 0.0625 > 0.05: cannot reject at 5 %
    assert SC.sf_ci([2.0] * 8) == (2.0, 2.0)


def test_cr_t_equals_database_t_when_clusters_are_balanced():
    rng = np.random.RandomState(1)
    G, m = 11, 40
    dbs = np.repeat(np.arange(G), m)
    d = rng.normal(0.3, 1.0, G * m) + np.repeat(rng.normal(0, 0.5, G), m)
    a, b = SC.cr_t(d, dbs), SC.t_db(d, dbs)
    assert a["est"] == pytest.approx(b["est"]) and a["se"] == pytest.approx(b["se"])
    assert a["df"] == 10
    q = sps.t.ppf(0.975, 10)
    assert a["lo"] == pytest.approx(a["est"] - q * a["se"]) and a["hi"] == pytest.approx(a["est"] + q * a["se"])
    assert a["p2"] == pytest.approx(2 * sps.t.sf(abs(a["est"] / a["se"]), 10))
    assert a["p_gt"] + a["p_lt"] == pytest.approx(1.0)


def test_cr_t_unbalanced_uses_item_weights_and_cr1_variance():
    d = np.array([1.0, 2.0, 3.0, 10.0, 0.0, 4.0, 5.0])
    dbs = np.array(["a", "a", "a", "b", "c", "c", "c"])
    c = SC.cr_t(d, dbs)
    est = d.mean()
    s = [(d[dbs == g] - est).sum() for g in "abc"]
    se = math.sqrt(3 / 2.0 * sum(x * x for x in s)) / 7.0
    assert c["est"] == pytest.approx(est) and c["se"] == pytest.approx(se)
    assert SC.t_db(d, dbs)["est"] == pytest.approx(np.mean([2.0, 10.0, 3.0]))


def test_design_effect_is_one_without_clustering():
    rng = np.random.RandomState(2)
    d = rng.normal(size=60)
    assert SC.deff_cr(d, np.arange(60)) == pytest.approx(1.0, rel=1e-9)


def test_pseudo_values_of_a_mean_are_the_cluster_means():
    m = np.array([1.0, 4.0, 2.5, 7.0, 3.0])
    G = len(m)
    loo = [(m.sum() - x) / (G - 1) for x in m]
    assert SC.pseudo(m.mean(), loo, G) == pytest.approx(m)


def test_hull():
    assert SC.hull((1, 2), (0, 1.5), None, (None, None)) == (0, 2)
    assert SC.hull(None) == (None, None)


# ------------------------------------------------------------------------------------------------ grammar
def test_grammar_is_complete_and_unique():
    ks = KS.expected_keys()
    names = [k for k, _ in ks]
    assert len(names) == len(set(names))
    assert len(KS.DUAL_IDS) == 110 and set(KS.DUAL_IDS) == set(K.DUAL_IDS)
    assert len(KS.LOOP_IDS) == 20
    for i in KS.SMALL_IDS:
        for t in KS.CI_TYPES:
            assert "small.%s.%s_ci" % (i, t) in names
        for t in KS.P_TYPES:
            assert "small.%s.p_%s" % (i, t) in names
    assert {t for _, t in ks} <= {"pct", "pp", "x2", "x3s", "int", "cny4", "p", "p4", "str"}
    assert all(n.startswith(("small.", "deff.", "margin.", "cost.nodisc.", "gate.")) for n in names)


# ------------------------------------------------------------------------------------------------ published file
def _doc():
    p = os.path.join(RESULTS, "rv2s_summary.json")
    if not os.path.exists(p):
        pytest.skip("results/posthoc2/rv2s_summary.json absent")
    return json.load(open(p))


def test_published_file_matches_grammar_and_has_no_api_calls():
    d = _doc()
    assert d["_meta"]["scope"] == "posthoc-secondary" and d["_meta"]["api_calls"] == 0 and d["_meta"]["B"] == 10000
    assert d["_meta"]["missing_keys"] == [] and d["_meta"]["recomputation_check"]["mismatches"] == 0
    keys = d["keys"]
    exp = dict(KS.expected_keys())
    assert set(keys) == set(exp)
    for k, e in keys.items():
        assert e["kind"] == exp[k] and e["v"] is not None, k


def test_published_small_cluster_readings_are_internally_consistent():
    keys = _doc()["keys"]
    rv2 = os.path.join(RESULTS, "rv2_summary.json")
    rv = json.load(open(rv2))["keys"] if os.path.exists(rv2) else None
    for i in KS.SMALL_IDS:
        g = lambda f: keys["small.%s.%s" % (i, f)]["v"]
        cl, it, wide, old, wideu = g("cl_ci"), g("item_ci"), g("wide_ci"), g("old_ci"), g("wideu_ci")
        assert old == [min(cl[0], it[0]), max(cl[1], it[1])], i
        for t in ("cl", "item", "t", "sf"):
            iv = g(t + "_ci")
            assert wide[0] <= iv[0] + 1e-12 and wide[1] >= iv[1] - 1e-12, (i, t)
        for t in ("tdb", "sfu"):
            iv = g(t + "_ci")
            assert wideu[0] <= iv[0] + 1e-12 and wideu[1] >= iv[1] - 1e-12, (i, t)
        assert wideu[0] <= wide[0] + 1e-12 and wideu[1] >= wide[1] - 1e-12
        for t in ("t", "tdb", "sf", "sfu"):
            assert 0.0 <= g("p_" + t) <= 1.0
        assert g("g") in (11, 19)
        if rv is not None and i in KS.DUAL_IDS:
            assert g("class_cl") == rv["dual.%s.class_cluster" % i]["v"] and g("class_item") == rv["dual.%s.class_item" % i]["v"], i
            assert abs(g("est") - rv["dual.%s.est" % i]["v"]) < 1e-9
            assert g("cl_ci") == rv["dual.%s.ci" % i]["v"]


def test_sign_flip_p_never_below_the_enumeration_floor():
    keys = _doc()["keys"]
    for i in KS.SMALL_IDS:
        g = keys["small.%s.g" % i]["v"]
        assert keys["small.%s.p_sf" % i]["v"] >= 2.0 ** (1 - g) - 1e-12
        assert keys["small.%s.p_sfu" % i]["v"] >= 2.0 ** (1 - g) - 1e-12


def test_published_design_effect_audit_agrees_with_the_dual_rows():
    keys = _doc()["keys"]
    import csv
    p = os.path.join(RESULTS, "dual.csv")
    if not os.path.exists(p):
        pytest.skip("dual.csv absent")
    vals = [float(r["deff"]) for r in csv.DictReader(open(p)) if r["deff"]]
    assert keys["deff.all.n"]["v"] == len(vals) == 110
    assert keys["deff.all.min"]["v"] == pytest.approx(min(vals)) and keys["deff.all.max"]["v"] == pytest.approx(max(vals))
    assert keys["deff.all.n_lt1"]["v"] == sum(v < 1 for v in vals) and keys["deff.all.n_gt2"]["v"] == sum(v > 2 for v in vals)
    for tag in KS.SETUP4:
        assert keys["deff.setup4.%s.b2000" % tag]["v"] == pytest.approx(keys["deff.setup4.%s.stored" % tag]["v"], abs=0.006)


def test_published_cost_ratios_reproduce_rv2_point_values():
    keys = _doc()["keys"]
    rv2 = os.path.join(RESULTS, "rv2_summary.json")
    if not os.path.exists(rv2):
        pytest.skip("rv2_summary.json absent")
    rv = json.load(open(rv2))["keys"]
    # with the discount the new ratios equal the RV2 Pareto-claim ratios; without it they equal the ratio of the nodisc totals
    for cl, (b, cfg) in (("plus_zs_vs_plus_gated", ("plus", "gated")), ("plus_zs_vs_27b_gated", ("27b", "gated")), ("plus_zs_vs_flash_gated", ("flash", "gated"))):
        a = keys["cost.nodisc.ratio.%s.%s.vs_plus_zs.disc" % (b, cfg)]["v"]
        assert a == pytest.approx(rv["pareto.claim.%s.cost_ratio" % cl]["v"], rel=1e-9)
        n = keys["cost.nodisc.ratio.%s.%s.vs_plus_zs.nodisc" % (b, cfg)]["v"]
        assert n == pytest.approx(rv["cost.%s.%s.cny_q_nodisc" % (b, cfg)]["v"] / rv["cost.plus.zs.cny_q_nodisc"]["v"], rel=0.02)
    # gate counts agree with the failing-root counts of RV2
    for b in K.BK:
        assert keys["gate.bird.%s.gated.fired" % b]["v"] == rv["roots.%s.n_fail.gated" % b]["v"]
        assert keys["gate.hh.bird.%s.gated.in.b" % b]["v"] == rv["loop.%s.eff_in.gated.b" % b]["v"]
