"""Tests of code/tools/rv2_analyze.py and rv2_keys.py (SAP Amendment 10, section 23; scope posthoc-secondary). Synthetic inputs only: no LLM call, no worker machine data.
The last block checks the published results/posthoc2/rv2_summary.json when it is present (set RV2_RESULTS to another directory if needed).
Run on a worker machine (needs numpy, scipy, pyyaml): cd code && python -m pytest tests/test_rv2_analyze.py"""
import collections
import csv
import importlib.util
import json
import math
import os
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(HERE, ".."))
spec = importlib.util.spec_from_file_location("rv2_analyze", os.path.join(TOOLS, "rv2_analyze.py"))
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
K = A.K
S = A.S
RESULTS = os.environ.get("RV2_RESULTS", os.path.join(HERE, "..", "..", "results", "posthoc2"))


# ------------------------------------------------------------------------------------------------ grammar
def test_grammar_has_no_duplicates_and_known_kinds():
    ks = K.expected_keys()
    names = [k for k, _ in ks]
    assert len(names) == len(set(names))
    assert {t for _, t in ks} <= {"pct", "pctv2", "pp", "fpp", "x1", "x2", "x3", "x3s", "int", "cny2", "cny4", "tok", "p", "str", "pctv", "pp2"}
    assert len(names) == 2688


def test_every_dual_id_has_all_eighteen_fields():
    names = {k for k, _ in K.expected_keys()}
    fields = ("est", "n", "g", "ci", "item_ci", "t_ci", "p_mcnemar", "b", "c", "deff", "loo_min", "loo_max", "rule", "class_cluster", "class_item", "class_t",
              "reading_changes", "reading_changes_t")
    for i in ("P1p.pooled", "W1.flash", "P5p", "P3", "XB2", "PF4_noise_cost"):
        for f in fields:
            assert "dual.%s.%s" % (i, f) in names, (i, f)
    for f in fields:
        assert "dual.RES.S_same.C1_flash.corr.%s" % f in names


def test_put_rejects_unknown_and_duplicate_keys():
    A.OUT.clear()
    with pytest.raises(KeyError):
        A.put("loop.nonsense", 1)
    A.put("cost.price_date", "2026-10-26")
    with pytest.raises(KeyError):
        A.put("cost.price_date", "x")
    A.OUT.clear()


def test_no_network_or_model_module_is_imported():
    src = open(os.path.join(TOOLS, "rv2_analyze.py")).read()
    for bad in ("import requests", "import openai", "nl2sql.llm", "import httpx", "urllib.request", "import socket"):
        assert bad not in src, bad


# ------------------------------------------------------------------------------------------------ reading classes (SAP 23.3)
def test_class_boundaries():
    assert A.cls_ni(-1.49, 1.0) == "non-inferior" and A.cls_ni(-1.5, 1.0) == "not shown" and A.cls_ni(-1.6, 1.0) == "not shown"
    assert A.cls_ratio(0.5, 0.79) == "saving shown" and A.cls_ratio(0.5, 0.80) == "not shown"
    assert A.cls_ratio1(1.1, 1.2) == "above 1" and A.cls_ratio1(0.8, 0.9) == "below 1" and A.cls_ratio1(0.9, 1.1) == "contains 1"
    assert A.cls_sup(0.01, 2) == "positive" and A.cls_sup(-2, -0.01) == "negative" and A.cls_sup(0.0, 2) == "contains 0"
    assert A.cls_band(0.1, 2.0, 1.0) == "small improvement" and A.cls_band(0.1, 2.0, 1.6) == "improves"
    assert A.cls_band(-2.0, -0.1, -1.0) == "small decline" and A.cls_band(-3, -0.1, -1.6) == "harms"
    assert A.cls_band(-1.0, 1.0, 0.0) == "no difference larger than 1.5 pp" and A.cls_band(-2.0, 1.0, 0.0) == "inconclusive"


def test_rule_of_assigns_the_governing_rule():
    assert A.rule_of("P1p.flash")[0] == "NI(1.5)" and A.rule_of("P1p_ratio.plus")[0] == "RATIO(0.80)"
    assert A.rule_of("XB3_ratio")[0] == "RATIO(0.80)" and A.rule_of("XB3_27b")[0] == "NI(1.5)"
    assert A.rule_of("PF3b")[0] == "RATIO(1)" and A.rule_of("PF2_n0")[0] == "BAND(1.5)"
    assert A.rule_of("W1.flash")[0] == "SUP(0)" and A.rule_of("P5p")[0] == "SUP(0)" and A.rule_of("P3_aurc")[0] == "SUP(0)"
    assert A.rule_of("RES.S_same.C1_plus.orig")[0] == "BAND(1.5)" and A.rule_of("RES.S_same.C1_plus.shift")[0] == "SUP(0)"
    assert A.rule_of("RES.S_all.C3_auroc.corr")[0] == "SUP(0)" and A.rule_of("RES.S_same.C4.corr")[0] == "BAND(1.5)"


# ------------------------------------------------------------------------------------------------ statistics
def test_exact_mcnemar_matches_binomial_tail():
    from scipy import stats as sps
    for b, c in ((13, 15), (125, 88), (1, 14), (4, 4), (0, 0)):
        x, y, p = A.mc(np.array([1] * b + [0] * c + [1, 0]), np.array([0] * b + [1] * c + [1, 0]))
        assert (x, y) == (b, c)
        want = 1.0 if b + c == 0 else min(1.0, 2.0 * sps.binom.cdf(min(b, c), b + c, 0.5))
        assert abs(p - want) < 1e-9, (b, c, p, want)


def test_item_bootstrap_draw_equals_the_primary_machinery():
    v = (np.arange(37) % 3 == 0).astype(float)
    assert np.array_equal(A.item_means(v, 200), S.item_boot_means(v, 200))
    num, den = np.arange(1.0, 38.0), np.arange(2.0, 39.0)
    assert np.allclose(A.item_ratio(num, den, 200), S.item_boot_ratio(num, den, 200))


def test_jackknife_t_interval_equals_classical_t_interval_for_a_mean():
    rng = np.random.RandomState(3)
    x = rng.normal(size=9)
    g = len(x)
    loo = [float(np.delete(x, i).mean()) for i in range(g)]
    lo, hi = A.jack_ci(float(x.mean()), loo, g)
    from scipy import stats as sps
    se = x.std(ddof=1) / math.sqrt(g)
    t = sps.t.ppf(0.975, g - 1)
    assert abs(lo - (x.mean() - t * se)) < 1e-9 and abs(hi - (x.mean() + t * se)) < 1e-9
    assert A.jack_ci(0.0, [], 2) == [None, None]


def test_cluster_and_item_intervals_agree_when_clusters_are_exchangeable():
    rng = np.random.RandomState(5)
    dbs = np.array(["d%d" % (i % 11) for i in range(1100)])
    d = rng.binomial(1, 0.5, size=1100).astype(float) - rng.binomial(1, 0.5, size=1100)
    e = A.est_mean(d, dbs, 2000)
    assert e["g"] == 11 and e["n"] == 1100
    assert 0.5 < e["deff"] < 2.0
    assert e["ci"][0] < e["est"] < e["ci"][1] and e["item_ci"][0] < e["est"] < e["item_ci"][1]
    assert e["loo_min"] <= e["est"] <= e["loo_max"]


def test_reading_changes_is_cluster_class_unequal_item_class():
    A.OUT.clear()
    del A.DUAL_ROWS[:]
    e = {"est": 2.4, "ci": [-0.1, 4.8], "item_ci": [0.6, 4.2], "t_ci": [-0.5, 5.3], "deff": 1.77, "loo_min": 1.5, "loo_max": 3.2, "g": 11, "n": 1534}
    A.put_dual("W1.flash", e, (125, 88, 0.013))
    assert A.OUT["dual.W1.flash.class_cluster"]["v"] == "contains 0" and A.OUT["dual.W1.flash.class_item"]["v"] == "positive"
    assert A.OUT["dual.W1.flash.reading_changes"]["v"] == "yes" and A.OUT["dual.W1.flash.reading_changes_t"]["v"] == "no"
    assert A.OUT["dual.W1.flash.b"]["v"] == 125 and A.OUT["dual.W1.flash.p_mcnemar"]["v"] == 0.013
    A.OUT.clear()
    del A.DUAL_ROWS[:]


# ------------------------------------------------------------------------------------------------ cost model
class _FakeRun(object):
    def __init__(self, usage):
        self.usage = usage


PRICES = {"cache_hit_ratio": 0.2, "batch_ratio": 0.5, "models": {"m": {"in": 2.0, "out": 8.0}}}


def test_run_costs_discount_nodiscount_and_embedding_exclusion():
    usage = [{"kind": "chat", "model": "m", "pt": 1000, "ct": 100, "cached_tokens": 400},
             {"kind": "chat", "model": "m", "pt": 1000, "ct": 100, "cached_tokens": 0, "mode": "batch"},
             {"kind": "embed", "model": "e", "pt": 500}, {"kind": "rerank", "model": "r", "pt": None}]
    t = A.run_costs(_FakeRun(usage), PRICES)
    want_disc = ((600 * 2.0 + 400 * 2.0 * 0.2 + 100 * 8.0) + 0.5 * (1000 * 2.0 + 100 * 8.0)) / 1e6
    want_nod = ((1000 * 2.0 + 100 * 8.0) + 0.5 * (1000 * 2.0 + 100 * 8.0)) / 1e6
    assert abs(t["cny_disc"] - want_disc) < 1e-12 and abs(t["cny_nodisc"] - want_nod) < 1e-12
    assert t["calls"] == 2 and t["pt"] == 2000 and t["ct"] == 200 and t["cached"] == 400
    assert t["embed_req"] == 1 and t["rerank_req"] == 1 and t["emb_tok_missing"] == 1
    assert t["cny_disc"] < t["cny_nodisc"]


def test_cached_tokens_never_exceed_prompt_tokens():
    t = A.run_costs(_FakeRun([{"kind": "chat", "model": "m", "pt": 100, "ct": 0, "cached_tokens": 5000}]), PRICES)
    assert t["cached"] == 100 and abs(t["cny_disc"] - 100 * 2.0 * 0.2 / 1e6) < 1e-15


# ------------------------------------------------------------------------------------------------ published results
def _summary():
    p = os.path.join(RESULTS, "rv2_summary.json")
    if not os.path.exists(p):
        pytest.skip("results/posthoc2/rv2_summary.json absent")
    return json.load(open(p))


def test_published_summary_matches_the_grammar_exactly():
    d = _summary()
    m, ks = d["_meta"], d["keys"]
    assert m["scope"] == "posthoc-secondary" and m["api_calls"] == 0 and m["B"] == 10000 and not m["missing_keys"]
    assert set(ks) == {k for k, _ in K.expected_keys()}
    kinds = dict(K.expected_keys())
    for k, e in ks.items():
        assert e["kind"] == kinds[k], k
        assert e["v"] is not None, k


def test_published_dual_rows_are_internally_consistent():
    d = _summary()["keys"]
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "dual.csv"))))
    assert len(rows) >= 100
    for r in rows:
        pre = "dual.%s." % r["id"]
        assert (d[pre + "class_cluster"]["v"] != d[pre + "class_item"]["v"]) == (d[pre + "reading_changes"]["v"] == "yes"), r["id"]
        assert abs(float(r["est"]) - d[pre + "est"]["v"]) < 1e-9 and abs(float(r["ci_lo"]) - d[pre + "ci"]["v"][0]) < 1e-9
        assert d[pre + "ci"]["v"][0] <= d[pre + "ci"]["v"][1] and d[pre + "item_ci"]["v"][0] <= d[pre + "item_ci"]["v"][1]
        if d[pre + "b"]["v"] != "n/a":
            assert d[pre + "n"]["v"] >= d[pre + "b"]["v"] + d[pre + "c"]["v"]
            assert 0.0 <= d[pre + "p_mcnemar"]["v"] <= 1.0
        assert d[pre + "g"]["v"] in (11, 19), r["id"]


def test_published_loop_identities():
    d = _summary()["keys"]
    for b in ("plus", "flash", "27b"):
        for defn in ("in", "btw"):
            g, f = d["loop.%s.eff_%s.gated" % (b, defn)]["v"], d["loop.%s.eff_%s.fixed" % (b, defn)]["v"]
            sh = d["loop.%s.share_%s" % (b, defn)]["v"]
            assert abs(sh - g / f) < 1e-9, (b, defn)
            assert abs(d["loop.%s.margin_over_eff_%s" % (b, defn)]["v"] - 1.5 / f) < 1e-9
        ex = lambda a: d["loop.%s.ex.%s" % (b, a)]["v"]
        assert abs(d["loop.%s.stack" % b]["v"] - 100.0 * (ex("rootonly") - ex("zs"))) < 1e-6, b
    pooled = np.mean([d["loop.%s.eff_in.gated" % b]["v"] for b in ("plus", "flash", "27b")])
    assert abs(pooled - d["loop.pooled.eff_in.gated"]["v"]) < 1e-9


def test_published_cost_identities():
    d = _summary()["keys"]
    for b, cfgs in K.COST_CFG.items():
        for c in cfgs:
            p = "cost.%s.%s." % (b, c)
            assert d[p + "cny_q"]["v"] <= d[p + "cny_q_nodisc"]["v"] + 1e-12, p
            assert abs(d[p + "cny_1000"]["v"] - 1000 * d[p + "cny_q"]["v"]) < 1e-9, p
            assert d[p + "calls"]["v"] >= 1.0 - 1e-9 and d[p + "lat_p95"]["v"] >= d[p + "lat_p50"]["v"] - 1e-9
    assert abs(d["cost.plus.route_extra.cny_q"]["v"] - (d["cost.plus.route.cny_q"]["v"] - d["cost.plus.gated.cny_q"]["v"])) < 1e-9
    assert d["cost.price_date"]["v"] == "2026-10-26"


def test_published_helped_hurt_counts_add_up():
    d = _summary()["keys"]
    rows = {(r["set"], r["kind"]): int(r["n"]) for r in csv.DictReader(open(os.path.join(RESULTS, "revisions.csv")))}
    assert rows[("dev", "helped")] + rows[("dev", "hurt")] + rows[("dev", "neutral")] == rows[("dev", "n")]
    lab = sum(d["rev.label.%s.%s" % (l, k)]["v"] for l in K.LABELS_E10 for k in ("helped", "hurt", "neutral"))
    assert lab == rows[("dev", "n")]
    assert d["rev.net"]["v"] is not None and 0.0 <= d["rev.sign_p"]["v"] <= 1.0


def test_published_routing_coverage_is_the_same_for_both_scores():
    d = _summary()["keys"]
    for cv in (40, 60, 80):
        assert d["route.v3.cov%d.n_acc" % cv]["v"] == d["route.route_ca.cov%d.n_acc" % cv]["v"]
    assert d["route.check.completions"]["v"] is not None and d["route.check.requests"]["v"] is not None
