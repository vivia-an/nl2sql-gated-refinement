"""Tests of code/tools/ext1_analyze.py (SAP Amendment 8, section 21). Synthetic runs only: no LLM call, no worker machine data.
Run: cd code && python -m pytest tests/test_ext1_analyze.py   (needs numpy)"""
import importlib.util
import json
import os

import pytest

np = pytest.importorskip("numpy")

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ext1_analyze", os.path.join(HERE, "..", "tools", "ext1_analyze.py"))
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
S = E.S


# ------------------------------------------------------------------------------------------------ helpers
DBS = ["d%d" % i for i in range(8)]


def make_run(root, arm, bk, ex, calls, actions=None, n_sys_error=0, plain=None, sql_tag=None):
    d = os.path.join(root, "%s-%s-s0" % (arm, bk))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "preds.jsonl"), "w") as f:
        for i, v in enumerate(ex):
            qid = "spider_realistic:%s:%d" % (DBS[i % len(DBS)], i)
            r = {"qid": qid, "db_id": DBS[i % len(DBS)], "ex_ts": int(v), "ex": int(v if plain is None else plain[i]), "calls": calls, "exec_ok": True,
                 "actions": (actions[i] if actions else ["accept"]), "final_sql": "SELECT %s %d" % (sql_tag or arm, i), "prompt_tokens": 100, "completion_tokens": 10,
                 "latency_s": 1.0, "error_class": None}
            if i < n_sys_error:
                r.update({"error_class": "sys_error:X", "ex_ts": None})
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(d, "usage.jsonl"), "w") as f:
        for i in range(len(ex)):
            f.write(json.dumps({"model": "qwen3.7-plus-2026-05-26", "pt": 1000, "ct": 100, "cached_tokens": 0, "cached": bool(i % 2), "kind": "chat"}) + "\n")
        f.write(json.dumps({"model": "qwen3.7-text-embedding", "pt": 5000, "ct": 0, "kind": "embed", "cached": False}) + "\n")
    return d


@pytest.fixture(autouse=True)
def small_n(monkeypatch):
    monkeypatch.setattr(E, "N_EXPECTED", 120)


def planted_runs(tmp_path, gain_weak=0.15, gain_plus=0.0, seed=0):
    rng = np.random.RandomState(seed)
    n = 120
    root = str(tmp_path / "runs")
    base = (rng.rand(n) < 0.6).astype(int)
    for bk in E.BACKBONES:
        gain = gain_plus if bk == "plus" else gain_weak
        zs = base.copy()
        flip_up = (rng.rand(n) < gain) & (zs == 0)
        gated = zs.copy()
        gated[flip_up] = 1
        fixed = gated.copy()
        make_run(root, "zs", bk, zs, 1.0)
        make_run(root, "gated", bk, gated, 2.0, actions=[["revise", "accept"] if i < 6 else ["accept"] for i in range(n)])
        make_run(root, "fixed", bk, fixed, 3.2)
        make_run(root, "rootonly", bk, gated, 2.0)
    return root


# ------------------------------------------------------------------------------------------------ loading rules
def test_ex_ts_value_rules():
    assert E.ex_ts_value({"ex_ts": 1}) == 1 and E.ex_ts_value({"ex_ts": 0}) == 0
    assert E.ex_ts_value({"error_class": "api_fail"}) == 0                    # SAP 10: api_fail counts as wrong
    assert E.ex_ts_value({"error_class": "sys_error:score:X", "ex_ts": 1}) is None   # infrastructure failure is not a measurement
    assert E.ex_ts_value({"ex_ts": None}) is None
    assert E.ex_ts_value({}) is None


def test_incomplete_arm_is_flagged(tmp_path):
    root = str(tmp_path)
    make_run(root, "zs", "plus", [1] * 120, 1.0)
    make_run(root, "zs", "flash", [1] * 120, 1.0, n_sys_error=2)
    make_run(root, "zs", "27b", [1] * 100, 1.0)
    assert E.load_arm(root, "zs", "plus").complete
    a = E.load_arm(root, "zs", "flash")
    assert (not a.complete) and a.n_sys_error == 2 and "incomplete" in a.status()
    assert not E.load_arm(root, "zs", "27b").complete
    assert E.load_arm(root, "gated", "plus") is None


# ------------------------------------------------------------------------------------------------ statistics
def test_contrast_hand_calculation():
    a = np.array([1, 1, 1, 0, 0, 1, 0, 1], int)
    b = np.array([1, 0, 0, 0, 1, 1, 0, 0], int)
    dbs = np.array(["x", "x", "y", "y", "z", "z", "w", "w"])
    o = E.contrast(a, b, dbs, 2000)
    assert abs(o["delta_pp"] - 100 * (a - b).mean()) < 1e-9
    assert (o["b"], o["c"]) == (3, 1)
    assert abs(o["mcnemar_p_two"] - S.mcnemar_exact(3, 1)["p_two"]) < 1e-12
    assert o["ci_lo"] <= o["delta_pp"] <= o["ci_hi"] and o["n_dbs"] == 4
    # the same call gives the same interval (seeded)
    assert E.contrast(a, b, dbs, 2000)["ci_lo"] == o["ci_lo"]


def test_pooled_is_mean_of_backbone_differences():
    rng = np.random.RandomState(3)
    dbs = np.array([DBS[i % 8] for i in range(120)])
    ea = {"flash": (rng.rand(120) < .7).astype(int), "27b": (rng.rand(120) < .7).astype(int)}
    eb = {"flash": (rng.rand(120) < .6).astype(int), "27b": (rng.rand(120) < .6).astype(int)}
    o = E.pooled(ea, eb, dbs, 1000)
    per = [100 * (ea[k] - eb[k]).mean() for k in ea]
    assert abs(o["delta_pp"] - np.mean(per)) < 1e-9
    assert abs(o["per_backbone"]["flash"] - per[0]) < 1e-9
    assert o["ci_lo"] < o["delta_pp"] < o["ci_hi"]


def test_xb3_block_ratio_and_bounds():
    rng = np.random.RandomState(5)
    dbs = np.array([DBS[i % 8] for i in range(120)])
    ex = {k: (rng.rand(120) < .7).astype(int) for k in E.BACKBONES}
    cg = {k: np.full(120, 2.0) for k in E.BACKBONES}
    cf = {k: np.full(120, 3.2) for k in E.BACKBONES}
    o = E.xb3_block(ex, ex, cg, cf, dbs, 1000)
    assert abs(o["ratio"]["mean"] - 2.0 / 3.2) < 1e-12 and o["ratio"]["ci_hi"] < E.RATIO_NULL      # 0.625 < 0.80, zero variance
    assert o["delta_pp"] == 0.0 and o["p_a_1s"] < 0.01 and o["ratio"]["p_b_1s"] < 0.01 and o["p_1s"] < 0.01 and o["cl_ok"]
    o2 = E.xb3_block(ex, ex, {k: np.full(120, 3.0) for k in E.BACKBONES}, cf, dbs, 1000)
    assert o2["ratio"]["mean"] > 0.9 and o2["p_1s"] > 0.5 and not o2["cl_ok"]                       # no call saving: the IUT p is the large one


# ------------------------------------------------------------------------------------------------ reading rules (SAP 21.5)
def test_read_xb1_all_lines():
    pb = {"flash": 4.0, "27b": 3.0}
    r = E.read_xb1(0.01, 0.01, 1.2, 5.0, 3.5, pb)
    assert r["verdict"] == "confirmed" and r["replicates_b"] and r["size_consistent_with_bird"] is True
    r = E.read_xb1(0.01, 0.01, 1.2, 5.0, 3.5, {"flash": 7.0, "27b": -0.5})
    assert r["verdict"] == "supported on average, not on each backbone" and not r["replicates_b"]
    assert E.read_xb1(0.2, 0.03, -0.5, 6.0, 3.0, pb)["verdict"] == "supported at item level, not at database level"
    assert E.read_xb1(0.2, 0.2, -0.5, 6.0, 3.0, pb)["verdict"] == "direction consistent, not confirmed"
    assert E.read_xb1(0.8, 0.8, -3.0, 2.0, -0.4, {"flash": -1.0, "27b": 0.2})["verdict"] == "not confirmed"
    c = E.read_xb1(0.99, 0.99, -5.0, -0.5, -2.7, {"flash": -3.0, "27b": -2.4})
    assert c["contradicted"] and "contradicted" in c["text"]
    assert E.read_xb1(0.01, 0.01, 1.2, 2.0, 1.5, pb)["size_consistent_with_bird"] is False        # CI [1.2, 2.0] does not contain +2.93


def test_read_xb2_lines():
    assert E.read_xb2(0.4, 5.0, 2.7)["line"] == "gain"
    assert E.read_xb2(-6.0, -0.2, -3.0)["line"] == "loss"
    r = E.read_xb2(-2.0, 2.0, 0.0)
    assert r["line"] == "no_detectable_gain" and r["equivalent_3pp"] and r["non_inferior_3pp"]
    r = E.read_xb2(-2.0, 4.0, 1.0)
    assert r["line"] == "no_detectable_gain" and r["non_inferior_3pp"] and not r["equivalent_3pp"] and "equivalence not shown" in r["text"]
    r = E.read_xb2(-4.5, 3.0, -0.8)
    assert not r["non_inferior_3pp"] and "does not exclude" in r["text"]


def test_read_xb3_words():
    r = E.read_xb3(0.01, 0.01, -0.4, 0.66, True, 0.5, 0.62)
    assert r["verdict"] == "confirmed" and r["replicates_a"]
    assert E.read_xb3(0.2, 0.2, -2.5, 0.66, False, -0.8, 0.62)["verdict"] == "direction consistent, not confirmed"
    r = E.read_xb3(0.9, 0.9, -4.0, 0.66, False, -2.6, 0.62)
    assert r["verdict"] == "not confirmed" and "contradicted" in r["text"]
    assert "no call saving" in E.read_xb3(0.9, 0.9, -1.0, 0.99, False, 0.0, 0.95)["text"]
    assert E.read_xb3(0.2, 0.2, -2.0, 0.7, False, -0.5, 0.65)["text"] == "fewer calls, accuracy parity not shown"


def test_holm_family_has_two_members():
    h = S.holm({"XB1": 0.012, "XB3": 0.03}, m=2)
    assert abs(h["XB1"] - 0.024) < 1e-12 and abs(h["XB3"] - 0.03) < 1e-12        # (m - i) * p, then running max
    h = S.holm({"XB1": None, "XB3": 0.03}, m=2)                                  # a test that was not run counts with p = 1 in the family size
    assert h["XB1"] is None and abs(h["XB3"] - 0.06) < 1e-12


# ------------------------------------------------------------------------------------------------ cost and stop rule
def test_run_cost_cache_conventions(tmp_path):
    d = make_run(str(tmp_path), "zs", "plus", [1] * 10, 1.0)
    P = E.load_prices(os.path.join(HERE, "..", "..", "exp", "prices.yaml"))
    full = E.run_cost(d, P, cache_free=False)
    paid = E.run_cost(d, P, cache_free=True)
    one = (1000 * 2.0 + 100 * 8.0) / 1e6
    assert abs(full["chat"] - 10 * one) < 1e-12                      # every chat row at list price, hit or not
    assert abs(paid["chat"] - 5 * one) < 1e-12                       # half of the rows are cache hits: free in the spend figure
    assert abs(full["embed"] - 5000 * 0.5 / 1e6) < 1e-12
    assert full["unpriced"] == 0


def test_stop_rule_thresholds():
    assert E.stop_rule(10.0, 5.0) == 0
    assert E.stop_rule(100.5, 0.0) == 2
    assert E.stop_rule(95.0, 56.0) == 3                              # next arm would pass the hard cap of 150
    assert E.stop_rule(149.0, 1.0) == 2 and E.stop_rule(149.5, 1.0) == 3        # 150.0 exactly is allowed (cap is 'never exceeded'); above the soft stop it is code 2
    assert E.stop_rule(99.0, 1.0) == 0


# ------------------------------------------------------------------------------------------------ cases and end to end
def test_case_rule_is_sorted_and_limited(tmp_path):
    root = str(tmp_path)
    a = np.array([1, 0] * 60, int)
    b = np.array([0, 1] * 60, int)
    for arm, ex in (("gated", a), ("zs", b)):
        make_run(root, arm, "flash", ex, 1.0)
    arms = {("flash", arm): E.load_arm(root, arm, "flash") for arm in ("gated", "zs")}
    al = E.align({k[1]: v for k, v in arms.items()})
    out = E.case_rows(al, "gated", "zs", {})
    g = out["gated_right_zs_wrong"]
    assert g["n_discordant"] == 60 and len(g["cases"]) == 3
    qs = [c["qid"] for c in g["cases"]]
    assert qs == sorted(qs) and qs == sorted(q for q in al["qids"] if q in {c["qid"] for c in g["cases"]})
    expected = sorted(q for q in al["qids"] if int(q.split(":")[-1]) % 2 == 0)[:3]    # gated right on even item indices
    assert qs == expected and qs != ["spider_realistic:d0:0", "spider_realistic:d2:2", "spider_realistic:d4:4"]     # ascending qid STRING order, not index order
    assert out["zs_right_gated_wrong"]["n_discordant"] == 60


def test_end_to_end_planted_effects(tmp_path):
    root = planted_runs(tmp_path, gain_weak=0.20, gain_plus=0.0)
    out = str(tmp_path / "out")
    rc = E.main(["analyze", "--runs", root, "--out", out, "--B", "800", "--prices", os.path.join(HERE, "..", "..", "exp", "prices.yaml"), "--data-file", ""])
    assert rc == 0
    R = json.load(open(os.path.join(out, "ext1_summary.json")))
    assert R["_meta"]["scope"] == "external-validity" and R["_meta"]["split"] == "spider_realistic"
    assert R["EXT1.arms_incomplete_or_missing"] == []
    assert R["EXT1.XB1.delta_ex"] > 5 and R["EXT1.XB1.verdict"] in ("confirmed", "supported at item level, not at database level", "supported on average, not on each backbone")
    assert R["EXT1.XB2.line"] == "no_detectable_gain" and abs(R["EXT1.XB2.delta_ex"]) < 1e-9
    assert abs(R["EXT1.XB3.calls_ratio"] - 2.0 / 3.2) < 1e-9 and R["EXT1.XB3.delta_ex"] == 0.0
    assert R["EXT1.XB3.p_adj"] >= R["EXT1.XB3.p_2s"] - 1e-12 and R["EXT1.XB1.p_adj"] >= R["EXT1.XB1.p_2s"] - 1e-12
    assert R["EXT1.arm.plus.zs.calls_per_q"] == 1.0 and abs(R["EXT1.gate.plus.gate_fired_share"] - 6 / 120.0) < 1e-12
    assert R["EXT1.sens_no_world_1.XB1.verdict"] == R["EXT1.XB1.verdict"]          # no world_1 in the synthetic data: the sensitivity run must reproduce the headline
    for f in ("ext1_arms.csv", "ext1_contrasts.csv", "ext1_cases.json"):
        assert os.path.exists(os.path.join(out, f))
    # nothing is a \res macro key of the confirmatory papers: the keys carry the EXT1. prefix only
    assert all(k.startswith("EXT1.") or k == "_meta" for k in R)


def test_missing_arms_are_reported_not_invented(tmp_path):
    root = str(tmp_path / "runs")
    n = 120
    for bk in E.BACKBONES:
        make_run(root, "zs", bk, [1] * n, 1.0)
    make_run(root, "gated", "plus", [1] * n, 2.0)                    # no fixed, no flash/27b gated
    out = str(tmp_path / "out")
    E.main(["analyze", "--runs", root, "--out", out, "--B", "300", "--prices", os.path.join(HERE, "..", "..", "exp", "prices.yaml"), "--data-file", ""])
    R = json.load(open(os.path.join(out, "ext1_summary.json")))
    assert R["EXT1.XB1.status"].startswith("not run") and R["EXT1.XB3.status"].startswith("not run") and R["EXT1.XB2.status"] == "ok"
    assert "EXT1.XB1.delta_ex" not in R and "EXT1.XB3.delta_ex" not in R and "EXT1.XB1.verdict" not in R
    assert "plus/gated" in R["EXT1.arms_complete"] and "flash/gated" in R["EXT1.arms_incomplete_or_missing"]
