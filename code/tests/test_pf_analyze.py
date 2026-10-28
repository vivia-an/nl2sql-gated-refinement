"""Offline tests of tools/pf_analyze.py (SAP Amendment 9, section 22): reading rules, paired clustered contrasts, helped / hurt, flag rates, missing arms."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import pf_analyze as A  # noqa: E402

DBS = ["d1", "d2", "d3", "d4"]


def mk_run(tmp_path, name, ex, root_ex=None, root_score=None, calls=None, iters=None, pf=None):
    d = tmp_path / name
    d.mkdir()
    n = len(ex)
    rows, cands = [], []
    for i in range(n):
        q = "bird_dev:%s:%d" % (DBS[i % 4], i)
        r = {"qid": q, "db_id": DBS[i % 4], "ex": int(ex[i]), "calls": (calls[i] if calls is not None else 3), "prompt_tokens": 1000, "completion_tokens": 100,
             "cached_tokens": 0, "iterations": int(iters[i] if iters is not None else 0), "actions": [], "final_sql": "SELECT %d" % i, "error_class": None}
        if pf:
            r["pf"] = pf[i]
        rows.append(r)
        if root_ex is not None:
            cands.append({"qid": q, "cand_idx": 0, "action": "root", "sql": "SELECT r%d" % i, "ex": int(root_ex[i]), "score": float(root_score[i]), "semantic": 9.0})
    (d / "preds.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if cands:
        (d / "cands.jsonl").write_text("\n".join(json.dumps(c) for c in cands) + "\n")
    return A.load_run(str(d))


def test_reading_rule_covers_all_outcomes():
    assert A.reading(2.0, 0.5, 3.5) == "improves (exploratory)"
    assert A.reading(1.0, 0.2, 1.8).startswith("small improvement")
    assert A.reading(0.2, -0.9, 1.1) == "no difference larger than 1.5 pp"
    assert A.reading(-2.0, -3.5, -0.5) == "harms (exploratory)"
    assert A.reading(-0.8, -1.2, -0.1).startswith("small decline")
    assert A.reading(0.5, -2.0, 3.0).startswith("inconclusive")
    assert A.reading(0.0, -1.5, 0.5).startswith("inconclusive")            # a bound at exactly the margin is not inside the open band


def test_pf3_and_curve_verdicts():
    assert A.pf3_verdict("improves (exploratory)", 1.1, 1.3) == "buys accuracy with calls"
    assert A.pf3_verdict("improves (exploratory)", 0.8, 0.95).startswith("dominates")
    assert "stays preferred" in A.pf3_verdict("no difference larger than 1.5 pp", 1.2, 1.4)
    assert A.pf3_verdict("harms (exploratory)", 0.9, 1.1).startswith("no accuracy gain; calls ratio")
    assert A.curve_verdict("improves (exploratory)", [60, 61, 62]).startswith("supported")
    assert "not monotone" in A.curve_verdict("improves (exploratory)", [60, 62, 61])
    assert A.curve_verdict("no difference larger than 1.5 pp", [60, 61, 62]).startswith("not shown")
    assert A.curve_verdict("harms (exploratory)", [60, 59, 58]).startswith("not shown")


def test_contrast_matches_manual_and_pairing(tmp_path):
    n = 80
    rng = np.random.RandomState(0)
    base = rng.randint(0, 2, n)
    better = base.copy()
    better[:8] = 1                                    # at most 8 flips up
    runs = {"full_fixed": mk_run(tmp_path, "f", base), "pf_n0": mk_run(tmp_path, "a", base), "pf_n1000_q100": mk_run(tmp_path, "b", better)}
    res = A.analyze(runs, B=300)
    c = {x["name"]: x for x in res["contrasts"]}
    d = (better - base).mean() * 100
    assert abs(c["PF1"]["d_pp"] - d) < 1e-9
    assert c["PF1"]["a_only"] == int(((better == 1) & (base == 0)).sum()) and c["PF1"]["b_only"] == 0
    assert c["PF1"]["lo_pp"] <= d <= c["PF1"]["hi_pp"]
    assert c["PF2"]["status"] == "not run" and c["PF3a"]["status"] == "not run"          # arms that were not run
    assert c["PF2_n0"]["d_pp"] == 0.0 and c["PF2_n0"]["reading"] == "no difference larger than 1.5 pp"
    assert res["_meta"]["arms_missing"] == ["pf_n250_q100", "pf_n1000_q085"]
    assert res["PF4"]["complete"] is False and res["PF4"]["points_present"] == [0, 1000]


def test_helped_hurt_flag_rates_and_identity(tmp_path):
    ex = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    root_ex = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    score = np.array([6.0, 9.5, 9.0, 7.0, 9.9, 9.9, 8.0, 8.4])           # flagged: items 0, 3, 6, 7
    iters = np.array([1, 0, 0, 1, 0, 0, 2, 1])
    pf = [{"n": 0, "q": 1.0, "parse_fail": int(i == 6), "cases": [{"id": "x", "verdict": i % 2, "oracle": 1}]} for i in range(8)]
    runs = {"full_fixed": mk_run(tmp_path, "f", ex, root_ex, score, iters=iters), "pf_n0": mk_run(tmp_path, "a", ex, root_ex, score, iters=iters, pf=pf)}
    res = A.analyze(runs, B=100)
    a = {x["label"]: x for x in res["arms"]}["pf_n0"]
    # revisions: items 0 (0 -> 1 helped), 3 (1 -> 0 hurt), 6 (0 -> 0), 7 (0 -> 0)
    assert a["items_with_revision"] == 4 and a["helped"] == 1 and a["hurt"] == 1 and a["net_helped_minus_hurt_pp"] == 0.0
    assert a["critic_flag_rate_pp"] == 50.0
    assert a["critic_detection_pp"] == 100 * 3 / 4 and a["critic_false_flag_pp"] == 100 * 1 / 4          # wrong roots 0,1,6,7: flagged 0,6,7 ; right roots 2,3,4,5: flagged 3
    assert a["parse_fail_items"] == 1 and a["root_sql_identical_to_full_fixed_pp"] == 100.0
    assert a["share_items_with_rejected_case_pp"] == 50.0


def test_cost_and_calls_ratio(tmp_path):
    ex = np.ones(40, int)
    runs = {"full_gated": mk_run(tmp_path, "g", ex, calls=[2] * 40), "pf_n1000_q085": mk_run(tmp_path, "p", ex, calls=[4] * 40)}
    res = A.analyze(runs, B=100)
    p = res["PF3_cost"]["PF3"]
    assert abs(p["calls_ratio"] - 2.0) < 1e-9 and abs(p["lo"] - 2.0) < 1e-9 and abs(p["hi"] - 2.0) < 1e-9
    assert p["verdict"].startswith("no accuracy gain at higher cost")
    a = res["arms"][0]
    assert abs(a["cost_cny_per_q"] - (1000 * 2.0 + 100 * 8.0) / 1e6) < 1e-12


def test_outputs_written_and_flat_keys(tmp_path):
    ex = np.array([1, 0, 1, 1] * 10)
    runs = {"pf_n0": mk_run(tmp_path, "a", ex), "pf_n1000_q100": mk_run(tmp_path, "b", ex)}
    res = A.analyze(runs, B=50)
    out = tmp_path / "out"
    A.write_outputs(res, str(out))
    s = json.load(open(str(out / "pf_summary.json")))
    assert s["_meta"]["scope"] == "exploratory-posthoc" and "PF1.d_pp" in s and "arm.pf_n0.ex_pp" in s and "curve.n1000.ex_pp" in s
    for f in ("pf_arms.csv", "pf_contrasts.csv", "pf_curve.csv", "pf_cases.json"):
        assert (out / f).exists()


def test_price_constants_match_prices_yaml():
    p = os.path.join(os.path.dirname(__file__), "..", "..", "exp", "prices.yaml")
    assert A.price_check(p)
