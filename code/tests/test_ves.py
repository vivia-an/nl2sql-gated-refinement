import json
import math
import os
import sqlite3

import pytest

from nl2sql.eval import ves, ves_score
from nl2sql.eval.bird_official import rves_reward


@pytest.fixture()
def dbp(tmp_path):
    p = str(tmp_path / "t.sqlite")
    c = sqlite3.connect(p)
    c.execute("create table a (id integer primary key, name text, v real)")
    c.executemany("insert into a values (?,?,?)", [(i, "n%d" % (i % 5), i * 1.0) for i in range(1, 3001)])
    c.commit()
    c.close()
    return p


def test_item_values_follow_official_formulas():
    assert ves.item_ves(1.0) == 100.0
    assert abs(ves.item_ves(4.0) - 200.0) < 1e-9 and ves.item_ves(0.0) == 0.0
    # Mini-Dev reward bands
    assert [rves_reward(x) for x in (0, 3, 2, 1.5, 1, 0.75, 0.5, 0.3, 0.25, 0.1)] == [0, 1.25, 1.25, 1, 1, 0.75, 0.75, 0.5, 0.5, 0.25]
    assert abs(ves.item_rves(2.5) - math.sqrt(1.25) * 100) < 1e-9 and ves.item_rves(0.0) == 0.0


def test_trimmed_mean_rules():
    # official 3-sigma rule can only trim when n >= 11 (|z| <= sqrt(n-1)): with 5 repetitions it is the plain mean
    v, fb = ves.trimmed_mean([1.0, 1.0, 1.0, 1.0, 50.0])
    assert abs(v - 10.8) < 1e-9 and not fb
    xs = [1.0] * 19 + [100.0]
    v, fb = ves.trimmed_mean(xs)
    assert abs(v - 1.0) < 1e-9 and not fb                    # 20 samples: the outlier is removed
    v, fb = ves.trimmed_mean([2.0, 2.0, 2.0])                # std = 0 -> official code would divide by zero; we fall back
    assert v == 2.0 and fb
    v, fb = ves.trimmed_mean([1.0, 2.0, 3.0, 4.0, 100.0], trim="minmax")
    assert abs(v - 3.0) < 1e-9


def test_measure_ratio_same_query_is_about_one(dbp):
    for timing in ves.TIMING_MODES:
        m = ves.measure_ratio(dbp, "select name, sum(v) from a group by name", "select name, sum(v) from a group by name", 5, timing)
        assert m["fail"] is None and len(m["pred_times"]) == 5 and len(m["gold_times"]) == 5
        assert 0.1 < m["time_ratio"] < 10


def test_measure_ratio_failure_and_gold_reuse(dbp):
    m = ves.measure_ratio(dbp, "select nonsense from a", "select 1", 3)
    assert m["time_ratio"] == 0.0 and "OperationalError" in m["fail"]
    m = ves.measure_ratio(dbp, "select count(*) from a x, a y, a z", "select 1", 2, timeout_s=0.2)
    assert m["time_ratio"] == 0.0 and m["fail"]               # timeout -> 0, never raises
    # reused gold timings: ratio_i = gold_i / pred_i with the supplied gold list
    m = ves.measure_ratio(dbp, "select count(*) from a", "select 1", 4, "official", gold_times=[1.0, 1.0, 1.0, 1.0])
    assert m["gold_times"] == [1.0] * 4 and m["time_ratio"] > 100        # pred takes << 1 s, so the ratio is huge


def test_ves_score_cli_and_summary(tmp_path, dbp, monkeypatch):
    from nl2sql.data import Item
    items = [Item("q%d" % i, "d%d" % (i % 2), "?", "", "", dbp, "bird_train_HT", {}) for i in range(4)]
    gold = {"q%d" % i: "select name, sum(v) from a group by name" for i in range(4)}
    monkeypatch.setattr(ves_score.data, "load_split", lambda name, with_gold=None: items)
    monkeypatch.setattr(ves_score.data, "load_gold", lambda split, run_id="": gold)
    preds = tmp_path / "preds.jsonl"
    rows = [{"qid": "q%d" % i, "db_id": "d%d" % (i % 2), "ex": 1 if i < 3 else 0, "final_sql": gold["q%d" % i]} for i in range(4)]
    preds.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ves_score.main(["--preds", str(preds), "--split", "bird_train_HT", "--iters", "3"])
    out = ves_score.load_rows(str(tmp_path / "ves.jsonl"))
    assert len(out) == 4 and out[3]["ves"] == 0.0 and out[3]["rves"] == 0.0           # EX=0 item contributes 0
    assert all(r["ves"] > 0 and r["rves"] > 0 for r in out[:3])
    s = json.load(open(str(tmp_path / "ves.summary.json")))
    assert s["n"] == 4 and s["n_ex1"] == 3 and s["ves"] > 0 and s["coverage"] == 1.0
    ves_score.main(["--preds", str(preds), "--split", "bird_train_HT", "--iters", "3"])         # resumable: nothing new is appended
    assert len(ves_score.load_rows(str(tmp_path / "ves.jsonl"))) == 4
    # summarize_run picks the file up
    from nl2sql.analysis.summarize import summarize_run
    S = summarize_run(str(preds))
    assert S["ves"] is not None and S["rves"] is not None and S["ves_coverage"] == 1.0
