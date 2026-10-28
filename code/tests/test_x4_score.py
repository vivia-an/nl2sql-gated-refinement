"""X4: scoring robustness - exceptions in score_sql, oversize gold results, deterministic gold/pred timeouts. Offline sqlite only."""
import sqlite3

import pytest

from nl2sql import db, paths, run, sqlcache
from nl2sql.data import Item
from nl2sql.eval import score as S
from nl2sql.eval.score import Evaluator, compact_gold, rows_digest

SLOW = "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM r WHERE x < %d) SELECT count(*) FROM r"


@pytest.fixture()
def dbp(tmp_path):
    p = str(tmp_path / "t.sqlite")
    c = sqlite3.connect(p)
    c.execute("create table a (id integer primary key, name text, v real)")
    c.executemany("insert into a values (?,?,?)", [(i, "n%d" % (i % 5), i * 1.0) for i in range(1, 501)])
    c.commit()
    c.close()
    return p


@pytest.fixture(autouse=True)
def fresh_gold_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CACHE", tmp_path / "gcache")
    monkeypatch.setattr(S, "_GC", None)
    yield
    S._GC = None


def make_eval(item, gold, timeout_s=30, gold_timeout_s=120):
    ev = Evaluator.__new__(Evaluator)
    ev.split, ev.timeout_s, ev.gold_timeout_s = "toy", timeout_s, max(timeout_s, gold_timeout_s)
    ev.gold, ev.gold2, ev.items, ev.use_ts = {item.qid: gold}, {}, {}, False
    return ev


def item(dbp, qid="toy:t:1"):
    return Item(qid=qid, db_id="t", question="q", evidence="", gold_sql="", db_path=dbp, split="toy", meta={})


# ---------------------------------------------------------------------------------------------- score_sql exceptions
class FakeEv:
    gold = {"toy:t:1": "select 1"}

    def __init__(self, bad):
        self.bad = bad

    def score_sql(self, item, sql, pred_res=None):
        if item.qid in self.bad:
            raise sqlite3.DataError("string or blob too big")
        return {"ex": 1, "exec_ok": True, "soft_f1": 1.0, "jaccard": 1.0, "overlap": 1.0, "auto_class": None}


class FakeSys:
    def run(self, item, seed=0):
        return {"final_sql": "select 1"}


def test_score_exception_becomes_sys_error_row_not_a_lost_loop(dbp):
    it = item(dbp, "toy:t:1")
    row, cands = run.process_item(FakeSys(), FakeEv({"toy:t:1"}), it, 0, {}, {"system": "x", "model": "m"})
    assert row["error_class"] == "sys_error:score:DataError" and row["ex"] is None and row["exec_ok"] is False
    assert "string or blob too big" in row["err_msg"]
    row2, _ = run.process_item(FakeSys(), FakeEv(set()), it, 0, {}, {"system": "x", "model": "m"})
    assert row2["ex"] == 1 and row2["error_class"] is None


def test_candidate_scoring_exception_is_contained(dbp):
    class Sys2:
        def run(self, item, seed=0):
            return {"final_sql": "select 1", "candidates": [{"sql": "select 2"}]}

    class Ev(FakeEv):
        def score_sql(self, item, sql, pred_res=None):
            if sql == "select 2":
                raise RuntimeError("boom")
            return super().score_sql(item, sql)

    row, cands = run.process_item(Sys2(), Ev(set()), item(dbp), 0, {}, {"system": "x", "model": "m"})
    assert row["ex"] == 1 and cands[0]["ex"] is None


def test_summary_counts_sys_error(tmp_path):
    import json
    from nl2sql.analysis.summarize import summarize_run
    rows = [{"qid": "a", "db_id": "d", "ex": 1, "error_class": None}, {"qid": "b", "db_id": "d", "ex": None, "error_class": "sys_error:score:DataError"},
            {"qid": "c", "db_id": "d", "ex": 0, "error_class": "api_fail"}, {"qid": "d", "db_id": "d", "ex": 0, "auto_class": "gold_fail"}]
    p = tmp_path / "preds.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    s = summarize_run(str(p))
    assert s["sys_error"] == 1 and s["api_fail"] == 1 and s["gold_fail"] == 1 and s["n_scored"] == 3


# ---------------------------------------------------------------------------------------------- oversize gold
def test_rows_digest_is_set_semantics():
    assert rows_digest([(1, "a"), (2, "b")]) == rows_digest([(2, "b"), (1, "a"), (1, "a")])
    assert rows_digest([(1.0,)]) == rows_digest([(1,)])
    assert rows_digest([(1, "a")]) != rows_digest([(1, "b")])
    assert rows_digest([(1,), (2,)]) != rows_digest([(1,)])


def test_compact_gold_threshold():
    small = {"ok": True, "rows": [(i,) for i in range(10)], "cols": ["x"], "error": None, "truncated": False}
    assert compact_gold(small, max_bytes=10000) is small
    stub = compact_gold(small, max_bytes=50)
    assert stub["big"] and stub["rows"] is None and stub["n_rows"] == 10 and stub["rows_digest"] == rows_digest(small["rows"])
    fail = {"ok": False, "rows": [], "error": "x"}
    assert compact_gold(fail, max_bytes=1) is fail


def test_oversize_gold_scores_via_digest_and_is_cached(dbp, monkeypatch):
    monkeypatch.setattr(S, "GOLD_MAX_BYTES", 200)                       # every non-trivial gold result is "oversize" here
    it = item(dbp)
    ev = make_eval(it, "select id from a")                              # 500 rows
    good = ev.score_sql(it, "select id from a order by id desc")
    assert good["ex"] == 1 and good["gold_fail"] is False and good["soft_f1"] is None and good["jaccard"] is None
    bad = ev.score_sql(it, "select id from a where id < 400")
    assert bad["ex"] == 0 and bad["auto_class"] == "gold_big"
    err = ev.score_sql(it, "select nope from a")
    assert err["ex"] == 0 and err["auto_class"] != "gold_big"
    # the stub is what the cache holds (no pickle of the rows, no exception), and a second evaluator reads it
    key = S.hashlib.sha256((it.db_path + "\x1f" + "select id from a").encode()).hexdigest()
    cached = S.gold_cache().get(key)
    assert cached["big"] and cached["n_rows"] == 500
    assert make_eval(it, "select id from a").score_sql(it, "select id from a")["ex"] == 1


def test_gold_cache_put_refuses_oversize_instead_of_raising(monkeypatch):
    monkeypatch.setattr(S, "GOLD_MAX_BYTES", 100)
    assert S.gold_cache().put("k", {"rows": [(i,) for i in range(1000)]}) is False
    assert S.gold_cache().get("k") is None


def test_gold_cache_unreadable_pickle_is_a_miss():
    gc = S.gold_cache()
    gc._db.execute("INSERT OR REPLACE INTO g VALUES (?,?)", ("bad", b"\x80not a pickle"))
    assert gc.get("bad") is None
    assert gc._db.query_one("SELECT v FROM g WHERE key=?", ("bad",)) is None      # dropped


# ---------------------------------------------------------------------------------------------- deterministic timeouts
def test_gold_gets_generous_timeout_pred_keeps_short(dbp):
    it = item(dbp)
    gold = SLOW % 3000000
    ev = make_eval(it, gold, timeout_s=0.05, gold_timeout_s=60)
    r = ev.score_sql(it, gold)                                         # same SQL as prediction: times out at 0.05 s ...
    assert r["gold_fail"] is False                                     # ... but the gold ran with 60 s and succeeded
    assert r["exec_ok"] is False and r["ex"] == 0 and r["auto_class"] == "timeout"
    assert ev.gold_exec(it)["ok"]


def test_gold_failure_beyond_limit_is_gold_fail_and_cached(dbp):
    it = item(dbp)
    gold = SLOW % 100000000
    ev = make_eval(it, gold, timeout_s=0.2, gold_timeout_s=0.5)
    r = ev.score_sql(it, "select 1")
    assert r["gold_fail"] is True and r["ex"] == 0 and r["auto_class"] == "gold_fail"
    key = S.hashlib.sha256((it.db_path + "\x1f" + gold).encode()).hexdigest()
    assert S.gold_cache().get(key)["gold_timeout_s"] == 0.5
    # a timeout cached under a smaller limit is not final: recomputed with the larger limit
    ev2 = make_eval(it, "select count(*) from a", timeout_s=0.2, gold_timeout_s=5)
    key2 = S.hashlib.sha256((it.db_path + "\x1f" + "select count(*) from a").encode()).hexdigest()
    S.gold_cache().put(key2, {"ok": False, "rows": [], "cols": [], "error": "timeout: interrupted", "truncated": False, "gold_timeout_s": 1.0})
    assert ev2.gold_exec(it)["ok"]


def test_execute_robust_retries_cold_timeouts_once(dbp, monkeypatch):
    calls = []

    def fake_execute(path, sql, timeout_s=30, max_rows=10000):
        calls.append(timeout_s)
        if len(calls) == 1:
            return {"ok": False, "rows": [], "cols": [], "error": "timeout: interrupted", "elapsed_s": 30.0, "truncated": False}
        return {"ok": True, "rows": [(1,)], "cols": ["x"], "error": None, "elapsed_s": 1.0, "truncated": False}

    monkeypatch.setattr(db, "execute", fake_execute)
    monkeypatch.setattr(db, "_LAST_USE", {})
    r = db.execute_robust(dbp, "select 1", timeout_s=30)
    assert r["ok"] and r["attempts"] == 2 and r["cold_start"] and calls == [30, 30]
    # warm db: no retry
    calls.clear()
    db._LAST_USE[dbp] = db.time.monotonic()
    monkeypatch.setattr(db, "execute", lambda *a, **k: {"ok": False, "rows": [], "cols": [], "error": "timeout: interrupted", "elapsed_s": 30.0, "truncated": False})
    r = db.execute_robust(dbp, "select 1", timeout_s=30)
    assert not r["ok"] and r["attempts"] == 1 and not r["cold_start"]
    # retry that also times out is final; non-timeout errors are never retried
    db._LAST_USE.clear()
    n = []
    monkeypatch.setattr(db, "execute", lambda *a, **k: (n.append(1), {"ok": False, "rows": [], "cols": [], "error": "timeout: x", "elapsed_s": 1, "truncated": False})[1])
    assert db.execute_robust(dbp, "select 1")["attempts"] == 2 and len(n) == 2
    db._LAST_USE.clear()
    n.clear()
    monkeypatch.setattr(db, "execute", lambda *a, **k: (n.append(1), {"ok": False, "rows": [], "cols": [], "error": "no such table: z", "elapsed_s": 1, "truncated": False})[1])
    assert db.execute_robust(dbp, "select 1")["attempts"] == 1 and len(n) == 1


def test_timed_out_query_leaves_db_cold_finished_query_warms_it(dbp):
    db._LAST_USE.pop(dbp, None)
    assert db.is_cold(dbp)
    db.execute(dbp, SLOW % 100000000, timeout_s=0.2)
    assert db.is_cold(dbp)
    db.execute(dbp, "select 1", timeout_s=5)
    assert not db.is_cold(dbp)
