import os
import sqlite3

import pytest

from nl2sql import paths
from nl2sql.eval import spider_ts

TS = paths.DATA / "eval_tools" / "test-suite-sql-eval"
pytestmark = pytest.mark.skipif(not (TS / "exec_eval.py").exists(), reason="official test-suite code not on this machine")


@pytest.fixture()
def suite(tmp_path, monkeypatch):
    """a mini test suite: main db + one neighbour db that differs in content (what the distilled suites are)."""
    pytest.importorskip("sqlparse")
    root = tmp_path / "database" / "mini"
    root.mkdir(parents=True)
    for name, rows in (("mini.sqlite", [(1, "a"), (2, "b"), (3, "c")]), ("mini_v1.sqlite", [(1, "a"), (2, "b"), (3, "c"), (2, "z"), (5, "e")])):
        c = sqlite3.connect(str(root / name))
        c.execute("create table t (x integer, y text)")
        c.executemany("insert into t values (?,?)", rows)
        c.commit()
        c.close()
    monkeypatch.setenv("NL2SQL_TS_DBS", str(tmp_path / "database"))
    monkeypatch.setenv("NL2SQL_TS_DIR", str(TS))
    pool = spider_ts.TSPool(2, timeout_s=60)
    yield pool
    pool.close()


def test_match_semantics(suite):
    g = "SELECT y FROM t WHERE x = 2"
    assert suite.match("mini", g, g)["ex_ts"] == 1
    assert suite.match("mini", "SELECT y FROM t WHERE x >= 2 AND x <= 2", g)["ex_ts"] == 1
    # coincidentally right on the main db (only 'b' has x=2) but wrong on the neighbour db (x=2 also has y='z'): plain EX accepts it
    r = suite.match("mini", "SELECT y FROM t WHERE y = 'b'", g)
    assert r["ex_ts"] == 0 and r["error"] is None
    assert suite.match("mini", "SELECT nope FROM t", g)["ex_ts"] == 0                 # execution error -> 0
    assert suite.match("mini", "", g) == {"ex_ts": 0, "error": "empty_pred"}
    assert suite.match("nodb", g, g) == {"ex_ts": None, "error": "no_suite_db"}
    r = suite.match("mini", g, "SELECT broken FROM t")                                 # gold error: reported, scored 0
    assert r["ex_ts"] == 0 and r["error"].startswith("gold_error")


def test_distinct_removed_and_order_sensitivity(suite):
    g = "SELECT DISTINCT y FROM t"
    assert suite.match("mini", "SELECT y FROM t", g)["ex_ts"] == 1        # official: DISTINCT is removed from both unless keep_distinct
    assert suite.match("mini", "SELECT y FROM t ORDER BY y DESC", "SELECT y FROM t ORDER BY y ASC")["ex_ts"] == 0    # gold has ORDER BY
    assert suite.match("mini", "SELECT y FROM t ORDER BY y DESC", "SELECT y FROM t")["ex_ts"] == 1                    # no ORDER BY in gold


def test_timeout_kills_and_respawns_worker(suite):
    suite.timeout_s = 3
    slow = "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM r) SELECT count(*) FROM r"
    r = suite.match("mini", slow, "SELECT count(*) FROM t")
    assert r == {"ex_ts": 0, "error": "timeout"}
    assert suite.match("mini", "SELECT count(*) FROM t", "SELECT count(*) FROM t")["ex_ts"] == 1      # pool still healthy
