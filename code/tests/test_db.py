import sqlite3
import time

import pytest

from nl2sql import db


@pytest.fixture()
def dbp(tmp_path):
    p = str(tmp_path / "t.sqlite")
    c = sqlite3.connect(p)
    c.execute("create table a (id integer primary key, name text, v real)")
    c.executemany("insert into a values (?,?,?)", [(i, "n%d" % (i % 5), i * 1.0) for i in range(1, 501)])
    c.commit()
    c.close()
    return p


def test_timeout_enforced(dbp):
    t = time.time()
    r = db.execute(dbp, "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM r) SELECT count(*) FROM r", timeout_s=1.0)
    assert not r["ok"] and r["error"].startswith("timeout")
    assert time.time() - t < 4.0
    assert db.error_kind(r["error"]) == "timeout"


def test_timeout_cross_join(dbp):
    t = time.time()
    r = db.execute(dbp, "select count(*) from a x, a y, a z, a w, a u", timeout_s=1.0)
    assert not r["ok"]
    assert time.time() - t < 4.0


def test_ok_and_truncation(dbp):
    r = db.execute(dbp, "select * from a", max_rows=10)
    assert r["ok"] and len(r["rows"]) == 10 and r["truncated"]
    r = db.execute(dbp, "select * from a", max_rows=None)
    assert len(r["rows"]) == 500 and not r["truncated"]


def test_errors(dbp):
    assert db.error_kind(db.execute(dbp, "select nope from a")["error"]) == "identifier"
    assert db.error_kind(db.execute(dbp, "selec * from a")["error"]) == "syntax"
    assert db.error_kind(db.execute(dbp, "select * from zzz")["error"]) == "identifier"
    assert not db.execute(dbp, "")["ok"]


def test_readonly(dbp):
    r = db.execute(dbp, "delete from a")
    assert not r["ok"]
    assert db.execute(dbp, "select count(*) from a")["rows"][0][0] == 500


def test_ex_semantics():
    # BIRD: set equality of row tuples -> order and duplicates ignored, column order matters
    assert db.ex_match([(1, "a"), (2, "b")], [(2, "b"), (1, "a")]) == 1
    assert db.ex_match([(1,), (1,), (2,)], [(1,), (2,)]) == 1
    assert db.ex_match([(1, "a")], [("a", 1)]) == 0
    assert db.ex_match([], []) == 1
    assert db.ex_match([(1,)], []) == 0
    assert db.ex_match([(1.0,)], [(1,)]) == 1          # python equality, as in the official script
    assert db.ex_match([(None,)], [(None,)]) == 1
    assert db.rows_key([(1,), (1,), (2,)]) == db.rows_key([(2,), (1,)])


def test_explain_and_counts(dbp):
    plan = db.explain(dbp, "select * from a where name='n1'")
    assert any("SCAN" in l for l in plan)
    plan = db.explain(dbp, "select * from a where id=3")
    assert any("SEARCH" in l for l in plan)
    assert db.explain(dbp, "select * from nope") == []
    assert db.row_counts(dbp) == {"a": 500}
