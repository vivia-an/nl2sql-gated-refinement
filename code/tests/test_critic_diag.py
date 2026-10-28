"""Offline tests of the injected-error diagnostic (tools/critic_diag.py, SAP Amendment 4): rule-based edits, validity filter, analysis on synthetic
records.  No LLM call, no network."""
import json
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import critic_diag as D  # noqa: E402


def edits(cat, sql, schema=None):
    toks = D._tokens(sql)
    return [(D._splice(sql, c["edits"]), c["info"]) for c in D.GENERATORS[cat](sql, toks, schema)]


SCHEMA = {"tables": {"emp": [{"name": "id", "type": "INTEGER"}, {"name": "age", "type": "INTEGER"}, {"name": "pay", "type": "INTEGER"},
                             {"name": "name", "type": "TEXT"}, {"name": "city", "type": "TEXT"}],
                     "dept": [{"name": "did", "type": "INTEGER"}, {"name": "dname", "type": "TEXT"}]}}


def test_agg_swap_pairs_and_case():
    assert [s for s, _ in edits("agg_maxmin", "SELECT MAX(age) FROM emp")] == ["SELECT MIN(age) FROM emp"]
    assert [s for s, _ in edits("agg_maxmin", "select min(age) from emp")] == ["select max(age) from emp"]
    assert sorted(s for s, _ in edits("agg_sumavg", "SELECT SUM(a), AVG(b) FROM t")) == ["SELECT AVG(a), AVG(b) FROM t", "SELECT SUM(a), SUM(b) FROM t"]
    assert edits("agg_maxmin", "SELECT name FROM emp WHERE name = 'MAX(x)'") == []         # inside a string literal


def test_count_distinct_both_directions_and_star():
    got = [s for s, _ in edits("count_distinct", "SELECT COUNT(city), COUNT(DISTINCT name), COUNT(*) FROM emp")]
    assert sorted(got) == ["SELECT COUNT(DISTINCT city), COUNT(DISTINCT name), COUNT(*) FROM emp",
                           "SELECT COUNT(city), COUNT(name), COUNT(*) FROM emp"]


def test_cmp_flip_and_limit_and_distinct():
    assert sorted(s for s, _ in edits("cmp_flip", "SELECT id FROM emp WHERE age > 30 AND pay <= 5")) == [
        "SELECT id FROM emp WHERE age < 30 AND pay <= 5", "SELECT id FROM emp WHERE age > 30 AND pay >= 5"]
    assert edits("cmp_flip", "SELECT id FROM emp WHERE name = '>'") == []
    assert [s for s, _ in edits("limit_change", "SELECT id FROM emp ORDER BY age DESC LIMIT 1")] == ["SELECT id FROM emp ORDER BY age DESC LIMIT 2"]
    assert edits("limit_change", "SELECT id FROM emp LIMIT 3") == []
    assert [s for s, _ in edits("distinct_drop", "SELECT DISTINCT city FROM emp")] == ["SELECT city FROM emp"]
    assert edits("distinct_drop", "SELECT COUNT(DISTINCT city) FROM emp") == []


def test_order_flip_explicit_and_implicit():
    assert [s for s, _ in edits("order_flip", "SELECT id FROM emp ORDER BY age DESC LIMIT 1")] == ["SELECT id FROM emp ORDER BY age ASC LIMIT 1"]
    assert [s for s, _ in edits("order_flip", "SELECT id FROM emp ORDER BY age LIMIT 1")] == ["SELECT id FROM emp ORDER BY age DESC LIMIT 1"]
    got = [s for s, _ in edits("order_flip", "SELECT id FROM emp ORDER BY age DESC, pay ASC")]
    assert got == ["SELECT id FROM emp ORDER BY age ASC, pay ASC"]                          # first key only


def test_where_drop_conjuncts_between_or_single():
    got = sorted(s for s, _ in edits("where_drop", "SELECT id FROM emp WHERE age > 30 AND city = 'X' AND pay BETWEEN 1 AND 5 ORDER BY id"))
    assert got == ["SELECT id FROM emp WHERE age > 30 AND city = 'X' ORDER BY id", "SELECT id FROM emp WHERE age > 30 AND pay BETWEEN 1 AND 5 ORDER BY id",
                   "SELECT id FROM emp WHERE city = 'X' AND pay BETWEEN 1 AND 5 ORDER BY id"]
    assert edits("where_drop", "SELECT id FROM emp WHERE age > 30 OR city = 'X'") == []     # top-level OR: not a conjunction
    single = edits("where_drop", "SELECT id FROM emp WHERE age > 30 GROUP BY id")
    assert [s for s, _ in single] == ["SELECT id FROM emp GROUP BY id"]
    assert single[0][1]["n_conjuncts"] == 1
    sub = edits("where_drop", "SELECT id FROM emp WHERE age > (SELECT AVG(age) FROM emp WHERE city = 'X') AND pay > 1")
    assert len(sub) == 2 + 1                                                                   # outer: 2 conjuncts; the subquery WHERE: 1


def test_col_swap_alias_type_and_ambiguity():
    got = edits("col_swap", "SELECT T1.age FROM emp AS T1 WHERE T1.city = 'X'", SCHEMA)
    assert sorted(s for s, _ in got) == ["SELECT T1.age FROM emp AS T1 WHERE T1.name = 'X'", "SELECT T1.id FROM emp AS T1 WHERE T1.city = 'X'",
                                         "SELECT T1.pay FROM emp AS T1 WHERE T1.city = 'X'"]
    assert all(i["table"] == "emp" for _, i in got)
    # a text column is never swapped for an integer one, and a bare column resolves only if exactly one table of the query has it
    assert not any("did" in s or "dname" in s for s, _ in got)
    bare = edits("col_swap", "SELECT dname FROM dept WHERE did > 1", SCHEMA)
    assert sorted(s for s, _ in bare) == []                                                   # dept has one TEXT and one INTEGER column: no sibling of the same type
    q = edits("col_swap", "SELECT `age` FROM emp", SCHEMA)
    assert "SELECT `id` FROM emp" in [s for s, _ in q]                                         # quoting style kept


def test_validity_filter_runs_against_sqlite(tmp_path, monkeypatch):
    p = str(tmp_path / "t.sqlite")
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE emp (id INTEGER, age INTEGER, city TEXT)")
    c.executemany("INSERT INTO emp VALUES (?,?,?)", [(1, 20, "A"), (2, 30, "A"), (3, 40, "B"), (4, 40, "B")])
    c.commit()
    c.close()

    class It:
        db_path = p
        gold_sql = "SELECT DISTINCT city FROM emp"
    it = It()
    g = D.exec_gold(it)
    assert g is not None
    # DISTINCT drop: invisible to the official set-EX, visible to the bag comparator (category rule)
    assert D.valid_injection("distinct_drop", it, g, "SELECT city FROM emp") is not None
    assert D.valid_injection("agg_maxmin", it, g, "SELECT city FROM emp") is None            # set comparator: same set -> not an error
    assert D.valid_injection("cmp_flip", it, g, "SELECT city FROM emp WHERE age > 100") is None   # empty result is not a silent error
    assert D.valid_injection("cmp_flip", it, g, "SELECT nope FROM emp") is None              # does not execute
    assert D.valid_injection("cmp_flip", it, g, "SELECT DISTINCT city FROM emp WHERE age < 30") is not None
    assert D.valid_injection("cmp_flip", it, g, it.gold_sql) is None                         # identical text


def _rec(i, cat, db, pert_score, gold_score=None, trig_match=1, root=0):
    def v(score, root_match):
        ck = "ex_bag" if cat == "distinct_drop" else "ex_set"
        res = {"ok": True, "ex_set": root_match, "ex_bag": root_match, "n_rows": 1}
        return {"sql": "S", "critic": {"syntax": 10.0, "semantic": score, "efficiency": 10.0, "score": 3 + 0.5 * score + 2, "status": "ok_nonempty", "parse_ok": True,
                                       "issues": [{"type": "semantic", "description": "x"}] if score < 7 else [], "suggestions": []},
                "triggered": 3 + 0.5 * score + 2 < 8.5, "must_fix": 3 + 0.5 * score + 2 < 7.5, "abstain_theta_h": False, "gated_would_revise": False,
                "root_res": res, "actions": ["revise"], "rounds": [{"sql": "S2", "noop": False, "critic": {}, "score_res": dict(res, ex_set=trig_match, ex_bag=trig_match)}],
                "final_sql": "S2", "final_k": 1, "final_score": 9.0, "final_res": dict(res, ex_set=trig_match, ex_bag=trig_match)}
    r = {"id": "D%03d" % i, "cat": cat, "qid": "q%d" % i, "db_id": db, "variants": {"pert": v(pert_score, root)}}
    if gold_score is not None:
        r["variants"]["gold"] = v(gold_score, 1)
    return r


def test_analyze_synthetic(tmp_path):
    pytest.importorskip("numpy")
    recs = []
    k = 0
    for cat in ("agg_maxmin", "distinct_drop"):
        for d in range(6):
            for rep in range(3):
                recs.append(_rec(k, cat, "db%d" % d, pert_score=3.0 if rep < 2 else 9.0, gold_score=9.0 if rep == 0 else None, trig_match=1 if rep == 0 else 0))
                k += 1
    items = {"items": [{"id": r["id"]} for r in recs]}
    ip, rp = tmp_path / "items.json", tmp_path / "records.jsonl"
    ip.write_text(json.dumps(items))
    rp.write_text("\n".join(json.dumps(r) for r in recs))

    class A:
        pass
    a = A()
    a.items, a.run, a.out, a.B = str(ip), str(rp), str(tmp_path / "out"), 200
    D.cmd_analyze(a)
    s = json.load(open(str(tmp_path / "out" / "critic_diag_summary.json")))
    assert s["_meta"]["scope"] == "diagnostic-injected"
    c = s["per_category"]["agg_maxmin"]
    assert c["n_pert"] == 18 and abs(c["detect_triggered"]["rate"] - 12 / 18.0) < 1e-9
    assert c["fp_triggered_gold"]["rate"] == 0.0 and c["auroc_paired_controls"]["auroc"] == 1.0
    assert abs(c["repair_round1_all"]["rate"] - 6 / 18.0) < 1e-9
    assert s["pooled"]["all"]["n_pert"] == 36
    assert os.path.exists(str(tmp_path / "out" / "critic_diag_fig.json"))
    cs = json.load(open(str(tmp_path / "out" / "critic_diag_cases.json")))
    assert {c["class"] for c in cs["cases"]} >= {"A_detected_and_repaired", "B_detected_not_repaired", "C_missed", "D_gold_false_positive"} - {"D_gold_false_positive"}


def test_diag_does_not_touch_frozen_artefacts():
    """The tool must not change the lock-relevant files: prompts and models.yaml are only read."""
    from nl2sql import config as C
    before = C.prompts_sha()
    assert isinstance(before, str) and len(before) >= 16
    src = open(D.__file__).read()
    for bad in ("open(os.path.join(str(paths.PROMPTS_DIR)", "models.yaml\", \"w\"", "dev_lock.json\", \"w\""):
        assert bad not in src
