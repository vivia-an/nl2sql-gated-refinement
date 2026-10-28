"""X7 tests (offline: scripted fake LLM, toy sqlite databases, synthetic preds; no API call, no dev / Spider data).
Covers: value probing for empty-result revisions (m5.probe_values), confidence-routed compute (route.adaptive) incl. the pure decision
function, legacy byte-identity with the switches off, tools/adaptive_replay.py, tools/value_check.py, tools/empty_roots.py."""
import json
import os
import sqlite3
import sys

import pytest

from nl2sql import config as C, data, db, paths, prompts_util as P
from nl2sql.systems.ours import m5 as M5, pipeline as PL, probe as PR
from nl2sql.systems.ours.defaults import DEFAULTS
from nl2sql.systems.ours.route import adaptive_decide
from test_x5 import BAD, CRIT9, GATED, GOOD, ScriptLLM, fake_retrieve, make_sys, mk_item, toydb  # noqa: F401  (toydb is a fixture)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

CONFIGS = paths.EXP_DIR / "configs"


class SchemaStore:
    def __init__(self, schema):
        self.schema = schema


class SchemaRes:
    """Stand-in for pipeline.Resources that serves the REAL schema of the toy database (types are needed to find text columns)."""

    def __init__(self, schema):
        self.s = SchemaStore(schema)

    def db_store(self, item):
        return self.s

    def corpus_for(self, item):
        return None

    def ensure_terms(self, item):
        pass


def make_probe_sys(monkeypatch, cls, llm, cfg, toydb):
    s = make_sys(monkeypatch, cls, llm, cfg)
    s.pipe.res = SchemaRes(data._build_schema(toydb, "toy", "t"))
    return s


PROBE = {"m5": {"policy": "gated", "gate_empty": True, "probe_values": True}, "route": {"score": "exec", "theta_h": 7.5}, "ddl": {"quote_keywords": True}}
EMPTY_A = 'SELECT id FROM "transaction" WHERE "order" = \'A\''          # stored values of "order" are a, b, c (lower case)
FIX_A = 'SELECT id FROM "transaction" WHERE "order" = \'a\''


# ------------------------------------------------------------------------------------------------ 1. probe module
@pytest.fixture()
def pdb(tmp_path):
    p = str(tmp_path / "p.sqlite")
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE emp (id INTEGER PRIMARY KEY, name TEXT, dept TEXT, salary REAL, bio TEXT, code VARCHAR(5))")
    rows = [(i, "n%d" % i, ["Sales", "R&D", "HR"][i % 3], 10.0 * i, "x" * 200, "C%d" % (i % 12)) for i in range(1, 31)]
    con.executemany("INSERT INTO emp VALUES (?,?,?,?,?,?)", rows)
    con.execute("CREATE TABLE dept (dept TEXT, city TEXT)")
    con.executemany("INSERT INTO dept VALUES (?,?)", [("Sales", "Oslo"), ("HR", "Rome")])
    con.commit()
    con.close()
    return p


def sch(p):
    return data._build_schema(p, "toy", "t")


def test_is_text_type():
    for t in ("TEXT", "text", "VARCHAR(10)", "CHAR", "NVARCHAR", "CLOB", "", None):
        assert PR.is_text_type(t), t
    for t in ("INTEGER", "REAL", "NUMERIC", "DATE", "DATETIME", "BOOLEAN", "FLOAT"):
        assert not PR.is_text_type(t), t


def lc(sql, p):
    return [(c["table"], c["column"], c["literals"]) for c in PR.literal_columns(sql, sch(p))]


def test_literal_columns_where_join_having_aliases_and_scopes(pdb):
    assert lc("SELECT id FROM emp WHERE dept = 'sales'", pdb) == [("emp", "dept", ["sales"])]
    assert lc("SELECT T1.id FROM emp AS T1 JOIN dept AS T2 ON T1.dept = T2.dept WHERE T2.city = 'oslo' AND T1.name = 'n1'", pdb) == \
        [("dept", "city", ["oslo"]), ("emp", "name", ["n1"])]
    # IN list, LIKE, <>, literal on the left, lower(col)
    assert lc("SELECT id FROM emp WHERE dept IN ('a', 'b') OR name LIKE '%n1%' OR 'x' = code", pdb) == \
        [("emp", "dept", ["a", "b"]), ("emp", "name", ["%n1%"]), ("emp", "code", ["x"])]
    assert lc("SELECT id FROM emp WHERE lower(dept) = 'sales'", pdb) == [("emp", "dept", ["sales"])]
    # literal in a JOIN ... ON and in HAVING
    assert lc("SELECT e.id FROM emp e JOIN dept d ON e.dept = d.dept AND d.city = 'rome'", pdb) == [("dept", "city", ["rome"])]
    assert lc("SELECT dept, count(*) FROM emp GROUP BY dept HAVING dept = 'Sales'", pdb) == [("emp", "dept", ["Sales"])]
    # NOT counted: numeric columns, select-list literals / CASE, column = column, unknown columns, IN (subquery)
    assert lc("SELECT id FROM emp WHERE salary = 100 AND id = 3", pdb) == []
    assert lc("SELECT CASE WHEN dept = 'Sales' THEN 1 ELSE 0 END FROM emp", pdb) == []
    assert lc("SELECT e.id FROM emp e JOIN dept d ON e.dept = d.dept", pdb) == []
    assert lc("SELECT id FROM emp WHERE nosuch = 'x'", pdb) == []
    assert lc("SELECT id FROM emp WHERE dept IN (SELECT dept FROM dept WHERE city = 'Oslo')", pdb) == [("dept", "city", ["Oslo"])]
    # same alias in two scopes resolves per scope; an unqualified column present in two tables is attributed to both
    assert lc("SELECT * FROM emp t WHERE t.dept = 'a' AND t.id IN (SELECT 1 FROM dept t WHERE t.city = 'b')", pdb) == \
        [("emp", "dept", ["a"]), ("dept", "city", ["b"])]
    assert lc("SELECT * FROM emp JOIN dept ON emp.name = dept.city WHERE dept = 'zz'", pdb) == [("dept", "dept", ["zz"]), ("emp", "dept", ["zz"])]
    # numbers compared with a text column count (a text column filtered with a number is a typical format error)
    assert lc("SELECT id FROM emp WHERE code = 5", pdb) == [("emp", "code", ["5"])]
    # unparsable / empty SQL never raises
    assert lc("SELEC bad (", pdb) == [] and lc("", pdb) == []


def test_probe_column_values_complete_flag_length_filter_and_timeout(pdb, tmp_path):
    p = PR.probe_column(pdb, "emp", "dept", n=8)
    assert sorted(p["values"]) == ["HR", "R&D", "Sales"] and p["complete"] is True
    p = PR.probe_column(pdb, "emp", "code", n=8)
    assert len(p["values"]) == 8 and p["complete"] is False and len(set(p["values"])) == 8             # 12 distinct codes, only 8 shown
    assert PR.probe_column(pdb, "emp", "bio", n=8, max_len=40) is None                                    # long text is not probed
    assert PR.probe_column(pdb, "emp", "nosuch") is None and PR.probe_column(pdb, "nosuch", "x") is None
    assert PR.probe_column(pdb, "emp", 'na"me', n=3) is None                                             # identifiers are quoted, never executed raw
    # timeout guard: a scan that cannot finish within the limit returns None instead of blocking
    big = str(tmp_path / "big.sqlite")
    con = sqlite3.connect(big)
    con.execute("CREATE TABLE t (c TEXT)")
    con.executemany("INSERT INTO t VALUES (?)", (("v%d" % (i % 3),) for i in range(400000)))
    con.commit()
    con.close()
    assert PR.probe_column(big, "t", "c", n=8, timeout_s=0.0001) is None
    assert sorted(PR.probe_column(big, "t", "c", n=8, timeout_s=20)["values"]) == ["v0", "v1", "v2"]
    # read-only: the database file is unchanged
    before = open(pdb, "rb").read()
    PR.probe_failed_sql(pdb, "SELECT id FROM emp WHERE dept = 'x'", sch(pdb))
    assert open(pdb, "rb").read() == before


def test_probe_failed_sql_text_and_limits(pdb):
    s = sch(pdb)
    pr = PR.probe_failed_sql(pdb, "SELECT id FROM emp WHERE dept = 'sales' AND code = 'c1' AND name = 'zz' AND bio = 'q'", s, n=2, max_cols=2)
    assert [(p["column"], p["complete"]) for p in pr] == [("dept", False), ("code", False)]                # bio skipped (long); max 2 columns
    txt = PR.probe_text(pr, 2)
    assert txt.splitlines()[0].startswith('- "emp"."dept" (the query compares it with \'sales\'): ') and txt.splitlines()[0].endswith(", ...")
    assert PR.probe_cols(pr) == ["emp.dept", "emp.code"]
    p3 = PR.probe_failed_sql(pdb, "SELECT id FROM emp WHERE dept = 'it''s'", s, n=8)
    assert "'it''s'" in PR.probe_text(p3) and p3[0]["complete"] is True


def test_probe_near_lists_resembling_values_first_and_is_off_by_default(pdb):
    con = sqlite3.connect(pdb)
    con.execute("CREATE TABLE town (name TEXT)")
    con.executemany("INSERT INTO town VALUES (?)", [("Town %02d" % i,) for i in range(30)] + [("Bishopville SC",), ("Bishopville",), ("it's Bish",)])
    con.commit()
    con.close()
    s = sch(pdb)
    q = "SELECT * FROM town WHERE name = 'bishopville'"
    plain = PR.probe_failed_sql(pdb, q, s, n=4)[0]
    assert plain["near"] == 0 and not any("ishop" in v for v in plain["values"])                            # first 4 distinct values only
    near = PR.probe_failed_sql(pdb, q, s, n=4, near=True)[0]
    assert near["near"] == 2 and sorted(near["values"][:2]) == ["Bishopville", "Bishopville SC"] and len(near["values"]) == 4
    assert "the first 2 values resemble it" in PR.probe_text([near], 4) and "resemble" not in PR.probe_text([plain], 4)
    assert PR.near_tokens(["%a%", "x_", "O'Brien Co", "12"]) == ["brien"] and PR.near_tokens(["it's"]) == []
    # a complete list is never reordered / extended
    comp = PR.probe_failed_sql(pdb, "SELECT id FROM emp WHERE dept = 'sales'", s, n=8, near=True)[0]
    assert comp["complete"] is True and comp["near"] == 0 and comp["values"] == PR.probe_failed_sql(pdb, "SELECT id FROM emp WHERE dept = 'sales'", s, n=8)[0]["values"]


# ------------------------------------------------------------------------------------------------ 2. probe in the gated loop
def test_probe_template_is_the_gated_template_plus_one_block():
    base = P._read("rev_exec.user")
    probe = P._read("rev_exec_probe.user")
    head, tail = base.split("# Execution feedback\n", 1)
    assert probe.startswith(head) and probe.endswith("# Execution feedback\n" + tail)
    mid = probe[len(head): len(probe) - len("# Execution feedback\n" + tail)]
    assert mid.startswith("# Values stored in the filtered columns\n") and mid.endswith("<<probe>>\n\n")
    # rendering with an empty-able optional placeholder is not needed: the probe template needs <<probe>>
    with pytest.raises(KeyError):
        P.render("rev_exec_probe.user", sections="s", question="q", evidence="e", graph="", sql="x", preview="p", feedback="f")


def run_probe(monkeypatch, toydb, gen, rev, cfg=None):
    llm = ScriptLLM(gen=gen, rev=rev)
    sysm = make_probe_sys(monkeypatch, PL.FullGated, llm, C.deep_update(PROBE, cfg or {}), toydb)
    return llm, sysm.run(mk_item(toydb))


def test_probe_values_go_into_the_revision_prompt_and_are_logged(monkeypatch, toydb):
    llm, out = run_probe(monkeypatch, toydb, EMPTY_A, FIX_A)
    assert llm.tags() == ["gen", "rev"]
    rev = llm.calls[1]["user"]
    assert "# Values stored in the filtered columns" in rev and "returned no rows" in rev
    assert '- "transaction"."order" (the query compares it with \'A\'): \'a\', \'b\', \'c\'' in rev          # complete list: no trailing "..."
    assert rev.index("# Execution\n") < rev.index("# Values stored") < rev.index("# Execution feedback")
    assert out["final_sql"] == FIX_A and out["final_status"] == "ok_nonempty" and out["actions"] == ["revise", "accept"]
    assert out["probed"] == 1 and out["probe_cols"] == ["transaction.order"]
    c = out["candidates"]
    assert [x["probed"] for x in c] == [1, 0] and [x["probe_cols"] for x in c] == [["transaction.order"], []]       # logged on the candidate that was revised


def test_probe_all_null_aggregate_is_degenerate_too(monkeypatch, toydb):
    sql = 'SELECT max(amt) FROM "transaction" WHERE "order" = \'zz\''                # one row of NULL -> ok_empty
    llm, out = run_probe(monkeypatch, toydb, sql, 'SELECT max(amt) FROM "transaction" WHERE "order" = \'b\'')
    assert llm.tags() == ["gen", "rev"] and out["probed"] == 1 and out["final_status"] == "ok_nonempty"


def test_probe_not_used_for_errors_nor_when_nothing_applies_and_prompt_equals_gated(monkeypatch, toydb):
    # execution error: the legacy gated template (no probe block) even with probe_values on
    llm, out = run_probe(monkeypatch, toydb, BAD, GOOD)
    assert "Values stored" not in llm.calls[1]["user"] and out["probed"] == 0 and out["probe_cols"] == []
    # empty result whose literal sits on a numeric column: nothing to probe -> the prompt is byte-identical to plain gated + gate_empty
    empty = 'SELECT id FROM "transaction" WHERE amt > 100'
    llm1, out1 = run_probe(monkeypatch, toydb, empty, GOOD)
    llm0 = ScriptLLM(gen=empty, rev=GOOD)
    cfg0 = {"m5": {"policy": "gated", "gate_empty": True}, "route": {"score": "exec", "theta_h": 7.5}, "ddl": {"quote_keywords": True}}
    make_sys(monkeypatch, PL.FullGated, llm0, cfg0).run(mk_item(toydb))
    assert llm1.calls[1]["user"] == llm0.calls[1]["user"] and out1["probed"] == 0


def test_probe_off_gives_the_legacy_prompt_and_rows(monkeypatch, toydb):
    cfg0 = {"m5": {"policy": "gated", "gate_empty": True}, "route": {"score": "exec", "theta_h": 7.5}, "ddl": {"quote_keywords": True}}
    llm = ScriptLLM(gen=EMPTY_A, rev=FIX_A)
    out = make_probe_sys(monkeypatch, PL.FullGated, llm, cfg0, toydb).run(mk_item(toydb))
    assert "Values stored" not in llm.calls[1]["user"] and not ({"probed", "probe_cols"} & set(out))
    assert all("probed" not in c for c in out["candidates"])
    # and the prompt is exactly what the pre-X7 code rendered from rev_exec.user
    ctx = fake_retrieve(None, None, None, {"ddl": {"quote_keywords": True}}, None)
    prev = M5.exec_summary(db.execute(toydb, EMPTY_A))
    want = P.render("rev_exec.user", sections=ctx.sections(), question=mk_item(toydb).question, evidence="(none)", graph="", sql=EMPTY_A,
                    preview=prev, quote_rule=P.QUOTE_RULE, feedback=M5.exec_feedback_text(db.execute(toydb, EMPTY_A), "ok_empty"))
    assert llm.calls[1]["user"] == want


def test_probe_failure_never_fails_the_question(monkeypatch, toydb):
    def boom(*a, **k):
        raise RuntimeError("probe exploded")
    monkeypatch.setattr(PR, "probe_failed_sql", boom)
    llm, out = run_probe(monkeypatch, toydb, EMPTY_A, FIX_A)
    assert llm.tags() == ["gen", "rev"] and out["final_sql"] == FIX_A and out["probed"] == 0 and "Values stored" not in llm.calls[1]["user"]


def test_probe_applies_to_every_empty_node_in_the_chain(monkeypatch, toydb):
    llm = ScriptLLM(gen=EMPTY_A, rev=lambda u, n, k: 'SELECT id FROM "transaction" WHERE "order" = \'B\'' if k == 2 else FIX_A)
    s = make_probe_sys(monkeypatch, PL.FullGated, llm, PROBE, toydb)
    out = s.run(mk_item(toydb))
    assert llm.tags() == ["gen", "rev", "rev"] and out["probed"] == 2 and out["probe_cols"] == ["transaction.order"]
    assert "Values stored" in llm.calls[1]["user"] and "Values stored" in llm.calls[2]["user"] and "'B'" in llm.calls[2]["user"]


def test_probe_config_validation(monkeypatch):
    llm = ScriptLLM()
    with pytest.raises(ValueError, match="probe_values"):
        make_sys(monkeypatch, PL.FullGated, llm, {"m5": {"probe_values": True}, "route": {"score": "exec"}})                 # gate_empty missing
    with pytest.raises(ValueError, match="probe_values"):
        make_sys(monkeypatch, PL.FullFixed, llm, {"m5": {"probe_values": True, "gate_empty": True}})                         # p0 policy
    make_sys(monkeypatch, PL.FullGated, llm, {"m5": {"probe_values": True, "gate_empty": True}, "route": {"score": "exec"}})


# ------------------------------------------------------------------------------------------------ 3. adaptive decision (pure)
def K(*xs):
    return [None if x is None else frozenset([(x,)]) for x in xs]


def test_adaptive_decide_fast_vote_ties_and_failures():
    d = adaptive_decide(K(1, 1, 1, 1, 1), None, 1.0)
    assert d["path"] == "fast" and d["chosen"] == 0 and d["cons"] == 1.0 and d["agree"] == 4 and d["n"] == 4
    d = adaptive_decide(K(1, 1, 1, 1, 2), None, 1.0)                                    # 3/4 < 1.0 -> vote; root group (4) wins
    assert d["path"] == "vote" and d["chosen"] == 0 and d["cons"] == 0.75 and d["sizes"] == [4, 1] and not d["tie"]
    assert adaptive_decide(K(1, 1, 1, 1, 2), None, 0.75)["path"] == "fast"
    assert adaptive_decide(K(1, 1, 1, 2, 2), None, 0.75)["path"] == "vote" and adaptive_decide(K(1, 1, 1, 2, 2), None, 0.5)["path"] == "fast"
    d = adaptive_decide(K(1, 2, 2, 2, 1), None, 1.0)                                    # sample group (3) beats root group (2)
    assert d["path"] == "vote" and d["chosen"] == 1 and d["sizes"] == [3, 2]
    d = adaptive_decide(K(1, 2, 2, 1, 1), None, 1.0)                                    # root group 3 vs 2
    assert d["chosen"] == 0
    d = adaptive_decide(K(1, 1, 2, 2), None, 1.0)                                       # 2 vs 2 tie, root in it -> root
    assert d["chosen"] == 0 and d["tie"] is True and d["tied"] == []
    d = adaptive_decide(K(1, 2, 2, 3, 3), None, 1.0)                                    # root alone; tie 2 vs 2 among samples -> earliest group
    assert d["chosen"] == 1 and d["tie"] and d["path"] == "vote"
    d = adaptive_decide(K(1, 2, 2, 3, 3), None, 1.0, tiebreak="v3")
    assert d["tied"] == [1, 3] and d["chosen"] == 1
    d = adaptive_decide(K(1, 1, 2, 2), None, 1.0, tiebreak="v3")                        # root's group is among the tied groups (index 0)
    assert d["tied"] == [0, 2] and d["chosen"] == 0
    # execution failures never agree and never vote
    d = adaptive_decide(K(None, 2, 2, None), None, 1.0)
    assert d["cons"] == 0.0 and d["chosen"] == 1 and d["sizes"] == [2]
    d = adaptive_decide(K(None, None, None), None, 1.0)
    assert d["chosen"] == 0 and d["path"] == "vote" and d["sizes"] == []
    d = adaptive_decide(K(1, None, None, None), None, 1.0)                              # root alone executes: it wins by default
    assert d["chosen"] == 0 and d["cons"] == 0.0
    # theta 0 = always fast; no samples = fast
    assert adaptive_decide(K(1, 2, 3), None, 0.0)["path"] == "fast" and adaptive_decide(K(1), None, 1.0)["path"] == "fast"
    with pytest.raises(ValueError):
        adaptive_decide(K(1, 2), None, 1.0, tiebreak="bogus")
    with pytest.raises(ValueError):
        adaptive_decide(K(1, 2), None, 1.0, empty="bogus")


def test_adaptive_decide_empty_demote():
    keys = [frozenset(), frozenset(), frozenset(), frozenset([(5,)]), frozenset([(5,)])]       # root and sample 1 empty (2), samples 2-3 non-empty (2)
    emp = [True, True, False, False, False]
    assert adaptive_decide(keys, emp, 1.0, empty="vote")["chosen"] == 0                         # tie 2:2 -> root (empty)
    d = adaptive_decide(keys, emp, 1.0, empty="demote")
    assert d["chosen"] == 3 and not d["tie"]                                                   # a non-empty group outranks the empty one
    keys2 = [frozenset(), frozenset(), frozenset(), frozenset(), frozenset([(5,)])]
    emp2 = [True, True, True, True, False]
    assert adaptive_decide(keys2, emp2, 1.0, empty="vote")["chosen"] == 0 and adaptive_decide(keys2, emp2, 1.0, empty="demote")["chosen"] == 4
    assert adaptive_decide(keys2, emp2, 0.75, empty="demote")["path"] == "fast"                 # fast path unaffected


# ------------------------------------------------------------------------------------------------ 4. adaptive in the pipeline
AD = {"m5": {"policy": "gated"}, "ddl": {"quote_keywords": True},
      "route": {"score": "consistency", "consistency": {"k": 4, "temperature": 0.7, "n_per_call": 4},
                "adaptive": {"enabled": True, "theta_fast": 1.0, "tiebreak": "root", "empty": "vote"}}}
CNT = 'SELECT count(*) FROM "transaction"'            # {(3,)}
SUM = 'SELECT sum(amt) FROM "transaction"'           # {(13.5,)}
MAX = 'SELECT max(amt) FROM "transaction"'           # {(7.5,)}


def cons_n(samples):
    def h(user, n, k):
        assert n == len(samples)
        return samples if n > 1 else samples[0]
    return h


def run_ad(monkeypatch, toydb, gen, samples, cfg=None, **kw):
    llm = ScriptLLM(gen=gen, cons=cons_n(samples), **kw)
    out = make_probe_sys(monkeypatch, PL.FullGated, llm, C.deep_update(AD, cfg or {}), toydb).run(mk_item(toydb))
    return llm, out


def test_adaptive_fast_path_all_samples_agree(monkeypatch, toydb):
    llm, out = run_ad(monkeypatch, toydb, CNT, [CNT, 'SELECT count(id) FROM "transaction"', CNT, CNT])
    assert llm.tags() == ["gen", "cons"] and llm.calls[1]["n"] == 4                 # one sampling request, no critic, no revision
    assert out["adaptive_path"] == "fast" and out["adaptive_choice"] == "root" and out["adaptive_cons"] == 1.0 and out["final_sql"] == CNT
    assert out["actions"] == ["adaptive_fast"] and out["iterations"] == 0 and not out["abstained"]
    assert out["consistency"] == 1.0 and out["cons_bits"] == [1, 1, 1, 1] and out["route_kind"] == "consistency" and out["route_score"] == 10.0
    # candidates: the root (cand_idx 0) and the four samples
    assert [c["action"] for c in out["candidates"]] == ["root"] + ["sample"] * 4 and out["candidates"][0]["sql"] == CNT


def test_adaptive_vote_picks_the_majority_result_without_any_revision(monkeypatch, toydb):
    llm, out = run_ad(monkeypatch, toydb, SUM, [CNT, CNT, 'SELECT count(id) FROM "transaction"', MAX])
    assert llm.tags() == ["gen", "cons"]                                            # no rev, no crit
    assert out["adaptive_path"] == "vote" and out["adaptive_choice"] == "sample" and out["final_sql"] == CNT and out["adaptive_sizes"] == [3, 1, 1]
    assert out["actions"] == ["adaptive_vote", "pick_sample"] and out["iterations"] == 0 and out["adaptive_cons"] == 0.0
    assert out["consistency"] == 0.75 and out["cons_bits"] == [1, 1, 1, 0]          # routing consistency is relative to the FINAL answer and out["n_agree"] == 3 and out["confidence"] == out["route_score"] == 7.5
    assert out["final_status"] == "ok_nonempty"


def test_adaptive_vote_keeps_the_root_when_it_wins_or_ties(monkeypatch, toydb):
    llm, out = run_ad(monkeypatch, toydb, CNT, [CNT, CNT, SUM, MAX])
    assert out["adaptive_path"] == "vote" and out["adaptive_choice"] == "root" and out["actions"] == ["adaptive_vote", "pick_root"] and out["adaptive_tie"] is False
    llm, out = run_ad(monkeypatch, toydb, CNT, [CNT, SUM, SUM, MAX], cfg={"route": {"consistency": {"k": 4}}})        # root 2 / sum 2 -> tie -> root
    assert out["adaptive_choice"] == "root" and out["adaptive_tie"] is True and out["final_sql"] == CNT
    llm, out = run_ad(monkeypatch, toydb, CNT, [SUM, SUM, MAX, MAX])                  # root alone (1), tie 2:2 among samples -> earliest group (SUM)
    assert out["final_sql"] == SUM and out["adaptive_tie"] is True


def test_adaptive_root_that_fails_is_outvoted_without_revision(monkeypatch, toydb):
    llm, out = run_ad(monkeypatch, toydb, BAD, [GOOD, GOOD, BAD, "SELEC x"])
    assert llm.tags() == ["gen", "cons"] and out["final_sql"] == GOOD and out["adaptive_path"] == "vote" and out["adaptive_cons"] == 0.0
    assert out["final_status"] == "ok_nonempty" and out["consistency"] == 0.5
    llm, out = run_ad(monkeypatch, toydb, BAD, [BAD, BAD, BAD, BAD])                  # nothing executes: the root stays (no revision by design)
    assert out["final_sql"] == BAD and out["final_status"] != "ok_nonempty" and llm.tags() == ["gen", "cons"] and out["adaptive_sizes"] == []
    assert out["consistency"] == 0.0 and out["cons_bits"] == [0, 0, 0, 0] and out["adaptive_cons"] == 0.0       # two failures are not agreement


def test_adaptive_theta_fast_levels(monkeypatch, toydb):
    s = [CNT, CNT, CNT, SUM]
    for theta, path in ((1.0, "vote"), (0.75, "fast"), (0.5, "fast")):
        llm, out = run_ad(monkeypatch, toydb, CNT, s, cfg={"route": {"adaptive": {"theta_fast": theta}}})
        assert out["adaptive_path"] == path and out["final_sql"] == CNT
    s = [CNT, SUM, SUM, SUM]
    llm, out = run_ad(monkeypatch, toydb, CNT, s, cfg={"route": {"adaptive": {"theta_fast": 0.75}}})
    assert out["adaptive_path"] == "vote" and out["final_sql"] == SUM


def test_adaptive_tiebreak_v3_scores_each_tied_group_once(monkeypatch, toydb):
    def crit(user, n, k):
        return '{"semantic": 9, "issues": [], "suggestions": []}' if "sum(amt)" in user.split("# SQL")[1] else '{"semantic": 2, "issues": [], "suggestions": []}'
    llm, out = run_ad(monkeypatch, toydb, CNT, [CNT, SUM, SUM, MAX][:4], cfg={"route": {"adaptive": {"tiebreak": "v3"}}}, crit=crit)
    # groups: CNT {root, s1} = 2, SUM {s2, s3} = 2, MAX = 1 -> tie CNT vs SUM -> two critic calls, SUM has the better V3
    assert llm.tags() == ["gen", "cons", "crit", "crit"] and out["final_sql"] == SUM and out["adaptive_choice"] == "sample"
    assert out["adaptive_v3"] == {"0": pytest.approx(0.3 * 10 + 0.5 * 2 + 0.2 * 10), "2": pytest.approx(0.3 * 10 + 0.5 * 9 + 0.2 * 10)}
    llm, out = run_ad(monkeypatch, toydb, CNT, [CNT, SUM, SUM, MAX], cfg={"route": {"adaptive": {"tiebreak": "v3"}}},
                      crit=lambda u, n, k: '{"semantic": 5, "issues": [], "suggestions": []}')
    assert out["final_sql"] == CNT                                                    # equal V3 -> root
    llm, out = run_ad(monkeypatch, toydb, CNT, [CNT, CNT, SUM, MAX], cfg={"route": {"adaptive": {"tiebreak": "v3"}}})
    assert llm.tags() == ["gen", "cons"]                                              # no tie -> no critic call


def test_adaptive_without_candidate_records_still_works(monkeypatch, toydb):
    llm, out = run_ad(monkeypatch, toydb, SUM, [CNT, CNT, CNT, MAX], cfg={"record_candidates": False})
    assert out["final_sql"] == CNT and "candidates" not in out and out["adaptive_choice"] == "sample"


def test_adaptive_sample_requests_follow_n_per_call(monkeypatch, toydb):
    llm = ScriptLLM(gen=CNT, cons=lambda u, n, k: [CNT] * n)
    cfg = C.deep_update(AD, {"route": {"consistency": {"k": 3, "n_per_call": 2}}})
    out = make_probe_sys(monkeypatch, PL.FullGated, llm, cfg, toydb).run(mk_item(toydb), seed=10)
    assert [(c["n"], c["seed"]) for c in llm.calls if c["tag"] == "cons"] == [(2, 11), (1, 12)] and out["n_samples"] == 3


def test_adaptive_config_validation(monkeypatch):
    llm = ScriptLLM()
    base = {"route": {"score": "consistency", "adaptive": {"enabled": True}}}
    with pytest.raises(ValueError, match="gated"):
        make_sys(monkeypatch, PL.FullFixed, llm, C.deep_update(base, {"route": {"consistency": {"k": 4}}}))
    with pytest.raises(ValueError, match="consistency.k"):
        make_sys(monkeypatch, PL.FullGated, llm, C.deep_update(base, {"route": {"consistency": {"k": 0}}, "m5": {"policy": "gated"}}))
    with pytest.raises(ValueError, match="theta_fast"):
        make_sys(monkeypatch, PL.FullGated, llm, C.deep_update(AD, {"route": {"adaptive": {"theta_fast": 1.5}}}))
    with pytest.raises(ValueError, match="tiebreak"):
        make_sys(monkeypatch, PL.FullGated, llm, C.deep_update(AD, {"route": {"adaptive": {"tiebreak": "x"}}}))
    with pytest.raises(ValueError, match="probe_values|adaptive"):
        make_sys(monkeypatch, PL.FullGated, llm, C.deep_update(AD, {"m5": {"gate_empty": True, "probe_values": True}}))
    make_sys(monkeypatch, PL.FullGated, llm, AD)


# ------------------------------------------------------------------------------------------------ 5. legacy byte-identity
def test_switches_default_off_and_legacy_rows_carry_no_x7_keys(monkeypatch, toydb):
    assert DEFAULTS["m5"]["probe_values"] is False and DEFAULTS["route"]["adaptive"]["enabled"] is False
    new_keys = {"probed", "probe_cols", "adaptive_path", "adaptive_cons", "adaptive_n", "adaptive_choice", "adaptive_sizes", "adaptive_tie"}
    llm = ScriptLLM(gen=BAD, rev=GOOD)
    out = make_sys(monkeypatch, PL.FullGated, llm, GATED).run(mk_item(toydb))
    assert not (new_keys & set(out)) and all(not ({"probed", "probe_cols"} & set(c)) for c in out["candidates"])
    llm = ScriptLLM(gen=GOOD, crit=CRIT9)
    out = make_sys(monkeypatch, PL.FullFixed, llm, {}).run(mk_item(toydb))
    assert not (new_keys & set(out)) and llm.tags() == ["gen", "crit"]


def test_x7_configs_load_and_existing_configs_do_not_enable_x7():
    for p in sorted(CONFIGS.glob("*.yaml")):
        cfg = C.load_config(str(p))
        m5, route = cfg.get("m5") or {}, cfg.get("route") or {}
        if p.stem.startswith(("full_gated_probe", "full_gated_empty", "full_gated_adaptive")):
            continue
        assert not m5.get("probe_values") and not (route.get("adaptive") or {}).get("enabled"), p.name
    pr = C.load_config(str(CONFIGS / "full_gated_probe.yaml"))
    assert pr["ddl"]["quote_keywords"] is True and pr["m5"]["probe_values"] is True and pr["m5"]["gate_empty"] is True and pr["m5"]["policy"] == "gated"
    ad = C.load_config(str(CONFIGS / "full_gated_adaptive.yaml"))
    assert ad["route"]["adaptive"]["enabled"] is True and ad["route"]["adaptive"]["theta_fast"] == 1.0 and ad["route"]["consistency"]["k"] == 4
    # the adaptive config shares root and sampling prompts (and thus the cache keys) with the X5 consistency run
    x5 = C.load_config(str(CONFIGS / "full_gated_cons.yaml"))
    assert ad["ddl"] == x5["ddl"] and ad["route"]["consistency"] == x5["route"]["consistency"] and ad["temperature"] == x5["temperature"]
    pn = C.load_config(str(CONFIGS / "full_gated_probe_near.yaml"))
    assert pn["m5"]["probe_near"] is True and pn["m5"]["probe_values"] is True and pn["ddl"]["quote_keywords"] is True
    for name in ("full_gated_probe", "full_gated_probe_zero", "full_gated_probe_near", "full_gated_adaptive", "full_gated_empty"):
        PL.Pipeline(object(), "m", C.load_config(str(CONFIGS / (name + ".yaml"))), resources=object())      # validation only: constructing must not raise


# ------------------------------------------------------------------------------------------------ 6. offline tools
import adaptive_replay as AR  # noqa: E402
import empty_roots as ER  # noqa: E402
import value_check as VC  # noqa: E402
from nl2sql.data import Item  # noqa: E402


def test_value_check_literal_extraction_and_text_classes():
    assert VC.extract_literals("SELECT a FROM t WHERE x = 'Foo' AND y IN ('a','b') AND z LIKE '%bar%' AND n > 5 AND m = 2.50 HAVING count(*) > 3") == \
        [("Foo", False), ("a", False), ("b", False), ("bar", False), ("5", True), ("2.50", True), ("3", True)]
    assert VC.extract_literals("SELECT 'sel' FROM t WHERE x = 1") == [("1", True)]                 # select-list literal is not a filter
    assert VC.extract_literals("SELEC bad 'q' 7") == [("q", False), ("7", True)]                   # regex fallback
    text = "How many Female users in New-York state? Evidence: year 2013, 1,000 users, gender = female"
    c = VC.classify_literal
    assert c("Female", False, text) == "verbatim" and c("female", False, text) == "verbatim"
    assert c("FEMALE", False, text) == "case"
    assert c("New York", False, text) == "format" and c("NewYork", False, text) == "format"
    assert c("users", False, "user list") == "stem" and c("Berlin", False, "Berlni city") == "fuzzy"
    assert c("F", False, "female users") == "case"                                                  # substring ignoring case counts as approximate text
    assert c("Y", False, "was it a hit?") == "absent" and c("xyz", False, text) == "absent"
    assert c("2013", True, text) == "verbatim" and c("1000", True, text) == "number" and c("201", True, text) == "absent" and c("7", True, text) == "absent"
    assert VC.present_in("Foo", False, [("Foo", False)]) and VC.present_in("bar", False, [("%bar%", False)]) and VC.present_in("5", True, [("5.0", True)])
    assert not VC.present_in("Foo", False, [("foo", False)]) and not VC.present_in("5", True, [("6", True)])


def test_value_check_analyse_row_levels_and_summary(pdb):
    s = sch(pdb)
    it = Item(qid="toy:t:1", db_id="t", question="How many staff work in sales?", evidence="", gold_sql="SELECT count(*) FROM emp WHERE dept = 'Sales' AND code = 'C1'",
              db_path=pdb, split="toy", meta={})
    a = VC.analyse_row(it, "SELECT count(*) FROM emp WHERE dept = 'sales'", s)
    assert a["level"] == "absent" and a["pure_value_error"] is False                                 # 'C1' absent; the column set differs (code missing)
    lits = {l["lit"]: l for l in a["gold_literals"]}
    assert lits["Sales"]["present"] is False and lits["Sales"]["text_class"] == "case" and lits["Sales"]["exists_in_gold_col"] is True
    assert lits["C1"]["text_class"] == "absent" and lits["C1"]["exists_in_gold_col"] is True and lits["C1"]["n_distinct"] == 12
    it2 = Item(qid="toy:t:2", db_id="t", question="How many in sales?", evidence="dept = Sales", gold_sql="SELECT count(*) FROM emp WHERE dept = 'Sales'",
               db_path=pdb, split="toy", meta={})
    a = VC.analyse_row(it2, "SELECT count(*) FROM emp WHERE dept = 'sales'", s)
    assert a["level"] == "verbatim_only" and a["pure_value_error"] is True
    a = VC.analyse_row(it2, "SELECT count(*) FROM emp WHERE dept = 'Sales'", s)
    assert a["level"] == "none_missing" and a["pure_value_error"] is False
    it3 = Item(qid="toy:t:3", db_id="t", question="How many?", evidence="", gold_sql="SELECT count(*) FROM emp", db_path=pdb, split="toy", meta={})
    assert VC.analyse_row(it3, "SELECT 1", s)["level"] == "no_gold_literal"
    rows = [VC.analyse_row(it, "SELECT count(*) FROM emp WHERE dept = 'sales'", s), VC.analyse_row(it2, "SELECT count(*) FROM emp WHERE dept = 'sales'", s)]
    sm = VC.summarise(rows, 10)
    assert sm["levels"] == {"absent": 1, "verbatim_only": 1} and sm["upper_bound_value_grounding"]["n"] == 1 and sm["upper_bound_value_grounding"]["pp_of_all_questions"] == 10.0
    assert sm["upper_bound_value_grounding"]["retrievable_by_lookup_n"] == 1 and sm["absent_literals"]["card_le_20"] == 1 and sm["pure_value_error"] == 1


def test_empty_roots_analysis_on_a_toy_run(tmp_path, monkeypatch, pdb):
    from nl2sql import data
    from nl2sql.eval import score
    s = sch(pdb)
    items = {
        "toy:t:1": Item(qid="toy:t:1", db_id="t", question="q1", evidence="", gold_sql="SELECT id FROM emp WHERE dept = 'Sales'", db_path=pdb, split="bird_train_HR", meta={}),
        "toy:t:2": Item(qid="toy:t:2", db_id="t", question="q2", evidence="", gold_sql="SELECT id FROM emp WHERE id > 1000", db_path=pdb, split="bird_train_HR", meta={}),
        "toy:t:3": Item(qid="toy:t:3", db_id="t", question="q3", evidence="", gold_sql="SELECT id FROM emp WHERE name = 'zz'", db_path=pdb, split="bird_train_HR", meta={}),
    }
    run = tmp_path / "run"
    run.mkdir()
    sqls = {"toy:t:1": "SELECT id FROM emp WHERE dept = 'sales'", "toy:t:2": "SELECT id FROM emp WHERE name = 'zz'", "toy:t:3": "SELECT id FROM emp WHERE dept = 'zz'"}
    with open(run / "preds.jsonl", "w") as f, open(run / "cands.jsonl", "w") as g:
        for q, sql in sqls.items():
            ex_root = 0 if q == "toy:t:1" else 1           # q2, q3: gold is empty too -> the empty root is right
            f.write(json.dumps({"qid": q, "db_id": "t", "split": "bird_train_HR", "ex": ex_root, "iterations": 0}) + "\n")
            g.write(json.dumps({"qid": q, "cand_idx": 0, "sql": sql, "status": "ok_empty", "ex": ex_root, "n_rows": 0, "n_cols": 1}) + "\n")
        f.write(json.dumps({"qid": "toy:t:4", "db_id": "t", "split": "bird_train_HR", "ex": 0, "iterations": 0}) + "\n")
        g.write(json.dumps({"qid": "toy:t:4", "cand_idx": 0, "sql": "SELECT count(*) FROM emp WHERE name = 'zz'", "status": "ok_nonempty", "ex": 0, "n_rows": 1, "n_cols": 1}) + "\n")
    items["toy:t:4"] = Item(qid="toy:t:4", db_id="t", question="q4", evidence="", gold_sql="SELECT count(*) FROM emp", db_path=pdb, split="bird_train_HR", meta={})
    monkeypatch.setattr(ER, "load_items", lambda split: items)
    monkeypatch.setattr(data, "get_schema", lambda it: s)

    class FakeEv:
        def __init__(self, *a, **k):
            pass

        def gold_exec(self, item):
            r = db.execute(item.db_path, item.gold_sql, max_rows=None)
            return {"ok": r["ok"], "rows": r["rows"]}
    monkeypatch.setattr(score, "Evaluator", FakeEv)
    r = ER.analyse_run("toy", str(run), 8)
    assert r["n_empty_roots"] == 3 and r["gold_nonempty"] == 1 and r["gold_empty"] == 2 and r["root_right"] == 2 and r["wrong_with_gold_nonempty"] == 1
    row = [x for x in r["rows"] if x["qid"] == "toy:t:1"][0]
    assert row["cause"] == "same_column_other_value" and row["format_only"] is True and row["probe_shown"] and row["probe_visible"] and row["probe_complete"]
    assert row["gold_literal_exists_in_root_column"] is True and r["probe_visible"] == 1 and r["probe_visible_complete_list"] == 1
    assert r["zero_aggregate_roots"] == {"n": 1, "root_wrong": 1, "wrong_with_nonzero_gold": 1}
    assert ER.is_empty([]) and ER.is_empty([(None, None)]) and not ER.is_empty([(0,)]) and not ER.is_empty([(None,), (None,)])


def test_adaptive_replay_map_samples_and_costs():
    from nl2sql.util import strip_sql
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "# Schema ... Question: how many things? Evidence: none"}]
    other = [{"role": "system", "content": "s"}, {"role": "user", "content": "Question: second one"}]
    arch = [{"messages": msgs, "params": {"temperature": 0.0, "max_tokens": 1024}, "text": "```sql\nSELECT 1\n```"},
            {"messages": msgs, "params": {"temperature": 0.7, "n": 4, "seed": 1}, "text": ["SELECT 1", "SELECT 2", "```sql\nSELECT 1;\n```", "SELECT 3"]},
            {"messages": other, "params": {"temperature": 0.0, "max_tokens": 1024}, "text": "SELECT 1"},
            {"messages": other, "params": {"temperature": 0.7, "n": 4, "seed": 1}, "text": ["SELECT 9"] * 4},
            {"messages": [{"role": "user", "content": "crit"}], "params": {"temperature": 0.0}, "text": "{}"}]
    roots = {"q1": {"sql": "SELECT 1"}, "q2": {"sql": "SELECT 1"}}
    items = {"q1": Item(qid="q1", db_id="d", question="how many things?", evidence="", gold_sql="", db_path="", split="x", meta={}),
             "q2": Item(qid="q2", db_id="d", question="second one", evidence="", gold_sql="", db_path="", split="x", meta={})}
    out, info = AR.map_samples(arch, roots, items)
    assert out["q1"] == [strip_sql(x) for x in arch[1]["text"]] and out["q2"] == ["SELECT 9"] * 4 and info == {"sampling_requests": 2, "unmatched": 0, "ambiguous": 0}
    row = {"stages": {"m1": {"calls": 1, "prompt": 200, "completion": 100}, "gen": {"calls": 1, "prompt": 3000, "completion": 50},
                      "cons": {"calls": 1, "prompt": 3000, "completion": 200}, "crit": {"calls": 1, "prompt": 3000, "completion": 10}, "rev": {"calls": 2, "prompt": 10, "completion": 10}}}
    assert AR.stage_cost(row, ("m1", "gen", "cons")) == (3, 6200, 350)
    assert AR.stage_cost(row, ("m1", "gen", "cons"), k=2) == (3, 6200, 250)            # sampling completion scaled by k / 4
    assert AR.stage_cost(row, ("m1", "gen", "rev", "rev_t2")) == (4, 3210, 160)


def test_adaptive_replay_policy_table_matches_the_live_decision(tmp_path, monkeypatch):
    import numpy as np

    def cand(key, ex, empty=False):
        return {"ok": key is not None, "key": key, "empty": empty, "ex": ex, "err": None}
    # 4 questions of 2 databases; samples = 4 each
    rows = {
        "q1": {"root": cand("A", 1), "samples": [cand("A", 1)] * 4},                                    # all agree -> fast, right
        "q2": {"root": cand("A", 0), "samples": [cand("B", 1)] * 3 + [cand("A", 0)]},                   # vote flips to the right majority
        "q3": {"root": cand("A", 1), "samples": [cand("A", 1), cand("A", 1), cand("A", 1), cand("B", 0)]},   # 3/4: fast at 0.75, vote keeps the root at 1.0
        "q4": {"root": cand("E", 0, True), "samples": [cand("E", 0, True), cand("E", 0, True), cand("C", 1), cand("C", 1)]},   # empty root, tie 3:2... see below
    }
    preds = {q: {"qid": q, "db_id": "d1" if q in ("q1", "q2") else "d2", "ex": e, "calls": 4, "split": "bird_train_HR",
                 "stages": {"m1": {"calls": 1, "prompt": 100, "completion": 10}, "gen": {"calls": 1, "prompt": 1000, "completion": 20},
                            "cons": {"calls": 1, "prompt": 1000, "completion": 80}, "crit": {"calls": 1, "prompt": 1000, "completion": 5}}}
             for q, e in (("q1", 1), ("q2", 0), ("q3", 1), ("q4", 0))}
    roots = {q: {"sql": "x", "ex": rows[q]["root"]["ex"]} for q in rows}
    monkeypatch.setattr(AR, "collect", lambda cons_dir, procs, cache: ({"rows": rows, "info": {}, "split": "bird_train_HR"}, preds, roots, "bird_train_HR"))

    class A:
        ks, thetas, empty, boot, procs, samples_cache = [4], [1.0, 0.75], ["vote", "demote"], 200, 1, None
    res = AR.replay_set("t", "dummy", None, A)
    assert res["n"] == 4 and res["refs"]["root_only"]["ex"] == pytest.approx(0.5) and res["refs"]["gated_same_roots"]["ex"] == pytest.approx(0.5)
    assert res["refs"]["oracle_root_plus_4_samples"]["ex"] == 1.0
    by = {(c["theta_fast"], c["empty"]): c for c in res["configs"]}
    # theta 1.0 / vote: q1 fast ok, q2 -> sample group B (3 vs 2) right, q3 root group 4 vs 1 right, q4 root E (3) vs C (2) -> root wrong  => 3/4
    assert by[(1.0, "vote")]["ex"] == 0.75 and by[(1.0, "vote")]["frac_fast"] == 0.25
    # demote: q4 prefers the non-empty group C -> right => 4/4
    assert by[(1.0, "demote")]["ex"] == 1.0
    # theta 0.75: q3 (3/4) is fast, q2's cons = 1/4 stays vote
    assert by[(0.75, "vote")]["frac_fast"] == 0.5 and by[(0.75, "vote")]["ex"] == 0.75
    assert by[(1.0, "vote")]["calls_per_q"] == 3.0 and res["refs"]["root_only"]["calls_per_q"] == 2.0
    g = by[(1.0, "demote")]["vs_gated_same_roots"]
    assert g["diff_pp"] == pytest.approx(50.0) and g["b_right_a_wrong"] == 2 and g["a_right_b_wrong"] == 0 and g["n_databases"] == 2
    assert "vote_always" in {c["theta_fast"] for c in res["configs"]} and "EX" in AR.table("t", res)


def test_run_py_passes_x7_fields_through():
    from nl2sql import run

    class Sys:
        def run(self, item, seed=0):
            return {"final_sql": "select 1", "abstained": False, "critic": {"score": 10.0}, "probed": 2, "probe_cols": ["t.a", "t.b"],
                    "adaptive_path": "vote", "adaptive_cons": 0.25, "adaptive_n": 4, "adaptive_choice": "sample", "adaptive_sizes": [3, 1, 1],
                    "adaptive_tie": False, "adaptive_v3": {"0": 7.0}}

    class Ev:
        gold = {}

        def score_sql(self, item, sql):
            return {"ex": 1, "exec_ok": True}

    it = Item(qid="x:d:1", db_id="d", question="q", evidence="", gold_sql="", db_path="", split="s", meta={})
    row, _ = run.process_item(Sys(), Ev(), it, 0, {}, {"system": "s", "model": "m"})
    for k in ("probed", "probe_cols", "adaptive_path", "adaptive_cons", "adaptive_n", "adaptive_choice", "adaptive_sizes", "adaptive_tie", "adaptive_v3"):
        assert k in row, k
    json.dumps(row)


# ------------------------------------------------------------------------------------------------ 7. gate_zero (degenerate one-cell zero results)
ZERO_A = 'SELECT count(*) FROM "transaction" WHERE "order" = \'A\''
FIX_CNT = 'SELECT count(*) FROM "transaction" WHERE "order" = \'a\''
ZERO_CFG = {"m5": {"policy": "gated", "gate_zero": True}, "route": {"score": "exec", "theta_h": 7.5}, "ddl": {"quote_keywords": True}}


def test_is_zero_result_and_node_helpers():
    ok = lambda rows: {"ok": True, "rows": rows}                    # noqa: E731
    assert M5.is_zero_result(ok([(0,)])) and M5.is_zero_result(ok([(0.0,)]))
    for rows in ([], [(None,)], [("0",)], [(1,)], [(0, 0)], [(0,), (0,)], [(False,)], [(True,)]):
        assert not M5.is_zero_result(ok(rows)), rows
    assert not M5.is_zero_result({"ok": False, "rows": [], "error": "x"})
    n = {"critic": {"status": "ok_nonempty"}, "res": ok([(0,)])}
    assert M5.gate_node(n, False, True) and not M5.gate_node(n, False, False) and not M5.gate_node(n, True, False)
    assert M5.node_rank(n, True) == 1 and M5.node_rank(n, False) == 2
    e = {"critic": {"status": "ok_empty"}, "res": ok([])}
    assert M5.gate_node(e, True, False) and not M5.gate_node(e, False, True) and M5.node_rank(e, True) == 1 and M5.node_rank({"critic": {"status": "err_syntax"}, "res": {}}, True) == 0


def test_gate_zero_revises_a_zero_count_with_probe_and_replaces_it(monkeypatch, toydb):
    llm = ScriptLLM(gen=ZERO_A, rev=FIX_CNT)
    out = make_probe_sys(monkeypatch, PL.FullGated, llm, C.deep_update(ZERO_CFG, {"m5": {"probe_values": True}}), toydb).run(mk_item(toydb))
    assert llm.tags() == ["gen", "rev"] and out["final_sql"] == FIX_CNT and out["actions"] == ["revise", "accept"]
    rev = llm.calls[1]["user"]
    assert M5.ZERO_FEEDBACK in rev and "returned no rows" not in rev and "'a', 'b', 'c'" in rev and out["probed"] == 1
    # off: a zero count is accepted as before (legacy behaviour)
    llm = ScriptLLM(gen=ZERO_A, rev=FIX_CNT)
    out = make_probe_sys(monkeypatch, PL.FullGated, llm, C.deep_update(ZERO_CFG, {"m5": {"gate_zero": False}}), toydb).run(mk_item(toydb))
    assert llm.tags() == ["gen"] and out["actions"] == ["accept"] and out["final_sql"] == ZERO_A


def test_gate_zero_keeps_a_true_zero_when_the_revision_fails_or_repeats(monkeypatch, toydb):
    llm = ScriptLLM(gen=ZERO_A, rev="SELECT nope FROM nowhere")
    out = make_probe_sys(monkeypatch, PL.FullGated, llm, C.deep_update(ZERO_CFG, {"m5": {"t_max": 2}}), toydb).run(mk_item(toydb))
    assert out["final_sql"] == ZERO_A and out["final_status"] == "ok_nonempty"                  # error node (rank 0) never replaces the executable zero root
    llm = ScriptLLM(gen=ZERO_A, rev=ZERO_A)                                                  # "return the same SQL" = a legitimate zero
    out = make_probe_sys(monkeypatch, PL.FullGated, llm, ZERO_CFG, toydb).run(mk_item(toydb))
    assert out["actions"] == ["revise", "revise_noop", "stop_best"] and out["final_sql"] == ZERO_A


def test_gate_zero_config_and_legacy_rows():
    assert DEFAULTS["m5"]["gate_zero"] is False
    with pytest.raises(ValueError, match="gate_zero"):
        PL.Pipeline(object(), "m", {"m5": {"gate_zero": True}}, resources=object())               # p0 policy
    PL.Pipeline(object(), "m", {"m5": {"policy": "gated", "gate_zero": True, "probe_values": True}, "route": {"score": "exec"}}, resources=object())
    assert M5.exec_feedback_text({"ok": True, "rows": [(0,)]}, "ok_nonempty") == "- no execution problem reported"           # default unchanged


def test_x7_adopt_rule_on_synthetic_runs(tmp_path):
    import x7_adopt as AD

    def write(d, rows):
        (tmp_path / d).mkdir()
        with open(tmp_path / d / "preds.jsonl", "w") as f:
            for q, db_, ex, calls in rows:
                f.write(json.dumps({"qid": q, "db_id": db_, "ex": ex, "calls": calls}) + "\n")
    dbs = ["a", "b", "c", "d"]
    ref, good, bad = [], [], []
    for i, d in enumerate(dbs):
        for j in range(40):
            q = "bird_train:%s:%d" % (d, j)
            ref.append((q, d, int(j % 2 == 0), 2))
            good.append((q, d, 1 if (j % 2 == 0 or j % 5 == 0) else 0, 3))          # extra right answers in every db, never a loss
            bad.append((q, d, 0 if (j % 2 == 0 and d in "ab") else int(j % 2 == 0), 3))   # loses answers in two dbs
    for name, rows in (("ref", ref), ("good", good), ("bad", bad)):
        write(name, rows)
    out = AD.main(["--ref-hr", str(tmp_path / "ref"), "--ref-ht", str(tmp_path / "ref"), "--cand", "good=%s,%s" % (tmp_path / "good", tmp_path / "good"),
                   "--cand", "bad=%s,%s" % (tmp_path / "bad", tmp_path / "bad"), "--boot", "500"])
    g, b = out["good"], out["bad"]
    assert g["verdict"]["adopt"] is True and g["HT"]["ci95_pp"][0] > 0 and g["HR"]["calls_ratio"] == 1.5 and g["verdict"]["hr_dbs_negative"] == 0
    assert b["verdict"]["adopt"] is False and b["verdict"]["hr_sign_consistent"] is False and b["HT"]["diff_pp"] < 0
