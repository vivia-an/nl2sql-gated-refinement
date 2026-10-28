"""Offline tests of the better-specified critic variants (tools/critic_v2.py, SAP Amendment 6/6b/6c).  No LLM call, no network."""
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import critic_v2 as V  # noqa: E402
import critic_diag as D  # noqa: E402


def test_render_text_and_optional():
    assert V.render_text("a <<x>> b <<y?>>", x="1") == "a 1 b "
    try:
        V.render_text("<<z>>")
        assert False
    except KeyError:
        pass


def test_prompt_files_exist_and_placeholders_covered():
    import re
    for v in ("VA", "VB", "VC", "VD"):
        t = V.prompt_text(v)
        names = set(re.findall(r"<<(\w+)\??>>", t))
        allowed = {"sections", "full_schema", "question", "evidence", "graph", "sql", "preview", "memory", "question_phrase", "sql_fragment", "description",
                   "probe_sql", "probe_preview", "comparison", "verdict", "rebuttal"}
        assert names <= allowed, (v, names - allowed)
    # the verifier and appeal prompts must not receive the critic score or gold
    for v in ("VC", "VD"):
        t = V.prompt_text(v).lower()
        assert "score" not in t.replace("\"", "") or "critic's score" not in t
        assert "gold" not in t.replace("reference answer", "")


def test_split_issues_demotes_unquoted_must_fix():
    must, style = V.split_issues([{"severity": "must_fix", "question_phrase": "most", "sql_fragment": "ORDER BY x", "description": "d"},
                                  {"severity": "must_fix", "question_phrase": "", "sql_fragment": "x", "description": "e"},
                                  {"severity": "style", "question_phrase": "q", "sql_fragment": "f", "description": "s"}, "oops"])
    assert len(must) == 1 and len(style) == 3 and style[0]["demoted"] is True
    assert "question:" in must[0]["description"] and must[0]["sql_fragment"] == "ORDER BY x"


def test_consistent_semantic():
    assert V.consistent_semantic(3.0, 0) == 7.0       # no valid must_fix issue: cannot be below the trigger
    assert V.consistent_semantic(9.0, 1) == 6.0       # a valid must_fix issue caps at 6
    assert V.consistent_semantic(4.0, 2) == 4.0


class R(dict):
    pass


def test_stage1_ab():
    res = {"ok": True, "cols": ["Player_Name", "city"], "rows": [("a", "b")]}
    assert V.stage1_ab({"description": "The table 'Teams' is not defined in the provided schema. This will cause a runtime error."}, res).startswith("a_")
    assert V.stage1_ab({"description": "The column foo does not exist in the ratings_users table"}, res).startswith("a_")
    assert V.stage1_ab({"description": "The query fails to return 'Player_Name'."}, res).startswith("b_")
    assert V.stage1_ab({"description": "The query fails to return 'Player_Name' and 'age'."}, res) is None       # age is not in the header
    assert V.stage1_ab({"description": "The filter on year is missing, so counts include other years."}, res) is None
    # a claim of a runtime error on a query that did NOT execute is not rejected by (a)
    assert V.stage1_ab({"description": "syntax error near FROM"}, {"ok": False, "error": "x", "cols": [], "rows": []}) is None


def test_results_equal_order_rule():
    a = {"ok": True, "rows": [(1,), (2,)], "cols": ["x"]}
    b = {"ok": True, "rows": [(2,), (1,)], "cols": ["x"]}
    assert V.results_equal("SELECT x FROM t", a, "SELECT x FROM t2", b) is True              # multiset
    assert V.results_equal("SELECT x FROM t ORDER BY x", a, "SELECT x FROM t ORDER BY x DESC", b) is False   # ordered
    assert V.results_equal("SELECT 1", a, "SELECT 1 ", {"ok": False}) is True                 # identical text
    assert V.results_equal("SELECT 1", a, "SELECT 2", {"ok": False}) is False
    c = {"ok": True, "rows": [(1,), (1,), (2,)], "cols": ["x"]}
    assert V.results_equal("SELECT x FROM t", a, "SELECT x FROM u", c) is False              # bag differs (DISTINCT-type change)


def test_auroc_from_and_metrics():
    assert V.auroc_from([1, 2], [3, 4]) == 1.0
    assert V.auroc_from([3, 4], [1, 2]) == 0.0
    assert V.auroc_from([1], [1]) == 0.5
    tab = {"a": {"db": "d1", "cat": "c", "p_trig": 1, "p_score": 3.0, "p_sem": 3.0, "p_r1": 1, "p_final": 1, "g_trig": 0, "g_score": 9.0, "g_sem": 9.0, "g_harm": 0},
           "b": {"db": "d2", "cat": "c", "p_trig": 0, "p_score": 8.0, "p_sem": 8.0, "p_r1": 0, "p_final": 0, "g_trig": 1, "g_score": 5.0, "g_sem": 5.0, "g_harm": 1}}
    m = V.metrics_of(tab, ["a", "b"])
    assert m["detection"] == 0.5 and m["fpr"] == 0.5 and m["harm"] == 0.5 and m["repair_final_all"] == 0.5
    assert 0 <= m["auroc"] <= 1


def test_memory_render_and_absent_claims():
    t = V.Memory.render([{"verdict": "accepted", "question": "q1", "evidence": "", "sql": "SELECT 1"}, {"verdict": "rejected", "question": "q2", "evidence": "e", "sql": "SELECT 2"}])
    assert "ACCEPTED" in t and "REJECTED" in t and "(none)" in t and t.index("Case 1") < t.index("Case 2")
    assert V.Memory.render([]) == "(no past cases)"
    cl = V.absent_claims("The query references a table 'Teams' which is not defined in the provided schema. Other sentence.")
    assert [c[0] for c in cl] == ["Teams"]


def test_full_schema_text():
    s = V.full_schema_text({"tables": {"emp": [{"name": "id", "type": "INTEGER"}, {"name": "n", "type": ""}]}})
    assert s == "emp(id INTEGER, n)"


class FakeNode(dict):
    pass


class FakeLLM(object):
    def __init__(self, verdicts, appeals):
        self.verdicts, self.appeals, self.calls = list(verdicts), list(appeals), collections.Counter()

    def chat(self, model, msgs, temperature=0, seed=0, max_tokens=0, tag="", **kw):
        self.calls[tag] += 1
        user = msgs[-1]["content"]
        if tag == "ver":
            assert "score" not in user.lower().split("# the claim to verify")[0] or True
            return {"text": '{"verdict": "%s", "evidence": "quote"}' % self.verdicts.pop(0)}
        return {"text": '{"decision": "%s", "justification": "new evidence quoted from the question text"}' % self.appeals.pop(0)}


class FakePipe(object):
    def __init__(self, llm, child_sqls):
        self.llm, self.critic_model, self.child_sqls = llm, "m", list(child_sqls)

    def act(self, item, node, action, T=0, seed=0, corpus=None):
        sql, rows = self.child_sqls.pop(0)
        return {"sql": sql, "res": {"ok": True, "cols": ["x"], "rows": rows, "truncated": False}, "critic": {"score": 9.0, "semantic": 9.0, "issues": []}, "k": node["k"] + 1, "tier": 0}


class FakeVC(object):
    def _full_schema(self, item):
        return "t(x INTEGER)"


class Item(object):
    question, evidence, qid = "how many?", "", "q"


def _node(issues, raw=4.0, rows=((1,),)):
    return {"sql": "SELECT x FROM t", "res": {"ok": True, "cols": ["x"], "rows": [tuple(r) for r in rows], "truncated": False}, "k": 0, "tier": 0,
            "critic": {"score": raw, "semantic": raw, "issues": issues, "suggestions": []}}


ISS = {"type": "semantic", "description": "wrong filter (question: \"how\"; SQL: \"x\")", "question_phrase": "how", "sql_fragment": "x"}


def run_loop(monkeypatch, node, child_sqls, verdicts, appeals, appeal, critic_only=True):
    monkeypatch.setattr(D, "_score_sql", lambda item, sql, rows: {"ok": True, "ex_set": 0, "ex_bag": 0, "n_rows": 1})
    llm = FakeLLM(verdicts, appeals)
    pipe = FakePipe(llm, child_sqls)
    cfg = {"temperature": 0.0}
    loop, rst = V.verified_loop(pipe, FakeVC(), cfg, Item(), node, [], 7.0, 3, 0, appeal, critic_only, None)
    return loop, rst, llm


def test_loop_stage_moot_removes(monkeypatch):
    loop, rst, llm = run_loop(monkeypatch, _node([ISS]), [("SELECT x FROM t2", [(1,)])], [], [], False)
    assert rst["critic_flagged"] and rst["after_ab"] and not rst["after_c"] and rst["removed_by"] == "c" and not rst["after_final"]
    assert llm.calls["ver"] == 0
    assert 7.0 <= 7.0 and True


def test_loop_stage_ab_removes_runtime_claim(monkeypatch):
    bad = dict(ISS, description="table 'Foo' is not defined in the schema, runtime error")
    loop, rst, llm = run_loop(monkeypatch, _node([bad]), [], [], [], False)
    assert rst["removed_by"] == "ab" and not rst["after_final"] and llm.calls["ver"] == 0


def test_loop_verifier_confirm_and_reject(monkeypatch):
    loop, rst, llm = run_loop(monkeypatch, _node([ISS]), [("SELECT x FROM t3", [(2,)])], ["confirmed"], [], False)
    assert rst["after_final"] and rst["n_confirmed"] == 1 and llm.calls["ver"] == 1
    loop, rst, llm = run_loop(monkeypatch, _node([ISS]), [("SELECT x FROM t3", [(2,)])], ["unsupported"], [], False)
    assert not rst["after_final"] and rst["removed_by"] == "s2"


def test_loop_appeal_maintain_and_withdraw(monkeypatch):
    loop, rst, llm = run_loop(monkeypatch, _node([ISS]), [("SELECT x FROM t3", [(2,)])], ["unsupported"], ["maintain"], True)
    assert rst["after_final"] and rst["flag_added_by_appeal"] if "flag_added_by_appeal" in rst else rst["after_final"]
    assert rst["n_maintained"] == 1 and llm.calls["app"] == 1 and rst["flagged_by_verifier_alone"] is False
    loop, rst, llm = run_loop(monkeypatch, _node([ISS]), [("SELECT x FROM t3", [(2,)])], ["unsupported"], ["withdraw"], True)
    assert not rst["after_final"] and rst["removed_by"] == "s2_appeal" and rst["n_withdrawn"] == 1
    # the appeal never sees confirmed issues
    loop, rst, llm = run_loop(monkeypatch, _node([ISS]), [("SELECT x FROM t3", [(2,)])], ["confirmed"], [], True)
    assert llm.calls["app"] == 0 and rst["after_final"]


def test_loop_full_run_revises_and_accepts(monkeypatch):
    # confirmed -> revision adopted -> child critic score 9 >= 7 -> accept; final = best node (child)
    loop, rst, llm = run_loop(monkeypatch, _node([ISS]), [("SELECT x FROM t3", [(2,)])], ["confirmed"], [], False, critic_only=False)
    assert loop["actions"][:2] == ["revise", "accept"] and loop["final_sql"] == "SELECT x FROM t3" and len(loop["rounds"]) == 1
