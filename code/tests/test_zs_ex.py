"""Tests of the round-3 control `zs_ex` (code/nl2sql/systems/ours/zs_ex.py; internal note R3EXP): the single call of zs_ours plus exactly the few-shot examples that the
root-only arm retrieved. No LLM call, no network: a fake client records the messages. Run on a worker machine: cd code && python -m pytest tests/test_zs_ex.py"""
import collections
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
pytest.importorskip("numpy")

from nl2sql import config as C, data, paths  # noqa: E402
from nl2sql import prompts_util as P  # noqa: E402
from nl2sql.systems import ours  # noqa: E402
from nl2sql.systems.ours import m3, zs_ex  # noqa: E402

Entry = collections.namedtuple("Entry", "question evidence sql")
E1 = Entry("How many schools?", "school = schools table", "SELECT count(*) FROM schools")
E2 = Entry("List the names.", "", "SELECT name FROM t")


class FakeLLM(object):
    def __init__(self):
        self.calls = []

    def chat(self, model, messages, **kw):
        self.calls.append((model, messages, kw))
        return {"text": "```sql\nSELECT 1\n```"}


class FakeItem(object):
    qid, question, evidence = "bird_dev:db:1", "How many?", "ev"


def make(tmp_path, ids):
    run = tmp_path / "rootonly"
    run.mkdir()
    with open(str(run / "preds.jsonl"), "w") as f:
        for q, i in ids.items():
            f.write(json.dumps({"qid": q, "exemplar_ids": i}) + "\n")
    model = "qwen3.7-plus-2026-05-26"
    key = C.model_key(model)
    cfg = {"temperature": 0.0, "ddl": {"quote_keywords": True}, "m6": {"path": "/nonexistent/store.sqlite"}, "models": {"embed": "x"},
           "zs_ex": {"sources": {key: {"0": str(run)}}}}          # absolute: os.path.join(RUNS, abs) = abs
    llm = FakeLLM()
    s = zs_ex.ZeroShotExamples(llm, model, cfg)
    s._corpus = type("Corp", (), {"entries": {"a": E1, "b": E2}})()          # bypass the frozen store
    return s, llm


def test_registered_under_the_name_zs_ex():
    import inspect
    src = inspect.getsource(ours)
    assert '"zs_ex": ZeroShotExamples' in src
    assert zs_ex.ZeroShotExamples.name == "zs_ex"


def test_examples_block_is_the_pipeline_format():
    ctx = m3.Context(ddl="DDL", examples=[E1, E2])
    sec = ctx.sections()
    blk = zs_ex.examples_block([E1, E2])
    assert sec == "# Schema\nDDL\n\n" + blk
    assert "Evidence: school = schools table\n" in blk and "Question: List the names.\nSQL: SELECT name FROM t" in blk       # no Evidence line for an entry without evidence
    assert zs_ex.examples_block([]) == ""


def test_prompt_is_zs_prompt_plus_examples_only(tmp_path, monkeypatch):
    s, llm = make(tmp_path, {"bird_dev:db:1": ["a", "b"]})
    sc = {"ddl": "RAW DDL"}
    monkeypatch.setattr(data, "get_schema", lambda item: sc)
    monkeypatch.setattr(data, "render_ddl", lambda sc_, quote_keywords=False: "DDL<%s>" % quote_keywords)
    msgs = s._req(FakeItem(), 0)
    base = P.render("gen_zs.user", schema="DDL<True>", question="How many?", evidence="ev", quote_rule=P.QUOTE_RULE)
    with_ex = P.render("gen_zs.user", schema="DDL<True>\n\n" + zs_ex.examples_block([E1, E2]), question="How many?", evidence="ev", quote_rule=P.QUOTE_RULE)
    assert msgs[1]["content"] == with_ex and msgs[1]["content"] != base
    assert msgs[0] == {"role": "system", "content": P.render("common.system")}
    # the examples are the only difference: removing the block gives the zs prompt
    assert msgs[1]["content"].replace("\n\n" + zs_ex.examples_block([E1, E2]), "") == base


def test_order_of_examples_follows_the_stored_ids(tmp_path, monkeypatch):
    s, _ = make(tmp_path, {"bird_dev:db:1": ["b", "a"]})
    monkeypatch.setattr(data, "get_schema", lambda item: {"ddl": "D"})
    monkeypatch.setattr(data, "render_ddl", lambda sc_, quote_keywords=False: "D")
    u = s._req(FakeItem(), 0)[1]["content"]
    assert u.index("List the names.") < u.index("How many schools?")


def test_missing_ids_is_an_error_not_a_silent_zero_shot(tmp_path, monkeypatch):
    s, _ = make(tmp_path, {"bird_dev:db:1": []})
    monkeypatch.setattr(data, "get_schema", lambda item: {"ddl": "D"})
    with pytest.raises(ValueError):
        s._req(FakeItem(), 0)
    with pytest.raises(ValueError):
        s._req(FakeItem(), 7)              # no source configured for seed 7


def test_run_returns_the_example_ids_and_one_call(tmp_path, monkeypatch):
    s, llm = make(tmp_path, {"bird_dev:db:1": ["a", "b"]})
    monkeypatch.setattr(data, "get_schema", lambda item: {"ddl": "D"})
    monkeypatch.setattr(data, "render_ddl", lambda sc_, quote_keywords=False: "D")
    r = s.run(FakeItem(), seed=0)
    assert r["exemplar_ids"] == ["a", "b"] and r["final_sql"] == "SELECT 1" and len(llm.calls) == 1
    assert llm.calls[0][2]["temperature"] == 0.0 and llm.calls[0][2]["seed"] == 0
    assert s.plan_requests(FakeItem(), 0) is None
