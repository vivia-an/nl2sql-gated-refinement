"""MEMOFIX regression tests (offline: stub LLM, toy sqlite database; no API call, no dev data).
MIN/M2/memosql-s0 failed on all 500 items because the runner's cwd is <repo>/code while memory_path is repo-relative
("exp/frozen/memo_memory_v1.json"): memo_sql opened it with a bare open() -> FileNotFoundError -> error_class "system_error" on
every row, hidden from the run summary (sys_error 0).  Two fixes are covered: path resolution, and the runner counting system_error."""
import hashlib
import json
import os

import numpy as np
import pytest

from nl2sql import data, paths, run
from nl2sql.systems.baselines import _common as BC, memo_sql as MEMO
from test_x5 import mk_item, toydb  # noqa: F401  (toydb is a fixture)

MEM_REL = "exp/frozen/memo_memory_v1.json"


class StubLLM:
    def chat(self, model, messages, *, temperature=0.0, seed=None, max_tokens=1024, n=1, tag="", **kw):
        t = '```sql\nSELECT count(*) FROM "transaction"\n```'
        if tag == "memo.decompose":
            t = '```json\n["count rows"]\n```'
        elif tag == "memo.reflect":
            t = "OK"
        elif tag == "memo.critic":
            t = '```json\n{"correct": true, "error_types": [], "suggestion": ""}\n```'
        return {"text": t if n == 1 else [t] * n, "prompt_tokens": 1, "completion_tokens": 1, "latency_s": 0.0}

    def embed(self, texts, model):
        return [np.frombuffer(hashlib.sha256(t.encode()).digest(), dtype=np.uint8).astype(np.float32).tolist() for t in texts]


@pytest.fixture()
def foreign_cwd(tmp_path, monkeypatch):
    d = tmp_path / "elsewhere"
    d.mkdir()
    monkeypatch.chdir(d)
    MEMO._MEM.clear()
    yield d
    MEMO._MEM.clear()


def test_frozen_memory_file_is_repo_relative_and_exists():
    assert (paths.REPO / MEM_REL).is_file()


def test_resolve_repo_path_is_cwd_independent(foreign_cwd, tmp_path):
    assert not os.path.exists(MEM_REL)                          # premise: a bare open(MEM_REL) fails from this cwd (the original bug)
    with pytest.raises(FileNotFoundError):
        open(MEM_REL)
    assert MEMO.resolve_repo_path(MEM_REL) == str(paths.REPO / MEM_REL)
    absf = str(tmp_path / "x.json")
    assert MEMO.resolve_repo_path(absf) == absf                 # absolute paths are untouched
    assert MEMO.resolve_repo_path("no/such/file.json") == "no/such/file.json"     # unresolvable: unchanged (the error names the config value)


def test_get_memory_loads_the_frozen_store_from_any_cwd(foreign_cwd):
    mem = MEMO.get_memory(StubLLM(), "emb", MEM_REL, None)
    assert len(mem.entries) > 0 and mem.emb.shape[0] == len(mem.entries)


def test_memosql_light_runs_from_a_foreign_cwd_with_the_frozen_config(foreign_cwd, toydb, monkeypatch):
    sch = data._build_schema(toydb, "toy", "t")
    monkeypatch.setattr(BC, "get_schema_cached", lambda item: sch)
    monkeypatch.setattr(MEMO, "get_pool_emb", lambda llm, cfg, model: ([], np.zeros((0, 32), np.float32)))
    import yaml
    cfg = yaml.safe_load(open(paths.REPO / "exp/configs/frozen/memosql_light.yaml"))
    assert cfg["memory_path"] == MEM_REL
    s = MEMO.MemoSQL(StubLLM(), "m", cfg)
    row = s.run(mk_item(toydb), seed=0)
    assert row["error_class"] is None, row["extra"].get("error")
    assert row["final_sql"] and row["calls"] > 0 and row["extra"]["memory"] is True


def test_runner_turns_a_system_error_row_into_a_counted_requeueable_sys_error(toydb):
    class Ev:
        gold = {"toy:t:1": "select 1"}

        def score_sql(self, item, sql, pred_res=None):
            return {"ex": 0, "exec_ok": False, "soft_f1": 0.0, "jaccard": 0.0, "overlap": 0.0, "auto_class": "exec_error:other"}

    class BadSys:
        def run(self, item, seed=0):
            return {"final_sql": "", "error_class": "system_error", "extra": {"error": "FileNotFoundError: x.json", "trace": "tb"}}

    class OkSys:
        def run(self, item, seed=0):
            return {"final_sql": "select 1", "error_class": None}

    row, _ = run.process_item(BadSys(), Ev(), mk_item(toydb), 0, {}, {"system": "memosql", "model": "m"})
    assert str(row["error_class"]).startswith("sys_error") and "FileNotFoundError: x.json" in row["err_msg"]
    row2, _ = run.process_item(OkSys(), Ev(), mk_item(toydb), 0, {}, {"system": "memosql", "model": "m"})
    assert row2["error_class"] is None
    # api_fail rows of a baseline keep their own class (SAP 10: api_fail counts as a wrong answer, not a sys_error)
    class ApiSys:
        def run(self, item, seed=0):
            return {"final_sql": "", "error_class": "api_fail"}
    row3, _ = run.process_item(ApiSys(), Ev(), mk_item(toydb), 0, {}, {"system": "memosql", "model": "m"})
    assert row3["error_class"] == "api_fail"
