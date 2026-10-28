"""X4: runner loop robustness (run.py), env_builder, stream.py, rl/env per-question outcomes, get_schema thread safety. Offline."""
import gzip
import json
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pytest
import yaml

from nl2sql import data, paths, run, stream
from nl2sql.data import Item
from nl2sql.eval import score as S
from nl2sql.llm import LLMClient
from nl2sql.rl import env_builder as EB
from nl2sql.rl.env import A_ACCEPT, A_HUMAN, A_REVISE, TreeSet, evaluate_policy
from nl2sql.rl.features import STATE_DIM

MODEL = "qwen3.7-plus-2026-05-26"


@pytest.fixture()
def toydb(tmp_path):
    p = str(tmp_path / "toy.sqlite")
    c = sqlite3.connect(p)
    c.execute("create table a (id integer primary key, name text)")
    c.executemany("insert into a values (?,?)", [(i, "n%d" % i) for i in range(1, 21)])
    c.commit()
    c.close()
    return p


def toy_items(dbp, n=5, split="bird_train_HT"):
    return [Item(qid="toy:t:%d" % i, db_id="t%d" % (i % 2), question="q%d" % i, evidence="", gold_sql="select id from a", db_path=dbp,
                 split=split, meta={"fold": i % 2}) for i in range(n)]


@pytest.fixture(autouse=True)
def local_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CACHE", tmp_path / "podcache")
    monkeypatch.setattr(S, "_GC", None)
    yield
    S._GC = None


def mock_client_factory(tmp_path):
    def fac(cache_path=None, usage_path=None, key_file=None, max_concurrency_per_model=8, **kw):
        kf = tmp_path / "k.env"
        kf.write_text("QWEN_API_KEY=sk-TEST\n")

        def handler(request):
            return httpx.Response(200, json={"choices": [{"message": {"content": "select id from a"}}],
                                             "usage": {"prompt_tokens": 5, "completion_tokens": 2}})

        c = LLMClient(cache_path, usage_path, key_file=str(kf), max_concurrency_per_model=max_concurrency_per_model)
        c._http = httpx.Client(transport=httpx.MockTransport(handler))
        return c
    return fac


class ChatSystem:
    def __init__(self, llm, model, cfg):
        self.llm, self.model = llm, model

    def run(self, item, seed=0):
        r = self.llm.chat(self.model, [{"role": "user", "content": item.question}], seed=seed, tag="gen")
        out = {"final_sql": r["text"], "critic": None, "actions": ["accept"]}
        if item.qid.endswith(":3"):
            out["confidence"] = np.float32(0.25)           # numpy scalar: used to make json.dumps raise in the writer
        return out


def run_main(tmp_path, monkeypatch, items, bad_qids=()):
    monkeypatch.setattr(run.data, "load_split", lambda name, with_gold=None: items)
    monkeypatch.setattr(run.data, "load_gold", lambda split, run_id="": {i.qid: i.gold_sql for i in items})
    monkeypatch.setattr(run, "get_system", lambda name: ChatSystem)
    monkeypatch.setattr(run, "LLMClient", mock_client_factory(tmp_path))
    real = S.Evaluator.score_sql

    def flaky(self, item, sql, pred_res=None):
        if item.qid in bad_qids:
            raise sqlite3.DataError("string or blob too big")
        return real(self, item, sql, pred_res)

    monkeypatch.setattr(S.Evaluator, "score_sql", flaky)
    cfgp = tmp_path / "cfg.yaml"
    cfgp.write_text(yaml.safe_dump({"temperature": 0.0}))
    out = tmp_path / "out"
    run.main(["--system", "fake", "--split", "bird_train_HT", "--model", "primary_actor", "--config", str(cfgp), "--out", str(out),
              "--exp", "t", "--workers", "3", "--cache", str(tmp_path / "callcache"), "--no-tuning-log"])
    return out


def test_runner_survives_score_exception_numpy_values_and_archives(tmp_path, monkeypatch, capsys, toydb):
    items = toy_items(toydb)
    out = run_main(tmp_path, monkeypatch, items, bad_qids={"toy:t:2"})
    rows = [json.loads(l) for l in open(out / "preds.jsonl")]
    assert sorted(r["qid"] for r in rows) == sorted(i.qid for i in items)           # no result dropped
    by = {r["qid"]: r for r in rows}
    assert by["toy:t:2"]["error_class"].startswith("sys_error:score:") and by["toy:t:2"]["ex"] is None
    assert by["toy:t:0"]["ex"] == 1 and by["toy:t:3"]["confidence"] == 0.25         # numpy float coerced, row written
    assert (out / "preds.jsonl.failed1").exists()                                   # retried once, failed again
    summ = json.load(open(out / "summary.json"))
    assert summ["sys_error"] == 1 and summ["api_fail"] == 0 and "cache_db_errors" in summ["client_stats"]
    txt = capsys.readouterr().out
    assert '"sys_error": 1' in txt and "WARNING" in txt
    arch = out / "llm_cache.jsonl.gz"
    assert arch.exists()
    recs = [json.loads(l) for l in gzip.open(str(arch), "rt")]
    assert len(recs) == 5 and (out / "llm_cache.manifest.json").exists()


def test_runner_resume_does_not_duplicate_and_retries_sys_errors(tmp_path, monkeypatch, toydb):
    items = toy_items(toydb, 3)
    out = run_main(tmp_path, monkeypatch, items, bad_qids={"toy:t:1"})
    monkeypatch.undo()
    out2 = run_main(tmp_path, monkeypatch, items, bad_qids=())                      # same out dir, healthy now
    assert out2 == out
    rows = [json.loads(l) for l in open(out / "preds.jsonl")]
    assert len(rows) == 3 and all(r["error_class"] is None and r["ex"] == 1 for r in rows)


# ------------------------------------------------------------------------------------------------ env_builder
def test_shard_and_done_logic(tmp_path):
    items = list(range(10))
    assert EB.shard_items(items, None) == items
    assert EB.shard_items(items, "1/3") == [1, 4, 7] and sorted(sum((EB.shard_items(items, "%d/3" % k) for k in range(3)), [])) == items
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"qid": "a", "nodes": []}) + "\n" + json.dumps({"qid": "b", "error": "x"}) + "\nnot json\n")
    assert EB.read_done([str(p), str(tmp_path / "missing.jsonl"), None]) == {"a"}


def test_build_one_converts_any_exception(monkeypatch, toydb):
    it = toy_items(toydb, 1)[0]
    monkeypatch.setattr(EB, "build_tree", lambda *a, **k: (_ for _ in ()).throw(sqlite3.DatabaseError("boom")))
    kind, rec = EB.build_one(None, None, it)
    assert kind == "error" and "DatabaseError" in rec["error"] and rec["qid"] == it.qid
    monkeypatch.setattr(EB, "build_tree", lambda *a, **k: {"qid": it.qid, "nodes": [{"x": np.float32(1.0)}]})
    kind, t = EB.build_one(None, None, it)
    assert kind == "tree"


def test_env_builder_main_continues_after_errors_and_shards(tmp_path, monkeypatch, toydb):
    items = toy_items(toydb, 6, split="bird_train_HR")
    monkeypatch.setattr(EB.C, "load_config", lambda p: {})
    monkeypatch.setattr(EB.C, "resolve_model", lambda m: MODEL)
    monkeypatch.setattr(EB, "resolved_cfg", lambda c: c)
    monkeypatch.setattr(EB, "FullFixed", lambda llm, model, cfg: object())
    monkeypatch.setattr(EB, "Evaluator", lambda split, run_id="": None)
    monkeypatch.setattr(EB.data, "load_split", lambda name, with_gold=None: items)

    def bt(sys_, ev, item, seed=0):
        if item.qid == "toy:t:2":
            raise ValueError("unexpected")                    # not an LLMError: used to end the whole result loop
        return {"qid": item.qid, "db_id": item.db_id, "fold": None, "nodes": [{"id": 0, "v": np.float64(1.5)}]}

    monkeypatch.setattr(EB, "build_tree", bt)
    out = tmp_path / "trees" / "trees.jsonl"
    args = ["--config", "x.yaml", "--out", str(out), "--workers", "3", "--cache", str(tmp_path / "cc")]
    EB.main(args)
    done = [json.loads(l)["qid"] for l in open(out)]
    assert sorted(done) == sorted(i.qid for i in items if i.qid != "toy:t:2")
    errs = [json.loads(l) for l in open(str(out) + ".errors.jsonl")]
    assert len(errs) == 1 and errs[0]["qid"] == "toy:t:2" and "ValueError" in errs[0]["error"]
    EB.main(args)                                              # restart: only the failed tree is retried
    assert len(open(out).read().splitlines()) == 5 and len(open(str(out) + ".errors.jsonl").read().splitlines()) == 2
    # sharding: two shards cover everything exactly once (separate outputs)
    for k in range(2):
        EB.main(["--config", "x.yaml", "--out", str(tmp_path / ("s%d" % k) / "trees.jsonl"), "--shard", "%d/2" % k, "--cache", str(tmp_path / "cc")])
    got = sorted(json.loads(l)["qid"] for k in range(2) for l in open(str(tmp_path / ("s%d" % k) / "trees.jsonl")))
    assert got == sorted(i.qid for i in items if i.qid != "toy:t:2")
    assert (tmp_path / "trees" / "usage.jsonl").parent.exists()


# ------------------------------------------------------------------------------------------------ stream.py
def test_pool_corpus_built_once_under_concurrency(monkeypatch):
    calls = []

    def slow_vecs(llm, model, items, name):
        calls.append(name)
        time.sleep(0.2)
        return np.zeros((0, 4), dtype=np.float32)

    monkeypatch.setattr(stream, "_pool_vecs", slow_vecs)
    monkeypatch.setattr(stream.data, "load_split", lambda name, with_gold=None: [])
    monkeypatch.setattr(stream, "_POOL_CACHE", {})
    cfg = {"models": {"embed": "e"}}
    with ThreadPoolExecutor(6) as ex:
        cs = list(ex.map(lambda i: stream._pool_corpus(None, cfg, "static", {}, seed=i), range(6)))
    assert calls == ["bird_train_pool"] and len(cs) == 6 and len({id(c) for c in cs}) == 6


class _FakeCorpus:
    def __init__(self):
        self.entries, self.n = {}, 0

    def add_entry(self, eid, *a, **k):
        self.entries[eid] = type("E", (), {"polluted": False})()

    def on_use(self, *a, **k): pass
    def evict(self): pass
    def _matrix(self): pass
    def stats(self): return {"n": len(self.entries)}


class _FakeSys:
    boom_at = None
    calls = 0

    def __init__(self, *a, **k):
        self.pipe = type("P", (), {})()

    def run(self, item, seed=0, corpus=None):
        _FakeSys.calls += 1
        if _FakeSys.boom_at is not None and _FakeSys.calls == _FakeSys.boom_at:
            raise RuntimeError("simulated crash")
        return {"final_sql": "select 1", "abstained": False, "exemplar_ids": [], "critic": {"score": 8.0}}


class _FakeEv:
    def __init__(self, *a, **k): pass
    def score_sql(self, item, sql): return {"ex": 1}


def _patch_unit(monkeypatch, items):
    monkeypatch.setattr(stream, "stream_split", lambda name: (items, "bird_train_HR", ["t0"]))
    monkeypatch.setattr(stream, "_pool_corpus", lambda *a, **k: _FakeCorpus())
    monkeypatch.setattr(stream, "Resources", lambda llm, cfg: type("R", (), {"db_store": lambda self, it: type("S", (), {"schema": {}})()})())
    monkeypatch.setattr(stream, "FullSystem", _FakeSys)
    monkeypatch.setattr(stream, "Evaluator", _FakeEv)
    monkeypatch.setattr(stream.pollution, "pollute_entries", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(stream, "resolved_cfg", lambda c: c)


def test_e4_unit_restart_rewrites_events_instead_of_duplicating(tmp_path, monkeypatch, toydb):
    items = []
    for i in range(14):
        role = "seed" if i < 3 else ("stream" if i < 9 else "probe")
        items.append(Item(qid="toy:t0:%d" % i, db_id="t0", question="q", evidence="", gold_sql="select 1", db_path=toydb, split="x", meta={"role": role}))
    _patch_unit(monkeypatch, items)
    cfg = {"m6": {"params": {}}}
    ref = tmp_path / "ref"
    _FakeSys.boom_at, _FakeSys.calls = None, 0
    assert stream.run_unit(None, cfg, MODEL, "t0", "ws_full", 0, 1, str(ref), {}, workers=1)[1] == "ok"
    n_ref = len(open(ref / "ws_full-p00-perm1-t0" / "events.jsonl").read().splitlines())
    crash = tmp_path / "crash"
    _FakeSys.calls, _FakeSys.boom_at = 0, 9                  # crash after some events were written
    with pytest.raises(RuntimeError):
        stream.run_unit(None, cfg, MODEL, "t0", "ws_full", 0, 1, str(crash), {}, workers=1)
    partial = len(open(crash / "ws_full-p00-perm1-t0" / "events.jsonl").read().splitlines())
    assert 0 < partial < n_ref
    _FakeSys.boom_at = None
    assert stream.run_unit(None, cfg, MODEL, "t0", "ws_full", 0, 1, str(crash), {}, workers=1)[1] == "ok"
    assert len(open(crash / "ws_full-p00-perm1-t0" / "events.jsonl").read().splitlines()) == n_ref
    assert stream.run_unit(None, cfg, MODEL, "t0", "ws_full", 0, 1, str(crash), {}, workers=1)[1] == "skipped"


def test_ht_ministream_roles(tmp_path):
    try:
        items = stream.ht_ministream_items()
        sp = json.load(open(paths.SPLITS_JSON))
    except (OSError, KeyError, FileNotFoundError):
        pytest.skip("BIRD train data not available here")
    n_db = len(sp["ht_dbs"])
    roles = [i.meta["role"] for i in items]
    assert roles.count("seed") == 12 * n_db and roles.count("stream") == 40 * n_db and roles.count("probe") == 24 * n_db
    assert len({i.qid for i in items}) == len(items) and {i.db_id for i in items} == set(sp["ht_dbs"])
    again = stream.ht_ministream_items()
    assert [(i.qid, i.meta["role"]) for i in items] == [(i.qid, i.meta["role"]) for i in again]      # deterministic draw
    assert all(i.gold_sql for i in items)
    items2, ev_split, dbs = stream.stream_split("HT")
    assert ev_split == "bird_train_HT_full" and dbs == sorted(sp["ht_dbs"])
    with pytest.raises(ValueError):
        stream.stream_split("XX")


# ------------------------------------------------------------------------------------------------ rl/env per-question outcomes
def _tree(qid, db, fold, root_ex, child_ex):
    st = [0.0] * STATE_DIM

    def node(i, parent, action, ex, score, calls):
        return {"id": i, "parent": parent, "action": action, "weight": 1, "state": list(st), "ex": ex, "score": score, "tier": 1,
                "tok_actor": 1000, "tok_crit": 0, "calls": calls}
    return {"qid": qid, "db_id": db, "fold": fold, "nodes": [node(0, None, "root", root_ex, 5.0, 3), node(1, 0, "revise", child_ex, 9.0, 2)]}


def test_evaluate_policy_returns_per_question_outcomes():
    ts = TreeSet([_tree("q1", "A", 0, 0, 1), _tree("q2", "B", 1, 1, 0), _tree("q3", "B", 1, 1, 1)])
    acc = evaluate_policy(ts, lambda s, m: A_ACCEPT)
    rev = evaluate_policy(ts, lambda s, m: A_REVISE if m[A_REVISE] else A_ACCEPT)
    hum = evaluate_policy(ts, lambda s, m: A_HUMAN, c_h=0.25)
    for res in (acc, rev, hum):
        assert [p["qid"] for p in res["per_question"]] == ["q1", "q2", "q3"]
        assert [p["db"] for p in res["per_question"]] == ["A", "B", "B"]
        for k_agg, k_pq in (("ex_all", "ex"), ("calls", "calls"), ("deferral", "deferred"), ("ret", "ret")):
            assert res[k_agg] == pytest.approx(np.mean([p[k_pq] for p in res["per_question"]]))
    assert [p["ex"] for p in acc["per_question"]] == [0, 1, 1] and [p["calls"] for p in acc["per_question"]] == [3, 3, 3]
    assert [p["ex"] for p in rev["per_question"]] == [1, 0, 1] and [p["calls"] for p in rev["per_question"]] == [5, 5, 5]
    assert [p["deferred"] for p in hum["per_question"]] == [1, 1, 1] and hum["ex_all"] == 0
    # subset + pairing on qid
    sub = evaluate_policy(ts, lambda s, m: A_ACCEPT, qidx=[2, 0])
    assert [p["qid"] for p in sub["per_question"]] == ["q3", "q1"]
    assert evaluate_policy(ts, lambda s, m: A_ACCEPT, per_question=False)["per_question"] == []
    assert evaluate_policy(ts, lambda s, m: A_ACCEPT, qidx=[])["per_question"] == []
    json.dumps(rev)                                              # serialisable as is (train.py dumps it)


# ------------------------------------------------------------------------------------------------ data.get_schema
def test_get_schema_concurrent_first_build_has_no_tmp_collision(tmp_path, monkeypatch, toydb):
    monkeypatch.setattr(data, "_SCHEMA_MEM", {})
    it = Item(qid="toy:t:1", db_id="t", question="q", evidence="", gold_sql="", db_path=toydb, split="toy", meta={})
    barrier = threading.Barrier(12)

    def go(_):
        barrier.wait()
        return data.get_schema(it)

    for rnd in range(3):                                         # a fresh cache dir (and memo) each round
        monkeypatch.setattr(paths, "CACHE", tmp_path / ("c%d" % rnd))
        monkeypatch.setattr(data, "_SCHEMA_MEM", {})
        with ThreadPoolExecutor(12) as ex:
            outs = list(ex.map(go, range(12)))
        assert all(o["ddl"] == outs[0]["ddl"] for o in outs) and "CREATE TABLE a" in outs[0]["ddl"]
        files = os.listdir(tmp_path / ("c%d" % rnd) / "schema")
        assert len(files) == 1 and files[0].endswith(".json")          # no leftover *.tmp
        barrier.reset()
