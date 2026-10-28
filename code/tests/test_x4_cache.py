"""X4: cache layout, self-healing sqlite, per-run archive. Offline: fake HTTP transport, no LLM calls."""
import gzip
import json
import multiprocessing
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from nl2sql import cache_archive as CA
from nl2sql import sqlcache
from nl2sql.llm import LLMClient, _Shards

MODEL = "qwen3.7-plus-2026-05-26"
CODE = os.path.join(os.path.dirname(__file__), "..")


class Fake:
    def __init__(self):
        self.n = 0

    def __call__(self, request):
        self.n += 1
        body = json.loads(request.content)
        txt = "SELECT %d" % (len(json.dumps(body["messages"])) % 97)
        return httpx.Response(200, json={"choices": [{"message": {"content": txt}}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 0}}})


def make(tmp_path, fake=None, cache="cache", usage="usage.jsonl"):
    kf = tmp_path / "k.env"
    kf.write_text("QWEN_API_KEY=sk-SECRET-TEST\n")
    c = LLMClient(str(tmp_path / cache), str(tmp_path / usage), key_file=str(kf), backoff_base_s=0.01, backoff_cap_s=0.02)
    c._http = httpx.Client(transport=httpx.MockTransport(fake or Fake()))
    return c


def msgs(i):
    return [{"role": "user", "content": "q%d" % i}]


# ------------------------------------------------------------------------------------------------ layout
def test_default_cache_is_pod_local_and_env_overrides():
    env = {k: v for k, v in os.environ.items() if k != "NL2SQL_CACHE"}
    env["PYTHONPATH"] = CODE
    out = subprocess.check_output([sys.executable, "-c", "from nl2sql import paths; print(paths.CACHE)"], env=env, universal_newlines=True).strip()
    assert out == os.path.join(os.path.expanduser("~"), ".cache", "nl2sql")
    env["NL2SQL_CACHE"] = "/tmp/x4_override_cache"
    out = subprocess.check_output([sys.executable, "-c", "from nl2sql import paths; print(paths.CACHE)"], env=env, universal_newlines=True).strip()
    assert out == "/tmp/x4_override_cache"


def test_cache_keys_unchanged_by_x4():
    # registered configs must keep hitting their caches: key = sha256(model|canon(params)|canon(messages)); pin one value
    k = LLMClient.cache_key("m", LLMClient._params(0.0, 0, 100, False, 1), [{"role": "user", "content": "a"}])
    import hashlib
    from nl2sql.util import canon_json
    h = hashlib.sha256()
    h.update(b"m\x1f" + canon_json({"temperature": 0.0, "seed": 0, "top_p": 1.0, "max_tokens": 100, "extra_body": {"enable_thinking": False}}).encode()
             + b"\x1f" + canon_json([{"role": "user", "content": "a"}]).encode())
    assert k == h.hexdigest()


def test_wal_mode_on_pod_local_file(tmp_path):
    c = make(tmp_path)
    c.chat(MODEL, msgs(1), seed=0)
    f = [p for p in os.listdir(tmp_path / "cache") if p.startswith("calls_")][0]
    import sqlite3
    con = sqlite3.connect(str(tmp_path / "cache" / f))
    assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    con.close()


# ------------------------------------------------------------------------------------------------ corruption handling
def test_startup_quarantines_malformed_shard_and_run_continues(tmp_path):
    c0 = make(tmp_path)
    keys = []
    for i in range(30):
        c0.chat(MODEL, msgs(i), seed=0)
        keys.append(LLMClient.cache_key(MODEL, LLMClient._params(0.0, 0, 2048, False, 1), msgs(i)))
    c0.close()
    shard = tmp_path / "cache" / "calls_03.sqlite"
    assert shard.exists()
    for suf in ("-wal", "-shm"):
        try:
            os.remove(str(shard) + suf)
        except OSError:
            pass
    shard.write_bytes(b"this is not a sqlite database " * 400)
    before = sqlcache.stats()["quarantined"]
    fake = Fake()
    c = make(tmp_path, fake)
    assert sqlcache.stats()["quarantined"] == before + 1
    assert any(".corrupt-" in p for p in os.listdir(tmp_path / "cache"))
    for i in range(30):                               # all still answerable; rows of the bad shard are re-issued, nothing raises
        r = c.chat(MODEL, msgs(i), seed=0)
        assert r["text"].startswith("SELECT")
    lost = sum(1 for k in keys if int(k[0], 16) % 16 == 3)
    assert fake.n == lost                             # only the quarantined shard's rows needed the API again


def test_runtime_corruption_is_a_miss_not_an_error(tmp_path):
    fake = Fake()
    c = make(tmp_path, fake)
    for i in range(20):
        c.chat(MODEL, msgs(i), seed=0)
    n0 = fake.n
    for p in os.listdir(tmp_path / "cache"):          # wreck every shard while the client is alive
        if p.endswith(".sqlite"):
            with open(str(tmp_path / "cache" / p), "r+b") as f:
                f.write(b"\x00garbage garbage garbage " * 600)
            for suf in ("-wal", "-shm"):
                try:
                    os.remove(str(tmp_path / "cache" / p) + suf)
                except OSError:
                    pass

    def work(i):                                       # fresh threads -> fresh connections hit the wrecked files
        return c.chat(MODEL, msgs(i), seed=0)["text"]

    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(work, range(20)))
    assert all(o.startswith("SELECT") for o in outs)
    assert fake.n >= n0                                # re-issued where needed, no exception
    s = sqlcache.stats()
    assert s["quarantined"] >= 1 or s["db_errors"] >= 1


def test_garbage_row_is_dropped_and_replaced(tmp_path):
    fake = Fake()
    c = make(tmp_path, fake)
    key = LLMClient.cache_key(MODEL, LLMClient._params(0.0, 0, 2048, False, 1), msgs(7))
    c.cache.put(key, MODEL, {"model": MODEL}, "{not json", 1, 1, 0, 0.0, "realtime")
    r = c.chat(MODEL, msgs(7), seed=0)
    assert not r["cached"] and fake.n == 1 and r["text"].startswith("SELECT")
    r2 = c.chat(MODEL, msgs(7), seed=0)
    assert r2["cached"] and fake.n == 1               # the good row replaced the bad one


def test_cache_put_failure_does_not_fail_the_call(tmp_path, monkeypatch):
    c = make(tmp_path)

    def boom(*a, **k):
        raise AssertionError("must not be reached")

    monkeypatch.setattr(sqlcache.SafeDB, "_conn", lambda self: (_ for _ in ()).throw(__import__("sqlite3").OperationalError("disk I/O error")))
    monkeypatch.setattr(sqlcache.time, "sleep", lambda s: None)
    r = c.chat(MODEL, msgs(1), seed=0)                # read -> miss (error counted), API call, write dropped
    assert r["text"].startswith("SELECT") and not r["cached"]
    assert sqlcache.stats()["dropped_writes"] >= 1 and sqlcache.stats()["miss_on_error"] >= 1


def _writer(args):
    path, wid = args
    s = _Shards(path)
    for i in range(150):
        k = "%02x%062d" % ((i * 7 + wid) % 256, i)     # overlapping + distinct keys
        s.put(k, "m", {"i": i}, json.dumps({"text": "t%d" % i}), 1, 1, 0, 0.0, "realtime")
    return True


def test_multiprocess_writers_on_one_pod_keep_file_healthy(tmp_path):
    path = str(tmp_path / "shared_cache")
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(4) as pool:
        assert all(pool.map(_writer, [(path, w) for w in range(4)]))
    for p in os.listdir(path):
        if p.endswith(".sqlite"):
            ok, detail = sqlcache.check_file(os.path.join(path, p))
            assert ok, (p, detail)
    s = _Shards(path)
    assert s.get("%02x%062d" % (7 % 256, 1)) is not None
    assert not any(".corrupt-" in p for p in os.listdir(path))


def test_embedding_cache_survives_corruption(tmp_path):
    c = make(tmp_path)
    c._emb_db().execute("INSERT OR REPLACE INTO rr VALUES (?,?)", ("zz", "garbage"))
    # rerank cache row unreadable -> bad row counted, would recompute (offline here): proves no crash on parse
    c.offline = True
    from nl2sql.llm import LLMError
    with pytest.raises(LLMError):                     # offline rerank miss is the only allowed failure
        c.rerank("q", ["a", "b"], "gte-rerank-v2")


# ------------------------------------------------------------------------------------------------ archive
def test_export_import_roundtrip_and_audit(tmp_path):
    run = tmp_path / "run1"
    run.mkdir()
    c = make(tmp_path, cache="cacheA", usage="run1/usage.jsonl")
    for i in range(12):
        c.chat(MODEL, msgs(i), seed=0, tag="gen")
    c.chat(MODEL, msgs(3), seed=0, tag="gen")        # cache hit: key logged too
    man = CA.export_run(str(run), store=c.cache)
    assert man["n_keys"] == 12 and man["n_archived"] == 12 and man["n_missing_in_cache"] == 0
    arch = run / "llm_cache.jsonl.gz"
    assert arch.exists() and (os.stat(str(arch)).st_mode & 0o222) == 0          # read-only
    assert CA.verify_archive(str(arch)) == {"ok": True, "n": 12, "error": None}
    # warm a different, empty cache from the archive: the same calls are then cache hits and need no API
    res = CA.import_archives([str(tmp_path)], str(tmp_path / "cacheB"))
    assert res["inserted"] == 12 and res["bad"] == 0
    fake = Fake()
    c2 = make(tmp_path, fake, cache="cacheB", usage="u2.jsonl")
    outs = [c2.chat(MODEL, msgs(i), seed=0, tag="gen") for i in range(12)]
    assert fake.n == 0 and all(o["cached"] for o in outs)
    assert [o["text"] for o in outs] == [c.chat(MODEL, msgs(i), seed=0)["text"] for i in range(12)]
    # idempotent import
    assert CA.import_archives([str(arch)], str(tmp_path / "cacheB"))["inserted"] == 0


def test_archive_merges_on_resume_and_detects_tampering(tmp_path):
    run = tmp_path / "run2"
    run.mkdir()
    c = make(tmp_path, cache="cacheC", usage="run2/usage.jsonl")
    for i in range(5):
        c.chat(MODEL, msgs(i), seed=0)
    CA.export_run(str(run), store=c.cache)
    for i in range(5, 9):
        c.chat(MODEL, msgs(i), seed=0)
    man = CA.export_run(str(run), store=c.cache)
    assert man["n_archived"] == 9 and man["n_new"] == 4
    arch = str(run / "llm_cache.jsonl.gz")
    # tamper: change the request text of one record, keep the key
    recs = [json.loads(l) for l in gzip.open(arch, "rt")]
    r = json.loads(recs[0]["req"])
    r["messages"][0]["content"] = "TAMPERED"
    recs[0]["req"] = json.dumps(r)
    bad = str(tmp_path / "bad.jsonl.gz")
    with gzip.open(bad, "wt") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")
    v = CA.verify_archive(bad)
    assert not v["ok"] and "key does not match" in v["error"]
    res = CA.import_archives([bad], str(tmp_path / "cacheD"))
    assert res["bad"] == 1 and res["inserted"] == 0


def test_cache_export_cli(tmp_path):
    run = tmp_path / "runs" / "E" / "r1"
    run.mkdir(parents=True)
    c = make(tmp_path, cache="cacheE", usage="runs/E/r1/usage.jsonl")
    c.chat(MODEL, msgs(1), seed=0)
    env = dict(os.environ, PYTHONPATH=CODE)
    tool = os.path.join(CODE, "tools", "cache_export.py")
    out = subprocess.check_output([sys.executable, tool, "export", str(tmp_path / "runs"), "--cache", str(tmp_path / "cacheE")], env=env, universal_newlines=True)
    assert '"n_archived": 1' in out
    out = subprocess.check_output([sys.executable, tool, "verify", str(tmp_path / "runs")], env=env, universal_newlines=True)
    assert '"ok": true' in out
    out = subprocess.check_output([sys.executable, tool, "import", str(tmp_path / "runs"), "--cache", str(tmp_path / "cacheF")], env=env, universal_newlines=True)
    assert '"inserted": 1' in out
