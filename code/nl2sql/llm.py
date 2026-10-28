"""DashScope (OpenAI-compatible) client.

Features (CONTRACT section 2, EXPERIMENT_SPEC section 0):
  * per-model concurrency semaphore (default 8 per process), 429/5xx/timeout retry with exponential backoff
  * sqlite request cache (sharded), key = sha256(model | canonical-JSON(temperature, seed, top_p, max_tokens,
    extra_body[, n]) | messages); a hit returns the originally recorded tokens and latency (cost tables are
    cache-independent) and is flagged cached=True
  * usage.jsonl: one line per call incl. cached_tokens (implicit context cache) and mode (realtime|batch|cache)
  * embeddings + rerank with their own caches; counted separately from chat calls
  * Batch API mode (file upload + /batches, 24h window): `batch_prefill` fills the same cache, so the normal
    pipeline then runs with cache hits. Batch is 50 % of real-time price but has no context-cache discount.
The API key is read from a file only; it never appears in logs, errors or argv.
"""
import asyncio
import concurrent.futures
import hashlib
import io
import json
import os
import random
import threading
import time
import zlib

import httpx
import numpy as np

from . import meter as _meter
from . import paths
from . import sqlcache
from .config import check_model_allowed, models_cfg
from .util import canon_json

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
RERANK_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
RETRY_STATUS = {429, 500, 502, 503, 504}


def _is_capacity_throttle(r):
    """F1 (2026-10-26): DashScope answers a provider-side capacity throttle with HTTP 400 'ServiceUnavailable: Too many requests. Your requests are
    being throttled due to system capacity limits' (seen in the M6 burn-in, which has no item-level retry pass). Transient, retried like a 429."""
    try:
        e = r.json().get("error") or {}
        txt = ("%s %s" % (e.get("code"), e.get("message", ""))).lower()
    except Exception:
        txt = (r.text or "")[:300].lower()
    return "throttl" in txt or "too many requests" in txt


class LLMError(Exception):
    def __init__(self, msg, status=None, code=None):
        super().__init__(msg)
        self.status = status
        self.code = code


def read_key(key_file):
    """The key is read from `key_file` (one line `QWEN_API_KEY=<key>`; default $NL2SQL_KEY_FILE or ~/.nl2sql/qwen.env);
    if that file does not exist, from the environment variable QWEN_API_KEY or DASHSCOPE_API_KEY. It is never taken from a command line."""
    key_file = key_file or paths.KEY_FILE
    if not os.path.exists(key_file):
        for _name in ("QWEN_API_KEY", "DASHSCOPE_API_KEY"):
            if os.environ.get(_name, "").strip():
                return os.environ[_name].strip()
    with open(key_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("QWEN_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise LLMError("QWEN_API_KEY not found in key file")


class _Shards:
    """Sharded sqlite KV store of LLM calls (thread-safe, self-healing, see sqlcache). `path` ending in .sqlite = single file (tests).
    Never raises on cache trouble: a read problem is a miss, a write problem drops the write (both counted in sqlcache.STATS)."""

    SCHEMA = ("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, model TEXT, ts REAL, req BLOB, resp TEXT, "
              "pt INTEGER, ct INTEGER, cached_tokens INTEGER, latency REAL, mode TEXT)")

    def __init__(self, path, n_shards=16):
        self.path = str(path)
        self.single = self.path.endswith(".sqlite")
        if not self.single:
            os.makedirs(self.path, exist_ok=True)
        self.n = 1 if self.single else n_shards
        self._dbs = {}
        self._dbs_lock = threading.Lock()
        # startup integrity check of every existing shard (quarantine = rename, a fresh file is created lazily)
        for idx in range(self.n):
            if os.path.exists(self._file(idx)):
                self._db(idx)

    def _file(self, idx):
        return self.path if self.single else os.path.join(self.path, "calls_%02d.sqlite" % idx)

    def _db(self, idx):
        d = self._dbs.get(idx)
        if d is None:
            with self._dbs_lock:
                d = self._dbs.get(idx)
                if d is None:
                    d = self._dbs[idx] = sqlcache.SafeDB(self._file(idx), [self.SCHEMA])
        return d

    def _idx(self, key):
        return 0 if self.single else int(key[0], 16) % self.n

    def get(self, key):
        return self._db(self._idx(key)).query_one(
            "SELECT resp, pt, ct, cached_tokens, latency, mode FROM calls WHERE key=?", (key,))

    def get_full(self, key):
        return self._db(self._idx(key)).query_one(
            "SELECT key, model, ts, req, resp, pt, ct, cached_tokens, latency, mode FROM calls WHERE key=?", (key,))

    def drop(self, key):
        """delete an unreadable row so that the fresh response can be stored (INSERT OR IGNORE would keep the bad one)."""
        sqlcache.bump("bad_rows")
        return self._db(self._idx(key)).execute("DELETE FROM calls WHERE key=?", (key,))

    def put_raw(self, row):
        return self._db(self._idx(row[0])).execute("INSERT OR IGNORE INTO calls VALUES (?,?,?,?,?,?,?,?,?,?)", tuple(row))

    def put(self, key, model, req_obj, resp_json, pt, ct, cached_tokens, latency, mode):
        blob = zlib.compress(canon_json(req_obj).encode("utf-8"), 3)
        return self.put_raw((key, model, time.time(), blob, resp_json, pt, ct, cached_tokens, latency, mode))


class LLMClient:
    def __init__(self, cache_path, usage_path, key_file=None, max_concurrency_per_model=8,
                 timeout_s=120.0, max_attempts=6, backoff_base_s=2.0, backoff_cap_s=60.0, offline=False):
        self.cache = _Shards(cache_path)
        base = str(cache_path)
        emb_path = base[:-7] + "_emb.sqlite" if base.endswith(".sqlite") else os.path.join(base, "emb.sqlite")
        self._emb_path = emb_path
        self._emb = None
        self.usage_path = usage_path
        self._usage_lock = threading.Lock()
        if usage_path:
            os.makedirs(os.path.dirname(os.path.abspath(usage_path)), exist_ok=True)
        self.key_file = key_file
        self._key = None
        self.max_conc = max_concurrency_per_model
        self._sems = {}
        self._sem_lock = threading.Lock()
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        self.offline = offline            # tests: raise on cache miss instead of calling the API
        self._http = None
        self._http_lock = threading.Lock()
        self._pool = None
        self.stats = {"calls": 0, "cache_hits": 0, "n429": 0, "n5xx": 0, "ntimeout": 0, "retries": 0, "failures": 0,
                      "embed_calls": 0, "rerank_calls": 0}
        self._stats_lock = threading.Lock()

    # ---------------------------------------------------------------- plumbing
    def _inc(self, k, v=1):
        with self._stats_lock:
            self.stats[k] = self.stats.get(k, 0) + v

    def _headers(self):
        if self._key is None:
            self._key = read_key(self.key_file)
        return {"Authorization": "Bearer " + self._key}

    def _client(self):
        if self._http is None:
            with self._http_lock:
                if self._http is None:
                    self._http = httpx.Client(timeout=httpx.Timeout(self.timeout_s, connect=20.0),
                                              limits=httpx.Limits(max_connections=256, max_keepalive_connections=64))
        return self._http

    def _sem(self, model):
        with self._sem_lock:
            s = self._sems.get(model)
            if s is None:
                s = self._sems[model] = threading.BoundedSemaphore(self.max_conc)
            return s

    def _log_usage(self, rec):
        if not self.usage_path:
            return
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with self._usage_lock:
            fd = os.open(self.usage_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)

    def _backoff(self, attempt, retry_after=None):
        d = min(self.backoff_cap_s, self.backoff_base_s * (2 ** attempt))
        if retry_after:
            d = max(d, min(float(retry_after), self.backoff_cap_s))
        time.sleep(d * (0.75 + 0.5 * random.random()))

    def _request(self, method, url, model_for_sem=None, **kw):
        """HTTP with retry; returns httpx.Response (2xx) or raises LLMError."""
        last = None
        for attempt in range(self.max_attempts):
            sem = self._sem(model_for_sem) if model_for_sem else None
            if sem:
                sem.acquire()
            try:
                r = self._client().request(method, url, headers=self._headers(), **kw)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                self._inc("ntimeout")
                last = LLMError("transport/timeout: %s" % type(e).__name__)
                r = None
            finally:
                if sem:
                    sem.release()
            if r is not None:
                if r.status_code < 300:
                    return r
                if r.status_code in RETRY_STATUS or (r.status_code == 400 and _is_capacity_throttle(r)):
                    self._inc("n429" if r.status_code == 429 else "n5xx")
                    last = LLMError("http %d" % r.status_code, status=r.status_code)
                    if attempt + 1 < self.max_attempts:
                        self._inc("retries")
                        self._backoff(attempt, r.headers.get("retry-after"))
                    continue
                code = None
                try:
                    code = (r.json().get("error") or {}).get("code")
                    msg = (r.json().get("error") or {}).get("message", "")[:300]
                except Exception:
                    msg = r.text[:300]
                raise LLMError("http %d %s: %s" % (r.status_code, code, msg), status=r.status_code, code=code)
            elif attempt + 1 < self.max_attempts:
                self._inc("retries")
                self._backoff(attempt)
        self._inc("failures")
        raise last or LLMError("request failed")

    # ---------------------------------------------------------------- chat
    @staticmethod
    def _params(temperature, seed, max_tokens, enable_thinking, n, top_p=1.0):
        p = {"temperature": temperature, "seed": seed, "top_p": top_p, "max_tokens": max_tokens,
             "extra_body": {"enable_thinking": bool(enable_thinking)}}
        if n != 1:
            p["n"] = n
        return p

    @staticmethod
    def cache_key(model, params, messages):
        h = hashlib.sha256()
        h.update(model.encode("utf-8"))
        h.update(b"\x1f")
        h.update(canon_json(params).encode("utf-8"))
        h.update(b"\x1f")
        h.update(canon_json(messages).encode("utf-8"))
        return h.hexdigest()

    def chat(self, model, messages, *, temperature=0.0, seed=None, max_tokens=2048, enable_thinking=False, n=1,
             tag="", top_p=1.0):
        check_model_allowed(model)
        params = self._params(temperature, seed, max_tokens, enable_thinking, n, top_p)
        key = self.cache_key(model, params, messages)
        m = _meter.current()
        row = self.cache.get(key)
        if row is not None:
            try:
                resp, pt, ct, cached_tokens, lat, mode = row
                out = json.loads(resp)
                out["text"]
                int(pt or 0), int(ct or 0), float(lat or 0.0)
            except Exception:                 # garbage blob in the cache: forget the row, treat as a miss (never an error)
                self.cache.drop(key)
                self._inc("cache_bad_rows")
                row = None
        if row is not None:
            self._inc("cache_hits")
            self._inc("calls")
            if m:
                m.add_llm(tag, pt, ct, cached_tokens or 0, lat, True)
            self._log_usage({"ts": round(time.time(), 3), "kind": "chat", "model": model, "tag": tag, "pt": pt, "ct": ct,
                             "cached_tokens": cached_tokens or 0, "latency_s": lat, "cached": True, "mode": mode,
                             "attempts": 0, "key": key})
            return {"text": out["text"], "prompt_tokens": pt, "completion_tokens": ct,
                    "cached_tokens": cached_tokens or 0, "latency_s": lat, "cached": True}
        if self.offline:
            raise LLMError("offline cache miss")
        body = {"model": model, "messages": messages, "temperature": temperature, "top_p": top_p,
                "max_tokens": max_tokens}
        if seed is not None:
            body["seed"] = seed
        if n != 1:
            body["n"] = n
        body["enable_thinking"] = bool(enable_thinking)
        t0 = time.time()
        n429_before = self.stats["n429"]
        try:
            if enable_thinking:
                texts, pt, ct, cached_tokens = self._chat_stream(model, body)
            else:
                r = self._request("POST", BASE_URL + "/chat/completions", model_for_sem=model, json=body)
                j = r.json()
                texts = [(c.get("message") or {}).get("content") or "" for c in j["choices"]]
                u = j.get("usage") or {}
                pt, ct = int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
                cached_tokens = int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or u.get("cached_tokens") or 0)
        except LLMError as e:
            self._log_usage({"ts": round(time.time(), 3), "kind": "chat", "model": model, "tag": tag, "error": str(e)[:200]})
            raise
        lat = time.time() - t0
        text = texts if n != 1 else texts[0]
        self.cache.put(key, model, {"model": model, "params": params, "messages": messages},
                       json.dumps({"text": text}, ensure_ascii=False), pt, ct, cached_tokens, lat, "realtime")
        self._inc("calls")
        if m:
            m.add_llm(tag, pt, ct, cached_tokens, lat, False)
        self._log_usage({"ts": round(time.time(), 3), "kind": "chat", "model": model, "tag": tag, "pt": pt, "ct": ct,
                         "cached_tokens": cached_tokens, "latency_s": round(lat, 3), "cached": False, "mode": "realtime",
                         "n429": self.stats["n429"] - n429_before, "key": key})
        return {"text": text, "prompt_tokens": pt, "completion_tokens": ct, "cached_tokens": cached_tokens,
                "latency_s": lat, "cached": False}

    def _chat_stream(self, model, body):
        """enable_thinking=True requires streaming on this endpoint (E2.T only)."""
        body = dict(body, stream=True, stream_options={"include_usage": True})
        last = None
        for attempt in range(self.max_attempts):
            sem = self._sem(model)
            sem.acquire()
            try:
                texts, pt, ct, cached_tokens = {}, 0, 0, 0
                with self._client().stream("POST", BASE_URL + "/chat/completions", headers=self._headers(), json=body) as r:
                    if r.status_code >= 300:
                        txt = r.read().decode("utf-8", "replace")[:300]
                        if r.status_code in RETRY_STATUS:
                            self._inc("n429" if r.status_code == 429 else "n5xx")
                            last = LLMError("http %d" % r.status_code, status=r.status_code)
                            raise _Retry()
                        raise LLMError("http %d: %s" % (r.status_code, txt), status=r.status_code)
                    for line in r.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        d = line[5:].strip()
                        if d == "[DONE]":
                            break
                        j = json.loads(d)
                        for c in j.get("choices") or []:
                            delta = c.get("delta") or {}
                            texts[c.get("index", 0)] = texts.get(c.get("index", 0), "") + (delta.get("content") or "")
                        u = j.get("usage")
                        if u:
                            pt, ct = int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
                            cached_tokens = int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
                return [texts[i] for i in sorted(texts)] or [""], pt, ct, cached_tokens
            except _Retry:
                pass
            except (httpx.TimeoutException, httpx.TransportError) as e:
                self._inc("ntimeout")
                last = LLMError("transport/timeout: %s" % type(e).__name__)
            finally:
                sem.release()
            self._inc("retries")
            self._backoff(attempt)
        self._inc("failures")
        raise last or LLMError("stream failed")

    async def achat(self, model, messages, **kw):
        loop = asyncio.get_running_loop()
        if self._pool is None:
            self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=64)
        import contextvars
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(self._pool, lambda: ctx.run(lambda: self.chat(model, messages, **kw)))

    # ---------------------------------------------------------------- embeddings / rerank
    def _emb_db(self):
        if self._emb is None:
            with self._http_lock:
                if self._emb is None:
                    self._emb = sqlcache.SafeDB(self._emb_path, [
                        "CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, v BLOB)",
                        "CREATE TABLE IF NOT EXISTS rr (key TEXT PRIMARY KEY, v TEXT)"])
        return self._emb

    def snapshot_stats(self):
        """client counters + cache-health counters (sqlcache.STATS: db_errors, quarantined, dropped_writes, miss_on_error, bad_rows)."""
        d = dict(self.stats)
        d.update({"cache_" + k: v for k, v in sqlcache.stats().items()})
        return d

    def embed_np(self, texts, model, batch=10):
        """float32 matrix [len(texts), dim]. Cache key = sha256(model|text)."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        check_model_allowed(model)
        keys = [hashlib.sha256((model + "\x1f" + t).encode("utf-8")).hexdigest() for t in texts]
        out = [None] * len(texts)
        edb = self._emb_db()
        need = []
        for i, k in enumerate(keys):
            row = edb.query_one("SELECT v FROM emb WHERE key=?", (k,))
            if row and row[0] is not None and len(row[0]) >= 4 and len(row[0]) % 4 == 0:
                out[i] = np.frombuffer(row[0], dtype=np.float32)
            else:
                need.append(i)
        m = _meter.current()
        # de-duplicate identical missing texts
        uniq = {}
        for i in need:
            uniq.setdefault(texts[i], []).append(i)
        todo = list(uniq.keys())
        chunks = [todo[s_:s_ + batch] for s_ in range(0, len(todo), batch)]
        if chunks and self.offline:
            raise LLMError("offline embed miss")

        def call(chunk):
            t0 = time.time()
            r = self._request("POST", BASE_URL + "/embeddings", model_for_sem=model,
                              json={"model": model, "input": [t if t.strip() else " " for t in chunk], "encoding_format": "float"})
            j = r.json()
            data = sorted(j["data"], key=lambda d: d["index"])
            return chunk, [np.asarray(d["embedding"], dtype=np.float32) for d in data], j.get("usage") or {}, time.time() - t0

        if len(chunks) > 1:
            if self._pool is None:
                self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=64)
            results = list(self._pool.map(call, chunks))
        else:
            results = [call(c_) for c_ in chunks]
        for chunk, vecs, u, el in results:
            self._inc("embed_calls")
            if m:
                m.add_aux("embed")
            self._log_usage({"ts": round(time.time(), 3), "kind": "embed", "model": model, "n": len(chunk),
                             "pt": int(u.get("prompt_tokens") or u.get("total_tokens") or 0), "latency_s": round(el, 3)})
            for t, v in zip(chunk, vecs):
                k = hashlib.sha256((model + "\x1f" + t).encode("utf-8")).hexdigest()
                edb.execute("INSERT OR REPLACE INTO emb VALUES (?,?)", (k, v.tobytes()))
                for i in uniq[t]:
                    out[i] = v
        return np.stack(out)

    def embed(self, texts, model):
        return self.embed_np(texts, model).tolist()

    def rerank(self, query, docs, model):
        """relevance scores aligned with docs. Cached by (model, query, docs)."""
        if not docs:
            return []
        check_model_allowed(model)
        key = hashlib.sha256(canon_json([model, query, docs]).encode("utf-8")).hexdigest()
        edb = self._emb_db()
        m = _meter.current()
        row = edb.query_one("SELECT v FROM rr WHERE key=?", (key,))
        if row:
            try:
                cached = json.loads(row[0])
                if len(cached) != len(docs):
                    raise ValueError("length")
                if m:
                    m.add_aux("rerank")
                return cached
            except Exception:
                sqlcache.bump("bad_rows")         # unreadable row: recompute and overwrite below
        if self.offline:
            raise LLMError("offline rerank miss")
        t0 = time.time()
        r = self._request("POST", RERANK_URL, model_for_sem=model,
                          json={"model": model, "query": query, "documents": list(docs), "top_n": len(docs)})
        j = r.json()
        res = j.get("results") or (j.get("output") or {}).get("results") or []
        scores = [0.0] * len(docs)
        for it in res:
            scores[it["index"]] = float(it["relevance_score"])
        self._inc("rerank_calls")
        if m:
            m.add_aux("rerank")
        self._log_usage({"ts": round(time.time(), 3), "kind": "rerank", "model": model, "n": len(docs),
                         "latency_s": round(time.time() - t0, 3)})
        edb.execute("INSERT OR REPLACE INTO rr VALUES (?,?)", (key, json.dumps(scores)))
        return scores

    # ---------------------------------------------------------------- Batch API
    def batch_prefill(self, model, requests, *, poll_s=30.0, max_wait_s=24 * 3600, chunk=20000, state_dir=None,
                      endpoint="/v1/chat/completions"):
        """Run `requests` (list of dicts with messages + optional temperature/seed/max_tokens/enable_thinking/tag)
        through the Batch API and store the results in the call cache, so that later chat() calls with the same
        parameters are cache hits. Requests already cached are skipped. One file = one model + one thinking mode.
        Returns {"submitted","skipped","ok","failed","batch_ids"}. Batch price is 50 % of real-time; no context-cache
        discount applies (official doc, checked 2026-10-26). `state_dir` stores batch ids so a crashed driver can resume."""
        check_model_allowed(model)
        todo, seen = [], set()
        skipped = 0
        for rq in requests:
            params = self._params(rq.get("temperature", 0.0), rq.get("seed"), rq.get("max_tokens", 2048),
                                  rq.get("enable_thinking", False), rq.get("n", 1))
            key = self.cache_key(model, params, rq["messages"])
            if key in seen:
                continue
            seen.add(key)
            if self.cache.get(key) is not None:
                skipped += 1
                continue
            todo.append((key, params, rq))
        res = {"submitted": len(todo), "skipped": skipped, "ok": 0, "failed": 0, "batch_ids": []}
        for s in range(0, len(todo), chunk):
            part = todo[s:s + chunk]
            lines = []
            for key, params, rq in part:
                body = {"model": model, "messages": rq["messages"], "temperature": params["temperature"],
                        "top_p": params["top_p"], "max_tokens": params["max_tokens"],
                        "enable_thinking": params["extra_body"]["enable_thinking"]}
                if params["seed"] is not None:
                    body["seed"] = params["seed"]
                lines.append(json.dumps({"custom_id": key, "method": "POST", "url": endpoint, "body": body}, ensure_ascii=False))
            data = ("\n".join(lines) + "\n").encode("utf-8")
            r = self._request("POST", BASE_URL + "/files", files={"file": ("batch.jsonl", io.BytesIO(data), "application/jsonl")},
                              data={"purpose": "batch"})
            fid = r.json()["id"]
            r = self._request("POST", BASE_URL + "/batches", json={"input_file_id": fid, "endpoint": endpoint,
                                                                   "completion_window": "24h"})
            bid = r.json()["id"]
            res["batch_ids"].append(bid)
            if state_dir:
                os.makedirs(state_dir, exist_ok=True)
                with open(os.path.join(state_dir, "batch_%s.json" % bid), "w") as f:
                    json.dump({"batch_id": bid, "model": model, "n": len(part), "ts": time.time()}, f)
            ok, bad = self._collect_batch(bid, model, {k: (p, rq) for k, p, rq in part}, poll_s, max_wait_s)
            res["ok"] += ok
            res["failed"] += bad
        return res

    def collect_batch(self, batch_id, model, requests, poll_s=30.0, max_wait_s=24 * 3600):
        """Resume: poll an existing batch and load its results into the cache."""
        m = {}
        for rq in requests:
            params = self._params(rq.get("temperature", 0.0), rq.get("seed"), rq.get("max_tokens", 2048),
                                  rq.get("enable_thinking", False), rq.get("n", 1))
            m[self.cache_key(model, params, rq["messages"])] = (params, rq)
        return self._collect_batch(batch_id, model, m, poll_s, max_wait_s)

    def _collect_batch(self, bid, model, by_key, poll_s, max_wait_s):
        t0 = time.time()
        while True:
            j = self._request("GET", BASE_URL + "/batches/" + bid).json()
            st = j.get("status")
            if st in ("completed", "failed", "expired", "cancelled"):
                break
            if time.time() - t0 > max_wait_s:
                raise LLMError("batch %s still %s after %ds" % (bid, st, max_wait_s))
            time.sleep(poll_s)
        ok = bad = 0
        for fkey in ("output_file_id", "error_file_id"):
            fid = j.get(fkey)
            if not fid:
                continue
            txt = self._request("GET", BASE_URL + "/files/%s/content" % fid).text
            for line in txt.splitlines():
                if not line.strip():
                    continue
                o = json.loads(line)
                key = o.get("custom_id")
                resp = (o.get("response") or {})
                body = resp.get("body") or {}
                if key in by_key and resp.get("status_code") == 200 and body.get("choices"):
                    params, rq = by_key[key]
                    u = body.get("usage") or {}
                    text = (body["choices"][0].get("message") or {}).get("content") or ""
                    self.cache.put(key, model, {"model": model, "params": params, "messages": rq["messages"]},
                                   json.dumps({"text": text}, ensure_ascii=False),
                                   int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)), 0, 0.0, "batch")
                    self._log_usage({"ts": round(time.time(), 3), "kind": "chat", "model": model, "tag": rq.get("tag", ""),
                                     "pt": int(u.get("prompt_tokens", 0)), "ct": int(u.get("completion_tokens", 0)),
                                     "cached_tokens": 0, "latency_s": 0.0, "cached": False, "mode": "batch", "key": key})
                    ok += 1
                else:
                    bad += 1
        return ok, bad

    def close(self):
        if self._http is not None:
            self._http.close()


class _Retry(Exception):
    pass
