"""Shared plumbing for the B1 baselines.

* Imports X2's core modules when present, otherwise the local shim (`_shim.py`, tests only).
* `BaselineSystem.run()` wraps a subclass' `solve()`: per-question accounting (calls / tokens / latency),
  final execution check, CONTRACT section 4 row.
* Accounting convention: `calls` = number of `LLMClient.chat` requests (a request with n>1 counts once, as in
  nl2sql.meter.Meter); `samples` (completions) is reported in `extra`.  Embedding calls are not LLM calls.
* Threads: nl2sql.meter uses contextvars, so worker threads must run inside `copy_context()`; use `Ctx.pmap`.
"""
from __future__ import annotations

import contextvars
import functools
import hashlib
import os
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_SHIM_SPLITS = os.environ.get("B1_SHIM_SPLITS") == "1"

try:
    from nl2sql.systems.base import System  # type: ignore
except Exception:  # pragma: no cover - X2 base.py absent
    from . import _shim
    System = _shim.System

try:
    from nl2sql import db as dbmod  # type: ignore
except Exception:  # pragma: no cover
    from . import _shim
    dbmod = _shim.db

try:
    from nl2sql import data as _real_data  # type: ignore
except Exception:  # pragma: no cover
    _real_data = None

try:
    from nl2sql.util import strip_sql as _strip_sql  # type: ignore
except Exception:  # pragma: no cover
    _strip_sql = None


class _DataAdapter:
    """data.py facade: real nl2sql.data when available; a deterministic stand-in split (shim) while
    exp/splits/train_holdout.json does not exist yet (B1 smoke tests only; NOT the official HT split)."""

    used_shim_split = False

    def load_split(self, name, **kw):
        if _real_data is not None and not _SHIM_SPLITS:
            try:
                return _real_data.load_split(name, **kw)
            except FileNotFoundError:
                pass
        from . import _shim
        self.used_shim_split = True
        return _shim.load_split(name)

    def get_schema(self, item_or_db, split=None):
        if _real_data is not None:
            return _real_data.get_schema(item_or_db, split)
        from . import _shim
        return _shim.get_schema(item_or_db, split)

    def render_ddl(self, schema, **kw):
        if _real_data is not None:
            return _real_data.render_ddl(schema, **kw)
        return schema["ddl"]


datamod = _DataAdapter()

DEFAULT_EMBED_MODEL = "qwen3.7-text-embedding"


class ApiFail(RuntimeError):
    pass


# ----------------------------------------------------------------------------- text helpers
_FENCE_ALL = re.compile(r"```(?:sql|sqlite|SQL)?[ \t]*\n?(.*?)```", re.S)
_XML_SQL = re.compile(r"<sql>(.*?)</sql>", re.S | re.I)


def extract_sql(text, which="first"):
    """Extract one SQL statement from an LLM reply.  which='first'|'last' picks the code block."""
    if text is None:
        return ""
    t = text.strip()
    blocks = _FENCE_ALL.findall(t)
    if blocks:
        blk = blocks[-1] if which == "last" else blocks[0]
        return _clean_sql(blk)
    m = _XML_SQL.findall(t)
    if m:
        return _clean_sql(m[-1] if which == "last" else m[0])
    if _strip_sql is not None:
        return _strip_sql(t)
    return _clean_sql(t)


def _clean_sql(s):
    s = s.strip()
    if not re.match(r"(?is)^\s*(select|with|\()", s):
        m2 = re.search(r"(?is)\b(select|with)\b", s)
        if m2:
            s = s[m2.start():]
    s = s.strip().rstrip(";").strip()
    return s


def one_line(sql):
    return " ".join(sql.split())


def fmt_rows(rows, max_rows=5, max_chars=40):
    out = []
    for r in rows[:max_rows]:
        out.append(" | ".join(_cut(c, max_chars) for c in r))
    return "\n".join(out)


def _cut(v, n):
    s = "NULL" if v is None else str(v)
    return s if len(s) <= n else s[: n - 3] + "..."


# ----------------------------------------------------------------------------- schema rendering
_SCHEMA_CACHE = {}
_SCHEMA_LOCK = threading.Lock()


def get_schema_cached(item):
    """Thread-safe memoised data.get_schema (X2's get_schema writes a pid-named tmp file, so concurrent threads of one
    process must not build the same schema at the same time)."""
    key = (item.db_id, getattr(item, "split", None) or "")
    with _SCHEMA_LOCK:
        if key in _SCHEMA_CACHE:
            return _SCHEMA_CACHE[key]
        sch = datamod.get_schema(item)
        _SCHEMA_CACHE[key] = sch
        return sch


def quote_flag(cfg):
    """X6: config `ddl.quote_keywords` (same key the `ours` systems read).  False/absent = legacy prompts, byte-identical."""
    return bool(((cfg or {}).get("ddl") or {}).get("quote_keywords"))


def qname(name, quote_keywords):
    """Identifier as shown in a prompt (data._q): reserved words double-quoted when quote_keywords, else unchanged."""
    return _real_data._q(name, True) if (quote_keywords and _real_data is not None) else name


def schema_text(item, *, notes=False, max_examples=3, max_desc=150, max_ex_chars=40, quote_keywords=False):
    """Schema block.  Default = data.get_schema()["ddl"], which already carries column descriptions and sample values
    (EXPERIMENT_SPEC zero-shot: "DDL + sample values").  notes=True appends a separate per-column note list (unused).
    quote_keywords=True (X6, config ddl.quote_keywords): the same block re-rendered by data.render_ddl(quote_keywords=True), i.e. SQLite
    reserved words used as table/column names are double-quoted exactly as for the `ours` systems; False = the stored legacy text."""
    sch = get_schema_cached(item)
    ddl = datamod.render_ddl(sch, quote_keywords=True) if quote_keywords else sch.get("ddl", "")
    if not notes:
        return ddl
    lines = []
    for t, cols in (sch.get("tables") or {}).items():
        for c in cols:
            bits = []
            d = (c.get("desc") or "").replace("\n", " ").strip()
            if d and d.lower() != str(c["name"]).lower():
                bits.append(d[:max_desc])
            ex = [str(e) for e in (c.get("examples") or [])][:max_examples]
            if ex:
                bits.append("e.g. " + ", ".join(_cut(e, max_ex_chars) for e in ex))
            if bits:
                lines.append("%s.%s: %s" % (qname(t, quote_keywords), qname(c["name"], quote_keywords), "; ".join(bits)))
    if not lines:
        return ddl
    return ddl + "\n\n-- Column notes (description; sample values)\n" + "\n".join("-- " + l for l in lines)


def raw_ddl(item, quote_keywords=False):
    """Plain CREATE TABLE statements with FK lines, no descriptions/samples (DAIL-SQL / AC-SQL style)."""
    sch = get_schema_cached(item)
    try:
        return datamod.render_ddl(sch, samples=False, descriptions=False, quote_keywords=quote_keywords)
    except Exception:
        return sch.get("ddl", "")


def question_block(item):
    q = "Question: " + item.question.strip()
    ev = (item.evidence or "").strip()
    if ev:
        q += "\nEvidence: " + ev
    return q


# ----------------------------------------------------------------------------- embeddings
_EMB_LOCK = threading.Lock()
_EMB_MEM = {}


def embed_matrix(llm, model, texts, cache_dir=None, name="emb", batch=64):
    """L2-normalised embeddings for `texts`; cached on disk (npz keyed by content hash) when cache_dir is given."""
    h = hashlib.sha256((model + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()
    key = (name, h)
    with _EMB_LOCK:
        if key in _EMB_MEM:
            return _EMB_MEM[key]
    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, "%s_%s.npy" % (name, h[:24]))
        if os.path.exists(path):
            m = np.load(path)
            with _EMB_LOCK:
                _EMB_MEM[key] = m
            return m
    chunks = [texts[i:i + 10] for i in range(0, len(texts), 10)]
    if len(chunks) > 4:
        with ThreadPoolExecutor(max_workers=8) as ex:
            parts = list(ex.map(lambda c: llm.embed(c, model), chunks))
    else:
        parts = [llm.embed(c, model) for c in chunks]
    vecs = [v for part in parts for v in part]
    m = np.asarray(vecs, dtype=np.float32)
    m /= (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    if path:
        tmp = path + ".tmp%d_%d" % (os.getpid(), threading.get_ident())
        with open(tmp, "wb") as f:
            np.save(f, m)
        os.replace(tmp, path)
    with _EMB_LOCK:
        _EMB_MEM[key] = m
    return m


def embed_one(llm, model, text):
    v = np.asarray(llm.embed([text], model)[0], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


# ----------------------------------------------------------------------------- per-run context
class Ctx:
    """Accounting + LLM call wrapper for one question."""

    def __init__(self, system, item, seed):
        self.system, self.item, self.seed = system, item, seed
        self.llm, self.model = system.llm, system.model
        self.cfg = system.cfg
        self.calls = 0
        self.samples = 0
        self.pt = 0
        self.ct = 0
        self.llm_lat = 0.0
        self.actions = []
        self.extra = {}
        self._lock = threading.Lock()
        self.t0 = time.time()

    # ---- LLM
    def chat(self, messages, *, model=None, temperature=None, seed=None, max_tokens=None, n=1, tag="", top_p=None):
        """One LLMClient.chat request. Returns str (n==1) or list[str]."""
        temp = self.cfg.get("temperature", 0.0) if temperature is None else temperature
        mt = max_tokens or self.cfg.get("max_tokens", 1024)
        sd = self.seed if seed is None else seed
        try:
            kw = {} if top_p is None else {"top_p": top_p}
            r = self.llm.chat(model or self.model, messages, temperature=temp, seed=sd, max_tokens=mt, n=n, tag=tag or self.system.name, **kw)
        except Exception as e:  # LLMClient already retried
            raise ApiFail(str(e)[:300])
        with self._lock:
            self.calls += 1
            self.samples += n
            self.pt += int(r.get("prompt_tokens", 0))
            self.ct += int(r.get("completion_tokens", 0))
            self.llm_lat += float(r.get("latency_s", 0.0))
        return r["text"]

    def user(self, prompt, system=None, **kw):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.chat(msgs, **kw)

    def pmap(self, fn, args_list, workers=None):
        """Run fn(*args) for each tuple in parallel threads, preserving order; propagates contextvars (Meter)."""
        if not args_list:
            return []
        workers = workers or min(len(args_list), int(self.cfg.get("parallel", 8)))
        if workers <= 1:
            return [fn(*a) for a in args_list]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(contextvars.copy_context().run, fn, *a) for a in args_list]
            return [f.result() for f in futs]

    # ---- execution
    def execute(self, sql, timeout_s=30):
        if not sql or not sql.strip():
            return {"ok": False, "rows": [], "error": "empty sql", "elapsed_s": 0.0, "truncated": False}
        return dbmod.execute(self.item.db_path, sql, timeout_s=timeout_s)

    def act(self, a):
        self.actions.append(a)


def result_key(res):
    """BIRD EX equivalence class of an execution result (set of row tuples)."""
    if not res["ok"]:
        return None
    try:
        return frozenset(res["rows"])
    except TypeError:
        return frozenset(tuple(map(str, r)) for r in res["rows"])


# ----------------------------------------------------------------------------- base class
class BaselineSystem(System):
    """Subclasses implement solve(item, seed, ctx) -> (sql, info) where info may hold
    iterations, error_class, critic, extra."""

    name = "baseline"
    default_cfg: dict = {}

    def __init__(self, llm, model, cfg=None):
        merged = dict(self.default_cfg)
        merged.update(cfg or {})
        super().__init__(llm, model, merged)
        self.cfg = merged
        self.qk = quote_flag(merged)          # X6: ddl.quote_keywords (default false = legacy prompts)

    def solve(self, item, seed, ctx):  # pragma: no cover
        raise NotImplementedError

    def run(self, item, seed=0):
        ctx = Ctx(self, item, seed)
        t0 = time.time()
        sql, info = "", {}
        err = None
        try:
            sql, info = self.solve(item, seed, ctx)
            sql = sql or ""
        except ApiFail as e:
            err = "api_fail"
            info = {"extra": {"error": str(e)}}
        except Exception as e:  # keep the run going; the runner counts these
            err = "system_error"
            info = {"extra": {"error": "%s: %s" % (type(e).__name__, e), "trace": traceback.format_exc()[-800:]}}
        res = ctx.execute(sql) if sql else {"ok": False, "error": "no sql"}
        error_class = err or info.get("error_class")
        if error_class is None and not sql:
            error_class = "no_sql"
        extra = dict(ctx.extra)
        extra.update(info.get("extra") or {})
        extra["samples"] = ctx.samples
        row = {
            "qid": item.qid, "db_id": item.db_id, "split": item.split, "system": self.name, "model": self.model,
            "seed": seed, "final_sql": sql, "exec_ok": bool(res["ok"]), "abstained": False,
            "critic": info.get("critic"), "iterations": info.get("iterations", 1),
            "actions": info.get("actions", ctx.actions), "calls": ctx.calls,
            "prompt_tokens": ctx.pt, "completion_tokens": ctx.ct,
            "latency_s": round(time.time() - t0, 3), "error_class": error_class, "extra": extra,
        }
        return row


def run_index(seed):
    """Run index r in the sample-seed rule 1000*r + j (EXPERIMENT_SPEC section 0): seeds 0/101/202 -> r=0/1/2."""
    return {0: 0, 101: 1, 202: 2}.get(seed, seed)
