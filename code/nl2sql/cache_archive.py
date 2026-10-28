"""Per-run, read-only archive of the LLM call-cache rows a run used, and import of such archives into a (machine-local) cache.

The live cache is machine-local and may be lost with the worker machine; the archive makes a run reproducible and auditable without it:
  <run dir>/llm_cache.jsonl.gz       one JSON object per line, sorted by key (sha256(model|params|messages), see LLMClient.cache_key)
  <run dir>/llm_cache.manifest.json  counts, sha256 of the .gz, creation time
Line fields: key, model, ts, req (canonical JSON text of {"model","params","messages"}), resp (JSON text {"text": ...}), pt, ct,
cached_tokens, latency, mode. `key` is re-derived from `req` on export, import and verify, so a record cannot be altered unnoticed.
Which rows a run used: the `key` field of its usage*.jsonl rows (chat calls, cache hits included). Runs that predate this field
cannot be exported. Embedding/rerank rows are not archived (cheap to recompute, deterministic per model).
Archives are append-only across restarts of a run: an existing archive is merged, never shrunk. File mode 0444.
CLI: tools/cache_export.py (export / import / verify / check)."""
import glob
import gzip
import hashlib
import io
import json
import os
import time
import zlib

from .llm import LLMClient, _Shards

ARCHIVE = "llm_cache.jsonl.gz"
MANIFEST = "llm_cache.manifest.json"


def usage_keys(run_dir):
    """ordered unique chat-call keys from every usage*.jsonl directly in run_dir."""
    seen, out = set(), []
    for f in sorted(glob.glob(os.path.join(run_dir, "usage*.jsonl"))):
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"key"' not in line:
                    continue
                try:
                    k = json.loads(line).get("key")
                except ValueError:
                    continue
                if k and k not in seen:
                    seen.add(k)
                    out.append(k)
    return out


def _derive_key(req_text):
    r = json.loads(req_text)
    return LLMClient.cache_key(r["model"], r["params"], r["messages"])


def _row_to_rec(row):
    key, model, ts, req, resp, pt, ct, cached, lat, mode = row
    req_text = zlib.decompress(req).decode("utf-8")
    return {"key": key, "model": model, "ts": ts, "req": req_text, "resp": resp, "pt": pt, "ct": ct, "cached_tokens": cached,
            "latency": lat, "mode": mode}


def _rec_to_row(rec):
    blob = zlib.compress(rec["req"].encode("utf-8"), 3)
    return (rec["key"], rec["model"], rec["ts"], blob, rec["resp"], rec["pt"], rec["ct"], rec["cached_tokens"], rec["latency"], rec["mode"])


def read_archive(path):
    """-> list of records (raises ValueError on a malformed line or a key that does not match its request)."""
    recs = []
    with gzip.open(path, "rb") as f:
        for n, line in enumerate(io.TextIOWrapper(f, encoding="utf-8"), 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if _derive_key(rec["req"]) != rec["key"]:
                raise ValueError("%s line %d: key does not match request" % (path, n))
            recs.append(rec)
    return recs


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def _write_archive(path, recs):
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=6) as gz:      # mtime=0: byte-reproducible
            for rec in sorted(recs, key=lambda r: r["key"]):
                gz.write((json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    os.chmod(tmp, 0o444)
    os.replace(tmp, path)


def export_run(run_dir, cache_path=None, out_dir=None, store=None):
    """Archive the cache rows used by `run_dir` (see module doc). Returns a stats dict (also written as the manifest)."""
    out_dir = out_dir or run_dir
    keys = usage_keys(run_dir)
    store = store or _Shards(cache_path)
    arch = os.path.join(out_dir, ARCHIVE)
    have = {}
    if os.path.exists(arch):
        try:
            have = {r["key"]: r for r in read_archive(arch)}
        except Exception:                 # damaged previous archive: keep it aside, rebuild from the cache
            os.replace(arch, arch + ".bad-%d" % int(time.time()))
    n_new = n_missing = 0
    for k in keys:
        if k in have:
            continue
        row = store.get_full(k)
        if row is None:
            n_missing += 1
            continue
        try:
            rec = _row_to_rec(row)
            if _derive_key(rec["req"]) != k:
                raise ValueError("key mismatch")
        except Exception:
            n_missing += 1
            continue
        have[k] = rec
        n_new += 1
    if not keys and not have:
        return {"n_keys": 0, "n_archived": 0, "n_new": 0, "n_missing": 0, "archive": None}
    _write_archive(arch, list(have.values()))
    man = {"n_keys": len(keys), "n_archived": len(have), "n_new": n_new, "n_missing_in_cache": n_missing, "archive": ARCHIVE,
           "archive_sha256": _sha256_file(arch), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    mp = os.path.join(out_dir, MANIFEST)
    with open(mp + ".tmp", "w") as f:
        json.dump(man, f, indent=1)
    os.replace(mp + ".tmp", mp)
    return man


def find_archives(targets):
    out = []
    for t in targets:
        if os.path.isdir(t):
            for root, _, files in os.walk(t):
                if ARCHIVE in files:
                    out.append(os.path.join(root, ARCHIVE))
        elif os.path.exists(t):
            out.append(t)
    return sorted(set(out))


def import_archives(targets, cache_path):
    """Warm a cache with archive rows (INSERT OR IGNORE: an existing row wins). Returns {"archives","records","inserted_attempts","bad"}."""
    store = _Shards(cache_path)
    res = {"archives": 0, "records": 0, "already": 0, "inserted": 0, "bad": 0}
    for p in find_archives(targets):
        res["archives"] += 1
        try:
            recs = read_archive(p)
        except Exception as e:
            res["bad"] += 1
            res.setdefault("errors", []).append("%s: %s" % (p, str(e)[:150]))
            continue
        for rec in recs:
            res["records"] += 1
            if store.get(rec["key"]) is not None:
                res["already"] += 1
                continue
            if store.put_raw(_rec_to_row(rec)):
                res["inserted"] += 1
    return res


def verify_archive(path):
    """-> {"ok", "n", "error"}; checks every key against its request and the manifest hash if a manifest sits next to it."""
    try:
        recs = read_archive(path)
    except Exception as e:
        return {"ok": False, "n": 0, "error": str(e)[:200]}
    mp = os.path.join(os.path.dirname(path), MANIFEST)
    if os.path.exists(mp):
        try:
            m = json.load(open(mp))
            if m.get("archive_sha256") != _sha256_file(path):
                return {"ok": False, "n": len(recs), "error": "sha256 differs from manifest"}
        except ValueError:
            return {"ok": False, "n": len(recs), "error": "manifest unreadable"}
    return {"ok": True, "n": len(recs), "error": None}


def finish_run(run_dir, cache_path=None, log=print, store=None):
    """Best-effort call at the end of a runner: never raises."""
    try:
        man = export_run(run_dir, cache_path, store=store)
        log("[cache] archived %s rows (%s not in cache) -> %s/%s" % (man.get("n_archived"), man.get("n_missing_in_cache", 0), run_dir, ARCHIVE))
        return man
    except Exception as e:
        log("[cache] archive export failed (non-fatal): %s: %s" % (type(e).__name__, str(e)[:150]))
        return None
