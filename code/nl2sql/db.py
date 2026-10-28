"""SQLite execution with a hard timeout (progress handler + interrupt timer), EX semantics, EXPLAIN.

EX follows the official BIRD evaluation.py: execute prediction and gold on the same database; correct iff
set(pred_rows) == set(gold_rows) (row tuples as returned by sqlite3; duplicates and row order ignored; column
order matters through the tuples). Any execution error or timeout of either query gives 0.
Databases are opened read-only (`mode=ro&immutable=1`), one connection per call (thread-safe)."""
import sqlite3
import threading
import time

from . import meter as _meter


def _connect(db_path):
    uri = "file:%s?mode=ro&immutable=1" % db_path.replace("?", "%3f").replace("#", "%23")
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


def _run(db_path, sql, timeout_s, max_rows):
    t0 = time.time()
    conn = None
    timer = None
    try:
        conn = _connect(db_path)
        deadline = t0 + timeout_s
        conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 20000)
        timer = threading.Timer(timeout_s + 0.5, conn.interrupt)
        timer.daemon = True
        timer.start()
        cur = conn.cursor()
        cur.execute(sql)
        if max_rows is None:
            rows = cur.fetchall()
            trunc = False
        else:
            rows = cur.fetchmany(max_rows + 1)
            trunc = len(rows) > max_rows
            rows = rows[:max_rows]
        cols = [d[0] for d in (cur.description or [])]
        return {"ok": True, "rows": rows, "cols": cols, "error": None, "elapsed_s": time.time() - t0, "truncated": trunc}
    except sqlite3.Error as e:
        el = time.time() - t0
        msg = str(e)
        if "interrupted" in msg.lower() or el >= timeout_s:
            msg = "timeout: " + msg
        return {"ok": False, "rows": [], "cols": [], "error": msg, "elapsed_s": el, "truncated": False}
    except Exception as e:  # e.g. recursion / memory errors from the driver
        return {"ok": False, "rows": [], "cols": [], "error": "%s: %s" % (type(e).__name__, e),
                "elapsed_s": time.time() - t0, "truncated": False}
    finally:
        if timer:
            timer.cancel()
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# db file -> time.monotonic() of the last query that FINISHED (no timeout) on it in THIS process (a proxy for "its pages are in the OS page cache")
_LAST_USE = {}
COLD_AFTER_S = 900.0


def is_cold(db_path):
    """True if this process has not touched `db_path` in the last COLD_AFTER_S seconds (first use, or long idle)."""
    t = _LAST_USE.get(db_path)
    return t is None or time.monotonic() - t > COLD_AFTER_S


def execute(db_path, sql, timeout_s=30, max_rows=10000):
    """{"ok","rows","error","elapsed_s","truncated"} (+ "cols"). max_rows=None: fetch everything (scoring)."""
    if not sql or not sql.strip():
        return {"ok": False, "rows": [], "cols": [], "error": "empty sql", "elapsed_s": 0.0, "truncated": False}
    r = _run(db_path, sql, timeout_s, max_rows)
    if r["ok"] or not str(r.get("error") or "").startswith("timeout"):
        _LAST_USE[db_path] = time.monotonic()      # a timed-out query proves nothing about the cache: the db stays "cold"
    m = _meter.current()
    if m:
        m.add_exec(r["elapsed_s"])
    return r


def execute_robust(db_path, sql, timeout_s=30, max_rows=None, retry_cold=True):
    """`execute` with a deterministic timeout policy for SCORING (EX labels must not depend on whether a multi-GB database happened to
    be in the OS page cache; R0 found 7 gold_fail + 1 flipped prediction on HR for exactly this reason):
      * the first attempt runs with `timeout_s`;
      * if it timed out AND the database was cold when the attempt started (`is_cold`: this process had not touched the file
        recently), it is retried ONCE with the same timeout - the first attempt has read the hot pages, so the retry measures the
        query rather than the disk;
      * a warm database is never retried; a retry that also times out is final ("timeout: ...").
    Result = `execute`'s dict plus "attempts" (1|2) and "cold_start"."""
    cold = is_cold(db_path)
    r = execute(db_path, sql, timeout_s=timeout_s, max_rows=max_rows)
    r["attempts"], r["cold_start"] = 1, cold
    if retry_cold and cold and not r["ok"] and str(r.get("error") or "").startswith("timeout"):
        first = r["elapsed_s"]
        r = execute(db_path, sql, timeout_s=timeout_s, max_rows=max_rows)
        r["attempts"], r["cold_start"], r["first_attempt_s"] = 2, True, first
    return r


def ex_match(pred_rows, gold_rows):
    """BIRD official: set equality of row tuples."""
    try:
        return int(set(map(tuple, pred_rows)) == set(map(tuple, gold_rows)))
    except TypeError:
        return int(sorted(map(repr, pred_rows)) == sorted(map(repr, gold_rows)))


def rows_key(rows):
    """Hashable key such that equal keys <=> ex_match==1 (used by execution voting)."""
    return frozenset(map(tuple, rows))


def explain(db_path, sql, timeout_s=10):
    """EXPLAIN QUERY PLAN detail lines ([] on error)."""
    r = _run(db_path, "EXPLAIN QUERY PLAN " + sql.strip().rstrip(";"), timeout_s, 1000)
    if not r["ok"]:
        return []
    return [str(row[3]) if len(row) >= 4 else str(row) for row in r["rows"]]


def error_kind(err):
    """syntax | identifier | timeout | other, from a sqlite error message."""
    if not err:
        return None
    e = err.lower()
    if e.startswith("timeout"):
        return "timeout"
    if "no such column" in e or "no such table" in e or "ambiguous column" in e or "no such function" in e:
        return "identifier"
    if "syntax error" in e or "incomplete input" in e or "unrecognized token" in e or "near " in e:
        return "syntax"
    return "other"


def row_counts(db_path):
    """{table: n_rows} (user tables only)."""
    out = {}
    r = _run(db_path, "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'", 10, 10000)
    for (t,) in r["rows"]:
        q = _run(db_path, 'SELECT COUNT(*) FROM "%s"' % t.replace('"', '""'), 30, 1)
        out[t] = q["rows"][0][0] if q["ok"] and q["rows"] else -1
    return out
