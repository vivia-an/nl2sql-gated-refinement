"""Self-healing sqlite helpers for the on-disk caches (LLM call cache, embedding/rerank cache, gold-execution cache).

Why (R0 CRITICAL 2, 2026-10-26): several processes on several worker machines wrote the same sqlite files on the shared cpfs volume, whose file
locks are not coherent across mounts -> 13/16 call-cache shards became "database disk image is malformed", and every read raised
DatabaseError, which the runner turned into `sys_error` rows with EX = 0.  Rules implemented here:

  * the cache directory is machine-local by default (paths.CACHE); a path on a shared volume (NL2SQL_SHARED_FS_PREFIXES) gets a loud warning and the
    old DELETE journal (WAL is unsafe on network file systems);
  * on a machine-local disk: WAL + busy_timeout (several threads/processes of ONE worker machine may share a file);
  * startup integrity check (PRAGMA quick_check) of every existing file: a malformed file is QUARANTINED (renamed to
    `<name>.corrupt-<time>-<pid>`, with its -wal/-shm) and a fresh file is created;
  * a DatabaseError at run time is never raised to the caller: corruption -> quarantine + reconnect; anything else -> bounded
    retry; in the end the call degrades to a cache MISS (reads) or a dropped write, is counted in `STATS` and logged.
    A cache problem can therefore cost money (a re-issued call) but can never change a result or yield EX = 0.
Env: NL2SQL_SQLITE_JOURNAL (force a journal mode), NL2SQL_CACHE_CHECK=0 (skip the startup check)."""
import logging
import os
import sqlite3
import sys
import threading
import time

log = logging.getLogger("nl2sql.cache")
STATS = {"db_errors": 0, "quarantined": 0, "dropped_writes": 0, "miss_on_error": 0, "bad_rows": 0}
_STATS_LOCK = threading.Lock()
_WARNED = set()

_CORRUPT_MARKERS = ("malformed", "not a database", "corrupt", "disk image", "file is encrypted")
_NO_RETRY = (sqlite3.DataError, sqlite3.IntegrityError, sqlite3.ProgrammingError, sqlite3.NotSupportedError)


def bump(k, n=1):
    with _STATS_LOCK:
        STATS[k] = STATS.get(k, 0) + n


def stats():
    with _STATS_LOCK:
        return dict(STATS)


def _warn_once(key, msg):
    if key in _WARNED:
        return
    _WARNED.add(key)
    log.info(msg)
    print("[cache] " + msg, file=sys.stderr, flush=True)


def is_shared_volume(path):
    pre = [p for p in os.environ.get("NL2SQL_SHARED_FS_PREFIXES", "").split(",") if p]     # e.g. "/mnt/shared/"
    return bool(pre) and os.path.realpath(str(path)).startswith(tuple(pre))


def journal_mode_for(path):
    forced = os.environ.get("NL2SQL_SQLITE_JOURNAL")
    if forced:
        return forced.upper()
    return "DELETE" if is_shared_volume(path) else "WAL"


def is_corruption(exc):
    msg = str(exc).lower()
    return isinstance(exc, sqlite3.DatabaseError) and any(m in msg for m in _CORRUPT_MARKERS)


def check_file(path, timeout=60.0):
    """(ok, detail). A busy/locked file is reported ok (cannot judge); only a definite corruption is not ok."""
    try:
        c = sqlite3.connect(path, timeout=timeout)
        try:
            rows = c.execute("PRAGMA quick_check").fetchall()
        finally:
            c.close()
    except sqlite3.OperationalError as e:
        if is_corruption(e):
            return False, str(e)
        return True, "unchecked: %s" % e
    except sqlite3.DatabaseError as e:
        return False, str(e)
    if len(rows) == 1 and rows[0][0] == "ok":
        return True, "ok"
    return False, "; ".join(str(r[0]) for r in rows[:3])


def quarantine(path, expect_ino=None, reason=""):
    """Rename a bad database (+ -wal/-shm/-journal) out of the way. If `expect_ino` is given and the file at `path` is a different
    file by now (another process already replaced it), nothing is moved. Returns True if this call moved the file."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return False
    if expect_ino is not None and st.st_ino != expect_ino:
        return False
    dst = "%s.corrupt-%s-%d" % (path, time.strftime("%Y%m%d-%H%M%S"), os.getpid())
    moved = False
    for suf in ("", "-wal", "-shm", "-journal"):
        try:
            os.rename(path + suf, dst + suf)
            moved = moved or suf == ""
        except FileNotFoundError:
            pass
        except OSError as e:
            log.error("quarantine rename %s failed: %s", path + suf, e)
    if moved:
        bump("quarantined")
        _warn_once("q:" + dst, "QUARANTINED malformed cache file %s -> %s (%s)" % (path, dst, str(reason)[:120]))
    return moved


class SafeDB:
    """One sqlite file, one connection per thread, never raises on cache trouble (see module docstring)."""

    def __init__(self, path, schema_sqls=(), check=None, retries=6, timeout=120.0):
        self.path = str(path)
        self.schema = list(schema_sqls)
        self.retries = retries
        self.timeout = timeout
        self._tl = threading.local()
        self._lock = threading.Lock()
        self._gen = 0
        self.last_error = None
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        if is_shared_volume(self.path):
            _warn_once("shared:" + d, "cache file %s is on a shared volume: concurrent writers from several machines can corrupt it. "
                                      "Use a local NL2SQL_CACHE and archive with tools/cache_export.py" % self.path)
        if check is None:
            check = os.environ.get("NL2SQL_CACHE_CHECK", "1") != "0"
        if check:
            self.startup_check()

    def startup_check(self):
        if not os.path.exists(self.path):
            return True
        ok, detail = check_file(self.path)
        if not ok:
            quarantine(self.path, reason=detail)
        return ok

    # -- connection management
    def _conn(self):
        ent = getattr(self._tl, "ent", None)
        if ent is not None and ent[0] == self._gen:
            return ent[1], ent[2]
        if ent is not None:
            try:
                ent[1].close()
            except Exception:
                pass
        gen = self._gen
        c = sqlite3.connect(self.path, timeout=self.timeout, isolation_level=None)
        c.execute("PRAGMA busy_timeout=%d" % int(self.timeout * 1000))
        jm = journal_mode_for(self.path)
        try:
            c.execute("PRAGMA journal_mode=%s" % jm)
        except sqlite3.OperationalError:
            pass                                   # another process holds the file: keep its mode
        c.execute("PRAGMA synchronous=%s" % ("NORMAL" if jm == "WAL" else "OFF"))
        for s in self.schema:
            c.execute(s)
        ino = os.stat(self.path).st_ino
        self._tl.ent = (gen, c, ino)
        return c, ino

    def _heal(self, ino, exc):
        with self._lock:
            quarantine(self.path, expect_ino=ino, reason=exc)
            self._gen += 1                         # every thread reconnects (to the fresh file)

    def _do(self, fn, what):
        last = None
        for attempt in range(self.retries):
            ino = None
            try:
                c, ino = self._conn()
                return True, fn(c)
            except sqlite3.Error as e:
                last = e
                bump("db_errors")
                log.warning("cache %s on %s failed (attempt %d): %s", what, self.path, attempt + 1, e)
                if is_corruption(e):
                    self._heal(ino, e)
                elif isinstance(e, _NO_RETRY):
                    break
                else:
                    time.sleep(min(0.5 * (attempt + 1), 3.0))
        self.last_error = last
        return False, None

    # -- API
    def query_one(self, sql, args=()):
        """first row or None; None also on error (counted as miss_on_error)."""
        ok, row = self._do(lambda c: c.execute(sql, args).fetchone(), "read")
        if not ok:
            bump("miss_on_error")
            return None
        return row

    def execute(self, sql, args=()):
        ok, _ = self._do(lambda c: c.execute(sql, args), "write")
        if not ok:
            bump("dropped_writes")
        return ok

    def iter_rows(self, sql, args=()):
        ok, rows = self._do(lambda c: c.execute(sql, args).fetchall(), "scan")
        return rows if ok else []
