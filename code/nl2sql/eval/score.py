"""Evaluator: execution accuracy (BIRD semantics), soft-F1, Jaccard/overlap, auto error class.
Gold results are cached in the (machine-local) gold cache (sqlite, keyed by sha256(db_path|sql)); gold SQL is loaded through
data.load_gold, which logs locked-split access.

Timeout policy (deterministic w.r.t. IO state, X4):
  * GOLD is executed once with a generous timeout (`gold_timeout_s`, default 120 s) and the outcome is cached. It is `gold_fail`
    (EX = 0, stays in the denominator) only if it errors or exceeds 120 s. A cached *timeout* failure recorded with a smaller limit is
    recomputed; real SQL errors are cached for good.
  * PREDICTIONS get `timeout_s` (default 30 s); a timeout while the database file was cold (not touched by this process recently) is
    retried once (db.execute_robust), a warm database is not retried.
  * Gold results whose pickle would exceed ~64 MB (NL2SQL_GOLD_MAX_BYTES) are not cached as rows: a stub with row count and an
    order-insensitive digest is cached instead; EX works through the digest, soft-F1 / Jaccard / overlap are None for such items."""
import hashlib
import os
import pickle

from .. import data, paths, sqlcache
from ..db import execute, execute_robust, ex_match
from .bird_official import calculate_f1_score
from .errors import classify
from . import spider_ts


GOLD_MAX_BYTES = int(os.environ.get("NL2SQL_GOLD_MAX_BYTES", str(64 * 1024 * 1024)))   # results above this are not pickled
GOLD_TIMEOUT_S = 120.0       # generous: a gold query is executed once and its outcome cached
PRED_TIMEOUT_S = 30.0


class _GoldCache:
    """sqlite KV (self-healing, see sqlcache). get() returns None on a miss OR any trouble (corruption, unreadable pickle);
    put() returns False if the value was not stored (too big, db trouble)."""

    def __init__(self, path):
        self.path = str(path)
        self._db = sqlcache.SafeDB(self.path, ["CREATE TABLE IF NOT EXISTS g (key TEXT PRIMARY KEY, v BLOB)"])

    def get(self, key):
        r = self._db.query_one("SELECT v FROM g WHERE key=?", (key,))
        if not r:
            return None
        try:
            return pickle.loads(r[0])
        except Exception:
            sqlcache.bump("bad_rows")
            self._db.execute("DELETE FROM g WHERE key=?", (key,))
            return None

    def put(self, key, v):
        blob = pickle.dumps(v, 4)
        if len(blob) > GOLD_MAX_BYTES:
            return False
        return self._db.execute("INSERT OR IGNORE INTO g VALUES (?,?)", (key, blob))


_GC = None


def gold_cache():
    global _GC
    if _GC is None:
        _GC = _GoldCache(paths.CACHE / "gold_exec.sqlite")
    return _GC


def _norm(v):
    return int(v) if isinstance(v, float) and v.is_integer() else v


def rows_digest(rows):
    """Order- and duplicate-insensitive digest of a result set: digest(A) == digest(B) <=> set(A) == set(B) (with overwhelming
    probability; integral floats are normalised to ints, as in Python's 1 == 1.0). Used instead of the rows for oversize gold results."""
    seen = set(tuple(_norm(v) for v in r) for r in rows)
    h = 0
    for t in seen:
        h = (h + int.from_bytes(hashlib.sha1(repr(t).encode("utf-8", "replace")).digest()[:16], "big")) % (1 << 128)
    return "%d:%032x" % (len(seen), h)


def compact_gold(r, max_bytes=None):
    """Oversize gold result (pickle > max_bytes) -> stub {ok, rows: None, big: True, n_rows, rows_digest, cols}; else r unchanged.
    The stub supports EX (digest compare) but not soft-F1 / Jaccard / overlap (reported as None)."""
    max_bytes = GOLD_MAX_BYTES if max_bytes is None else max_bytes
    if not r.get("ok") or not r.get("rows"):
        return r
    rows = r["rows"]
    n = len(rows)
    smp = rows[:200]
    est = sum(len(repr(x)) for x in smp) / float(len(smp)) * n
    if est <= max_bytes / 4.0 and len(pickle.dumps(r, 4)) <= max_bytes:
        return r
    return {"ok": True, "rows": None, "cols": r.get("cols", []), "error": None, "truncated": False, "big": True, "n_rows": n,
            "rows_digest": rows_digest(rows)}


def jaccard_overlap(pred, gold):
    """pred/gold: execute dicts. Returns (jaccard, overlap)."""
    if not pred["ok"] or not gold["ok"]:
        return 0.0, 0.0
    P, G = set(map(tuple, pred["rows"])), set(map(tuple, gold["rows"]))
    if not P and not G:
        return 1.0, 1.0
    inter = len(P & G)
    j = inter / len(P | G)
    mn = min(len(P), len(G))
    return j, (inter / mn if mn else 0.0)


class Evaluator:
    def __init__(self, split, run_id="", timeout_s=PRED_TIMEOUT_S, gold_timeout_s=GOLD_TIMEOUT_S):
        self.split = split
        self.timeout_s = timeout_s
        self.gold_timeout_s = max(gold_timeout_s, timeout_s)
        self.gold = data.load_gold(split, run_id=run_id)
        self.gold2 = {}
        if split == "ambiqt_mix":      # two valid interpretations: EX = match with either (declared proxy)
            self.gold2 = {i.qid: i.meta.get("gold_sql_2", "") for i in data.load_split(split, with_gold=True)}
        self.items = {}
        # Spider splits additionally get the official test-suite EX (`ex_ts`); NL2SQL_TS=0 switches it off
        self.use_ts = split in spider_ts.TS_SPLITS and os.environ.get("NL2SQL_TS", "1") != "0"

    def _ts(self, item, sql, pred_res):
        """{"ex_ts": 0|1|None} for Spider splits, {} otherwise (additive field)."""
        if not self.use_ts:
            return {}
        g = self.gold.get(item.qid, "")
        if not g or not spider_ts.has_suite(item.db_id):
            return {"ex_ts": None}
        if not pred_res["ok"]:
            return {"ex_ts": 0}
        r = spider_ts.shared_pool().match(item.db_id, sql, g)
        return {"ex_ts": r["ex_ts"], "ts_error": r["error"]} if r["error"] else {"ex_ts": r["ex_ts"]}

    def gold_exec(self, item):
        g = self.gold.get(item.qid, "")
        if not g:
            return None
        key = hashlib.sha256((item.db_path + "\x1f" + g).encode()).hexdigest()
        r = gold_cache().get(key)
        if r is not None and not r.get("ok") and str(r.get("error") or "").startswith("timeout") \
                and r.get("gold_timeout_s", 0) < self.gold_timeout_s:
            r = None                      # cached under a smaller limit: not final
        if r is None:
            r = execute(item.db_path, g, timeout_s=self.gold_timeout_s, max_rows=None)
            r.pop("elapsed_s", None)
            if r["ok"]:
                r = compact_gold(r)
            else:
                r["gold_timeout_s"] = self.gold_timeout_s
            gold_cache().put(key, r)
        return r

    def score_sql(self, item, sql, pred_res=None):
        """-> dict(ex, exec_ok, soft_f1, jaccard, overlap, gold_fail, auto_class, pred_rows). ex None if no gold."""
        if pred_res is None:
            pred_res = execute_robust(item.db_path, sql, timeout_s=self.timeout_s, max_rows=None)
        g = self.gold_exec(item)
        out = {"exec_ok": bool(pred_res["ok"]), "pred_error": pred_res.get("error")}
        if g is None:
            out.update({"ex": None, "soft_f1": None, "jaccard": None, "overlap": None, "gold_fail": None, "auto_class": None})
            out.update(self._ts(item, sql, pred_res))
            return out
        if not g["ok"]:
            out.update({"ex": 0, "soft_f1": 0.0, "jaccard": 0.0, "overlap": 0.0, "gold_fail": True, "auto_class": "gold_fail"})
            out.update(self._ts(item, sql, pred_res))
            return out
        if not pred_res["ok"]:
            ex, f1, (j, o) = 0, 0.0, (0.0, 0.0)
        elif g.get("big"):
            ex = int(rows_digest(pred_res["rows"]) == g["rows_digest"])
            f1 = j = o = None
        else:
            ex = ex_match(pred_res["rows"], g["rows"])
            g2 = self.gold2.get(item.qid)
            if g2 and not ex:
                r2 = execute(item.db_path, g2, timeout_s=self.gold_timeout_s, max_rows=None)
                if r2["ok"]:
                    ex = ex_match(pred_res["rows"], r2["rows"])
            try:
                f1 = calculate_f1_score(pred_res["rows"], g["rows"])
            except Exception:
                f1 = 0.0
            j, o = jaccard_overlap(pred_res, g)
        big = bool(g.get("big"))
        out.update({"ex": ex, "soft_f1": None if big else round(f1, 6), "jaccard": None if big else round(j, 6),
                    "overlap": None if big else round(o, 6), "gold_fail": False,
                    "auto_class": None if ex else ("gold_big" if big and pred_res["ok"] else classify(pred_res, g))})
        out.update(self._ts(item, sql, pred_res))
        return out
