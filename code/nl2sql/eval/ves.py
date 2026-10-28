"""VES / R-VES: execution-time measurement and per-item values.

Two timing conventions are implemented, each mirroring the script it comes from (cross-checked in
tests/crosscheck_ves_official.py against the vendored official files):
  * "official": DAMO-ConvAI bird/llm/src/evaluation_ves.py  -> new connection, time ONLY `cursor.execute(sql)`
                (no fetch), ratio_i = gold_time_i / pred_time_i, 3-sigma trimmed mean over `iters` repetitions.
  * "minidev":  bird-bench/mini_dev evaluation/evaluation_ves.py -> time of execute + fetchall + close.
VES   (headline, "official" timing): per item sqrt(time_ratio) * 100, 0 if the item is not EX-correct.
R-VES (Mini-Dev, "minidev" timing):  per item sqrt(reward(time_ratio)) * 100 with the reward bands of the
                Mini-Dev script (`bird_official.rves_reward`), 0 if not EX-correct.
Only SQL execution is timed (no API latency is involved anywhere in this module).

Deviations from the official scripts (all deliberate, none changes the definition of the metric):
  * connections are read-only/immutable (`db._connect`), the official scripts open the file read-write;
  * `time.perf_counter` instead of `time.time`;
  * each single execution has a hard timeout (default 30 s, sqlite progress handler); the official scripts use
    func_timeout(30 * iters) around the whole item. A timeout/error gives time_ratio 0 in both;
  * if the 3-sigma filter leaves nothing (identical timings, std = 0; the official code then divides by zero and
    the exception path scores the item 0), the plain mean is used and `trim_fallback` is set;
  * the EX gate is the EX already stored in preds.jsonl (ex == 1), not a second comparison.
NOTE (statistics): with the population std used by the official `clean_abnormal`, |z| <= sqrt(n-1) for n samples, so
the 3-sigma rule can only remove a point when n >= 11. With iters = 5 the "trimmed mean" is the plain mean.
`trim="minmax"` (drop the largest and smallest ratio) is offered as a robust alternative; it is NOT the official rule.
"""
import math
import sqlite3
import time

import numpy as np

from ..db import _connect
from .bird_official import clean_abnormal, rves_reward, ves_value

TIMING_MODES = ("official", "minidev")


class _Timeout(Exception):
    pass


def time_once(db_path, sql, timing="official", timeout_s=30.0):
    """Wall time (s) of one execution; raises sqlite3.Error / _Timeout on failure."""
    conn = None
    try:
        conn = _connect(db_path)
        deadline = time.perf_counter() + timeout_s
        conn.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 20000)
        cur = conn.cursor()
        if timing == "official":
            t0 = time.perf_counter()
            cur.execute(sql)
            dt = time.perf_counter() - t0
        elif timing == "minidev":
            t0 = time.perf_counter()
            cur.execute(sql)
            cur.fetchall()
            conn.close()
            conn = None
            dt = time.perf_counter() - t0
        else:
            raise ValueError(timing)
        return dt
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e).lower():
            raise _Timeout(str(e))
        raise
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def trimmed_mean(ratios, trim="sigma3"):
    """Returns (value, fallback_used)."""
    if trim == "minmax":
        r = sorted(ratios)
        r = r[1:-1] if len(r) >= 3 else r
        return sum(r) / len(r), False
    kept = clean_abnormal(ratios)
    if not kept:
        return float(sum(ratios) / len(ratios)), True
    return sum(kept) / len(kept), False


def measure_ratio(db_path, pred_sql, gold_sql, iters=5, timing="official", trim="sigma3", timeout_s=30.0, gold_times=None):
    """time_ratio of one EX-correct item = trimmed mean over `iters` of gold_time / pred_time.
    gold_times: optional list of previously measured gold times (len >= iters) reused instead of re-timing the gold
    (SPEC: gold timings measured once per batch). Returns dict(time_ratio, pred_times, gold_times, fail, trim_fallback).
    Never raises; failures give time_ratio 0 (as in the official scripts)."""
    pt, gt, ratios = [], [], []
    try:
        for i in range(iters):
            p = time_once(db_path, pred_sql, timing, timeout_s)
            g = gold_times[i] if gold_times is not None else time_once(db_path, gold_sql, timing, timeout_s)
            pt.append(p)
            gt.append(g)
            ratios.append(g / p)
        val, fb = trimmed_mean(ratios, trim)
        return {"time_ratio": val, "pred_times": pt, "gold_times": gt, "fail": None, "trim_fallback": fb}
    except Exception as e:      # incl. ZeroDivisionError (p == 0), timeout, sqlite errors
        return {"time_ratio": 0.0, "pred_times": pt, "gold_times": gt, "fail": "%s: %s" % (type(e).__name__, str(e)[:120]),
                "trim_fallback": False}


def item_ves(time_ratio):
    return ves_value(time_ratio) if time_ratio else 0.0


def item_rves(time_ratio):
    return math.sqrt(rves_reward(time_ratio)) * 100 if time_ratio else 0.0


def aggregate(values):
    """Mean over ALL items (items that are not EX-correct enter as 0)."""
    return float(np.mean(values)) if len(values) else None
