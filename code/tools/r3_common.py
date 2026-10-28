"""Shared helpers of the round-3 post hoc analyses (internal note R3EXP, SAP section 26): run map, loaders, alignment, summary-file writer.
numpy only; runs on a worker machine (host python 3.6 has no numpy). Imports analysis_pipeline.{runs,stats} read-only (same bootstrap machinery as RV2)."""
import collections
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from analysis_pipeline import runs as R, stats as S  # noqa: E402

B_DEFAULT = 10000
SEED_BOOT = S.SEED_BOOT
MARGIN = 1.5
BK = ("plus", "flash", "27b")
ARMS = ("zs", "rootonly", "gated", "fixed")
SEEDS = (0, 101, 202, 303)
REP = (101, 202, 303)            # the replication: three runs at T = 0.7, reported apart from the headline run (seed 0, T = 0), never averaged with it
MODEL_KEY = {"plus": "qwen37plus", "flash": "qwen37flash", "27b": "qwen3527b"}

# headline run (seed 0, T = 0) of every arm: the paths of rv2_analyze.RUNS
S0 = {("plus", "zs"): "MIN/M1/zs-s0", ("plus", "rootonly"): "MIN/M3/rootonly-s0", ("plus", "gated"): "MIN/M1/gated-s0", ("plus", "fixed"): "MIN/M1/fixed-s0"}
for _b in ("flash", "27b"):
    for _a in ARMS:
        S0[(_b, _a)] = "MIN/M5/%s-%s-s0" % (_a, _b)
# seeds 101 / 202 / 303 (T = 0.7). Plus single call and gated loop at 101 / 202 exist since the stability runs of MIN/M1; everything else is the R3 / R3B queues
SEEDRUN = {}
for _b in BK:
    for _a in ARMS:
        for _s in REP:
            SEEDRUN[(_b, _a, _s)] = "R3/%s-%s-s%d" % (_a, _b, _s)
for _s in (101, 202):
    SEEDRUN[("plus", "zs", _s)] = "MIN/M1/zs-s%d" % _s
    SEEDRUN[("plus", "gated", _s)] = "MIN/M1/gated-s%d" % _s
EXRUN = {(b, s): ("R3/zsex-%s-s%d" % (b, s)) for b in BK for s in SEEDS}


def run_path(b, arm, seed):
    if seed == 0:
        return S0[(b, arm)]
    return SEEDRUN[(b, arm, seed)]


def jl(path):
    return R.jl(path)


def load_preds(runs_root, rel):
    """qid -> row of preds.jsonl, or None if the run is absent."""
    p = os.path.join(runs_root, rel, "preds.jsonl")
    if not os.path.exists(p):
        return None
    d = {}
    for r in jl(p):
        d[r["qid"]] = r
    return d


def ex_of(row):
    return R.ex_value(row)


def check_run(rows, n_expected=1534):
    """-> (ok, note). A run enters a seed analysis only if complete and free of api_fail / sys_error rows."""
    if rows is None:
        return False, "absent"
    bad = sum(1 for r in rows.values() if r.get("error_class") == "api_fail" or str(r.get("error_class")).startswith("sys_error"))
    if len(rows) < n_expected:
        return False, "incomplete (%d rows)" % len(rows)
    if bad:
        return False, "%d api_fail/sys_error rows" % bad
    return True, "ok"


def pp(x):
    return None if x is None else 100.0 * float(x)


def fnum(x):
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else x


class Summary(object):
    """results/r3/*.json writer: {"_meta": {...}, "keys": {key: {"v", "kind", "ci", "item_ci", "n", "meaning"}}} (RV2 format)."""

    def __init__(self, scope="posthoc-secondary", api_calls=0, extra=None):
        self.meta = {"scope": scope, "api_calls": api_calls, "B": B_DEFAULT, "seed_boot": SEED_BOOT, "created": None}
        if extra:
            self.meta.update(extra)
        self.keys = collections.OrderedDict()

    def put(self, key, v, kind, ci=None, item_ci=None, n=None, meaning=None):
        def _f(x):
            if isinstance(x, (list, tuple)):
                return [fnum(i) for i in x]
            if isinstance(x, (np.floating, float)):
                return fnum(x)
            if isinstance(x, np.integer):
                return int(x)
            if isinstance(x, np.bool_):
                return bool(x)
            return x
        e = {"v": _f(v), "kind": kind}
        if ci is not None:
            e["ci"] = _f(ci)
        if item_ci is not None:
            e["item_ci"] = _f(item_ci)
        if n is not None:
            e["n"] = int(n)
        if meaning:
            e["meaning"] = meaning
        self.keys[key] = e

    def dump(self, path, extra_meta=None):
        import time
        self.meta["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if extra_meta:
            self.meta.update(extra_meta)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"_meta": self.meta, "keys": self.keys}, f, indent=1, sort_keys=False)


def reading(lo, hi):
    """SUP(0) reading of SAP 23.3: positive iff lo > 0, negative iff hi < 0, else contains 0."""
    if lo is None or hi is None:
        return "n/a"
    if lo > 0:
        return "positive"
    if hi < 0:
        return "negative"
    return "contains 0"


def boot_ci(vec, dbs, B=B_DEFAULT, boot=None):
    """(estimate, cluster CI, item CI) of mean(vec) in percentage points."""
    vec = np.asarray(vec, float)
    cb = boot or S.ClusterBoot(dbs, B=B)
    est = 100.0 * vec.mean()
    ci = tuple(100.0 * x for x in S.pctl_ci(cb.means(vec)))
    ic = tuple(100.0 * x for x in S.pctl_ci(S.item_boot_means(vec, B=B)))
    return est, ci, ic


def hull(*cis):
    lo = min(c[0] for c in cis)
    hi = max(c[1] for c in cis)
    return (lo, hi)


def write_csv(path, header, rows):
    """plain CSV (floats with 4 decimals); rows are lists in header order."""
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([("%.4f" % x) if isinstance(x, float) else ("" if x is None else x) for x in r])


def ci2(x):
    return (None, None) if x is None else (x[0], x[1])
