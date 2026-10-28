"""Small shared helpers: result files, provenance, key helpers."""
import hashlib
import json
import math
import os
import subprocess
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def sha256_file(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_hash(repo=REPO):
    try:
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        p = os.path.join(repo, "code", "GIT_HASH.txt")
        return open(p).read().strip() if os.path.exists(p) else "unknown"


def clean(v):
    """JSON-safe scalar: numpy -> python, NaN/inf -> None."""
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return [clean(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    return v


# Scope of a results file (CONTRACT 0 / SAP 13.0): only "confirmatory" and "tuning-export" files may sit in the top-level results/ directory
# and be read by make_macros. "tuning-set" = analysis of BIRD-train HR/HT runs (dry runs, design evidence); "synthetic" = self-test data.
SCOPES = ("confirmatory", "tuning-export", "tuning-set", "synthetic")
PAPER_SCOPES = ("confirmatory", "tuning-export")
_DEFAULT_SCOPE = ["confirmatory"]


def set_default_scope(scope):
    assert scope in SCOPES, scope
    _DEFAULT_SCOPE[0] = scope


def detect_scope(runs_root, split, dev_split="bird_dev"):
    """synthetic if <runs_root>/SYNTHETIC exists; tuning-set if the split is not the dev split; else confirmatory."""
    if runs_root and os.path.exists(os.path.join(runs_root, "SYNTHETIC")):
        return "synthetic"
    return "confirmatory" if split == dev_split else "tuning-set"


class Result(object):
    """Flat key/value collector for one results/<name>.json."""

    def __init__(self, name, scope=None):
        self.name = name
        self.d = {}
        self.scope = scope or _DEFAULT_SCOPE[0]
        self.meta = {"name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_hash(), "notes": [], "scope": self.scope}

    def set(self, key, value):
        v = clean(value)
        if v is None:
            return                              # never write null: a missing key means "not produced" (make_macros reports it)
        self.d[key] = v

    def update(self, d, prefix=""):
        for k, v in d.items():
            self.set(prefix + k, v)

    def ci(self, prefix, mean, lo, hi, scale=1.0):
        self.set(prefix + ".mean", None if mean is None else mean * scale)
        self.set(prefix + ".ci_lo", None if lo is None else lo * scale)
        self.set(prefix + ".ci_hi", None if hi is None else hi * scale)

    def note(self, msg):
        self.meta["notes"].append(msg)

    def write(self, path):
        top = os.path.abspath(os.path.join(REPO, "results"))
        if self.scope not in PAPER_SCOPES and os.path.dirname(os.path.abspath(path)) == top:
            raise SystemExit("refusing to write %s results into %s (scope %s must never reach the paper; use another --results directory)" % (self.scope, top, self.scope))
        self.meta["scope"] = self.scope
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        out = dict(self.d)
        out["_meta"] = self.meta
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
        os.replace(tmp, path)
        return path


def mean_se_ci(x):
    x = np.asarray(x, float)
    return float(x.mean())
