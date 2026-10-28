"""R1 shared helpers. Runs on the worker machine (py3.12 venv: numpy/scipy/sklearn). Reads ONLY tuning data (HT-300, HR); never dev.
Paths are the worker machine shared disk; outputs go to $R/r1/out (own dir)."""
import json, os, collections, math
import numpy as np

R = os.environ.get("NL2SQL_ROOT", os.path.expanduser("~/nl2sql_work"))
OUT = os.path.join(R, "r1", "out")
os.makedirs(OUT, exist_ok=True)
HR_RUN = R + "/runs/P2_HR/P2_HR-full_fixed-qwen37plus-s0-38ee3ec2"
HT_RUN = R + "/runs/P2_base/P2_base-full_fixed-qwen37plus-s0-d37d2097"
HT_ZS = R + "/runs/P2_base/P2_base-zs_ours-qwen37plus-s0-80485cc1"
HT_ZS_B1 = R + "/runs/P2_base/P2_base-zs-qwen37plus-s0-80485cc1"
TREES = R + "/runs/P2_E6/trees_1101.jsonl"
SPLIT = R + "/exp/splits/train_holdout.json"
TRAIN = R + "/data/bird/train/train/train.json"
SEED = 20261101


def jl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def by_qid(rows):
    return {r["qid"]: r for r in rows}


def cluster_boot(dbs, stat_fn, B=2000, seed=SEED):
    """db-clustered bootstrap. dbs: array of db ids per item (aligned with whatever stat_fn indexes).
    stat_fn(idx_array) -> float. Resample databases with replacement; all items of a drawn db kept."""
    rng = np.random.RandomState(seed)
    dbs = np.asarray(dbs)
    u = np.unique(dbs)
    members = {d: np.where(dbs == d)[0] for d in u}
    est = stat_fn(np.arange(len(dbs)))
    vals = []
    for _ in range(B):
        draw = rng.choice(u, size=len(u), replace=True)
        idx = np.concatenate([members[d] for d in draw])
        try:
            v = stat_fn(idx)
        except Exception:
            v = float("nan")
        vals.append(v)
    vals = np.array(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return est, float(lo), float(hi)


def fmt(ci, d=3, pct=False):
    e, lo, hi = ci
    if pct:
        return "%.1f [%.1f, %.1f]" % (100 * e, 100 * lo, 100 * hi)
    return "%.*f [%.*f, %.*f]" % (d, e, d, lo, d, hi)


def auroc(score, y):
    """AUROC of score for predicting y=1 (ties = 1/2). Returns nan if one class empty."""
    score = np.asarray(score, float); y = np.asarray(y, int)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(score)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def dump(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
