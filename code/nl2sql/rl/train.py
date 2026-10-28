"""E6 training protocol. Cells: c_h in {0.1,0.25,0.5,0.75} x lambda in {0,0.01,0.03} (12). Grouped 4-fold CV on HR (train 6 dbs,
validate 2) x 3 seeds, then final controllers on all HR (5 seeds per cell; deployed = seed with the median training return in the last 20 %
of updates). Baselines evaluated on the same trees. CPU only; ~1-3 min per training.

python -m nl2sql.rl.train cv    --trees trees.jsonl --out dir [--cells 12] [--seeds 3] [--updates 200] [--algo ppo]
python -m nl2sql.rl.train final --trees trees.jsonl --out dir [--seeds 5]
python -m nl2sql.rl.train base  --trees trees.jsonl --out dir
"""
import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .env import TreeSet, evaluate_policy
from .features import FEATURE_NAMES
from .ppo import export_npz, greedy_policy, train

KI = FEATURE_NAMES.index('k_over_tmax')
CH = [0.1, 0.25, 0.5, 0.75]
LAM = [0.0, 0.01, 0.03]
CELLS = list(itertools.product(CH, LAM))


def _job(args):
    trees, c_h, lam, fold, seed, algo, updates, a4, out = args
    ts_all = TreeSet.load(trees)
    qtr = [i for i, f in enumerate(ts_all.fold) if f != fold]
    qva = [i for i, f in enumerate(ts_all.fold) if f == fold]
    model, info = train(ts_all, c_h, lam, seed, algo, updates, a4=a4, q_subset=qtr)
    ev = evaluate_policy(ts_all, greedy_policy(model), qva, c_h=c_h, lam=lam, a4=a4)
    rec = {"c_h": c_h, "lam": lam, "fold": fold, "seed": seed, "algo": algo, "val": ev, "train_return_last20": info["train_return_last20"]}
    with open(os.path.join(out, "cv.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def _final(args):
    trees, c_h, lam, seed, algo, updates, a4, out = args
    ts = TreeSet.load(trees)
    model, info = train(ts, c_h, lam, seed, algo, updates, a4=a4)
    p = os.path.join(out, "ctl_%s_ch%s_lam%s_s%d.npz" % (algo, c_h, lam, seed))
    export_npz(model, p)
    ev = evaluate_policy(ts, greedy_policy(model), None, c_h=c_h, lam=lam, a4=a4, per_question=False)   # train-set eval: aggregates only
    rec = {"c_h": c_h, "lam": lam, "seed": seed, "algo": algo, "path": p, "train_eval": ev, "train_return_last20": info["train_return_last20"]}
    with open(os.path.join(out, "final.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def baselines(ts, out, c_h=0.5, lam=0.0):
    """P0 threshold grid, fixed-k revisions, greedy oracle, logistic-regression stopper (same 27 features)."""
    res = {}
    ACC, REV, RR, ESC, HUM = 0, 1, 2, 3, 4

    def p0(theta, tmax=3):
        def pol(s, mask):
            score = s[3] * 10.0
            k = int(round(s[KI] * 3))          # k/T_max feature index (see FEATURE_NAMES)
            if score >= theta or k >= tmax or not mask[REV]:
                return ACC
            return REV
        return pol
    for th in [7.0, 7.5, 8.0, 8.5, 9.0, 9.5]:
        res["P0_theta%.1f" % th] = evaluate_policy(ts, p0(th), None, c_h, lam)
    for k in range(4):
        def fk(s, mask, k=k):
            kk = int(round(s[KI] * 3))
            return REV if (kk < k and mask[REV]) else ACC
        res["fixed_k%d" % k] = evaluate_policy(ts, fk, None, c_h, lam)
    # LR stopper: label = "a revise improves EX_new > EX_cur" estimated on the trees' revise children
    X, y = [], []
    for n in range(len(ts.state)):
        ch = ts.children[n].get(REV)
        if ch is not None:
            X.append(ts.state[n])
            y.append(float(sum(w * ts.ex[c] for c, w in zip(*ch)) > ts.ex[n]))
    if len(set(y)) > 1:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=500, C=1.0).fit(np.array(X), np.array(y))
        for thr in [0.3, 0.5, 0.7]:
            def pol(s, mask, thr=thr):
                return REV if (mask[REV] and lr.predict_proba(s.reshape(1, -1))[0, 1] > thr) else ACC
            res["LR_stop_thr%.1f" % thr] = evaluate_policy(ts, pol, None, c_h, lam)
    # greedy oracle: DP over the tree with perfect knowledge (upper bound)
    def oracle_value(n, step, memo={}):
        best = (1.0 if ts.ex[n] else -1.0)
        best = max(best, -c_h)
        if step < 6:
            for a, (c, w) in ts.children[n].items():
                if a == ESC:
                    continue
                v = sum(wi * (-lam * ts.tok[ci] + oracle_value(ci, step + 1)) for ci, wi in zip(c, w))
                best = max(best, v)
        return best
    res["oracle_upper_bound_return"] = float(np.mean([oracle_value(ts.root[q], 0) for q in range(len(ts.qids))]))
    json.dump(res, open(os.path.join(out, "baselines.json"), "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cv", "final", "base"])
    ap.add_argument("--trees", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--algo", default="ppo")
    ap.add_argument("--cells", type=int, default=12)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--a5", action="store_true", help="E6-tier: allow escalate")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    a4 = not a.a5
    if a.mode == "base":
        r = baselines(TreeSet.load(a.trees), a.out)
        print(json.dumps({k: (v if not isinstance(v, dict) else {x: round(y, 3) for x, y in v.items() if x not in ("actions", "per_question")}) for k, v in r.items()}, indent=0))
        return
    cells = CELLS[: a.cells]
    if a.mode == "cv":
        jobs = [(a.trees, c, l, f, s, a.algo, a.updates, a4, a.out) for (c, l) in cells for f in range(4) for s in range(a.seeds)]
        with ProcessPoolExecutor(max_workers=a.procs) as ex:
            for r in ex.map(_job, jobs):
                print(json.dumps({k: r[k] for k in ("c_h", "lam", "fold", "seed")}), "val ex_all %.3f calls %.2f defer %.3f" % (r["val"]["ex_all"], r["val"]["calls"], r["val"]["deferral"]), flush=True)
    else:
        jobs = [(a.trees, c, l, s, a.algo, a.updates, a4, a.out) for (c, l) in cells for s in range(a.seeds)]
        with ProcessPoolExecutor(max_workers=a.procs) as ex:
            for r in ex.map(_final, jobs):
                print(r["path"], "train ex_all %.3f" % r["train_eval"]["ex_all"], flush=True)


if __name__ == "__main__":
    main()
