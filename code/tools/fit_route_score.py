#!/usr/bin/env python3
"""Fit the logistic routing score of M7 (X5, HEADROOM row 2) offline from preds.jsonl files. numpy only (no sklearn / scipy).

  fit_route_score.py --preds RUN/preds.jsonl [RUN2/preds.jsonl ...] --out exp/frozen/route_coef_hr.json
                     [--features v3,consistency,nonempty] [--eval-preds HT_RUN/preds.jsonl] [--k M] [--C 1.0] [--boot 1000]

Rows come from runs of the ours-systems with route.consistency.k > 0 (e.g. exp/configs/full_gated_cons.yaml) on tuning data (HR; HT is the
held-out check, `--eval-preds`). Features (all read from preds.jsonl):
  v3           critic.score of the final answer (0-10; needs a run with the V3 critic on the final answer)
  exec         critic.syntax (execution sub-score, 0-10)
  consistency  `consistency` (fraction of the k extra samples whose result equals the final result); `--k M` recomputes it from the first
               M entries of `cons_bits` (sensitivity to the number of samples)
  nonempty     final_status == ok_nonempty (falls back to exec_ok when the field is missing)
Label: ex (0/1; rows with ex null are dropped). Model: L2-regularised logistic regression on standardised features (intercept unpenalised).
Evaluation = leave-one-database-out (LODO) predictions pooled over all rows: AUROC, AURC (area under the risk-coverage curve, ties random),
EX of the accepted fraction at fixed coverages, each against the V3-only and consistency-only baselines, with db-clustered bootstrap CIs
(databases resampled with replacement) and the paired AUROC difference vs V3. Success rule of HEADROOM section 5 row 2: AUROC gain >= 0.03
with the paired CI above 0 and a lower AURC.
Output JSON = {"features", "coef" (raw feature scale), "intercept", "link": "logistic", ...}, read by nl2sql.systems.ours.route.RouteScorer
(route.coef_path). The routing score is 10 * sigmoid(intercept + coef . features)."""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

FEATURES = ("v3", "exec", "consistency", "nonempty")
COVERAGES = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)


# ----------------------------------------------------------------------------------------------- data
def load_rows(paths, features, k=None):
    """-> (X [n, f] float, y [n] int, dbs [n] str, qids list, dropped dict)."""
    X, y, dbs, qids = [], [], [], []
    dropped = {"no_label": 0, "missing_feature": 0}
    for p in paths:
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("ex") is None:
                    dropped["no_label"] += 1
                    continue
                feat = row_features(r, k)
                if any(feat.get(n) is None for n in features):
                    dropped["missing_feature"] += 1
                    continue
                X.append([float(feat[n]) for n in features])
                y.append(int(r["ex"]))
                dbs.append(r.get("db_id") or str(r["qid"]).split(":")[1])
                qids.append(r["qid"])
    return np.array(X, float).reshape(len(X), len(features)), np.array(y, int), np.array(dbs), qids, dropped


def row_features(r, k=None):
    c = r.get("critic") or {}
    cons = r.get("consistency")
    bits = r.get("cons_bits")
    if k and bits:
        cons = float(np.mean(bits[:k])) if len(bits[:k]) else None
    st = r.get("final_status")
    nonempty = (1.0 if st == "ok_nonempty" else 0.0) if st is not None else (1.0 if r.get("exec_ok") else None)
    v3 = c.get("score") if c.get("semantic") is not None else None
    return {"v3": v3, "exec": c.get("syntax"), "consistency": cons, "nonempty": nonempty}


# ----------------------------------------------------------------------------------------------- model
def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def fit_logistic(X, y, C=1.0, iters=100, tol=1e-9):
    """L2 logistic regression (penalty 1/(2C) |w|^2 on standardised features; intercept free) by Newton / IRLS.
    -> dict(mean, sd, w_std, b_std, coef (raw scale), intercept (raw scale))."""
    X = np.asarray(X, float)
    mean, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = (X - mean) / sd
    n, f = Z.shape
    A = np.hstack([np.ones((n, 1)), Z])
    w = np.zeros(f + 1)
    lam = np.eye(f + 1) / float(C)
    lam[0, 0] = 0.0
    for _ in range(iters):
        p = _sigmoid(A @ w)
        g = A.T @ (p - y) + lam @ w
        W = np.clip(p * (1 - p), 1e-6, None)
        H = (A * W[:, None]).T @ A + lam + 1e-9 * np.eye(f + 1)
        step = np.linalg.solve(H, g)
        w = w - step
        if np.max(np.abs(step)) < tol:
            break
    w_std, b_std = w[1:], w[0]
    coef = w_std / sd
    intercept = b_std - float(np.sum(w_std * mean / sd))
    return {"mean": mean, "sd": sd, "w_std": w_std, "b_std": float(b_std), "coef": coef, "intercept": float(intercept)}


def predict(model, X):
    return _sigmoid(np.asarray(X, float) @ model["coef"] + model["intercept"])


# ----------------------------------------------------------------------------------------------- metrics
def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, int)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):                      # average ranks for ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def risk_coverage(score, y, covs=COVERAGES, reps=20, seed=1):
    """-> (AURC, {coverage: EX of the accepted fraction}); ties in score are broken randomly (mean over `reps` shuffles)."""
    rng = np.random.RandomState(seed)
    n = len(y)
    acc = {c: [] for c in covs}
    aurc = []
    for _ in range(reps):
        o = np.lexsort((rng.rand(n), -np.asarray(score, float)))
        cum = np.cumsum(np.asarray(y, float)[o]) / np.arange(1, n + 1)
        aurc.append(float((1.0 - cum).mean()))
        for c in covs:
            acc[c].append(float(cum[max(1, int(round(c * n))) - 1]))
    return float(np.mean(aurc)), {str(c): float(np.mean(v)) for c, v in acc.items()}


def cluster_boot(dbs, fn, B=1000, seed=20261101):
    """db-clustered bootstrap: fn(idx) -> float on a resample of databases (all rows of a drawn db kept). -> (estimate, lo, hi)."""
    rng = np.random.RandomState(seed)
    u = np.unique(dbs)
    mem = {d: np.where(dbs == d)[0] for d in u}
    est = fn(np.arange(len(dbs)))
    vals = []
    for _ in range(B):
        idx = np.concatenate([mem[d] for d in rng.choice(u, len(u), replace=True)])
        try:
            v = fn(idx)
        except Exception:
            v = float("nan")
        if v == v:
            vals.append(v)
    if not vals:
        return est, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def lodo_predict(X, y, dbs, C=1.0):
    p = np.zeros(len(y))
    for d in np.unique(dbs):
        te = dbs == d
        tr = ~te
        if tr.sum() < 5 or len(np.unique(y[tr])) < 2:
            p[te] = float(y[tr].mean()) if tr.any() else 0.5
            continue
        p[te] = predict(fit_logistic(X[tr], y[tr], C), X[te])
    return p


def evaluate(scores, y, dbs, base_name, B=1000):
    """scores: {name: array}; paired AUROC / AURC difference of each against scores[base_name], db-clustered CI."""
    out = {}
    for nm, s in scores.items():
        a = cluster_boot(dbs, lambda idx, s=s: auroc(s[idx], y[idx]), B=B)
        aurc, rc = risk_coverage(s, y)
        out[nm] = {"auroc": a[0], "auroc_ci": [a[1], a[2]], "aurc": aurc, "ex_at_coverage": rc}
    if base_name in scores:
        b = scores[base_name]
        for nm, s in scores.items():
            if nm == base_name:
                continue
            d = cluster_boot(dbs, lambda idx, s=s, b=b: auroc(s[idx], y[idx]) - auroc(b[idx], y[idx]), B=B)
            out[nm]["auroc_gain_vs_" + base_name] = {"gain": d[0], "ci": [d[1], d[2]]}
            out[nm]["aurc_gain_vs_" + base_name] = out[base_name]["aurc"] - out[nm]["aurc"]       # positive = lower risk than the baseline
    return out


def success_rule(ev, name, base):
    g = (ev.get(name) or {}).get("auroc_gain_vs_" + base)
    if not g:
        return None
    return bool(g["gain"] >= 0.03 and g["ci"][0] > 0 and ev[name]["aurc"] < ev[base]["aurc"])


# ----------------------------------------------------------------------------------------------- main
def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preds", nargs="+", required=True, help="training preds.jsonl file(s) (HR)")
    ap.add_argument("--eval-preds", nargs="*", default=[], help="held-out preds.jsonl file(s) (HT): evaluated with the model fitted on --preds")
    ap.add_argument("--features", default="v3,consistency,nonempty")
    ap.add_argument("--k", type=int, default=None, help="use only the first k entries of cons_bits for the consistency feature")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    feats = [f.strip() for f in a.features.split(",") if f.strip()]
    bad = [f for f in feats if f not in FEATURES]
    if bad:
        sys.exit("unknown features %s (known %s)" % (bad, FEATURES))
    X, y, dbs, qids, dropped = load_rows(a.preds, feats, a.k)
    if len(y) < 30 or len(np.unique(y)) < 2:
        sys.exit("too few usable rows (%d, dropped %s)" % (len(y), dropped))
    model = fit_logistic(X, y, a.C)
    p_lodo = lodo_predict(X, y, dbs, a.C)
    scores = {"lr_lodo(%s)" % "+".join(feats): p_lodo}
    base = None
    for nm in ("v3", "consistency", "exec"):
        if nm in FEATURES:
            Xb, yb, db_b, _, _ = load_rows(a.preds, [nm], a.k)
            if len(yb) == len(y):                       # same rows (all other features present)
                scores[nm] = Xb[:, 0]
    base = "v3" if "v3" in scores else None
    ev = evaluate(scores, y, dbs, base, B=a.boot) if base else evaluate(scores, y, dbs, "none", B=a.boot)
    out = {"features": feats, "coef": [float(c) for c in model["coef"]], "intercept": model["intercept"], "link": "logistic", "scale": 10.0,
           "standardised": {"mean": [float(m) for m in model["mean"]], "sd": [float(s) for s in model["sd"]],
                            "w_std": [float(w) for w in model["w_std"]], "b_std": model["b_std"]},
           "fit": {"C": a.C, "k": a.k, "n": int(len(y)), "n_pos": int(y.sum()), "dbs": sorted(set(dbs.tolist())), "dropped": dropped},
           "lodo": ev, "success_rule_met": success_rule(ev, list(scores)[0], base) if base else None,
           "inputs": [{"path": p, "sha256": _sha(p)} for p in a.preds], "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    if a.eval_preds:
        Xh, yh, dh, _, dr = load_rows(a.eval_preds, feats, a.k)
        if len(yh) >= 10 and len(np.unique(yh)) == 2:
            sc = {"lr_fitted_on_train": predict(model, Xh)}
            for nm in ("v3", "consistency", "exec"):
                if nm in FEATURES:
                    Xb, yb, _, _, _ = load_rows(a.eval_preds, [nm], a.k)
                    if len(yb) == len(yh):
                        sc[nm] = Xb[:, 0]
            out["heldout"] = {"n": int(len(yh)), "dropped": dr, "inputs": [{"path": p, "sha256": _sha(p)} for p in a.eval_preds],
                              "metrics": evaluate(sc, yh, dh, "v3" if "v3" in sc else "none", B=a.boot)}
            out["heldout"]["success_rule_met"] = success_rule(out["heldout"]["metrics"], "lr_fitted_on_train", "v3") if "v3" in sc else None
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    print("[fit_route_score] n=%d pos=%d dbs=%d features=%s" % (len(y), int(y.sum()), len(set(dbs.tolist())), feats))
    for nm, m in ev.items():
        print("  %-28s AUROC %.3f [%.3f, %.3f]  AURC %.3f  EX@60%% %.3f" % (nm, m["auroc"], m["auroc_ci"][0], m["auroc_ci"][1], m["aurc"],
                                                                   m["ex_at_coverage"]["0.6"]))
    print("  coef %s intercept %.4f -> %s" % ([round(c, 4) for c in out["coef"]], out["intercept"], a.out))
    return out


if __name__ == "__main__":
    main()
