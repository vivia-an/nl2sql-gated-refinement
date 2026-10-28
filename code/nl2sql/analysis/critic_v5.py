"""E5 learned critic V5: data loading, grouped CV, calibration metrics, export.

Inputs (candidate-level records with gold EX; HR candidates for the real training, any smoke file for tests)
  --cands   one or more cands.jsonl (rows written by run.py: sql/action/tier/syntax/semantic/efficiency/score/status/n_rows/n_cols/
            k/top_sim/mean_ws/... + qid/db_id/cand_idx/ex)
  --trees   one or more trees.jsonl from rl/env_builder.py (nodes with 27-dim `state`, sub-scores, ex); HR candidates of the spec
  --v4      optional jsonl re-scoring the same candidates with the cross-vendor critic: {qid, sql, syntax, semantic, efficiency, score}
  --enrich-split  e.g. bird_train_HR: run EXPLAIN / execute on the db to add plan features and result-set agreement (CPU only)
Models (fixed hyper-parameters, no tuning on dev):  lr (impute+standardise+logistic regression), hgb (sklearn
HistGradientBoostingClassifier, the spec's V5), lgbm (LightGBM, if installed).
Protocol: group-by-db K-fold (spec: 4 folds x 2 dbs, dbs sorted by n_q desc in snake order; trees carry their fold id, which is
used when present), out-of-fold probabilities, cross-fitted isotonic calibration (calibrator for fold f is fitted on the
OOF predictions of the other folds), metrics AUROC / AUPRC / Brier / ECE (15 equal-mass bins) raw and calibrated,
db-clustered bootstrap CIs and paired Delta-AUROC against the V3 score, top-1 selection accuracy per question.
The exported drop-in is trained on ALL records with the LIVE feature set (per-candidate features only); its calibrator is
the isotonic fitted on all OOF predictions.

python -m nl2sql.analysis.critic_v5 --cands a/cands.jsonl --out DIR [--trees t.jsonl] [--models lr,hgb,lgbm] [--features live,set]
       [--export hgb] [--folds 4] [--seed 0] [--bootstrap 2000] [--enrich-split bird_train_HR]"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

from ..systems.ours.critic_features import NAN, feature_names, qf_fields, record_features, vectorize
from ..systems.ours.learned_critic import save_bundle


# ------------------------------------------------------------------ loading
def read_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def recs_from_cands(rows, src=0):
    """src = index of the input file: candidates of different runs for the same question are separate candidate sets (gid)."""
    out = []
    seen = {}
    for r in rows:
        if r.get("ex") is None or not r.get("qid"):
            continue
        seen[(r["qid"], r.get("cand_idx"))] = r            # a resumed/appended file may repeat a row: keep the last
    for r in seen.values():
        rec = dict(r)
        rec["gid"] = "%s|%s" % (src, r["qid"])
        rec.update(qf_fields(r.get("qf")))
        rec["ctx_full_schema"] = r.get("full_schema")
        rec["y"] = int(r["ex"])
        out.append(rec)
    _derive_prev(out)
    return out


def _derive_prev(recs):
    """older cands.jsonl rows lack prev_score/changed: derive them from the previous candidate of the same question
    (full_fixed candidates form a revision chain)."""
    by = defaultdict(list)
    for r in recs:
        by[r["gid"]].append(r)
    for q, rs in by.items():
        rs.sort(key=lambda r: r.get("cand_idx", 0))
        for i, r in enumerate(rs):
            if "prev_score" not in r or "changed" not in r:
                if i == 0:
                    r.setdefault("prev_score", None)
                    r.setdefault("changed", None)
                else:
                    p = rs[i - 1]
                    r["prev_score"] = p.get("score")
                    r["changed"] = (p.get("status"), p.get("n_rows"), p.get("n_cols")) != (r.get("status"), r.get("n_rows"), r.get("n_cols"))


def recs_from_trees(trees, src="t"):
    out = []
    for t in trees:
        nodes = t.get("nodes") or []
        by_id = {n["id"]: n for n in nodes}
        for n in nodes:
            if n.get("ex") is None:
                continue
            s = n["state"]
            par = by_id.get(n.get("parent"))
            rec = {"qid": t["qid"], "gid": "%s|%s" % (src, t["qid"]), "db_id": t["db_id"], "cand_idx": n["id"], "fold": t.get("fold"), "sql": n.get("sql"),
                   "syntax": n.get("syntax"), "semantic": n.get("semantic"), "efficiency": n.get("efficiency"), "score": n.get("score"),
                   "status": n.get("status"), "n_rows": math.expm1(s[10] * math.log1p(1000)), "n_cols": s[11] * 10,
                   "top_sim": s[16], "mean_sim": s[17], "mean_ws": s[18] * 10, "q_tokens": s[19], "n_objects": s[20], "has_time": s[21],
                   "qf_n_tables": s[22], "qf_n_columns": s[23], "has_evidence": s[24], "k": n.get("k"), "tier": n.get("tier"),
                   "action": n.get("action"), "prev_score": par.get("score") if par else None,
                   "changed": (s[12] > 0.5) if par else None, "y": int(n["ex"]), "weight": n.get("weight", 1)}
            out.append(rec)
    return out


def merge_v4(recs, v4_rows):
    idx = {(r["qid"], re.sub(r"\s+", " ", (r.get("sql") or "").strip())): r for r in v4_rows}
    n = 0
    for r in recs:
        v = idx.get((r["qid"], re.sub(r"\s+", " ", (r.get("sql") or "").strip())))
        if v:
            r["v4_syntax"], r["v4_semantic"], r["v4_efficiency"], r["v4_score"] = v.get("syntax"), v.get("semantic"), v.get("efficiency"), v.get("score")
            n += 1
    return n


def enrich_db(recs, split):
    """Add EXPLAIN plan features and result-set agreement with the other candidates of the question (needs the sqlite files)."""
    from .. import data, db
    items = {i.qid: i for i in data.load_split(split)}
    from ..systems.ours.critic_features import plan_features
    keys = {}
    for r in recs:
        it = items.get(r["qid"])
        if it is None or not r.get("sql"):
            continue
        r.update(plan_features(db.explain(it.db_path, r["sql"])))
        res = db.execute(it.db_path, r["sql"], timeout_s=30, max_rows=10000)
        keys[(r["gid"], r["cand_idx"])] = db.rows_key(res["rows"]) if res["ok"] else None
    by = defaultdict(list)
    for r in recs:
        by[r["gid"]].append(r)
    for q, rs in by.items():
        for r in rs:
            k = keys.get((q, r["cand_idx"]))
            others = [keys.get((q, o["cand_idx"])) for o in rs if o is not r]
            r["agree_res"] = (sum(1 for o in others if o is not None and o == k) / len(others)) if (others and k is not None) else (0.0 if others else NAN)


def add_set_features(recs):
    def norm(s):
        return re.sub(r"\s+", " ", (s or "").strip().lower()).rstrip(";")
    by = defaultdict(list)
    for r in recs:
        by[r["gid"]].append(r)
    for q, rs in by.items():
        n = len(rs)
        mx = max((r.get("score") or 0.0) for r in rs)
        order = sorted(range(n), key=lambda i: -(rs[i].get("score") or 0.0))
        rank = {i: j for j, i in enumerate(order)}
        for i, r in enumerate(rs):
            others = [o for o in rs if o is not r]
            r["agree_sql"] = (sum(1 for o in others if norm(o.get("sql")) == norm(r.get("sql"))) / len(others)) if others else NAN
            sig = lambda x: (x.get("status"), round(x.get("n_rows") or 0), x.get("n_cols"))
            r["agree_sig"] = (sum(1 for o in others if sig(o) == sig(r)) / len(others)) if others else NAN
            r.setdefault("agree_res", NAN)
            r["score_minus_max"] = (r.get("score") or 0.0) - mx
            r["score_rank_frac"] = rank[i] / (n - 1) if n > 1 else 0.0
            r["n_cands"] = float(n)


def build_matrix(recs, names, with_v4):
    setk = ("agree_sql", "agree_sig", "agree_res", "score_minus_max", "score_rank_frac", "n_cands")
    X = np.asarray([vectorize({**record_features(r, with_v4=with_v4), **{k: r.get(k, NAN) for k in setk}}, names) for r in recs],
                   dtype=np.float64)
    X[~np.isfinite(X)] = np.nan
    y = np.asarray([r["y"] for r in recs], dtype=int)
    return X, y


# ------------------------------------------------------------------ folds
def snake_folds(db_counts, k=4):
    """SPEC section 1.2.8: dbs sorted by n_q desc, assigned in snake order f0..f(k-1), f(k-1)..f0."""
    order = sorted(db_counts, key=lambda d: (-db_counts[d], d))
    seq = list(range(k)) + list(range(k - 1, -1, -1))
    return {d: seq[i % len(seq)] for i, d in enumerate(order)}


def assign_folds(recs, k=4):
    if all(r.get("fold") is not None for r in recs):
        return np.asarray([int(r["fold"]) for r in recs])
    nq = defaultdict(set)
    for r in recs:
        nq[r["db_id"]].add(r["qid"])          # n_q = distinct questions per db
    fm = snake_folds({d: len(v) for d, v in nq.items()}, k)
    return np.asarray([fm[r["db_id"]] for r in recs])


# ------------------------------------------------------------------ models
def make_model(name, seed=0):
    if name == "lr":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_depth=3, min_samples_leaf=20, l2_regularization=1.0,
                                              early_stopping=False, random_state=seed)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=20, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, verbose=-1, n_jobs=2)
    raise ValueError(name)


class _Const:
    """fallback when a training fold has a single class"""
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.tile([1 - self.p, self.p], (len(X), 1))


def fit_model(name, X, y, seed=0):
    if len(set(y.tolist())) < 2:
        return _Const(float(y.mean()) if len(y) else 0.5)
    m = make_model(name, seed)
    m.fit(X, y)
    return m


def oof_predict(name, X, y, folds, seed=0):
    p = np.full(len(y), np.nan)
    for f in sorted(set(folds.tolist())):
        te = folds == f
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        p[te] = fit_model(name, X[tr], y[tr], seed).predict_proba(X[te])[:, 1]
    return p


def crossfit_isotonic(p, y, folds):
    from sklearn.isotonic import IsotonicRegression
    out = np.full(len(y), np.nan)
    for f in sorted(set(folds.tolist())):
        te = folds == f
        tr = ~te & np.isfinite(p)
        if tr.sum() < 10 or len(set(y[tr].tolist())) < 2:
            out[te] = p[te]
            continue
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p[tr], y[tr])
        out[te] = ir.predict(p[te])
    return out


# ------------------------------------------------------------------ metrics
def ece_equal_mass(p, y, bins=15):
    order = np.argsort(p, kind="stable")
    chunks = np.array_split(order, min(bins, len(p)))
    e = 0.0
    rel = []
    for c in chunks:
        if len(c) == 0:
            continue
        conf, acc = float(p[c].mean()), float(y[c].mean())
        e += len(c) / len(p) * abs(conf - acc)
        rel.append({"n": int(len(c)), "mean_p": round(conf, 4), "frac_pos": round(acc, 4)})
    return float(e), rel


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else float("nan")


def metrics(y, p, pcal=None):
    from sklearn.metrics import average_precision_score, brier_score_loss
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    out = {"n": int(len(y)), "pos_rate": float(y.mean()) if len(y) else None, "auroc": _auc(y, p)}
    if len(set(y.tolist())) == 2:
        out["auprc"] = float(average_precision_score(y, p))
    out["brier"] = float(brier_score_loss(y, np.clip(p, 0, 1)))
    out["ece_raw"], out["reliability_raw"] = ece_equal_mass(np.clip(p, 0, 1), y)
    if pcal is not None:
        pc = pcal[ok]
        out["brier_cal"] = float(brier_score_loss(y, np.clip(pc, 0, 1)))
        out["ece_cal"], out["reliability_cal"] = ece_equal_mass(np.clip(pc, 0, 1), y)
    return out


def cluster_boot(y, p1, p2, groups, B=2000, seed=20261101):
    """db-clustered bootstrap: CI of AUROC(p1) and of AUROC(p1) - AUROC(p2) (paired; p2 may be None)."""
    rnd = np.random.RandomState(seed)
    gs = sorted(set(groups.tolist()))
    idx = {g: np.where(groups == g)[0] for g in gs}
    a1, d = [], []
    for _ in range(B):
        pick = rnd.choice(len(gs), len(gs), replace=True)
        ii = np.concatenate([idx[gs[j]] for j in pick])
        yy = y[ii]
        if len(set(yy.tolist())) < 2:
            continue
        x1 = _auc(yy, p1[ii])
        a1.append(x1)
        if p2 is not None:
            d.append(x1 - _auc(yy, p2[ii]))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 20 else [None, None]
    out = {"auroc_ci": q(a1), "boot_valid": len(a1)}
    if p2 is not None:
        out["delta_auroc_vs_v3"] = _auc(y, p1) - _auc(y, p2)
        out["delta_auroc_ci"] = q(d)
    return out


def pick_accuracy(recs, p):
    """per question: EX of the highest-scored candidate (ties -> earliest), averaged over questions."""
    by = defaultdict(list)
    for i, r in enumerate(recs):
        by[r["gid"]].append(i)
    sel, oracle, root = [], [], []
    for q, ii in by.items():
        ii = sorted(ii, key=lambda i: recs[i].get("cand_idx", 0))
        best = max(ii, key=lambda i: (p[i] if np.isfinite(p[i]) else -1, -ii.index(i)))
        sel.append(recs[best]["y"])
        oracle.append(max(recs[i]["y"] for i in ii))
        root.append(recs[ii[0]]["y"])
    return {"n_questions": len(sel), "top1": float(np.mean(sel)), "oracle": float(np.mean(oracle)), "first_candidate": float(np.mean(root))}


# ------------------------------------------------------------------ main
def run(recs, out_dir, models=("lr", "hgb"), feature_sets=("live",), k=4, seed=0, boot=2000, export=None, export_path=None):
    os.makedirs(out_dir, exist_ok=True)
    with_v4 = any(r.get("v4_score") is not None for r in recs)
    add_set_features(recs)
    groups = np.asarray([r["db_id"] for r in recs])
    folds = assign_folds(recs, k)
    y = np.asarray([r["y"] for r in recs], dtype=int)
    report = {"n_candidates": len(recs), "n_questions": len({r["qid"] for r in recs}), "n_dbs": len(set(groups.tolist())),
              "pos_rate": float(y.mean()), "with_v4": with_v4, "folds": {int(f): int((folds == f).sum()) for f in sorted(set(folds.tolist()))},
              "fold_dbs": {int(f): sorted(set(groups[folds == f].tolist())) for f in sorted(set(folds.tolist()))}, "models": {}}
    # baselines: raw critic scores as probabilities (score / 10)
    v3 = np.asarray([(r.get("score") or 0.0) / 10.0 for r in recs])
    base = {"v3_score": v3}
    if with_v4:
        base["v4_score"] = np.asarray([(r["v4_score"] if r.get("v4_score") is not None else np.nan) / 10.0 for r in recs])
    for name, p in base.items():
        pc = crossfit_isotonic(p, y, folds)
        m = metrics(y, p, pc)
        m.update(cluster_boot(y, p, None, groups, boot))
        m["selection"] = pick_accuracy(recs, p)
        report["models"][name] = m
    oof_rows = [{"qid": r["qid"], "gid": r["gid"], "db_id": r["db_id"], "cand_idx": r.get("cand_idx"), "y": r["y"], "v3": float(v3[i])} for i, r in enumerate(recs)]
    exported = None
    for fs in feature_sets:
        names = feature_names(with_v4=with_v4, with_set=(fs == "set"))
        X, _ = build_matrix(recs, names, with_v4)
        for mname in models:
            try:
                p = oof_predict(mname, X, y, folds, seed)
            except ImportError as e:
                report["models"]["%s/%s" % (fs, mname)] = {"error": "import: %s" % e}
                continue
            pc = crossfit_isotonic(p, y, folds)
            key = "%s/%s" % (fs, mname)
            m = metrics(y, p, pc)
            m.update(cluster_boot(y, p, v3, groups, boot))
            m["per_fold_auroc"] = {int(f): _auc(y[folds == f], p[folds == f]) for f in sorted(set(folds.tolist()))}
            m["selection"] = pick_accuracy(recs, p)
            m["n_features"] = len(names)
            report["models"][key] = m
            for i in range(len(recs)):
                oof_rows[i][key] = float(p[i])
                oof_rows[i][key + "+iso"] = float(pc[i])
            if fs == "live" and (export == mname) and export_path:
                from sklearn.isotonic import IsotonicRegression
                model = fit_model(mname, X, y, seed)
                cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p[np.isfinite(p)], y[np.isfinite(p)])
                meta = {"n_candidates": len(recs), "n_questions": report["n_questions"], "dbs": sorted(set(groups.tolist())), "seed": seed,
                        "cv_auroc": m["auroc"], "cv_ece_cal": m.get("ece_cal"), "trained_on": "all records, live feature set",
                        "calibrator": "isotonic fitted on out-of-fold predictions"}
                save_bundle(export_path, model, names, mname, cal, with_v4, meta)
                exported = export_path
    with open(os.path.join(out_dir, "oof.jsonl"), "w") as f:
        for r in oof_rows:
            f.write(json.dumps(r) + "\n")
    report["exported"] = exported
    json.dump(report, open(os.path.join(out_dir, "report.json"), "w"), indent=1)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", nargs="*", default=[])
    ap.add_argument("--trees", nargs="*", default=[])
    ap.add_argument("--v4", default=None)
    ap.add_argument("--enrich-split", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="lr,hgb")
    ap.add_argument("--features", default="live,set")
    ap.add_argument("--export", default=None, help="model name to export as drop-in (live features), e.g. hgb")
    ap.add_argument("--export-path", default=None)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args(argv)
    recs = []
    for i, p in enumerate(a.cands):
        recs += recs_from_cands(read_jsonl(p), src="c%d" % i)
    for i, p in enumerate(a.trees):
        recs += recs_from_trees(read_jsonl(p), src="t%d" % i)
    if not recs:
        sys.exit("no records")
    if a.v4:
        print("v4 merged:", merge_v4(recs, read_jsonl(a.v4)))
    if a.enrich_split:
        enrich_db(recs, a.enrich_split)
    rep = run(recs, a.out, tuple(a.models.split(",")), tuple(a.features.split(",")), a.folds, a.seed, a.bootstrap, a.export,
              a.export_path or (os.path.join(a.out, "critic_v5.pkl") if a.export else None))
    brief = {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ("auroc", "auroc_ci", "ece_raw", "ece_cal", "brier", "delta_auroc_vs_v3")}
             for k, v in rep["models"].items()}
    print(json.dumps({"n_candidates": rep["n_candidates"], "n_questions": rep["n_questions"], "n_dbs": rep["n_dbs"], "pos_rate": rep["pos_rate"], "models": brief}, indent=1))


if __name__ == "__main__":
    main()
