"""Q2a: critic quality on roots and on final answers, HR (trees roots + live run) and HT-300 (live run cands). Signals available at
decision time only (no gold). Learned critics: db-grouped CV on HR (leave-one-db-out), and HR->HT transfer. No LLM."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
from trees import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

STAT = ["ok_nonempty", "ok_empty", "err_syntax", "err_identifier", "timeout"]


def feats_tree_root(q):
    n = q.root; s = n["state"]
    return {"syn": n["syntax"], "sem": n["semantic"], "eff": n["efficiency"], "score": n["score"],
            "exec_ok": float(n["status"] in ("ok_nonempty", "ok_empty")), "nonempty": float(n["status"] == "ok_nonempty"),
            "log_rows": s[10], "n_cols": s[11], "sim_top1": s[16], "sim_mean": s[17], "q_tokens": s[19], "n_objects": s[20], "n_tables": s[22], "n_columns": s[23],
            "has_evidence": s[24]}


def feats_cand(c):
    return {"syn": c["syntax"], "sem": c["semantic"], "eff": c["efficiency"], "score": c["score"],
            "exec_ok": float(c["status"] in ("ok_nonempty", "ok_empty")), "nonempty": float(c["status"] == "ok_nonempty"),
            "log_rows": math.log1p(c["n_rows"]) / math.log1p(1000), "n_cols": min(c["n_cols"], 10) / 10.0, "sim_top1": c["top_sim"] or 0.0}


Qs = load_trees()
HRX = [feats_tree_root(q) for q in Qs]
HRy = np.array([q.root["ex"] for q in Qs]); HRd = np.array([q.db for q in Qs])
htc = [c for c in jl(HT_RUN + "/cands.jsonl") if c["cand_idx"] == 0]
HTX = [feats_cand(c) for c in htc]; HTy = np.array([c["ex"] for c in htc]); HTd = np.array([c["db_id"] for c in htc])
res = {"n_HR_roots": len(HRy), "n_HT_roots": len(HTy)}

# live finals
hp = jl(HR_RUN + "/preds.jsonl"); tp = jl(HT_RUN + "/preds.jsonl")
res["root_score_dist"] = {}
for nm, X, y in (("HR", HRX, HRy), ("HT", HTX, HTy)):
    sc = np.array([x["score"] for x in X])
    res["root_score_dist"][nm] = {"EX_root": float(y.mean()), "frac_ge_9.5": float((sc >= 9.5).mean()), "frac_ge_9.5_among_wrong": float((sc[y == 0] >= 9.5).mean()),
                                  "frac_ge_8.5_among_wrong": float((sc[y == 0] >= 8.5).mean()), "frac_lt_8.5": float((sc < 8.5).mean()),
                                  "frac_eq_10": float((sc == 10).mean()), "frac_eq_10_among_wrong": float((sc[y == 0] == 10).mean()),
                                  "EX_when_score_ge_9.5": float(y[sc >= 9.5].mean()), "EX_when_score_lt_9.5": float(y[sc < 9.5].mean()),
                                  "sem_dist_wrong": {str(k): int(v) for k, v in collections.Counter(np.array([x["sem"] for x in X])[y == 0]).items()},
                                  "sem_dist_right": {str(k): int(v) for k, v in collections.Counter(np.array([x["sem"] for x in X])[y == 1]).items()}}


def auc_ci(sig, y, d):
    sig = np.asarray(sig, float)
    return cluster_boot(d, lambda idx: auroc(sig[idx], y[idx]), B=1000)


tab = []
for nm, X, y, d in (("HR", HRX, HRy, HRd), ("HT", HTX, HTy, HTd)):
    for key in ("score", "sem", "syn", "eff", "exec_ok", "nonempty", "log_rows", "sim_top1"):
        sig = np.array([x[key] for x in X], float)
        if key == "log_rows":
            sig = -np.abs(sig - np.median(sig))     # (unused ordering; kept only as a weak prior)
        a = auc_ci(sig, y, d)
        tab.append({"set": nm, "signal": key, "AUROC": a[0], "lo": a[1], "hi": a[2]})
res["auroc_root_simple"] = tab

# learned critic: LODO on HR, HR->HT
fk = ["syn", "sem", "eff", "score", "exec_ok", "nonempty", "log_rows", "n_cols", "sim_top1"]


def mk(model):
    if model == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))
    return HistGradientBoostingClassifier(max_depth=3, max_iter=120, learning_rate=0.05, min_samples_leaf=20, random_state=0)


def M(X, keys):
    return np.array([[x[k] for k in keys] for x in X], float)


learned = {}
for model in ("lr", "hgb"):
    for fs_name, keys in (("v3_subscores", ["syn", "sem", "eff", "score"]), ("live_features", fk), ("live_plus_q", fk + ["q_tokens", "n_objects", "n_tables", "n_columns", "has_evidence"])):
        Xa = M(HRX, keys); p = np.zeros(len(HRy))
        for dbn in np.unique(HRd):
            tr_, te_ = HRd != dbn, HRd == dbn
            if len(set(HRy[tr_])) < 2: continue
            m = mk(model).fit(Xa[tr_], HRy[tr_]); p[te_] = m.predict_proba(Xa[te_])[:, 1]
        a = cluster_boot(HRd, lambda idx: auroc(p[idx], HRy[idx]), B=1000)
        entry = {"HR_LODO_AUROC": a}
        if fs_name != "live_plus_q":
            m = mk(model).fit(Xa, HRy); pt = m.predict_proba(M(HTX, keys))[:, 1]
            entry["HR_to_HT_AUROC"] = cluster_boot(HTd, lambda idx: auroc(pt[idx], HTy[idx]), B=1000)
        learned["%s/%s" % (model, fs_name)] = entry
res["learned_root_critic"] = learned

# calibration of raw V3 score/10 as probability
def ece(p, y, bins=10):
    o = np.argsort(p); b = np.array_split(o, bins); e = 0
    for ix in b:
        if len(ix): e += len(ix) / len(p) * abs(p[ix].mean() - y[ix].mean())
    return e
res["calibration_v3"] = {nm: {"ECE_equal_mass10": float(ece(np.array([x["score"] for x in X]) / 10.0, y)), "Brier": float(np.mean((np.array([x["score"] for x in X]) / 10.0 - y) ** 2)),
                              "mean_score/10": float(np.mean([x["score"] for x in X]) / 10.0), "EX": float(y.mean())} for nm, X, y in (("HR", HRX, HRy), ("HT", HTX, HTy))}

# risk-coverage on roots and on finals with thresholds
def rc_table(sig, y, d, thr):
    out = []
    for t in thr:
        acc = sig >= t
        cov = acc.mean()
        e = cluster_boot(d, lambda idx: (y[idx][acc[idx]].mean() if acc[idx].sum() else float("nan")), B=1000)
        c = cluster_boot(d, lambda idx: acc[idx].mean(), B=1000)
        out.append({"theta": t, "coverage": c, "EX_accepted": e, "n_accepted": int(acc.sum()), "EX_rejected": float(y[~acc].mean()) if (~acc).sum() else None})
    return out


rc = {}
thr = [10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.0, 0.0]
sc_r = np.array([x["score"] for x in HRX]); rc["HR_root"] = rc_table(sc_r, HRy, HRd, thr)
sc_t = np.array([x["score"] for x in HTX]); rc["HT_root"] = rc_table(sc_t, HTy, HTd, thr)
hf_sc = np.array([p["critic"]["score"] if p.get("critic") else 0 for p in hp]); hf_ex = np.array([p["ex"] for p in hp]); hf_d = np.array([p["db_id"] for p in hp])
tf_sc = np.array([p["critic"]["score"] if p.get("critic") else 0 for p in tp]); tf_ex = np.array([p["ex"] for p in tp]); tf_d = np.array([p["db_id"] for p in tp])
rc["HR_final_live"] = rc_table(hf_sc, hf_ex, hf_d, thr)
rc["HT_final_live"] = rc_table(tf_sc, tf_ex, tf_d, thr)
res["risk_coverage"] = rc
res["auroc_final_live"] = {"HR": cluster_boot(hf_d, lambda idx: auroc(hf_sc[idx], hf_ex[idx]), B=1000), "HT": cluster_boot(tf_d, lambda idx: auroc(tf_sc[idx], tf_ex[idx]), B=1000)}


def aurc(sig, y, reps=50, seed=0):
    rng = np.random.RandomState(seed); out = []
    for _ in range(reps):
        o = np.lexsort((rng.rand(len(sig)), -sig))
        risk = 1 - np.cumsum(y[o]) / np.arange(1, len(y) + 1)
        out.append(risk.mean())
    return float(np.mean(out))


res["AURC"] = {"HR_root_V3": aurc(sc_r, HRy), "HT_root_V3": aurc(sc_t, HTy), "HR_final_V3": aurc(hf_sc, hf_ex), "HT_final_V3": aurc(tf_sc, tf_ex),
               "oracle_HR_root": aurc(HRy.astype(float), HRy), "random_HR_root": float(1 - HRy.mean()), "oracle_HT_root": aurc(HTy.astype(float), HTy), "random_HT_root": float(1 - HTy.mean())}
# live selected P0 policy: where do accepted-after-revision stand
dump("q2a_critic.json", res)
print(json.dumps(res, indent=1, default=lambda o: float(o) if hasattr(o, "item") else str(o)))
