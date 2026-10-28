#!/usr/bin/env python3
"""SAP 13.6 pre-lock decisions, mechanically (X6; numpy only; worker machine venv or any python with numpy).  Tuning data only (HR / HT), never dev.

  selection_rules.py --hr-fixed R/runs/X5/1b-hr-quote --hr-gated R/runs/X5/1b-hr-gated \\
                     --ht-fixed R/runs/X5/1b-ht-quote --ht-gated R/runs/X5/1b-ht-gated \\
                     [--hr-cons R/runs/X5/2-hr-cons [--ht-cons R/runs/X5/2-ht-cons]] [--k 2] [--boot 10000] --out R/exp/selection_decisions.json

Each run argument is a run directory holding preds.jsonl, or a preds.jsonl file.  Rows are paired on qid; rows with ex null in either run are dropped (counted).

RULE 3 (primary system `full`).  full := full_gated iff ALL of
  (i)   HR: lower 95 % bound of EX(gated) - EX(fixed) > -1.5 pp          (paired, database-clustered percentile bootstrap, B = 10 000)
  (ii)  HR: upper 95 % bound of R = sum calls(gated) / sum calls(fixed) < 0.80   (same bootstrap; ratio of sums over resampled databases)
  (iii) HT-300: point estimate of EX(gated) - EX(fixed) >= -2.0 pp  and  point R < 0.80
otherwise full := full_fixed.  "95 % bound" = the 2.5 / 97.5 percentile of the bootstrap (two-sided 95 % interval); the one-sided 5 / 95 percentile
bounds are written next to them for information only.

RULE 4 (routing score route_ca, P3).  Needs the consistency runs (full_gated_cons.yaml, V3 of the final answer + >= k cons_bits).  route_ca = logistic
regression on [v3, consistency (first k cons_bits, default k = 2), nonempty].  Adopt (P3 kept) iff on HR, with leave-one-database-out predictions,
  (i)  AUROC(route_ca) - AUROC(v3) >= 0.03 with the lower 95 % bound of the paired db-clustered bootstrap (B = 10 000) > 0, AND
  (ii) accepted-EX at 60 % coverage (quantile rule, ties random: fit_route_score.risk_coverage) of route_ca >= that of v3 + 2 pp (point estimate).
otherwise P3 is dropped (m = 4).  If --ht-cons is given, the model fitted on all of HR is evaluated on HT and reported (information, not a decision input).

Output (exp/selection_decisions.json, read by tools/lock_gate.py --require, the freeze checklist (not released) step 2 and analysis_pipeline.runs.load_decisions):
  {"rule3": "gated"|"fixed", "full": "full_gated"|"full_fixed", "rule4_p3": "keep"|"drop"|null, "route_ca_adopted": bool|null, "m_family": 5|4,
   "b_ref_calls": HR mean calls/q of the primary, "b_ref_calls_gated", "b_ref_calls_fixed", "p2_calls_ceiling": 1.15 * b_ref_calls,
   "decided": date, "rule3_detail": {...}, "rule4_detail": {...}, "inputs": [{path, sha256, n_rows}], "tool": ..., "boot": B, "seed": ...}
`m_family` counts P3 only (5, or 4 when P3 is dropped); the P5 inclusion rule (n_R >= 40) is applied on dev at the lock.  Without --hr-cons rule 4 is
not evaluated (rule4_p3 null, m_family 5 by default) - do not freeze in that state."""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fit_route_score as FR  # noqa: E402

SEED = 20261101
MARGIN_PP = -1.5            # rule 3 (i)
RATIO_MAX = 0.80            # rule 3 (ii), (iii)
HT_MIN_PP = -2.0            # rule 3 (iii)
AUROC_GAIN = 0.03           # rule 4 (i)
ACC60_GAIN = 0.02           # rule 4 (ii)
FEATURES = ["v3", "consistency", "nonempty"]


# ------------------------------------------------------------------------------------------------ io
def preds_path(p):
    return os.path.join(p, "preds.jsonl") if os.path.isdir(p) else p


def load_rows(p):
    rows = {}
    with open(preds_path(p)) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["qid"]] = r
    return rows


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pair(rows_fixed, rows_gated):
    """-> dict(db, ex_f, ex_g, c_f, c_g) arrays over the qids present in both runs with ex not null in both; plus drop counts."""
    qids = sorted(q for q in rows_fixed if q in rows_gated)
    keep = [q for q in qids if rows_fixed[q].get("ex") is not None and rows_gated[q].get("ex") is not None]
    drop = {"only_in_fixed": len(set(rows_fixed) - set(rows_gated)), "only_in_gated": len(set(rows_gated) - set(rows_fixed)),
            "ex_null": len(qids) - len(keep)}
    f = lambda rows, k, d=0.0: np.array([float(rows[q].get(k) if rows[q].get(k) is not None else d) for q in keep], float)  # noqa: E731
    dbs = np.array([rows_fixed[q].get("db_id") or q.split(":")[1] for q in keep])
    return {"qids": keep, "db": dbs, "ex_f": f(rows_fixed, "ex"), "ex_g": f(rows_gated, "ex"),
            "c_f": f(rows_fixed, "calls"), "c_g": f(rows_gated, "calls")}, drop


# ------------------------------------------------------------------------------------------------ rule 3
def _db_sums(dbs, cols):
    u = np.unique(dbs)
    return u, np.array([[c[dbs == d].sum() for c in cols] + [float((dbs == d).sum())] for d in u], float)


def boot_ex_and_ratio(pr, B, seed=SEED):
    """Database-clustered bootstrap of (mean ex_g - ex_f, sum c_g / sum c_f).  Resampling databases with replacement keeps every item of a drawn
    database, so mean = sum(d) / sum(n) over the drawn databases (identical to resampling the concatenated items, cf. paired_compare.cluster_ci)."""
    u, S = _db_sums(pr["db"], [pr["ex_g"] - pr["ex_f"], pr["c_g"], pr["c_f"]])
    m = len(u)
    out = {"n_databases": int(m)}
    if m < 2:
        return out
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, m, size=(B, m))
    T = S[idx].sum(1)                                  # [B, 4]
    d = 100.0 * T[:, 0] / T[:, 3]
    r = T[:, 1] / np.maximum(T[:, 2], 1e-9)
    out["diff_pp_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    out["diff_pp_lo_onesided95"] = float(np.percentile(d, 5))
    out["ratio_ci95"] = [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
    out["ratio_hi_onesided95"] = float(np.percentile(r, 95))
    return out


def point(pr):
    n = max(len(pr["qids"]), 1)
    return {"n_pairs": len(pr["qids"]), "ex_fixed": float(pr["ex_f"].mean()) if len(pr["qids"]) else None,
            "ex_gated": float(pr["ex_g"].mean()) if len(pr["qids"]) else None,
            "diff_pp": 100.0 * float((pr["ex_g"] - pr["ex_f"]).mean()) if len(pr["qids"]) else None,
            "calls_per_q_fixed": float(pr["c_f"].sum() / n), "calls_per_q_gated": float(pr["c_g"].sum() / n),
            "ratio": float(pr["c_g"].sum() / max(pr["c_f"].sum(), 1e-9)),
            "gated_right_fixed_wrong": int(((pr["ex_g"] == 1) & (pr["ex_f"] == 0)).sum()),
            "fixed_right_gated_wrong": int(((pr["ex_f"] == 1) & (pr["ex_g"] == 0)).sum())}


def rule3(hr_fixed, hr_gated, ht_fixed, ht_gated, B=10000):
    hr, hr_drop = pair(hr_fixed, hr_gated)
    ht, ht_drop = pair(ht_fixed, ht_gated)
    if not hr["qids"] or not ht["qids"]:
        raise SystemExit("rule 3: no paired questions (HR %d, HT %d)" % (len(hr["qids"]), len(ht["qids"])))
    p_hr, p_ht = point(hr), point(ht)
    b_hr = boot_ex_and_ratio(hr, B)
    if "diff_pp_ci95" not in b_hr:
        raise SystemExit("rule 3: HR has fewer than 2 databases, no clustered bootstrap")
    c1 = b_hr["diff_pp_ci95"][0] > MARGIN_PP
    c2 = b_hr["ratio_ci95"][1] < RATIO_MAX
    c3 = (p_ht["diff_pp"] >= HT_MIN_PP) and (p_ht["ratio"] < RATIO_MAX)
    return {"decision": "gated" if (c1 and c2 and c3) else "fixed",
            "conditions": {"i_hr_dEX_lower95_gt_-1.5pp": bool(c1), "ii_hr_ratio_upper95_lt_0.80": bool(c2), "iii_ht_point_dEX_ge_-2.0pp_and_ratio_lt_0.80": bool(c3)},
            "hr": dict(p_hr, **b_hr, dropped=hr_drop), "ht": dict(p_ht, dropped=ht_drop)}


# ------------------------------------------------------------------------------------------------ rule 4
def _acc60(score, y):
    return FR.risk_coverage(score, y, covs=(0.6,))[1]["0.6"]


def rule4(hr_cons_path, ht_cons_path=None, k=2, B=10000):
    X, y, dbs, qids, dropped = FR.load_rows([preds_path(hr_cons_path)], FEATURES, k)
    if len(y) < 30 or len(np.unique(y)) < 2 or len(np.unique(dbs)) < 2:
        raise SystemExit("rule 4: too few usable HR rows (%d rows, %d dbs, dropped %s); the consistency run needs v3 of the final answer and cons_bits"
                         % (len(y), len(np.unique(dbs)), dropped))
    p_lodo = FR.lodo_predict(X, y, dbs)
    v3 = X[:, 0]
    a_ca, a_v3 = FR.auroc(p_lodo, y), FR.auroc(v3, y)
    est, lo, hi = FR.cluster_boot(dbs, lambda idx: FR.auroc(p_lodo[idx], y[idx]) - FR.auroc(v3[idx], y[idx]), B=B, seed=SEED)
    acc_ca, acc_v3 = _acc60(p_lodo, y), _acc60(v3, y)
    c1 = bool(est >= AUROC_GAIN and lo > 0)
    c2 = bool(acc_ca - acc_v3 >= ACC60_GAIN - 1e-12)
    out = {"decision": "keep" if (c1 and c2) else "drop", "k": k, "features": FEATURES,
           "conditions": {"i_dAUROC_ge_0.03_and_lower95_gt_0": c1, "ii_acc60_ge_v3_plus_2pp": c2},
           "hr": {"n": int(len(y)), "n_databases": int(len(np.unique(dbs))), "dropped": dropped, "auroc_route_ca_lodo": a_ca, "auroc_v3": a_v3,
                  "dAUROC": est, "dAUROC_ci95": [lo, hi], "acc60_route_ca_lodo": acc_ca, "acc60_v3": acc_v3, "dAcc60_pp": 100.0 * (acc_ca - acc_v3),
                  "aurc_route_ca_lodo": FR.risk_coverage(p_lodo, y)[0], "aurc_v3": FR.risk_coverage(v3, y)[0]}}
    if ht_cons_path:
        Xh, yh, dh, _, dr = FR.load_rows([preds_path(ht_cons_path)], FEATURES, k)
        if len(yh) >= 10 and len(np.unique(yh)) == 2 and len(np.unique(dh)) >= 2:
            ph = FR.predict(FR.fit_logistic(X, y), Xh)
            e, l, h = FR.cluster_boot(dh, lambda idx: FR.auroc(ph[idx], yh[idx]) - FR.auroc(Xh[idx, 0], yh[idx]), B=B, seed=SEED)
            out["ht_heldout_info"] = {"n": int(len(yh)), "dropped": dr, "dAUROC": e, "dAUROC_ci95": [l, h],
                                      "acc60_route_ca": _acc60(ph, yh), "acc60_v3": _acc60(Xh[:, 0], yh)}
    return out


# ------------------------------------------------------------------------------------------------ main
def decide(hr_fixed, hr_gated, ht_fixed, ht_gated, hr_cons=None, ht_cons=None, k=2, B=10000, inputs=None):
    r3 = rule3(load_rows(hr_fixed), load_rows(hr_gated), load_rows(ht_fixed), load_rows(ht_gated), B)
    r4 = rule4(hr_cons, ht_cons, k, B) if hr_cons else None
    gated = r3["decision"] == "gated"
    b_gated, b_fixed = r3["hr"]["calls_per_q_gated"], r3["hr"]["calls_per_q_fixed"]
    b_ref = b_gated if gated else b_fixed
    adopted = None if r4 is None else (r4["decision"] == "keep")
    return {"rule3": r3["decision"], "full": "full_gated" if gated else "full_fixed",
            "rule4_p3": None if r4 is None else r4["decision"], "route_ca_adopted": adopted,
            "m_family": 4 if adopted is False else 5,
            "m_family_note": "counts P3 only; P5 is in the family iff n_R >= 40 on dev (SAP 13.3), applied at the lock",
            "b_ref_calls": round(b_ref, 4), "b_ref_calls_gated": round(b_gated, 4), "b_ref_calls_fixed": round(b_fixed, 4),
            "p2_calls_ceiling": round(1.15 * b_ref, 4), "baseline_window": [round(0.85 * b_ref, 4), round(1.15 * b_ref, 4)],
            "decided": time.strftime("%Y-%m-%d"), "rule3_detail": r3, "rule4_detail": r4, "inputs": inputs or [],
            "boot": B, "seed": SEED, "tool": "code/tools/selection_rules.py (X6)",
            "fallback_note": None if gated else "full_fixed: full_gated is still run on dev and P1 is still tested (SAP 13.6 rule 3, 'Otherwise')"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for n in ("hr-fixed", "hr-gated", "ht-fixed", "ht-gated"):
        ap.add_argument("--" + n, required=True)
    ap.add_argument("--hr-cons", default=None)
    ap.add_argument("--ht-cons", default=None)
    ap.add_argument("--k", type=int, default=2, help="number of consistency samples used by route_ca (SAP: k = 2)")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    paths = {"hr_fixed": a.hr_fixed, "hr_gated": a.hr_gated, "ht_fixed": a.ht_fixed, "ht_gated": a.ht_gated, "hr_cons": a.hr_cons, "ht_cons": a.ht_cons}
    inputs = []
    for n, p in paths.items():
        if p:
            pp = preds_path(p)
            if not os.path.exists(pp):
                sys.exit("missing %s: %s" % (n, pp))
            inputs.append({"name": n, "path": pp, "sha256": sha256_file(pp), "n_rows": sum(1 for l in open(pp) if l.strip())})
    res = decide(a.hr_fixed, a.hr_gated, a.ht_fixed, a.ht_gated, a.hr_cons, a.ht_cons, a.k, a.boot, inputs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    r3 = res["rule3_detail"]
    print("[selection_rules] rule 3 -> %s  %s" % (res["rule3"], r3["conditions"]))
    print("  HR dEX %+.2f pp ci95 [%.2f, %.2f]  R %.3f ci95 [%.3f, %.3f] | HT dEX %+.2f pp R %.3f | B_ref %.3f calls/q"
          % (r3["hr"]["diff_pp"], r3["hr"]["diff_pp_ci95"][0], r3["hr"]["diff_pp_ci95"][1], r3["hr"]["ratio"], r3["hr"]["ratio_ci95"][0],
             r3["hr"]["ratio_ci95"][1], r3["ht"]["diff_pp"], r3["ht"]["ratio"], res["b_ref_calls"]))
    r4 = res["rule4_detail"]
    if r4:
        h = r4["hr"]
        print("[selection_rules] rule 4 -> %s  %s\n  HR dAUROC %+.3f ci95 [%.3f, %.3f]  acc@60%% route_ca %.3f vs v3 %.3f"
              % (res["rule4_p3"], r4["conditions"], h["dAUROC"], h["dAUROC_ci95"][0], h["dAUROC_ci95"][1], h["acc60_route_ca_lodo"], h["acc60_v3"]))
    else:
        print("[selection_rules] rule 4 not evaluated (no --hr-cons): rule4_p3 = null; do not freeze in this state")
    print("  -> %s" % a.out)
    return res


if __name__ == "__main__":
    main()
