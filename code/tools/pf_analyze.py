#!/usr/bin/env python3
"""Analysis of task PF (SAP Amendment 9, docs/SAP.md section 22): feedback-calibrated Critic `patent_fb` against the reference arms on BIRD dev.
scope = exploratory-posthoc. Intervals only (database-clustered percentile bootstrap, B = 10000, seed 20261101, ONE shared resample for all arms,
`analysis_pipeline.stats.ClusterBoot`); no p-values; nothing here is a `\\res{}` macro or enters a Holm family.

  python tools/pf_analyze.py --ref zs=RUN full_fixed=RUN full_gated=RUN route_ca=RUN --arm pf_n0=RUN pf_n250_q100=RUN pf_n1000_q100=RUN pf_n1000_q085=RUN \
         --out ../results/pf [--B 10000] [--spent 123.4]
A RUN is a directory with preds.jsonl (and cands.jsonl, usage not needed). Arms that were not run (budget drop order, SAP 22.8) are simply omitted:
every contrast that needs them is reported as "not run". Python >= 3.6, numpy."""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)

from analysis_pipeline import runs as R  # noqa: E402
from analysis_pipeline import stats as S  # noqa: E402

SCOPE = "exploratory-posthoc"
MARGIN = 1.5                       # pp, SAP 22.3
THETA_ACC = 8.5                    # the P0 revision trigger: Score < 8.5 = flagged
PRICES_IN, PRICES_OUT, CACHE_RATIO = 2.0, 8.0, 0.2       # qwen3.7-plus-2026-05-26, exp/prices.yaml (checked against the file by price_check)
ARM_ORDER = ["pf_n0", "pf_n250_q100", "pf_n1000_q100", "pf_n1000_q085"]
REF_ORDER = ["zs", "full_fixed", "full_gated", "route_ca"]
V3_LIKE = ("full_fixed", "pf_n0", "pf_n250_q100", "pf_n1000_q100", "pf_n1000_q085")     # arms whose root Score is the weighted Critic score
THROMBOSIS = "thrombosis_prediction"


# ------------------------------------------------------------------------------------------------------ reading rules (SAP 22.6)
def reading(d, lo, hi, m=MARGIN):
    """d, lo, hi in percentage points."""
    if lo > 0:
        return "improves (exploratory)" if d >= m else "small improvement of unclear practical relevance (exploratory)"
    if hi < 0:
        return "harms (exploratory)" if d <= -m else "small decline of unclear practical relevance (exploratory)"
    if lo > -m and hi < m:
        return "no difference larger than 1.5 pp"
    return "inconclusive; the interval contains both 0 and effects of at least 1.5 pp"


def pf3_verdict(read_a, ratio_lo, ratio_hi):
    better = read_a.startswith("improves") or read_a.startswith("small improvement")
    if better and ratio_hi <= 1.0:
        return "dominates (accuracy up, calls not higher)"
    if better:
        return "buys accuracy with calls"
    if ratio_lo > 1.0:
        return "no accuracy gain at higher cost; the gated loop stays preferred"
    return "no accuracy gain; calls ratio interval contains 1"


def curve_verdict(pf1_read, pts):
    """SAP 22.6: the expectation is supported only if PF1 improves AND the points are non-decreasing in n."""
    mono = all(pts[i] <= pts[i + 1] for i in range(len(pts) - 1))
    improves = pf1_read.startswith("improves") or pf1_read.startswith("small improvement")
    if improves and mono:
        return "supported (exploratory): improvement at n = 1000 and non-decreasing curve"
    if improves:
        return "improves with n = 1000, curve not monotone"
    return "not shown: this implementation of the Critic update does not show the expected gain on these data"


# ------------------------------------------------------------------------------------------------------ loading
def jl(path):
    return R.jl(path)


def load_run(path):
    rows = {}
    for r in jl(os.path.join(path, "preds.jsonl")):
        rows[r["qid"]] = r
    cands = {}
    cp = os.path.join(path, "cands.jsonl")
    if os.path.exists(cp):
        for c in jl(cp):
            cands.setdefault(c["qid"], []).append(c)
        for q in cands:
            cands[q].sort(key=lambda c: c.get("cand_idx", 0))
    return {"path": path, "rows": rows, "cands": cands}


def price_row(r):
    pt, ct = r.get("prompt_tokens", 0) or 0, r.get("completion_tokens", 0) or 0
    cached = min(r.get("cached_tokens", 0) or 0, pt)
    return ((pt - cached) * PRICES_IN + cached * PRICES_IN * CACHE_RATIO + ct * PRICES_OUT) / 1e6


def ci_pp(boot):
    lo, hi = S.pctl_ci(boot)
    return 100.0 * lo, 100.0 * hi


# ------------------------------------------------------------------------------------------------------ per-arm vectors
def arm_vectors(run, qids):
    rows = [run["rows"][q] for q in qids]
    ex = np.array([R.ex_value(r) for r in rows], float)
    v = {"ex": ex, "calls": np.array([r.get("calls", 0) or 0 for r in rows], float),
         "pt": np.array([r.get("prompt_tokens", 0) or 0 for r in rows], float), "ct": np.array([r.get("completion_tokens", 0) or 0 for r in rows], float),
         "cost": np.array([price_row(r) for r in rows], float), "iters": np.array([r.get("iterations", 0) or 0 for r in rows], float),
         "embed": np.array([r.get("embed_calls", 0) or 0 for r in rows], float), "rerank": np.array([r.get("rerank_calls", 0) or 0 for r in rows], float)}
    rex, rscore, rsql = [], [], []
    for q in qids:
        cs = run["cands"].get(q) or []
        root = cs[0] if cs else None
        rex.append(np.nan if root is None or root.get("ex") is None else float(root["ex"]))
        rscore.append(np.nan if root is None else float(root.get("score", np.nan)))
        rsql.append(root.get("sql") if root else None)
    v["root_ex"], v["root_score"], v["root_sql"] = np.array(rex), np.array(rscore), rsql
    v["pf_parse_fail"] = np.array([((r.get("pf") or {}).get("parse_fail") or 0) for r in rows], float)
    v["has_rejected_case"] = np.array([float(any(c.get("verdict") == 0 for c in ((r.get("pf") or {}).get("cases") or []))) if r.get("pf") else np.nan for r in rows])
    v["n_cases"] = np.array([len((r.get("pf") or {}).get("cases") or []) if r.get("pf") else np.nan for r in rows])
    return v


def helped_hurt(v):
    """items with at least one revision: helped = root wrong -> final right; hurt = root right -> final wrong (SAP 22.4)."""
    rev = (v["iters"] > 0) & ~np.isnan(v["root_ex"])
    helped = (rev & (v["root_ex"] == 0) & (v["ex"] == 1)).astype(float)
    hurt = (rev & (v["root_ex"] == 1) & (v["ex"] == 0)).astype(float)
    return rev.astype(float), helped, hurt


def flag_stats(v):
    flag = (v["root_score"] < THETA_ACC).astype(float)
    ok = ~np.isnan(v["root_ex"]) & ~np.isnan(v["root_score"])
    if not ok.any():
        return {}
    wrong, right = ok & (v["root_ex"] == 0), ok & (v["root_ex"] == 1)
    return {"flag_rate": float(flag[ok].mean()) if ok.any() else None,
            "detection": float(flag[wrong].mean()) if wrong.any() else None, "false_flag": float(flag[right].mean()) if right.any() else None,
            "n_wrong_roots": int(wrong.sum()), "n_right_roots": int(right.sum()), "flag": flag}


# ------------------------------------------------------------------------------------------------------ contrasts
def contrast(boot, name, a, b, ex, dbs, mask=None):
    """EX(a) - EX(b), paired, in pp with the clustered interval, McNemar counts, leave-one-db-out range and sign share."""
    d = ex[a] - ex[b]
    est = 100.0 * float(d.mean())
    lo, hi = ci_pp(boot.means(d))
    b_ = int(((ex[a] == 1) & (ex[b] == 0)).sum())
    c_ = int(((ex[a] == 0) & (ex[b] == 1)).sum())
    loo = [100.0 * v for _, v in boot.loo(lambda idx: d[idx].mean())]
    per_db = {}
    for db in sorted(set(boot.dbs.tolist())):
        m = boot.dbs == db
        per_db[db] = float(d[m].mean())
    same = float(np.mean([np.sign(x) == np.sign(est) for x in per_db.values() if x != 0] or [np.nan])) if est != 0 else None
    out = {"name": name, "a": a, "b": b, "d_pp": est, "lo_pp": lo, "hi_pp": hi, "a_only": b_, "b_only": c_, "loo_min_pp": min(loo), "loo_max_pp": max(loo),
           "share_dbs_same_sign": same, "n": int(len(d)), "reading": reading(est, lo, hi)}
    if mask is not None:
        dm = d[mask]
        out["d_pp_no_thrombosis"] = 100.0 * float(dm.mean())
        bm = S.ClusterBoot(boot.dbs[mask], B=boot.B, seed=S.SEED_BOOT)
        out["lo_pp_no_thrombosis"], out["hi_pp_no_thrombosis"] = ci_pp(bm.means(dm))
    return out


def ratio_ci(boot, num, den):
    r = float(num.sum() / max(den.sum(), 1e-12))
    lo, hi = S.pctl_ci(boot.ratio(num, den))
    return r, lo, hi


# ------------------------------------------------------------------------------------------------------ main analysis
def analyze(runs, B=10000, spent=None):
    labels = [l for l in REF_ORDER + ARM_ORDER if l in runs]
    common = None
    for l in labels:
        s = set(q for q, r in runs[l]["rows"].items() if R.ex_value(r) is not None)
        common = s if common is None else (common & s)
    qids = sorted(common)
    dbs = np.array([runs[labels[0]]["rows"][q].get("db_id") or q.split(":")[1] for q in qids])
    n_all = {l: len(runs[l]["rows"]) for l in labels}
    boot = S.ClusterBoot(dbs, B=B, seed=S.SEED_BOOT)
    V = {l: arm_vectors(runs[l], qids) for l in labels}
    ex = {l: V[l]["ex"] for l in labels}
    nothr = dbs != THROMBOSIS
    res = {"_meta": {"scope": SCOPE, "label": "exploratory-posthoc, pre-specified", "sap": "docs/SAP.md section 22 (Amendment 9)", "B": B, "seed": S.SEED_BOOT,
                     "n_items": len(qids), "n_dbs": len(set(dbs.tolist())), "arms_present": [l for l in labels if l in ARM_ORDER],
                     "arms_missing": [l for l in ARM_ORDER if l not in runs], "refs_present": [l for l in labels if l in REF_ORDER],
                     "rows_per_run": n_all, "dropped_for_null_ex": {l: n_all[l] - len(qids) for l in labels},
                     "feedback": "SIMULATED ORACLE feedback (EX against the BIRD-train reference answer, optional noise q): an upper bound for an ideal user",
                     "spent_cny_reported": spent}}
    arms = []
    for l in labels:
        v = V[l]
        e = v["ex"]
        lo, hi = ci_pp(boot.means(e))
        rev, hp, ht = helped_hurt(v)
        fs = flag_stats(v) if l in V3_LIKE else {}
        row = {"label": l, "n": int(len(e)), "ex_pp": 100 * float(e.mean()), "ex_lo": lo, "ex_hi": hi,
               "ex_no_thrombosis_pp": 100 * float(e[nothr].mean()),
               "calls_per_q": float(v["calls"].mean()), "prompt_tokens_per_q": float(v["pt"].mean()), "completion_tokens_per_q": float(v["ct"].mean()),
               "cost_cny_per_q": float(v["cost"].mean()), "cost_cny_total": float(v["cost"].sum()), "cost_cny_per_correct": float(v["cost"].sum() / max(e.sum(), 1)),
               "embed_calls_per_q": float(v["embed"].mean()), "rerank_calls_per_q": float(v["rerank"].mean()),
               "revision_share_pp": 100 * float(rev.mean()), "items_with_revision": int(rev.sum()), "helped": int(hp.sum()), "hurt": int(ht.sum()),
               "net_helped_minus_hurt_pp": 100 * float((hp - ht).mean())}
        nlo, nhi = ci_pp(boot.means(hp - ht))
        row["net_lo"], row["net_hi"] = nlo, nhi
        if fs:
            row.update({"critic_flag_rate_pp": 100 * fs["flag_rate"], "critic_detection_pp": 100 * fs["detection"] if fs["detection"] is not None else None,
                        "critic_false_flag_pp": 100 * fs["false_flag"] if fs["false_flag"] is not None else None,
                        "n_wrong_roots": fs["n_wrong_roots"], "n_right_roots": fs["n_right_roots"]})
        if l in ARM_ORDER:
            row["parse_fail_items"] = int((v["pf_parse_fail"] > 0).sum())
            hr = v["has_rejected_case"]
            if not np.all(np.isnan(hr)):
                row["share_items_with_rejected_case_pp"] = 100 * float(np.nanmean(hr))
                fl = (v["root_score"] < THETA_ACC).astype(float)
                row["flag_rate_with_rejected_case_pp"] = 100 * float(fl[hr == 1].mean()) if (hr == 1).any() else None
                row["flag_rate_without_rejected_case_pp"] = 100 * float(fl[hr == 0].mean()) if (hr == 0).any() else None
        if l != "full_fixed" and "full_fixed" in V and l in ARM_ORDER:
            a, b = V[l]["root_sql"], V["full_fixed"]["root_sql"]
            row["root_sql_identical_to_full_fixed_pp"] = 100 * float(np.mean([x == y for x, y in zip(a, b)]))
        arms.append(row)
    res["arms"] = arms
    # --- contrasts
    C = []

    def add(name, a, b, need=()):
        if all(x in ex for x in (a, b) + tuple(need)):
            C.append(contrast(boot, name, a, b, ex, dbs, nothr))
        else:
            C.append({"name": name, "a": a, "b": b, "status": "not run"})

    add("PF1", "pf_n1000_q100", "pf_n0")
    add("PF2", "pf_n1000_q085", "full_fixed")
    add("PF2_q100", "pf_n1000_q100", "full_fixed")
    add("PF2_n0", "pf_n0", "full_fixed")
    add("PF3a", "pf_n1000_q085", "full_gated")
    add("PF3a_q100", "pf_n1000_q100", "full_gated")
    add("PF4_250_vs_0", "pf_n250_q100", "pf_n0")
    add("PF4_1000_vs_250", "pf_n1000_q100", "pf_n250_q100")
    add("PF4_noise_cost", "pf_n1000_q100", "pf_n1000_q085")
    for l in [x for x in ARM_ORDER if x in ex]:
        add("ref_" + l + "_vs_zs", l, "zs")
    if "full_fixed" in ex and "full_gated" in ex:
        add("ref_fixed_vs_gated", "full_fixed", "full_gated")
    res["contrasts"] = C
    cd = {c["name"]: c for c in C}
    # --- PF3 (b): calls ratio and verdict
    pf3 = {}
    for nm, arm in (("PF3", "pf_n1000_q085"), ("PF3_q100", "pf_n1000_q100")):
        if arm in V and "full_gated" in V:
            r, lo, hi = ratio_ci(boot, V[arm]["calls"], V["full_gated"]["calls"])
            ra = cd["PF3a" if nm == "PF3" else "PF3a_q100"]["reading"]
            pf3[nm] = {"calls_ratio": r, "lo": lo, "hi": hi, "accuracy_reading": ra, "verdict": pf3_verdict(ra, lo, hi)}
            r2, lo2, hi2 = ratio_ci(boot, V[arm]["cost"], V["full_gated"]["cost"])
            pf3[nm].update({"cost_ratio": r2, "cost_lo": lo2, "cost_hi": hi2})
    res["PF3_cost"] = pf3
    ratio_fixed = {}
    for arm in [x for x in ARM_ORDER if x in V and "full_fixed" in V]:
        r, lo, hi = ratio_ci(boot, V[arm]["calls"], V["full_fixed"]["calls"])
        ratio_fixed[arm] = {"calls_ratio_vs_fixed": r, "lo": lo, "hi": hi}
    res["calls_vs_fixed"] = ratio_fixed
    # --- learning curve
    pts = [(0, "pf_n0"), (250, "pf_n250_q100"), (1000, "pf_n1000_q100")]
    curve = []
    for n_, l in pts:
        if l in ex:
            lo, hi = ci_pp(boot.means(ex[l]))
            curve.append({"n": n_, "label": l, "ex_pp": 100 * float(ex[l].mean()), "lo": lo, "hi": hi})
    res["curve"] = curve
    if "PF1" in cd and cd["PF1"].get("reading"):
        present = [p["ex_pp"] for p in curve]
        res["PF4"] = {"points_present": [p["n"] for p in curve], "monotone_nondecreasing": all(present[i] <= present[i + 1] for i in range(len(present) - 1)),
                      "complete": len(curve) == 3, "expectation": curve_verdict(cd["PF1"]["reading"], present)}
    # --- cases (SAP 22.6: first three discordant per direction by qid; no gold, no question text; dev gold is read by the evaluator only)
    cases = {}
    for nm in ("PF1", "PF2"):
        c = cd.get(nm)
        if not c or "a" not in c or "status" in c:
            continue
        a, b = c["a"], c["b"]
        up = [q for i, q in enumerate(qids) if ex[a][i] == 1 and ex[b][i] == 0][:3]
        dn = [q for i, q in enumerate(qids) if ex[a][i] == 0 and ex[b][i] == 1][:3]

        def desc(q):
            i = qids.index(q)
            o = {"qid": q, "db_id": dbs[i]}
            for l in (a, b):
                r = runs[l]["rows"][q]
                o[l] = {"final_sql": r.get("final_sql"), "ex": r.get("ex"), "actions": r.get("actions"), "calls": r.get("calls"),
                        "candidates": [{"action": cc.get("action"), "score": cc.get("score"), "semantic": cc.get("semantic"), "ex": cc.get("ex")}
                                       for cc in runs[l]["cands"].get(q, [])]}
                if r.get("pf"):
                    o[l]["memory_verdicts"] = [cc["verdict"] for cc in r["pf"].get("cases", [])]
            return o
        cases[nm] = {"a": a, "b": b, "a_right_b_wrong": [desc(q) for q in up], "a_wrong_b_right": [desc(q) for q in dn]}
    res["cases"] = cases
    return res


# ------------------------------------------------------------------------------------------------------ outputs
def flat(res):
    out = {"_meta": res["_meta"]}
    for a in res["arms"]:
        for k, v in a.items():
            if k != "label":
                out["arm.%s.%s" % (a["label"], k)] = v
    for c in res["contrasts"]:
        for k, v in c.items():
            if k != "name":
                out["%s.%s" % (c["name"], k)] = v
    for nm, d in res["PF3_cost"].items():
        for k, v in d.items():
            out["%s.%s" % (nm, k)] = v
    for nm, d in res["calls_vs_fixed"].items():
        for k, v in d.items():
            out["calls_vs_fixed.%s.%s" % (nm, k)] = v
    for p in res["curve"]:
        out["curve.n%d.ex_pp" % p["n"]] = p["ex_pp"]
        out["curve.n%d.lo" % p["n"]] = p["lo"]
        out["curve.n%d.hi" % p["n"]] = p["hi"]
    for k, v in (res.get("PF4") or {}).items():
        out["PF4." + k] = v
    return out


def write_outputs(res, out):
    os.makedirs(out, exist_ok=True)
    json.dump(flat(res), open(os.path.join(out, "pf_summary.json"), "w"), indent=1, sort_keys=True, default=float)
    json.dump(res["cases"], open(os.path.join(out, "pf_cases.json"), "w"), indent=1, default=float)
    for fname, rows in (("pf_arms.csv", res["arms"]), ("pf_contrasts.csv", res["contrasts"]), ("pf_curve.csv", res["curve"])):
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(out, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)


def price_check(prices_path):
    """the constants above equal exp/prices.yaml (qwen3.7-plus-2026-05-26)."""
    txt = open(prices_path).read()
    return ("qwen3.7-plus-2026-05-26:   {in: 2.0,  out: 8.0}" in txt) and ("cache_hit_ratio: 0.2" in txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", nargs="*", default=[], help="label=run_dir for zs full_fixed full_gated route_ca")
    ap.add_argument("--arm", nargs="*", default=[], help="label=run_dir for the pf arms")
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--spent", type=float, default=None)
    a = ap.parse_args()
    runs = {}
    for kv in list(a.ref) + list(a.arm):
        k, p = kv.split("=", 1)
        runs[k] = load_run(p)
    here_prices = os.path.join(os.path.dirname(CODE), "exp", "prices.yaml")
    assert price_check(here_prices), "price constants of pf_analyze.py differ from exp/prices.yaml"
    res = analyze(runs, a.B, a.spent)
    write_outputs(res, a.out)
    print(json.dumps({"arms": [(x["label"], round(x["ex_pp"], 2)) for x in res["arms"]],
                      "contrasts": [(c["name"], round(c.get("d_pp", float("nan")), 2), c.get("reading", c.get("status"))) for c in res["contrasts"]]}, indent=1))


if __name__ == "__main__":
    main()
