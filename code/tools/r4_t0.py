#!/usr/bin/env python3
"""R4T0: temperature-0 rerun of the locked headline run (internal note R4T0, SAP section 27). Post hoc, descriptive. No model call. numpy + scipy; runs on a worker machine (host python 3.6 has no numpy).

    worker machine:  PYTHONPATH=<code> python code/tools/r4_t0.py --runs $R/runs --out results/r4_t0 [--B 10000] [--cost results/r4_t0/cost.json]

Inputs. The locked run s0 (seed parameter 0, T = 0; paths of r3_common.S0) and the two fresh reruns r1 and r2 (runs/R4T0/<arm>-<bk>-r<k>: same configs, same model, seed parameter 0, T = 0,
empty chat-call cache; queue exp/queues/R4T0.jsonl), for the 12 decomposition arms (single call, root-only, gated loop, fixed loop on Plus, Flash and 27B, BIRD dev 1 534). For the comparison with the
replication the three T = 0.7 runs (seeds 101 / 202 / 303) are read from the same tree and analysed with the same code.

(a) per arm: EX of s0 / r1 / r2, SD over the three (ddof 1) with its chi-square interval, share of questions whose final SQL text differs (whitespace-normalised; raw text beside it) and share whose
    correctness differs, for s0-r1, s0-r2, r1-r2 (and the mean of the two pairs with s0), the share of questions whose EX is not the same in all three runs, root-SQL differences of the chain arms,
    share of retrieved example sets that differ, and the identity of the requests (cache keys of usage.jsonl) with the s0 run of the same arm.
(b) every contrast of the replication table (stack, gated loop, fixed loop, gated minus fixed, gated minus single call, fixed minus single call; plus, flash, 27b, pooled over the three; W1 =
    mean of Flash and 27B gated minus single call) for the T = 0 triple: the estimate is the mean over the three runs of the per-run paired difference of the same-run arms (= the mean of the run-averaged
    per-question difference); intervals on that run-averaged vector: database-clustered bootstrap (B = 10 000, seed 20261101), item-level bootstrap, cluster-robust t(G-1), exact sign-flip over the
    size-weighted database means; WIDEST = hull of these four (the definition of revision list 4 A2: the four intervals the Methods define, applied to every setting). The between-run range and the t(2)
    interval of the three run values stay separate columns; the hull with them instead of the two small-cluster intervals is given as `wide_alt` (the R3 definition). Reading rule SUP(0). The s0 reading is
    that of the locked run alone under the same definition; "agrees" = same reading word. P1' = pooled gated minus fixed against the margin -1.5 pp (lower end of the widest interval).
(c) the same contrasts for the T = 0.7 replication under the same definition, side by side with the T = 0 triple, and the difference (T = 0.7 minus T = 0) of every contrast with its intervals."""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r3_common as C  # noqa: E402
from r3_common import S  # noqa: E402
import small_cluster as SC  # noqa: E402

T0 = ("s0", "r1", "r2")
T07 = tuple("s%d" % s for s in C.REP)
PAIRS = (("s0", "r1"), ("s0", "r2"), ("r1", "r2"))
ARMS = C.ARMS                       # zs, rootonly, gated, fixed
ARM_NAME = {"zs": "single call", "rootonly": "root-only", "gated": "gated loop", "fixed": "fixed loop"}
CONTRASTS = [("stack", "rootonly", "zs"), ("loop_g", "gated", "rootonly"), ("loop_f", "fixed", "rootonly"), ("g_minus_f", "gated", "fixed"),
             ("g_minus_zs", "gated", "zs"), ("f_minus_zs", "fixed", "zs")]
SCOPES = ("plus", "flash", "27b", "pooled3")
MARGIN = C.MARGIN
_IDX = {}


def run_rel(b, arm, lab):
    """published path of a run under the runs root."""
    if lab == "s0":
        return C.S0[(b, arm)]
    if lab in ("r1", "r2"):
        return "R4T0/%s-%s-%s" % (arm, b, lab)
    return C.run_path(b, arm, int(lab[1:]))


def norm_sql(s):
    return " ".join((s or "").split())


# ------------------------------------------------------------------------------------------------------------------------------- data
class RunSet(object):
    """rows / EX / SQL of 12 arms x labels on the questions common to all of them."""

    def __init__(self, runs_root, labels):
        self.root, self.labels = runs_root, tuple(labels)
        self.rows, self.note = {}, {}
        for b in C.BK:
            for arm in ARMS:
                for lab in self.labels:
                    rows = C.load_preds(runs_root, run_rel(b, arm, lab))
                    ok, why = C.check_run(rows)
                    self.note[(b, arm, lab)] = why
                    if ok:
                        self.rows[(b, arm, lab)] = rows
        missing = [k for k in self.note if k not in self.rows]
        self.complete = not missing
        common = None
        for rows in self.rows.values():
            q = set(k for k, r in rows.items() if C.ex_of(r) is not None)
            common = q if common is None else common & q
        self.qids = sorted(common or [])
        first = next(iter(self.rows.values())) if self.rows else {}
        self.dbs = np.array([first[q]["db_id"] for q in self.qids])
        self.ex = {k: np.array([C.ex_of(rows[q]) for q in self.qids], float) for k, rows in self.rows.items()}
        self.sql = {k: [norm_sql(rows[q].get("final_sql")) for q in self.qids] for k, rows in self.rows.items()}
        self.raw = {k: [(rows[q].get("final_sql") or "").strip() for q in self.qids] for k, rows in self.rows.items()}
        self.calls = {k: np.array([float(rows[q].get("calls") or 0.0) for q in self.qids], float) for k, rows in self.rows.items()}
        self.exemplars = {k: [tuple(rows[q].get("exemplar_ids") or ()) if rows[q].get("exemplar_ids") is not None else None for q in self.qids] for k, rows in self.rows.items()}

    def have(self, b, arm, labs):
        return all((b, arm, l) in self.ex for l in labs)


# ------------------------------------------------------------------------------------------------------------------------------- intervals
def item_means(v, B):
    v = np.asarray(v, float)
    key = (len(v), B, S.SEED_BOOT)
    if key not in _IDX:
        _IDX[key] = np.random.RandomState(S.SEED_BOOT).randint(0, len(v), size=(B, len(v)))
    return v[_IDX[key]].mean(1)


def four(vec, dbs, boot, B):
    """the four intervals of the Methods on one per-question difference vector (0/1 units) -> pp. cluster, item, cluster-robust t(G-1), exact sign-flip, widest (hull of the four)."""
    v = np.asarray(vec, float)
    est = 100.0 * float(v.mean())
    cl = tuple(100.0 * x for x in S.pctl_ci(boot.means(v)))
    it = tuple(100.0 * x for x in S.pctl_ci(item_means(v, B)))
    d = v * 100.0
    tb = SC.cr_t(d, dbs)
    t_iv = (float(tb["lo"]), float(tb["hi"]))
    _, n_g, m_g = SC.db_level(d, dbs)
    sf = SC.sf_ci(m_g, n_g / n_g.sum())
    sf = (float(sf[0]), float(sf[1])) if sf[0] is not None else (None, None)
    wide = SC.hull(cl, it, t_iv, sf)
    return {"est": est, "cl": cl, "item": it, "t": t_iv, "sf": sf, "wide": wide}


T2 = 4.302652729911275          # t(0.975, 2)


def contrast(dby, dbs, boot, B, labels):
    """dby: {label: vector}. -> estimate (mean of the runs), intervals on the run-averaged vector, per-run values and single-run readings."""
    avg = np.mean([dby[l] for l in labels], axis=0)
    out = four(avg, dbs, boot, B)
    per = {l: 100.0 * float(dby[l].mean()) for l in labels}
    vals = [per[l] for l in labels]
    out["per"] = per
    out["sd_runs"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
    out["range"] = (min(vals), max(vals))
    if len(vals) == 3:
        m, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
        out["t2"] = (m - T2 * sd / math.sqrt(3), m + T2 * sd / math.sqrt(3))
    else:
        out["t2"] = out["range"]
    out["wide_alt"] = SC.hull(out["cl"], out["item"], out["range"], out["t2"])
    out["reading"] = {"cluster": C.reading(*out["cl"]), "item": C.reading(*out["item"]), "wide": C.reading(*out["wide"]), "wide_alt": C.reading(*out["wide_alt"])}
    out["single"] = {}
    for l in labels:
        f = four(dby[l], dbs, boot, B)
        f["reading"] = {"cluster": C.reading(*f["cl"]), "wide": C.reading(*f["wide"])}
        out["single"][l] = f
    return out


def contrasts_of(R, labels, B):
    """-> {(cid, scope): stats} for a RunSet R and a list of labels (all runs present)."""
    boot = S.ClusterBoot(R.dbs, B=B)
    per_b = {}
    res = {}
    for cid, xa, ya in CONTRASTS:
        for b in C.BK:
            if not (R.have(b, xa, labels) and R.have(b, ya, labels)):
                continue
            dby = {l: R.ex[(b, xa, l)] - R.ex[(b, ya, l)] for l in labels}
            per_b[(cid, b)] = dby
            res[(cid, b)] = contrast(dby, R.dbs, boot, B, labels)
        if all((cid, b) in per_b for b in C.BK):
            dby = {l: np.mean([per_b[(cid, b)][l] for b in C.BK], axis=0) for l in labels}
            per_b[(cid, "pooled3")] = dby
            res[(cid, "pooled3")] = contrast(dby, R.dbs, boot, B, labels)
    if all(("g_minus_zs", b) in per_b for b in ("flash", "27b")):
        dby = {l: np.mean([per_b[("g_minus_zs", b)][l] for b in ("flash", "27b")], axis=0) for l in labels}
        per_b[("w1", "weak2")] = dby
        res[("w1", "weak2")] = contrast(dby, R.dbs, boot, B, labels)
    return res, per_b, boot


def agreement(t0, ref_label="s0"):
    """flags: the triple's reading against the locked run's alone, cluster and widest."""
    s = t0["single"][ref_label]
    return {"cluster": t0["reading"]["cluster"] == s["reading"]["cluster"], "wide": t0["reading"]["wide"] == s["reading"]["wide"]}


# ------------------------------------------------------------------------------------------------------------------------------- (a) non-determinism
def sd_interval(sd, n=3):
    from scipy import stats as st
    df = n - 1
    return (sd * math.sqrt(df / st.chi2.ppf(0.975, df)), sd * math.sqrt(df / st.chi2.ppf(0.025, df)))


def root_sqls(runs_root, rel):
    p = os.path.join(runs_root, rel, "cands.jsonl")
    if not os.path.exists(p):
        return None
    return {r["qid"]: norm_sql(r.get("sql")) for r in C.jl(p) if r.get("cand_idx") == 0}


def nondeterminism(R, runs_root):
    out = {"arm": {}}
    for b in C.BK:
        for arm in ARMS:
            if not R.have(b, arm, T0):
                continue
            ex = {l: R.ex[(b, arm, l)] for l in T0}
            sq = {l: R.sql[(b, arm, l)] for l in T0}
            rw = {l: R.raw[(b, arm, l)] for l in T0}
            e = {"ex": {l: 100.0 * float(ex[l].mean()) for l in T0}}
            vals = list(e["ex"].values())
            e["ex_mean"], e["ex_sd"] = float(np.mean(vals)), float(np.std(vals, ddof=1))
            e["ex_sd_ci"] = sd_interval(e["ex_sd"])
            e["ex_range"] = (min(vals), max(vals))
            n = len(R.qids)
            e["sql_diff"], e["sql_diff_raw"], e["ex_diff"], e["ex_flip"] = {}, {}, {}, {}
            for x, y in PAIRS:
                k = "%s_%s" % (x, y)
                e["sql_diff"][k] = 100.0 * sum(1 for a, c in zip(sq[x], sq[y]) if a != c) / n
                e["sql_diff_raw"][k] = 100.0 * sum(1 for a, c in zip(rw[x], rw[y]) if a != c) / n
                e["ex_diff"][k] = 100.0 * float((ex[x] != ex[y]).mean())
                e["ex_flip"][k] = {"right_to_wrong": int(((ex[x] == 1) & (ex[y] == 0)).sum()), "wrong_to_right": int(((ex[x] == 0) & (ex[y] == 1)).sum())}
            for key in ("sql_diff", "sql_diff_raw", "ex_diff"):
                e[key]["vs_s0"] = 0.5 * (e[key]["s0_r1"] + e[key]["s0_r2"])
                e[key]["all_pairs"] = (e[key]["s0_r1"] + e[key]["s0_r2"] + e[key]["r1_r2"]) / 3.0
            allsame = np.all([ex[T0[0]] == ex[l] for l in T0[1:]], axis=0)
            e["ex_unstable"] = 100.0 * float((~allsame).mean())
            e["all_wrong"] = 100.0 * float(np.all([ex[l] == 0 for l in T0], axis=0).mean())
            e["all_right"] = 100.0 * float(np.all([ex[l] == 1 for l in T0], axis=0).mean())
            # retrieved example sets (absent for the single call)
            exm = {l: R.exemplars[(b, arm, l)] for l in T0}
            if all(v is not None for l in T0 for v in exm[l]):
                e["exemplars_differ"] = {"%s_%s" % (x, y): 100.0 * sum(1 for a, c in zip(exm[x], exm[y]) if a != c) / n for x, y in PAIRS}
                e["exemplars_differ"]["vs_s0"] = 0.5 * (e["exemplars_differ"]["s0_r1"] + e["exemplars_differ"]["s0_r2"])
                same = [i for i in range(n) if exm["s0"][i] == exm["r1"][i] == exm["r2"][i]]
                e["same_exemplars_n"] = len(same)
                e["sql_diff_same_exemplars_vs_s0"] = (100.0 * float(np.mean([0.5 * ((sq["s0"][i] != sq["r1"][i]) + (sq["s0"][i] != sq["r2"][i])) for i in same]))) if same else None
                e["sql_diff_same_exemplars_r1_r2"] = (100.0 * float(np.mean([sq["r1"][i] != sq["r2"][i] for i in same]))) if same else None
            # root SQL (cands.jsonl cand 0) for the chain arms
            if arm != "zs":
                rt = {l: root_sqls(runs_root, run_rel(b, arm, l)) for l in T0}
                if all(v is not None for v in rt.values()):
                    e["root_diff"] = {"%s_%s" % (x, y): 100.0 * sum(1 for q in R.qids if rt[x].get(q) != rt[y].get(q)) / n for x, y in PAIRS}
                    e["root_diff"]["vs_s0"] = 0.5 * (e["root_diff"]["s0_r1"] + e["root_diff"]["s0_r2"])
            out["arm"]["%s|%s" % (b, arm)] = e
    A = out["arm"]

    def mean_of(keys, path):
        v = []
        for k in keys:
            x = A[k]
            for p in path:
                x = x.get(p) if isinstance(x, dict) else None
                if x is None:
                    break
            if x is not None:
                v.append(x)
        return float(np.mean(v)) if v else None
    groups = {"all": list(A)}
    for b in C.BK:
        groups["bk." + b] = [k for k in A if k.startswith(b + "|")]
    for arm in ARMS:
        groups["type." + arm] = [k for k in A if k.endswith("|" + arm)]
    out["group"] = {}
    for g, ks in groups.items():
        if not ks:
            continue
        out["group"][g] = {"n_arms": len(ks), "sql_diff_vs_s0": mean_of(ks, ("sql_diff", "vs_s0")), "sql_diff_all_pairs": mean_of(ks, ("sql_diff", "all_pairs")),
                           "sql_diff_r1_r2": mean_of(ks, ("sql_diff", "r1_r2")), "ex_diff_vs_s0": mean_of(ks, ("ex_diff", "vs_s0")), "ex_diff_all_pairs": mean_of(ks, ("ex_diff", "all_pairs")),
                           "ex_diff_r1_r2": mean_of(ks, ("ex_diff", "r1_r2")), "ex_unstable": mean_of(ks, ("ex_unstable",)), "ex_sd": mean_of(ks, ("ex_sd",)),
                           "sql_diff_min": min(A[k]["sql_diff"]["vs_s0"] for k in ks), "sql_diff_max": max(A[k]["sql_diff"]["vs_s0"] for k in ks),
                           "ex_diff_min": min(A[k]["ex_diff"]["vs_s0"] for k in ks), "ex_diff_max": max(A[k]["ex_diff"]["vs_s0"] for k in ks),
                           "ex_sd_min": min(A[k]["ex_sd"] for k in ks), "ex_sd_max": max(A[k]["ex_sd"] for k in ks)}
        if g == "all":
            ch = [k for k in A if "root_diff" in A[k]]
            out["group"][g]["root_diff_vs_s0"] = mean_of(ch, ("root_diff", "vs_s0")) if ch else None
            ex_ = [k for k in A if "exemplars_differ" in A[k]]
            out["group"][g]["exemplars_differ_vs_s0"] = mean_of(ex_, ("exemplars_differ", "vs_s0")) if ex_ else None
    return out


# ------------------------------------------------------------------------------------------------------------------------------- requests
def usage_rows(runs_root, rel):
    p = os.path.join(runs_root, rel, "usage.jsonl")
    if not os.path.exists(p):
        return None
    out = []
    for r in C.jl(p):
        if r.get("kind") == "chat" and r.get("key"):
            out.append((r["key"], r.get("tag") or "", bool(r.get("cached"))))
    return out


def requests_identity(runs_root, R):
    """per arm and rerun: share of chat requests (usage rows with a cache key) whose key also occurs in the s0 run of the same arm; share served from the cache inside the run."""
    out = {}
    for b in C.BK:
        for arm in ARMS:
            ref = usage_rows(runs_root, run_rel(b, arm, "s0"))
            if ref is None:
                continue
            refk = set(k for k, _, _ in ref)
            refk_tag = {}
            for k, t, _ in ref:
                refk_tag.setdefault(t, set()).add(k)
            for lab in ("r1", "r2"):
                rows = usage_rows(runs_root, run_rel(b, arm, lab))
                if rows is None:
                    continue
                keys = [k for k, _, _ in rows]
                e = {"n_rows": len(rows), "n_unique": len(set(keys)), "in_s0_rows": 100.0 * sum(1 for k in keys if k in refk) / max(1, len(keys)),
                     "in_s0_unique": 100.0 * len(set(keys) & refk) / max(1, len(set(keys))), "cached_rows": 100.0 * sum(1 for _, _, c in rows if c) / max(1, len(rows)), "by_tag": {}}
                for tag in sorted(set(t for _, t, _ in rows)):
                    kk = [k for k, t, _ in rows if t == tag]
                    e["by_tag"][tag] = {"n": len(kk), "in_s0": 100.0 * sum(1 for k in kk if k in refk_tag.get(tag, set())) / max(1, len(kk))}
                out["%s|%s|%s" % (b, arm, lab)] = e
    if out:
        vals = [e["in_s0_rows"] for e in out.values()]
        out["_all"] = {"in_s0_rows_mean": float(np.mean(vals)), "in_s0_rows_min": float(min(vals)), "in_s0_rows_max": float(max(vals)),
                       "n_rows": int(sum(e["n_rows"] for k, e in out.items() if k != "_all"))}
    return out


# ------------------------------------------------------------------------------------------------------------------------------- calls ratio (P1')
def calls_ratio(R, labels):
    out = {}
    for b in C.BK:
        if R.have(b, "gated", labels) and R.have(b, "fixed", labels):
            out[b] = {l: float(R.calls[(b, "gated", l)].sum() / R.calls[(b, "fixed", l)].sum()) for l in labels}
    if all(b in out for b in C.BK):
        out["pooled3"] = {l: float(np.mean([out[b][l] for b in C.BK])) for l in labels}
    return out


# ------------------------------------------------------------------------------------------------------------------------------- run
def run(runs_root, B):
    R0 = RunSet(runs_root, T0)
    res = {"notes": {"complete_T0": R0.complete, "n_qids": len(R0.qids), "run_notes": {"%s|%s|%s" % k: v for k, v in sorted(R0.note.items()) if v != "ok"}}}
    if not R0.complete:
        res["notes"]["incomplete"] = True
    res["nd"] = nondeterminism(R0, runs_root) if R0.rows else {}
    res["requests"] = requests_identity(runs_root, R0)
    c0, per0, boot0 = contrasts_of(R0, T0, B) if R0.complete else ({}, {}, None)
    res["T0"] = {"%s|%s" % k: v for k, v in c0.items()}
    for k, v in res["T0"].items():
        v["agrees"] = agreement(v)
    res["calls_T0"] = calls_ratio(R0, T0) if R0.complete else {}
    # T = 0.7 replication, same definition
    R7 = RunSet(runs_root, T07)
    res["notes"]["complete_T07"] = R7.complete
    c7, per7, boot7 = contrasts_of(R7, T07, B) if R7.complete else ({}, {}, None)
    res["T07"] = {"%s|%s" % k: v for k, v in c7.items()}
    res["calls_T07"] = calls_ratio(R7, T07) if R7.complete else {}
    # (c) comparison; the setting difference needs the same questions in both sets
    res["cmp"] = {}
    if c0 and c7 and R0.qids == R7.qids:
        for k in c0:
            a, b_ = c0[k], c7[k]
            dvec = np.mean([per7[k][l] for l in T07], axis=0) - np.mean([per0[k][l] for l in T0], axis=0)
            sd = four(dvec, R0.dbs, boot0, B)
            res["cmp"]["%s|%s" % k] = {"t0_est": a["est"], "t0_reading_wide": a["reading"]["wide"], "t0_reading_cluster": a["reading"]["cluster"],
                                       "t07_est": b_["est"], "t07_reading_wide": b_["reading"]["wide"], "t07_reading_cluster": b_["reading"]["cluster"],
                                       "same_wide": a["reading"]["wide"] == b_["reading"]["wide"], "same_cluster": a["reading"]["cluster"] == b_["reading"]["cluster"],
                                       "setting_diff": sd, "setting_reading_wide": C.reading(*sd["wide"]), "setting_reading_cluster": C.reading(*sd["cl"]),
                                       "t07_est_inside_t0_wide": bool(a["wide"][0] <= b_["est"] <= a["wide"][1]),
                                       "t0_runs_negative": int(sum(1 for l in T0 if a["per"][l] < 0)), "t0_runs_positive": int(sum(1 for l in T0 if a["per"][l] > 0))}
    elif c0 and c7:
        res["notes"]["cmp_skipped"] = "the T = 0 and the T = 0.7 sets have different question sets"
    return res


# ------------------------------------------------------------------------------------------------------------------------------- reduced scope (infrastructure loss): s0 against ONE rerun r1
PARTIAL_OK = (("flash", "fixed"),)          # named in advance: the one incomplete rerun that is analysed, labelled partial (all other incomplete reruns are left out)


def load_loose(runs_root, rel):
    """qid -> row of a possibly incomplete run, rows with api_fail / sys_error dropped."""
    rows = C.load_preds(runs_root, rel)
    if rows is None:
        return None
    return {q: r for q, r in rows.items() if not (r.get("error_class") == "api_fail" or str(r.get("error_class")).startswith("sys_error"))}


def pair_arm(runs_root, b, arm, B, s0rows, r1rows):
    qids = sorted(q for q in s0rows if q in r1rows and C.ex_of(s0rows[q]) is not None and C.ex_of(r1rows[q]) is not None)
    n = len(qids)
    dbs = np.array([s0rows[q]["db_id"] for q in qids])
    e0 = np.array([C.ex_of(s0rows[q]) for q in qids], float)
    e1 = np.array([C.ex_of(r1rows[q]) for q in qids], float)
    boot = S.ClusterBoot(dbs, B=B)
    f = four(e1 - e0, dbs, boot, B)
    bb = int(((e0 == 1) & (e1 == 0)).sum())
    cc = int(((e0 == 0) & (e1 == 1)).sum())
    mc = S.mcnemar_exact(bb, cc)
    sq0, sq1 = [norm_sql(s0rows[q].get("final_sql")) for q in qids], [norm_sql(r1rows[q].get("final_sql")) for q in qids]
    rw0, rw1 = [(s0rows[q].get("final_sql") or "").strip() for q in qids], [(r1rows[q].get("final_sql") or "").strip() for q in qids]
    out = {"n": n, "complete": n >= 1534, "n_dbs": len(set(dbs.tolist())), "ex_s0": 100.0 * float(e0.mean()), "ex_r1": 100.0 * float(e1.mean()), "diff": f,
           "reading_cluster": C.reading(*f["cl"]), "reading_wide": C.reading(*f["wide"]), "b": bb, "c": cc, "mcnemar_p": mc["p_two"],
           "sql_diff": 100.0 * sum(1 for x, y in zip(sq0, sq1) if x != y) / n, "sql_diff_raw": 100.0 * sum(1 for x, y in zip(rw0, rw1) if x != y) / n,
           "ex_changed": 100.0 * (bb + cc) / n, "sd_net": 100.0 * math.sqrt(bb + cc) / n, "sd_run": 100.0 * math.sqrt(bb + cc) / n / math.sqrt(2.0),
           "qids": qids, "ex0": e0, "ex1": e1, "dbs": dbs}
    x0 = [tuple(s0rows[q].get("exemplar_ids")) if s0rows[q].get("exemplar_ids") is not None else None for q in qids]
    x1 = [tuple(r1rows[q].get("exemplar_ids")) if r1rows[q].get("exemplar_ids") is not None else None for q in qids]
    if all(v is not None for v in x0 + x1):
        out["exemplars_differ"] = 100.0 * sum(1 for x, y in zip(x0, x1) if x != y) / n
        same = [i for i in range(n) if x0[i] == x1[i]]
        out["n_same_exemplars"] = len(same)
        out["sql_diff_same_exemplars"] = (100.0 * sum(1 for i in same if sq0[i] != sq1[i]) / len(same)) if same else None
    if arm != "zs":
        a, c_ = root_sqls(runs_root, run_rel(b, arm, "s0")), root_sqls(runs_root, run_rel(b, arm, "r1"))
        if a is not None and c_ is not None:
            out["root_diff"] = 100.0 * sum(1 for q in qids if a.get(q) != c_.get(q)) / n
    return out


def read_archive_resp(runs_root, rel):
    """key -> response text of the call-cache archive of a run (None if absent)."""
    import gzip
    p = os.path.join(runs_root, rel, "llm_cache.jsonl.gz")
    if not os.path.exists(p):
        return None
    out = {}
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["key"]] = r["resp"]
    return out


def response_identity(runs_root, b, arm):
    """Requests with the same cache key in s0 and r1 (byte-identical model, parameters and messages): share whose stored answer differs. By stage tag."""
    a, c = read_archive_resp(runs_root, run_rel(b, arm, "s0")), read_archive_resp(runs_root, run_rel(b, arm, "r1"))
    if a is None or c is None:
        return None
    u0, u1 = usage_rows(runs_root, run_rel(b, arm, "s0")) or [], usage_rows(runs_root, run_rel(b, arm, "r1")) or []
    tag = {}
    for k, t, _ in u0 + u1:
        tag.setdefault(k, t)
    both = sorted(set(a) & set(c))
    out = {"n_common": len(both), "n_diff": sum(1 for k in both if a[k] != c[k]), "by_tag": {}}
    out["diff_pct"] = 100.0 * out["n_diff"] / max(1, len(both))
    for t in sorted(set(tag.get(k, "") for k in both)):
        ks = [k for k in both if tag.get(k, "") == t]
        out["by_tag"][t] = {"n": len(ks), "diff_pct": 100.0 * sum(1 for k in ks if a[k] != c[k]) / max(1, len(ks))}
    return out


# ------------------------------------------------------------------------------------------------------------------------------- round 5: every contrast s0 against r1, noise scale, chain check
CTR_SCOPES = ("plus", "flash", "27b", "pooled3")
_BOOTS = {}


def _boot(dbs, B):
    k = (dbs.tobytes(), B) if hasattr(dbs, "tobytes") else (tuple(dbs), B)
    if k not in _BOOTS:
        _BOOTS[k] = S.ClusterBoot(dbs, B=B)
    return _BOOTS[k]


def contrast_runs(keep, cid, scope, B):
    """One contrast of the paper (CONTRASTS, or W1 = mean of Flash and 27B gated minus single call) in the locked run s0 and in the rerun r1 on the questions common to all arms it uses.
    -> None if an arm is missing. Noise scale (pp), all from the s0-versus-r1 discordance of the arms: NET difference between two runs that are identical in distribution.
      per arm        sd_net = 100 sqrt(b + c) / n          (b, c = discordant counts; Var of the net count, multinomial, ignoring (b - c)^2 / n)
      two-arm        sqrt(sd_net_x^2 + sd_net_y^2)         ASSUMES the two arms' run-to-run changes are independent across arms (approximation; same-chain arms share roots, so they are
                                                            positively correlated and the true SD of the difference is smaller)
      pooled / W1    sqrt(sum over backbones of the two-arm variances) / k       (k = 3 backbones, 2 for W1; again independence across arms and backbones)
    `sd_direct` = 100 sqrt(sum v_i^2) / n with v_i the per-question change (r1 minus s0) of the contrast itself: no independence assumption between arms, only between questions."""
    if scope == "weak2":
        if cid != "w1":
            return None
        bks, xa, ya = ("flash", "27b"), "gated", "zs"
    else:
        if cid == "w1":
            return None
        spec = [c for c in CONTRASTS if c[0] == cid]
        if not spec:
            return None
        bks, xa, ya = (C.BK if scope == "pooled3" else (scope,)), spec[0][1], spec[0][2]
    if not all((b, a) in keep for b in bks for a in (xa, ya)):
        return None
    qs = None
    for b in bks:
        for a in (xa, ya):
            q = set(keep[(b, a)]["qids"])
            qs = q if qs is None else qs & q
    qs = sorted(qs)
    n = len(qs)
    if n == 0:
        return None

    def vec(b, a, which):
        e = keep[(b, a)]
        idx = {q: i for i, q in enumerate(e["qids"])}
        return np.array([e[which][idx[q]] for q in qs], float)
    d0 = np.mean([vec(b, xa, "ex0") - vec(b, ya, "ex0") for b in bks], axis=0)
    d1 = np.mean([vec(b, xa, "ex1") - vec(b, ya, "ex1") for b in bks], axis=0)
    first = keep[(bks[0], xa)]
    dbs = np.array([first["dbs"][{q: i for i, q in enumerate(first["qids"])}[q]] for q in qs])
    boot = _boot(dbs, B)
    f0, f1, fc = four(d0, dbs, boot, B), four(d1, dbs, boot, B), four(d1 - d0, dbs, boot, B)
    var = 0.0
    for b in bks:
        for a in (xa, ya):
            e0, e1 = vec(b, a, "ex0"), vec(b, a, "ex1")
            var += (100.0 * math.sqrt(float(((e0 == 1) & (e1 == 0)).sum() + ((e0 == 0) & (e1 == 1)).sum())) / n) ** 2
    sd_indep = math.sqrt(var) / len(bks)
    sd_direct = 100.0 * math.sqrt(float(((d1 - d0) ** 2).sum())) / n
    out = {"n": n, "complete": n >= 1534, "s0": f0, "r1": f1, "change": fc,
           "reading_s0_wide": C.reading(*f0["wide"]), "reading_r1_wide": C.reading(*f1["wide"]),
           "reading_s0_cluster": C.reading(*f0["cl"]), "reading_r1_cluster": C.reading(*f1["cl"]),
           "sd_indep": sd_indep, "sd_direct": sd_direct,
           "z_indep": (fc["est"] / sd_indep) if sd_indep > 0 else 0.0, "z_direct": (fc["est"] / sd_direct) if sd_direct > 0 else 0.0}
    out["same_wide"] = out["reading_s0_wide"] == out["reading_r1_wide"]
    out["same_cluster"] = out["reading_s0_cluster"] == out["reading_r1_cluster"]
    return out


def chain_check(runs_root, keep, B):
    """Is the root shared inside the chain as in s0? Share of questions where the root SQL (cands.jsonl, cand 0) of the gated loop equals that of the fixed loop (and of root-only equals gated), in
    s0 and in r1; and the gated-minus-fixed contrast restricted to the questions with the same root in r1 (sensitivity when the cache of the fixed loop was lost, as for Flash)."""
    out = {}
    for b in C.BK:
        if not all((b, a) in keep for a in ("gated", "fixed")):
            continue
        rt = {(a, l): root_sqls(runs_root, run_rel(b, a, l)) for a in ("gated", "fixed", "rootonly") for l in ("s0", "r1")}
        e = {}
        for l in ("s0", "r1"):
            g, f_, r_ = rt[("gated", l)], rt[("fixed", l)], rt[("rootonly", l)]
            if g is not None and f_ is not None:
                qs = sorted(set(g) & set(f_))
                e["gf_root_same_" + l] = 100.0 * sum(1 for q in qs if g[q] == f_[q]) / max(1, len(qs))
            if g is not None and r_ is not None:
                qs = sorted(set(g) & set(r_))
                e["gr_root_same_" + l] = 100.0 * sum(1 for q in qs if g[q] == r_[q]) / max(1, len(qs))
        g, f_ = rt[("gated", "r1")], rt[("fixed", "r1")]
        if g is not None and f_ is not None:
            gk, fk = keep[(b, "gated")], keep[(b, "fixed")]
            ig, jf = {q: i for i, q in enumerate(gk["qids"])}, {q: i for i, q in enumerate(fk["qids"])}
            qs = [q for q in sorted(set(ig) & set(jf)) if g.get(q) is not None and g.get(q) == f_.get(q)]
            if qs:
                dbs = np.array([gk["dbs"][ig[q]] for q in qs])
                d0 = np.array([gk["ex0"][ig[q]] - fk["ex0"][jf[q]] for q in qs])
                d1 = np.array([gk["ex1"][ig[q]] - fk["ex1"][jf[q]] for q in qs])
                boot = _boot(dbs, B)
                e["gf_same_root_n"] = len(qs)
                e["gf_same_root_s0"], e["gf_same_root_r1"] = four(d0, dbs, boot, B), four(d1, dbs, boot, B)
        out[b] = e
    return out


def round_contrasts(runs_root, keep, B):
    ctr = {}
    for cid, _, _ in CONTRASTS:
        for sc in CTR_SCOPES:
            r = contrast_runs(keep, cid, sc, B)
            if r is not None:
                ctr["%s|%s" % (cid, sc)] = r
    r = contrast_runs(keep, "w1", "weak2", B)
    if r is not None:
        ctr["w1|weak2"] = r
    return ctr, chain_check(runs_root, keep, B)


def run_partial(runs_root, B):
    res = {"mode": "partial", "arm": {}, "notes": {"left_out": {}}}
    keep = {}
    for b in C.BK:
        for arm in ARMS:
            s0 = load_loose(runs_root, run_rel(b, arm, "s0"))
            r1 = load_loose(runs_root, run_rel(b, arm, "r1"))
            tag = "%s|%s" % (b, arm)
            if r1 is None:
                res["notes"]["left_out"][tag] = "no rerun"
                continue
            if s0 is None or len(s0) < 1534:
                res["notes"]["left_out"][tag] = "s0 incomplete"
                continue
            complete = len(r1) >= 1534
            if not complete and (b, arm) not in PARTIAL_OK:
                res["notes"]["left_out"][tag] = "incomplete rerun (%d rows), not analysed" % len(r1)
                continue
            e = pair_arm(runs_root, b, arm, B, s0, r1)
            keep[(b, arm)] = e
            res["arm"][tag] = {k: v for k, v in e.items() if k not in ("qids", "ex0", "ex1", "dbs")}
            ri = response_identity(runs_root, b, arm) if complete else None
            if ri is not None:
                res["arm"][tag]["resp"] = ri
    # contrasts of the locked run (the paper's key contrasts) for the size comparison
    R = RunSet(runs_root, ("s0",))
    res["notes"]["s0_complete"] = R.complete
    if R.complete:
        c0, _, _ = contrasts_of(R, ("s0",), B)
        res["lock"] = {"%s|%s" % k: {"est": v["est"], "cl": v["cl"], "item": v["item"], "wide": v["wide"], "reading": v["reading"]} for k, v in c0.items()}
    # a contrast re-estimated in the rerun, where both arms of one backbone were rerun (same questions)
    res["pair"] = {}
    for b in C.BK:
        if (b, "zs") in keep and (b, "fixed") in keep:
            z, f_ = keep[(b, "zs")], keep[(b, "fixed")]
            qs = sorted(set(z["qids"]) & set(f_["qids"]))
            iz = {q: i for i, q in enumerate(z["qids"])}
            jf = {q: i for i, q in enumerate(f_["qids"])}
            dbs = np.array([z["dbs"][iz[q]] for q in qs])
            d0 = np.array([f_["ex0"][jf[q]] - z["ex0"][iz[q]] for q in qs])
            d1 = np.array([f_["ex1"][jf[q]] - z["ex1"][iz[q]] for q in qs])
            boot = S.ClusterBoot(dbs, B=B)
            res["pair"]["f_minus_zs|%s" % b] = {"n": len(qs), "complete": len(qs) >= 1534, "s0": four(d0, dbs, boot, B), "r1": four(d1, dbs, boot, B), "change": four(d1 - d0, dbs, boot, B)}
    res["ctr"], res["chain"] = round_contrasts(runs_root, keep, B)
    res["requests"] = requests_identity(runs_root, None)
    return res


def build_keys_partial(res, cost=None, cost_prev=None):
    K = {}

    def put(key, v, kind, ci=None, item_ci=None, meaning=None):
        e = {"v": conv(v), "kind": kind}
        if ci is not None:
            e["ci"] = [float(x) for x in ci]
        if item_ci is not None:
            e["item_ci"] = [float(x) for x in item_ci]
        if meaning:
            e["meaning"] = meaning
        K[key] = e
    for tag, e in res["arm"].items():
        b, arm = tag.split("|")
        p = "nd.%s.%s." % (b, arm)
        nm = "%s, %s" % (b, ARM_NAME[arm])
        put(p + "n", e["n"], "int", meaning="questions compared (s0 and r1 both scored), %s" % nm)
        put(p + "complete", "yes" if e["complete"] else "no", "str", meaning="rerun r1 complete (1534 questions), %s" % nm)
        put(p + "n_dbs", e["n_dbs"], "int", meaning="databases among the compared questions, %s" % nm)
        put(p + "ex_s0", e["ex_s0"], "pctv2", meaning="EX (%%) of s0 on the compared questions, %s" % nm)
        put(p + "ex_r1", e["ex_r1"], "pctv2", meaning="EX (%%) of rerun r1 on the compared questions, %s" % nm)
        put(p + "ex_diff", e["diff"]["est"], "pp2", ci=e["diff"]["cl"], item_ci=e["diff"]["item"], meaning="r1 minus s0 EX (pp), %s; ci = cluster, item_ci = item level" % nm)
        put(p + "ex_diff_wide", list(e["diff"]["wide"]), "pp2", meaning="widest interval (cluster, item-level, t(G-1), sign-flip) of r1 minus s0, %s" % nm)
        put(p + "ex_diff_reading", e["reading_wide"], "str", meaning="SUP(0) reading of the widest interval of r1 minus s0, %s" % nm)
        put(p + "sql_diff", e["sql_diff"], "pctv2", meaning="share (%%) of questions whose final SQL text differs between s0 and r1 (whitespace-normalised), %s" % nm)
        put(p + "sql_diff_raw", e["sql_diff_raw"], "pctv2", meaning="the same without whitespace normalisation, %s" % nm)
        put(p + "ex_changed", e["ex_changed"], "pctv2", meaning="share (%%) of questions whose correctness differs between s0 and r1, %s" % nm)
        put(p + "b", e["b"], "int", meaning="questions right in s0 and wrong in r1, %s" % nm)
        put(p + "c", e["c"], "int", meaning="questions wrong in s0 and right in r1, %s" % nm)
        put(p + "mcnemar_p", e["mcnemar_p"], "p", meaning="exact two-sided McNemar p of b against c, %s" % nm)
        put(p + "sd_net", e["sd_net"], "x2", meaning="implied SD (pp) of the net EX difference between two identical runs of this arm = 100 sqrt(b + c) / n, %s" % nm)
        put(p + "sd_run", e["sd_run"], "x2", meaning="implied SD (pp) of the EX of one run of this arm = sd_net / sqrt(2), %s" % nm)
        if "exemplars_differ" in e:
            put(p + "exemplars_differ", e["exemplars_differ"], "pctv2", meaning="share (%%) of questions with a different retrieved example set, %s" % nm)
        if "root_diff" in e:
            put(p + "root_diff", e["root_diff"], "pctv2", meaning="share (%%) of questions with a different root SQL, %s" % nm)
        if e.get("sql_diff_same_exemplars") is not None:
            put(p + "sql_diff_same_exemplars", e["sql_diff_same_exemplars"], "pctv2", meaning="share (%%) of final SQL texts that differ on the questions whose retrieved example set is the same in s0 and r1, %s" % nm)
        if e.get("resp"):
            put(p + "resp_n", e["resp"]["n_common"], "int", meaning="requests with the same cache key in s0 and r1 (identical model, parameters and messages), %s" % nm)
            put(p + "resp_diff", e["resp"]["diff_pct"], "pctv2", meaning="share (%%) of those requests whose stored answer differs, %s" % nm)
            for t, v in e["resp"]["by_tag"].items():
                put(p + "resp_n_" + t, v["n"], "int", meaning="identical requests of stage %s, %s" % (t, nm))
                put(p + "resp_diff_" + t, v["diff_pct"], "pctv2", meaning="share (%%) of identical %s requests whose answer differs, %s" % (t, nm))
    full = [e for e in res["arm"].values() if e["complete"]]
    if full:
        put("nd.complete.n_arms", len(full), "int", meaning="complete reruns analysed")
        put("nd.complete.sql_diff_mean", float(np.mean([e["sql_diff"] for e in full])), "pctv2", meaning="mean over the complete reruns of the share of changed final SQL texts")
        put("nd.complete.sql_diff_range", [min(e["sql_diff"] for e in full), max(e["sql_diff"] for e in full)], "pctv2", meaning="lowest and highest share of changed final SQL texts, complete reruns")
        put("nd.complete.ex_changed_mean", float(np.mean([e["ex_changed"] for e in full])), "pctv2", meaning="mean share of questions whose correctness changed, complete reruns")
        put("nd.complete.ex_changed_range", [min(e["ex_changed"] for e in full), max(e["ex_changed"] for e in full)], "pctv2", meaning="lowest and highest share of changed correctness, complete reruns")
        put("nd.complete.ex_diff_maxabs", max(abs(e["diff"]["est"]) for e in full), "x2", meaning="largest absolute r1 minus s0 EX difference (pp), complete reruns")
        put("nd.complete.ex_diff_range", [min(e["diff"]["est"] for e in full), max(e["diff"]["est"] for e in full)], "pp2", meaning="lowest and highest r1 minus s0 EX difference (pp), complete reruns")
        put("nd.complete.n_ex_diff_excl0", sum(1 for e in full if e["reading_wide"] != "contains 0"), "int", meaning="complete reruns whose r1 minus s0 EX difference excludes 0 under the widest interval")
    for k, e in res.get("pair", {}).items():
        cid, b = k.split("|")
        p = "pair.%s.%s." % (cid, b)
        put(p + "complete", "yes" if e["complete"] else "no", "str", meaning="both rerun arms complete, %s %s" % (cid, b))
        put(p + "n", e["n"], "int", meaning="questions in the pair contrast %s %s" % (cid, b))
        put(p + "s0", e["s0"]["est"], "pp2", ci=e["s0"]["cl"], item_ci=e["s0"]["item"], meaning="fixed minus single call (pp) in s0 on the compared questions, %s" % b)
        put(p + "r1", e["r1"]["est"], "pp2", ci=e["r1"]["cl"], item_ci=e["r1"]["item"], meaning="fixed minus single call (pp) in r1 on the compared questions, %s" % b)
        put(p + "change", e["change"]["est"], "pp2", ci=e["change"]["cl"], item_ci=e["change"]["item"], meaning="r1 minus s0 of that contrast (pp), %s" % b)
        put(p + "change_wide", list(e["change"]["wide"]), "pp2", meaning="widest interval of that change, %s" % b)
        put(p + "reading_s0_wide", C.reading(*e["s0"]["wide"]), "str", meaning="reading of the s0 contrast, widest interval, %s" % b)
        put(p + "reading_r1_wide", C.reading(*e["r1"]["wide"]), "str", meaning="reading of the r1 contrast, widest interval, %s" % b)
    lock = res.get("lock") or {}
    for k, e in lock.items():
        cid, sc = k.split("|")
        p = "lock.%s.%s." % (cid, sc)
        put(p + "est", e["est"], "pp2", ci=e["cl"], item_ci=e["item"], meaning="locked run s0: paired EX difference (pp), %s %s" % (cid, sc))
        put(p + "wide", list(e["wide"]), "pp2", meaning="locked run s0: widest interval, %s %s" % (cid, sc))
        put(p + "reading_wide", e["reading"]["wide"], "str", meaning="locked run s0: reading of the widest interval, %s %s" % (cid, sc))
    if lock and full:
        mx = max(abs(e["diff"]["est"]) for e in full)
        ests = {k: abs(e["est"]) for k, e in lock.items()}
        put("cmp.n_contrasts", len(lock), "int", meaning="key contrasts of the locked run compared")
        put("cmp.n_below_maxabs", sum(1 for v in ests.values() if v <= mx), "int", meaning="key contrasts whose absolute locked estimate is not larger than the largest run-to-run EX difference")
        for cid, sc in (("g_minus_f", "pooled3"), ("g_minus_f", "flash"), ("loop_g", "plus"), ("loop_g", "pooled3"), ("loop_f", "pooled3"), ("stack", "pooled3")):
            kk = "%s|%s" % (cid, sc)
            if kk in lock:
                put("cmp.%s.%s.below_maxabs" % (cid, sc), "yes" if ests[kk] <= mx else "no", "str", meaning="absolute locked estimate of %s %s not larger than the largest run-to-run EX difference" % (cid, sc))
    ctr = res.get("ctr") or {}
    for k, e in ctr.items():
        cid, sc = k.split("|")
        p = "ctr.%s.%s." % (cid, sc)
        nm = "%s %s" % (cid, sc)
        put(p + "n", e["n"], "int", meaning="questions common to the arms of the contrast %s" % nm)
        put(p + "complete", "yes" if e["complete"] else "no", "str", meaning="all 1534 questions in the contrast %s" % nm)
        put(p + "s0", e["s0"]["est"], "pp2", ci=e["s0"]["cl"], item_ci=e["s0"]["item"], meaning="locked run s0: paired EX difference (pp) of %s; ci = cluster, item_ci = item level" % nm)
        put(p + "r1", e["r1"]["est"], "pp2", ci=e["r1"]["cl"], item_ci=e["r1"]["item"], meaning="rerun r1: paired EX difference (pp) of %s" % nm)
        put(p + "change", e["change"]["est"], "pp2", ci=e["change"]["cl"], item_ci=e["change"]["item"], meaning="r1 minus s0 of the contrast (pp), %s" % nm)
        put(p + "s0_wide", list(e["s0"]["wide"]), "pp2", meaning="s0: widest interval of %s" % nm)
        put(p + "r1_wide", list(e["r1"]["wide"]), "pp2", meaning="r1: widest interval of %s" % nm)
        put(p + "change_wide", list(e["change"]["wide"]), "pp2", meaning="widest interval of the change r1 minus s0 of %s" % nm)
        put(p + "reading_s0_wide", e["reading_s0_wide"], "str", meaning="reading (SUP(0), widest interval) of s0, %s" % nm)
        put(p + "reading_r1_wide", e["reading_r1_wide"], "str", meaning="reading (widest interval) of r1, %s" % nm)
        put(p + "reading_s0_cluster", e["reading_s0_cluster"], "str", meaning="reading (cluster interval) of s0, %s" % nm)
        put(p + "reading_r1_cluster", e["reading_r1_cluster"], "str", meaning="reading (cluster interval) of r1, %s" % nm)
        put(p + "same_wide", "yes" if e["same_wide"] else "no", "str", meaning="r1 has the same widest reading as s0, %s" % nm)
        put(p + "same_cluster", "yes" if e["same_cluster"] else "no", "str", meaning="r1 has the same cluster reading as s0, %s" % nm)
        put(p + "sd_indep", e["sd_indep"], "x2", meaning="implied SD (pp) of the net change of %s between two identical runs, from the discordant counts of its arms assuming independence across arms (and backbones): approximation" % nm)
        put(p + "sd_direct", e["sd_direct"], "x2", meaning="the same from the per-question change of the contrast itself, 100 sqrt(sum v^2) / n (no independence assumption between arms), %s" % nm)
        put(p + "z_indep", e["z_indep"], "x2", meaning="observed change divided by sd_indep, %s" % nm)
        put(p + "z_direct", e["z_direct"], "x2", meaning="observed change divided by sd_direct, %s" % nm)
    if ctr:
        put("ctr.n_total", len(ctr), "int", meaning="contrasts compared between s0 and r1 (per backbone and pooled)")
        put("ctr.n_same_wide", sum(1 for e in ctr.values() if e["same_wide"]), "int", meaning="contrasts whose widest reading in r1 equals that in s0")
        put("ctr.n_same_cluster", sum(1 for e in ctr.values() if e["same_cluster"]), "int", meaning="contrasts whose cluster reading in r1 equals that in s0")
        put("ctr.changed_wide", ", ".join(k.replace("|", " ") for k, e in ctr.items() if not e["same_wide"]) or "none", "str", meaning="contrasts whose widest reading changes between s0 and r1")
        put("ctr.changed_cluster", ", ".join(k.replace("|", " ") for k, e in ctr.items() if not e["same_cluster"]) or "none", "str", meaning="contrasts whose cluster reading changes between s0 and r1")
        put("ctr.change_maxabs", max(abs(e["change"]["est"]) for e in ctr.values()), "x2", meaning="largest absolute change (pp) of a contrast between s0 and r1")
        put("ctr.n_change_excl0_wide", sum(1 for e in ctr.values() if C.reading(*e["change"]["wide"]) != "contains 0"), "int", meaning="contrasts whose change r1 minus s0 excludes 0 under the widest interval")
        sd1 = [e["sd_net"] for e in res["arm"].values()]
        put("noise.arm.sd_net_range", [min(sd1), max(sd1)], "x2", meaning="lowest and highest implied SD (pp) of the net EX difference between two identical runs, single arm")
        put("noise.arm.sd_net_mean", float(np.mean(sd1)), "x2", meaning="mean over the arms of that SD (pp)")
        two = [e["sd_indep"] for k, e in ctr.items() if k.split("|")[1] in ("plus", "flash", "27b")]
        pool = [e["sd_indep"] for k, e in ctr.items() if k.endswith("|pooled3")]
        if two:
            put("noise.two_arm.sd_indep_range", [min(two), max(two)], "x2", meaning="lowest and highest implied SD (pp) of a two-arm contrast per backbone, independence across arms (approximation)")
        if pool:
            put("noise.pooled.sd_indep_range", [min(pool), max(pool)], "x2", meaning="lowest and highest implied SD (pp) of a pooled three-backbone contrast, independence across arms and backbones (approximation)")
        gf = ctr.get("g_minus_f|pooled3")
        if gf:
            put("p1p.s0_est", gf["s0"]["est"], "pp2", ci=gf["s0"]["cl"], item_ci=gf["s0"]["item"], meaning="P1': pooled gated minus fixed, locked run (pp)")
            put("p1p.r1_est", gf["r1"]["est"], "pp2", ci=gf["r1"]["cl"], item_ci=gf["r1"]["item"], meaning="P1': pooled gated minus fixed, rerun r1 (pp)")
            put("p1p.r1_wide", list(gf["r1"]["wide"]), "pp2", meaning="P1': widest interval of the rerun r1")
            put("p1p.margin", -MARGIN, "pp2", meaning="P1' margin (pp)")
            put("p1p.s0_clearance_wide", gf["s0"]["wide"][0] + MARGIN, "pp2", meaning="lower end of the widest interval of s0 minus the margin (pp); positive = margin kept")
            put("p1p.r1_clearance_wide", gf["r1"]["wide"][0] + MARGIN, "pp2", meaning="lower end of the widest interval of r1 minus the margin (pp)")
            put("p1p.r1_clearance_cluster", gf["r1"]["cl"][0] + MARGIN, "pp2", meaning="lower end of the cluster interval of r1 minus the margin (pp)")
            put("p1p.r1_met_wide", "yes" if gf["r1"]["wide"][0] > -MARGIN else "no", "str", meaning="widest lower end of r1 above -1.5 pp")
            put("p1p.r1_met_cluster", "yes" if gf["r1"]["cl"][0] > -MARGIN else "no", "str", meaning="cluster lower end of r1 above -1.5 pp")
    for b, e in (res.get("chain") or {}).items():
        for fld in ("gf_root_same_s0", "gf_root_same_r1", "gr_root_same_s0", "gr_root_same_r1"):
            if fld in e:
                put("chain.%s.%s" % (b, fld), e[fld], "pctv2", meaning="share (%%) of questions with the same root SQL in %s, %s (%s)" % (
                    "gated and fixed" if fld.startswith("gf") else "root-only and gated", b, fld[-2:]))
        if "gf_same_root_n" in e:
            put("chain.%s.gf_same_root_n" % b, e["gf_same_root_n"], "int", meaning="questions where the gated and fixed r1 roots are the same, %s" % b)
            put("chain.%s.gf_same_root_s0" % b, e["gf_same_root_s0"]["est"], "pp2", ci=e["gf_same_root_s0"]["cl"], meaning="gated minus fixed (pp) in s0 on those questions, %s" % b)
            put("chain.%s.gf_same_root_r1" % b, e["gf_same_root_r1"]["est"], "pp2", ci=e["gf_same_root_r1"]["cl"], meaning="gated minus fixed (pp) in r1 on those questions, %s" % b)
    rq = res.get("requests") or {}
    r1s = {k: e for k, e in rq.items() if k != "_all" and k.endswith("|r1") and "|".join(k.split("|")[:2]).replace("|", "|") in res["arm"]}
    for k, e in r1s.items():
        b, arm, _ = k.split("|")
        put("req.%s.%s.in_s0_rows" % (b, arm), e["in_s0_rows"], "pctv2", meaning="share (%%) of rerun chat requests whose cache key occurs in the s0 run of the same arm, %s %s" % (b, arm))
        put("req.%s.%s.cached_rows" % (b, arm), e["cached_rows"], "pctv2", meaning="share (%%) of rerun chat requests served from the run's own cache, %s %s" % (b, arm))
        put("req.%s.%s.n_rows" % (b, arm), e["n_rows"], "int", meaning="chat request rows of the rerun, %s %s" % (b, arm))
        for tag, t in e["by_tag"].items():
            put("req.%s.%s.tag_%s_in_s0" % (b, arm, tag), t["in_s0"], "pctv2", meaning="share (%%) of the %s requests found in s0, %s %s" % (tag, b, arm))
    lf = os.path.join(os.path.dirname(os.path.dirname(HERE)), "exp", "dev_lock.json")
    if os.path.exists(lf):
        import hashlib
        raw = open(lf, "rb").read()
        ents = json.loads(raw.decode("utf-8")).get("entries", [])
        put("lock_file.n_entries", len(ents), "int", meaning="entries of the lock file after the dated rerun entries")
        put("lock_file.n_t0", sum(1 for e in ents if e.get("date", "") >= "2026-10-28" and e.get("seeds") == [0]), "int", meaning="dated entries with seed 0 only (the registration of the T 0 rerun)")
        put("lock_file.sha12", hashlib.sha256(raw).hexdigest()[:12], "str", meaning="first 12 hex characters of the sha256 of the lock file")
    if cost:
        put("cost.total_cny", cost["total"]["cny_disc"], "cny2", meaning="CNY of everything that ran in the R4T0 queue (completed and aborted runs), with the cache discount")
        put("cost.list_cny", cost["total"]["cny_nodisc"], "cny2", meaning="the same at list price")
        put("cost.calls", cost["total"]["calls"], "int", meaning="billed chat calls of the R4T0 queue")
        if cost_prev:
            put("cost.round5_cny", cost["total"]["cny_disc"] - cost_prev["total"]["cny_disc"], "cny2", meaning="CNY of the round-5 completion runs (all R4T0 minus the first batch), with the cache discount")
            put("cost.round5_list_cny", cost["total"]["cny_nodisc"] - cost_prev["total"]["cny_nodisc"], "cny2", meaning="the same at list price")
            put("cost.round5_calls", cost["total"]["calls"] - cost_prev["total"]["calls"], "int", meaning="billed chat calls of the round-5 completion runs")
    return K


# ------------------------------------------------------------------------------------------------------------------------------- outputs
def conv(o):
    if isinstance(o, dict):
        return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [conv(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def f3(x):
    return "" if x is None else "%.3f" % x


def write_csvs(res, out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "arms_nd.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backbone", "arm", "ex_s0", "ex_r1", "ex_r2", "ex_mean", "ex_sd", "sd_lo", "sd_hi", "sql_diff_s0_r1", "sql_diff_s0_r2", "sql_diff_r1_r2", "sql_diff_vs_s0",
                    "sql_diff_raw_vs_s0", "ex_diff_s0_r1", "ex_diff_s0_r2", "ex_diff_r1_r2", "ex_diff_vs_s0", "ex_unstable", "flip_s0_r1(r->w;w->r)", "flip_s0_r2(r->w;w->r)", "root_diff_vs_s0",
                    "exemplars_differ_vs_s0", "sql_diff_same_exemplars_vs_s0"])
        for k, e in res["nd"].get("arm", {}).items():
            b, a = k.split("|")
            w.writerow([b, a] + [f3(e["ex"][l]) for l in T0] + [f3(e["ex_mean"]), f3(e["ex_sd"]), f3(e["ex_sd_ci"][0]), f3(e["ex_sd_ci"][1])] +
                       [f3(e["sql_diff"][p]) for p in ("s0_r1", "s0_r2", "r1_r2", "vs_s0")] + [f3(e["sql_diff_raw"]["vs_s0"])] +
                       [f3(e["ex_diff"][p]) for p in ("s0_r1", "s0_r2", "r1_r2", "vs_s0")] + [f3(e["ex_unstable"])] +
                       ["%d;%d" % (e["ex_flip"][p]["right_to_wrong"], e["ex_flip"][p]["wrong_to_right"]) for p in ("s0_r1", "s0_r2")] +
                       [f3((e.get("root_diff") or {}).get("vs_s0")), f3((e.get("exemplars_differ") or {}).get("vs_s0")), f3(e.get("sql_diff_same_exemplars_vs_s0"))])
    for name, key in (("contrasts_T0.csv", "T0"), ("contrasts_T07.csv", "T07")):
        if not res.get(key):
            continue
        labs = T0 if key == "T0" else T07
        with open(os.path.join(out, name), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["contrast", "scope"] + ["est_" + l for l in labs] + ["mean", "sd_runs", "cl_lo", "cl_hi", "item_lo", "item_hi", "t_lo", "t_hi", "sf_lo", "sf_hi", "wide_lo", "wide_hi",
                                                                             "range_lo", "range_hi", "t2_lo", "t2_hi", "wide_alt_lo", "wide_alt_hi", "reading_cluster", "reading_wide", "reading_wide_alt"] +
                       (["s0_est", "s0_wide_lo", "s0_wide_hi", "s0_reading_cluster", "s0_reading_wide", "agrees_cluster", "agrees_wide", "r1_reading_wide", "r2_reading_wide"] if key == "T0" else []))
            for k, e in res[key].items():
                c, sc = k.split("|")
                row = [c, sc] + [f3(e["per"][l]) for l in labs] + [f3(e["est"]), f3(e["sd_runs"])] + [f3(x) for x in (e["cl"] + e["item"] + e["t"] + e["sf"] + e["wide"] + e["range"] + e["t2"] + e["wide_alt"])] + \
                      [e["reading"]["cluster"], e["reading"]["wide"], e["reading"]["wide_alt"]]
                if key == "T0":
                    s0 = e["single"]["s0"]
                    row += [f3(s0["est"]), f3(s0["wide"][0]), f3(s0["wide"][1]), s0["reading"]["cluster"], s0["reading"]["wide"],
                            "yes" if e["agrees"]["cluster"] else "no", "yes" if e["agrees"]["wide"] else "no", e["single"]["r1"]["reading"]["wide"], e["single"]["r2"]["reading"]["wide"]]
                w.writerow(row)
    if res.get("cmp"):
        with open(os.path.join(out, "compare_T07.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["contrast", "scope", "t0_est", "t0_reading_wide", "t07_est", "t07_reading_wide", "same_reading_wide", "same_reading_cluster", "setting_diff", "setting_wide_lo",
                        "setting_wide_hi", "setting_reading_wide", "t07_est_inside_t0_wide", "t0_runs_negative", "t0_runs_positive"])
            for k, e in res["cmp"].items():
                c, sc = k.split("|")
                w.writerow([c, sc, f3(e["t0_est"]), e["t0_reading_wide"], f3(e["t07_est"]), e["t07_reading_wide"], "yes" if e["same_wide"] else "no", "yes" if e["same_cluster"] else "no",
                            f3(e["setting_diff"]["est"]), f3(e["setting_diff"]["wide"][0]), f3(e["setting_diff"]["wide"][1]), e["setting_reading_wide"],
                            "yes" if e["t07_est_inside_t0_wide"] else "no", e["t0_runs_negative"], e["t0_runs_positive"]])


# ------------------------------------------------------------------------------------------------------------------------------- keys (RV4T0.*)
def build_keys(res, cost=None):
    """flat key dict in the format of the RV2 / RV3 summary files: {key: {"v","kind","ci","item_ci","n","meaning"}}."""
    K = {}

    def put(key, v, kind, ci=None, item_ci=None, meaning=None):
        e = {"v": conv(v), "kind": kind}
        if ci is not None:
            e["ci"] = [float(x) for x in ci]
        if item_ci is not None:
            e["item_ci"] = [float(x) for x in item_ci]
        if meaning:
            e["meaning"] = meaning
        K[key] = e
    nd = res.get("nd") or {}
    for k, e in nd.get("arm", {}).items():
        b, arm = k.split("|")
        p = "arm.%s.%s." % (b, arm)
        nm = "%s, %s" % (b, ARM_NAME[arm])
        for l in T0:
            put(p + "ex_" + l, e["ex"][l], "pctv2", meaning="EX (%%) of run %s, %s, T 0" % (l, nm))
        put(p + "ex_mean", e["ex_mean"], "pctv2", meaning="mean EX (%%) over s0, r1, r2, %s" % nm)
        put(p + "ex_sd", e["ex_sd"], "x2", meaning="SD (pp, ddof 1) of EX over s0, r1, r2, %s" % nm)
        put(p + "ex_sd_ci", list(e["ex_sd_ci"]), "x2", meaning="chi-square interval of that SD (2 degrees of freedom), %s" % nm)
        put(p + "ex_range", list(e["ex_range"]), "pctv2", meaning="lowest and highest EX (%%) of the three T 0 runs, %s" % nm)
        for pr in ("s0_r1", "s0_r2", "r1_r2", "vs_s0"):
            put(p + "sql_diff_" + pr, e["sql_diff"][pr], "pctv2", meaning="share (%%) of questions whose final SQL text differs, %s, %s" % (pr, nm))
            put(p + "ex_diff_" + pr, e["ex_diff"][pr], "pctv2", meaning="share (%%) of questions whose correctness differs, %s, %s" % (pr, nm))
        put(p + "sql_diff_raw_vs_s0", e["sql_diff_raw"]["vs_s0"], "pctv2", meaning="as sql_diff_vs_s0 without whitespace normalisation, %s" % nm)
        put(p + "ex_unstable", e["ex_unstable"], "pctv2", meaning="share (%%) of questions whose EX is not the same in s0, r1 and r2, %s" % nm)
        put(p + "flip_s0_r1", [e["ex_flip"]["s0_r1"]["right_to_wrong"], e["ex_flip"]["s0_r1"]["wrong_to_right"]], "int", meaning="questions right in s0 and wrong in r1, and wrong in s0 and right in r1, %s" % nm)
        put(p + "flip_s0_r2", [e["ex_flip"]["s0_r2"]["right_to_wrong"], e["ex_flip"]["s0_r2"]["wrong_to_right"]], "int", meaning="as flip_s0_r1 for r2, %s" % nm)
        if "root_diff" in e:
            put(p + "root_diff_vs_s0", e["root_diff"]["vs_s0"], "pctv2", meaning="share (%%) of questions whose root SQL differs from s0 (mean of r1, r2), %s" % nm)
        if "exemplars_differ" in e:
            put(p + "exemplars_differ_vs_s0", e["exemplars_differ"]["vs_s0"], "pctv2", meaning="share (%%) of questions whose retrieved example set differs from s0, %s" % nm)
            if e.get("sql_diff_same_exemplars_vs_s0") is not None:
                put(p + "sql_diff_same_exemplars_vs_s0", e["sql_diff_same_exemplars_vs_s0"], "pctv2", meaning="share (%%) of final SQL texts that differ from s0 on questions with the same example set in all three runs, %s" % nm)
    for g, e in nd.get("group", {}).items():
        for fld, v in e.items():
            if v is None:
                continue
            put("nd.%s.%s" % (g, fld), v, "int" if fld == "n_arms" else ("x2" if fld.startswith("ex_sd") else "pctv2"), meaning="%s over the arms of group %s" % (fld, g))
    rq = res.get("requests") or {}
    for k, e in rq.items():
        if k == "_all":
            continue
        b, arm, lab = k.split("|")
        put("req.%s.%s.%s.in_s0_rows" % (b, arm, lab), e["in_s0_rows"], "pctv2", meaning="share (%%) of the chat requests of the rerun whose cache key occurs in the s0 run of the same arm, %s %s %s" % (b, arm, lab))
        put("req.%s.%s.%s.cached_rows" % (b, arm, lab), e["cached_rows"], "pctv2", meaning="share (%%) of the chat requests answered from the run's own cache (roots shared inside the chain), %s %s %s" % (b, arm, lab))
    if "_all" in rq:
        for fld in ("in_s0_rows_mean", "in_s0_rows_min", "in_s0_rows_max"):
            put("req.all." + fld, rq["_all"][fld], "pctv2", meaning="share (%%) of rerun requests found in s0, %s over arms and reruns" % fld)
        put("req.all.n_rows", rq["_all"]["n_rows"], "int", meaning="chat request rows of the six reruns' arms (usage.jsonl)")
    for k, e in (res.get("T0") or {}).items():
        cid, sc = k.split("|")
        p = "c.%s.%s." % (cid, sc)
        s0 = e["single"]["s0"]
        put(p + "est", e["est"], "pp2", ci=e["cl"], item_ci=e["item"], meaning="T 0 triple (s0, r1, r2): mean paired EX difference (pp), %s %s; ci = cluster, item_ci = item level" % (cid, sc))
        for l in T0:
            put(p + "run_" + l, e["per"][l], "pp2", meaning="paired EX difference (pp) of run %s alone, %s %s" % (l, cid, sc))
        put(p + "sd_runs", e["sd_runs"], "x2", meaning="SD over the three runs of the per-run difference (pp), %s %s" % (cid, sc))
        put(p + "t_ci", list(e["t"]), "pp2", meaning="cluster-robust t(G-1) interval, %s %s" % (cid, sc))
        put(p + "sf_ci", list(e["sf"]), "pp2", meaning="exact sign-flip interval, %s %s" % (cid, sc))
        put(p + "wide", list(e["wide"]), "pp2", meaning="widest interval = hull of the cluster, item-level, t(G-1) and sign-flip intervals, %s %s" % (cid, sc))
        put(p + "range", list(e["range"]), "pp2", meaning="between-run range of the three per-run differences, %s %s" % (cid, sc))
        put(p + "wide_alt", list(e["wide_alt"]), "pp2", meaning="hull of cluster, item-level, between-run range and t(2) interval (the R3 definition), %s %s" % (cid, sc))
        put(p + "reading_cluster", e["reading"]["cluster"], "str", meaning="SUP(0) reading of the cluster interval, T 0 triple, %s %s" % (cid, sc))
        put(p + "reading_wide", e["reading"]["wide"], "str", meaning="SUP(0) reading of the widest interval, T 0 triple, %s %s" % (cid, sc))
        put(p + "reading_wide_alt", e["reading"]["wide_alt"], "str", meaning="reading of wide_alt, T 0 triple, %s %s" % (cid, sc))
        put(p + "s0_est", s0["est"], "pp2", ci=s0["cl"], item_ci=s0["item"], meaning="locked run s0 alone: paired EX difference (pp), %s %s" % (cid, sc))
        put(p + "s0_wide", list(s0["wide"]), "pp2", meaning="locked run s0 alone: widest interval, %s %s" % (cid, sc))
        put(p + "s0_reading_cluster", s0["reading"]["cluster"], "str", meaning="reading of s0 alone, cluster interval, %s %s" % (cid, sc))
        put(p + "s0_reading_wide", s0["reading"]["wide"], "str", meaning="reading of s0 alone, widest interval, %s %s" % (cid, sc))
        for l in ("r1", "r2"):
            put(p + l + "_reading_wide", e["single"][l]["reading"]["wide"], "str", meaning="reading of run %s alone, widest interval, %s %s" % (l, cid, sc))
        put(p + "agrees_cluster", "yes" if e["agrees"]["cluster"] else "no", "str", meaning="triple reading (cluster) equals the s0 reading (cluster), %s %s" % (cid, sc))
        put(p + "agrees_wide", "yes" if e["agrees"]["wide"] else "no", "str", meaning="triple reading (widest) equals the s0 reading (widest), %s %s" % (cid, sc))
    T0c = res.get("T0") or {}
    if T0c:
        ag_w = sum(1 for e in T0c.values() if e["agrees"]["wide"])
        ag_c = sum(1 for e in T0c.values() if e["agrees"]["cluster"])
        put("agree.n_total", len(T0c), "int", meaning="contrasts of the replication table")
        put("agree.n_wide", ag_w, "int", meaning="contrasts whose T 0 triple reading (widest) equals the s0 reading")
        put("agree.n_cluster", ag_c, "int", meaning="contrasts whose T 0 triple reading (cluster) equals the s0 reading")
        put("agree.changed_wide", ", ".join(k.replace("|", " ") for k, e in T0c.items() if not e["agrees"]["wide"]) or "none", "str", meaning="contrasts whose widest reading changes against s0")
        put("agree.changed_cluster", ", ".join(k.replace("|", " ") for k, e in T0c.items() if not e["agrees"]["cluster"]) or "none", "str", meaning="contrasts whose cluster reading changes against s0")
        # single runs against s0's reading: how often does one more T = 0 run give another reading
        for l in ("r1", "r2"):
            n_same = sum(1 for e in T0c.values() if e["single"][l]["reading"]["wide"] == e["single"]["s0"]["reading"]["wide"])
            put("agree.single_%s_same_as_s0" % l, n_same, "int", meaning="contrasts for which run %s alone has the same widest reading as s0 alone" % l)
        gf = T0c.get("g_minus_f|pooled3")
        if gf:
            put("p1p.est", gf["est"], "pp2", ci=gf["cl"], item_ci=gf["item"], meaning="P1': pooled gated minus fixed, T 0 triple (pp)")
            put("p1p.wide", list(gf["wide"]), "pp2", meaning="P1': widest interval of the T 0 triple")
            put("p1p.margin", -MARGIN, "pp2", meaning="P1' margin (pp)")
            put("p1p.clearance_wide", gf["wide"][0] + MARGIN, "pp2", meaning="lower end of the widest interval minus the margin (pp); positive = margin kept")
            put("p1p.clearance_cluster", gf["cl"][0] + MARGIN, "pp2", meaning="lower end of the cluster interval minus the margin (pp)")
            put("p1p.met_wide", "yes" if gf["wide"][0] > -MARGIN else "no", "str", meaning="widest lower end above -1.5 pp")
            put("p1p.met_cluster", "yes" if gf["cl"][0] > -MARGIN else "no", "str", meaning="cluster lower end above -1.5 pp")
            s0 = gf["single"]["s0"]
            put("p1p.s0_clearance_wide", s0["wide"][0] + MARGIN, "pp2", meaning="the same for the locked run alone")
    for k, e in (res.get("calls_T0") or {}).items():
        put("calls.%s.ratio_mean" % k, float(np.mean(list(e.values()))), "x3", meaning="gated / fixed billed calls, mean over s0, r1, r2, %s" % k)
        put("calls.%s.ratio_range" % k, [min(e.values()), max(e.values())], "x3", meaning="range over the three runs, %s" % k)
    for k, e in (res.get("T07") or {}).items():
        cid, sc = k.split("|")
        p = "c07.%s.%s." % (cid, sc)
        put(p + "est", e["est"], "pp2", ci=e["cl"], item_ci=e["item"], meaning="T 0.7 replication under the same definition: mean paired EX difference (pp), %s %s" % (cid, sc))
        put(p + "wide", list(e["wide"]), "pp2", meaning="T 0.7 replication: widest interval (four intervals of the Methods), %s %s" % (cid, sc))
        put(p + "reading_wide", e["reading"]["wide"], "str", meaning="T 0.7 replication: reading of the widest interval, %s %s" % (cid, sc))
        put(p + "reading_cluster", e["reading"]["cluster"], "str", meaning="T 0.7 replication: reading of the cluster interval, %s %s" % (cid, sc))
    for k, e in (res.get("cmp") or {}).items():
        cid, sc = k.split("|")
        p = "cmp.%s.%s." % (cid, sc)
        put(p + "same_wide", "yes" if e["same_wide"] else "no", "str", meaning="T 0 triple and T 0.7 replication have the same widest reading, %s %s" % (cid, sc))
        put(p + "same_cluster", "yes" if e["same_cluster"] else "no", "str", meaning="same cluster reading at T 0 and T 0.7, %s %s" % (cid, sc))
        put(p + "setting_diff", e["setting_diff"]["est"], "pp2", ci=e["setting_diff"]["cl"], item_ci=e["setting_diff"]["item"], meaning="T 0.7 minus T 0 of the contrast (pp), %s %s" % (cid, sc))
        put(p + "setting_wide", list(e["setting_diff"]["wide"]), "pp2", meaning="widest interval of that setting difference, %s %s" % (cid, sc))
        put(p + "setting_reading_wide", e["setting_reading_wide"], "str", meaning="reading of the setting difference (widest), %s %s" % (cid, sc))
        put(p + "t0_runs_negative", e["t0_runs_negative"], "int", meaning="runs of the T 0 triple whose per-run estimate is below 0, %s %s" % (cid, sc))
    cm = res.get("cmp") or {}
    if cm:
        put("cmp.n_total", len(cm), "int", meaning="contrasts compared between T 0 and T 0.7")
        put("cmp.n_same_wide", sum(1 for e in cm.values() if e["same_wide"]), "int", meaning="contrasts with the same widest reading at T 0 and T 0.7")
        put("cmp.n_same_cluster", sum(1 for e in cm.values() if e["same_cluster"]), "int", meaning="contrasts with the same cluster reading at T 0 and T 0.7")
        for nm, kk in (("gf_flash", "g_minus_f|flash"), ("gf_pooled", "g_minus_f|pooled3")):
            if kk in cm:
                e = cm[kk]
                put("cmp.%s.t0_negative_wide" % nm, "yes" if e["t0_reading_wide"] == "negative" else "no", "str", meaning="gated minus fixed (%s) reads negative under the widest interval at T 0" % nm)
                put("cmp.%s.t0_negative_cluster" % nm, "yes" if e["t0_reading_cluster"] == "negative" else "no", "str", meaning="same under the cluster interval at T 0")
    if cost:
        put("cost.total_cny", cost["total"]["cny_disc"], "cny2", meaning="CNY of the six T 0 reruns, with the cache discount")
        put("cost.list_cny", cost["total"]["cny_nodisc"], "cny2", meaning="CNY of the six T 0 reruns at list price")
        put("cost.calls", cost["total"]["calls"], "int", meaning="billed chat calls of the reruns")
        for lab in ("r1", "r2"):
            sel = [v for k, v in cost.get("by_run", {}).items() if k.endswith("-" + lab)]
            if sel:
                put("cost.%s_cny" % lab, sum(v["cny_disc"] for v in sel), "cny2", meaning="CNY of rerun %s, with the cache discount" % lab)
    return K



def write_partial_csv(res, out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "arms_partial.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backbone", "arm", "n", "complete", "n_dbs", "ex_s0", "ex_r1", "diff_r1_minus_s0", "cl_lo", "cl_hi", "item_lo", "item_hi", "wide_lo", "wide_hi", "reading_wide",
                    "sql_diff", "ex_changed", "b_right_to_wrong", "c_wrong_to_right", "mcnemar_p", "root_diff", "exemplars_differ", "sd_net_pp"])
        for k, e in res["arm"].items():
            b, a = k.split("|")
            d = e["diff"]
            w.writerow([b, a, e["n"], "yes" if e["complete"] else "no", e["n_dbs"], f3(e["ex_s0"]), f3(e["ex_r1"]), f3(d["est"])] + [f3(x) for x in d["cl"] + d["item"] + d["wide"]] +
                       [e["reading_wide"], f3(e["sql_diff"]), f3(e["ex_changed"]), e["b"], e["c"], "%.4f" % e["mcnemar_p"], f3(e.get("root_diff")), f3(e.get("exemplars_differ")), f3(e.get("sd_net"))])
    if res.get("ctr"):
        with open(os.path.join(out, "contrasts_r1.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["contrast", "scope", "n", "est_s0", "s0_wide_lo", "s0_wide_hi", "reading_s0_wide", "est_r1", "r1_wide_lo", "r1_wide_hi", "reading_r1_wide", "change_r1_minus_s0",
                        "change_cl_lo", "change_cl_hi", "change_wide_lo", "change_wide_hi", "same_reading_wide", "same_reading_cluster", "sd_indep_pp", "sd_direct_pp", "z_indep", "z_direct"])
            for k, e in res["ctr"].items():
                c_, sc = k.split("|")
                w.writerow([c_, sc, e["n"], f3(e["s0"]["est"]), f3(e["s0"]["wide"][0]), f3(e["s0"]["wide"][1]), e["reading_s0_wide"], f3(e["r1"]["est"]), f3(e["r1"]["wide"][0]), f3(e["r1"]["wide"][1]),
                            e["reading_r1_wide"], f3(e["change"]["est"]), f3(e["change"]["cl"][0]), f3(e["change"]["cl"][1]), f3(e["change"]["wide"][0]), f3(e["change"]["wide"][1]),
                            "yes" if e["same_wide"] else "no", "yes" if e["same_cluster"] else "no", f3(e["sd_indep"]), f3(e["sd_direct"]), f3(e["z_indep"]), f3(e["z_direct"])])
    with open(os.path.join(out, "size_comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "item", "estimate_pp", "widest_lo", "widest_hi", "reading"])
        for k, e in res["arm"].items():
            d = e["diff"]
            w.writerow(["rerun r1 minus s0", k.replace("|", " ") + ("" if e["complete"] else " (partial, n=%d)" % e["n"]), f3(d["est"]), f3(d["wide"][0]), f3(d["wide"][1]), e["reading_wide"]])
        for k, e in res.get("pair", {}).items():
            w.writerow(["change of the contrast, r1 minus s0", k.replace("|", " ") + ("" if e["complete"] else " (partial, n=%d)" % e["n"]), f3(e["change"]["est"]), f3(e["change"]["wide"][0]), f3(e["change"]["wide"][1]),
                        C.reading(*e["change"]["wide"])])
        for k, e in (res.get("lock") or {}).items():
            w.writerow(["locked-run contrast", k.replace("|", " "), f3(e["est"]), f3(e["wide"][0]), f3(e["wide"][1]), e["reading"]["wide"]])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=C.B_DEFAULT)
    ap.add_argument("--cost", default=None, help="cost.json of tools/r3_cost.py --dirs R4T0")
    ap.add_argument("--cost-prev", default=None, help="cost.json of the first batch (results/r4_t0/cost_round4.json): the round-5 cost is the difference")
    ap.add_argument("--mode", default="auto", choices=["auto", "full", "partial"], help="full = triple s0/r1/r2 for all arms; partial = s0 against r1 of the arms that ran (reduced scope after the worker machine loss)")
    a = ap.parse_args(argv)
    mode = a.mode
    if mode == "auto":
        mode = "full" if RunSet(a.runs, T0).complete else "partial"
    res = run(a.runs, a.B) if mode == "full" else run_partial(a.runs, a.B)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "t0.json" if mode == "full" else "t0_partial.json"), "w") as f:
        json.dump(conv(res), f, indent=1)
    cost = json.load(open(a.cost)) if a.cost and os.path.exists(a.cost) else None
    if mode == "full":
        write_csvs(res, a.out)
        K = build_keys(res, cost)
    else:
        write_partial_csv(res, a.out)
        cprev = json.load(open(a.cost_prev)) if a.cost_prev and os.path.exists(a.cost_prev) else None
        K = build_keys_partial(res, cost, cprev)
    import time
    meta = {"scope": "posthoc-secondary", "api_calls": int(cost["total"]["calls"]) if cost else 0, "B": a.B, "seed_boot": S.SEED_BOOT, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "complete": bool(res["notes"].get("complete_T0")) if mode == "full" else not res["notes"]["left_out"],
            "mode": mode if (mode == "full" or res["notes"]["left_out"]) else "r1",
            "n_qids": res["notes"]["n_qids"] if mode == "full" else 1534, "missing_keys": []}
    with open(os.path.join(a.out, "rv4t0_summary.json"), "w") as f:
        json.dump({"_meta": meta, "keys": K}, f, indent=1)
    if mode == "full":
        print("[r4_t0] qids %d; complete T0 %s; T0 contrasts %d; T07 contrasts %d; keys %d" % (res["notes"]["n_qids"], res["notes"].get("complete_T0"), len(res.get("T0") or {}), len(res.get("T07") or {}), len(K)))
        for k, v in res["notes"].get("run_notes", {}).items():
            print("[r4_t0] run %s: %s" % (k, v))
    else:
        print("[r4_t0] partial mode: arms %s; left out %s; keys %d" % (sorted(res["arm"]), res["notes"]["left_out"], len(K)))


if __name__ == "__main__":
    main()
