#!/usr/bin/env python3
"""R3 / E-S: replication of the decomposition. Per backbone and decomposition arm (single call, root-only, gated loop, fixed loop) the headline run (seed 0, T = 0,
locked) and a REPLICATION of three runs at T = 0.7 (seeds 101, 202, 303; the protocol of the existing Plus stability runs). The headline run is reported apart and
is never averaged with the replication (author decision 2026-10-28: a T = 0 run and T = 0.7 runs are different estimands). Post hoc, descriptive (SAP section 26).

    worker machine:  PYTHONPATH=<code> python code/tools/r3_seeds.py --runs $R/runs --out results/r3 [--B 10000]

For every arm: EX per run, the T = 0.7 mean, SD over the three T = 0.7 runs (ddof 1), range; the seed-0 EX beside it. For every paired contrast: per-run estimates, the
estimate on the run-averaged per-question correctness (mean over the three T = 0.7 runs of the 0/1 EX), the database-clustered percentile bootstrap (B = 10 000,
seed 20261101) and the item-level paired bootstrap on that run-averaged vector, the between-run range, the t interval (2 df) of the three per-run estimates, the hull
of the four ("wide"), the reading, and the reading of the seed-0 run (cluster and item hull) with the flag whether the two agree. Also: stack-versus-loop split per backbone and the root-regeneration audit (how many questions have different root SQL in
the arms of one run, and the loop effect on questions with a shared root). No model call."""
import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3_common as C  # noqa: E402
from r3_common import S  # noqa: E402

CONTRASTS = [
    # id, (x arm, y arm) on one backbone: x minus y
    ("stack", "rootonly", "zs"), ("loop_g", "gated", "rootonly"), ("loop_f", "fixed", "rootonly"), ("g_minus_f", "gated", "fixed"),
    ("g_minus_zs", "gated", "zs"), ("f_minus_zs", "fixed", "zs"),
]
POOLED = ("stack", "loop_g", "loop_f", "g_minus_f", "g_minus_zs", "f_minus_zs")     # mean over the three backbones (the P1' estimator for g_minus_f)
SCOPES = ("plus", "flash", "27b", "pooled3")
T2 = 4.302652729911275          # t(0.975, 2): three runs


def sd_interval(sd, n=3):
    from scipy import stats as st
    df = n - 1
    return (sd * np.sqrt(df / st.chi2.ppf(0.975, df)), sd * np.sqrt(df / st.chi2.ppf(0.025, df)))


class Data(object):
    def __init__(self, runs_root, qids=None):
        self.root = runs_root
        self.rows = {}              # (b, arm, seed) -> {qid: row}
        self.note = {}
        for b in C.BK:
            for arm in C.ARMS:
                for s in C.SEEDS:
                    try:
                        rel = C.run_path(b, arm, s)
                    except KeyError:
                        continue
                    rows = C.load_preds(runs_root, rel)
                    ok, why = C.check_run(rows)
                    self.note[(b, arm, s)] = why
                    if ok:
                        self.rows[(b, arm, s)] = rows
        common = None
        for rows in self.rows.values():
            q = set(k for k, r in rows.items() if C.ex_of(r) is not None)
            common = q if common is None else common & q
        self.qids = sorted(common if qids is None else common & set(qids))
        self.dbs = np.array([self.rows[next(iter(self.rows))][q]["db_id"] for q in self.qids])
        self.ex = {k: np.array([C.ex_of(rows[q]) for q in self.qids], float) for k, rows in self.rows.items()}
        self.calls = {k: np.array([float(rows[q].get("calls") or 0.0) for q in self.qids], float) for k, rows in self.rows.items()}

    def seeds(self, b, arm):
        return [s for s in C.SEEDS if (b, arm, s) in self.ex]

    def reps(self, b, arm):
        return [s for s in C.REP if (b, arm, s) in self.ex]

    def avg(self, b, arm, seeds):
        return np.mean([self.ex[(b, arm, s)] for s in seeds], axis=0)


def rep_avg(d_by_seed):
    """mean over the T = 0.7 replication runs of the per-question vectors (seed 0 is never part of it); None if no replication run is present."""
    v = [d_by_seed[s] for s in sorted(d_by_seed) if s in C.REP]
    return np.mean(v, axis=0) if v else None


def contrast_stats(d_by_seed, dbs, boot, B):
    """d_by_seed: {seed: per-question difference vector}, may contain seed 0 (T = 0, reported apart) and the replication seeds 101 / 202 / 303 (T = 0.7).
    The estimate is the mean over the replication runs; the intervals are those of the run-averaged vector."""
    seeds = [s for s in sorted(d_by_seed) if s in C.REP]
    avg_vec = rep_avg(d_by_seed)
    per = {s: 100.0 * d_by_seed[s].mean() for s in seeds}
    est, ci, ic = C.boot_ci(avg_vec, dbs, B=B, boot=boot)
    vals = [per[s] for s in seeds]
    out = {"per": per, "est": est, "ci": ci, "item_ci": ic, "n_runs": len(seeds), "seeds": seeds,
           "range": (min(vals), max(vals)), "sd": (float(np.std(vals, ddof=1)) if len(vals) >= 2 else None)}
    if len(vals) == 3:
        m, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
        out["t_ci"] = (m - T2 * sd / np.sqrt(3), m + T2 * sd / np.sqrt(3))
        out["wide"] = C.hull(ci, ic, out["range"], out["t_ci"])
    else:
        out["t_ci"] = None
        out["wide"] = C.hull(ci, ic, out["range"])
    # the headline run (seed 0, T = 0) read the way the paper reads it, reported apart
    if 0 in d_by_seed:
        e0, c0, i0 = C.boot_ci(d_by_seed[0], dbs, B=B, boot=boot)
        out["s0"] = {"est": e0, "ci": c0, "item_ci": i0, "wide": C.hull(c0, i0)}
        out["agrees"] = {"cluster": C.reading(*ci) == C.reading(*c0), "wide": C.reading(*out["wide"]) == C.reading(*out["s0"]["wide"])}
    return out


def run(runs_root, B):
    D = Data(runs_root)
    boot = S.ClusterBoot(D.dbs, B=B)
    res = {"arms": {}, "contrasts": {}, "split": {}, "notes": {"n_qids": len(D.qids), "run_notes": {"%s|%s|%d" % k: v for k, v in sorted(D.note.items())}}}
    # ----------------------------------------------------------------------------- arms
    for b in C.BK:
        for arm in C.ARMS:
            ss = D.reps(b, arm)
            if not ss:
                continue
            ex = [100.0 * D.ex[(b, arm, s)].mean() for s in ss]
            e = {"per": dict(zip(ss, ex)), "mean": float(np.mean(ex)), "n_runs": len(ss), "range": (min(ex), max(ex)),
                 "s0": (100.0 * D.ex[(b, arm, 0)].mean()) if (b, arm, 0) in D.ex else None}
            if len(ss) >= 2:
                e["sd"] = float(np.std(ex, ddof=1))
                if len(ss) == 3:
                    e["sd_ci"] = sd_interval(e["sd"])
            res["arms"]["%s|%s" % (b, arm)] = e
    # ----------------------------------------------------------------------------- contrasts per backbone (runs present for both arms)
    per_b = {}
    for cid, xa, ya in CONTRASTS:
        for b in C.BK:
            ss = [s for s in C.SEEDS if (b, xa, s) in D.ex and (b, ya, s) in D.ex]
            if not [s for s in ss if s in C.REP]:
                continue
            dseed = {s: D.ex[(b, xa, s)] - D.ex[(b, ya, s)] for s in ss}
            per_b[(cid, b)] = (dseed, rep_avg(dseed))
            res["contrasts"]["%s|%s" % (cid, b)] = contrast_stats(dseed, D.dbs, boot, B)
    # ----------------------------------------------------------------------------- pooled over the three backbones (seeds present for all three)
    for cid in POOLED:
        if not all((cid, b) in per_b for b in C.BK):
            continue
        ss = sorted(set.intersection(*[set(per_b[(cid, b)][0]) for b in C.BK]))
        if not [s for s in ss if s in C.REP]:
            continue
        dseed = {s: np.mean([per_b[(cid, b)][0][s] for b in C.BK], axis=0) for s in ss}
        res["contrasts"]["%s|pooled3" % cid] = contrast_stats(dseed, D.dbs, boot, B)
    # W1: gated minus single call, mean of Flash and 27B
    if all(("g_minus_zs", b) in per_b for b in ("flash", "27b")):
        ss = sorted(set(per_b[("g_minus_zs", "flash")][0]) & set(per_b[("g_minus_zs", "27b")][0]))
        if [s for s in ss if s in C.REP]:
            dseed = {s: np.mean([per_b[("g_minus_zs", b)][0][s] for b in ("flash", "27b")], axis=0) for s in ss}
            res["contrasts"]["w1|weak2"] = contrast_stats(dseed, D.dbs, boot, B)
    # ----------------------------------------------------------------------------- P1' calls ratio: gated / fixed, sum of calls (per backbone) and mean of the three ratios (pooled)
    res["p1p_calls"] = calls_ratio(D, boot)
    # ----------------------------------------------------------------------------- stack versus loop
    for b in C.BK:
        if ("stack", b) in per_b and ("loop_g", b) in per_b:
            ss = sorted(set(per_b[("stack", b)][0]) & set(per_b[("loop_g", b)][0]))
            ds = {s: per_b[("stack", b)][0][s] - per_b[("loop_g", b)][0][s] for s in ss}
            st = contrast_stats(ds, D.dbs, boot, B)
            tot = res["contrasts"].get("g_minus_zs|%s" % b)
            sg = res["contrasts"].get("stack|%s" % b)
            lg = res["contrasts"].get("loop_g|%s" % b)
            sp = {"stack_minus_loop": st, "stack": sg["est"], "loop_g": lg["est"], "total_g": tot["est"] if tot else None}
            lo, hi = st["wide"]
            sp["verdict"] = "stack larger" if lo > 0 else ("loop larger" if hi < 0 else "not separable")
            sp["verdict_cluster"] = "stack larger" if st["ci"][0] > 0 else ("loop larger" if st["ci"][1] < 0 else "not separable")
            if tot is not None and (tot["ci"][0] > 0 or tot["ci"][1] < 0):
                sp["share_defined"] = True
                tvec = rep_avg(per_b[("g_minus_zs", b)][0])
                lvec = rep_avg(per_b[("loop_g", b)][0])
                tb, lb = boot.means(tvec), boot.means(lvec)
                sp["share_loop"] = float(lvec.mean() / tvec.mean())
                sp["share_loop_ci"] = S.pctl_ci(lb / np.where(tb == 0, np.nan, tb))
            else:
                sp["share_defined"] = False
                sp["share_loop"] = float(lg["est"] / tot["est"]) if tot and tot["est"] != 0 else None
            res["split"][b] = sp
    # ----------------------------------------------------------------------------- regeneration audit (root candidates)
    res["regen"] = regen_audit(D, runs_root, boot, B)
    return D, res


def calls_ratio(D, boot):
    out = {}
    for b in C.BK:
        ss = [s for s in C.REP if (b, "gated", s) in D.calls and (b, "fixed", s) in D.calls]
        if not ss:
            continue
        num = np.mean([D.calls[(b, "gated", s)] for s in ss], axis=0)
        den = np.mean([D.calls[(b, "fixed", s)] for s in ss], axis=0)
        per = {s: float(D.calls[(b, "gated", s)].sum() / D.calls[(b, "fixed", s)].sum()) for s in ss}
        s0 = float(D.calls[(b, "gated", 0)].sum() / D.calls[(b, "fixed", 0)].sum()) if (b, "gated", 0) in D.calls and (b, "fixed", 0) in D.calls else None
        out[b] = {"per": per, "est": float(num.sum() / den.sum()), "ci": S.pctl_ci(boot.ratio(num, den)), "item_ci": S.pctl_ci(S.item_boot_ratio(num, den, B=boot.B)),
                  "range": (min(per.values()), max(per.values())), "n_runs": len(ss), "num": num, "den": den, "seeds": ss, "s0": s0}
    if all(b in out for b in C.BK):
        ss = sorted(set.intersection(*[set(out[b]["seeds"]) for b in C.BK]))
        if ss:
            per = {s: float(np.mean([D.calls[(b, "gated", s)].sum() / D.calls[(b, "fixed", s)].sum() for b in C.BK])) for s in ss}
            nums = [np.mean([D.calls[(b, "gated", s)] for s in ss], axis=0) for b in C.BK]
            dens = [np.mean([D.calls[(b, "fixed", s)] for s in ss], axis=0) for b in C.BK]
            bt = np.mean([boot.ratio(n_, d_) for n_, d_ in zip(nums, dens)], axis=0)
            ib = np.mean([S.item_boot_ratio(n_, d_, B=boot.B) for n_, d_ in zip(nums, dens)], axis=0)
            s0p = float(np.mean([out[b]["s0"] for b in C.BK])) if all(out[b]["s0"] is not None for b in C.BK) else None
            out["pooled3"] = {"per": per, "est": float(np.mean([n_.sum() / d_.sum() for n_, d_ in zip(nums, dens)])), "ci": S.pctl_ci(bt), "item_ci": S.pctl_ci(ib),
                              "range": (min(per.values()), max(per.values())), "n_runs": len(ss), "seeds": ss, "s0": s0p}
    for v in out.values():
        v.pop("num", None)
        v.pop("den", None)
    return out


def regen_audit(D, runs_root, boot, B):
    out = {}
    pairs = (("gated", "fixed"), ("gated", "rootonly"), ("fixed", "rootonly"))
    for b in C.BK:
        for s in C.SEEDS:
            if not all((b, a, s) in D.ex for a in ("gated", "fixed", "rootonly")):
                continue
            roots = {}
            for a in ("gated", "fixed", "rootonly"):
                p = os.path.join(runs_root, C.run_path(b, a, s), "cands.jsonl")
                if not os.path.exists(p):
                    roots = None
                    break
                roots[a] = {r["qid"]: r for r in C.jl(p) if r.get("cand_idx") == 0}
            if roots is None:
                continue
            q_index = {q: i for i, q in enumerate(D.qids)}
            e = {"n": len(D.qids)}
            for x, y in pairs:
                diff = np.array([roots[x].get(q, {}).get("sql", "").strip() != roots[y].get(q, {}).get("sql", "").strip() for q in D.qids])
                e["diff.%s_%s" % (x, y)] = int(diff.sum())
                e["_mask.%s_%s" % (x, y)] = diff
            rex = {a: np.array([float(roots[a].get(q, {}).get("ex") or 0) for q in D.qids]) for a in roots}
            for x, y in pairs:
                e["rootex.%s_minus_%s" % (x, y)] = 100.0 * float((rex[x] - rex[y]).mean())
            # loop effects on questions whose root is identical in the two runs compared
            for cid, x, y in (("g_minus_f", "gated", "fixed"), ("loop_g", "gated", "rootonly"), ("loop_f", "fixed", "rootonly")):
                keep = ~e["_mask.%s_%s" % (x, y)]
                d = (D.ex[(b, x, s)] - D.ex[(b, y, s)])[keep]
                est, ci, ic = C.boot_ci(d, D.dbs[keep], B=B)
                e["shared.%s" % cid] = {"est": est, "ci": ci, "item_ci": ic, "n": int(keep.sum()),
                                        "all": 100.0 * float((D.ex[(b, x, s)] - D.ex[(b, y, s)]).mean())}
            out["%s|%d" % (b, s)] = {k: v for k, v in e.items() if not k.startswith("_mask")}
    return out


def write_outputs(D, res, outdir):
    os.makedirs(outdir, exist_ok=True)

    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    with open(os.path.join(outdir, "rep.json"), "w") as f:
        json.dump(conv(res), f, indent=1)
    f3 = lambda x: "" if x is None else "%.3f" % x
    with open(os.path.join(outdir, "rep_arms.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backbone", "arm", "n_runs", "ex_s0_T0", "ex_s101", "ex_s202", "ex_s303", "mean_T07", "sd_T07", "sd_lo", "sd_hi", "min", "max"])
        for k, e in res["arms"].items():
            b, a = k.split("|")
            sd_ci = e.get("sd_ci") or (None, None)
            w.writerow([b, a, e["n_runs"], f3(e.get("s0"))] + [f3(e["per"].get(s)) for s in C.REP] +
                       [f3(e["mean"]), f3(e.get("sd")), f3(sd_ci[0]), f3(sd_ci[1]), f3(e["range"][0]), f3(e["range"][1])])
    with open(os.path.join(outdir, "rep_contrasts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "scope", "n_runs", "est_s101", "est_s202", "est_s303", "mean_T07", "cl_lo", "cl_hi", "item_lo", "item_hi", "range_lo", "range_hi",
                    "t_lo", "t_hi", "wide_lo", "wide_hi", "reading_cluster", "reading_wide", "s0_est", "s0_cl_lo", "s0_cl_hi", "s0_reading_cluster", "s0_reading_wide",
                    "agrees_with_s0_cluster", "agrees_with_s0_wide"])
        for k, e in res["contrasts"].items():
            c, sc = k.split("|")
            t = e.get("t_ci") or (None, None)
            s0 = e.get("s0")
            ag = e.get("agrees")
            w.writerow([c, sc, e["n_runs"]] + [f3(e["per"].get(s)) for s in C.REP] +
                       [f3(e["est"]), f3(e["ci"][0]), f3(e["ci"][1]), f3(e["item_ci"][0]), f3(e["item_ci"][1]), f3(e["range"][0]), f3(e["range"][1]), f3(t[0]), f3(t[1]),
                        f3(e["wide"][0]), f3(e["wide"][1]), C.reading(*e["ci"]), C.reading(*e["wide"]),
                        f3(s0["est"]) if s0 else "", f3(s0["ci"][0]) if s0 else "", f3(s0["ci"][1]) if s0 else "",
                        C.reading(*s0["ci"]) if s0 else "", C.reading(*s0["wide"]) if s0 else "",
                        ("yes" if ag["cluster"] else "no") if ag else "", ("yes" if ag["wide"] else "no") if ag else ""])
    with open(os.path.join(outdir, "rep_regen.csv"), "w", newline="") as f:
        w = csv.writer(f)
        keys = ["diff.gated_fixed", "diff.gated_rootonly", "diff.fixed_rootonly", "rootex.gated_minus_fixed", "rootex.gated_minus_rootonly", "rootex.fixed_minus_rootonly"]
        w.writerow(["backbone", "seed", "n"] + keys + ["shared.g_minus_f(est,n,all)", "shared.loop_g(est,n,all)", "shared.loop_f(est,n,all)"])
        for k, e in res["regen"].items():
            b, s = k.split("|")
            w.writerow([b, s, e["n"]] + [("%.3f" % e[x]) if isinstance(e[x], float) else e[x] for x in keys] +
                       ["%.3f;%d;%.3f" % (e["shared.%s" % c]["est"], e["shared.%s" % c]["n"], e["shared.%s" % c]["all"]) for c in ("g_minus_f", "loop_g", "loop_f")])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=C.B_DEFAULT)
    a = ap.parse_args(argv)
    D, res = run(a.runs, a.B)
    write_outputs(D, res, a.out)
    print("[r3_seeds] qids %d; arms %d; contrasts %d; regen cells %d" % (len(D.qids), len(res["arms"]), len(res["contrasts"]), len(res["regen"])))
    for k, v in D.note.items():
        if v != "ok":
            print("[r3_seeds] run %s: %s" % (k, v))


if __name__ == "__main__":
    main()
