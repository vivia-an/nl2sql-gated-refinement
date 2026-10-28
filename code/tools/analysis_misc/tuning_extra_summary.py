#!/usr/bin/env python3
"""Tuning-set summaries that exist only inside results/raw-ignored (untracked) -> committed JSON (stdlib only, Python 3.6).

    python3 code/tools/analysis_misc/tuning_extra_summary.py

Writes (scope = "tuning-set"; NEVER confirmatory, never read by make_macros.py because they sit in subdirectories):
  results/x5/x5_tuning_summary.json   identifier-quoting check, gated vs fixed (HR-1101, HT-300), routing-score fit, arbitration probe, M3 forced routing
  results/x8/x8_pooled.json           per-backbone and pooled paired contrasts on HT-300 (flash, 27b, plus): gated vs zero-shot, gated vs fixed, calls ratio, root failures

Every number is read from the named raw files or computed from per-question rows with the stated rule; nothing is typed in.
Paired contrasts: percentage points; 95 % interval = percentile bootstrap over the 4 HT databases (B = 10000, seed 20261101); exact two-sided McNemar.
These are TUNING-SET numbers (BIRD train, 4 resp. 8 databases): design evidence, not results."""
import json
import math
import os
import random

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RAW = os.path.join(REPO, "results", "raw-ignored")
B, SEED = 10000, 20261101


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def js(path):
    return json.load(open(path))


def mcnemar_two(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.factorial(n) // (math.factorial(i) * math.factorial(n - i)) for i in range(k + 1)) / float(2 ** n)
    return min(1.0, 2 * p)


def pct(lo_hi_sorted, q):
    i = min(len(lo_hi_sorted) - 1, max(0, int(round(q * (len(lo_hi_sorted) - 1)))))
    return lo_hi_sorted[i]


def boot_ci(per_db, stat, rng):
    """per_db: {db: vector of numbers}; stat(list of per-db vectors) -> float; percentile 95 % over databases."""
    dbs = sorted(per_db)
    vals = []
    for _ in range(B):
        draw = [per_db[rng.choice(dbs)] for _ in dbs]
        vals.append(stat(draw))
    vals.sort()
    return pct(vals, 0.025), pct(vals, 0.975)


def load_preds(d):
    return {r["qid"]: r for r in jl(os.path.join(d, "preds.jsonl"))}


def load_roots(d):
    return {r["qid"]: r for r in jl(os.path.join(d, "cands.jsonl")) if r.get("action") == "root"}


def main():
    rng = random.Random(SEED)
    src = {"HT": {"flash": {"zs": "X8/x8-qwen37flash-zs", "fixed": "X8/x8-qwen37flash-fixed", "gated": "X8/x8-qwen37flash-gated"},
                  "b27": {"zs": "X8/x8-qwen3527b-zs", "fixed": "X8/x8-qwen3527b-fixed", "gated": "X8/x8-qwen3527b-gated"},
                  "plus": {"fixed": "X5/x5-1b-ht-quote", "gated": "X5/x5-1b-ht-gated"}}}
    P = {m: {s: load_preds(os.path.join(RAW, p)) for s, p in d.items()} for m, d in src["HT"].items()}
    R = {m: {s: load_roots(os.path.join(RAW, p)) for s, p in d.items() if s == "gated"} for m, d in src["HT"].items()}
    out = {"_meta": {"scope": "tuning-set", "split": "bird_train_HT (300 q, 4 dbs)", "script": "code/tools/analysis_misc/tuning_extra_summary.py",
                     "note": "design evidence only; bootstrap over 4 databases, B=10000, seed %d" % SEED}}

    def paired(a, b, ma, mb):
        """per-db vectors of (a_ex - b_ex) for questions in both."""
        qs = sorted(set(P[ma][a]) & set(P[mb][b]))
        per_db = {}
        n10 = n01 = 0
        for q in qs:
            x, y = int(P[ma][a][q]["ex"] or 0), int(P[mb][b][q]["ex"] or 0)
            per_db.setdefault(P[ma][a][q]["db_id"], []).append(x - y)
            n10 += int(x == 1 and y == 0)
            n01 += int(x == 0 and y == 1)
        return qs, per_db, n10, n01

    def put_ci(key, mean, lo, hi):
        out[key + ".mean"], out[key + ".ci_lo"], out[key + ".ci_hi"] = mean, lo, hi

    def contrast(key, a, b, ma, mb=None):
        mb = mb or ma
        qs, per_db, n10, n01 = paired(a, b, ma, mb)
        tot = sum(sum(v) for v in per_db.values())
        mean = 100.0 * tot / len(qs)
        lo, hi = boot_ci(per_db, lambda dr: 100.0 * sum(sum(v) for v in dr) / sum(len(v) for v in dr), rng)
        put_ci(key + ".delta_ex", mean, lo, hi)
        out[key + ".n"], out[key + ".n10"], out[key + ".n01"] = len(qs), n10, n01
        out[key + ".p"] = mcnemar_two(n10, n01)
        return per_db

    names = {"flash": "flash", "b27": "b27", "plus": "plus"}
    # per-backbone: ex, calls
    for m in ("flash", "b27", "plus"):
        for s in P[m]:
            rows = list(P[m][s].values())
            out["TS.x8.%s.%s.ex" % (m, s)] = sum(int(r["ex"] or 0) for r in rows) / float(len(rows))
            out["TS.x8.%s.%s.calls" % (m, s)] = sum(r["calls"] for r in rows) / float(len(rows))
        out["TS.x8.%s.n" % m] = len(P[m]["gated"])
        out["TS.x8.%s.calls_ratio" % m] = sum(r["calls"] for r in P[m]["gated"].values()) / float(sum(r["calls"] for r in P[m]["fixed"].values()))
    out["TS.x8.n_dbs"] = len({r["db_id"] for r in P["flash"]["gated"].values()})

    # gated vs zs (weak backbones) and pooled
    pd_gz = {}
    for m in ("flash", "b27"):
        pd_gz[m] = contrast("TS.x8.%s.gated_vs_zs" % m, "gated", "zs", m)
    # gated vs fixed (3 backbones) and pooled
    pd_gf = {}
    for m in ("flash", "b27", "plus"):
        pd_gf[m] = contrast("TS.x8.%s.gated_vs_fixed" % m, "gated", "fixed", m)

    def pooled(key, pds):
        dbs = sorted(set.intersection(*[set(p) for p in pds.values()]))

        def stat(draw_dbs):
            # draw_dbs: list of db names drawn (we resample names, then average each backbone's db-pooled difference, then average backbones)
            ms = []
            for p in pds.values():
                tot = sum(sum(p[d]) for d in draw_dbs)
                n = sum(len(p[d]) for d in draw_dbs)
                ms.append(100.0 * tot / n)
            return sum(ms) / len(ms)
        mean = stat(dbs)
        vals = sorted(stat([rng.choice(dbs) for _ in dbs]) for _ in range(B))
        put_ci(key + ".delta_ex", mean, pct(vals, 0.025), pct(vals, 0.975))
        out[key + ".n_backbones"] = len(pds)
        out[key + ".all_same_sign"] = all((sum(sum(v) for v in p.values()) > 0) for p in pds.values())

    pooled("TS.x8.w1_pooled", pd_gz)
    pooled("TS.x8.p1_pooled", pd_gf)
    # pooled calls ratio (ratio of sums per backbone, averaged) with bootstrap over dbs
    def ratio_stat(draw):
        rs = []
        for m in ("flash", "b27", "plus"):
            g = sum(sum(P[m]["gated"][q]["calls"] for q in dbq[m]["g"][d]) for d in draw)
            f = sum(sum(P[m]["fixed"][q]["calls"] for q in dbq[m]["g"][d]) for d in draw)
            rs.append(g / float(f))
        return sum(rs) / len(rs)
    dbq = {}
    for m in ("flash", "b27", "plus"):
        d = {}
        for q, r in P[m]["gated"].items():
            if q in P[m]["fixed"]:
                d.setdefault(r["db_id"], []).append(q)
        dbq[m] = {"g": d}
    dbs_all = sorted(dbq["plus"]["g"])
    vals = sorted(ratio_stat([rng.choice(dbs_all) for _ in dbs_all]) for _ in range(B))
    put_ci("TS.x8.p1_pooled.calls_ratio", ratio_stat(dbs_all), pct(vals, 0.025), pct(vals, 0.975))

    # root failures and repair (gated runs)
    tot_fail = tot_fix = tot_exe = 0
    for m in ("flash", "b27", "plus"):
        roots = R[m]["gated"]
        fail = [q for q, r in roots.items() if str(r.get("status", "")).startswith("err") or r.get("status") == "timeout"]
        fixed = sum(int(P[m]["gated"][q]["ex"] or 0) for q in fail)
        exe = sum(1 for q in fail if P[m]["gated"][q].get("exec_ok"))
        out["TS.x8.%s.n_root_fail" % m], out["TS.x8.%s.n_repaired" % m], out["TS.x8.%s.n_executable" % m] = len(fail), fixed, exe
        tot_fail, tot_fix, tot_exe = tot_fail + len(fail), tot_fix + fixed, tot_exe + exe
    out["TS.x8.pooled.n_root_fail"], out["TS.x8.pooled.n_repaired"], out["TS.x8.pooled.n_executable"] = tot_fail, tot_fix, tot_exe
    out["TS.x8.pooled.repair_rate"] = tot_fix / float(tot_fail) if tot_fail else None
    out["TS.x8.pooled.executable_rate"] = tot_exe / float(tot_fail) if tot_fail else None
    # root-only arm (the root candidate of the gated run, i.e. the pipeline without any revision) vs zero-shot and vs the gated answer
    def root_contrast(key, ma_roots, other, label):
        per_db, n10, n01 = {}, 0, 0
        for q, r in ma_roots.items():
            if q not in other:
                continue
            x, y = int(r.get("ex") or 0), int(other[q])
            per_db.setdefault(r["db_id"], []).append(x - y)
            n10, n01 = n10 + int(x == 1 and y == 0), n01 + int(x == 0 and y == 1)
        n = sum(len(v) for v in per_db.values())
        mean = 100.0 * sum(sum(v) for v in per_db.values()) / n
        lo, hi = boot_ci(per_db, lambda dr: 100.0 * sum(sum(v) for v in dr) / sum(len(v) for v in dr), rng)
        put_ci(key + ".delta_ex", mean, lo, hi)
        out[key + ".n"], out[key + ".n10"], out[key + ".n01"] = n, n10, n01
    for m in ("flash", "b27", "plus"):
        roots = R[m]["gated"]
        out["TS.x8.%s.rootonly.ex" % m] = sum(int(r.get("ex") or 0) for r in roots.values()) / float(len(roots))
        root_contrast("TS.x8.%s.gated_vs_rootonly" % m, {q: dict(r, ex=P[m]["gated"][q]["ex"]) for q, r in roots.items()},
                      {q: int(r.get("ex") or 0) for q, r in roots.items()}, "gated vs root")
        if m in ("flash", "b27"):
            root_contrast("TS.x8.%s.rootonly_vs_zs" % m, roots, {q: int(r["ex"] or 0) for q, r in P[m]["zs"].items()}, "root vs zs")
    json.dump(out, open(os.path.join(REPO, "results", "x8", "x8_pooled.json"), "w"), indent=1, sort_keys=True)

    # ---------------------------------------------------------------- X5
    x5 = {"_meta": {"scope": "tuning-set", "sources": "results/raw-ignored/X5/x5-cmp-4/*.json, x5-2-fit/route_coef_hr.json, x5-3-probe-*/probe_*.json",
                    "script": "code/tools/analysis_misc/tuning_extra_summary.py"}}
    c4 = os.path.join(RAW, "X5", "x5-cmp-4")
    bf = js(os.path.join(c4, "1a_beer_factory.json"))
    x5["TS.quote.n"] = bf["n_pairs"]
    for tag, k in (("legacy", "legacy_ddl"), ("quoted", "quoted_ddl")):
        x5["TS.quote.%s.ex_root" % tag] = bf[k]["root"]["ex"]
        x5["TS.quote.%s.fail_rate" % tag] = bf[k]["root"]["exec_error_or_timeout"]
        x5["TS.quote.%s.ex" % tag] = bf[k]["ex"]
    for tag, f in (("hr", "1b_hr.json"), ("ht", "1b_ht.json")):
        d = js(os.path.join(c4, f))
        x5["TS.%s.n" % tag] = d["n_pairs"]
        x5["TS.%s.fixed.ex" % tag], x5["TS.%s.gated.ex" % tag] = d["p0_quoted"]["ex"], d["gated"]["ex"]
        x5["TS.%s.fixed.calls" % tag], x5["TS.%s.gated.calls" % tag] = d["p0_quoted"]["calls_per_q"], d["gated"]["calls_per_q"]
        x5["TS.%s.calls_ratio" % tag] = d["gated"]["calls_per_q"] / d["p0_quoted"]["calls_per_q"]
        x5["TS.%s.gated_vs_fixed.delta_ex.mean" % tag] = d["diff_pp"]["b_minus_a"]
        x5["TS.%s.gated_vs_fixed.delta_ex.ci_lo" % tag], x5["TS.%s.gated_vs_fixed.delta_ex.ci_hi" % tag] = d["diff_pp"]["ci95"]
        x5["TS.%s.gated_vs_fixed.n10" % tag], x5["TS.%s.gated_vs_fixed.n01" % tag] = d["diff_pp"]["b_right_a_wrong"], d["diff_pp"]["a_right_b_wrong"]
        x5["TS.%s.gated_vs_fixed.n_dbs" % tag] = d["diff_pp"]["n_databases"]
        x5["TS.%s.root.fail_rate" % tag] = d["p0_quoted"]["root"]["exec_error_or_timeout"]
    m3 = js(os.path.join(c4, "4_m3_ht.json"))
    x5["TS.m3.n"] = m3["n_pairs"]
    x5["TS.m3.default.ex"], x5["TS.m3.forced.ex"] = m3["fallback6000"]["ex"], m3["fallback0"]["ex"]
    x5["TS.m3.forced.delta_ex.mean"] = m3["diff_pp"]["b_minus_a"]
    x5["TS.m3.forced.delta_ex.ci_lo"], x5["TS.m3.forced.delta_ex.ci_hi"] = m3["diff_pp"]["ci95"]
    rf = js(os.path.join(RAW, "X5", "x5-2-fit", "route_coef_hr.json"))
    for tag, blk, lrk in (("hr", rf["lodo"], "lr_lodo(v3+consistency+nonempty)"), ("ht", rf["heldout"]["metrics"], "lr_fitted_on_train")):
        v3, lr = blk["v3"], blk[lrk]
        x5["TS.route.%s.v3.auroc" % tag], x5["TS.route.%s.ca.auroc" % tag] = v3["auroc"], lr["auroc"]
        x5["TS.route.%s.v3.aurc" % tag], x5["TS.route.%s.ca.aurc" % tag] = v3["aurc"], lr["aurc"]
        x5["TS.route.%s.v3.acc_ex_cov60" % tag], x5["TS.route.%s.ca.acc_ex_cov60" % tag] = v3["ex_at_coverage"]["0.6"], lr["ex_at_coverage"]["0.6"]
        g = lr["auroc_gain_vs_v3"]
        x5["TS.route.%s.dauroc.mean" % tag], (x5["TS.route.%s.dauroc.ci_lo" % tag], x5["TS.route.%s.dauroc.ci_hi" % tag]) = g["gain"], g["ci"]
    x5["TS.route.hr.n"], x5["TS.route.ht.n"] = rf["fit"]["n"], rf["heldout"]["n"]
    for tag, f in (("revise", "x5-3-probe-revise/probe_revise.json"), ("pool", "x5-3-probe-pool/probe_pool.json")):
        s = js(os.path.join(RAW, "X5", f))["summary"]
        x5["TS.arb.%s.n_mixed" % tag] = s["n_mixed"]
        x5["TS.arb.%s.judge_acc" % tag], x5["TS.arb.%s.revise_always_acc" % tag] = s["judge_acc_on_mixed"], s["revise_always_acc_on_mixed"]
        x5["TS.arb.%s.delta_ex.mean" % tag] = s["judged_minus_root_pp"]["diff"]
        x5["TS.arb.%s.delta_ex.ci_lo" % tag], x5["TS.arb.%s.delta_ex.ci_hi" % tag] = s["judged_minus_root_pp"]["ci"]
    # EX by status of the root candidate (HR, strong backbone, quoting on): fixed loop vs gated loop
    def status_group(st):
        return "err" if (str(st).startswith("err") or st == "timeout") else ("empty" if st == "ok_empty" else "nonempty")
    for lab, d in (("full_fixed", "x5-1b-hr-quote"), ("full_gated", "x5-1b-hr-gated")):
        pr, rt = load_preds(os.path.join(RAW, "X5", d)), load_roots(os.path.join(RAW, "X5", d))
        for g in ("err", "empty", "nonempty"):
            qs = [q for q, r in rt.items() if q in pr and status_group(r.get("status")) == g]
            n = len(qs)
            x5["TS.rs.%s.%s.n" % (lab, g)] = n
            if n:
                x5["TS.rs.%s.%s.ex_root" % (lab, g)] = sum(int(rt[q].get("ex") or 0) for q in qs) / float(n)
                x5["TS.rs.%s.%s.ex_final" % (lab, g)] = sum(int(pr[q]["ex"] or 0) for q in qs) / float(n)
                x5["TS.rs.%s.%s.gain" % (lab, g)] = sum(1 for q in qs if not int(rt[q].get("ex") or 0) and int(pr[q]["ex"] or 0))
                x5["TS.rs.%s.%s.harm" % (lab, g)] = sum(1 for q in qs if int(rt[q].get("ex") or 0) and not int(pr[q]["ex"] or 0))
    os.makedirs(os.path.join(REPO, "results", "x5"), exist_ok=True)
    json.dump(x5, open(os.path.join(REPO, "results", "x5", "x5_tuning_summary.json"), "w"), indent=1, sort_keys=True)
    print("wrote results/x5/x5_tuning_summary.json (%d keys), results/x8/x8_pooled.json (%d keys)" % (len(x5), len(out)))


if __name__ == "__main__":
    main()
