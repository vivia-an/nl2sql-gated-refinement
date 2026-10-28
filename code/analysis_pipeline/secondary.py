"""Secondary analyses: E1 (main table, S-A, per-db, root status), E11 (cost), E2 (per backbone), E3 (ablations).

Flat keys: results/E1.json ... Each function takes the run Registry and a Result and only adds keys for runs that exist.
Percentages: levels are fractions in [0, 1]; differences are in percentage points (names say `delta_ex`, `delta_pp`)."""
import os

import numpy as np

from . import runs as R, stats as S
from .util import Result

THR = "thrombosis_prediction"
E1_ORDER = ["zs", "dail", "sdebug", "macsql", "acsql_s", "acsql_x", "memosql", "alphasql", "bon_vote", "sc_cot", "bon_critic",
            "full_fixed", "full_gated", "full_gated_route", "full_rl", "full_x"]


# ------------------------------------------------------------------------------------------ prices and cost
def load_prices(path):
    if not path or not os.path.exists(path):
        return None
    d = R.read_yaml(path) or {}
    return d


def _price(prices, model):
    return (prices or {}).get("models", {}).get(model)


def run_cost(run, prices):
    """-> dict(total_cny, per_q, source, note) priced with list prices (exp/prices.yaml). Cache-independent: every chat row of usage*.jsonl counts at its
    recorded token numbers whether or not it was a cache hit. Embedding / rerank are priced separately (prices UNVERIFIED) and excluded from `total_cny`."""
    if not prices:
        return None
    ratio = float(prices.get("cache_hit_ratio", 0.2))
    bratio = float(prices.get("batch_ratio", 0.5))
    chat = emb = 0.0
    unpriced = 0
    files = run.usage_files()
    if files:
        for f in files:
            for r in R.jl(f):
                pr = _price(prices, r.get("model"))
                if pr is None:
                    unpriced += 1
                    continue
                pt, ct = r.get("pt", 0) or 0, r.get("ct", 0) or 0
                if r.get("kind") in ("embed", "rerank"):
                    emb += pt * pr["in"] / 1e6
                    continue
                cached = min(r.get("cached_tokens", 0) or 0, pt)
                y = ((pt - cached) * pr["in"] + cached * pr["in"] * ratio + ct * pr["out"]) / 1e6
                chat += y * (bratio if r.get("mode") == "batch" else 1.0)
        src = "usage"
    else:
        for r in run.rows.values():
            pr = _price(prices, r.get("model"))
            if pr is None:
                unpriced += 1
                continue
            pt, ct = r.get("prompt_tokens", 0) or 0, r.get("completion_tokens", 0) or 0
            cached = min(r.get("cached_tokens", 0) or 0, pt)
            chat += ((pt - cached) * pr["in"] + cached * pr["in"] * ratio + ct * pr["out"]) / 1e6
        src = "preds"
    n = max(len(run.rows), 1)
    return {"total_cny": chat, "per_q": chat / n, "embed_rerank_cny": emb, "source": src, "unpriced": unpriced}


# ------------------------------------------------------------------------------------------ per-run summary
def ves_values(run):
    p = os.path.join(run.path, "ves.jsonl")
    if not os.path.exists(p):
        return None
    return R.jl(p)


STAGES = ("m1", "gen", "crit", "rev")


def summarize(run, B, prices=None):
    """Own-row summary of one run with db-clustered CIs (own boot per run, shared seed)."""
    rows = [run.rows[q] for q in sorted(run.rows)]
    ex = [R.ex_value(r) for r in rows]
    keep = [i for i, v in enumerate(ex) if v is not None]
    rows = [rows[i] for i in keep]
    ex = np.array([ex[i] for i in keep], int)
    dbs = np.array([r.get("db_id") for r in rows])
    boot = S.ClusterBoot(dbs, B)
    out = {"n": len(rows), "n_dropped_ex_null": len(keep) != len(run.rows) and (len(run.rows) - len(keep)) or 0}
    ci = lambda v: S.pctl_ci(boot.means(np.asarray(v, float)))
    out["ex"] = (float(ex.mean()),) + ci(ex)
    for k in ("soft_f1", "jaccard"):
        v = R.col(rows, k, np.nan)
        if np.isfinite(v).any():
            vv = np.where(np.isfinite(v), v, 0.0)
            out[k] = (float(vv.mean()),) + ci(vv)
    acc = np.array([0 if r.get("abstained") else 1 for r in rows])
    out["coverage"] = float(acc.mean())
    out["ex_accepted"] = float(ex[acc == 1].mean()) if acc.sum() else None
    execok = np.array([1 if r.get("exec_ok") else 0 for r in rows])
    silent = ((execok == 1) & (ex == 0)).astype(float)
    out["silent_error"] = (float(silent.mean()),) + ci(silent)
    out["silent_share_of_errors"] = float(silent.sum() / max((ex == 0).sum(), 1))
    out["exec_ok"] = float(execok.mean())
    out["abstained_wrong_ex"] = float((ex * acc).mean())                        # EX_all with abstentions counted wrong
    calls = R.col(rows, "calls")
    pt, ct = R.col(rows, "prompt_tokens"), R.col(rows, "completion_tokens")
    lat = R.col(rows, "latency_s")
    out["calls"] = float(calls.mean())
    for st in STAGES:                                                           # answer-path calls per question by pipeline stage (figure F3a)
        out["calls_" + st] = float(np.mean([((r.get("stages") or {}).get(st) or {}).get("calls", 0) or 0 for r in rows]))
    out["prompt_tokens"] = float(pt.mean()); out["completion_tokens"] = float(ct.mean()); out["tokens"] = float((pt + ct).mean())
    out["latency_mean"] = float(lat.mean()); out["latency_p50"] = float(np.percentile(lat, 50)); out["latency_p95"] = float(np.percentile(lat, 95))
    out["tokens_per_correct"] = float((pt + ct).sum() / max(ex.sum(), 1))
    out["api_fail"] = int(sum(1 for r in rows if r.get("error_class") == "api_fail"))
    out["parse_fail"] = int(sum(1 for r in rows if r.get("error_class") == "parse_fail"))
    out["by_db"] = {d: (float(ex[dbs == d].mean()), int((dbs == d).sum())) for d in sorted(set(dbs.tolist()))}
    vr = ves_values(run)
    if vr:
        for key in ("ves", "rves"):
            vals = [(r.get(key), r.get("db_id")) for r in vr if r.get(key) is not None]
            if vals:
                b2 = S.ClusterBoot([d for _, d in vals], B)
                v = np.array([x for x, _ in vals], float)
                out[key] = (float(v.mean()),) + S.pctl_ci(b2.means(v))
        out["ves_coverage"] = len(vr) / float(max(len(run.rows), 1))
    c = run_cost(run, prices)
    if c:
        out["cost_per_q"] = c["per_q"]; out["cost_total"] = c["total_cny"]; out["cost_source"] = c["source"]; out["cost_per_correct"] = c["total_cny"] / max(ex.sum(), 1)
        out["embed_rerank_cny"] = c["embed_rerank_cny"]
    return out


def put_summary(res, pre, s):
    for k in ("ex", "soft_f1", "jaccard", "silent_error", "ves", "rves"):
        if k in s:
            res.ci("%s.%s" % (pre, k), *s[k])
    for k in ("n", "coverage", "ex_accepted", "silent_share_of_errors", "exec_ok", "abstained_wrong_ex", "calls", "calls_m1", "calls_gen", "calls_crit", "calls_rev", "prompt_tokens", "completion_tokens", "tokens", "latency_mean", "latency_p50",
              "latency_p95", "tokens_per_correct", "api_fail", "parse_fail", "cost_per_q", "cost_total", "cost_per_correct", "embed_rerank_cny", "ves_coverage"):
        if s.get(k) is not None:
            res.set("%s.%s.mean" % (pre, k) if k not in ("n", "api_fail", "parse_fail") else "%s.%s" % (pre, k), s[k])


# ------------------------------------------------------------------------------------------ pairwise helpers
def pair_block(res, pre, ex_a, ex_b, dbs, B, boot=None, two_sided=True):
    """ΔEX = a - b (pp) with db-clustered CI, exact McNemar, conditional OR. Returns dict with p."""
    ex_a, ex_b = np.asarray(ex_a, int), np.asarray(ex_b, int)
    boot = boot or S.ClusterBoot(dbs, B)
    d = ex_a - ex_b
    bt = 100.0 * (boot.means(ex_a) - boot.means(ex_b))
    lo, hi = S.pctl_ci(bt)
    b = int(((ex_a == 1) & (ex_b == 0)).sum())
    c = int(((ex_a == 0) & (ex_b == 1)).sum())
    mc = S.mcnemar_exact(b, c)
    ilo, ihi = S.pctl_ci(100.0 * S.item_boot_means(d, min(B, 5000)))
    res.ci(pre + ".delta_ex", 100.0 * d.mean(), lo, hi)
    res.set(pre + ".delta_ex.item_ci_lo", ilo); res.set(pre + ".delta_ex.item_ci_hi", ihi)
    res.set(pre + ".p", mc["p_two"]); res.set(pre + ".p_mid", mc["p_mid_two"]); res.set(pre + ".b", b); res.set(pre + ".c", c)
    res.set(pre + ".or", mc["or"]); res.set(pre + ".or_lo", mc["or_lo"]); res.set(pre + ".or_hi", mc["or_hi"])
    res.set(pre + ".n", int(len(d)))
    return {"delta": 100.0 * d.mean(), "p": mc["p_two"], "ci": (lo, hi), "b": b, "c": c}


def _s(x):
    return "%.1f" % x


# ------------------------------------------------------------------------------------------ E1
def _failed_run(r):
    """True when no row of the run has a final SQL and no row made a model call (e.g. the Memo-SQL M2 run: 500 rows, `error_class: system_error`, `pred_error: empty sql`)."""
    rows = list(r.rows.values())
    return bool(rows) and not any((x.get("final_sql") or "").strip() for x in rows) and not any((x.get("calls") or 0) for x in rows)


def run_e1(reg, res, B, prices, decisions=None, bstar=None):
    decisions = decisions or {}
    model = reg.primary
    full_label = decisions.get("full") or "full_gated"
    runs = {}
    for lab in E1_ORDER:
        try:
            r = reg.get(lab)
        except R.RunError as e:
            res.note("E1: %s" % e)
            r = None
        if r is not None:
            runs[lab] = r
    for lab, r in reg.baselines().items():                       # variants at several budgets (system@cfg)
        if lab not in runs:
            runs[lab] = r
    for lab in [l for l, r in runs.items() if _failed_run(r)]:        # a run in which no row produced any SQL is a technical failure (SAP 13.6: repeated, not scored as 0 % EX)
        res.set("E1.%s.%s.status" % (lab, model), "n/a: all %d rows failed without a model call (empty SQL, system_error); the run must be repeated" % runs[lab].n_rows)
        res.note("E1: run %s has no prediction in any row; excluded from E1 / E11 and from the S-A family" % runs[lab].run_id)
        del runs[lab]
    if not runs:
        res.set("E1.status", "not run: no complete r0 run on %s" % reg.split)
        return {}
    res.set("E1.status", "ok")
    res.set("E1.systems", ", ".join(runs))
    summ = {}
    for lab, r in runs.items():
        s = summarize(r, B, prices)
        summ[lab] = s
        put_summary(res, "E1.%s.%s" % (lab, model), s)
        if r.subset:
            res.set("E1.%s.subset_n" % lab, r.n_rows)                  # subset baseline: its EX is on its own items; comparisons with `full` use the common items
        for d, (v, n) in s["by_db"].items():
            res.set("E1.perdb.%s.%s.ex" % (lab, d), v); res.set("E1.perdb.%s.%s.n" % (lab, d), n)
        # stability over r0, r1, r2
        reps = [reg.get(lab, rep=k) for k in (1, 2)] if lab in R.LABELS else []
        reps = [x for x in reps if x is not None]
        if reps:
            vals = [s["ex"][0]] + [float(np.mean([R.ex_value(x_) for x_ in rr.rows.values() if R.ex_value(x_) is not None])) for rr in reps]
            res.set("E1.%s.%s.ex_rep.mean" % (lab, model), float(np.mean(vals)))
            res.set("E1.%s.%s.ex_rep.sd" % (lab, model), float(np.std(vals, ddof=1)) if len(vals) > 1 else None)
            res.set("E1.%s.%s.ex_rep.n" % (lab, model), len(vals))
    # ---- comparisons: vs zs (every system), full vs everyone (S-A, Holm), full vs best baseline, vs B*
    full = runs.get(full_label)
    if "zs" in runs:
        for lab in runs:
            if lab == "zs":
                continue
            A = R.align({"a": runs[lab], "b": runs["zs"]})
            if len(A["qids"]):
                pair_block(res, "E1.vs_zs.%s.%s" % (lab, model), A["ex"]["a"], A["ex"]["b"], A["dbs"], B)
    if full is not None:
        res.set("E1.full_label", full_label)
        others = [l for l in runs if l != full_label and l != "full_gated_route"]
        pv = {}
        for lab in others:
            A = R.align({"a": full, "b": runs[lab]})
            if not len(A["qids"]):
                continue
            o = pair_block(res, "E1.full_vs.%s" % lab, A["ex"]["a"], A["ex"]["b"], A["dbs"], B)
            pv[lab] = o["p"]
        m = len(pv) + (0 if full_label == "full_gated" and "full_fixed" in pv else 0)
        adj = S.holm(pv, max(m, len(pv)))
        for lab, p in adj.items():
            res.set("E1.full_vs.%s.p_adj" % lab, p)
        res.set("E1.full_vs.family_m", len(pv))
        base_labs = [l for l in runs if R.SYSTEM_TO_BASELINE.get(l.split("@")[0]) or l.split("@")[0] in R.BASELINE_SYSTEMS]
        if base_labs:
            best = sorted(base_labs, key=lambda l: (-summ[l]["ex"][0], summ[l]["calls"]))[0]
            res.set("E1.full_vs_best.label", best)
            A = R.align({"a": full, "b": runs[best]})
            o = pair_block(res, "E1.full_vs_best", A["ex"]["a"], A["ex"]["b"], A["dbs"], B)
            res.set("E1.full_vs_best.p_adj", adj.get(best))
        if bstar and bstar in runs:
            A = R.align({"a": full, "b": runs[bstar]})
            pair_block(res, "E1.vs_bstar.%s" % full_label, A["ex"]["a"], A["ex"]["b"], A["dbs"], B)
            res.set("E1.bstar", bstar)
        # per-db deltas and leave-one-db-out (plot input)
        for other in [x for x in ("zs", "full_fixed", bstar) if x and x in runs and x != full_label]:
            A = R.align({"a": full, "b": runs[other]})
            if not len(A["qids"]):
                continue
            d = (A["ex"]["a"] - A["ex"]["b"]).astype(float)
            pre = "E1.perdb_delta.%s_vs_%s" % (full_label, other)
            boot = S.ClusterBoot(A["dbs"], B)
            for j, dbn in enumerate(boot.u):
                idx = boot.members[j]
                bt = 100.0 * S.item_boot_means(d[idx], 2000)
                lo, hi = S.pctl_ci(bt)
                res.set("%s.%s.delta_pp" % (pre, dbn), 100.0 * d[idx].mean()); res.set("%s.%s.ci_lo" % (pre, dbn), lo); res.set("%s.%s.ci_hi" % (pre, dbn), hi)
                res.set("%s.%s.n" % (pre, dbn), int(len(idx)))
            for dbn, v in boot.loo(lambda idx: 100.0 * d[idx].mean()):
                res.set("%s.%s.loo_delta_pp" % (pre, dbn), v)
            res.set(pre + ".overall_pp", 100.0 * d.mean())
    # ---- root status table: EX by root status for the loop systems; gain / harm vs the root
    for lab in ("full_fixed", "full_gated"):
        r = runs.get(lab)
        if r is None or not r.cands:
            continue
        roots = r.roots()
        st_map = {"err": ("err_syntax", "err_identifier", "timeout"), "empty": ("ok_empty",), "nonempty": ("ok_nonempty",)}
        for nm, sts in st_map.items():
            items = [(q, roots[q]) for q in roots if q in r.rows and roots[q].get("status") in sts and R.ex_value(r.rows[q]) is not None and roots[q].get("ex") is not None]
            pre = "E1.rootstatus.%s.%s" % (lab, nm)
            res.set(pre + ".n", len(items))
            if items:
                fin = np.array([R.ex_value(r.rows[q]) for q, _ in items])
                rt = np.array([int(c["ex"]) for _, c in items])
                res.set(pre + ".ex_root", float(rt.mean())); res.set(pre + ".ex_final", float(fin.mean()))
                res.set(pre + ".gain", int(((fin == 1) & (rt == 0)).sum())); res.set(pre + ".harm", int(((fin == 0) & (rt == 1)).sum()))
                res.set(pre + ".revised_share", float(np.mean([1.0 if (r.rows[q].get("iterations") or 0) > 0 else 0.0 for q, _ in items])))
    return runs


# ------------------------------------------------------------------------------------------ E11
def run_e11(reg, res, B, prices, runs=None):
    runs = runs or {}
    model = reg.primary
    for lab, r in runs.items():
        s = summarize(r, min(B, 1000), prices)
        pre = "E11.%s.%s" % (lab, model)
        for k in ("calls", "prompt_tokens", "completion_tokens", "tokens", "latency_p50", "latency_p95", "latency_mean", "tokens_per_correct", "cost_per_q", "cost_per_correct"):
            if s.get(k) is not None:
                res.set("%s.%s.mean" % (pre, k), s[k])
        if s.get("embed_rerank_cny") is not None:
            res.set(pre + ".embed_rerank_cny.mean", s["embed_rerank_cny"])
    if runs:
        res.set("E11.status", "ok")
        res.note("E11 cost = chat tokens x list price (exp/prices.yaml, DashScope catalogue, read 2026-10-26); embedding/rerank priced separately with UNVERIFIED unit prices and excluded from cost_per_q")
    else:
        res.set("E11.status", "not run")


# ------------------------------------------------------------------------------------------ E2
def run_e2(reg, res, B, labels=("zs", "dail", "full_fixed", "full_gated"), prices=None):
    models = sorted(set(r.modelkey for r in reg.all if r.split == reg.split))
    gains, zsex, ps, used = [], [], {}, []
    for mk in models:
        cur = {}
        for lab in labels:
            try:
                r = reg.get(lab, model=mk)
            except R.RunError:
                r = None
            if r is not None:
                cur[lab] = r
                s = summarize(r, min(B, 2000), prices)
                put_summary(res, "E2.%s.%s" % (mk, lab), s)
        if "zs" in cur and "full_gated" in cur:
            A = R.align({"a": cur["full_gated"], "b": cur["zs"]})
            o = pair_block(res, "E2.%s.gain_gated_vs_zs" % mk, A["ex"]["a"], A["ex"]["b"], A["dbs"], min(B, 5000))
            gains.append(o["delta"]); zsex.append(float(A["ex"]["b"].mean())); ps[mk] = o["p"]; used.append(mk)
    if used:
        res.set("E2.status", "ok"); res.set("E2.models", ", ".join(used)); res.set("E2.n_models", len(used))
        q = S.benjamini_hochberg(ps)
        for mk in used:
            res.set("E2.%s.gain_q" % mk, q[mk])
        rho = S.spearman(zsex, gains)
        res.set("E2.gain.rank_corr", rho)
        res.set("E2.gain_vs_zs_ex.spearman", rho)
        res.note("E2 Spearman between the gain of full_gated over zs and zs EX across %d models is descriptive (n small); BH q = 0.10 within the exercise" % len(used))
    else:
        res.set("E2.status", "not run: need zs and full_gated r0 for at least one model")


# ------------------------------------------------------------------------------------------ E3
E3_LABELS = ["abl_m1", "abl_m2", "abl_m3", "abl_m3force", "abl_m5", "abl_m6", "abl_q", "full_fixed", "full_gated_blind"]


def run_e3(reg, res, B, full_label="full_gated"):
    base = reg.get(full_label)
    if base is None:
        res.set("E3.status", "not run: %s r0 missing" % full_label)
        return
    runs = {full_label: base}
    for lab in E3_LABELS:
        try:
            r = reg.get(lab)
        except R.RunError as e:
            res.note("E3: %s" % e)
            r = None
        if r is not None:
            runs[lab] = r
    if len(runs) < 2:
        res.set("E3.status", "not run: no ablation run complete")
        return
    A = R.align(runs)
    res.set("E3.status", "ok"); res.set("E3.configs", ", ".join(runs)); res.set("E3.n", len(A["qids"]))
    M = np.column_stack([A["ex"][l] for l in runs])
    cq = S.cochran_q(M)
    res.set("E3.cochran.q", cq["q"]); res.set("E3.cochran.df", cq["df"]); res.set("E3.cochran.p", cq["p"]); res.set("E3.cochran.k", len(runs))
    pv = {}
    boot = S.ClusterBoot(A["dbs"], B)
    for lab in runs:
        res.ci("E3.%s.ex" % lab, float(A["ex"][lab].mean()), *S.pctl_ci(boot.means(A["ex"][lab])))
        if lab == full_label:
            continue
        o = pair_block(res, "E3.%s.delta_vs_full" % lab, A["ex"][lab], A["ex"][full_label], A["dbs"], B, boot)
        pv[lab] = o["p"]
    adj = S.holm(pv, len(pv))
    for lab, p in adj.items():
        res.set("E3.%s.delta_vs_full.p_adj" % lab, p)
    if "full_fixed" in runs:                                   # semantic revision on top of the gate: fixed - gated
        pair_block(res, "E3.sem_revision.delta_pair", A["ex"]["full_fixed"], A["ex"][full_label], A["dbs"], B, boot)
        d = 100.0 * (boot.means(A["ex"]["full_fixed"]) - boot.means(A["ex"][full_label]))
        lo, hi = S.pctl_ci(d)
        res.ci("E3.sem_revision.delta", float(100.0 * (A["ex"]["full_fixed"].mean() - A["ex"][full_label].mean())), lo, hi)
    # error classes re-introduced by an ablation: items right in full, wrong in the ablation, by auto_class of the ablation's output
    for lab in runs:
        if lab == full_label:
            continue
        cnt = {}
        for i in range(len(A["qids"])):
            if A["ex"][full_label][i] == 1 and A["ex"][lab][i] == 0:
                c = A["rows"][lab][i].get("auto_class") or "other"
                cnt[c] = cnt.get(c, 0) + 1
        for c, v in cnt.items():
            res.set("E3.%s.reintroduced.%s" % (lab, c), v)


def run_e3_backbones(reg, res, B):
    """SAP 15.3 W1 reading rule: the root-only arm (`gated_rootonly`, M3 arm -M5) separates the pipeline's retrieval stack from the loop, per backbone.
    E3.<b>.rootonly_vs_zs.delta_ex = EX(root-only) - EX(zs); E3.<b>.loop_over_root.delta_ex = EX(full_gated) - EX(root-only). Paired, db-clustered, r0."""
    done = []
    for mk in sorted(set(r.modelkey for r in reg.all if r.split == reg.split)):
        arms = {}
        for lab in ("zs", "abl_m5", "full_gated"):
            try:
                arms[lab] = reg.get(lab, model=mk)
            except R.RunError as e:
                res.note("E3: %s" % e)
                arms[lab] = None
        if any(v is None for v in arms.values()):
            continue
        A = R.align(arms)
        boot = S.ClusterBoot(A["dbs"], B)
        pair_block(res, "E3.%s.rootonly_vs_zs" % mk, A["ex"]["abl_m5"], A["ex"]["zs"], A["dbs"], B, boot)
        pair_block(res, "E3.%s.loop_over_root" % mk, A["ex"]["full_gated"], A["ex"]["abl_m5"], A["dbs"], B, boot)
        done.append(mk)
    res.set("E3.backbones_with_rootonly", ", ".join(done) if done else "none")
