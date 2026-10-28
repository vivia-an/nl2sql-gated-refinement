"""E4 driver (S-E), E8 (Spider variants), E9 (contamination probes), setup keys, tuning-set export (T.*), E6 status."""
import glob
import json
import os
import subprocess

import numpy as np

from . import e4 as E4, runs as R, stats as S
from .secondary import pair_block, put_summary, summarize
from .util import REPO

SPIDER_SPLITS = ["spider_dev", "spider_syn", "spider_realistic", "spider_dk"]


# ------------------------------------------------------------------------------------------ E4 (all policies, curves, S-E)
def run_e4(e4_roots, res, B):
    if not e4_roots:
        res.set("E4.status", "not run: no E4 stream directory given")
        return
    units, notes = E4.load_all(e4_roots)
    for n in notes:
        res.note("E4: " + n)
    if not units:
        res.set("E4.status", "not run: no complete stream unit (events.jsonl + done) found")
        return
    pols = sorted(set(u["policy"] for u in units))
    levels = sorted(set(u["p"] for u in units))
    res.set("E4.status", "ok"); res.set("E4.policies", ", ".join(pols)); res.set("E4.pollution_levels", ", ".join(str(p) for p in levels))
    res.set("E4.n_units", len(units)); res.set("E4.perms", ", ".join(str(x) for x in sorted(set(u["perm"] for u in units))))
    for pol in pols:
        for p in levels:
            cv = E4.curves(units, pol, p, B=min(B, 2000))
            if not cv:
                continue
            pre = "E4.%s.p%d" % (pol, p)
            res.ci(pre + ".final_ex", cv["y"][-1], cv["ci_lo"][-1], cv["ci_hi"][-1])
            res.set(pre + ".init_ex.mean", cv["y"][0]); res.set(pre + ".auc.mean", cv["auc"]); res.set(pre + ".n_units", cv["n_units"])
            res.set("E4.curve.%s.p%d.x" % (pol, p), cv["x"]); res.set("E4.curve.%s.p%d.y" % (pol, p), cv["y"])
            res.set("E4.curve.%s.p%d.ci_lo" % (pol, p), cv["ci_lo"]); res.set("E4.curve.%s.p%d.ci_hi" % (pol, p), cv["ci_hi"])
            res.set("E4.curve.%s.p%d.online_x" % (pol, p), cv["online_x"]); res.set("E4.curve.%s.p%d.online_y" % (pol, p), cv["online_y"])
            ev = E4.eviction_diag(units, pol, p)
            if ev and (ev["evicted_polluted"] + ev["evicted_clean"]) > 0:
                for k in ("recall", "precision", "clean_share_of_evicted", "fpr"):
                    res.set("E4.evict.%s.p%d.%s.mean" % (pol, p, k), ev[k])
                # counts and recall under the names the text uses (E4.<policy>.p<pp>.*); same numbers as the E4.evict.* block above
                res.set(pre + ".evicted_clean", ev["evicted_clean"]); res.set(pre + ".evicted_polluted", ev["evicted_polluted"])
                res.set(pre + ".evicted_total", ev["evicted_clean"] + ev["evicted_polluted"])
                res.set(pre + ".recall_polluted", ev["recall"])
    # slopes (log-odds per 10 pp pollution) and the slope difference ws_full vs append
    pair = [p for p in ("ws_full", "append") if p in pols]
    if len(levels) >= 2:
        sl = E4.slope_model(units, pair if len(pair) == 2 else pols[:1], levels) if pair else {}
        others = [p for p in pols if p not in pair]
        for p in others:
            sl.update({k: v for k, v in E4.slope_model(units, [p], levels).items() if k == p})
        for pol, v in sl.items():
            if pol == "_diff":
                if "interaction" in v:
                    res.ci("E4.slope_diff.ws_full_minus_append", v["interaction"], v["ci_lo"], v["ci_hi"]); res.set("E4.slope_diff.p", v["p_lrt"])
                continue
            if "slope" in v:
                res.ci("E4.slope.%s" % pol, v["slope"], v["ci_lo"], v["ci_hi"]); res.set("E4.slope.%s.p" % pol, v["p"])
        if "ws_full" in sl and "slope" in sl["ws_full"]:
            res.ci("E4.slope.ws", sl["ws_full"]["slope"], sl["ws_full"]["ci_lo"], sl["ws_full"]["ci_hi"])
    if "ws_full" in pols:
        ev = {k: res.d.get("E4.evict.ws_full.p40.%s.mean" % k) for k in ("recall", "fpr", "precision")}
        for k, v in ev.items():
            if v is not None:
                res.set("E4.evict.%s" % k, v)
    # S-E: ws_full vs {static, ws_evict, memrl} at p = 40, vs {append, static, ws_evict, memrl} at p = 20 (m = 7, absent comparators count as p = 1)
    fam = [("static", 40), ("ws_evict", 40), ("memrl", 40), ("append", 20), ("static", 20), ("ws_evict", 20), ("memrl", 20)]
    pv, store = {}, {}
    for comp, p in fam:
        key = "%s@p%d" % (comp, p)
        if comp not in pols or "ws_full" not in pols or p not in levels:
            pv[key] = None
            continue
        c = E4.policy_contrast(units, "ws_full", comp, p, min(B, 5000))
        if c and "p_lrt" in c:
            pv[key] = c["p_lrt"]
            store[key] = (comp, p, c)
        else:
            pv[key] = None
    adj = S.holm(pv, len(fam))
    for key, (comp, p, c) in store.items():
        pre = "E4.contrast.ws_full_vs_%s.p%d" % (comp, p)
        res.ci(pre + ".logodds", c["logodds"], c["logodds_ci_lo"], c["logodds_ci_hi"]); res.ci(pre + ".diff_pp", c["diff_pp"], c["diff_pp_ci_lo"], c["diff_pp_ci_hi"])
        res.set(pre + ".p", c["p_lrt"]); res.set(pre + ".p_adj", adj[key])
    if "memrl@p40" in store:
        c = store["memrl@p40"][2]
        res.ci("E4.memrl_vs_ws.p40.delta", c["diff_pp"], c["diff_pp_ci_lo"], c["diff_pp_ci_hi"])      # ws_full minus memrl, pp
        res.set("E4.memrl_vs_ws.p40.p", c["p_lrt"]); res.set("E4.memrl_vs_ws.p40.p_adj", adj["memrl@p40"])
    comp0 = "append" if "append" in pols else ("static" if "static" in pols else None)
    if "ws_full" in pols and comp0 and 0 in levels:
        nh = E4.policy_contrast(units, "ws_full", comp0, 0, min(B, 5000), margin_pp=2.0, fit=False)
        if nh:
            res.ci("E4.ws_full.p0.noharm.delta", nh["diff_pp"], nh["diff_pp_ci_lo"], nh["diff_pp_ci_hi"]); res.set("E4.ws_full.p0.noharm.p_1s", nh.get("p_ni"))
    return units


# ------------------------------------------------------------------------------------------ E8
def run_e8(reg, res, B, labels=("zs", "dail", "full_fixed", "full_gated"), full_label="full_gated"):
    got = {}
    for sp in SPIDER_SPLITS:
        for lab in labels:
            try:
                r = reg.get(lab, split=sp)
            except R.RunError:
                r = None
            if r is not None:
                got[(sp, lab)] = r
                s = summarize(r, min(B, 2000))
                put_summary(res, "E8.%s.%s" % (sp, lab), s)
                ts = [x.get("ex_ts") for x in r.rows.values() if x.get("ex_ts") is not None]
                if ts:
                    res.set("E8.%s.%s.ex_ts.mean" % (sp, lab), float(np.mean(ts)))
    if not got:
        res.set("E8.status", "not run: no complete Spider-variant run")
        return
    res.set("E8.status", "ok")
    for sp in SPIDER_SPLITS[1:]:
        for lab in labels:
            if (sp, lab) in got and ("spider_dev", lab) in got:
                a, b = res.d.get("E8.%s.%s.ex.mean" % (sp, lab)), res.d.get("E8.spider_dev.%s.ex.mean" % lab)
                if a is not None and b is not None:
                    res.set("E8.%s.%s.drop_vs_dev_pp" % (sp, lab), 100.0 * (a - b))
    pv = {}
    for sp in SPIDER_SPLITS:
        for other in ("zs", "dail"):
            if (sp, full_label) in got and (sp, other) in got:
                A = R.align({"a": got[(sp, full_label)], "b": got[(sp, other)]})
                o = pair_block(res, "E8.%s.%s_vs_%s" % (sp, full_label, other), A["ex"]["a"], A["ex"]["b"], A["dbs"], min(B, 5000))
                pv["%s/%s" % (sp, other)] = o["p"]
    if pv:
        adj = S.holm(pv, 6)
        for k, p in adj.items():
            sp, other = k.split("/")
            res.set("E8.%s.%s_vs_%s.p_adj" % (sp, full_label, other), p)


# ------------------------------------------------------------------------------------------ E9
def run_e9(dirs, res):
    files = []
    for d in dirs or []:
        files += glob.glob(os.path.join(d, "**", "summary.json"), recursive=True)
    n_ok = 0
    for f in sorted(files):
        try:
            s = json.load(open(f))
        except ValueError:
            continue
        if "q_repro_0.9" not in s:
            continue
        mk = R.model_keys(os.path.join(REPO, "exp", "models.yaml")).get(s.get("model"), s.get("model"))
        sp = s.get("split")
        pre = "E9.%s.%s" % (mk, sp)
        n = s.get("n_ok") or s.get("n") or 0
        for src, nm in (("q_repro_0.9", "q_repro_09"), ("q_repro_0.7", "q_repro_07"), ("sql_exact", "sql_exact")):
            if src in s and n:
                lo, hi = S.wilson(int(round(s[src] * n)), n)
                res.ci("%s.%s" % (pre, nm), s[src], lo, hi)
        if "table_overlap" in s:
            res.set("%s.table_overlap.mean" % pre, s["table_overlap"])
        res.set(pre + ".n", n)
        n_ok += 1
    res.set("E9.status", "ok" if n_ok else "not run: no probe summary.json found")


# ------------------------------------------------------------------------------------------ setup keys
def run_setup(reg, res, repo=REPO, plan_commit=None, B=None, e4_roots=None):
    tr = os.path.join(repo, "exp", "splits", "train_holdout.json")
    if os.path.exists(tr):
        d = json.load(open(tr))
        res.set("setup.holdout.n_dbs", len(d["held_out_dbs"]))
        res.set("setup.holdout.n_questions", len(d["hr_order"]) + len(d["ht300_qids"]))
        res.set("setup.hr.n_dbs", len(d["hr_dbs"])); res.set("setup.hr.n_questions", len(d["hr_order"]))
        res.set("setup.ht.n_dbs", len(d["ht_dbs"])); res.set("setup.ht.n_questions", len(d["ht300_qids"]))
        res.note("setup.holdout = HR-1101 (8 dbs) + HT-300 (4 dbs): the tuning data of docs/TUNING.md and docs/HEADROOM.md")
    runs = [r for r in reg.all if r.split == reg.split and reg.complete(r)]
    if runs:
        r0 = max(runs, key=lambda r: r.n_rows)
        res.set("setup.bird_dev.n_questions", r0.n_rows)
        res.set("setup.bird_dev.n_dbs", len(set(x.get("db_id") for x in r0.rows.values())))
    res.set("setup.seeds", "0, 101, 202" if True else "")
    run_power_constants(reg, res, B, e4_roots)
    h = plan_commit
    if not h:
        try:
            h = subprocess.check_output(["git", "-C", repo, "log", "-1", "--format=%h", "--", "docs/SAP.md"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            h = None
    if h:
        res.set("setup.plan.commit", h)
    else:
        res.note("setup.plan.commit: no git history here (worker machine copy); pass --plan-commit $(git log -1 --format=%h -- docs/SAP.md) from the host")
    lock = os.path.join(repo, "exp", "dev_lock.json")
    if os.path.exists(lock):
        import hashlib
        res.set("setup.devlock.hash", hashlib.sha256(open(lock, "rb").read()).hexdigest()[:12])
    else:
        res.note("setup.devlock.hash: exp/dev_lock.json does not exist yet")


PLAN_RHO, PLAN_PD_NEAR, PLAN_PD_E4, PLAN_POWER, PLAN_ALPHA_HOLM = 0.01, 0.05, 0.20, 0.80, 0.05 / 5.0


def mde_pp(n, p_d, deff, alpha, power=PLAN_POWER):
    """SAP 9: MDE = (z_{1-alpha/2} + z_power) * sqrt(p_d * DEFF / n), in percentage points."""
    return 100.0 * (S.norm_ppf(1.0 - alpha / 2.0) + S.norm_ppf(power)) * float(np.sqrt(p_d * deff / float(n)))


def run_power_constants(reg, res, B=None, e4_roots=None):
    """setup.design_effect / setup.resolution_pp / setup.p4_mde_pp from the SAP 9 / 13.7 definitions, with n and the number of databases taken from the data.
    * design effect: DEFF = 1 + (m - 1) * rho, m = items per database on dev, rho = 0.01 (the planning value of SAP 9: ICC of the paired differences, assumed).
      Observed analogues (variance of the db-clustered bootstrap over the variance of the item-level bootstrap of the same paired difference) are written as
      setup.design_effect_emp.* and are NOT the key the text uses.
    * resolution: SAP 13.7 row 'p_d 0.05': MDE at n = dev items, DEFF as above, power 0.80, two-sided alpha 0.05/5 (Holm step 1, m = 5) -> 3.0 pp.
    * P4 MDE: SAP 9 K3 row: p_d 0.2, power 0.80, two-sided alpha 0.05, DEFF 1 (permutations add little information), n = distinct probe items of the E4 final checkpoint.
    * setup.boot_resolution_pp = 100 / (B + 1): smallest step of a bootstrap p-value (percentage points of probability, not of EX)."""
    r0 = [r for r in reg.all if r.split == reg.split and reg.complete(r)]
    if r0:
        big = max(r0, key=lambda r: r.n_rows)
        n = big.n_rows
        k = len(set(x.get("db_id") for x in big.rows.values()))
        deff = 1.0 + (n / float(k) - 1.0) * PLAN_RHO
        res.set("setup.design_effect", deff); res.set("setup.design_effect.rho_assumed", PLAN_RHO); res.set("setup.design_effect.items_per_db", n / float(k))
        res.set("setup.design_effect.basis", "planning design effect (SAP 9): 1 + (m - 1) * rho, m = %.1f items per database, rho = %.2f assumed; not estimated from these data (observed analogues: setup.design_effect_emp.*)" % (n / float(k), PLAN_RHO))
        res.set("setup.resolution_pp", mde_pp(n, PLAN_PD_NEAR, deff, PLAN_ALPHA_HOLM))
        res.set("setup.resolution_pp.basis", "SAP 13.7 row p_d 0.05: n=%d, DEFF=%.2f, power 0.80, two-sided alpha 0.01 (Holm step 1, m=5)" % (n, deff))
        # observed design effect of the paired differences (descriptive, not used by the text)
        pairs = [("full_gated_vs_full_fixed.qwen37plus", "full_gated", "full_fixed", "qwen37plus"), ("full_gated_vs_zs.qwen37plus", "full_gated", "zs", "qwen37plus"),
                 ("full_gated_vs_zs.qwen37flash", "full_gated", "zs", "qwen37flash"), ("full_gated_vs_zs.qwen3527b", "full_gated", "zs", "qwen3527b")]
        Bq = min(B or 2000, 2000)
        for tag, la, lb, mk in pairs:
            try:
                a, b = reg.get(la, model=mk), reg.get(lb, model=mk)
            except R.RunError:
                continue
            if a is None or b is None:
                continue
            A = R.align({"a": a, "b": b})
            d = (A["ex"]["a"] - A["ex"]["b"]).astype(float)
            boot = S.ClusterBoot(A["dbs"], Bq)
            vc, vi = float(np.var(boot.means(d))), float(np.var(S.item_boot_means(d, Bq)))
            if vi > 0:
                res.set("setup.design_effect_emp." + tag, vc / vi)
                res.set("setup.design_effect_emp.%s.rho" % tag, (vc / vi - 1.0) / max(n / float(k) - 1.0, 1.0))
    if B:
        res.set("setup.boot_resolution_pp", 100.0 / (B + 1.0))
    if e4_roots:
        units, _ = E4.load_all(e4_roots)
        rows = E4.probe_rows(units, "ws_full", 40)
        ni = len(set(rows["qid"].tolist()))
        if ni:
            res.set("setup.p4_mde_pp", mde_pp(ni, PLAN_PD_E4, 1.0, 0.05))
            res.set("setup.p4_mde_pp.n_probe_items", ni)


# ------------------------------------------------------------------------------------------ tuning export
def _flat(prefix, x, out, depth=0):
    if isinstance(x, dict):
        for k, v in x.items():
            _flat(prefix + "." + str(k).replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("'", "").replace("/", "_"), v, out, depth + 1)
    elif isinstance(x, list):
        if len(x) == 3 and all(isinstance(v, (int, float)) for v in x):
            out[prefix + ".mean"], out[prefix + ".ci_lo"], out[prefix + ".ci_hi"] = x      # [estimate, lo, hi] convention of the R1 analysis files
        elif x and all(isinstance(v, (int, float)) for v in x) and len(x) <= 12:
            for i, v in enumerate(x):
                out["%s.%d" % (prefix, i)] = v
    elif isinstance(x, bool):
        out[prefix] = x
    elif isinstance(x, (int, float)):
        out[prefix] = x
    elif isinstance(x, str) and len(x) < 120:
        out[prefix] = x


def run_tuning(res, results_dir):
    """T.* keys: flatten the tuning-set analyses in results/r1/*.json and results/p2/*_summary.json (BIRD-train data only; never confirmatory)."""
    n = 0
    for sub, pat in (("r1", "*.json"), ("p2", "*summary*.json"), ("p2", "base_ht300.json"), ("p2", "e4_tuning.json")):
        for f in sorted(glob.glob(os.path.join(results_dir, sub, pat))):
            try:
                d = json.load(open(f))
            except ValueError:
                continue
            tmp = {}
            _flat("T.%s.%s" % (sub, os.path.splitext(os.path.basename(f))[0]), d, tmp)
            for k, v in tmp.items():
                res.set(k, v)
            n += 1
    # curated aliases used by STORYLINE R8
    g = lambda k: res.d.get(k)
    ex = "T.r1.q1_errs.P0_replay_gain_harm_beer_vs_others"
    gb, go = g(ex + ".beer_factory_exec_error.gain"), g(ex + ".others_exec_error.gain")
    if gb is not None and go is not None and (gb + go) > 0:
        res.set("T.hr.repair_share_one_db", gb / float(gb + go))
        res.set("T.hr.repair_gains_one_db", gb); res.set("T.hr.repair_gains_total", gb + go)
    if g("T.r1.q1_headroom.excl_beer_factory.delta_pp") is not None:
        res.set("T.hr.p0_gain_excl_db", g("T.r1.q1_headroom.excl_beer_factory.delta_pp"))
    res.set("T.status", "ok" if n else "not run: no tuning json found")
    res.set("T.n_files", n)


# ------------------------------------------------------------------------------------------ E6
def run_e6(res):
    res.set("E6.status", "not run: MINIMAL_PLAN drops the live RL runs (D9); the negative result rests on the tuning-set CV in T.p2.e6_summary.* (labelled tuning set)")
