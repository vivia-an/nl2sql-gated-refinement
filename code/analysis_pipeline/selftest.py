"""Self-test of the analysis pipeline:  python -m analysis_pipeline.selftest   (worker machine venv: numpy + scipy)

1. statistics against scipy / textbook values (exact McNemar, Clopper-Pearson, Wilson, AUROC, Holm, chi-square);
2. planted effects on SYNTHETIC runs are recovered by primary.py (null effect -> not significant; planted effect -> detected; direction right);
3. missing runs -> `not run`, never an invented number; P5 with n_R < 40 -> descriptive, family m = 4; P3 dropped by decision -> m = 4;
4. scope guards: synthetic / tuning-set results refuse to enter results/ and make_macros.
Prints PASS/FAIL per check; exit code 1 on any failure."""
import os
import re
import shutil
import sys
import tempfile

import numpy as np

from . import e7 as E7, misc, primary, runs as R, secondary, stats as S, synth, weak
from .util import Result, set_default_scope

FAILS = []


def check(name, cond, info=""):
    print("%-4s %s %s" % ("PASS" if cond else "FAIL", name, info))
    if not cond:
        FAILS.append(name)


def t_stats():
    from scipy import stats as ss
    for b, c in [(10, 3), (7, 7), (0, 5), (25, 14)]:
        p = S.mcnemar_exact(b, c)["p_two"]
        ref = ss.binomtest(b, b + c, 0.5).pvalue
        check("mcnemar_exact(%d,%d)" % (b, c), abs(p - ref) < 1e-9, "%.6f vs %.6f" % (p, ref))
    for k, n in [(3, 20), (0, 10), (10, 10), (55, 100)]:
        lo, hi = S.clopper_pearson(k, n)
        ref = ss.binomtest(k, n).proportion_ci(0.95, method="exact")
        check("clopper_pearson(%d,%d)" % (k, n), abs(lo - ref.low) < 1e-6 and abs(hi - ref.high) < 1e-6)
        lo, hi = S.wilson(k, n)
        ref = ss.binomtest(k, n).proportion_ci(0.95, method="wilson")
        check("wilson(%d,%d)" % (k, n), abs(lo - ref.low) < 1e-6 and abs(hi - ref.high) < 1e-6)
    rng = np.random.RandomState(0)
    y = (rng.rand(300) < 0.5).astype(int)
    sc = y * 0.8 + rng.randn(300)
    u = ss.mannwhitneyu(sc[y == 1], sc[y == 0]).statistic / ((y == 1).sum() * (y == 0).sum())
    check("auroc vs Mann-Whitney", abs(S.auroc(sc, y) - u) < 1e-9)
    a = S.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None}, 5)
    check("holm m=5 with a missing test", abs(a["a"] - 0.05) < 1e-12 and abs(a["c"] - 0.12) < 1e-12 and abs(a["b"] - 0.12) < 1e-12 and a["d"] is None, str(a))
    check("chi2_sf", abs(S.chi2_sf(5.99, 2) - ss.chi2.sf(5.99, 2)) < 1e-6)
    check("norm_ppf", abs(S.norm_ppf(0.975) - 1.959964) < 1e-5)


def t_glmm():
    """crossed random intercepts: recovery of the fixed effect over replicates (true beta 0.6) and agreement with plain logistic regression when sigma -> 0."""
    from . import glmm
    cover, est = 0, []
    nrep = 12
    for rep in range(nrep):
        rng = np.random.RandomState(100 + rep)
        ndb, nq, nperm = 8, 40, 3
        db = np.repeat(np.arange(ndb), nq)
        qid = np.arange(ndb * nq)
        bdb, bq = rng.randn(ndb) * 0.5, rng.randn(ndb * nq) * 1.0
        ys, xs, ds, qs, ps = [], [], [], [], []
        for pol in (0, 1):
            for pm in range(nperm):
                eta = -0.2 + 0.6 * pol + bdb[db] + bq[qid]
                ys.append((rng.rand(len(db)) < 1 / (1 + np.exp(-eta))).astype(int))
                xs.append(np.full(len(db), pol)); ds.append(db); qs.append(qid); ps.append(np.full(len(db), pm))
        r = glmm.contrast(np.concatenate(ys), np.concatenate(xs), [np.concatenate(ds), np.concatenate(qs), np.concatenate(ps)], label="policy")
        est.append(r["beta"])
        cover += int(r["ci_lo"] <= 0.6 <= r["ci_hi"])
    check("GLMM: mean estimate of beta within 0.15 of 0.6 over %d replicates" % nrep, abs(np.mean(est) - 0.6) < 0.15, "mean %.3f" % np.mean(est))
    check("GLMM: Wald CI covers the true beta in >= 9 of 12 replicates", cover >= 9, "%d/12" % cover)
    rng = np.random.RandomState(7)
    x2 = rng.randint(0, 2, 3000)
    y2 = (rng.rand(3000) < 1 / (1 + np.exp(-(-0.3 + 0.5 * x2)))).astype(int)
    r2 = glmm.contrast(y2, x2, [rng.randint(0, 5, 3000)], label="x")
    p1, p0 = y2[x2 == 1].mean(), y2[x2 == 0].mean()
    plain = np.log(p1 / (1 - p1)) - np.log(p0 / (1 - p0))
    check("GLMM without real random variance equals the plain logit contrast", abs(r2["beta"] - plain) < 0.02, "%.4f vs %.4f" % (r2["beta"], plain))


def _run_primary(root, e4=None, decisions=None, B=400, split="bird_dev"):
    reg = R.Registry(root, {}, os.path.join(os.path.dirname(__file__), "..", "..", "exp", "models.yaml"), split)
    res, info = primary.run_primary(reg, e4 or [], decisions or {}, B=B)
    return res


def t_planted(tmp):
    set_default_scope("synthetic")
    # null world: no planted effect on EX between gated and fixed, no E4 resistance
    null = os.path.join(tmp, "null")
    synth.make_dev_world(null, seed=3, effects={"gated_minus_fixed": 0.0, "cons_strength": 0.0})
    synth.make_e4(null, seed=3, effects={"ws_drop": 0.14, "append_drop": 0.14, "memrl_drop": 0.14})
    r0 = _run_primary(null, [os.path.join(null, "MIN")])
    check("P1 null: gated is not detected as better in EX by luck (|delta| < 3 pp)", abs(r0.d["P1.delta_ex.mean"]) < 3.0, "delta %.2f" % r0.d["P1.delta_ex.mean"])
    check("P1 calls ratio near the planted design (<0.8)", r0.d["P1.calls_ratio.mean"] < 0.8, "%.3f" % r0.d["P1.calls_ratio.mean"])
    check("P3 null (no consistency signal): dAUROC CI contains 0 or tiny", r0.d["P3.dauroc.ci_lo"] < 0.03, "%.3f" % r0.d["P3.dauroc.ci_lo"])
    check("P4 null: ws_full vs append CI contains 0", r0.d["P4.logodds.ci_lo"] < 0 < r0.d["P4.logodds.ci_hi"], "[%.3f, %.3f]" % (r0.d["P4.logodds.ci_lo"], r0.d["P4.logodds.ci_hi"]))
    # effect world
    eff = os.path.join(tmp, "eff")
    synth.make_dev_world(eff, seed=4, effects={"gated_minus_fixed": 0.0, "cons_strength": 1.2, "repair_gated": 0.8, "repair_blind": 0.3})
    synth.make_e4(eff, seed=4, effects={"ws_drop": 0.02, "append_drop": 0.25})
    r1 = _run_primary(eff, [os.path.join(eff, "MIN")])
    check("P3 effect: dAUROC > 0 and CI excludes 0", r1.d["P3.dauroc.ci_lo"] > 0, "%.3f" % r1.d["P3.dauroc.mean"])
    check("P4 effect: log-odds > 0 and CI excludes 0", r1.d["P4.logodds.ci_lo"] > 0, "%.3f" % r1.d["P4.logodds.mean"])
    check("P5 effect: repair rate gated > blind (b > c)", r1.d["P5.b"] > r1.d["P5.c"], "b=%s c=%s" % (r1.d["P5.b"], r1.d["P5.c"]))
    check("P.family.m == 5 with all members", r1.d["P.family.m"] == 5)
    check("p_adj >= p_2s for every member", all(r1.d[t + ".p_adj"] >= r1.d[t + ".p_2s"] - 1e-12 for t in ("P1", "P2", "P3", "P4", "P5") if t + ".p_adj" in r1.d))
    check("every key finite", all(np.isfinite(v) for k, v in r1.d.items() if isinstance(v, float)))
    # decisions: P3 dropped -> m = 4
    r2 = _run_primary(eff, [os.path.join(eff, "MIN")], {"route_ca_adopted": False})
    check("P3 dropped by decision -> m = 4, verdict 'dropped'", r2.d["P.family.m"] == 4 and r2.d["P3.verdict"].startswith("dropped"))
    # missing runs
    only_e4 = os.path.join(tmp, "e4only")
    synth.make_e4(only_e4, seed=5)
    r3 = _run_primary(only_e4, [os.path.join(only_e4, "MIN")])
    nr = [t for t in ("P1", "P2", "P3", "P5") if r3.d[t + ".status"].startswith("not run")]
    check("no dev runs -> P1,P2,P3,P5 'not run' and no numbers", nr == ["P1", "P2", "P3", "P5"] and not any(k.startswith("P1.delta") or k.startswith("P3.dauroc") for k in r3.d))
    check("P4 still produced from E4 alone", r3.d["P4.status"] == "ok")
    empty = os.path.join(tmp, "empty")
    os.makedirs(empty)
    r4 = _run_primary(empty)
    check("empty runs dir -> all five 'not run'", all(r4.d[t + ".status"].startswith("not run") for t in ("P1", "P2", "P3", "P4", "P5")))
    # incomplete run is never used silently
    inc = os.path.join(tmp, "inc")
    synth.make_dev_world(inc, seed=6, n_scale=1.0)
    for d, _, fs in os.walk(inc):
        if "preds.jsonl" in fs and "full_gated-" in d and "blind" not in d:
            lines = open(os.path.join(d, "preds.jsonl")).read().splitlines()
            open(os.path.join(d, "preds.jsonl"), "w").write("\n".join(lines[:800]) + "\n")
    r5 = _run_primary(inc)
    check("truncated full_gated run -> P1 'not run' (incomplete)", r5.d["P1.status"].startswith("not run"))


def t_min_stems(tmp):
    """MINIMAL_PLAN naming: zs system `zs_ours`, subset baseline `memosql` (500 rows, config memosql_light), r0 gated run with config full_gated_route_ca."""
    import json
    from . import secondary
    root = os.path.join(tmp, "minstems")
    synth.make_dev_world(root, seed=8)
    for d in [d for d, _, fs in os.walk(root) if "config.yaml" in fs and "full_gated_cons_lr" in open(os.path.join(d, "config.yaml")).read()]:
        shutil.rmtree(d)                                  # in MIN the routing run IS the r0 gated run
    for d, _, fs in list(os.walk(root)):
        if "preds.jsonl" not in fs:
            continue
        m = json.load(open(os.path.join(d, "run_meta.json")))
        cfgp = os.path.join(d, "config.yaml")
        if m["system"] == "zs":
            m["system"] = "zs_ours"
        elif m["system"] == "sdebug":
            m["system"] = "memosql"
            lines = open(os.path.join(d, "preds.jsonl")).read().splitlines()[:500]
            open(os.path.join(d, "preds.jsonl"), "w").write("\n".join(lines) + "\n")
            open(cfgp, "w").write("argv:\n  config: exp/configs/frozen/memosql_light.yaml\nresolved: {}\n")
        elif m["system"] == "full_gated" and "configs/full_gated.yaml" in open(cfgp).read():
            open(cfgp, "w").write("argv:\n  config: exp/configs/frozen/full_gated_route_ca.yaml\nresolved: {}\n")
        json.dump(m, open(os.path.join(d, "run_meta.json"), "w"))
    reg = R.Registry(root, {}, os.path.join(os.path.dirname(__file__), "..", "..", "exp", "models.yaml"), "bird_dev")
    check("zs_ours resolves to label zs", reg.get("zs") is not None)
    check("full_gated_route_ca config resolves to full_gated and full_gated_route", reg.get("full_gated") is not None and reg.get("full_gated_route") is not None)
    res = Result("E1")
    secondary.run_e1(reg, res, 200, {}, {}, None)
    check("subset baseline (memosql, 500 rows) is used for E1 and flagged", res.d.get("E1.memosql.subset_n") == 500, str(res.d.get("E1.memosql.subset_n")))
    pr = Result("primary")
    primary.test_p2(reg, pr, 200, {}, "full_gated")
    check("P2 never picks the subset baseline as B*", pr.d.get("P2.bstar") not in (None, "memosql"), str(pr.d.get("P2.bstar")))


def t_weak(tmp):
    """SAP 15.3 backbone family (weak.py), E3 per backbone, E4 eviction counts, setup constants, X1, E7 with two bon_vote budgets."""
    import json
    set_default_scope("synthetic")
    root = os.path.join(tmp, "weak")
    full = ["zs", "full_fixed", "full_gated", "full_gated_blind", "abl_m5"]
    synth.make_dev_world(root, seed=21, models=("qwen37plus",), systems=full + ["full_gated_route", "bon_vote"], effects={"zs_delta": -0.6})
    synth.make_dev_world(root, seed=22, models=("qwen37flash",), systems=full, effects={"zs_delta": -0.6})
    synth.make_dev_world(root, seed=23, models=("qwen3527b",), systems=full, effects={"zs_delta": -0.6})
    synth.make_e4(root, seed=21)
    models_yaml = os.path.join(os.path.dirname(__file__), "..", "..", "exp", "models.yaml")
    reg = R.Registry(root, {}, models_yaml, "bird_dev")
    B = 300
    pres, _ = primary.run_primary(reg, [os.path.join(root, "MIN")], {}, B=B)
    res = Result("W1")
    weak.run_w1_stage(reg, res, B, pres)
    d = res.d
    # (1) per-backbone P1 on the primary backbone reproduces primary.json exactly (same code, same seed)
    check("P1.qwen37plus.* == P1.* of primary (delta, CI, calls ratio)",
          all(abs(d["P1.qwen37plus." + k] - pres.d["P1." + k]) < 1e-12 for k in ("delta_ex.mean", "delta_ex.ci_lo", "delta_ex.ci_hi", "calls_ratio.mean", "calls_ratio.ci_lo", "calls_ratio.ci_hi")))
    check("P1.n_backbones == 3 and S-J m == 3", d["P1.n_backbones"] == 3 and d["P1.sj.m"] == 3)
    mean3 = np.mean([d["P1.%s.delta_ex.mean" % m] for m in weak.BK])
    check("P1prime.delta_ex = mean of the per-backbone deltas", abs(d["P1prime.delta_ex.mean"] - mean3) < 1e-9, "%.4f vs %.4f" % (d["P1prime.delta_ex.mean"], mean3))
    check("P1prime ratio = mean of per-backbone ratios", abs(d["P1prime.calls_ratio.mean"] - np.mean([d["P1.%s.calls_ratio.mean" % m] for m in weak.BK])) < 1e-9)
    # (2) W1: mean over the two weak backbones, planted zero-shot deficit is detected, sign rule, scope of keys
    w = np.mean([d["W1.%s.delta_ex.mean" % m] for m in weak.WK])
    check("W1.delta_ex = mean of the two per-backbone deltas", abs(d["W1.delta_ex.mean"] - w) < 1e-9, "%.4f vs %.4f" % (d["W1.delta_ex.mean"], w))
    check("W1 planted gain over zero-shot: positive, CI above 0, sign consistent, n_backbones 2", d["W1.delta_ex.ci_lo"] > 0 and d["W1.sign_consistent"] is True and d["W1.n_backbones"] == 2)
    check("W1 verdict follows S.verdict", d["W1.verdict"] == "confirmed", d["W1.verdict"])
    # P5' (15.3): pooled over Bk, exact McNemar, n_R and the inclusion rule
    nR_sum = sum(d["P5prime.n_R.%s" % m] for m in weak.BK)
    check("P5prime.n_R = sum over backbones; in_family <=> n_R >= 40", d["P5prime.n_R"] == nR_sum and d["P5prime.in_family"] == (nR_sum >= weak.N_R_MIN), "n_R=%d" % nR_sum)
    if d["P5prime.n_R"]:
        mc = S.mcnemar_exact(d["P5prime.b"], d["P5prime.c"])
        check("P5prime exact McNemar p = S.mcnemar_exact(b, c); the Holm p is that one", abs(d["P5prime.mcnemar.p_1s"] - mc["p_one_greater"]) < 1e-12 and abs(d["P5prime.p_1s"] - mc["p_one_greater"]) < 1e-12)
        check("P5prime repair-rate difference = (b - c) / n_R in pp", abs(d["P5prime.diff_rate_pp.mean"] - 100.0 * (d["P5prime.b"] - d["P5prime.c"]) / d["P5prime.n_R"]) < 1e-9)
        check("P5prime pooled repair rate = weighted mean of the backbone rates", abs(d["P5prime.repair_rate.mean"] - sum(d["P5prime.by_backbone.%s.repair_rate.mean" % m] * d["P5prime.n_R.%s" % m] for m in weak.BK if ("P5prime.by_backbone.%s.repair_rate.mean" % m) in d) / d["P5prime.n_R"]) < 1e-9)
    # family P*: Holm recomputed from the members' p-values, m by the inclusion rules, P-values replace the 13.3 ones
    mem = [x.strip() for x in d["Pstar.family.members"].split(",")]
    two = {x: d[("Pstar.P3" if x == "P3" else x) + ".p_2s"] for x in mem}
    ref = S.holm(two, len(mem))
    check("family P*: m = number of members (4, or 3 when P3 / P5' drop out), Holm recomputed from p_2s",
          d["Pstar.family.m"] == len(mem) and all(abs(d[("Pstar.P3" if x == "P3" else x) + ".p_adj"] - ref[x]) < 1e-12 for x in mem), d["Pstar.family.members"])
    if "P5prime" in mem:
        check("P5' enters Holm with the exact one-sided McNemar p (SAP 15.3), doubled", abs(d["P5prime.p_2s"] - min(1.0, 2.0 * d["P5prime.mcnemar.p_1s"])) < 1e-12)
    check("family P*: P5' is a member iff its inclusion rule holds", ("P5prime" in mem) == bool(d["P5prime.in_family"]))
    check("W1 Holm: p_adj >= p_2s; P3 enters with the p of primary.json", d["W1.p_adj"] >= d["W1.p_2s"] - 1e-12 and abs(d["Pstar.P3.p_2s"] - pres.d["P3.p_2s"]) < 1e-12)
    check("no superseded m = 3 variants are written", not any(k.endswith(".p_adj_m3") or k.endswith(".verdict_m3") for k in d))
    check("primary.json carries the superseded-family label and keeps the 13.3 Holm values", "SUPERSEDED" in pres.d["P.family.label"] and "P1.p_adj" in pres.d)
    # sign-consistency rule: pooled test passes, one member <= 0 -> 'supported on average, not on each backbone'
    fake = Result("primary")
    fake.set("P3.status", "ok"); fake.set("P3.p_2s", 0.5); fake.set("P3.p_item_2s", 0.5)
    r2 = Result("W1")
    r2.set("W1.sign_consistent", False)
    o = {"p_1s": 1e-4, "p_item_1s": 1e-4, "cl_ok": True, "item_ok": True, "favourable": True}
    weak.run_pstar(reg, r2, None, o, None, fake)
    check("sign rule: confirmed pooled test with a non-positive member -> 'supported on average, not on each backbone'", r2.d["W1.verdict"] == weak.SJ_WORDS, r2.d.get("W1.verdict"))
    # secondary rows: P2 in S-A, P4 in S-E (m = 8) with Holm inside their own families
    fp = Result("primary")
    fp.set("P2.bstar", "macsql"); fp.set("P2.delta_ex.mean", -1.0); fp.set("P2.delta_ex.ci_lo", -2.0); fp.set("P2.delta_ex.item_ci_lo", -2.0)
    fp.set("P4.p_lrt", 0.04); fp.set("P4.logodds.ci_lo", 0.1); fp.set("P4.logodds.mean", 0.5); fp.set("P4.p_mcnemar", 0.01); fp.set("P4.diff_pp.mean", 1.0)
    fe1 = Result("E1"); fe1.set("E1.full_vs.macsql.p_adj", 0.02); fe1.set("E1.full_vs.family_m", 6)
    fe4 = Result("E4")
    for comp, pp, pv in (("static", 40, 0.01), ("ws_evict", 40, 0.2), ("memrl", 40, 0.03), ("append", 20, 0.5), ("static", 20, 0.04), ("ws_evict", 20, None), ("memrl", 20, 0.6)):
        if pv is not None:
            fe4.set("E4.contrast.ws_full_vs_%s.p%d.p" % (comp, pp), pv)
    sec = Result("Psec")
    weak.run_secondary_rows(sec, fp, fe1, fe4)
    exp = S.holm({"append@p40": 0.04, "static@p40": 0.01, "ws_evict@p40": 0.2, "memrl@p40": 0.03, "append@p20": 0.5, "static@p20": 0.04, "ws_evict@p20": None, "memrl@p20": 0.6}, 8)["append@p40"]
    check("secondary rows: P2.p_adj_sa = E1 Holm value (S-A); P4.p_adj_se = Holm over the 7 S-E rows + append@40 (m = 8)",
          sec.d["P2.p_adj_sa"] == 0.02 and sec.d["P4.se_family_m"] == 8 and abs(sec.d["P4.p_adj_se"] - exp) < 1e-12, "%.4f vs %.4f" % (sec.d["P4.p_adj_se"], exp))
    # make_macros: the paper's P1/P3/P5/P2/P4/P.family keys resolve to the P* / secondary values, never silently to the superseded family
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_macros_t", os.path.join(os.path.dirname(__file__), "..", "tools", "make_macros.py"))
    mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
    outd = {"P1.delta_ex": "old", "P1.delta_ex.ci": "old", "P1prime.delta_ex": "new", "P1prime.delta_ex.ci": "[new]", "P1.p_adj": "old", "P1prime.p_adj": "newp",
            "P3.verdict": "old", "Pstar.P3.verdict": "newv", "P5.n_R": "23", "P5prime.n_R": "135", "P5.repair_rate": "old",
            "P2.p_adj": "old", "P2.p_adj_sa": "sa", "P.family.m": "5", "Pstar.family.m": "4", "P1.calls_ratio.calls_gated": "keep", "P1.n_backbones": "3"}
    red, dropped = mm.apply_overrides(outd)
    check("make_macros OVERRIDES: P1/P3/P5/P2/P.family keys take the P* values, unrelated keys stay",
          outd["P1.delta_ex"] == "new" and outd["P1.delta_ex.ci"] == "[new]" and outd["P1.p_adj"] == "newp" and outd["P3.verdict"] == "newv" and outd["P5.n_R"] == "135"
          and outd["P2.p_adj"] == "sa" and outd["P.family.m"] == "4" and outd["P1.calls_ratio.calls_gated"] == "keep" and outd["P1.n_backbones"] == "3")
    check("make_macros OVERRIDES: a redirect without a target is dropped (reported missing), not served from the old family", "P5.repair_rate" not in outd and "P5.repair_rate" in dropped)
    # keys with a percent sign (tuning-set exports such as 'stable_wrong_>=80%_nodes...') must not start a TeX comment inside \csname
    import shutil, subprocess
    td = os.path.join(tmp, "mm_pct"); os.makedirs(os.path.join(td, "results")); os.makedirs(os.path.join(td, "paper", "sections"))
    odd = ["T.x.stable_wrong_>=80%_nodes.n", "T.x.agree_+~9_calls.n", "T.x.a#b&c^d$e{f}g\\h.n", "T.x.plain_ok-1.n"]
    rr = Result("T", "confirmatory")
    for i_, k_ in enumerate(odd):
        rr.set(k_, 10 + i_)
    rr.write(os.path.join(td, "results", "T.json"))
    open(os.path.join(td, "paper", "sections", "a.tex"), "w").write("\\res{T.x.plain_ok-1.n}\n")           # a raw % cannot occur in LaTeX source (comment), so only the macro FILE must cope with it
    mm.main(["--results", os.path.join(td, "results"), "--paper", os.path.join(td, "paper")])
    body = open(os.path.join(td, "paper", "results_macros.tex")).read()
    mac_lines = [ln for ln in body.splitlines() if "csname resv@T.x." in ln]
    safe_re = re.compile(r"^[A-Za-z0-9._\-+=<>/,()\[\]@*!?|:;']*$")
    def _rest(ln):            # name part of the csname with the escape sequences removed
        return re.sub(r"\\expandafter\\@gobble\\string\\.", "", ln.split("csname resv@")[1].split("\\endcsname")[0])
    bad = [ln for ln in mac_lines if not safe_re.match(_rest(ln))]
    check("make_macros: every special character (% ~ # & ^ $ { } backslash) of a key is escaped in the macro file; the rest of the name matches the safe set", not bad and len(mac_lines) == len(odd) and mm.unsafe_keys(odd) == sorted(odd[:3]), str(bad[:1]))
    ok_t, tail_t = mm.check_tex(odd + ["T.x.plain_ok-1.n"], os.path.join(td, "paper", "results_macros.tex"), os.path.join(td, "kc"))
    if ok_t is not None:
        check("make_macros --check-tex: pdflatex resolves every key, including those with % ~ # & ^ $ { } backslash", ok_t is True, tail_t[-200:])
    # (3) missing runs: no numbers, status says why
    only = os.path.join(tmp, "weak_plus_only")
    synth.make_dev_world(only, seed=24, models=("qwen37plus",), systems=full)
    reg1 = R.Registry(only, {}, models_yaml, "bird_dev")
    r3 = Result("W1")
    weak.run_w1_stage(reg1, r3, B, None)
    check("only qwen37plus: W1 'not run' with no W1.delta_ex, P1prime 'not run', P1.n_backbones == 1",
          r3.d["W1.status"].startswith("not run") and "W1.delta_ex.mean" not in r3.d and r3.d["P1prime.status"].startswith("not run") and r3.d["P1.n_backbones"] == 1)
    # (4) E3 per backbone: gated - root-only and root-only - zs
    e3 = Result("E3")
    secondary.run_e3_backbones(reg, e3, B)
    mk = "qwen37flash"
    g, ro, z = reg.get("full_gated", model=mk), reg.get("abl_m5", model=mk), reg.get("zs", model=mk)
    A = R.align({"g": g, "r": ro, "z": z})
    check("E3.<b>.loop_over_root = EX(gated) - EX(root-only), rootonly_vs_zs = EX(root-only) - EX(zs)",
          abs(e3.d["E3.%s.loop_over_root.delta_ex.mean" % mk] - 100.0 * (A["ex"]["g"].mean() - A["ex"]["r"].mean())) < 1e-9 and
          abs(e3.d["E3.%s.rootonly_vs_zs.delta_ex.mean" % mk] - 100.0 * (A["ex"]["r"].mean() - A["ex"]["z"].mean())) < 1e-9)
    # (5) E4 eviction counts under the names the text uses
    e4r = Result("E4")
    misc.run_e4([os.path.join(root, "MIN")], e4r, 200)
    ec, ep, tot = e4r.d["E4.ws_full.p40.evicted_clean"], e4r.d["E4.ws_full.p40.evicted_polluted"], e4r.d["E4.ws_full.p40.evicted_total"]
    check("E4.ws_full.p40.evicted_total == clean + polluted > 0 and recall_polluted == E4.evict recall",
          tot == ec + ep and tot > 0 and abs(e4r.d["E4.ws_full.p40.recall_polluted"] - e4r.d["E4.evict.ws_full.p40.recall.mean"]) < 1e-12)
    # (6) setup constants from the SAP 9 / 13.7 definitions
    st = Result("setup")
    misc.run_power_constants(reg, st, 10000, [os.path.join(root, "MIN")])
    deff = 1 + (1534 / 11.0 - 1) * 0.01
    check("setup.design_effect = 1 + (items per db - 1) * 0.01", abs(st.d["setup.design_effect"] - deff) < 1e-9, "%.4f" % st.d["setup.design_effect"])
    check("setup.resolution_pp reproduces SAP 13.7 (3.0 pp at p_d 0.05, DEFF 2.4, alpha 0.01)", abs(st.d["setup.resolution_pp"] - 3.0) < 0.1, "%.3f" % st.d["setup.resolution_pp"])
    n_probe = st.d["setup.p4_mde_pp.n_probe_items"]
    check("setup.p4_mde_pp = (1.96 + 0.84) sqrt(0.2 / n) with n = distinct probe items", abs(st.d["setup.p4_mde_pp"] - 100 * (S.norm_ppf(0.975) + S.norm_ppf(0.8)) * np.sqrt(0.2 / n_probe)) < 1e-9, "n=%d mde=%.2f" % (n_probe, st.d["setup.p4_mde_pp"]))
    check("setup.boot_resolution_pp = 100 / (B + 1)", abs(st.d["setup.boot_resolution_pp"] - 100.0 / 10001) < 1e-12)
    # (7) X1: cost ratio from the E11 cost model, descriptive scope
    prices = {"cache_hit_ratio": 0.2, "batch_ratio": 0.5, "models": {"qwen3.7-plus-2026-05-26": {"in": 2.0, "out": 8.0}, "qwen3.7-flash-2026-07-15": {"in": 0.2, "out": 0.8}, "qwen3.5-27b": {"in": 0.6, "out": 4.8}}}
    x1 = Result("X1")
    weak.run_x1(reg, x1, B, prices)
    gc, zc = secondary.run_cost(reg.get("full_gated", model="qwen37flash"), prices)["per_q"], secondary.run_cost(reg.get("zs", model="qwen37plus"), prices)["per_q"]
    check("X1 cost ratio = flash gated / plus zs per question (usage-based cost model)", abs(x1.d["X1.flash_vs_plus_zs.cost_ratio"] - gc / zc) < 1e-12)
    check("X1 is marked descriptive and carries the S-I bounds", x1.d["X1.scope"].startswith("descriptive") and x1.d["X1.flash_vs_plus_zs.cost_ratio_bound"] == 0.70 and x1.d["X1.flash_vs_plus_zs.margin_pp"] == 3.0)
    # (8) E7 does not abort when two bon_vote budgets (n = 2 and n = 3) are complete
    import shutil as sh
    for dd, _, fs in list(os.walk(root)):
        if "preds.jsonl" in fs and "bon_vote" in os.path.basename(dd) and "qwen37plus" in dd:
            dst = dd + "-n3copy"
            sh.copytree(dd, dst)
            m = json.load(open(os.path.join(dst, "run_meta.json"))); m["config_sha256"] = "x" * 8; m["run_id"] = m["run_id"] + "-n3copy"
            json.dump(m, open(os.path.join(dst, "run_meta.json"), "w"))
            open(os.path.join(dst, "config.yaml"), "w").write("argv:\n  config: exp/configs/bon_vote_n3.yaml\n  exp: MIN\nresolved: {}\n")
            break
    reg2 = R.Registry(root, {}, models_yaml, "bird_dev")
    e7r = Result("E7")
    E7.run_e7(reg2, e7r, 100)
    check("E7 with two bon_vote budgets: stage completes and writes the keys the text uses",
          e7r.d.get("E7.status") == "ok" and all(k in e7r.d for k in ("E7.route_ca.aurc.mean", "E7.v3.aurc.mean", "E7.route_ca.acc_ex_cov60.mean", "E7.v3.acc_ex_cov60.mean", "E7.route_ca_vs_v3.daurc.mean", "E7.route_ca_vs_v3.daurc.ci_lo")))


def t_scope(tmp):
    from .util import REPO
    import json
    import subprocess
    res = Result("x", scope="tuning-set")
    try:
        res.write(os.path.join(REPO, "results", "x.json"))
        ok = False
        os.remove(os.path.join(REPO, "results", "x.json"))
    except SystemExit:
        ok = True
    check("Result.write refuses tuning-set scope in results/", ok)
    d = os.path.join(tmp, "scope_results")
    os.makedirs(d)
    json.dump({"P1.x": 1.0, "_meta": {"scope": "synthetic"}}, open(os.path.join(d, "primary.json"), "w"))
    mm = os.path.join(os.path.dirname(__file__), "..", "tools", "make_macros.py")
    rc = subprocess.call([sys.executable, mm, "--results", d, "--paper", tmp, "--out", os.path.join(tmp, "m.tex"), "--report", os.path.join(tmp, "r.txt")], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    check("make_macros refuses synthetic results", rc == 4, "rc=%s" % rc)


def main():
    tmp = tempfile.mkdtemp(prefix="an_selftest_")
    try:
        t_stats()
        t_glmm()
        t_planted(tmp)
        t_min_stems(tmp)
        t_weak(tmp)
        t_scope(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n%d failure(s)" % len(FAILS) if FAILS else "\nall checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
