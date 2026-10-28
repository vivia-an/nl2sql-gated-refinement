#!/usr/bin/env python3
"""R3: merge the result files of the round-3 post hoc analyses into results/r3/rv3_summary.json, the file that make_xmacros.py reads (macro prefix `RV3.`).
Pure Python (no numpy), runs on the host. Format of results/posthoc2/rv2_summary.json: {"_meta": {...}, "keys": {key: {"v", "kind", "ci", "item_ci", "n", "meaning"}}}.

    python3 code/tools/r3_merge.py [--dir results/r3] [--rv2 results/posthoc2/rv2_summary.json] [--write-grammar]

Inputs (all in --dir): rep.json, samex.json, retrieval.json, routing.json, p5.json, q360.json, addendum.json, cost.json.
The grammar of the expected keys is code/tools/rv3_grammar.json (written once by --write-grammar after a complete run; make_xmacros.py checks the summary against it).
Every number is a copy of a field of the input files; no model is called here. A missing input field is an error (no silent skips)."""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BK = ("plus", "flash", "27b")
ARMS = ("zs", "rootonly", "gated", "fixed")
REP = ("101", "202", "303")
SCOPES = ("plus", "flash", "27b", "pooled3")
CIDS = ("stack", "loop_g", "loop_f", "g_minus_f", "g_minus_zs", "f_minus_zs")
SX_CIDS = ("ex_minus_zs", "root_minus_ex", "stack")
SX_ARMS = ("zs", "zsex", "rootonly")
ROUTE_SCORES = ("v3", "route_ca_frozen", "agreement_only", "nonempty_only", "hr_refit_3feat", "hr_refit_no_agreement", "hr_refit_v3_only", "dev_lodo_3feat", "dev_lodo_no_agreement")
ROUTE_DIFFS = ("route_ca_frozen_minus_v3", "hr_refit_no_agreement_minus_v3", "route_ca_frozen_minus_hr_refit_no_agreement", "dev_lodo_no_agreement_minus_v3",
               "dev_lodo_3feat_minus_dev_lodo_no_agreement", "agreement_only_minus_v3", "hr_refit_3feat_minus_v3")
COVS = ("0.4", "0.6", "0.8")
COVK = {"0.4": "40", "0.6": "60", "0.8": "80"}
P5_VARIANTS = (("all", "all"), ("minus_timeout_only", "no_timeout_only"), ("minus_any_timeout", "no_timeout"))
Q_SETS = ("S_same", "other_dev", "outside_release")
Q_ARMS = ("zs", "gated", "fixed")
INC_CFG = {"plus": ("rootonly", "gated", "fixed", "route"), "flash": ("rootonly", "gated", "fixed"), "27b": ("rootonly", "gated", "fixed")}
MDE_CIDS = ("stack", "loop_g", "loop_f", "g_minus_f", "g_minus_zs")
CRITIC_FIELDS = (("n_shared_root", "int"), ("helped", "int"), ("hurt", "int"), ("net_q", "int"), ("net_pp_all", "pp2"), ("p_two", "p"), ("p_one_greater", "p"),
                 ("revised_by_fixed_only", "int"))


class Out(object):
    def __init__(self):
        self.keys = {}
        self.order = []

    def put(self, key, v, kind, ci=None, item_ci=None, n=None, meaning=None):
        if key in self.keys:
            raise ValueError("duplicate key " + key)
        if v is None:
            raise ValueError("null value for " + key)
        e = {"v": v, "kind": kind}
        if ci is not None:
            e["ci"] = list(ci)
        if item_ci is not None:
            e["item_ci"] = list(item_ci)
        if n is not None:
            e["n"] = int(n)
        if meaning:
            e["meaning"] = meaning
        self.keys[key] = e
        self.order.append(key)


def yes(b):
    return "yes" if b else "no"


def reading(lo, hi):
    return "positive" if lo > 0 else ("negative" if hi < 0 else "contains 0")


def ptxt(ci, nd=2):
    return "[%+.*f, %+.*f]" % (nd, ci[0], nd, ci[1])


def load(d, name):
    with open(os.path.join(d, name)) as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------------------------- E-S (replication at T = 0.7)
TXT_REP = "three runs at T 0.7 (seeds 101, 202, 303), reported apart from the headline run (seed 0, T 0), never averaged with it"


def put_contrast(o, p, e, n_q, what, kind="pp2"):
    """one paired contrast of the T = 0.7 replication, with the reading of the seed-0 run beside it."""
    o.put(p + "est", e["est"], kind, ci=e["ci"], item_ci=e["item_ci"], n=n_q,
          meaning="%s: mean over the %s of the paired EX difference (pp); cluster interval (databases) and item-level interval of the run-averaged per-question correctness" % (what, TXT_REP))
    for s in REP:
        o.put(p + "s" + s, e["per"][s], kind)
    o.put(p + "sd", e["sd"], "x2", meaning="SD over the three per-run estimates (pp)")
    o.put(p + "range", tuple(e["range"]), kind, meaning="between-run range of the three per-run estimates")
    o.put(p + "t_ci", tuple(e["t_ci"]), kind, meaning="t interval (2 df) of the three per-run estimates")
    o.put(p + "wide", tuple(e["wide"]), kind, meaning="hull of the cluster interval, the item-level interval, the between-run range and the t interval")
    r0, ra = reading(*e["s0"]["ci"]), reading(*e["ci"])
    r0w, raw = reading(*e["s0"]["wide"]), reading(*e["wide"])
    o.put(p + "s0", e["s0"]["est"], kind, meaning="the headline run (seed 0, T 0), same contrast (pp); not part of the replication mean")
    o.put(p + "s0_ci", tuple(e["s0"]["ci"]), kind, meaning="cluster interval of the headline run")
    o.put(p + "s0_wide", tuple(e["s0"]["wide"]), kind, meaning="hull of the cluster and item-level intervals of the headline run")
    o.put(p + "reading", ra, "str", meaning="reading of the replication by its cluster interval (positive / negative / contains 0)")
    o.put(p + "reading_wide", raw, "str", meaning="reading of the replication by the widest interval")
    o.put(p + "reading_s0", r0, "str", meaning="reading of the headline run by its cluster interval")
    o.put(p + "reading_s0_wide", r0w, "str", meaning="reading of the headline run by the hull of its cluster and item-level intervals")
    o.put(p + "agrees", yes(r0 == ra), "str", meaning="the replication reading (cluster interval) is the same word as the headline run's")
    o.put(p + "agrees_wide", yes(r0w == raw), "str", meaning="the same with the widest intervals")


def emit_rep(o, S):
    n_q = S["notes"]["n_qids"]
    for b in BK:
        for a in ARMS:
            e = S["arms"]["%s|%s" % (b, a)]
            if e["n_runs"] != 3 or e["s0"] is None:
                raise ValueError("arm %s %s has %d replication runs (needs 3) or no seed-0 run" % (b, a, e["n_runs"]))
            p = "rep.arm.%s.%s." % (b, a)
            o.put(p + "s0", e["s0"], "pctv", meaning="EX (%%) of %s %s, headline run (seed 0, T 0)" % (b, a))
            for s in REP:
                o.put(p + "s" + s, e["per"][s], "pctv", meaning="EX (%%) of %s %s, seed %s (T 0.7)" % (b, a, s))
            o.put(p + "mean", e["mean"], "pctv", meaning="mean EX (%) over the three T 0.7 runs (seed 0 excluded)")
            o.put(p + "min", e["range"][0], "pctv"); o.put(p + "max", e["range"][1], "pctv")
            o.put(p + "sd", e["sd"], "x2", meaning="SD of EX (pp) over the three T 0.7 runs, ddof 1 (the T 0 run is not pooled)")
            o.put(p + "sd_lo", e["sd_ci"][0], "x2", meaning="lower end of the 95% interval of the SD (chi-square, 2 df)")
            o.put(p + "sd_hi", e["sd_ci"][1], "x2", meaning="upper end of the 95% interval of the SD (chi-square, 2 df)")
            o.put(p + "n_runs", e["n_runs"], "int")
    scopes = [(c, sc) for c in CIDS for sc in SCOPES] + [("w1", "weak2")]
    for cid, sc in scopes:
        e = S["contrasts"]["%s|%s" % (cid, sc)]
        if e["n_runs"] != 3:
            raise ValueError("contrast %s %s has %d runs" % (cid, sc, e["n_runs"]))
        put_contrast(o, "rep.%s.%s." % (cid, sc), e, n_q, "%s on %s" % (cid, sc))
    # P1' and W1 verdict words
    g = S["contrasts"]["g_minus_f|pooled3"]
    o.put("rep.p1p.margin", 1.5, "x1", meaning="non-inferiority margin of P1' (pp)")
    o.put("rep.p1p.margin_ok_cluster", yes(g["ci"][0] > -1.5), "str", meaning="lower end of the cluster interval of gated minus fixed (pooled, T 0.7 replication) is above -1.5 pp")
    o.put("rep.p1p.margin_ok_wide", yes(g["wide"][0] > -1.5), "str", meaning="the same for the lower end of the widest interval")
    o.put("rep.p1p.margin_ok_s0", yes(g["s0"]["ci"][0] > -1.5), "str", meaning="the same for the headline run's cluster interval")
    w = S["contrasts"]["w1|weak2"]
    o.put("rep.w1.claim_ok_cluster", yes(w["ci"][0] > 0), "str", meaning="lower end of the cluster interval of W1 (T 0.7 replication) is above 0")
    o.put("rep.w1.claim_ok_wide", yes(w["wide"][0] > 0), "str", meaning="the same for the widest interval")
    o.put("rep.w1.claim_ok_s0", yes(w["s0"]["ci"][0] > 0), "str", meaning="the same for the headline run's cluster interval")
    for sc in SCOPES:
        e = S["p1p_calls"][sc]
        p = "rep.calls.%s." % sc
        o.put(p + "est", e["est"], "x2", ci=e["ci"], item_ci=e["item_ci"], n=n_q, meaning="calls of the gated loop divided by calls of the fixed loop (ratio of sums; pooled3 = mean of the three ratios), mean over the T 0.7 runs")
        o.put(p + "s0", e["s0"], "x2", meaning="the same ratio in the headline run (seed 0, T 0)")
        for s in REP:
            o.put(p + "s" + s, e["per"][s], "x2")
        o.put(p + "range", tuple(e["range"]), "x2")
    # stack versus loop
    for b in BK:
        e = S["split"][b]
        p = "rep.split.%s." % b
        o.put(p + "stack_minus_loop", e["stack_minus_loop"]["est"], "pp2", ci=e["stack_minus_loop"]["ci"], item_ci=e["stack_minus_loop"]["item_ci"], n=n_q,
              meaning="stack contrast minus gated-loop contrast on %s, mean over the three T 0.7 runs (pp)" % b)
        o.put(p + "stack_minus_loop.wide", tuple(e["stack_minus_loop"]["wide"]), "pp2")
        o.put(p + "stack_minus_loop.s0", e["stack_minus_loop"]["s0"]["est"], "pp2", meaning="the same in the headline run (seed 0, T 0)")
        o.put(p + "stack", e["stack"], "pp2"); o.put(p + "loop_g", e["loop_g"], "pp2"); o.put(p + "total_g", e["total_g"], "pp2")
        o.put(p + "verdict_wide", e["verdict"], "str", meaning="stack larger / loop larger / not separable by the widest interval")
        o.put(p + "verdict_cluster", e["verdict_cluster"], "str")
        o.put(p + "share_defined", yes(e["share_defined"]), "str", meaning="the share of the total gain due to the loop is defined only where the total gain's cluster interval excludes 0")
        o.put(p + "share_loop", 100.0 * e["share_loop"], "pctv", meaning="gated-loop contrast as a percentage of the total (gated minus single call), T 0.7 replication")
        o.put(p + "share_loop_ci", ptxt([100.0 * x for x in e["share_loop_ci"]], 0) if e["share_defined"] else "undefined", "str")
    # regeneration audit (descriptive, per run, no averaging)
    for b in BK:
        for s in ("0",) + REP:
            e = S["regen"]["%s|%s" % (b, s)]
            p = "rep.regen.%s.%s." % (b, s)
            o.put(p + "n_diff_gf", e["diff.gated_fixed"], "int", meaning="questions whose root SQL differs between the gated and the fixed run of this backbone and seed")
            o.put(p + "n_diff_gr", e["diff.gated_rootonly"], "int"); o.put(p + "n_diff_fr", e["diff.fixed_rootonly"], "int")
            o.put(p + "rootex_gf", e["rootex.gated_minus_fixed"], "pp2", meaning="EX of the root candidates, gated minus fixed run (pp)")
            o.put(p + "rootex_gr", e["rootex.gated_minus_rootonly"], "pp2"); o.put(p + "rootex_fr", e["rootex.fixed_minus_rootonly"], "pp2")
            for cid in ("g_minus_f", "loop_g", "loop_f"):
                sh = e["shared." + cid]
                o.put(p + "shared." + cid, sh["est"], "pp2", ci=sh["ci"], item_ci=sh["item_ci"], n=sh["n"], meaning="%s on the questions with the same root SQL in both runs (pp)" % cid)
                o.put(p + "all." + cid, sh["all"], "pp2", meaning="%s on all questions (pp), for comparison" % cid)


# ----------------------------------------------------------------------------------------------------------- E-C
def emit_samex(o, X):
    n_q = X["notes"]["n_qids"]
    for b in BK:
        for a in SX_ARMS:
            e = X["arms"]["%s|%s" % (b, a)]
            if e["n_runs"] != 3 or e["s0"] is None:
                raise ValueError("samex arm %s %s has %d runs" % (b, a, e["n_runs"]))
            p = "samex.arm.%s.%s." % (b, a)
            o.put(p + "s0", e["s0"], "pctv", meaning="EX (%) of the headline run (seed 0, T 0)")
            for s in REP:
                o.put(p + "s" + s, e["per"][s], "pctv")
            o.put(p + "mean", e["mean"], "pctv", meaning="mean EX (%) over the three T 0.7 runs"); o.put(p + "sd", e["sd"], "x2", meaning="SD over the three T 0.7 runs")
    for cid in SX_CIDS:
        for sc in SCOPES:
            e = X["contrasts"]["%s|%s" % (cid, sc)]
            put_contrast(o, "samex.%s.%s." % (cid, sc), e, n_q,
                         {"ex_minus_zs": "single call plus the same examples minus single call", "root_minus_ex": "root-only minus single call plus the same examples",
                          "stack": "root-only minus single call"}[cid] + " (%s)" % sc)
    for sc in SCOPES:
        e = X["share"][sc]
        p = "samex.share.%s." % sc
        o.put(p + "defined", yes(e["defined"]), "str", meaning="defined only where the interval of the stack contrast excludes 0")
        o.put(p + "examples", e["examples"], "pp2"); o.put(p + "stack", e["stack"], "pp2")
        o.put(p + "share", 100.0 * e["share"], "pctv", meaning="examples-only contrast as a percentage of the stack contrast (read only where defined = yes), T 0.7 replication")
        o.put(p + "share_ci", ptxt([100.0 * x for x in e["share_ci"]], 0) if e["defined"] else "undefined", "str")
        o.put(p + "s0_defined", yes(e["s0"]["defined"]), "str", meaning="the same, headline run (seed 0)")
        o.put(p + "s0_share", 100.0 * e["s0"]["share"], "pctv", meaning="examples-only contrast as a percentage of the stack contrast in the headline run (read only where s0_defined = yes)")
    for b in BK:
        for s in ("0",) + REP:
            e = X["discord"]["%s|%s" % (b, s)]
            o.put("samex.discord.%s.%s.rate" % (b, s), e["rate_pct"], "pctv2", meaning="questions where single call plus examples and root-only differ in correctness (%)")


# ----------------------------------------------------------------------------------------------------------- E-R
def emit_retrieval(o, R):
    for b in BK:
        for a in ("rootonly", "gated", "fixed"):
            e = R["by_backbone"][b][a]
            p = "ret.%s.%s." % (b, a)
            o.put(p + "n", e["n"], "int"); o.put(p + "full_schema", e["full_schema"], "int", meaning="questions whose generation prompt carried the whole schema")
            o.put(p + "routed", e["routed"], "int", meaning="questions whose schema was cut down by the schema router")
    mx = max(R["by_backbone"][b][a]["max_ctx_tokens"] for b in BK for a in ("rootonly", "gated", "fixed"))
    o.put("ret.max_ctx_tokens", mx, "int", meaning="largest stored context size (estimated tokens) of any development question in any of these runs")
    o.put("ret.n_examples", R["by_backbone"]["plus"]["rootonly"]["n_examples_max"], "int", meaning="retrieved few-shot examples per question (minimum = maximum in all nine runs)")
    o.put("ret.k_t", R["k_t"], "int", meaning="tables kept by the schema router when it is active")
    o.put("ret.fallback_tokens", R["fallback_tokens"], "int", meaning="estimated tokens below which the whole schema is always passed")
    o.put("ret.dbs_gt_kt", R["dbs_more_than_kt_tables"], "int", meaning="development databases with more tables than k_t")
    o.put("ret.share_q_gt_kt", R["share_questions_dbs_more_than_kt_tables"], "pctv", meaning="share of development questions on those databases (%)")
    o.put("ret.max_db_tokens", R["max_tokens_any_db"], "int", meaning="largest median context size over the 11 development databases (estimated tokens)")
    o.put("ret.stratum_threshold", R["strata_threshold_tokens"], "int")
    for b in BK:
        for st in ("large", "small"):
            e = R["stack_by_stratum"][b][st]
            o.put("ret.stack.%s.%s" % (b, st), e["est"], "pp2", item_ci=e["item_ci"], n=e["n"],
                  meaning="root-only minus single call (seed 0, pp) on the questions of the %s databases (context above/below the threshold); %d databases" % (st, e["n_dbs"]))


def emit_routing(o, T):
    o.put("route.n", T["n"], "int"); o.put("route.ex_rate", T["ex_rate"], "pctv")
    o.put("route.hr_n", T["hr_n"], "int", meaning="questions of the HR run on which the refits are fitted")
    for nm in ROUTE_SCORES:
        e = T["scores"][nm]
        o.put("route.%s.auroc" % nm, e["auroc"], "x3", ci=e["auroc_ci"], n=T["n"])
        for c in COVS:
            o.put("route.%s.acc%s" % (nm, COVK[c]), e["acc_at_cov"][c], "pctv", meaning="EX (%%) of the answers accepted at coverage %s" % c)
    for nm in ROUTE_DIFFS:
        e = T["diff"][nm]
        o.put("route.diff.%s" % nm, e["est"], "x3s", ci=e["ci"], n=T["n"])
    c = T["coef_no_agreement"]
    o.put("route.coef_no_agreement.v3", c["coef"][0], "x3s"); o.put("route.coef_no_agreement.nonempty", c["coef"][1], "x3s")
    o.put("route.coef_no_agreement.intercept", c["intercept"], "x3s")
    V = T["vote"]
    o.put("route.vote.n_with_4_samples", V["n_questions_with_4_samples"], "int", meaning="questions whose four stored sample completions were recovered from the routing run's cache archive")
    o.put("route.vote.ex_final", V["ex_final"], "pctv")
    for v in ("A", "B"):
        e = V["variants"][v]
        p = "route.vote.%s." % v
        o.put(p + "ex", e["ex_vote"], "pctv", meaning="EX (%) of the majority-vote answer: " + e["label"])
        o.put(p + "auroc", e["auroc_vote_share"], "x3", ci=e["auroc_vote_share_ci"], n=T["n"])
        o.put(p + "auroc_route_minus", e["auroc_route_minus_vote"]["est"], "x3s", ci=e["auroc_route_minus_vote"]["ci"], n=T["n"],
              meaning="AUROC of the frozen route score (on the final answers) minus AUROC of the vote share (on the vote answers)")
        for c_ in COVS:
            x = e["cov"][c_]
            k = COVK[c_]
            o.put(p + "acc%s_route" % k, x["acc_route"], "pctv"); o.put(p + "acc%s_vote" % k, x["acc_vote"], "pctv")
            o.put(p + "cpa%s_route" % k, x["correct_per_100_asked_route"], "x1", meaning="correct accepted answers per 100 questions asked at coverage %s, route score" % c_)
            o.put(p + "cpa%s_vote" % k, x["correct_per_100_asked_vote"], "x1")


def emit_p5(o, P):
    for src, nm in P5_VARIANTS:
        e = P[src]
        p = "p5.%s." % nm
        o.put(p + "n", e["n"], "int"); o.put(p + "gated", e["repaired_gated"], "int"); o.put(p + "blind", e["repaired_blind"], "int")
        o.put(p + "rate_gated", e["rate_gated"], "pctv"); o.put(p + "rate_blind", e["rate_blind"], "pctv")
        o.put(p + "diff", e["diff_pp"], "pp2", ci=e["ci"], item_ci=e["item_ci"], n=e["n"], meaning="repair rate with the error text minus blind retry (pp), pooled failing roots")
        o.put(p + "b", e["b"], "int"); o.put(p + "c", e["c"], "int")
        o.put(p + "p_one", e["p_one_greater"], "p", meaning="exact McNemar, one-sided (error text better)"); o.put(p + "p_two", e["p_two"], "p")
    for b in BK:
        o.put("p5.timeout.%s" % b, P["timeout_roots"][b], "int", meaning="failing gated roots whose status is an execution time-out")
        o.put("p5.timeout_only.%s" % b, P["timeout_only_roots"][b], "int", meaning="of those, roots whose SQL executes without error in the fixed run (the failure is the time limit, not the query)")
        e = P["by_backbone"][b]
        o.put("p5.bb.%s.n" % b, e["n"], "int"); o.put("p5.bb.%s.diff" % b, e["diff_pp"], "pp2", item_ci=e["item_ci"])
        o.put("p5.bb.%s.b" % b, e["b"], "int"); o.put("p5.bb.%s.c" % b, e["c"], "int")


# ----------------------------------------------------------------------------------------------------------- E-Q
def emit_q360(o, Q):
    ch = Q["chain"]
    for k in ("dev_questions", "release_rows", "release_distinct_ids", "release_ids_in_dev", "db_discordant_removed", "S_all", "question_changed_in_S_all",
              "evidence_changed_in_S_all", "S_same"):
        o.put("q360.chain." + k, ch[k], "int")
    for k, v in Q["sets"].items():
        o.put("q360.set." + k, v, "int")
    o.put("q360.selection_inputs", "dev rows, corrected-release rows, list of questions whose gold query returns a different result on the dev and the Mini-Dev SQLite file (rescore_corrected.build_sets)", "str",
          meaning="the only inputs of the function that defines the 360 questions; no prediction, no EX label and no model output is an input")
    o.put("q360.selection_uses_outcomes", "no", "str")
    o.put("q360.selection_by_disagreement", "no", "str", meaning="the code that defines the subset does not use model disagreement")
    for b in BK:
        for a in Q_ARMS:
            for st in Q_SETS:
                o.put("q360.ex.%s.%s.%s" % (b, a, st), Q["ex"]["%s|%s" % (b, a)][st], "pctv")
        for st in Q_SETS:
            e = Q["contrast"]["g_minus_f|%s|%s" % (b, st)]
            o.put("q360.g_minus_f.%s.%s" % (b, st), e["est"], "pp2", ci=e["ci"], item_ci=e["item_ci"], n=e["n"], meaning="gated minus fixed (seed 0, original reference, pp) on the %s set (%d databases)" % (st, e["n_dbs"]))
            e = Q["contrast"]["stack|%s|%s" % (b, st)]
            o.put("q360.stack.%s.%s" % (b, st), e["est"], "pp2", ci=e["ci"], item_ci=e["item_ci"], n=e["n"])
            o.put("q360.disc.%s.%s" % (b, st), Q["discordance"][b][st], "pctv", meaning="gated and fixed differ in correctness (%) on the set")
    for sc, e in Q["bound"].items():
        p = "q360.bound.%s." % sc
        o.put(p + "shift_subset", e["shift_subset"], "pp2", ci=e["ci_shift_cluster"], item_ci=e["ci_shift_item"], meaning="shift of gated minus fixed on the 360 questions when the corrected reference replaces the original (pp)")
        o.put(p + "weight", 100.0 * e["weight"], "pctv", meaning="share of the 1 534 development questions in the 360 (%)")
        o.put(p + "full_no_shift_elsewhere", e["full_if_no_shift_elsewhere"], "pp2", meaning="shift on all 1 534 if no other question shifts")
        o.put(p + "full_same_rate", e["full_if_same_rate_elsewhere"], "pp2", meaning="shift on all 1 534 if every other question shifts at the same per-question rate")
        o.put(p + "orig_d0", e["orig_d0"], "pp2"); o.put(p + "corr_d1", e["corr_d1"], "pp2")


# ----------------------------------------------------------------------------------------------------------- addendum
def emit_addendum(o, A, C):
    refs = lambda b, c: ["plus_zs"] + (["rootonly"] if c != "rootonly" else []) + (["self_zs"] if b != "plus" else [])
    for b in BK:
        for c in INC_CFG[b]:
            for r in refs(b, c):
                e = A["inc"]["%s|%s|%s" % (b, c, r)]
                p = "inc.%s.%s.vs_%s." % (b, c, r)
                o.put(p + "dex", e["dex"], "pp2", ci=e["dex_ci"], item_ci=e["dex_item_ci"], n=e["n"], meaning="EX difference (pp), seed 0")
                o.put(p + "dcost_q", e["dcost_q_disc"], "cny4", meaning="extra CNY per question with the cache discount")
                o.put(p + "dcost_q_nodisc", e["dcost_q_nodisc"], "cny4", meaning="extra CNY per question at list price")
                o.put(p + "undefined_reason", e["undefined_reason"], "str", meaning="why the cost per additional correct answer is not interpretable ('none' = EX gain positive with both intervals above 0)")
                if e["ratio_defined"]:
                    o.put(p + "cny_per_correct", e["cny_per_correct_disc"], "cny4", ci=e.get("cny_per_correct_ci_disc"), meaning="extra CNY per additional correct answer (discounted); the interval is shown only where the EX interval excludes 0")
                    o.put(p + "cny_per_correct_nodisc", e["cny_per_correct_nodisc"], "cny4", ci=e.get("cny_per_correct_ci_nodisc"))
                    o.put(p + "cny_1000_per_pp", e["cny_1000_per_pp_disc"], "cny2", meaning="break-even: extra CNY per 1 000 questions per pp of EX (discounted)")
                    o.put(p + "cny_1000_per_pp_nodisc", e["cny_1000_per_pp_nodisc"], "cny2")
    M = A["mde"]
    o.put("mde.n", M["n"], "int"); o.put("mde.G", M["G"], "int", meaning="database clusters")
    o.put("mde.z_sum", M["z_sum"], "x3", meaning="z(0.975) + z(0.80)"); o.put("mde.t_sum", M["t_sum"], "x3", meaning="t(0.975, 10) + t(0.80, 10)")
    for cid in MDE_CIDS:
        for sc in SCOPES:
            e = M["cells"]["%s|%s" % (cid, sc)]
            p = "mde.%s.%s." % (cid, sc)
            o.put(p + "item_pp", e["mde_item_pp"], "x2", meaning="minimum detectable paired EX difference (pp), 80% power, two-sided alpha 0.05, item level: (z + z) * SD of the observed paired difference / sqrt(1534)")
            o.put(p + "cluster_pp", e["mde_cluster_pp"], "x2", meaning="the same with the database as the unit: (t + t) * SE (SD of the cluster bootstrap distribution)")
            o.put(p + "sd_pp", e["sd_pp"], "x2")
            if sc != "pooled3":
                o.put(p + "disc_rate", e["disc_rate_pct"], "pctv2", meaning="share of questions where the two arms differ in correctness (%)")
    o.put("mde.item_min", M["item_min"], "x2"); o.put("mde.item_max", M["item_max"], "x2")
    o.put("mde.cluster_min", M["cluster_min"], "x2"); o.put("mde.cluster_max", M["cluster_max"], "x2")
    m = A["mult"]
    o.put("mult.n_dual", m["n_dual"], "int", meaning="contrasts in the dual-interval rows of the supplement (RV2.dual)")
    o.put("mult.n_loop", m["n_loop"], "int", meaning="loop effects and stack contrasts with the small-cluster readings (RV2.small.loop)")
    o.put("mult.n_confirmatory", m["n_confirmatory"], "int", meaning="size of the confirmatory family P*, Holm")
    for b in BK:
        e = A["critic"][b]
        p = "critic.%s." % b
        for f, kd in CRITIC_FIELDS:
            o.put(p + f, e[f], kd)
        o.put(p + "net_ci_pp_all", tuple(e["net_ci_pp_all"]), "pp2", meaning="exact conditional (Clopper-Pearson on helped / (helped + hurt)) interval of the net gain, pp of all questions")
        t4 = e["table4"]
        for f in ("fixed_helped", "fixed_hurt", "gated_helped", "gated_hurt", "implied_net_q"):
            o.put(p + "table4." + f, t4[f], "int")
        o.put(p + "table4.implied_net_pp", t4["implied_net_pp"], "pp2")
    s = A["stack"]
    for t in ("cl_ci", "item_ci", "t_ci", "sf_ci", "wide_ci"):
        o.put("stack.pooled." + t, tuple(s[t]), "pp2")
    o.put("stack.pooled.wide_lo_x3", s["wide_lo_exact"], "x3s", meaning="exact lower end of the widest interval of the pooled stack contrast (three decimals; the one-decimal print is 0.0)")
    o.put("stack.pooled.wide_hi_x3", s["wide_hi_exact"], "x3s")
    o.put("stack.pooled.est", s["est"], "pp2")
    L = A["latency"]
    o.put("lat.status", "available", "str", meaning="timing logs exist: latency_s per question in every preds.jsonl; RV2.cost.<b>.<cfg>.lat_mean/lat_p50/lat_p95")
    o.put("lat.source", L["source"], "str")
    for b in BK:
        for c in ("rootonly", "gated", "fixed"):
            o.put("lat.ratio.%s.%s" % (b, c), L["ratio"]["%s|%s" % (b, c)], "x2", meaning="mean latency per question divided by that of the single call of the backbone (seed 0)")


def _cny_txt(v):
    return "%.2f" % v if v >= 1 else "%.3f" % v


def emit_aliases(o, A, R, Q, n_total):
    """names used by the round-3 text (internal note RW3 section 4): icer.* (text; a dash where no positive EX difference, 'unbounded' where the interval of the EX difference reaches 0),
    critinc.*, mde.<test>.*, posthoc.n_contrasts, retr.*, sel360.*."""
    dash = "--"
    for b in BK:
        ref = "plus_zs" if b == "plus" else "self_zs"
        for c in ("rootonly", "gated", "fixed"):
            e = A["inc"]["%s|%s|%s" % (b, c, ref)]
            p = "icer.%s.%s.vs_zs." % (b, c)
            for mode, suf in (("disc", "cny_disc"), ("nodisc", "cny_nodisc")):
                dq = e["dcost_q_" + mode]
                lo, hi = e["dex_ci"]
                if e["dex"] > 0 and dq > 0:
                    o.put(p + suf, _cny_txt(dq / (e["dex"] / 100.0)), "str", meaning="CNY per additional correct answer over the single call of the same backbone (%s cache discount)" % ("with" if mode == "disc" else "without"))
                else:
                    o.put(p + suf, dash, "str", meaning="no positive EX difference or no extra cost: the ratio is not defined")
                if hi > 0 and dq > 0:
                    lo_t = _cny_txt(dq / (hi / 100.0))
                    hi_t = "unbounded" if lo <= 0 else _cny_txt(dq / (lo / 100.0))
                    o.put(p + suf + ".ci", "[%s, %s]" % (lo_t, hi_t), "str",
                          meaning="95%% cluster interval of the cost per additional correct answer: [extra cost / upper end, extra cost / lower end] of the EX difference; unbounded above where that interval reaches 0")
                else:
                    lo_t = hi_t = dash
                    o.put(p + suf + ".ci", dash, "str", meaning="upper end of the EX-difference interval is not positive")
                o.put(p + suf + ".ci_lo", lo_t, "str"); o.put(p + suf + ".ci_hi", hi_t, "str")
    e = A["inc"]["plus|gated|plus_zs"]
    o.put("icer.plus.gated.vs_zs.dcost_1000", 1000.0 * e["dcost_q_disc"], "cny2", meaning="extra CNY per 1 000 questions, Plus gated loop over the single call, with the cache discount")
    o.put("icer.plus.gated.vs_zs.extra_correct_1000.ci", (10.0 * e["dex_ci"][0], 10.0 * e["dex_ci"][1]), "int", meaning="95% cluster interval of the extra correct answers per 1 000 questions (10 x the interval of the EX difference in pp)")
    for b in BK:
        c = A["critinc"][b]
        o.put("critinc.%s.d" % b, c["d"], "pp2", meaning="(fixed loop final minus its own root) minus (gated loop final minus its own root), pp: the Critic's own part of the loop effect")
        o.put("critinc.%s.ci" % b, tuple(c["ci"]), "pp2", meaning="95%% cluster interval of critinc.%s.d" % b)
        o.put("critinc.%s.item_ci" % b, tuple(c["item_ci"]), "pp2")
        o.put("critinc.%s.net_q" % b, c["net_q"], "int", meaning="the same in questions (net)")
    M = A["mde_named"]
    o.put("mde.P1p.pp", M["P1p_pp"], "x2", meaning="minimum detectable effect (pp), power 0.80, two-sided alpha 0.05, observed paired variance: P1' (pooled gated minus fixed), n = 1 534")
    o.put("mde.W1.pp", M["W1_pp"], "x2", meaning="the same for W1 (mean of Flash and 27B gated minus single call)")
    o.put("mde.P3.auroc", M["P3_auroc"], "x3", meaning="the same for the AUROC difference of P3 (DeLong standard error of the paired difference)")
    o.put("mde.P5p.pp", M["P5p_pp"], "x2", meaning="the same for the repair-rate difference of P5' (%d failing roots)" % M["P5p_n"])
    o.put("posthoc.n_contrasts", n_total, "int", meaning="post hoc contrasts examined (110 dual rows, 20 loop rows, round-3 contrasts); the confirmatory family P* is separate")
    for b in BK:
        o.put("retr.%s.routed_n" % b, R["by_backbone"][b]["rootonly"]["routed"], "int", meaning="development questions on which the retrieval module cut the schema (root-only run, %s)" % b)
    o.put("retr.plus.routed_share", 100.0 * R["by_backbone"]["plus"]["rootonly"]["routed"] / R["by_backbone"]["plus"]["rootonly"]["n"], "pctv", meaning="the same as a share of the development questions (Plus), %")
    ch = Q["chain"]
    o.put("sel360.rule", ("defined without any model output: the %d questions of the corrected release (all inside the %d development questions), minus %d questions whose gold query returns a different result "
                          "on the development and on the Mini-Dev SQLite file (%d left), keeping those whose question text and evidence text are unchanged (%d)"
                          % (ch["release_distinct_ids"], ch["dev_questions"], ch["db_discordant_removed"], ch["S_all"], ch["S_same"])), "str",
          meaning="selection rule of the 360-question subset (code/tools/rescore_corrected.py build_sets); the function takes the development rows, the corrected-release rows and the list of discordant questions")
    o.put("sel360.pipelines", "no evaluated pipeline or model output enters the rule, and the repository holds no record of how the release's authors chose the %d items; stored correctness is higher inside the subset than outside it (single call on Plus %.1f%% against %.1f%%)"
          % (ch["release_distinct_ids"], Q["ex"]["plus|zs"]["S_same"], Q["ex"]["plus|zs"]["other_dev"]), "str", meaning="which pipelines or models the selection used")


def emit_cost(o, K):
    t = K["total"]
    o.put("cost.total_cny", t["cny_disc"], "cny2", meaning="total cost of the round-3 runs (CNY, cache discount of exp/prices.yaml)")
    o.put("cost.total_cny_nodisc", t["cny_nodisc"], "cny2", meaning="the same at list price")
    o.put("cost.total_calls", t["calls"], "int", meaning="billed chat calls")
    if "s303" in K:
        o.put("cost.s303.cny", K["s303"]["cny_disc"], "cny2", meaning="of which the third T 0.7 replication run (seed 303: 12 decomposition runs and 3 same-examples runs; CNY, cache discount)")
        o.put("cost.s303.calls", K["s303"]["calls"], "int", meaning="billed chat calls of the seed-303 runs")
    for m, short in (("qwen3.7-plus-2026-05-26", "plus"), ("qwen3.7-flash-2026-07-15", "flash"), ("qwen3.5-27b", "27b")):
        o.put("cost.model.%s" % short, K["by_model"][m]["cny_disc"], "cny2")


def build(d, rv2_path=None):
    o = Out()
    S, X = load(d, "rep.json"), load(d, "samex.json")
    R, T, P = load(d, "retrieval.json"), load(d, "routing.json"), load(d, "p5.json")
    Q, A, K = load(d, "q360.json"), load(d, "addendum.json"), load(d, "cost.json")
    emit_rep(o, S); emit_samex(o, X); emit_retrieval(o, R); emit_routing(o, T); emit_p5(o, P); emit_q360(o, Q); emit_addendum(o, A, K); emit_cost(o, K)
    # number of post hoc contrasts added by this round (E-S: 6 + 1 contrasts x scopes; E-C: 3 x 4; E-R/E-Q cells that are contrasts)
    n_new = (len(CIDS) * len(SCOPES) + 1) + len(SX_CIDS) * len(SCOPES) + len(ROUTE_DIFFS) + (len(P5_VARIANTS) - 1) + 2 * len(BK) * len(Q_SETS) + 2 * len(BK) + len(BK)
    m = A["mult"]
    o.put("mult.n_r3", n_new, "int", meaning="post hoc contrasts added in round 3: T 0.7 replication (6 x 4 + W1), same-examples (3 x 4), routing AUROC differences, P5' variants, 360-subset contrasts (gated minus fixed and stack by set), ret strata, critic increments")
    o.put("mult.n_total", m["n_dual"] + m["n_loop"] + n_new, "int", meaning="post hoc contrasts in all result tables, round 3 included (confirmatory family excluded)")
    emit_aliases(o, A, R, Q, m["n_dual"] + m["n_loop"] + n_new)
    o.put("mult.expected_false_pos", 0.05 * (m["n_dual"] + m["n_loop"] + n_new), "x1", meaning="number of nominally significant contrasts expected by chance at alpha 0.05 if all nulls were true (an upper-bound illustration, not a correction)")
    meta = {"scope": "posthoc-secondary", "sap": "docs/SAP.md section 26", "api_calls": K["total"]["calls"], "api_calls_note": "billed chat calls of the round-3 runs (r3_cost.py); the analysis scripts themselves call no model", "B": 10000, "seed_boot": 20261101,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "inputs": sorted(os.listdir(d)), "missing_keys": [], "price_date": "2026-10-26"}
    return {"_meta": meta, "keys": o.keys}, o.order


def fmt_entry(e):
    """typeset-like text of one key (same number formats as make_xmacros.fnum, plain minus)."""
    sys.path.insert(0, HERE)
    import make_xmacros as X
    def one(v, kind):
        if isinstance(v, (list, tuple)):
            return "[%s, %s]" % (one(v[0], kind), one(v[1], kind))
        if kind == "str":
            return str(v)
        return X.fnum(v, kind)
    t = one(e["v"], e["kind"])
    if e.get("ci"):
        t += " cl" + one(e["ci"], e["kind"])
    if e.get("item_ci"):
        t += " it" + one(e["item_ci"], e["kind"])
    return t


def write_tsv(doc, order, path):
    with open(path, "w") as f:
        f.write("key\tkind\tvalue\tn\tmeaning\n")
        for k in order:
            e = doc["keys"][k]
            f.write("RV3.%s\t%s\t%s\t%s\t%s\n" % (k, e["kind"], fmt_entry(e).replace("\t", " "), e.get("n", ""), (e.get("meaning") or "").replace("\t", " ").replace("\n", " ")))


def appendix_md(doc, order):
    """markdown table of every base key with its value, grouped by family (used by --doc to refresh the generated block of internal note R3EXP)."""
    fam = []
    for k in order:
        f = ".".join(k.split(".")[:2]) if k.startswith(("rep.arm", "samex.arm", "rep.split", "rep.regen", "samex.share", "samex.discord", "route.diff", "route.vote")) else k.split(".")[0]
        if not fam or fam[-1][0] != f:
            fam.append((f, []))
        fam[-1][1].append(k)
    out = []
    for f, ks in fam:
        out.append("")
        out.append("#### `RV3.%s.*` (%d keys)" % (f, len(ks)))
        out.append("")
        out.append("| Key | Value |")
        out.append("|---|---|")
        for k in ks:
            out.append("| `RV3.%s` | %s |" % (k, fmt_entry(doc["keys"][k]).replace("|", "\\|")))
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(REPO, "results", "r3"))
    ap.add_argument("--write-grammar", action="store_true")
    ap.add_argument("--doc", help="refresh the generated key table between the markers <!-- RV3-KEYS:BEGIN --> and <!-- RV3-KEYS:END --> of this markdown file and write internal note RV3_keys")
    a = ap.parse_args(argv)
    doc, order = build(a.dir)
    with open(os.path.join(a.dir, "rv3_summary.json"), "w") as f:
        json.dump(doc, f, indent=1)
    if a.doc:
        write_tsv(doc, order, os.path.join(REPO, "results", "r3", "RV3_keys.tsv"))
        txt = open(a.doc).read()
        b, e = "<!-- RV3-KEYS:BEGIN -->", "<!-- RV3-KEYS:END -->"
        i, j = txt.index(b) + len(b), txt.index(e)
        open(a.doc, "w").write(txt[:i] + "\n" + appendix_md(doc, order) + txt[j:])
    if a.write_grammar:
        with open(os.path.join(HERE, "rv3_grammar.json"), "w") as f:
            json.dump([[k, doc["keys"][k]["kind"]] for k in order], f, indent=0)
    print("[r3_merge] %d keys" % len(order))


if __name__ == "__main__":
    main()
