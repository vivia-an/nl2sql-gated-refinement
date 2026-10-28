#!/usr/bin/env python3
"""R4: merge the result files of the round-4 statistics (code/tools/r4_stats.py, internal review note revision list 4 A1-A8) into results/r4/rv4_summary.json, the file that
make_xmacros.py reads (macro prefix `RV4.`). Pure Python (stdlib), runs on the host. Same file format as results/r3/rv3_summary.json.

    python3 code/tools/r4_merge.py [--dir results/r4] [--r3 results/r3] [--write-grammar] [--doc internal note R4A]

Inputs in --dir: widest.json, regen.json, critic_cost.json, margin_rep.json, mde_cluster.json, p5_units.json, silent.json (all from r4_stats.py, no model call); in --r3: addendum.json
(post hoc counts). The grammar of the expected keys is code/tools/rv4_grammar.json (written by --write-grammar after a complete run). Every number is a copy of a field of the input
files; a missing field is an error."""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from r3_merge import Out, load, reading  # noqa: E402

BK = ("plus", "flash", "27b")
CIDS = ("stack", "loop_g", "loop_f", "g_minus_f", "g_minus_zs", "f_minus_zs")
SCOPES = ("plus", "flash", "27b", "pooled3")
SX_CIDS = ("ex_minus_zs", "root_minus_ex", "stack")
P1P_MARGIN = -1.5          # pp, the pre-specified non-inferiority margin of P1'
MEAN_TXT = "round-4 statistics, post hoc, no model call (SAP section 27)"


def yes(b):
    return "yes" if b else "no"


def money(x, d=2):
    return "%.*f" % (d, x)


def iv_money(lo, hi, d=2):
    return "[%s, %s]" % (money(lo, d), "unbounded" if hi is None else money(hi, d))


# ----------------------------------------------------------------------------------------------------------------- A2 one widest-interval definition
def emit_wide(o, W):
    for fam, src, cids in (("rep", W["contrasts"], [(c, s) for c in CIDS for s in SCOPES] + [("w1", "weak2")]), ("samex", W["samex"], [(c, s) for c in SX_CIDS for s in SCOPES])):
        n_ok = n_ok_prev = n_ok_alt = 0
        for cid, sc in cids:
            e = src["%s|%s" % (cid, sc)]
            p = "wide.%s.%s.%s." % (fam, cid, sc)
            rw, sw = e["rep4"]["wide"], e["s0_4"]["wide"]
            o.put(p + "rep_wide", list(rw), "pp2", meaning="replication (T 0.7, run-averaged): hull of the clustered, item-level, cluster-robust t(G-1) and sign-flip intervals, pp; " + MEAN_TXT)
            o.put(p + "s0_wide", list(sw), "pp2", meaning="locked run (T 0): the same hull, pp")
            o.put(p + "rep_t", list(e["rep4"]["t"]), "pp2", meaning="replication: cluster-robust t(G-1) interval, pp")
            o.put(p + "rep_sf", list(e["rep4"]["sf"]), "pp2", meaning="replication: exact sign-flip interval (size-weighted database means), pp")
            o.put(p + "rep_reading", reading(*rw), "str", meaning="reading word of the replication under the single widest definition")
            o.put(p + "s0_reading", reading(*sw), "str", meaning="reading word of the locked run under the single widest definition")
            ag = reading(*rw) == reading(*sw)
            o.put(p + "agrees", yes(ag), "str", meaning="same reading word in both settings under the single widest definition")
            prev = e.get("agrees_old")
            o.put(p + "agrees_prev", yes(bool(prev)), "str", meaning="same flag under the definition of the earlier version (locked run: cluster and item hull; replication: cluster, item, run range, t(2))")
            if fam == "rep":
                # round 7 (revision list 7 D3): the unrounded lower end of the replication interval (the exact sign-flip search leaves about 1e-9 at zero, which the reading rule counts as 0)
                o.put(p + "rep_lo_raw", "%.1e" % rw[0], "str", meaning="unrounded lower end of rep_wide in scientific notation (a sign-flip end at zero is about +-1e-9)")
            if fam == "rep" and (cid, sc) == ("g_minus_f", "pooled3"):
                # round 7 (D9, R2 m7): generation-variance allowance for the replication, same formula as the rerun's (est - sqrt((est - lo)^2 + (1.96 s)^2)), with s the SD of the
                # three per-run estimates of the pooled contrast (a run-level scale, conservative for their mean)
                per = [float(x) for x in e["per"].values()]
                mu = sum(per) / len(per)
                sdr = math.sqrt(sum((x - mu) ** 2 for x in per) / (len(per) - 1))
                lo_a = e["est"] - math.sqrt((e["est"] - rw[0]) ** 2 + (1.96 * sdr) ** 2)
                o.put(p + "rep_run_sd", sdr, "x2", meaning="SD over the three per-run estimates of the contrast (pp)")
                o.put(p + "rep_allow_lo", lo_a, "pp2", meaning="replication: lower end of rep_wide widened in quadrature by 1.96 x rep_run_sd (pp)")
                o.put(p + "rep_allow_clearance", lo_a - P1P_MARGIN, "pp2", meaning="that widened lower end minus the margin of P1' (pp); positive = margin kept")
            n_ok += int(ag)
            # round 8 (V8 M-1): the same count if an end at zero (within 1e-6, the sign-flip tolerance) were read as positive
            alt = lambda w: reading(1e-6 if abs(w[0]) < 1e-6 else w[0], w[1])
            n_ok_alt += int(alt(rw) == alt(sw))
            n_ok_prev += int(bool(prev))
        o.put("wide.%s.n_contrasts" % fam, len(cids), "int", meaning="contrasts compared between the two settings")
        o.put("wide.%s.n_agree" % fam, n_ok, "int", meaning="contrasts whose reading word is the same in both settings (single widest definition)")
        o.put("wide.%s.n_agree_zero_positive" % fam, n_ok_alt, "int", meaning="the same count if an interval end at zero within 1e-6 were read as positive instead of containing 0")
        o.put("wide.%s.n_agree_prev" % fam, n_ok_prev, "int", meaning="the same count under the earlier definition")


# ----------------------------------------------------------------------------------------------------------------- A4 root regeneration
def emit_regen(o, R):
    cells = R["cells"]
    for b in BK:
        e = cells["%s|0" % b]["gated_fixed"]
        p = "regen.s0.%s." % b
        o.put(p + "n_diff", e["n_diff_root"], "int", meaning="questions whose root SQL differs between the gated and the fixed run, locked run")
        o.put(p + "share", 100.0 * e["share_diff_root"], "pctv2", meaning="share of the development questions with a different root between gated and fixed (locked run), percent")
        o.put(p + "net_diff", e["net_q_diff_root"], "int", meaning="net questions (gated right minus fixed right) on the questions with a different root")
        o.put(p + "net_all", e["net_q_all"], "int", meaning="net questions, gated minus fixed, all questions")
        o.put(p + "net_same", e["net_q_same_root"], "int", meaning="net questions on the questions with an identical root")
        o.put(p + "est_all", e["est_all_pp"], "pp2", meaning="gated minus fixed, all questions, pp")
        o.put(p + "est_same", e["est_same_root_pp"], "pp2", meaning="gated minus fixed on the questions with an identical root, pp")
        r = cells["%s|0" % b]["gated_rootonly"]
        o.put("regen.s0.%s.retry_net" % b, r["net_q_all"], "int", meaning="net questions, gated loop minus separate root-only run, locked run (the questions behind the retry's gain)")
        mx = max(cells["%s|%d" % (b, s)]["gated_fixed"]["n_diff_root"] for s in (101, 202, 303))
        o.put("regen.rep.%s.max_diff" % b, mx, "int", meaning="largest number of questions with a different gated/fixed root in one replication run")
    pool_diff = sum(cells["%s|0" % b]["gated_fixed"]["n_diff_root"] for b in ("flash", "27b"))
    pool_net = sum(cells["%s|0" % b]["gated_fixed"]["net_q_diff_root"] for b in ("flash", "27b"))
    o.put("regen.s0.cheap.n_diff", pool_diff, "int", meaning="Flash and 27B together: questions with a different gated/fixed root, locked run")
    o.put("regen.s0.cheap.net_diff", pool_net, "int", meaning="Flash and 27B together: net questions carried by them")
    o.put("regen.s0.cheap.share", 100.0 * pool_diff / (2.0 * R["n_qids"]), "pctv2", meaning="share of the Flash and 27B questions (2 x 1 534) with a different gated/fixed root, percent")


# ----------------------------------------------------------------------------------------------------------------- A1 Critic increment cost
def emit_critic(o, K):
    for b in BK:
        e = K[b]
        p = "ccost.%s." % b
        o.put(p + "dex", e["dex"], "pp2", ci=e["cl"], item_ci=e["item"], n=e["n"], meaning="Critic increment: (fixed final - fixed root) - (gated final - gated root), pp, locked run")
        o.put(p + "dex_wide", list(e["wide"]), "pp2", meaning="widest interval of the Critic increment (cluster, item-level, t(G-1), sign-flip), pp")
        o.put(p + "net_q", e["net_q"], "int", meaning="net questions of the Critic increment")
        o.put(p + "dcost_disc", e["dcost_1000_disc"], "cny2", meaning="extra CNY per 1,000 questions, fixed loop minus gated loop, with the cache discount")
        o.put(p + "dcost_nodisc", e["dcost_1000_nodisc"], "cny2", meaning="the same at full price")
        o.put(p + "extra_correct_1000", e["extra_correct_1000"], "pp", ci=e["extra_correct_1000_ci"], meaning="extra correct answers per 1,000 questions (10 x increment in pp); cluster interval")
        o.put(p + "extra_correct_1000_wide", list(e["extra_correct_1000_wide"]), "pp", meaning="extra correct answers per 1,000, widest interval")
        for nm in ("disc", "nodisc"):
            r0 = e["icer_%s_cl" % nm]
            o.put(p + "icer_" + nm, money(r0["est"]) if r0["defined"] else "undefined", "str",
                  meaning="CNY per extra correct answer of the Critic increment (%s), point estimate; 'undefined' where the increment is not positive" % nm)
            for iv, tag in (("cl", ""), ("wide", "_wide")):
                r = e["icer_%s_%s" % (nm, iv)]
                o.put(p + "icer_%s%s_iv" % (nm, tag), iv_money(r["lo"], r["hi"]) if (r["defined"] and r["lo"] is not None) else "undefined", "str",
                      meaning="interval of CNY per extra correct answer (%s, %s interval of the increment): [price at the largest gain the interval allows, price at the smallest; 'unbounded' if the interval of the increment includes 0]" % (nm, iv))
        o.put(p + "defined", yes(e["icer_disc_cl"]["defined"]), "str", meaning="increment positive and extra cost positive")
        # round 5 (A2): one basis for the price of a step, the difference of the FINAL answers of the two arms (locked run). 'fa' = fixed loop over gated loop (the Critic),
        # 'retry' = gated loop over the separate root-only run (the retry on execution failure)
        for blk, what in (("fa", "Critic: fixed-loop final answer minus gated-loop final answer"), ("retry", "retry: gated-loop final answer minus root-only final answer (separate run)")):
            f = e[blk]
            q = p + blk + "."
            d = 3 if blk == "retry" else 2          # the retry costs a few thousandths of a yuan per extra correct answer
            o.put(q + "dex", f["dex"], "pp2", ci=f["cl"], item_ci=f["item"], n=f["n"], meaning="EX difference of the final answers, " + what + ", pp, locked run")
            o.put(q + "dex_wide", list(f["wide"]), "pp2", meaning="widest interval (cluster, item-level, t(G-1), sign-flip) of " + blk + ".dex")
            o.put(q + "net_q", f["net_q"], "int", meaning="net questions behind " + blk + ".dex (the point estimate rests on this many questions)")
            o.put(q + "dcost_disc", f["dcost_1000_disc"], "cny2", meaning="extra CNY per 1,000 questions, " + what + ", with the cache discount")
            o.put(q + "dcost_nodisc", f["dcost_1000_nodisc"], "cny2", meaning="the same at full price")
            o.put(q + "extra_correct_1000", f["extra_correct_1000"], "pp", ci=f["extra_correct_1000_ci"], meaning="extra correct answers per 1,000 questions on this basis; cluster interval")
            for nm in ("disc", "nodisc"):
                r0 = f["icer_%s_cl" % nm]
                o.put(q + "icer_" + nm, money(r0["est"], d) if r0["defined"] else "undefined", "str",
                      meaning="CNY per extra correct answer, " + what + " (%s), point estimate; 'undefined' where the difference is not positive" % nm)
                for iv, tag in (("cl", ""), ("wide", "_wide")):
                    r = f["icer_%s_%s" % (nm, iv)]
                    o.put(q + "icer_%s%s_iv" % (nm, tag), iv_money(r["lo"], r["hi"], d) if (r["defined"] and r["lo"] is not None) else "undefined", "str",
                          meaning="interval of CNY per extra correct answer on this basis (%s, %s interval of the difference); 'unbounded' where that interval includes 0" % (nm, iv))
                rw = f["icer_%s_wide" % nm]
                o.put(q + "icer_%s_wide_lo" % nm, money(rw["lo"], d) if (rw["defined"] and rw["lo"] is not None) else "undefined", "str",
                      meaning="lower end of the widest interval of the price (%s): the price of the largest gain the data allow; the price is unbounded above where the interval of the difference includes 0" % nm)
            o.put(q + "defined", yes(f["icer_disc_cl"]["defined"]), "str", meaning="difference positive and extra cost positive")


# ----------------------------------------------------------------------------------------------------------------- A3 margin ratio, share kept in the replication
def _finite(iv, what):
    if iv[0] is None or iv[1] is None:
        raise ValueError("%s: open-ended interval %r (the key grammar holds closed intervals only)" % (what, iv))
    return list(iv)


def emit_margin(o, M):
    for tag, e in (("s0", M["s0"]), ("rep", M["rep"])):
        p = "rmargin.%s." % tag
        o.put(p + "ratio", e["margin_over_eff"], "x2", ci=e["margin_over_eff_ci"], item_ci=e["margin_over_eff_item_ci"], n=e["n"], meaning="margin (1.5 pp) divided by the fixed loop's pooled effect (final minus own root, within run)")
        o.put(p + "ratio_wide", _finite(e["margin_over_eff_wide"], p + "ratio_wide"), "x2",
              meaning="widest interval of the ratio: hull of the intervals of the clustered bootstrap, the item-level bootstrap, the cluster-robust t(G-1) and the sign-flip test of the fixed-loop effect, each mapped through 1.5 / x (round 5; the four-interval definition of the Methods)")
        o.put(p + "share", e["share_kept"], "x2", ci=e["share_kept_ci"], item_ci=e["share_kept_item_ci"], n=e["n"], meaning="share of the fixed loop's effect kept by the gated loop (gated effect / fixed effect, within run)")
        o.put(p + "share_wide", _finite(e["share_kept_wide"], p + "share_wide"), "x2",
              meaning="widest interval of the share kept: hull of the clustered and item-level bootstrap intervals and of the test-inversion intervals (cluster-robust t(G-1); exact sign flip) of the ratio gated effect / fixed effect (round 5)")
        o.put(p + "eff_fixed", e["eff_fixed"], "pp2", ci=e["eff_fixed_ci"], item_ci=e["eff_fixed_item_ci"], meaning="fixed loop effect, final minus root of the same run, pooled over backbones, pp")
        o.put(p + "eff_gated", e["eff_gated"], "pp2", ci=e["eff_gated_ci"], item_ci=e["eff_gated_item_ci"], meaning="gated loop effect, final minus root of the same run, pooled over backbones, pp")
        o.put(p + "gap", e["gap"], "pp2", ci=e["gap_ci"], item_ci=e["gap_item_ci"], meaning="gated loop effect minus fixed loop effect, both final minus own root within the run, pooled over backbones, pp: the quantity whose widest interval decides whether a share of 1 is excluded")
        o.put(p + "gap_wide", _finite(e["gap_wide"], p + "gap_wide"), "pp2", meaning="widest interval (four intervals) of the within-run difference gated effect minus fixed effect")
    o.put("rmargin.rep.ratio_range", list(M["rep_range"]["margin_over_eff"]), "x2", meaning="range of the margin ratio over the three replication runs")
    o.put("rmargin.rep.share_range", list(M["rep_range"]["share_kept"]), "x2", meaning="range of the share kept over the three replication runs")


# ----------------------------------------------------------------------------------------------------------------- round 5 (A4) loop share of the gain over the single call, locked run
def emit_loopshare(o, W):
    for b in ("flash", "27b"):
        loop, tot = W["contrasts"]["loop_g|%s" % b]["s0"]["est"], W["contrasts"]["g_minus_zs|%s" % b]["s0"]["est"]
        o.put("loopshare.s0.%s" % b, 100.0 * loop / tot, "pctv", meaning="locked run: the gated loop's contrast (gated minus root-only) as a share of the pipeline's gain over the single call (gated minus single call), point estimate, percent; round 5")


# ----------------------------------------------------------------------------------------------------------------- A6 MDE at question and database level
def emit_mde(o, D):
    o.put("mde4.G", D["G"], "int", meaning="databases (clusters) of the development set")
    o.put("mde4.t_sum", D["t_sum"], "x2", meaning="t(0.975, G-1) + t(0.80, G-1)")
    for t in ("P1p", "W1", "P5p"):
        e = D["tests"][t]
        o.put("mde4.%s.item" % t, e["mde_item_pp"], "x2", meaning="minimum detectable effect, questions (items) as the unit, pp")
        o.put("mde4.%s.cluster" % t, e["mde_cluster_pp"], "x2", meaning="minimum detectable effect, databases as the unit, pp")
        o.put("mde4.%s.n" % t, e["n"], "int", meaning="units at question level")
        o.put("mde4.%s.G" % t, e["G"], "int", meaning="databases contributing")
    e = D["tests"]["P3"]
    o.put("mde4.P3.item", e["mde_item"], "x3", meaning="minimum detectable AUROC difference, DeLong standard error (questions as the unit)")
    o.put("mde4.P3.cluster", e["mde_cluster"], "x3", meaning="minimum detectable AUROC difference, database-clustered bootstrap standard error")


# ----------------------------------------------------------------------------------------------------------------- A7 P5' units and silent errors
def emit_p5(o, P):
    o.put("p5u.n_roots", P["n_roots"], "int", meaning="failing roots pooled over the three backbones")
    o.put("p5u.n_questions", P["n_questions"], "int", meaning="distinct questions among the failing roots")
    for k in ("1", "2", "3"):
        o.put("p5u.fail_on_%s" % k, P["fail_on"][k], "int", meaning="questions whose root fails on %s backbone(s)" % k)
    q = P["question_unit"]
    o.put("p5u.diff", q["diff_pp"], "pp", ci=q["ci"], item_ci=q["item_ci"], n=q["n"], meaning="P5': gated minus blind retry repair rate, one unit per distinct question (mean over the backbones where the root failed), pp")
    o.put("p5u.rate_gated", q["rate_gated"], "pctv", meaning="repair rate with the error text, question as the unit, percent")
    o.put("p5u.rate_blind", q["rate_blind"], "pctv", meaning="repair rate of the blind retry, question as the unit, percent")
    o.put("p5u.b", q["b_questions"], "int", meaning="questions repaired more often with the error text")
    o.put("p5u.c", q["c_questions"], "int", meaning="questions repaired more often without it")
    o.put("p5u.p_two", q["p_two"], "p", meaning="exact two-sided sign test over the discordant questions")


def emit_silent(o, Sx):
    for b in BK:
        for arm in ("zs", "rootonly", "gated"):
            e = Sx["%s|%s" % (b, arm)]
            p = "silent.%s.%s." % (b, arm)
            o.put(p + "rate", e["silent_rate_pct"], "pctv", ci=e["silent_rate_ci"], n=e["n"], meaning="silent-error rate: share of questions answered by an executable query that is wrong, percent")
            o.put(p + "share_wrong", e["silent_share_of_wrong_pct"], "pctv", meaning="silent errors as a share of all wrong answers, percent")
            o.put(p + "reach", e["reach_share_pct"], "pctv", meaning="wrong answers whose query fails to execute (what the gate can reach), as a share of all wrong answers, percent")
            o.put(p + "n_wrong", e["n_wrong"], "int", meaning="wrong answers")
            o.put(p + "n_fail", e["n_exec_fail_wrong"], "int", meaning="wrong answers whose query fails to execute")
    for arm in ("rootonly", "gated"):
        e = Sx["pooled|" + arm]
        o.put("silent.pooled.%s.reach" % arm, e["reach_share_pct"], "pctv", meaning="pooled over the three backbones: wrong answers that fail to execute, percent of wrong answers")
        o.put("silent.pooled.%s.share_wrong" % arm, e["silent_share_of_wrong_pct"], "pctv", meaning="pooled: silent errors, percent of wrong answers")
        o.put("silent.pooled.%s.n_wrong" % arm, e["n_wrong"], "int", meaning="pooled wrong answers")
        o.put("silent.pooled.%s.n_fail" % arm, e["n_exec_fail_wrong"], "int", meaning="pooled wrong answers that fail to execute")


# ----------------------------------------------------------------------------------------------------------------- A5 post hoc count
def _csv_rows(path, keep=None):
    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if keep is None or keep(r)]


def mult_counts(r3, res):
    """the counts of the post hoc contrasts that are not table rows, READ from the result files that hold them (round 5, V5 L-7): replication, same-examples control, routing
    differences, P5' repeats, corrected-subset contrasts (gated minus fixed and stack, 3 backbones x 3 question sets), retrieval strata, Critic rows, Spider-Realistic decomposition rows."""
    c = {}
    c["rep"] = len(_csv_rows(os.path.join(r3, "rep_contrasts.csv")))
    c["samex"] = len(_csv_rows(os.path.join(r3, "samex_contrasts.csv")))
    c["route"] = len(_csv_rows(os.path.join(r3, "routing_diffs.csv")))
    p5 = load(r3, "p5.json")
    c["p5"] = sum(1 for k in ("minus_timeout_only", "minus_any_timeout") if k in p5)
    c["q360"] = len(_csv_rows(os.path.join(r3, "q360_contrasts.csv"), lambda r: r["contrast"] in ("g_minus_f", "stack")))
    c["ret"] = len(_csv_rows(os.path.join(r3, "retrieval_stack_strata.csv")))
    c["critic"] = len(_csv_rows(os.path.join(r3, "addendum_critic.csv")))
    c["spider"] = len(_csv_rows(os.path.join(res, "ext1", "ext1_contrasts.csv"), lambda r: r["contrast"].endswith(("gated_minus_rootonly", "rootonly_minus_zs"))))
    return c


def emit_mult(o, A3, counts):
    m = A3["mult"]
    parts = (("s32", m["n_dual"], "contrasts of the dual-interval table (Table S32)"), ("loop", m["n_loop"], "loop effects with the small-cluster readings"),
             ("rep", counts["rep"], "replication: 6 contrasts x 4 scopes and W1"), ("samex", counts["samex"], "same-examples control: 3 contrasts x 4 scopes"), ("route", counts["route"], "routing AUROC differences"),
             ("p5", counts["p5"], "P5' repeated without time-out roots"), ("q360", counts["q360"], "corrected-subset contrasts: gated minus fixed and stack, 3 backbones, 3 question sets"),
             ("ret", counts["ret"], "retrieval strata: stack on large and small schemas, 3 backbones"), ("critic", counts["critic"], "Critic increment, 3 backbones"),
             ("spider", counts["spider"], "Spider-Realistic decomposition: root-only minus single call and gated minus root-only, 3 backbones"))
    tot = 0
    for k, n, txt in parts:
        o.put("mult.n_" + k, n, "int", meaning=txt)
        tot += n
    prev = tot - counts["spider"]
    o.put("mult.n_round2", m["n_dual"] + m["n_loop"], "int", meaning="post hoc contrasts before the replication (110 table rows and 20 loop effects)")
    o.put("mult.n_round3", prev, "int", meaning="post hoc contrasts after the replication and its controls")
    o.put("mult.n_total", tot, "int", meaning="post hoc contrasts in all result tables (confirmatory family excluded), including the Spider-Realistic decomposition")
    o.put("mult.n_dup", len(SCOPES), "int", meaning="contrasts counted twice: the stack contrast of the same-examples control is the stack contrast of the replication, one per scope (round 5)")
    o.put("mult.n_distinct", tot - len(SCOPES), "int", meaning="post hoc contrasts after removing the stack contrasts that appear in both blocks")
    o.put("mult.expected_false_pos", 0.05 * tot, "x1", meaning="nominally significant contrasts expected by chance at 0.05 if every null held (illustration, not a correction)")


# ----------------------------------------------------------------------------------------------------------------- A15 lock file versions
LOCK_REVS = (("orig", "2d14769"), ("primary", "09a14f0"), ("r3", "fe59b33"))


def lock_versions(show=None):
    """Hash (sha256[:12]) and entry count of exp/dev_lock.json at three committed versions: the 16 entries of the stage-2 registration, the 20 entries at the end
    of the primary runs, and the 28 entries committed with the replication runs; and whether the first 16 entries are unchanged in the later versions."""
    import hashlib
    import subprocess

    def default_show(rev):
        return subprocess.check_output(["git", "show", "%s:exp/dev_lock.json" % rev], cwd=os.path.dirname(os.path.dirname(HERE))).decode("utf-8")
    show = show or default_show
    out = {}
    for tag, rev in LOCK_REVS:
        txt = show(rev)
        out[tag] = {"hash": hashlib.sha256(txt.encode("utf-8")).hexdigest()[:12], "entries": json.loads(txt)["entries"]}
    n0 = len(out["orig"]["entries"])
    out["orig_unchanged"] = all(out[t]["entries"][:n0] == out["orig"]["entries"] for t in ("primary", "r3"))
    return out


def emit_lock(o, L):
    for tag, txt in (("orig", "the stage-2 registration"), ("primary", "the end of the primary runs"), ("r3", "the commit of the replication runs")):
        o.put("lock.hash_" + tag, L[tag]["hash"], "str", meaning="sha256 (first 12 hex) of exp/dev_lock.json at " + txt)
        o.put("lock.n_" + tag, len(L[tag]["entries"]), "int", meaning="entries of exp/dev_lock.json at " + txt)
    o.put("lock.orig_unchanged", yes(L["orig_unchanged"]), "str", meaning="the entries of the stage-2 registration are identical in the later versions of the lock file")


def build(d, r3, res=None):
    o = Out()
    W, R, K = load(d, "widest.json"), load(d, "regen.json"), load(d, "critic_cost.json")
    M, D, P, Sx = load(d, "margin_rep.json"), load(d, "mde_cluster.json"), load(d, "p5_units.json"), load(d, "silent.json")
    A3 = load(r3, "addendum.json")
    emit_wide(o, W); emit_regen(o, R); emit_critic(o, K); emit_margin(o, M); emit_loopshare(o, W); emit_mde(o, D); emit_p5(o, P); emit_silent(o, Sx)
    emit_mult(o, A3, mult_counts(r3, os.path.join(REPO, "results") if res is None else res)); emit_lock(o, lock_versions())
    meta = {"scope": "posthoc-secondary", "sap": "docs/SAP.md section 27", "api_calls": 0, "B": 10000, "seed_boot": 20261101, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "inputs": sorted(os.listdir(d)), "missing_keys": []}
    return {"_meta": meta, "keys": o.keys}, o.order


def write_tsv(doc, order, path):
    sys.path.insert(0, HERE)
    import make_xmacros as X

    def one(v, kind):
        if isinstance(v, (list, tuple)):
            return "[%s, %s]" % (one(v[0], kind), one(v[1], kind))
        return str(v) if kind == "str" else X.fnum(v, kind)
    with open(path, "w") as f:
        f.write("key\tkind\tvalue\tn\tmeaning\n")
        for k in order:
            e = doc["keys"][k]
            t = one(e["v"], e["kind"])
            if e.get("ci"):
                t += " cl" + one(e["ci"], e["kind"])
            if e.get("item_ci"):
                t += " it" + one(e["item_ci"], e["kind"])
            f.write("RV4.%s\t%s\t%s\t%s\t%s\n" % (k, e["kind"], t.replace("\t", " "), e.get("n", ""), (e.get("meaning") or "").replace("\t", " ").replace("\n", " ")))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(REPO, "results", "r4"))
    ap.add_argument("--r3", default=os.path.join(REPO, "results", "r3"))
    ap.add_argument("--write-grammar", action="store_true")
    ap.add_argument("--tsv", default=os.path.join(REPO, "results", "r4", "RV4_keys.tsv"))
    a = ap.parse_args(argv)
    doc, order = build(a.dir, a.r3)
    with open(os.path.join(a.dir, "rv4_summary.json"), "w") as f:
        json.dump(doc, f, indent=1)
    write_tsv(doc, order, a.tsv)
    if a.write_grammar:
        with open(os.path.join(HERE, "rv4_grammar.json"), "w") as f:
            json.dump([[k, doc["keys"][k]["kind"]] for k in order], f, indent=0)
    print("[r4_merge] %d keys" % len(order))


if __name__ == "__main__":
    main()
