#!/usr/bin/env python3
"""X2DS key file (post hoc; non-Qwen backbone and the retry-only control; SAP Section 28). Pure Python (no numpy), no model call.

    python3 code/tools/x2ds_keys.py --write            # results/x2ds/{plus,flash,27b,dsv41flash}/x2ds_summary.json + pooled/x2ds_pooled.json + x2ds_runs.json + cost_*.json
                                                        #   -> results/x2ds/x2ds_keys.json (macro prefix X2DS.) and code/tools/x2ds_grammar.json
    python3 code/tools/x2ds_keys.py --check            # the key file equals what the inputs give, and its keys equal the grammar

Inputs are written by code/tools/x2ds_analyze.py / x2ds_pool.py / x2ds_runmeta.py (run: code/tools/x2ds_run_analysis.sh). Every number of the paper's X2DS block comes from the macros of this file.
Key families (b in plus, flash, 27b, ds; a in zs, zsretry, gated, fixed, rootonly; c = contrast id, e.g. zsretry_minus_zs; pool in q3 = Plus + Flash + 27B, w2 = Flash + 27B):
  <b>.arm.<a>.*      EX (%), calls per question, tokens per question, CNY per 1,000 questions at list price and with the cache discount, executable share, timeouts
  <b>.ctr.<c>        paired EX difference (pp) with its cluster interval (.ci) and item-level interval (.item_ci); .t CR1 t(G-1); .sf exact sign-flip; .wide hull of the four; .reading, .reading_cluster;
                     .b / .c discordant counts (b: first arm only right), .p exact McNemar p
  <b>.retry.*        what the retry did (questions revised, revisions, repaired, identical SQL with another correctness)
  <b>.calls_ratio_gf calls of the gated loop over the fixed loop with the cluster interval
  pool.<pool>.<c>.*  the same for the pooled difference (mean over the backbones of the per-question difference); pool.<pool>.share_retry
  n.<c>.*            counts over the four backbones;  retry.calls_min / calls_max;  meta.*, runs.*, cost.*
  <b>.step.<x>_over_<y>.*  one rung of the decision ladder (post hoc, 2026-10-29): extra CNY per 1,000 questions, CNY per extra correct answer (point, and at the
                     ends of the widest interval), the reading line of the step and whether its gain is detected
  <b>.ladder.*       the default option the ladder rule reaches on backbone b and the price of the next dearer option
The reading follows SUP(0) with the convention of the paper: an interval end that prints as 0.00 counts as containing 0."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(REPO, "results", "x2ds")
OUT = os.path.join(RES, "x2ds_keys.json")
GRAMMAR = os.path.join(HERE, "x2ds_grammar.json")
KINDS = ("pp2", "pctv", "pctv2", "x1", "x2", "x3", "p", "int", "str", "cny2", "tok")
BK = (("plus", "plus"), ("flash", "flash"), ("27b", "27b"), ("ds", "dsv41flash"))
ARMS = ("zs", "zsretry", "gated", "fixed", "rootonly")
CIDS = ("zsretry_minus_zs", "gated_minus_zsretry", "fixed_minus_zsretry", "rootonly_minus_zsretry", "gated_minus_zs", "fixed_minus_zs", "gated_minus_fixed", "gated_minus_rootonly",
        "fixed_minus_rootonly", "rootonly_minus_zs")
POOL_CIDS = ("zsretry_minus_zs", "gated_minus_zsretry", "gated_minus_zs", "fixed_minus_zsretry", "rootonly_minus_zsretry")
SCOPE = "posthoc-secondary"
# W7 (post hoc): three further arms, the rerun of the retry-only arm and of the non-Qwen chain (x2ds_rerun.py), and per step both reading-line
# conventions, the probability of a positive gain, the minimum detectable effect, prices at full price and net-benefit probabilities.
W7_ARMS = ("zsthink", "zsretryempty", "macsql", "gatedempty")
W7_CIDS = ("zsthink_minus_zs", "gated_minus_zsthink", "zsretryempty_minus_zs", "zsretryempty_minus_zsretry", "macsql_minus_zs", "gated_minus_macsql",
            "gated_minus_zsretryempty", "zsthink_minus_zsretryempty",
            "gatedempty_minus_zs", "gatedempty_minus_gated", "gatedempty_minus_zsretryempty", "fixed_minus_gatedempty")   # W8: the pipeline + empty gate
RERUN = os.path.join(RES, "x2ds_rerun.json")
VALUES = (("0.01", "v001"), ("0.1", "v01"), ("1.0", "v1"), ("10.0", "v10"))
# all steps of the evidence table: (x, y); rerun lines exist for the six steps among zs, zsretry, gated, fixed
W7_STEPS = (("zsretry", "zs"), ("gated", "zs"), ("gated", "zsretry"), ("fixed", "zs"), ("fixed", "zsretry"), ("fixed", "gated"),
            ("zsthink", "zs"), ("gated", "zsthink"), ("zsretryempty", "zs"), ("zsretryempty", "zsretry"), ("macsql", "zs"), ("gated", "macsql"),
            ("gated", "zsretryempty"), ("zsthink", "zsretryempty"),
            ("gatedempty", "zs"), ("gatedempty", "gated"), ("gatedempty", "zsretryempty"), ("fixed", "gatedempty"))
RV4T0 = os.path.join(REPO, "results", "r4_t0", "rv4t0_summary.json")
# decision ladder: options ordered by cost (single call < retry-only < gated pipeline < fixed pipeline, on every backbone);
# step x over y with the rerun contrast whose reading line applies (None: no rerun of either arm differs from the stored roots, or no rerun)
LADDER = ("zs", "zsretry", "gated", "fixed")
NAMES = {"zs": "single call", "zsretry": "single call with retry", "gated": "pipeline, gated loop", "fixed": "pipeline, fixed loop (Critic)"}
STEPS = (("zsretry", "zs", None), ("gated", "zs", "g_minus_zs"), ("fixed", "zs", "f_minus_zs"), ("gated", "zsretry", "g_minus_zs"),
         ("fixed", "zsretry", "f_minus_zs"), ("fixed", "gated", "g_minus_f"))


def price(dcost, gain_pp):
    """CNY per extra correct answer = extra CNY per 1,000 questions / extra correct answers per 1,000 (10 x gain in pp); printed like the RV4 prices."""
    if round(gain_pp, 2) <= 0:
        return "undefined"
    if dcost <= 0:
        return "cheaper"   # the dearer-by-EX arm costs no more: no price to pay
    v = dcost / (10.0 * gain_pp)
    return "%.2f" % v if v >= 0.1 else "%.3f" % v


def _nb_flip(c, vk):
    """net-benefit probability of the reversed step is not a function of the stored one (cost sign flips too): not supported."""
    raise ValueError("reverse W7 step needs its own contrast for net benefit")


def reading(lo, hi):
    """SUP(0) with ends read at the printed precision (an end that prints as 0.00 counts as containing 0)."""
    lo, hi = round(lo, 2), round(hi, 2)
    if lo > 0:
        return "positive"
    if hi < 0:
        return "negative"
    return "contains 0"


def expected_keys():
    with open(GRAMMAR) as f:
        g = json.load(f)
    out = [(k, kind) for k, kind in g]
    for k, kind in out:
        if kind not in KINDS:
            raise ValueError("x2ds_grammar.json: unknown kind %r at %s" % (kind, k))
    return out


def _load(path):
    with open(path) as f:
        return json.load(f)


def build(res=RES):
    keys = {}

    def put(k, v, kind, meaning, ci=None, item_ci=None):
        e = {"v": v, "kind": kind, "meaning": meaning}
        if ci is not None:
            e["ci"] = list(ci)
        if item_ci is not None:
            e["item_ci"] = list(item_ci)
        keys[k] = e

    def contrast(prefix, c, label):
        put(prefix, c["est_pp"], "pp2", "paired EX difference (pp), %s; ci = database-clustered bootstrap, item_ci = question-level bootstrap" % label, c["cluster_ci_pp"], c["item_ci_pp"])
        put(prefix + ".t", list(c["t_ci_pp"]), "pp2", "cluster-robust (CR1) t(G-1) interval of the difference, %s" % label)
        put(prefix + ".sf", list(c["signflip_ci_pp"]), "pp2", "exact sign-flip interval (databases weighted by size) of the difference, %s" % label)
        wide = [min(c["cluster_ci_pp"][0], c["item_ci_pp"][0], c["t_ci_pp"][0], c["signflip_ci_pp"][0]), max(c["cluster_ci_pp"][1], c["item_ci_pp"][1], c["t_ci_pp"][1], c["signflip_ci_pp"][1])]
        put(prefix + ".wide", wide, "pp2", "widest interval (hull of cluster, item-level, t(G-1), sign-flip) of the difference, %s" % label)
        put(prefix + ".reading", reading(*wide), "str", "SUP(0) reading under the widest interval, %s" % label)
        put(prefix + ".reading_cluster", reading(*c["cluster_ci_pp"]), "str", "SUP(0) reading under the database-clustered interval, %s" % label)
        return wide

    summ = {}
    for b, d in BK:
        summ[b] = _load(os.path.join(res, d, "x2ds_summary.json"))
    pooled = _load(os.path.join(res, "pooled", "x2ds_pooled.json"))
    runs = _load(os.path.join(res, "x2ds_runs.json"))
    cost1 = _load(os.path.join(res, "cost_X2DS.json"))
    cost2 = _load(os.path.join(res, "cost_X2DSR.json"))
    cost3 = _load(os.path.join(res, "cost_X2DSRR.json"))
    cost4 = _load(os.path.join(res, "cost_W7.json"))
    cost5 = _load(os.path.join(res, "cost_W7RR.json"))
    cost6 = _load(os.path.join(res, "cost_W8.json"))
    rv4t0 = _load(RV4T0)["keys"]
    rerun = _load(RERUN)

    n_pos = {c: {"wide": 0, "cluster": 0} for c in CIDS}
    calls_rt = []
    for b, _ in BK:
        s = summ[b]
        put(b + ".n", s["n_common"], "int", "questions compared (all arms scored), %s" % b)
        for a in ARMS + tuple(x for x in W7_ARMS if x in s["arms"]):
            v = s["arms"][a]
            p = "%s.arm.%s." % (b, a)
            put(p + "ex", v["ex_pct"], "pctv2", "EX (%%), %s, %s" % (b, a))
            put(p + "calls", v["calls_per_q"], "x3", "chat requests per question, %s, %s" % (b, a))
            put(p + "pt", v["prompt_tok_per_q"], "tok", "prompt tokens per question, %s, %s" % (b, a))
            put(p + "ct", v["completion_tok_per_q"], "tok", "completion tokens per question, %s, %s" % (b, a))
            put(p + "cached", v["cached_tok_per_q"], "tok", "recorded cached prompt tokens per question, %s, %s" % (b, a))
            put(p + "cny_list", v["cny_per_1000q_list"], "cny2", "CNY per 1,000 questions, every prompt token at the list price, %s, %s" % (b, a))
            put(p + "cny_disc", v["cny_per_1000q_cache_disc"], "cny2", "CNY per 1,000 questions, list price with cached tokens at 0.2 of the input price, %s, %s" % (b, a))
            put(p + "exec_ok", v["exec_ok_pct"], "pctv2", "share of final queries that execute (%%), %s, %s" % (b, a))
            put(p + "timeouts", v["pred_timeouts"], "int", "final queries that timed out when scored, %s, %s" % (b, a))
            put(p + "lat", v["lat_mean_s"], "x1", "mean summed request latency per question (s), %s, %s" % (b, a))
            put(p + "lat95", v["lat_p95_s"], "x1", "95th percentile of the summed request latency per question (s), %s, %s" % (b, a))
            put(p + "near_limit", v["n_exec_near_limit"], "int", "final queries whose scoring execution took 24 s or more (80%% of the 30 s limit), %s, %s" % (b, a))
            if "n_no_sql" in v:
                put(p + "no_sql", v["n_no_sql"], "int", "questions with no SQL in the answer (empty output after extraction), %s, %s" % (b, a))
                put(p + "not_exec", v["n_not_exec"], "int", "final queries that do not execute (error, timeout or no SQL), %s, %s" % (b, a))
                put(p + "ct_max", v["ct_max"], "int", "largest completion-token count of one question (reasoning included), %s, %s" % (b, a))
        calls_rt.append(s["arms"]["zsretry"]["calls_per_q"])
        for c in CIDS:
            v = s["contrasts"][c]
            p = "%s.ctr.%s" % (b, c)
            wide = contrast(p, v, "%s, %s" % (b, c.replace("_", " ")))
            put(p + ".b", v["discordant_x_only"], "int", "questions right only in the first arm of %s, %s" % (c, b))
            put(p + ".c", v["discordant_y_only"], "int", "questions right only in the second arm of %s, %s" % (c, b))
            put(p + ".p", v["mcnemar_exact_p"]["p_two"], "p", "exact McNemar p (two-sided), %s, %s" % (c, b))
            n_pos[c]["wide"] += 1 if reading(*wide) == "positive" else 0
            n_pos[c]["cluster"] += 1 if reading(*v["cluster_ci_pp"]) == "positive" else 0
        r = s["retry"]
        p = b + ".retry."
        put(p + "n_revised", r["n_revised"], "int", "questions whose root failed the execution gate and was revised, %s" % b)
        put(p + "share_revised", r["share_revised_pct"], "pctv", "share of questions revised (%%), %s" % b)
        put(p + "revisions", r["revisions_total"], "int", "revisions made, %s" % b)
        put(p + "rev_per_revised", r["revisions_per_revised"], "x2", "revisions per revised question, %s" % b)
        put(p + "n_exec_ok_of_revised", r["n_final_exec_ok_of_revised"], "int", "revised questions whose final query executes, %s" % b)
        put(p + "n_correct_of_revised", r["n_correct_of_revised"], "int", "revised questions whose final query is correct, %s" % b)
        put(p + "n_root_fail", r["n_root_exec_fail_zs"], "int", "single-call answers that fail to execute, %s" % b)
        put(p + "n_same_sql", r["n_sql_identical_to_zs"], "int", "questions whose final query text equals the single call's, %s" % b)
        put(p + "n_up_same_sql", r["n_ex_up_same_sql"], "int", "identical query text, scored correct only in the retry run (scoring noise), %s" % b)
        put(p + "n_down_same_sql", r["n_ex_down_same_sql"], "int", "identical query text, scored correct only in the single-call run (scoring noise), %s" % b)
        if "retry_empty" in s:
            r = s["retry_empty"]
            put(b + ".retry_empty.n_revised", r["n_revised"], "int", "questions revised by the empty-gate arm (root failed, timed out or returned an empty result), %s" % b)
            put(b + ".retry_empty.share_revised", r["share_revised_pct"], "pctv", "share of questions revised by the empty-gate arm (%%), %s" % b)
            put(b + ".retry_empty.n_up", r["n_up"], "int", "revised questions correct with the empty gate and wrong in the single call, %s" % b)
            put(b + ".retry_empty.n_down", r["n_down"], "int", "revised questions wrong with the empty gate and correct in the single call, %s" % b)
        if "gated_empty" in s:
            r = s["gated_empty"]
            put(b + ".gated_empty.n_changed", r["n_changed"], "int", "questions whose final query the empty gate changed in the pipeline (W8), %s" % b)
            put(b + ".gated_empty.n_up", r["n_up"], "int", "of these, correct with the empty gate and wrong in the pipeline, %s" % b)
            put(b + ".gated_empty.n_down", r["n_down"], "int", "of these, wrong with the empty gate and correct in the pipeline, %s" % b)
        lad = {}
        for x, y, line_id in STEPS:
            c = s["contrasts"]["%s_minus_%s" % (x, y)] if "%s_minus_%s" % (x, y) in s["contrasts"] else None
            if c is None:       # fixed over gated = minus gated_minus_fixed
                g = s["contrasts"]["%s_minus_%s" % (y, x)]
                c = {k: (0.0 - g[k] if k == "est_pp" else [0.0 - g[k][1], 0.0 - g[k][0]]) for k in ("est_pp", "cluster_ci_pp", "item_ci_pp", "t_ci_pp", "signflip_ci_pp")}
            est = c["est_pp"]
            wide = [min(c[k][0] for k in ("cluster_ci_pp", "item_ci_pp", "t_ci_pp", "signflip_ci_pp")), max(c[k][1] for k in ("cluster_ci_pp", "item_ci_pp", "t_ci_pp", "signflip_ci_pp"))]
            line = rv4t0["ctr.%s.%s.line_indep" % (line_id, b)]["v"] if (line_id and b != "ds") else None
            det = reading(*wide) == "positive" and (line is None or round(est, 2) >= round(line, 2))
            ax, ay = s["arms"][x], s["arms"][y]
            dd = ax["cny_per_1000q_cache_disc"] - ay["cny_per_1000q_cache_disc"]
            dl = ax["cny_per_1000q_list"] - ay["cny_per_1000q_list"]
            p = "%s.step.%s_over_%s." % (b, x, y)
            lab = "%s over %s, %s" % (NAMES[x], NAMES[y], b)
            put(p + "dex", est, "pp2", "EX gain of the step (pp), %s; ci = widest interval" % lab, wide)
            put(p + "dcost_disc", dd, "cny2", "extra CNY per 1,000 questions of the step, cache discount, %s" % lab)
            put(p + "dcost_list", dl, "cny2", "extra CNY per 1,000 questions of the step, every prompt token at the list price, %s" % lab)
            put(p + "price_disc", price(dd, est), "str", "CNY per extra correct answer of the step, cache discount, point estimate; undefined where the gain is not positive, %s" % lab)
            put(p + "price_list", price(dl, est), "str", "CNY per extra correct answer of the step at full price, point estimate, %s" % lab)
            put(p + "price_disc_best", price(dd, wide[1]), "str", "CNY per extra correct answer at the upper end of the widest interval (the lowest price the data allow), cache discount, %s" % lab)
            put(p + "price_disc_worst", price(dd, wide[0]) if round(wide[0], 2) > 0 else "unbounded", "str", "CNY per extra correct answer at the lower end of the widest interval (unbounded where it includes zero), cache discount, %s" % lab)
            put(p + "line", "none" if line is None else "%.2f" % line, "str", "reading line of the step (pp; twice the rerun-change scale, independence formula); none: the step shares stored roots (retry over single call) or the backbone was not rerun (ds), %s" % lab)
            put(p + "detected", "detected" if det else "not detected", "str", "gain of the step detected: widest interval above zero and the estimate at or above its reading line where it has one, %s" % lab)
            lad[(x, y)] = (det, dd, dl, est, wide)
        cur, path = "zs", ["zs"]
        while True:
            up = [x for x in LADDER[LADDER.index(cur) + 1:] if lad[(x, cur)][0]]
            if not up:
                break
            cur = up[0]
            path.append(cur)
        nxt = LADDER[LADDER.index(cur) + 1] if cur != LADDER[-1] else None
        put(b + ".ladder.default", NAMES[cur], "str", "default option the ladder rule reaches, %s (post hoc, hypothesis-generating)" % b)
        put(b + ".ladder.path", ", then ".join(NAMES[a] for a in path), "str", "options passed on the way, %s" % b)
        put(b + ".ladder.next", NAMES[nxt] if nxt else "none", "str", "next dearer option, not taken, %s" % b)
        if nxt:
            det, dd, dl, est, wide = lad[(nxt, cur)]
            put(b + ".ladder.next_dex", est, "pp2", "EX gain of the next dearer option over the default (pp), %s; ci = widest interval" % b, wide)
            put(b + ".ladder.next_price_disc", price(dd, est), "str", "CNY per extra correct answer of the next dearer option over the default, cache discount, %s" % b)
            put(b + ".ladder.next_price_list", price(dl, est), "str", "the same at full price, %s" % b)
            put(b + ".ladder.next_price_best", price(dd, wide[1]), "str", "the same at the upper end of the widest interval (lowest price the data allow), cache discount, %s" % b)
        # ---- W7 evidence table (post hoc): every step x over y
        rr = rerun["backbones"]["dsv41flash" if b == "ds" else b]
        for x, y in W7_STEPS:
            if x not in s["arms"] or y not in s["arms"]:
                continue
            cid = "%s_minus_%s" % (x, y)
            if cid in s["contrasts"]:
                c, sign = s["contrasts"][cid], 1.0
            else:
                c, sign = s["contrasts"]["%s_minus_%s" % (y, x)], -1.0
            def sg(v):
                return 0.0 + sign * v if sign > 0 else 0.0 - v
            def sgi(iv):
                return [sg(iv[0]), sg(iv[1])] if sign > 0 else [sg(iv[1]), sg(iv[0])]
            est = sg(c["est_pp"])
            wide = sgi([min(c[k][0] for k in ("cluster_ci_pp", "item_ci_pp", "t_ci_pp", "signflip_ci_pp")), max(c[k][1] for k in ("cluster_ci_pp", "item_ci_pp", "t_ci_pp", "signflip_ci_pp"))])
            ax, ay = s["arms"][x], s["arms"][y]
            dl = ax["cny_per_1000q_list"] - ay["cny_per_1000q_list"]
            dd = ax["cny_per_1000q_cache_disc"] - ay["cny_per_1000q_cache_disc"]
            p = "%s.w7.%s_over_%s." % (b, x, y)
            lab = "%s over %s, %s" % (x, y, b)
            put(p + "dex", est, "pp2", "EX gain of the step (pp), %s; ci = widest interval" % lab, wide)
            put(p + "reading", reading(*wide), "str", "SUP(0) reading of the step under the widest interval, %s" % lab)
            put(p + "p_pos", c["p_pos_cluster"] if sign > 0 else 1.0 - c["p_pos_cluster"], "x2", "share of database-clustered bootstrap draws with a positive gain, %s" % lab)
            put(p + "mde", c["mde80_pp"], "x1", "minimum detectable effect (pp; 80%% power, two-sided 5%%, database-clustered standard error), %s" % lab)
            put(p + "dcost_list", dl, "cny2", "extra CNY per 1,000 questions at full price, %s" % lab)
            put(p + "dcost_disc", dd, "cny2", "extra CNY per 1,000 questions with the cache discount, %s" % lab)
            put(p + "price_list", price(dl, est), "str", "CNY per extra correct answer at full price (point estimate; undefined where the gain is not positive), %s" % lab)
            put(p + "price_list_best", price(dl, wide[1]), "str", "CNY per extra correct answer at full price at the upper end of the widest interval (lowest price the data allow), %s" % lab)
            put(p + "price_disc", price(dd, est), "str", "CNY per extra correct answer with the cache discount (point estimate), %s" % lab)
            for vk, vname in VALUES:
                q = c["nb_p_list"][vk]
                put(p + "nb_" + vname, q if sign > 0 else _nb_flip(c, vk), "x2", "probability (database-clustered bootstrap) that the net benefit V x gain - extra cost is positive at full price, V = CNY %s per correct answer, %s" % (vk, lab))
            stp = c["same_text_x_only"] if sign > 0 else c["same_text_y_only"]
            stn = c["same_text_y_only"] if sign > 0 else c["same_text_x_only"]
            put(p + "st_flip_for", stp, "int", "questions with identical final SQL text in both arms, scored correct only in the dearer arm (re-execution flips), %s" % lab)
            put(p + "st_flip_against", stn, "int", "questions with identical final SQL text in both arms, scored correct only in the cheaper arm, %s" % lab)
            put(p + "st_dex", sg(c["st_est_pp"]), "pp2", "EX gain with identical SQL text scored identically (pp), %s; ci = widest interval" % lab, sgi(c["st_wide_ci_pp"]))
            put(p + "st_reading", reading(*sgi(c["st_wide_ci_pp"])), "str", "SUP(0) reading of the same-text gain under the widest interval, %s" % lab)
            put(p + "st_up", c["st_x_only"] if sign > 0 else c["st_y_only"], "int", "questions right only in the dearer arm, identical SQL text excluded, %s" % lab)
            put(p + "st_down", c["st_y_only"] if sign > 0 else c["st_x_only"], "int", "questions right only in the cheaper arm, identical SQL text excluded, %s" % lab)
            st = rr["step"].get("%s_over_%s" % (x, y))
            if st is not None:
                li, ld = st["line"], st["line_direct"]
                pos = reading(*wide) == "positive"
                di, dr = pos and round(est, 2) >= round(li, 2), pos and round(est, 2) >= round(ld, 2)
                put(p + "line_indep", li, "x2", "reading line (pp), twice the rerun-change scale with the arms treated as independent, %s" % lab)
                put(p + "line_direct", ld, "x2", "reading line (pp), twice the rerun change of the contrast itself (direct scale), %s" % lab)
                put(p + "det_indep", "detected" if di else "not detected", "str", "gain detected: widest interval above zero and estimate at or above the independence line, %s" % lab)
                put(p + "det_direct", "detected" if dr else "not detected", "str", "gain detected: widest interval above zero and estimate at or above the direct line, %s" % lab)
                put(p + "det", ("detected" if di else "not detected") if di == dr else "depends on the line", "str",
                    "verdict that holds under both line conventions (detected / not detected), else 'depends on the line', %s" % lab)
                put(p + "det_s", ("yes" if di else "no") if di == dr else "line", "str", "short verdict for tables: yes (detected under both lines), no, line (depends on the line), %s" % lab)
                put(p + "r1", st["r1"], "pp2", "the step in the rerun alone (pp), %s; ci = widest interval of the rerun" % lab, st["r1_wide"])
                put(p + "r1_reading", st["r1_reading_wide"], "str", "SUP(0) reading of the step in the rerun under the widest interval, %s" % lab)
            else:
                put(p + "line_indep", "none", "str", "reading line: none, the dearer arm was not rerun, %s" % lab)
                put(p + "det", "detected" if reading(*wide) == "positive" else "not detected", "str", "interval condition only (the arm was not rerun), %s" % lab)
                put(p + "det_s", "yes" if reading(*wide) == "positive" else "no", "str", "short verdict for tables, interval condition only (the arm was not rerun), %s" % lab)
        for k, v in rr["arm"].items():
            put("%s.rerun.%s.sd_net" % (b, k), v["sd_net"], "x2", "run-to-run SD (pp) of the net EX difference between the locked run and the rerun, arm %s, %s" % (k, b))
            put("%s.rerun.%s.ex_r1" % (b, k), v["ex_r1"], "pctv2", "EX of the rerun (%%), arm %s, %s" % (k, b))
        for cid in W7_CIDS:
            if cid in s["contrasts"]:
                v = s["contrasts"][cid]
                p2 = "%s.ctr.%s" % (b, cid)
                contrast(p2, v, "%s, %s" % (b, cid.replace("_", " ")))
                put(p2 + ".b", v["discordant_x_only"], "int", "questions right only in the first arm of %s, %s" % (cid, b))
                put(p2 + ".c", v["discordant_y_only"], "int", "questions right only in the second arm of %s, %s" % (cid, b))
                put(p2 + ".p", v["mcnemar_exact_p"]["p_two"], "p", "exact McNemar p (two-sided), %s, %s" % (cid, b))
        cr = s["calls_ratio_gated_over_fixed"]
        put(b + ".calls_ratio_gf", cr["est"], "x2", "calls of the gated loop over calls of the fixed loop, %s; ci = database-clustered" % b, cr["cluster_ci"])

    for c in CIDS:
        put("n.%s.pos_wide" % c, n_pos[c]["wide"], "int", "backbones (of 4) where %s is positive under the widest interval" % c)
        put("n.%s.pos_cluster" % c, n_pos[c]["cluster"], "int", "backbones (of 4) where %s is positive under the database-clustered interval" % c)
    gains = [summ[b]["contrasts"]["zsretry_minus_zs"]["est_pp"] for b, _ in BK]
    put("retry.gain_min", min(gains), "pp2", "smallest retry-only gain over the single call over the four backbones (pp)")
    put("retry.gain_max", max(gains), "pp2", "largest retry-only gain over the single call over the four backbones (pp)")
    ge = [summ[b]["contrasts"]["zsretryempty_minus_zs"]["est_pp"] for b, _ in BK if "zsretryempty_minus_zs" in summ[b]["contrasts"]]
    if ge:
        put("retry_empty.gain_min", min(ge), "pp2", "smallest gain of the single call + retry with the empty gate over the single call, over the four backbones (pp)")
        put("retry_empty.gain_max", max(ge), "pp2", "largest gain of the single call + retry with the empty gate over the single call, over the four backbones (pp)")
    for cid, name, lab in (("gatedempty_minus_gated", "empty_in_pipeline", "the pipeline + empty gate over the pipeline"),
                           ("gatedempty_minus_zsretryempty", "fe_empty", "the pipeline + empty gate over the empty-gate retry (front end with the empty trigger held fixed)"),
                           ("gated_minus_zsretry", "fe_err", "the pipeline over the single call + retry (front end with the error-only trigger held fixed)")):
        gv = [summ[b]["contrasts"][cid]["est_pp"] for b, _ in BK if cid in summ[b]["contrasts"]]
        put("%s.gain_min" % name, min(gv), "pp2", "smallest gain of %s over the four backbones (pp)" % lab)
        put("%s.gain_max" % name, max(gv), "pp2", "largest gain of %s over the four backbones (pp)" % lab)
    dg = summ["ds"]["contrasts"]["gated_minus_fixed"]
    dwide_lo = min(dg["cluster_ci_pp"][0], dg["item_ci_pp"][0], dg["t_ci_pp"][0], dg["signflip_ci_pp"][0])
    put("ds.margin_cluster_met", "yes" if round(dg["cluster_ci_pp"][0], 2) > -1.5 else "no", "str", "lower end of the clustered interval of gated minus fixed (DeepSeek) above the margin -1.5 pp")
    put("ds.margin_wide_met", "yes" if round(dwide_lo, 2) > -1.5 else "no", "str", "lower end of the widest interval of gated minus fixed (DeepSeek) above the margin -1.5 pp")
    put("retry.calls_min", min(calls_rt), "x3", "smallest calls per question of the retry-only arm over the four backbones")
    put("retry.calls_max", max(calls_rt), "x3", "largest calls per question of the retry-only arm over the four backbones")

    n_pb = len(BK) * len(CIDS)
    n_pool = len(pooled["pools"]) * len(POOL_CIDS)
    dup_b = ("gated_minus_zs", "fixed_minus_zs", "gated_minus_fixed", "gated_minus_rootonly", "fixed_minus_rootonly", "rootonly_minus_zs")     # contrasts of the locked run already counted (Qwen backbones)
    n_dup = 3 * len(dup_b) + len([1 for c in POOL_CIDS if c == "gated_minus_zs"]) * len(pooled["pools"])
    put("n.contrasts_total", n_pb + n_pool, "int", "contrasts of the X2DS block: ten per backbone on four backbones and five per pool on two pools")
    put("n.contrasts_dup", n_dup, "int", "of these, contrasts of the locked run on the three Qwen backbones (and the pooled gated minus single call) that are already counted among the post hoc contrasts of the locked run")
    put("n.contrasts_new", n_pb + n_pool - n_dup, "int", "contrasts added to the post hoc count (all contrasts of the retry-only arm, and all six non-retry contrasts on the non-Qwen backbone)")
    n_w7 = sum(1 for b, _ in BK for c in W7_CIDS if c in summ[b]["contrasts"])
    put("n.contrasts_w7", n_w7, "int", "contrasts of the further arms (thinking-on single call, empty-gate retry, MAC-SQL on Flash and 27B, pipeline + empty gate), added to the post hoc count")
    rv4 = _load(os.path.join(REPO, "results", "r4", "rv4_summary.json"))
    w9p0 = os.path.join(REPO, "results", "x2ds", "w9_harm.json")
    n_w9 = 4 * len(_load(w9p0)["backbones"]) if os.path.exists(w9p0) else 0      # W9: the four contrasts per backbone printed with intervals in Table S57
    put("n.contrasts_w9", n_w9, "int", "contrasts of the tuning-split run W9 printed with intervals in Table S57 (four per backbone), added to the post hoc count")
    n_all = rv4["keys"]["mult.n_distinct"]["v"] + (n_pb + n_pool - n_dup) + n_w7 + n_w9
    put("n.mult_total", n_all, "int", "distinct post hoc contrasts of the paper: the distinct RV4 count, the X2DS block, the W7 / W8 arms and W9 (confirmatory family excluded)")
    put("n.mult_expected", "%.1f" % (0.05 * n_all), "str", "expected number of intervals excluding zero by chance if every null held (5 per cent of all post hoc contrasts)")
    put("pool.n_dbs", pooled["n_dbs"], "int", "databases")
    for pid, pp in pooled["pools"].items():
        for c in POOL_CIDS:
            contrast("pool.%s.%s" % (pid, c), pp["contrasts"][c], "pooled over %s, %s" % ("/".join(pp["backbones"]), c.replace("_", " ")))
        share = 100.0 * pp["contrasts"]["zsretry_minus_zs"]["est_pp"] / pp["contrasts"]["gated_minus_zs"]["est_pp"]
        put("pool.%s.share_retry" % pid, share, "pctv", "point estimate of the retry-only difference as a share of the gated-loop difference over the single call (%%), pooled over %s; no interval" % "/".join(pp["backbones"]))

    b0 = summ["ds"]
    put("meta.ds.model_id", "deepseek-v4.1-flash-1017", "str", "model identifier of the non-Qwen backbone (Alibaba Cloud Model Studio)")
    for b, d in BK:
        pr = summ[b]["price_cny_per_mtok"]
        put("meta.price.%s.in" % b, pr["in"], "cny2", "list price, CNY per million input tokens, %s" % b)
        put("meta.price.%s.out" % b, pr["out"], "cny2", "list price, CNY per million output tokens, %s" % b)
    put("meta.cache_ratio", b0["cache_ratio"], "x1", "price of a recorded cached prompt token as a share of the input price (assumption)")
    put("meta.B", b0["B"], "int", "bootstrap resamples")
    rs = _load(os.path.join(res, "x2ds_rescore.json"))   # code/tools/x2ds_rescore.py: corrected references, 360-question subset
    put("rescore.n", rs["_meta"]["n_S_same"], "int", "questions of the corrected-reference subset (S_same of variant G1)")
    for b, d in BK:
        for cid, v in rs["backbones"][d]["contrasts"].items():
            p = "%s.rescore.%s." % (b, cid)
            put(p + "orig", v["orig"], "pp2", "contrast on the corrected-reference subset with the original references (pp), %s, %s" % (cid, b), v["orig_ci"])
            put(p + "corr", v["corrected"], "pp2", "contrast on the corrected-reference subset with the corrected references (pp; database-clustered CI), %s, %s" % (cid, b), v["corrected_ci"])
            put(p + "shift", v["shift"], "pp2", "shift corrected minus original (pp; database-clustered CI), %s, %s" % (cid, b), v["shift_ci"])
    sh = [v["shift"] for b, d in BK for cid, v in rs["backbones"][d]["contrasts"].items() if cid == "zsretryempty_minus_zs"]
    put("rescore.empty_shift_min", min(sh), "pp2", "smallest shift of the empty-gate retry over the single call under corrected references, four backbones (pp)")
    put("rescore.empty_shift_max", max(sh), "pp2", "largest shift of the empty-gate retry over the single call under corrected references, four backbones (pp)")
    eg = _load(os.path.join(res, "empty_gold.json"))     # code/tools/x2ds_empty_gold.py
    put("meta.dev.gold_empty", eg["gold_empty"], "int", "BIRD development questions whose reference result has no rows")
    put("meta.dev.gold_null_row", eg["gold_one_null_row"], "int", "BIRD development questions whose reference result is one row of NULLs (empty for the gate)")
    wr = _load(os.path.join(res, "w8_roots.json"))      # code/tools/x2ds_w8_roots.py: where the losses of the empty trigger come from
    for bk, b in (("plus", "plus"), ("flash", "flash"), ("27b", "27b"), ("dsv41flash", "ds")):
        r = wr["backbones"][bk]
        put("%s.w8roots.n_regen" % b, r["n_regen"], "int", "questions whose root differs between the pipeline and the pipeline + empty gate (request answered afresh), %s" % b)
        put("%s.w8roots.n_down" % b, r["n_down"], "int", "changed final queries turned wrong by the pipeline + empty gate, %s" % b)
        put("%s.w8roots.n_down_regen" % b, r["n_down_regen"], "int", "of these, questions with a regenerated root, %s" % b)
        put("%s.w8roots.n_same_root" % b, r["n_same_root"], "int", "questions with the same root in both runs, %s" % b)
        put("%s.w8roots.gain_same_root" % b, r["gain_same_root"], "pp2", "gain of the pipeline + empty gate over the pipeline on the questions with the same root (pp), %s" % b)
        put("%s.w8roots.lat_samecalls" % b, r["lat_delta_samecalls"], "x2", "mean latency change (s) of the pipeline + empty gate over the pipeline on questions whose request count did not change (run conditions), %s" % b)
        put("%s.w8roots.retry_down" % b, len(r["empty_vs_retry_down_qids"]), "int", "changed final queries turned wrong by the empty-gate retry against the single call + retry, %s" % b)
    q3 = [wr["backbones"][x]["lat_delta_samecalls"] for x in ("plus", "flash", "27b")]
    put("w8roots.lat_same_qwen_min", min(q3), "x1", "smallest latency change (s) on questions with unchanged request count, Qwen backbones")
    put("w8roots.lat_same_qwen_max", max(q3), "x1", "largest latency change (s) on questions with unchanged request count, Qwen backbones")
    put("w8roots.down_outside_regen", sum(r["n_down"] - r["n_down_regen"] for r in wr["backbones"].values()), "int", "questions turned wrong by the pipeline + empty gate whose root was the same in both runs, four backbones")
    put("w8roots.regen_min", min(r["n_regen"] for r in wr["backbones"].values()), "int", "fewest regenerated roots over the four backbones")
    put("w8roots.regen_max", max(r["n_regen"] for r in wr["backbones"].values()), "int", "most regenerated roots over the four backbones")
    put("w8roots.gain_same_min", min(r["gain_same_root"] for r in wr["backbones"].values()), "pp2", "smallest same-root gain of the pipeline + empty gate over the pipeline (pp)")
    put("w8roots.gain_same_max", max(r["gain_same_root"] for r in wr["backbones"].values()), "pp2", "largest same-root gain of the pipeline + empty gate over the pipeline (pp)")
    w9p = os.path.join(res, "w9_harm.json")          # code/tools/x2ds_w9.py (SAP Section 30): harm of the empty trigger on the tuning split HR
    if os.path.exists(w9p):
        w9 = _load(w9p)
        put("w9.n", w9["n"], "int", "HR questions in W9 (seven databases; bike_share_1 left out)")
        put("w9.n_db", w9["n_db"], "int", "HR databases in W9")
        cw9 = _load(os.path.join(res, "cost_W9.json"))["total"]
        put("w9.cost.calls", cw9["calls"], "int", "billed chat requests of W9 (twelve runs)")
        put("w9.cost.cny_disc", cw9["cny_disc"], "cny2", "CNY of W9, cached tokens at the discount factor")
        put("w9.cost.cny_list", cw9["cny_nodisc"], "cny2", "CNY of W9, every prompt token at list price")
        put("w9.gold_empty", w9["gold_empty"] + w9["gold_null_row"], "int", "W9 questions whose reference result is empty or one row of NULLs")
        put("w9.gold_fail", w9["gold_fail"], "int", "W9 reference queries that did not finish in the emptiness check (120 s; counted as non-empty)")
        # counts by trigger set and SQL identity from x2ds_w9_iv.py. Harm = a correct empty answer rewritten into a wrong query;
        # an unchanged query scored differently on two executions is a flip (inside or outside the set), not harm.
        ivall = _load(os.path.join(res, "w9_iv.json"))["backbones"]
        for bk, r in w9["backbones"].items():
            iv0 = ivall[bk]; d = iv0["decomp"]; g = lambda k, f: d.get(k, {}).get(f, 0)
            r["harm_as_scored"] = r["harm"]
            r["harm"] = iv0["harm_rewritten"]
            r["up_fired"], r["down_fired"] = g("in_chg", "up"), g("in_chg", "down")
            r["flips_same_sql_out"] = g("out_same", "up") + g("out_same", "down") + g("in_same", "up") + g("in_same", "down")
            r["flips_in"] = g("in_same", "up") + g("in_same", "down")
            r["flips_up"], r["flips_down"] = g("out_same", "up") + g("in_same", "up"), g("out_same", "down") + g("in_same", "down")
            r["out_chg_up"], r["out_chg_down"] = g("out_chg", "up"), g("out_chg", "down")
            r["net_fired_pp"] = 100.0 * (r["up_fired"] - r["down_fired"]) / w9["n"]
            r["harm_per_100_correct_empty"] = 100.0 * r["harm"] / r["right_r_empty_ref"]
            u = r["up_fired"] / float(r["n_fired"] - r["right_r_empty_ref"]); h = r["harm"] / float(r["right_r_empty_ref"])
            r["breakeven_pct"] = 100.0 * u / (u + h)          # share of correct answers among the model's empty answers at which repairs and harm balance
        for bk, b in (("plus", "plus"), ("flash", "flash"), ("27b", "27b"), ("dsv41flash", "ds")):
            r = w9["backbones"][bk]
            put("%s.w9.out_chg" % b, "%d / %d" % (r["out_chg_up"], r["out_chg_down"]), "str", "questions outside the trigger set whose final query changed (a root timed out when executed again), turned correct / wrong, HR (W9), %s" % b)
            put("%s.w9.flips_ud" % b, "%d / %d" % (r["flips_up"], r["flips_down"]), "str", "questions whose unchanged final query was scored differently (inside and outside the trigger set), turned correct / wrong, HR (W9), %s" % b)
            put("%s.w9.flips_in" % b, r["flips_in"], "int", "questions inside the trigger set whose unchanged query was scored differently, HR (W9), %s" % b)
            put("%s.w9.breakeven" % b, r["breakeven_pct"], "int", "share (%%) of correct answers among the model's empty answers at which the trigger's repairs and harm balance, HR (W9), %s" % b)
            for f, kind, m in (("ex_zs", "pctv2", "EX (%) of the single call"), ("ex_zsretry", "pctv2", "EX (%) of the single call + retry"),
                               ("ex_zsretryempty", "pctv2", "EX (%) of the empty-gate retry"), ("harm", "int", "questions with an empty reference right under the single call + retry and wrong under the empty-gate retry"),
                               ("harm_vs_zs", "int", "questions with an empty reference right under the single call and wrong under the empty-gate retry"),
                               ("repair_empty_ref", "int", "questions with an empty reference wrong under the single call + retry and right under the empty-gate retry"),
                               ("right_r_empty_ref", "int", "questions with an empty reference answered correctly by the single call + retry"),
                               ("correct_empty_kept", "int", "of these, questions where the empty-gate retry kept the query unchanged"),
                               ("up_er", "int", "questions turned correct by the empty-gate retry against the single call + retry"),
                               ("down_er", "int", "questions turned wrong by the empty-gate retry against the single call + retry"),
                               ("down_er_outside_empty_ref", "int", "of these, questions whose reference is not empty"),
                               ("n_fired", "int", "questions whose plain-retry answer is empty, where the empty trigger acts"),
                               ("up_fired", "int", "of these, questions whose query the trigger rewrote and that turned correct"),
                               ("down_fired", "int", "of these, questions whose query the trigger rewrote and that turned wrong"),
                               ("up_out", "int", "questions outside the trigger's reach turned correct (scoring of the same query, root timeouts)"),
                               ("down_out", "int", "questions outside the trigger's reach turned wrong (scoring of the same query, root timeouts)"),
                               ("flips_same_sql_out", "int", "questions whose final query is the same in both arms but scored differently (inside and outside the trigger set)"),
                               ("net_fired_pp", "pp2", "net effect of the empty trigger on the questions where it acts, in pp of all W9 questions"),
                               ("up_ez", "int", "questions turned correct by the empty-gate retry against the single call"),
                               ("down_ez", "int", "questions turned wrong by the empty-gate retry against the single call")):
                put("%s.w9.%s" % (b, f), r[f], kind, "%s, HR (W9), %s" % (m, b))
            # intervals: code/tools/x2ds_w9_iv.py -> results/x2ds/w9_iv.json; ci = the widest of the four intervals (B = 10,000), the paper's reading rule
            iv = _load(os.path.join(res, "w9_iv.json"))["backbones"][bk]["contrasts"]
            for cid, short, m in (("zsretryempty_minus_zsretry", "er", "the empty-gate retry minus the single call + retry"), ("zsretryempty_minus_zs", "ez", "the empty-gate retry minus the single call"),
                                  ("zsretry_minus_zs", "rz", "the single call + retry minus the single call")):
                c = iv[cid]
                put("%s.w9.dex_%s" % (b, short), c["est_pp"], "pp2", "EX of %s (pp), HR (W9), %s; ci = widest of the four intervals" % (m, b), c["wide_ci_pp"])
                put("%s.w9.dex_%s.reading" % (b, short), c["reading_wide"], "str", "reading of the widest interval, %s, HR (W9), %s" % (m, b))
                put("%s.w9.dex_%s.mde" % (b, short), c["mde80_pp"], "x2", "minimum detectable effect (pp; 80%% power, clustered standard error), %s, HR (W9), %s" % (m, b))
                st = c["same_text"]
                put("%s.w9.st_%s" % (b, short), st["est_pp"], "pp2", "EX of %s with identical final SQL scored identically (pp), HR (W9), %s; ci = widest of the four intervals" % (m, b), st["wide_ci_pp"])
                put("%s.w9.st_%s.reading" % (b, short), st["reading_wide"], "str", "reading of the widest interval, same-text, %s, HR (W9), %s" % (m, b))
                put("%s.w9.st_%s.mde" % (b, short), st["mde80_pp"], "x2", "minimum detectable effect (pp), same-text, %s, HR (W9), %s" % (m, b))
            if r["harm_per_100_correct_empty"] is not None:
                put("%s.w9.harm_rate" % b, r["harm_per_100_correct_empty"], "int", "harm per 100 correct empty answers of the single call + retry (%%), HR (W9), %s" % b)
        B = w9["backbones"].values()
        put("w9.harm_min", min(r["harm"] for r in B), "int", "fewest harmed questions over the four backbones (W9)")
        put("w9.harm_max", max(r["harm"] for r in B), "int", "most harmed questions over the four backbones (W9)")
        put("w9.dex_ez_min", min(r["dex_ez"] for r in B), "pp2", "smallest EX gain of the empty-gate retry over the single call on HR (W9)")
        put("w9.dex_ez_max", max(r["dex_ez"] for r in B), "pp2", "largest EX gain of the empty-gate retry over the single call on HR (W9)")
        put("w9.dex_er_min", min(r["dex_er"] for r in B), "pp2", "smallest EX difference of the empty-gate retry minus the single call + retry on HR (W9)")
        put("w9.dex_er_max", max(r["dex_er"] for r in B), "pp2", "largest EX difference of the empty-gate retry minus the single call + retry on HR (W9)")
        put("w9.dex_rz_min", min(v["contrasts"]["zsretry_minus_zs"]["est_pp"] for v in _load(os.path.join(res, "w9_iv.json"))["backbones"].values()), "pp2", "smallest EX gain of the single call + retry over the single call on HR (W9)")
        put("w9.dex_rz_max", max(v["contrasts"]["zsretry_minus_zs"]["est_pp"] for v in _load(os.path.join(res, "w9_iv.json"))["backbones"].values()), "pp2", "largest EX gain of the single call + retry over the single call on HR (W9)")
        IV = [v["contrasts"] for v in _load(os.path.join(res, "w9_iv.json"))["backbones"].values()]
        for short, cid in (("er", "zsretryempty_minus_zsretry"), ("ez", "zsretryempty_minus_zs"), ("rz", "zsretry_minus_zs")):
            put("w9.st_%s_min" % short, min(c[cid]["same_text"]["est_pp"] for c in IV), "pp2", "smallest same-text estimate over the four backbones, %s (W9)" % cid)
            put("w9.st_%s_max" % short, max(c[cid]["same_text"]["est_pp"] for c in IV), "pp2", "largest same-text estimate over the four backbones, %s (W9)" % cid)
            put("w9.st_%s_mde_min" % short, min(c[cid]["same_text"]["mde80_pp"] for c in IV), "x2", "smallest same-text minimum detectable effect, %s (W9)" % cid)
            put("w9.st_%s_mde_max" % short, max(c[cid]["same_text"]["mde80_pp"] for c in IV), "x2", "largest same-text minimum detectable effect, %s (W9)" % cid)
            put("w9.n_pos_%s" % short, sum(1 for c in IV if c[cid]["reading_wide"] == "positive"), "int", "backbones on which the widest interval of %s excludes zero from below (W9)" % cid)
            put("w9.n_pos_st_%s" % short, sum(1 for c in IV if c[cid]["same_text"]["reading_wide"] == "positive"), "int", "same, same-text (W9)")
        put("w9.breakeven_min", min(r["breakeven_pct"] for r in B), "int", "smallest break-even share (%) over the four backbones (W9)")
        put("w9.breakeven_max", max(r["breakeven_pct"] for r in B), "int", "largest break-even share (%) over the four backbones (W9)")
        put("w9.net_fired_q_min", min(r["up_fired"] - r["down_fired"] for r in B), "int", "smallest in-set net of the trigger (questions) over the four backbones (W9)")
        put("w9.net_fired_q_max", max(r["up_fired"] - r["down_fired"] for r in B), "int", "largest in-set net of the trigger (questions) over the four backbones (W9)")
        put("w9.share_empty_correct_min", min(100.0 * r["right_r_empty_ref"] / r["n_fired"] for r in B), "int", "smallest share (%) of the plain retry's empty answers that are correct (W9)")
        put("w9.share_empty_correct_max", max(100.0 * r["right_r_empty_ref"] / r["n_fired"] for r in B), "int", "largest share (%) of the plain retry's empty answers that are correct (W9)")
        put("w9.up_fired_pp_min", min(100.0 * r["up_fired"] / w9["n"] for r in B), "pp2", "smallest repairs of the empty trigger, pp of all W9 questions")
        put("w9.up_fired_pp_max", max(100.0 * r["up_fired"] / w9["n"] for r in B), "pp2", "largest repairs of the empty trigger, pp of all W9 questions")
        for f, kind, m in (("harm_per_100_correct_empty", "int", "harm per 100 correct empty answers of the single call + retry (%, rounded: denominators of 15 to 19)"),
                           ("n_fired", "int", "questions where the empty trigger acts"), ("up_fired", "int", "questions turned correct by the empty trigger"),
                           ("down_fired", "int", "questions turned wrong by the empty trigger"), ("net_fired_pp", "pp2", "net effect of the empty trigger where it acts (pp of all questions)"),
                           ("flips_same_sql_out", "int", "same final query scored differently (inside and outside the trigger set)"),
                           ("right_r_empty_ref", "int", "correct empty answers of the single call + retry")):
            vals = [r[f] for r in B if r.get(f) is not None]
            put("w9.%s_min" % f, min(vals), kind, "smallest over the four backbones (W9): " + m)
            put("w9.%s_max" % f, max(vals), kind, "largest over the four backbones (W9): " + m)
    dates = sorted(set(v["first_request_bj"][:10] for v in runs.values()) | set(v["last_request_bj"][:10] for v in runs.values()))
    put("meta.run_date_first", dates[0], "str", "first day of the X2DS / X2DSR requests (Beijing time)")
    put("meta.run_date_last", dates[-1], "str", "last day of the X2DS / X2DSR requests (Beijing time)")
    put("meta.price_date", "2026-10-26", "str", "date the list prices were read (exp/prices.yaml)")
    put("meta.git", sorted(set(v["git_hash"] for v in runs.values()))[0] if len(set(v["git_hash"] for v in runs.values())) == 1 else "mixed", "str", "git hash of the working tree of the runs")
    x2 = [v for k, v in runs.items() if k.startswith("X2DS/")]
    xr = [v for k, v in runs.items() if k.startswith("X2DSR/")]
    put("runs.x2ds.billed", sum(v["chat_billed"] for v in x2), "int", "chat requests billed in the four DeepSeek arms (X2DS)")
    put("runs.x2ds.from_cache", sum(v["chat_from_cache"] for v in x2), "int", "chat requests of the four DeepSeek arms answered from the call cache (shared roots)")
    put("runs.x2dsr.billed", sum(v["chat_billed"] for v in xr), "int", "chat requests billed in the four retry-only runs (X2DSR; the roots are stored answers)")
    put("runs.x2dsr.from_cache", sum(v["chat_from_cache"] for v in xr), "int", "chat requests of the retry-only runs answered from the call cache (the roots)")
    put("cost.x2ds.cny_disc", cost1["total"]["cny_disc"], "cny2", "CNY billed for the DeepSeek arms, cache discount")
    put("cost.x2ds.cny_list", cost1["total"]["cny_nodisc"], "cny2", "CNY billed for the DeepSeek arms, list price")
    put("cost.x2dsr.cny_disc", cost2["total"]["cny_disc"], "cny2", "CNY billed for the retry-only runs, cache discount")
    put("cost.x2dsr.cny_list", cost2["total"]["cny_nodisc"], "cny2", "CNY billed for the retry-only runs, list price")
    xrr = [v for k, v in runs.items() if k.startswith("X2DSRR/")]
    w7 = [v for k, v in runs.items() if k.startswith("W7/") or k.startswith("W7RR/")]
    put("runs.x2dsrr.billed", sum(v["chat_billed"] for v in xrr), "int", "chat requests billed in the reruns of the retry-only arm and of the DeepSeek chain (X2DSRR)")
    put("runs.w7.billed", sum(v["chat_billed"] for v in w7), "int", "chat requests billed in the thinking-on, empty-gate and MAC-SQL runs and the empty-gate rerun (W7, W7RR)")
    put("cost.x2dsrr.cny_disc", cost3["total"]["cny_disc"], "cny2", "CNY billed for the reruns (X2DSRR), cache discount")
    put("cost.x2dsrr.cny_list", cost3["total"]["cny_nodisc"], "cny2", "CNY billed for the reruns (X2DSRR), list price")
    put("cost.w7.cny_disc", cost4["total"]["cny_disc"] + cost5["total"]["cny_disc"], "cny2", "CNY billed for the W7 and W7RR runs, cache discount")
    put("cost.w7.cny_list", cost4["total"]["cny_nodisc"] + cost5["total"]["cny_nodisc"], "cny2", "CNY billed for the W7 and W7RR runs, list price")
    w8 = [v for k, v in runs.items() if k.startswith("W8/")]
    put("runs.w8.billed", sum(v["chat_billed"] for v in w8), "int", "chat requests billed in the pipeline + empty gate runs and their reruns (W8)")
    put("runs.w8.from_cache", sum(v["chat_from_cache"] for v in w8), "int", "chat requests of the W8 runs answered from the call cache (the archived gated runs)")
    put("cost.w8.cny_disc", cost6["total"]["cny_disc"], "cny2", "CNY billed for the W8 runs, cache discount")
    put("cost.w8.cny_list", cost6["total"]["cny_nodisc"], "cny2", "CNY billed for the W8 runs, list price")
    cs = (cost1, cost2, cost3, cost4, cost5, cost6)
    put("cost.total.cny_disc", sum(c["total"]["cny_disc"] for c in cs), "cny2", "CNY billed for X2DS, X2DSR, X2DSRR, W7, W7RR and W8, cache discount")
    put("cost.total.cny_list", sum(c["total"]["cny_nodisc"] for c in cs), "cny2", "CNY billed for X2DS, X2DSR, X2DSRR, W7, W7RR and W8, list price")
    api = sum(v["chat_billed"] for v in runs.values())
    meta = {"scope": SCOPE, "api_calls": api, "B": b0["B"], "created": "2026-10-29", "complete": True, "missing_keys": []}
    return {"_meta": meta, "keys": keys}


def write_grammar(doc):
    g = [[k, e["kind"]] for k, e in doc["keys"].items()]
    with open(GRAMMAR, "w") as f:
        json.dump(g, f, indent=0)
    return len(g)


def main(argv):
    if argv[1:] == ["--write"]:
        doc = build()
        os.makedirs(RES, exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(doc, f, indent=1)
        print("wrote %d keys to %s; grammar %d" % (len(doc["keys"]), OUT, write_grammar(doc)))
    elif argv[1:] == ["--check"]:
        doc = build()
        have = _load(OUT)
        gram = expected_keys()
        assert have == json.loads(json.dumps(doc)), "results/x2ds/x2ds_keys.json differs from what the inputs give; run --write"
        assert [k for k, _ in gram] == list(have["keys"]), "grammar differs from the key file"
        print("ok: %d keys" % len(gram))
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
