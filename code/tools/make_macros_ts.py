#!/usr/bin/env python3
"""TUNING-SET design-evidence macros  ->  paper/results_macros_ts.tex   (stdlib only, Python 3.6; runs on the host)

    python3 code/tools/make_macros_ts.py [--paper paper] [--out-dir results/ts]

Scope: tuning-set.  The values are legitimately reported TUNING results (held-out BIRD training databases HR / HT, never the development set) that the
paper quotes as design evidence, always framed as such in the text ("On the held-out tuning databases (HR, 8 databases) ...", Section "Development and
Tuning").  They are read ONLY from the tuning-set summary files below, whose _meta.scope must be 'tuning-set' (checked here), and they are written ONLY to
  * paper/results_macros_ts.tex   defines \\tsres{TS.<key>}  (typeset normally, not red; loaded in BOTH builds by main.tex)
  * results/ts/results_ts.json    flat key -> value, _meta.scope = "tuning-set"  (make_macros.py reads only top-level results/*.json, so this never reaches \\res)
  * results/ts/keymap_ts.md       key -> typeset value -> source file:key -> producing run
Separation rule (CONTRACT 0): a TS number never enters paper/results_macros.tex, never appears under \\res, and is never a confirmatory claim.
make_macros.py / make_tables.py refuse scope 'tuning-set'; this script refuses everything but 'tuning-set' inputs.
Sources: results/x5/x5_tuning_summary.json, results/x8/x8_pooled.json, results/p2/{base_ht300,e6_summary}.json, results/x7/{empty_roots,value_check,adaptive_replay}.json,
results/r1/{q1_headroom,q2a_critic,q1_errs}.json, exp/selection_decisions.json, exp/splits/train_holdout.json.
Formatting: make_macros.derive / classify with paper/macro_formats.json, so a value looks as it did in the reference build."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_macros as M  # noqa: E402
import make_macros_ref as MR  # noqa: E402  (Ref container, load, wilson only)

REPO = M.REPO
SCOPE = "tuning-set"
Ref, load = MR.Ref, MR.load
TUNING_INPUTS = ["results/x5/x5_tuning_summary.json", "results/x8/x8_pooled.json"]


def check_scopes():
    for rel in TUNING_INPUTS:
        sc = (load(rel).get("_meta") or {}).get("scope")
        if sc != "tuning-set":
            sys.exit("make_macros_ts: %s has _meta.scope %r; only 'tuning-set' summaries may feed the TS macros" % (rel, sc))


def build():
    check_scopes()
    R = Ref()
    x8 = load("results/x8/x8_pooled.json")
    x5 = load("results/x5/x5_tuning_summary.json")
    e6 = load("results/p2/e6_summary.json")
    emp = load("results/x7/empty_roots.json")
    vchk = load("results/x7/value_check.json")
    adp = load("results/x7/adaptive_replay.json")
    r1h = load("results/r1/q1_headroom.json")
    r1c = load("results/r1/q2a_critic.json")
    r1e = load("results/r1/q1_errs.json")
    sel = load("exp/selection_decisions.json")
    split = load("exp/splits/train_holdout.json")
    S8, S5 = "results/x8/x8_pooled.json", "results/x5/x5_tuning_summary.json"
    RUN8 = "queue X8 (HT-300, 4 dbs): runs/X8/x8-<model>-{zs,fixed,gated}"
    RUN5HR = "queue X5 jobs x5-1b-hr-{quote,gated} (HR-1101, qwen3.7-plus)"
    RUN5HT = "queue X5 jobs x5-1b-ht-{quote,gated} (HT-300, qwen3.7-plus)"
    # tuning-split sizes (facts of the split file, used by the framing macros \HRdb / \HTdb)
    SP = "exp/splits/train_holdout.json"
    R.put("TS.setup.hr.n_questions", len(split["hr_order"]), SP + ":hr_order", "exp/splits/make_splits.py")
    R.put("TS.setup.hr.n_dbs", len(split["hr_dbs"]), SP + ":hr_dbs", "exp/splits/make_splits.py")
    R.put("TS.setup.ht.n_questions", len(split["ht300_qids"]), SP + ":ht300_qids", "exp/splits/make_splits.py")
    R.put("TS.setup.ht.n_dbs", len(split["ht_dbs"]), SP + ":ht_dbs", "exp/splits/make_splits.py")
    R.put("TS.setup.pool.n_dbs", len(split["pool_dbs"]), SP + ":pool_dbs", "exp/splits/make_splits.py")
    R.put("TS.setup.pool.n_questions", 6918, "internal note X2 section 1 (bird_train_pool)", "exp/splits/make_splits.py")

    # ---------------------------------------------------------------- tuning-set keys (TS.*), cited in prose with the label "tuning set"
    runs5 = [("TS.arb.", "queue X5 x5-3-probe-{revise,pool}"), ("TS.m3.", "queue X5 x5-4-ht-{base,m3force}"), ("TS.quote.", "queue X5 x5-cmp-1a (131 beer_factory HR questions)"),
             ("TS.hr.", RUN5HR), ("TS.ht.", RUN5HT), ("TS.route.", "queue X5 x5-2-{hr,ht}-cons + x5-2-fit")]
    for kk in x5:
        for pre, run in runs5:
            if kk.startswith(pre) and not isinstance(x5[kk], (dict, list)):
                R.put(kk, x5[kk], S5 + ":" + kk, run)
    for kk in x8:
        if kk.startswith("TS.x8.") and not isinstance(x8[kk], (dict, list)):
            R.put(kk, x8[kk], S8 + ":" + kk, RUN8)
    R.put("TS.rs.full_fixed.nonempty.gain", x5["TS.rs.full_fixed.nonempty.gain"], S5, RUN5HR)
    R.put("TS.rs.full_fixed.nonempty.harm", x5["TS.rs.full_fixed.nonempty.harm"], S5, RUN5HR)
    d0 = sel["rule4_detail"]["hr"]
    R.ci("TS.sel.k2.dauroc", d0["dAUROC"], d0["dAUROC_ci95"][0], d0["dAUROC_ci95"][1], "exp/selection_decisions.json:rule4_detail.hr (k=2)", "code/tools/selection_rules.py --k 2")
    # the selection rule's own figure: HR, gated minus fixed, 95 % interval used by condition (i) of rule 3 (differs from the later X5 summary of the same pairs)
    r3 = sel["rule3_detail"]["hr"]
    R.ci("TS.sel.rule3.hr.delta_ex", r3["diff_pp"], r3["diff_pp_ci95"][0], r3["diff_pp_ci95"][1], "exp/selection_decisions.json:rule3_detail.hr (diff_pp, diff_pp_ci95)", "code/tools/selection_rules.py (rule 3, run once on 2026-10-26)")
    # controller (E6 tuning-set cross-validation, 12 cells)
    cv, cva = e6["cv"], e6["cv_a2c"]
    bp = max(cv, key=lambda r: r["ex_all"])
    ba = max(cva, key=lambda r: r["ex_all"])
    R.put("TS.e6.n_cells", len(cv), "results/p2/e6_summary.json:cv", "rl.train cv (HR, 4 folds x 3 seeds)")
    R.put("TS.e6.ppo_best_ex", bp["ex_all"], "results/p2/e6_summary.json:cv (best cell by EX)", "rl.train cv")
    R.put("TS.e6.ppo_best_calls", bp["calls"], "results/p2/e6_summary.json:cv (best cell by EX)", "rl.train cv")
    R.put("TS.e6.p0_ex", bp["P0_8.5"]["ex_all"], "results/p2/e6_summary.json:cv[best].P0_8.5", "rl.train base")
    R.put("TS.e6.p0_calls", bp["P0_8.5"]["calls"], "results/p2/e6_summary.json:cv[best].P0_8.5", "rl.train base")
    R.put("TS.e6.a2c_best_ex", ba["ex_all"], "results/p2/e6_summary.json:cv_a2c (best cell by EX)", "rl.train cv --algo a2c")
    R.put("TS.e6.a2c_best_calls", ba["calls"], "results/p2/e6_summary.json:cv_a2c (best cell by EX)", "rl.train cv --algo a2c")
    # X7 / X9 offline bounds
    hr = emp["P0_HR"]
    R.put("TS.empty.hr.n_empty", hr["n_empty_roots"], "results/x7/empty_roots.json:P0_HR", "tools/empty_roots.py on x5-1b-hr-quote")
    R.put("TS.empty.hr.root_wrong", hr["root_wrong"], "results/x7/empty_roots.json:P0_HR", "tools/empty_roots.py on x5-1b-hr-quote")
    R.put("TS.probe.hr.ub_pp", hr["probe_visible_pp_of_all_questions"], "results/x7/empty_roots.json:P0_HR", "tools/empty_roots.py (upper bound)")
    R.put("TS.vg.hr.ub_pp", vchk["gated_HR"]["upper_bound_value_grounding"]["retrievable_by_lookup_pp"], "results/x7/value_check.json:gated_HR", "tools/value_check.py (upper bound)")
    c0 = [c for c in adp["HR"]["configs"] if c["k"] == 4 and c["theta_fast"] == 1.0 and c["empty"] == "vote"][0]["vs_gated_same_roots"]
    R.ci("TS.adapt.hr.delta_ex", c0["diff_pp"], c0["ci95_pp"][0], c0["ci95_pp"][1], "results/x7/adaptive_replay.json:HR.configs[k=4,theta_fast=1.0]", "tools/adaptive_replay.py (HR replay)")
    # R1 analyses (tuning set, HR, pre-fix tree)
    ts = [s for s in r1h["oracle_sets"] if s["set"].startswith("whole tree (")][0]
    R.put("TS.r1.tree_unsolved_share", 1 - ts["oracle_EX"], "results/r1/q1_headroom.json:oracle_sets[whole tree]", "R1 offline analysis of stored HR trees")
    R.put("TS.r1.tree_nodes_per_q", ts["mean_nodes"], "results/r1/q1_headroom.json:oracle_sets[whole tree]", "R1 offline analysis of stored HR trees")
    R.put("TS.r1.ideal_gate_gain_pp", r1h["ideal_gate_P0chain"]["gain_pp_vs_replay_P0"], "results/r1/q1_headroom.json:ideal_gate_P0chain", "R1 offline analysis of stored HR trees")
    R.put("TS.r1.p0_beer_gain", r1e["P0_replay_gain_harm_beer_vs_others"]["beer_factory/exec_error"]["gain"], "results/r1/q1_errs.json", "R1 replay of P0 (pre-fix)")
    R.put("TS.r1.p0_exec_error_gain", r1e["P0_replay_by_root_status"]["exec_error"]["gain"], "results/r1/q1_errs.json", "R1 replay of P0 (pre-fix)")
    sw = r1c["root_score_dist"]["HR"]["sem_dist_wrong"]
    R.put("TS.r1.wrong_roots", sum(sw.values()), "results/r1/q2a_critic.json:root_score_dist.HR.sem_dist_wrong", "R1 critic analysis (HR roots)")
    R.put("TS.r1.wrong_roots_sem10", sw["10.0"], "results/r1/q2a_critic.json:root_score_dist.HR.sem_dist_wrong", "R1 critic analysis (HR roots)")
    # strong backbone, HT-300: gated minus single call, point estimate (the single-call run predates the identifier-quoting fix: legacy renderer)
    base = load("results/p2/base_ht300.json")
    zs_plus = base["P2_base-zs_ours-qwen37plus-s0-80485cc1"]
    R.put("TS.ht.gated_vs_zs.delta_ex.mean", 100.0 * (x5["TS.ht.gated.ex"] - zs_plus["ex"]), S5 + ":TS.ht.gated.ex - results/p2/base_ht300.json:zs_ours.ex", RUN5HT + "; runs/P2_base (legacy renderer for zs_ours) (point estimate, no CI)")
    return R


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paper", default=os.path.join(REPO, "paper"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "results", "ts"))
    a = ap.parse_args(argv)
    R = build()
    vals = R.v
    os.makedirs(a.out_dir, exist_ok=True)
    doc = {"_meta": {"scope": SCOPE, "note": "TUNING-SET design evidence quoted in the paper under \\tsres{}; framed as tuning-set evidence in the text. Not confirmatory. make_macros.py / make_tables.py refuse this scope.",
                     "generator": "code/tools/make_macros_ts.py"}}
    doc.update(vals)
    with open(os.path.join(a.out_dir, "results_ts.json"), "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    overrides = []
    ov = os.path.join(a.paper, "macro_formats.json")
    if os.path.exists(ov):
        overrides = [(k, v) for k, v in json.load(open(ov)).items()]
    out = M.derive(vals, overrides)
    lines = ["% GENERATED by code/tools/make_macros_ts.py -- do not edit. Loaded by main.tex in BOTH builds.",
             "% Every value below is a TUNING-SET design-evidence number (scope " + SCOPE + "), quoted only through \\tsres{} in sentences framed as tuning-set evidence.",
             "\\makeatletter",
             "\\def\\tsres#1{\\@ifundefined{tsv@#1}{\\resfallback{#1}}{\\csname tsv@#1\\endcsname}}"]
    for k in sorted(out):
        lines.append("\\expandafter\\gdef\\csname tsv@%s\\endcsname{%s}" % (k, out[k]))
    lines.append("\\makeatother")
    with open(os.path.join(a.paper, "results_macros_ts.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    used = scan_ts_uses(a.paper)
    rows = ["| KEY | value (as typeset) | source (file:key) | producing run | used in |", "|---|---|---|---|---|"]
    for k in sorted(out):
        src = R.src.get(k) or R.src.get(k + ".mean") or ("derived", "")
        rows.append("| `%s` | %s | %s | %s | %s |" % (k, str(out[k]).replace("|", "/").replace("\\ensuremath{-}", "-"), src[0].replace("|", "/"), src[1].replace("|", "/"), ", ".join(sorted(used.get(k, []))) or "-"))
    with open(os.path.join(a.out_dir, "keymap_ts.md"), "w") as f:
        f.write("# TS.* keys (scope tuning-set), generated by code/tools/make_macros_ts.py\n\nTyped with \\tsres{KEY} only; never under \\res{}; never confirmatory. Reference-mode keys are in results/ref/keymap.md.\n\n" + "\n".join(rows) + "\n")
    missing = sorted(k for k in used if k not in out)
    print("make_macros_ts: %d TS keys -> paper/results_macros_ts.tex; keymap %s/keymap_ts.md" % (len(vals), a.out_dir))
    print("make_macros_ts: %d \\tsres keys used in the paper, %d without a value%s" % (len(used), len(missing), (": " + ", ".join(missing)) if missing else ""))
    return 1 if missing else 0


def scan_ts_uses(paper_dir):
    import glob
    import re
    rx = re.compile(r"\\tsres\{([^}]*)\}")
    used = {}
    files = [os.path.join(paper_dir, "main.tex")] + sorted(glob.glob(os.path.join(paper_dir, "sections", "*.tex"))) + sorted(glob.glob(os.path.join(paper_dir, "tables", "*.tex"))) + sorted(glob.glob(os.path.join(paper_dir, "ref", "tables", "*.tex")))
    for f in files:
        if os.path.exists(f):
            for m in rx.finditer(M.strip_comments(open(f, encoding="utf-8", errors="replace").read())):
                used.setdefault(m.group(1).strip(), set()).add(os.path.relpath(f, paper_dir))
    return used


if __name__ == "__main__":
    sys.exit(main())
