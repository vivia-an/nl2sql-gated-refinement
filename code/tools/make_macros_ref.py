#!/usr/bin/env python3
"""TUNING-SET reference values for the paper draft  ->  paper/results_macros_ref.tex   (stdlib only, Python 3.6; runs on the host)

    python3 code/tools/make_macros_ref.py [--paper paper] [--ref-dir results/ref] [--keymap results/ref/keymap.md]

Purpose (N2, 2026-10-26): writing runs in parallel with the experiments. The paper text uses the confirmatory key names (\\res{P1.delta_ex}, ...);
this script maps each of them to the closest TUNING-SET number so that the whole story can be read before the development run exists.
  * results/ref/results_ref.json   flat key -> value, _meta.scope = "tuning-reference" (a scope make_macros.py / make_tables.py REFUSE,
                                   so nothing here can reach paper/results_macros.tex or paper/tables/)
  * paper/results_macros_ref.tex   defines \\res for REFERENCE MODE ONLY (main.tex loads it when \\refmodetrue): every value red with a dagger
  * results/ref/keymap.md          table KEY -> reference value (formatted) -> source file/key -> producing run
Replacement step when the confirmatory results exist: build normally (make_macros.py + make_tables.py). Nothing else changes; the
reference file is never edited by hand. Keys without a tuning analogue are simply absent here and print as bold [KEY] in reference mode.
Formatting uses make_macros.derive / classify (same rules and paper/macro_formats.json), so a reference value looks like the final one."""
import argparse
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_macros as M  # noqa: E402

REPO = M.REPO
SCOPE = "tuning-reference"
MODELS = {"qwen37plus": "plus", "qwen37flash": "flash", "qwen3527b": "b27"}


def load(rel):
    with open(os.path.join(REPO, rel)) as f:
        return json.load(f)


def wilson(k, n, z=1.959964):
    if n == 0:
        return None, None
    p = k / float(n)
    d = 1 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4.0 * n * n)) / d
    return c - h, c + h


class Ref(object):
    def __init__(self):
        self.v, self.src = {}, {}

    def put(self, key, value, source, run):
        if value is None:
            return
        self.v[key] = value
        self.src[key] = (source, run)

    def ci(self, key, mean, lo, hi, source, run):
        self.put(key + ".mean", mean, source, run)
        self.put(key + ".ci_lo", lo, source, run)
        self.put(key + ".ci_hi", hi, source, run)

    def copy_ci(self, key, d, dkey, source, run):
        self.ci(key, d[dkey + ".mean"], d[dkey + ".ci_lo"], d[dkey + ".ci_hi"], "%s:%s" % (source, dkey), run)


def build():
    R = Ref()
    x8 = load("results/x8/x8_pooled.json")
    x8s = load("results/x8/x8_tuning_summary.json")
    x5 = load("results/x5/x5_tuning_summary.json")
    base = load("results/p2/base_ht300.json")
    e6 = load("results/p2/e6_summary.json")
    e4 = load("results/e4_min/e4_p40_summary.json")
    emp = load("results/x7/empty_roots.json")
    vchk = load("results/x7/value_check.json")
    adp = load("results/x7/adaptive_replay.json")
    r1h = load("results/r1/q1_headroom.json")
    r1c = load("results/r1/q2a_critic.json")
    r1e = load("results/r1/q1_errs.json")
    sel = load("exp/selection_decisions.json")
    split = load("exp/splits/train_holdout.json")
    S8, S5, SB = "results/x8/x8_pooled.json", "results/x5/x5_tuning_summary.json", "results/p2/base_ht300.json"
    RUN8 = "queue X8 (HT-300, 4 dbs): runs/X8/x8-<model>-{zs,fixed,gated}"
    RUN5HT = "queue X5 jobs x5-1b-ht-{quote,gated} (HT-300, qwen3.7-plus)"
    RUN5HR = "queue X5 jobs x5-1b-hr-{quote,gated} (HR-1101, qwen3.7-plus)"

    # ---------------------------------------------------------------- setup (facts, not results)
    R.put("setup.bird_dev.n_questions", 1534, "SAP 13.3 / BIRD dev", "none (benchmark size)")
    R.put("setup.bird_dev.n_dbs", 11, "SAP 13.3 / BIRD dev", "none (benchmark size)")
    R.put("setup.hr.n_questions", len(split["hr_order"]), "exp/splits/train_holdout.json:hr_order", "exp/splits/make_splits.py")
    R.put("setup.hr.n_dbs", len(split["hr_dbs"]), "exp/splits/train_holdout.json:hr_dbs", "exp/splits/make_splits.py")
    R.put("setup.ht.n_questions", len(split["ht300_qids"]), "exp/splits/train_holdout.json:ht300_qids", "exp/splits/make_splits.py")
    R.put("setup.ht.n_dbs", len(split["ht_dbs"]), "exp/splits/train_holdout.json:ht_dbs", "exp/splits/make_splits.py")
    R.put("setup.pool.n_dbs", 56, "internal note X2 section 2 (POOL56)", "exp/splits/make_splits.py")
    R.put("setup.pool.n_questions", 6918, "internal note X2 section 1 (bird_train_pool)", "exp/splits/make_splits.py")
    try:
        h = subprocess.check_output(["git", "-C", REPO, "log", "-1", "--format=%h", "--", "docs/SAP.md"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        h = ""
    R.put("setup.plan.commit", (h or "unknown") + " (last commit touching SAP.md; Amendment 3 not committed yet)", "git log -1 -- docs/SAP.md", "author commit")
    R.put("setup.devlock.hash", "pending (no dev lock yet)", "exp/dev_lock.json absent", "author at the lock (F1)")
    R.put("setup.resolution_pp", 3.0, "SAP 13.7 / 15.6 (planning resolution, near-neighbour pairs)", "docs/SAP.md power calculation (plan constant)")
    R.put("setup.design_effect", 2.4, "SAP 13.7 / 15.6 (design effect)", "docs/SAP.md power calculation (plan constant)")
    R.put("setup.p4_mde_pp", 6.9, "SAP 13.3 P4 (MDE at alpha 0.05)", "docs/SAP.md power calculation (plan constant)")
    R.put("setup.e10.n_annotated", 300, "MINIMAL_PLAN M6 / STORYLINE R7 (planned sheet size)", "E10 annotation (not run)")

    # ---------------------------------------------------------------- H1 / P1 (gated vs fixed, pooled over three backbones, HT-300)
    R.put("P.family.m", 4, "SAP 15.3 (design)", "none")
    R.put("P.family.members", "P1, W1, P3, P5", "SAP 15.3 (design)", "none")
    R.put("P.family.dropped", "none", "SAP 15.3 (design)", "none")
    for h_ in ("P1", "W1", "P3", "P5", "P2", "P4"):
        R.put(h_ + ".status", "reference (tuning set)", "n/a", "n/a")
        R.put(h_ + ".verdict", "not tested (reference)", "n/a", "n/a")
    R.copy_ci("P1.delta_ex", x8, "TS.x8.p1_pooled.delta_ex", S8, RUN8 + " + " + RUN5HT)
    R.copy_ci("P1.calls_ratio", x8, "TS.x8.p1_pooled.calls_ratio", S8, RUN8 + " + " + RUN5HT)
    cr = "TS.x8.p1_pooled.calls_ratio"
    R.ci("P1.calls_saved", 1 - x8[cr + ".mean"], 1 - x8[cr + ".ci_hi"], 1 - x8[cr + ".ci_lo"], S8 + ":1-" + cr, RUN8 + " + " + RUN5HT)
    R.put("P1.n_backbones", x8["TS.x8.p1_pooled.n_backbones"], S8, RUN8)
    for mk, m in MODELS.items():
        R.copy_ci("P1.%s.delta_ex" % mk, x8, "TS.x8.%s.gated_vs_fixed.delta_ex" % m, S8, RUN8 if m != "plus" else RUN5HT)
        R.put("P1.%s.calls_ratio" % mk, x8["TS.x8.%s.calls_ratio" % m], S8, RUN8 if m != "plus" else RUN5HT)

    # ---------------------------------------------------------------- H2 / W1 (gated vs zs, two weak backbones)
    R.copy_ci("W1.delta_ex", x8, "TS.x8.w1_pooled.delta_ex", S8, RUN8)
    R.put("W1.n_backbones", x8["TS.x8.w1_pooled.n_backbones"], S8, RUN8)
    R.put("W1.sign_consistent", bool(x8["TS.x8.w1_pooled.all_same_sign"]), S8, RUN8)
    zs_plus = base["P2_base-zs_ours-qwen37plus-s0-80485cc1"]
    fx_plus = base["P2_base-full_fixed-qwen37plus-s0-d37d2097"]
    SBR = "runs/P2_base (HT-300, legacy renderer for zs_ours)"
    R.put("E1.vs_zs.full_gated.qwen37plus.delta_ex.mean", 100.0 * (x5["TS.ht.gated.ex"] - zs_plus["ex"]), S5 + ":TS.ht.gated.ex - " + SB + ":zs_ours.ex", RUN5HT + "; " + SBR + " (point estimate, no CI)")
    R.put("E1.vs_zs.full_fixed.qwen37plus.delta_ex.mean", 100.0 * (x5["TS.ht.fixed.ex"] - zs_plus["ex"]), S5 + ":TS.ht.fixed.ex - " + SB + ":zs_ours.ex", RUN5HT + "; " + SBR + " (point estimate, no CI)")

    # ---------------------------------------------------------------- E1 / E2 / E11 per backbone
    R.put("E1.zs.qwen37plus.ex", zs_plus["ex"], SB, SBR)
    R.put("E1.zs.qwen37plus.calls", zs_plus["calls_per_q"], SB, SBR)
    R.put("E1.zs.qwen37plus.soft_f1", zs_plus["soft_f1"], SB, SBR)
    R.put("E1.full_fixed.qwen37plus.ex", x5["TS.ht.fixed.ex"], S5, RUN5HT)
    R.put("E1.full_fixed.qwen37plus.calls", x5["TS.ht.fixed.calls"], S5, RUN5HT)
    R.put("E1.full_fixed.qwen37plus.soft_f1", fx_plus["soft_f1"], SB, "runs/P2_base (HT-300)")
    R.put("E1.full_gated.qwen37plus.ex", x5["TS.ht.gated.ex"], S5, RUN5HT)
    R.put("E1.full_gated.qwen37plus.calls", x5["TS.ht.gated.calls"], S5, RUN5HT)
    cost = {"plus": x8s["qwen37plus_ref"]["cost_cny_per_q"], "flash": x8s["qwen37flash"]["cost_cny_per_q"], "b27": x8s["qwen3527b"]["cost_cny_per_q"]}
    for mk, m in MODELS.items():
        for lab, ck in (("zs", "zs"), ("full_fixed", "fixed"), ("full_gated", "gated")):
            R.put("E11.%s.%s.cost_per_q.mean" % (lab, mk), cost[m][ck], "results/x8/x8_tuning_summary.json:cost_cny_per_q", "cost_report --by run, list price exp/prices.yaml")
    for lab, ck in (("full_fixed", "fixed"), ("full_gated", "gated")):
        R.put("E11.%s.qwen37plus.calls.mean" % lab, x5["TS.ht.%s.calls" % ck], S5, RUN5HT)
    R.put("E11.zs.qwen37plus.calls.mean", 1.0, SB, SBR)
    for mk, m in (("qwen37flash", "flash"), ("qwen3527b", "b27")):
        R.put("E2.%s.zs.ex" % mk, x8["TS.x8.%s.zs.ex" % m], S8, RUN8)
        R.put("E2.%s.full_gated.ex" % mk, x8["TS.x8.%s.gated.ex" % m], S8, RUN8)
        R.put("E2.%s.full_gated.calls" % mk, x8["TS.x8.%s.gated.calls" % m], S8, RUN8)
        R.copy_ci("E2.%s.gain_gated_vs_zs.delta_ex" % mk, x8, "TS.x8.%s.gated_vs_zs.delta_ex" % m, S8, RUN8)
        R.put("E2.%s.gain_gated_vs_zs.p" % mk, x8["TS.x8.%s.gated_vs_zs.p" % m], S8, RUN8 + " (exact two-sided McNemar)")
        R.copy_ci("E3.%s.rootonly_vs_zs.delta_ex" % mk, x8, "TS.x8.%s.rootonly_vs_zs.delta_ex" % m, S8, RUN8 + " root candidates of the gated run")
        R.copy_ci("E3.%s.loop_over_root.delta_ex" % mk, x8, "TS.x8.%s.gated_vs_rootonly.delta_ex" % m, S8, RUN8 + " root candidates of the gated run")
    R.put("E2.qwen37plus.zs.ex", zs_plus["ex"], SB, SBR)
    R.put("E2.qwen37plus.full_gated.ex", x5["TS.ht.gated.ex"], S5, RUN5HT)
    R.put("E2.qwen37plus.full_gated.calls", x5["TS.ht.gated.calls"], S5, RUN5HT)
    R.put("E2.qwen37plus.gain_gated_vs_zs.delta_ex.mean", 100.0 * (x5["TS.ht.gated.ex"] - zs_plus["ex"]), S5 + " - " + SB, RUN5HT + "; " + SBR + " (point estimate)")

    # S-I: flash + gated vs plus + zero-shot
    R.put("X1.flash_vs_plus_zs.delta_ex.mean", 100.0 * (x8["TS.x8.flash.gated.ex"] - zs_plus["ex"]), S8 + " - " + SB, RUN8 + "; " + SBR + " (point estimate)")
    R.put("X1.flash_vs_plus_zs.cost_ratio", cost["flash"]["gated"] / cost["plus"]["zs"], "results/x8/x8_tuning_summary.json:cost_cny_per_q", "flash gated / plus zs, list price")

    # ---------------------------------------------------------------- H4 / P5 (repair; HT-300 gated runs pooled over three backbones, Wilson intervals)
    nR, k = x8["TS.x8.pooled.n_root_fail"], x8["TS.x8.pooled.n_repaired"]
    lo, hi = wilson(k, nR)
    R.ci("P5.repair_rate", k / float(nR), lo, hi, S8 + ":TS.x8.pooled.n_repaired / n_root_fail (Wilson interval)", RUN8 + " + " + RUN5HT)
    ke = x8["TS.x8.pooled.n_executable"]
    lo, hi = wilson(ke, nR)
    R.ci("P5.executable_rate", ke / float(nR), lo, hi, S8 + ":TS.x8.pooled.n_executable / n_root_fail (Wilson interval)", RUN8 + " + " + RUN5HT)
    R.put("P5.n_R", nR, S8 + ":TS.x8.pooled.n_root_fail", RUN8 + " + " + RUN5HT)

    # ---------------------------------------------------------------- H3 / P3 and E7 (routing; HR leave-one-database-out)
    R.put("P3.v3.auroc.mean", x5["TS.route.hr.v3.auroc"], S5, "queue X5 jobs x5-2-hr-cons + x5-2-fit (HR, LODO)")
    R.put("P3.route_ca.auroc.mean", x5["TS.route.hr.ca.auroc"], S5, "queue X5 jobs x5-2-hr-cons + x5-2-fit (HR, LODO, k=4)")
    R.copy_ci("P3.dauroc", x5, "TS.route.hr.dauroc", S5, "queue X5 x5-2-fit (HR, LODO, k=4)")
    for s_, p_ in (("v3", "v3"), ("route_ca", "ca")):
        for met in ("auroc", "aurc", "acc_ex_cov60"):
            R.put("E7.%s.%s.mean" % (s_, met), x5["TS.route.hr.%s.%s" % (p_, met)], S5, "queue X5 x5-2-fit (HR, LODO)")
    R.copy_ci("E7.route_ca_vs_v3.dauroc", x5, "TS.route.hr.dauroc", S5, "queue X5 x5-2-fit (HR, LODO)")
    R.put("E7.route_ca_vs_v3.daurc.mean", x5["TS.route.hr.ca.aurc"] - x5["TS.route.hr.v3.aurc"], S5 + ":TS.route.hr.ca.aurc - v3.aurc", "queue X5 x5-2-fit (HR, LODO); no CI")

    # ---------------------------------------------------------------- root status (HR, strong backbone)
    for lab in ("full_fixed", "full_gated"):
        for g in ("err", "empty", "nonempty"):
            for st in ("n", "ex_root", "ex_final", "gain", "harm"):
                kk = "TS.rs.%s.%s.%s" % (lab, g, st)
                if kk in x5:
                    R.put("E1.rootstatus.%s.%s.%s" % (lab, g, st), x5[kk], S5 + ":" + kk, RUN5HR)

    # ---------------------------------------------------------------- ablation / M3 / quoting
    R.put("E3.full_fixed.ex.mean", x5["TS.m3.default.ex"], S5 + ":TS.m3.default.ex", "queue X5 x5-4-ht-base (HT-300, full_fixed re-run, fallback 6000)")
    R.put("E3.abl_m3force.ex.mean", x5["TS.m3.forced.ex"], S5, "queue X5 x5-4-ht-m3force (HT-300)")
    R.copy_ci("E3.abl_m3force.delta_vs_full.delta_ex", x5, "TS.m3.forced.delta_ex", S5, "queue X5 x5-4-ht-{base,m3force} (HT-300)")

    # ---------------------------------------------------------------- store experiment (E4 mini-stream, HR, p = 40 %, 2 permutations, no CI)
    fp = e4["final_probe_ex"]
    for pol in ("append", "ws_full", "memrl"):
        R.put("E4.%s.p40.final_ex.mean" % pol, fp[pol], "results/e4_min/e4_p40_summary.json:final_probe_ex", "runs/MIN/E4 min-e4-%s-p40-perm{1,2} (HR)" % pol)
    R.put("P4.ex_a", fp["ws_full"], "results/e4_min/e4_p40_summary.json", "runs/MIN/E4 min-e4-ws_full-p40-perm{1,2}")
    R.put("P4.ex_b", fp["append"], "results/e4_min/e4_p40_summary.json", "runs/MIN/E4 min-e4-append-p40-perm{1,2}")
    R.put("P4.diff_pp.mean", e4["ws_full_minus_append_pp"], "results/e4_min/e4_p40_summary.json", "runs/MIN/E4 (point estimate, no CI)")
    ev = e4["eviction_ws_full"]
    R.put("E4.ws_full.p40.evicted_total", ev["evicted_total"], "results/e4_min/e4_p40_summary.json:eviction_ws_full", "runs/MIN/E4 min-e4-ws_full-p40")
    R.put("E4.ws_full.p40.evicted_clean", ev["evicted_clean"], "results/e4_min/e4_p40_summary.json:eviction_ws_full", "runs/MIN/E4 min-e4-ws_full-p40")
    R.put("E4.ws_full.p40.recall_polluted", ev["recall_polluted"], "results/e4_min/e4_p40_summary.json:eviction_ws_full", "runs/MIN/E4 min-e4-ws_full-p40")

    # TS.* design-evidence keys moved to code/tools/make_macros_ts.py (scope tuning-set, \\tsres{}, loaded in both builds, not red; N3 2026-10-26)
    return R


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paper", default=os.path.join(REPO, "paper"))
    ap.add_argument("--ref-dir", default=os.path.join(REPO, "results", "ref"))
    ap.add_argument("--keymap", default=None)
    a = ap.parse_args(argv)
    R = build()
    vals = R.v
    os.makedirs(a.ref_dir, exist_ok=True)
    doc = {"_meta": {"scope": SCOPE, "note": "TUNING-SET reference values for the paper draft (reference mode). NOT results. make_macros.py and make_tables.py refuse this scope.",
                     "generator": "code/tools/make_macros_ref.py"}}
    doc.update(vals)
    with open(os.path.join(a.ref_dir, "results_ref.json"), "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    overrides = []
    ov = os.path.join(a.paper, "macro_formats.json")
    if os.path.exists(ov):
        overrides = [(k, v) for k, v in json.load(open(ov)).items()]
    out = M.derive(vals, overrides)
    used = M.scan_sources(a.paper)
    rt = os.path.join(a.paper, "ref", "tables")                    # tables of the reference build (make_tables.py --ref)
    if os.path.isdir(rt):
        import glob
        for f in glob.glob(os.path.join(rt, "*.tex")):
            for m in M.RES_RE.finditer(M.strip_comments(open(f, encoding="utf-8").read())):
                used.setdefault(m.group(1).strip(), set()).add("ref/tables/" + os.path.basename(f))
    lines = ["% GENERATED by code/tools/make_macros_ref.py -- do not edit. REFERENCE MODE ONLY (loaded by main.tex when \\refmodetrue).",
             "% Every value below is a TUNING-SET reference number (scope " + SCOPE + "), not a result of this study.",
             "\\makeatletter",
             "\\providecommand{\\refmark}{\\textsuperscript{\\textdagger}}",
             "\\def\\res#1{\\@ifundefined{resv@#1}{\\resfallback{#1}}{{\\color{red}\\csname resv@#1\\endcsname\\refmark}}}"]
    for k in sorted(out):
        lines.append("\\expandafter\\gdef\\csname resv@%s\\endcsname{%s}" % (k, out[k]))
    n_alias = 0
    for k in sorted(used):                                        # aliases of the confirmatory plan (P3.auroc_v3 -> P3.v3.auroc.mean, ...)
        if k not in out:
            r = M.resolve(k, out, vals)
            if r is not None:
                lines.append("\\expandafter\\gdef\\csname resv@%s\\endcsname{%s}" % (k, r))
                out[k] = r
                n_alias += 1
    lines.append("\\makeatother")
    with open(os.path.join(a.paper, "results_macros_ref.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    missing = sorted(k for k in used if k not in out)
    keymap = a.keymap or os.path.join(a.ref_dir, "keymap.md")
    rows = ["| KEY | reference value (as typeset) | source (file:key) | producing run |", "|---|---|---|---|"]
    for k in sorted(out):
        if k in used or k in vals:
            src = R.src.get(k) or R.src.get(k + ".mean") or ("alias", "")
            rows.append("| `%s` | %s | %s | %s |" % (k, str(out[k]).replace("|", "/").replace("\\ensuremath{-}", "-"), src[0].replace("|", "/"), src[1].replace("|", "/")))
    with open(keymap, "w") as f:
        f.write("<!-- TS.* design-evidence keys (tuning-set, \\tsres{}) are NOT in this table: see results/ts/keymap_ts.md (code/tools/make_macros_ts.py) -->\n\n")
        f.write("\n".join(rows) + "\n")
    miss_path = os.path.join(a.ref_dir, "missing_in_ref.txt")
    with open(miss_path, "w") as f:
        f.write("# keys used by the LaTeX sources that have no tuning-set reference value (print as bold [KEY] in reference mode)\n")
        for k in missing:
            f.write("%s    used in: %s\n" % (k, ", ".join(sorted(used[k]))))
    print("make_macros_ref: %d reference keys (+%d aliases) -> paper/results_macros_ref.tex; keymap %s" % (len(vals), n_alias, keymap))
    print("make_macros_ref: %d of %d \\res keys used in the paper have a reference value; %d without (see %s)" % (len(used) - len(missing), len(used), len(missing), miss_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
