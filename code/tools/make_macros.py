#!/usr/bin/env python3
"""results/*.json  ->  paper/results_macros.tex   (defines \\res{KEY}; stdlib only, Python 3.6 compatible, runs on the host)

    python3 code/tools/make_macros.py [--results results] [--paper paper] [--out paper/results_macros.tex]
                                      [--allow-missing] [--report results/_reports/missing_keys.txt]

1. Reads every top-level results/*.json (flat key -> value; keys starting with `_` and non-scalar values such as curves are skipped).
2. Formats each scalar for the text (rules below; first match wins; `paper/macro_formats.json` {regex: kind} overrides).
3. Writes \\gdef\\resv@<KEY>{<formatted>} for every key and for derived keys (`K` -> `K.mean`, `K.ci` -> `[lo, hi]`).
4. Scans paper/main.tex, paper/sections/*.tex, paper/tables/*.tex (comments removed) for \\res{KEY}; every key that cannot be produced is
   listed with the run / script expected to produce it (results/_reports/missing_keys.txt) and the exit code is 3 (unless --allow-missing).
   Missing keys still compile: \\res prints [KEY] in bold, as before.
5. Scope guard: result files whose _meta.scope is not 'confirmatory' / 'tuning-export' (tuning-set dry runs, synthetic self-tests) make the run fail (exit 4);
   `--preview` accepts synthetic files for layout checks and then requires --out below a directory whose name starts with '_'.
Red line (CONTRACT 0): no number reaches the paper except through this file.

Formatting kinds (metric = last key segment that is not a statistic suffix mean/ci_lo/ci_hi/item_ci_lo/item_ci_hi/sd):
  p      p-values (segments p, p_*, *_p): '<0.001' or 3 decimals
  pp     signed percentage points, 1 decimal (delta_ex, delta_pp, diff_pp, *.delta, loo_*, gain/harm in pp)
  pct    fraction -> percent, 1 decimal (ex, coverage, soft_f1, repair_rate, prevalence, recall, ...)
  x3     3 decimals (auroc, aurc, ece, brier, kappa, ac1, spearman, logodds, slope, sigma)
  x3s    signed 3 decimals (dauroc, daurc, dece)
  ppf    signed fraction-difference -> percentage points (dacc_ex_cov*)
  x2     2 decimals (calls, ratios, odds ratios, extra calls)
  x1     1 decimal (ves, rves, latency)
  tok    integer, thousands separated (tokens)
  cny    4 decimals (cost per question) / 2 decimals (cost totals)
  int    integer (n, counts)
  str    text, LaTeX-escaped"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STAT_SUFFIX = ("mean", "ci_lo", "ci_hi", "item_ci_lo", "item_ci_hi", "sd")

# first match wins; applied to the full key
FORMAT_RULES = [
    (r"(^|\.)(p|p_[a-z0-9_]+|[a-z0-9_]+_p|p_adj|p_adj_item)$", "p"),
    (r"(^|\.)(p_1s_margin[0-9_]+)$", "p"),
    (r"\.(dacc_ex_cov[0-9]+)(\.|$)", "ppf"),
    (r"\.(dauroc|daurc|dece|dbrier)(\.|$)", "x3s"),
    (r"(^|\.)(delta_ex|delta_pp|diff_pp|diff_rate_pp|loo_delta_pp|overall_pp|delta_vs_full|delta|gain_gated_vs_zs|margin_pp)(\.|$)", "pp"),
    (r"\.(loo_min|loo_max)$", "pp"),
    (r"\.(logodds|slope|slope_diff|sigma_[a-z]+|kappa|kappa_l5|ac1|spearman|rank_corr|auroc|aurc|ece|brier|ece_raw|ece_cal|brier_raw|brier_cal|delong_se|q|auc|b_ref_calls)(\.|$)", "x3"),
    (r"\.(rank_corr|spearman)$", "x3"),
    (r"(^|\.)(or|or_lo|or_hi|odds_ratio|calls|calls_ratio|calls_gated|calls_fixed|calls_full|calls_bstar|extra_calls|k_samples|tokens_ratio)(\.|$)", "x2"),
    (r"(^|\.)(ves|rves|latency_mean|latency_p50|latency_p95|latency_mean_s)(\.|$)", "x1"),
    (r"\.(tokens_per_q|prompt_tokens|completion_tokens|tokens|tokens_per_correct)(\.|$)", "tok"),
    (r"\.(cost_per_q|cost_per_correct|embed_rerank_cny)(\.|$)", "cny4"),
    (r"\.(cost_total|cost_per_1000q|cost_k)(\.|$)", "cny2"),
    (r"\.(n|n_[a-z_]+|b|c|n10|n01|df|k|m|n_R|api_fail|parse_fail|count|n_dbs|n_units)$", "int"),
    (r"(^|\.|_)family_m$", "int"),
    (r"^E7\.(cov[0-9]+|deferral(_cov[0-9]+)?)$", "int"),
    (r"\.(rs_c[0-9N]+(_cov[0-9]+)?|own\.rs_c[0-9N]+)(\.|$)", "x3"),
    (r"\.(ex|ex_[a-z_]+|ex_new|ex_ref|coverage|soft_f1|jaccard|silent_error|silent_share_of_errors|exec_ok|repair_rate|executable_rate|blind_rate|prevalence|share|recall|precision|fpr|clean_share_of_evicted|"
     r"final_ex|init_ex|calls_saved|acc_ex_cov[0-9]+|q_repro_[0-9]+|sql_exact|table_overlap|same_sign_share|top_db_share|revised_share|percent_agreement|abstained_wrong_ex|"
     r"ex_with_oracle_human|ex_rate|ex_root|ex_final|ex_accepted|ex_all|ex_rep|ves_coverage|ex_ts|rate|fail_rate|gain_rate)(\.|$)", "pct"),
    (r"\.(auc)\.", "pct"),
    (r"\.(ex_rep)\.sd$", "pct"),
]
ALIASES = [                                                  # requested key regex -> producing key template
    (r"^E10\.repair\.(.+)\.rate$", r"P5.by_class.\1.repair_rate.mean"),
    (r"^E1\.full_vs_best\.delta_ex$", r"E1.full_vs_best.delta_ex.mean"),
    (r"^E1\.full_vs_best\.ci$", r"E1.full_vs_best.delta_ex.ci"),
    (r"^setup\.e10\.n_annotated$", r"E10.n_annotated"),
    # short names used by the STORYLINE R3/R4 sentence templates -> keys the pipeline writes
    (r"^P1\.ex_gated$", r"P1.a.ex_new"), (r"^P1\.ex_fixed$", r"P1.a.ex_ref"),
    (r"^P1\.delta_ci$", r"P1.delta_ex.ci"), (r"^P2\.delta_ci$", r"P2.delta_ex.ci"),
    (r"^P2\.ex_bstar$", r"P2.ex_ref"), (r"^P2\.ex_full$", r"P2.ex_new"),
    (r"^P3\.auroc_v3$", r"P3.v3.auroc.mean"), (r"^P3\.auroc_ca$", r"P3.route_ca.auroc.mean"),
    (r"^P3\.dauroc_ci$", r"P3.dauroc.ci"),
    (r"^P4\.ex_ws$", r"P4.ex_a"), (r"^P4\.ex_append$", r"P4.ex_b"), (r"^P4\.ci$", r"P4.logodds.ci"),
    (r"^E1\.gated_vs_zs\.delta$", r"E1.full_vs.zs.delta_ex.mean"),
    (r"^E1\.gated_vs_zs\.ci$", r"E1.full_vs.zs.delta_ex.ci"),
    (r"^E1\.gated_vs_zs\.p_adj$", r"E1.full_vs.zs.p_adj"),
    (r"^E4\.p0\.delta$", r"E4.ws_full.p0.noharm.delta.mean"),
    (r"^S-E\.memrl\.delta$", r"E4.memrl_vs_ws.p40.delta.mean"),
    (r"^E7\.ca\.(.+)$", r"E7.route_ca.\1.mean"),
]
# SAP Amendment 3 (15.3) replaced family P of 13.3 by P*. Keys the paper writes as P1.* / P3.* / P5.* / P2.* / P4.* / P.family.* mean the P* (or secondary-row) values; the
# single-backbone 13.3 values stay in primary.json under the same names. These redirects take PRECEDENCE over the direct key; a redirect whose target does not exist makes the key
# missing (it is never silently served from the superseded family). First match wins.
_SFX = r"(\.(?:mean|ci|ci_lo|ci_hi|fmt|item_ci_lo|item_ci_hi))?"
OVERRIDES = [
    (r"^P1\.(delta_ex|calls_ratio|calls_saved)" + _SFX + "$", r"P1prime.\1\2"),       # "averaged over backbones" (pooled P1', 15.3)
    (r"^P1\.(p_adj|verdict)$", r"P1prime.\1"),
    (r"^P3\.(p_adj|verdict)$", r"Pstar.P3.\1"),
    (r"^P5\.(repair_rate|executable_rate|blind_rate|diff_rate_pp)" + _SFX + "$", r"P5prime.\1\2"),
    (r"^P5\.(n_R|verdict|p_adj)$", r"P5prime.\1"),
    (r"^P2\.(p_adj|verdict)$", r"P2.\1_sa"),                                            # secondary rows: Holm inside S-A / S-E (15.2)
    (r"^P4\.(p_adj|verdict)$", r"P4.\1_se"),
    (r"^P\.family\.(m|members|dropped)$", r"Pstar.family.\1"),
]
# key regex -> what produces it (shown in the missing-key report)
KEY_PRODUCERS = [
    (r"^setup\.(bird_dev)\.", "analysis_pipeline.misc.run_setup -> results/setup.json; needs a complete dev run (any system) under runs/"),
    (r"^setup\.(holdout|hr|ht)\.", "analysis_pipeline.misc.run_setup; needs exp/splits/train_holdout.json"),
    (r"^setup\.devlock\.", "exp/dev_lock.json (written by the author at the dev lock) -> run_setup"),
    (r"^setup\.plan\.commit", "git log of docs/SAP.md -> run_setup"),
    (r"^setup\.seeds", "run_setup (constant 0, 101, 202)"),
    (r"^W1\.", "weak.run_w1 (stage W1): dev r0 full_gated + zs on qwen37flash and qwen3527b (MINIMAL_PLAN M5'; SAP 15.3); p_adj/verdict need the primary stage in the same run_all call"),
    (r"^P1\.(n_backbones|qwen[a-z0-9]+\.)", "weak.run_p1_backbones (stage W1): dev r0 full_gated + full_fixed on every backbone of Bk (SAP 15.3)"),
    (r"^P1prime\.|^Pstar\.|^P5prime\.", "weak.run_p1_backbones / run_p5prime / run_pstar (stage W1): pooled P1', P5' and family P* (SAP 15.3); Holm values need the primary stage in the same run_all call"),
    (r"^P[24]\.(p_adj|verdict)_s[ae]$", "weak.run_secondary_rows (stage Psec): P2 in S-A, P4 in S-E (SAP 15.2); needs the E1 / E4 stages in the same call"),
    (r"^X1\.", "weak.run_x1 (stage X1): qwen37flash full_gated r0 vs qwen37plus zs r0 on dev + exp/prices.yaml (SAP 15.5 S-I, descriptive)"),
    (r"^setup\.design_effect", "misc.run_power_constants: PLANNING design effect 1 + (m-1)*rho, rho = 0.01 assumed (SAP 9); the observed analogues are setup.design_effect_emp.*"),
    (r"^setup\.(resolution_pp|p4_mde_pp|boot_resolution_pp)", "misc.run_power_constants (SAP 9 / 13.7 definitions; n from the dev runs, probe items from the E4 units)"),
    (r"^setup\.e10\.", "E10 annotation (MINIMAL_PLAN M6) has not been run: E10.n_annotated exists only with --annot-dir"),
    (r"^P1\.", "primary.py P1: dev runs M1 full_gated r0 + full_fixed r0 (MINIMAL_PLAN M1)"),
    (r"^P2\.", "primary.py P2: M1 full run + M2 matched baselines (self_debug, ac_sql, memo_sql, bon_vote; zs/dail from M1/D1)"),
    (r"^P3\.", "primary.py P3: routing run on dev (config full_gated_cons_lr: V3 of the final answer + k extra samples; SAP D4) + exp/frozen/route_coef_hr.json"),
    (r"^P4\.", "primary.py P4: E4 stream units (M4: ws_full vs append at p = 40, >= 2 permutations) passed with --e4-roots"),
    (r"^P5\.", "primary.py P5: full_gated r0 (cands.jsonl) + full_gated_blind r0 (SAP D3, NOT in MINIMAL_PLAN) [+ abl_q legacy arm]"),
    (r"^P\.", "primary.py family bookkeeping"),
    (r"^E1\.(.+)\.(corrected)", "corrected-gold re-scoring: NOT IMPLEMENTED (needs the corrected-label release; E10 follow-up)"),
    (r"^E1\.", "secondary.run_e1: dev r0 runs of the systems (M1, M2)"),
    (r"^E11\.", "secondary.run_e11: same runs as E1 (usage.jsonl for cost)"),
    (r"^E2\.", "secondary.run_e2: M5 runs (zs + full_gated on qwen37flash, qwen3527b) plus the primary backbone"),
    (r"^E3\.", "secondary.run_e3: M3 ablation runs (abl_m1, abl_m3/full_m3force, abl_m5, abl_q) + full_fixed"),
    (r"^E4\.", "misc.run_e4: E4 stream units (M4)"),
    (r"^E5\.", "e7.run_e5: full_fixed r0 with cands.jsonl (stored candidates; V1-V4 end-to-end variants were dropped by MINIMAL_PLAN)"),
    (r"^E6\.", "dropped by MINIMAL_PLAN (live RL runs); tuning-set CV only (T.p2.e6_summary.*)"),
    (r"^E7\.ambiqt", "AmbiQT ambiguity detection: NOT IMPLEMENTED (needs an ambiqt_mix run; add to e7.py when the run exists)"),
    (r"^E7\.", "e7.run_e7: routing run (see P3)"),
    (r"^E8\.", "misc.run_e8: Spider-Syn/Realistic/DK runs (dropped unless budget allows)"),
    (r"^E9\.", "misc.run_e9: tools/e9_probes.py summary.json files (M6) via --e9-dirs"),
    (r"^E10\.spot\.", "authors' blind spot-check of the LLM labels (human work, optional \\PH item in 04_setup): no producer; delete the item or add exp/annotation spot file and extend e10.py"),
    (r"^E10\.", "e10.run_e10: annotation directory (annotation_key.csv, A1/A2 labels, adjudicated.csv, llm_labels.*) via --annot-dir"),
    (r"^T\.", "misc.run_tuning: results/r1/*.json, results/p2/*.json (tuning set)"),
]
RES_RE = re.compile(r"\\res\{([^{}]+)\}")


# ------------------------------------------------------------------------------------------ formatting
def _tex_escape(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in str(s))


def _minus(txt):
    return txt.replace("-", r"\ensuremath{-}")


def _num(v, kind):
    sign = lambda x, s: (("+" if x > 0 else "") + s)
    if kind == "p":
        return "<0.001" if v < 0.001 else "%.3f" % v
    if kind == "pp":
        return sign(v, "%.1f" % v)
    if kind == "ppf":
        return sign(v, "%.1f" % (100.0 * v))
    if kind == "pct":
        return "%.1f" % (100.0 * v)
    if kind == "x3":
        return "%.3f" % v
    if kind == "x3s":
        return sign(v, "%.3f" % v)
    if kind == "x2":
        return "%.2f" % v
    if kind == "x1":
        return "%.1f" % v
    if kind == "tok":
        return "{:,}".format(int(round(v)))
    if kind == "cny4":
        return "%.4f" % v
    if kind == "cny2":
        return "%.2f" % v
    if kind == "int":
        return "%d" % int(round(v))
    return ("%d" % v) if float(v).is_integer() and abs(v) > 1 else "%.3f" % v


def classify(key, overrides):
    for rx, kind in overrides:
        if re.search(rx, key):
            return kind
    for rx, kind in FORMAT_RULES:
        if re.search(rx, key):
            return kind
    return None


def fmt_value(key, v, overrides):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, str):
        return _tex_escape(v)
    if isinstance(v, int) and classify(key, overrides) in (None, "int"):
        return _minus(str(v))
    kind = classify(key, overrides)
    return _minus(_num(float(v), kind or "default"))


# ------------------------------------------------------------------------------------------ loading
def load_results(results_dir, preview=False):
    vals, src, shas, errors = {}, {}, {}, []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        name = os.path.basename(f)
        if name.startswith("_"):
            continue
        raw = open(f, "rb").read()
        shas[name] = hashlib.sha256(raw).hexdigest()
        try:
            d = json.loads(raw.decode("utf-8"))
        except ValueError as e:
            errors.append("%s: invalid JSON (%s)" % (name, e))
            continue
        if not isinstance(d, dict):
            errors.append("%s: top level must be a flat object" % name)
            continue
        scope = (d.get("_meta") or {}).get("scope", "confirmatory") if isinstance(d.get("_meta"), dict) else "confirmatory"
        if scope not in (("confirmatory", "tuning-export", "synthetic") if preview else ("confirmatory", "tuning-export")):       # dry runs on tuning data / synthetic self-tests must never feed the paper
            errors.append("%s: _meta.scope = %r (only 'confirmatory' and 'tuning-export' results may sit in %s)" % (name, scope, results_dir))
            continue
        for k, v in d.items():
            if k.startswith("_") or isinstance(v, (list, dict)) or v is None:
                continue
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                errors.append("%s: %s is %r" % (name, k, v))
                continue
            if k in vals and vals[k] != v:
                errors.append("key %s defined in %s and %s with different values (%r vs %r)" % (k, src[k], name, vals[k], v))
                continue
            vals[k] = v
            src[k] = name
    return vals, src, shas, errors


def derive(vals, overrides):
    """-> dict key -> formatted text, including derived K (from K.mean) and K.ci."""
    out = {}
    for k, v in vals.items():
        try:
            out[k] = fmt_value(k, v, overrides)
        except (TypeError, ValueError) as e:
            raise SystemExit("cannot format %s=%r: %s" % (k, v, e))
    # cost per 1 000 questions (CNY, two decimals) next to every cost per question: `<x>.cost_per_q[.mean|.ci_lo|.ci_hi]` -> `<x>.cost_per_1000q[...]` (+ alias `.cost_k`, the name of the EXT1 / PF keys)
    for k in list(vals):
        m_ = re.match(r"^(.*)\.cost_per_q(\.mean|\.ci_lo|\.ci_hi)?$", k)
        if m_ and isinstance(vals[k], (int, float)) and not isinstance(vals[k], bool):
            for nm in ("cost_per_1000q", "cost_k"):
                k2 = "%s.%s%s" % (m_.group(1), nm, m_.group(2) or "")
                if k2 not in vals:
                    vals2 = 1000.0 * float(vals[k])
                    out[k2] = fmt_value(k2, vals2, overrides)
                    if (m_.group(2) or "") in ("", ".mean"):
                        out.setdefault("%s.%s" % (m_.group(1), nm), out[k2])
                    lo_, hi_ = "%s.%s.ci_lo" % (m_.group(1), nm), "%s.%s.ci_hi" % (m_.group(1), nm)
                    if (m_.group(2) or "") == ".ci_hi" and lo_ in out:
                        out["%s.%s.ci" % (m_.group(1), nm)] = "[%s, %s]" % (out[lo_], out[hi_])
    for k in list(vals):
        if k.endswith(".mean"):
            base = k[:-5]
            out.setdefault(base, out[k])
            lo, hi = base + ".ci_lo", base + ".ci_hi"
            if lo in out and hi in out:
                out[base + ".ci"] = "[%s, %s]" % (out[lo], out[hi])
                out[base + ".fmt"] = "%s %s" % (out[k], out[base + ".ci"])
        if k.endswith(".ci_lo"):
            base = k[:-6]
            hi = base + ".ci_hi"
            if hi in out:
                out.setdefault(base + ".ci", "[%s, %s]" % (out[k], out[hi]))
    return out


TEX_SAFE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-+=<>/,()[]@*!?|:;'")


def tex_name(key):
    """key as it may appear inside \\csname ... \\endcsname in a .tex file. Characters outside TEX_SAFE (% comment, ~ active, # parameter, & $ ^ _-like specials, \\ escape, braces, quotes,
    spaces, non-ASCII ...) are written as \\expandafter\\@gobble\\string\\<char>: one catcode-12 character token, exactly what \\csname builds from `\\res{key}` at use time, so the
    lookup works for the raw key. (`\\string%` and `\\string~` do not work: the tokenizer comments out / activates the character before \\string sees it.) The macro file is read between
    \\makeatletter and \\makeatother, so \\@gobble is available. `_` is catcode 8 but legal inside \\csname and is kept (thousands of keys use it)."""
    return "".join(c if (c in TEX_SAFE or c == "_") else ("\\expandafter\\@gobble\\string\\" + c) for c in key)


def unsafe_keys(keys):
    """keys whose spelling in the macro file differs from the key (they still resolve by their raw name through \\res / \\csname)."""
    return sorted(k for k in keys if tex_name(k) != k)


def check_tex(keys, macros_path, workdir):
    """compile a tiny document that loads the macro file and tests \\ifcsname for EVERY key (spelled with tex_name) and resolves each through the `\\csname resv@<key>` route that \\res uses.
    -> (ok, log tail). Skipped (ok=None) when pdflatex is missing."""
    import shutil
    if not shutil.which("pdflatex"):
        return None, "pdflatex not found"
    os.makedirs(workdir, exist_ok=True)
    body = ["\\documentclass{article}\\def\\resfallback#1{[#1]}", "\\input{%s}" % os.path.abspath(macros_path).replace("\\", "/"), "\\begin{document}", "\\makeatletter", "\\newcount\\badkeys"]
    for k in sorted(keys):
        body.append("\\expandafter\\ifx\\csname resv@%s\\endcsname\\relax\\advance\\badkeys by 1\\typeout{BADKEY %d}\\fi" % (tex_name(k), len(k)))
    body += ["\\typeout{BADKEYS=\\the\\badkeys}", "\\makeatother", "\\end{document}"]
    tex = os.path.join(workdir, "keycheck.tex")
    open(tex, "w", encoding="utf-8").write("\n".join(body) + "\n")
    pr = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "keycheck.tex"], cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = pr.stdout.decode("utf-8", "replace")
    return (pr.returncode == 0 and "BADKEYS=0" in log and "Missing" not in log), log[-400:]


def apply_overrides(out, extra=()):
    """redirect the keys of OVERRIDES in place; -> {key: target} actually redirected, and the list of keys dropped because the target is absent."""
    done, dropped = {}, []
    for k in sorted(set(out) | set(extra)):                      # `extra` = keys the paper uses that the results do not carry under that name (e.g. P5.p_adj)
        for rx, tmpl in OVERRIDES:
            m = re.match(rx, k)
            if m:
                tgt = m.expand(tmpl)
                if tgt in out and tgt != k:
                    done[k] = (tgt, out[tgt])
                else:
                    dropped.append(k)
                break
    for k, (tgt, v) in done.items():
        out[k] = v
    for k in dropped:
        out.pop(k, None)
    return {k: t for k, (t, _) in done.items()}, dropped


def resolve(key, out, vals):
    if key in out:
        return out[key]
    for rx, tmpl in ALIASES:
        m = re.match(rx, key)
        if m:
            tgt = m.expand(tmpl)
            if tgt in out:
                return out[tgt]
    return None


def strip_comments(tex):
    lines = []
    for ln in tex.splitlines():
        out, i = [], 0
        while i < len(ln):
            c = ln[i]
            if c == "\\" and i + 1 < len(ln):
                out.append(ln[i:i + 2])
                i += 2
                continue
            if c == "%":
                break
            out.append(c)
            i += 1
        lines.append("".join(out))
    return "\n".join(lines)


def scan_sources(paper_dir):
    files = [os.path.join(paper_dir, "main.tex")] + sorted(glob.glob(os.path.join(paper_dir, "sections", "*.tex"))) + sorted(glob.glob(os.path.join(paper_dir, "tables", "*.tex")))
    # round 5 (V5 L-2): the supplement uses \res keys too; its text, its generated tables and its main file are scanned like the main text
    files += [os.path.join(paper_dir, "supplement.tex")] + sorted(glob.glob(os.path.join(paper_dir, "supp_text", "*.tex"))) + sorted(glob.glob(os.path.join(paper_dir, "supp_tables", "*.tex")))
    used = {}
    for f in files:
        if not os.path.exists(f):
            continue
        txt = strip_comments(open(f, encoding="utf-8", errors="replace").read())
        for m in RES_RE.finditer(txt):
            k = m.group(1).strip()
            if k == "KEY" or "#" in k:
                continue
            used.setdefault(k, set()).add(os.path.relpath(f, paper_dir))
    kf = os.path.join(paper_dir, "res_keys_used.txt")        # release: the manuscript sources are not distributed; this file lists the keys they reference
    if os.path.exists(kf):
        for ln in open(kf, encoding="utf-8"):
            k = ln.strip()
            if k and not k.startswith("#"):
                used.setdefault(k, set()).add("res_keys_used.txt")
    return used


def producer_of(key):
    for rx, desc in KEY_PRODUCERS:
        if re.search(rx, key):
            return desc
    return "no registered producer - check the key spelling against docs/ANALYSIS_RUNBOOK.md (key catalogue)"


def status_hint(key, vals):
    pre = key.split(".")[0]
    cand = [k for k in vals if k.endswith(".status") and (k.split(".")[0] == pre)]
    return "; ".join("%s = %s" % (k, vals[k]) for k in sorted(cand)[:3])


# ------------------------------------------------------------------------------------------ main
def git_hash():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=os.path.join(REPO, "results"))
    ap.add_argument("--paper", default=os.path.join(REPO, "paper"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--check-tex", action="store_true", help="compile a document that tests every generated key with pdflatex (CI gate for unsafe characters in key names)")
    ap.add_argument("--allow-missing", action="store_true", help="exit 0 although referenced keys are missing (drafts)")
    ap.add_argument("--preview", action="store_true", help="accept synthetic results (layout previews); --out must be below a directory starting with '_'")
    a = ap.parse_args(argv)
    if a.preview and not (a.out and any(x.startswith("_") for x in os.path.abspath(a.out).split(os.sep))):
        sys.stderr.write("make_macros: --preview needs --out below a directory whose name starts with '_' (e.g. paper/_preview/results_macros.tex)\n")
        return 5
    out_path = a.out or os.path.join(a.paper, "results_macros.tex")
    report = a.report or os.path.join(a.results, "_reports", "missing_keys.txt")
    overrides = []
    ov = os.path.join(a.paper, "macro_formats.json")
    if os.path.exists(ov):
        overrides = [(k, v) for k, v in json.load(open(ov)).items()]
    vals, src, shas, errors = load_results(a.results, a.preview)
    if errors:
        sys.stderr.write("make_macros: INVALID RESULTS\n  " + "\n  ".join(errors) + "\n")
        return 4
    out = derive(vals, overrides)
    used = scan_sources(a.paper)
    redirected, dropped = apply_overrides(out, used)
    missing = {k: v for k, v in used.items() if resolve(k, out, vals) is None}
    unformatted = sorted(k for k in vals if isinstance(vals[k], float) and classify(k, overrides) is None and k in {u for u in used})
    lines = ["% GENERATED by code/tools/make_macros.py -- do not edit. Source: results/*.json",
             "%% generated %s, git %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), git_hash())]
    for n, h in sorted(shas.items()):
        lines.append("%% %s sha256 %s" % (n, h))
    lines += ["%% %d keys redirected to the family P* / secondary-row values (OVERRIDES in make_macros.py): %s" % (len(redirected), ", ".join(sorted(k for k in redirected if k in used)))]
    lines += ["\\makeatletter",
              "\\def\\res#1{\\@ifundefined{resv@#1}{\\resfallback{#1}}{\\csname resv@#1\\endcsname}}"]
    for k in sorted(out):
        lines.append("\\expandafter\\gdef\\csname resv@%s\\endcsname{%s}" % (tex_name(k), out[k]))
    for rx, tmpl in ALIASES:                                    # materialise aliases that are referenced
        for k in used:
            m = re.match(rx, k)
            if m and k not in out:
                tgt = m.expand(tmpl)
                if tgt in out:
                    lines.append("\\expandafter\\gdef\\csname resv@%s\\endcsname{%s}" % (tex_name(k), out[tgt]))
    lines.append("\\makeatother")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("make_macros: %d keys from %d result files -> %s" % (len(out), len(shas), out_path))
    uk = unsafe_keys(out)
    if uk:
        print("make_macros: %d keys contain characters TeX cannot take raw in a control-sequence name; written escaped (name unchanged for \\res): %s" % (len(uk), ", ".join(uk[:5]) + (" ..." if len(uk) > 5 else "")))
    if dropped:
        print("make_macros: %d redirected keys dropped (target absent): %s" % (len(dropped), ", ".join(sorted(dropped)[:10])))
    print("make_macros: %d distinct \\res keys referenced in %s; %d resolved, %d missing" % (len(used), a.paper, len(used) - len(missing), len(missing)))
    if unformatted:
        print("make_macros: WARNING no format rule for: %s (printed with the default rule)" % ", ".join(unformatted))
    if a.check_tex:
        ok, tail = check_tex(list(out) + [k for k in used if k in out], out_path, tempfile.mkdtemp(prefix="keycheck_"))
        print("make_macros: --check-tex over %d keys: %s" % (len(out), {True: "OK", False: "FAILED", None: "skipped (%s)" % tail}[ok]))
        if ok is False:
            print(tail)
            return 6
    os.makedirs(os.path.dirname(os.path.abspath(report)), exist_ok=True)
    with open(report, "w", encoding="utf-8") as f:
        f.write("# keys referenced by the LaTeX sources that results/*.json cannot provide yet (%s)\n" % time.strftime("%Y-%m-%d %H:%M"))
        for k in sorted(missing):
            hint = status_hint(k, vals)
            f.write("%s\n    used in: %s\n    expected from: %s\n%s" % (k, ", ".join(sorted(missing[k])), producer_of(k), ("    status: %s\n" % hint) if hint else ""))
    if missing:
        print("make_macros: MISSING KEYS (full list with producers: %s)" % report)
        for k in sorted(missing)[:60]:
            print("  %-48s <- %s" % (k, producer_of(k)[:100]))
        if len(missing) > 60:
            print("  ... %d more" % (len(missing) - 60))
        if not a.allow_missing:
            print("make_macros: FAILING because keys are missing (use --allow-missing for drafts)")
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
