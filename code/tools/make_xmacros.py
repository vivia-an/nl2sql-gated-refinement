#!/usr/bin/env python3
"""exploratory / robustness / external-validity results  ->  paper/results_macros_x.tex   (defines \\xres{KEY}; stdlib only, Python 3.6, runs on the host)

    python3 code/tools/make_xmacros.py [--results results] [--out paper/results_macros_x.tex] [--allow-missing]
                                       [--report results/_reports/missing_xkeys.txt] [--doc internal note XMACROS]

Sources (nothing else is read; every value comes from these files, no number is typed in this script):
  ROB.   results/robust/rescore.json, overlap.json        scope robustness-posthoc   (SAP 20)
  DIAG.  results/diag/critic_diag_summary.json            scope diagnostic-injected  (SAP 16)
  FBK.   results/diag/feedback_calibration.json           scope diagnostic-posthoc   (SAP 17)
  VAD.   results/diag/critic_v2_summary.json              scope diagnostic-posthoc   (SAP 18)
  VE.    results/diag/critic_ve_summary.json              scope diagnostic-posthoc   (SAP 19)
  EXT1.  results/ext1/ext1_summary.json                   scope external-validity    (SAP 21)
  E10.   results/e10/agreement.json, discordant_summary.json (LLM annotation, exploratory)
  PF.    results/pf/pf_summary.json                       scope exploratory-posthoc  (SAP 22, Amendment 9); simulated oracle feedback; skipped while the file is absent
Formatting follows make_macros.py / STYLE_SPEC section 7 (kinds below); the true minus is \\ensuremath{-}; an interval is "[lo, hi]".
For every estimate K with an interval the script writes K, K.ci, K.ci_lo, K.ci_hi, K.fmt ("est [lo, hi]"); a sample size is K.n (and K.k for counts).
Keys that tables print with two decimals carry a twin K.d2 (and K.ci.d2).
Strict mode (default): a missing source file or field, or a file whose _meta.scope is wrong, fails (exit 3 / 4) and nothing is written.
--allow-missing writes what exists and lists the rest in --report (exit 0). The macro file is never written in strict mode on error."""
import argparse
import collections
import hashlib
import json
import math
import os
import re
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MISSING = object()

# ------------------------------------------------------------------------------------------ sources
SOURCES = collections.OrderedDict([
    ("rescore", ("robust/rescore.json", ("robustness-posthoc",))),
    ("overlap", ("robust/overlap.json", ("robustness-posthoc",))),
    ("diag", ("diag/critic_diag_summary.json", ("diagnostic-injected",))),
    ("fbk", ("diag/feedback_calibration.json", ("diagnostic-posthoc",))),
    ("v2", ("diag/critic_v2_summary.json", ("diagnostic-posthoc",))),
    ("ve", ("diag/critic_ve_summary.json", ("diagnostic-posthoc",))),
    ("ext1", ("ext1/ext1_summary.json", ("external-validity",))),
    ("e10agree", ("e10/agreement.json", (None,))),
    ("e10disc", ("e10/discordant_summary.json", (None,))),
    ("pf", ("pf/pf_summary.json", ("exploratory-posthoc", "diagnostic-posthoc"))),
    ("rv2", ("posthoc2/rv2_summary.json", ("posthoc-secondary",))),
    ("rv2s", ("posthoc2/rv2s_summary.json", ("posthoc-secondary",))),
    ("rv3", ("r3/rv3_summary.json", ("posthoc-secondary",))),
    ("rv4", ("r4/rv4_summary.json", ("posthoc-secondary",))),
    ("rv4t0", ("r4_t0/rv4t0_summary.json", ("posthoc-secondary",))),
])
LABELS = {
    "ROB": "ROB (robustness-posthoc)",
    "DIAG": "EXPL (injected errors, diagnostic)",
    "FBK": "EXPL (simulated oracle feedback, post hoc)",
    "VAD": "EXPL (injected errors, post hoc)",
    "VE": "EXPL (simulated user, post hoc)",
    "EXT1": "EXT (external-validity)",
    "E10": "EXPL (LLM annotation)",
    "PF": "EXPL (feedback-updated Critic, simulated oracle feedback, post hoc)",
    "RV2": "POSTHOC2 (post hoc, following a critical reading of the draft by the authors with Claude Sonnet 5.5 (see Acknowledgments); computed on existing runs; SAP Amendment 10)",
    "RV3": "POSTHOC3 (post hoc, requested by the round-3 referees; new runs for seeds and the same-examples control; SAP section 26)",
    "RV4T0": "POSTHOC4-T0 (post hoc, requested by the round-4 referees; temperature-0 rerun of part of the locked run, reduced scope after the loss of compute worker machines; SAP section 27)",
    "RV4": "POSTHOC4 (post hoc, requested by the round-4 referees; computed on existing runs, no model call; SAP section 27)",
}


# ------------------------------------------------------------------------------------------ formatting
def _minus(txt):
    return txt.replace("-", r"\ensuremath{-}")


def _tex_escape(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in str(s))


def _signed(v, nd):
    s = "%.*f" % (nd, v)
    if float(s) == 0.0:                      # no "-0.0" and no "+0.0"
        return "%.*f" % (nd, 0.0)
    return ("+" if v > 0 else "") + s


TWO_DECIMAL = {"pp": "pp2", "fpp": "fpp2", "pctv": "pctv2"}


def fnum(v, kind):
    """one number -> text (before the minus is typeset). kinds: pp fpp (fraction -> pp) pct (fraction -> %) pctv (already %) pctv2 x3 x3s x2 x2s x1 int p tok cny4 cny2 + pp2 fpp2."""
    v = float(v)
    if kind == "pp":
        return _signed(v, 1)
    if kind == "pp2":
        return _signed(v, 2)
    if kind == "fpp":
        return _signed(100.0 * v, 1)
    if kind == "fpp2":
        return _signed(100.0 * v, 2)
    if kind == "pct":
        return "%.1f" % (100.0 * v)
    if kind == "pctv":
        return "%.1f" % v
    if kind == "pctv2":
        return "%.2f" % v
    if kind == "x3":
        return "%.3f" % v
    if kind == "x3s":
        return _signed(v, 3)
    if kind == "p4":
        return "<0.0001" if v < 0.0001 else "%.4f" % v
    if kind == "x2":
        return "%.2f" % v
    if kind == "x2s":
        return _signed(v, 2)
    if kind == "x1":
        return "%.1f" % v
    if kind == "int":
        return "%d" % int(round(v))
    if kind == "p":
        return "<0.001" if v < 0.001 else "%.3f" % v
    if kind == "p4":
        return "<0.0001" if v < 0.0001 else "%.4f" % v
    if kind == "tok":
        return "{:,}".format(int(round(v)))
    if kind == "cny4":
        return "%.4f" % v
    if kind == "cny2":
        return "%.2f" % v
    raise ValueError("unknown kind %r" % kind)


def render_raw(raw, kind):
    if isinstance(raw, bool):
        return "yes" if raw else "no"
    if kind == "str":
        return _tex_escape(raw)
    if isinstance(raw, tuple):
        return "[%s, %s]" % (_minus(fnum(raw[0], kind)), _minus(fnum(raw[1], kind)))
    return _minus(fnum(raw, kind))


class Entry(object):
    __slots__ = ("key", "raw", "kind", "src", "field", "meaning", "label", "two")

    def __init__(self, key, raw, kind, src, field, meaning, label, two=False):
        self.key, self.raw, self.kind, self.src, self.field, self.meaning, self.label, self.two = key, raw, kind, src, field, meaning, label, two

    def text(self, kind=None):
        return render_raw(self.raw, kind or self.kind)


# ------------------------------------------------------------------------------------------ context
class Ctx(object):
    def __init__(self, results_dir, allow_missing=False):
        self.dir = results_dir
        self.allow_missing = allow_missing
        self.data, self.paths, self.shas = {}, {}, {}
        self.entries = collections.OrderedDict()
        self.errors = []
        self.scope_errors = []
        self.na = []                      # fields that are null in the source by design (undefined statistic); listed in the catalogue
        self.absent = set()
        for name, (rel, scopes) in SOURCES.items():
            p = os.path.join(results_dir, rel)
            self.paths[name] = "results/" + rel
            if not os.path.exists(p):
                self.absent.add(name)
                continue
            raw = open(p, "rb").read()
            self.shas["results/" + rel] = hashlib.sha256(raw).hexdigest()
            d = json.loads(raw.decode("utf-8"))
            self.data[name] = d
            sc = (d.get("_meta") or {}).get("scope") if isinstance(d, dict) and isinstance(d.get("_meta"), dict) else None
            if sc not in scopes:
                self.scope_errors.append("%s: _meta.scope = %r, expected one of %r" % (self.paths[name], sc, scopes))
        self._reported_absent = set()
        self.prefix = ""
        self.group = None

    # --- access
    def get(self, src, path):
        if src in self.absent:
            if src not in self._reported_absent and not (src == "pf"):
                self._reported_absent.add(src)
                self.errors.append("source file missing: %s" % self.paths[src])
            return MISSING
        cur = self.data.get(src)
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            elif isinstance(cur, list) and isinstance(p, int) and 0 <= p < len(cur):
                cur = cur[p]
            else:
                self.errors.append("%s: missing field %s" % (self.paths[src], "/".join(str(x) for x in path)))
                return MISSING
        if cur is None:
            self.errors.append("%s: null at %s" % (self.paths[src], "/".join(str(x) for x in path)))
            return MISSING
        return cur

    def get_quiet(self, src, path):
        cur = self.data.get(src)
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return MISSING
        return cur

    def has(self, src, path):
        cur = self.data.get(src)
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return False
        return cur is not None

    def keys(self, src, path):
        cur = self.data.get(src)
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return []
        return list(cur.keys()) if isinstance(cur, dict) else []

    # --- adding
    def _put(self, key, raw, kind, src, field, meaning, two=False):
        full = self.prefix + key if not key.startswith(self.prefix) else key
        if full in self.entries:
            raise RuntimeError("duplicate key %s" % full)
        self.entries[full] = Entry(full, raw, kind, src, field, meaning, LABELS[self.group], two)
        return full

    def val(self, key, src, path, kind, meaning, scale=None, two=False):
        v = self.get(src, path)
        if v is MISSING:
            return None
        if scale is not None:
            v = scale(v)
        if isinstance(v, bool) or kind == "str":
            pass
        return self._put(key, v, kind, self.paths[src], "/".join(str(x) for x in path), meaning, two)

    def ci(self, key, src, path, kind, meaning, two=False):
        """path -> [lo, hi] list"""
        v = self.get(src, path)
        if v is MISSING:
            return None
        lo, hi = v
        f = "/".join(str(x) for x in path)
        self._put(key + ".ci", (lo, hi), kind, self.paths[src], f, "95% interval of " + key + " (" + meaning + ")", two)
        self._put(key + ".ci_lo", lo, kind, self.paths[src], f + "[0]", "lower end of " + key + ".ci", two)
        self._put(key + ".ci_hi", hi, kind, self.paths[src], f + "[1]", "upper end of " + key + ".ci", two)

    def ci2(self, key, src, p_lo, p_hi, kind, meaning, two=False):
        lo, hi = self.get(src, p_lo), self.get(src, p_hi)
        if lo is MISSING or hi is MISSING:
            return None
        f = "%s , %s" % ("/".join(str(x) for x in p_lo), "/".join(str(x) for x in p_hi))
        self._put(key + ".ci", (lo, hi), kind, self.paths[src], f, "95% interval of " + key + " (" + meaning + ")", two)
        self._put(key + ".ci_lo", lo, kind, self.paths[src], "/".join(str(x) for x in p_lo), "lower end of " + key + ".ci", two)
        self._put(key + ".ci_hi", hi, kind, self.paths[src], "/".join(str(x) for x in p_hi), "upper end of " + key + ".ci", two)

    def est(self, key, src, p_est, p_ci, kind, meaning, scale=None, two=False, n=None):
        """estimate + list interval (+ n)"""
        k = self.val(key, src, p_est, kind, meaning, scale, two)
        self.ci(key, src, p_ci, kind, meaning, two)
        if n is not None:
            self.val(key + ".n", src, n, "int", "sample size of " + key)
        return k

    def est2(self, key, src, p_est, p_lo, p_hi, kind, meaning, scale=None, two=False, n=None):
        k = self.val(key, src, p_est, kind, meaning, scale, two)
        self.ci2(key, src, p_lo, p_hi, kind, meaning, two)
        if n is not None:
            self.val(key + ".n", src, n, "int", "sample size of " + key)
        return k

    def rate(self, key, src, path, meaning, kind="pct"):
        """{n, k, rate, lo, hi} object -> K, K.ci*, K.n, K.k"""
        v = self.get(src, path)
        if v is MISSING:
            return None
        f = "/".join(str(x) for x in path)
        self._put(key, v["rate"], kind, self.paths[src], f + "/rate", meaning)
        self._put(key + ".ci", (v["lo"], v["hi"]), kind, self.paths[src], f + "/lo,hi", "95% interval of " + key)
        self._put(key + ".ci_lo", v["lo"], kind, self.paths[src], f + "/lo", "lower end of " + key + ".ci")
        self._put(key + ".ci_hi", v["hi"], kind, self.paths[src], f + "/hi", "upper end of " + key + ".ci")
        self._put(key + ".n", v["n"], "int", self.paths[src], f + "/n", "denominator of " + key)
        self._put(key + ".k", v["k"], "int", self.paths[src], f + "/k", "numerator of " + key)

    def derived(self, key, raw, kind, note, meaning, src_note):
        self._put(key, raw, kind, src_note, "derived: " + note, meaning)

    def raw_of(self, full):
        e = self.entries.get(full)
        return e.raw if e is not None else MISSING


def start(ctx, prefix, group):
    ctx.prefix, ctx.group = prefix, group


# ------------------------------------------------------------------------------------------ ROB
def rob_contrast_specs(ctx):
    """name -> (path below analysis, unit kind, has_rule, has_sens). Names of contrasts come from the file (contrasts/*) with a fixed renaming."""
    ren = {"q27b": "27b", "pooled3": "pooled", "pooled_weak": "pooled", "plus_control": "plus"}
    out = collections.OrderedDict()
    base = ("variants", "G1", "sets", "S_same", "analysis")
    for ck in ctx.keys("rescore", base + ("contrasts",)):
        parts = ck.split(".")
        if ck.startswith("H2b."):
            _, b, what = parts[0], parts[1], ".".join(parts[2:])
            nm = "H2b.%s.%s" % (ren.get(b, b), {"gated_minus_rootonly": "loop", "rootonly_minus_zs": "stack"}[what])
        elif ck.startswith("H3."):
            nm = "H3." + parts[1].replace("_minus_", "_")
        else:
            nm = parts[0] + "." + ren.get(parts[1], parts[1])
        out[nm] = (("contrasts", ck), "pp")
    for k, kind in (("dauroc", "x3s"), ("daurc", "x3s"), ("dacc60", "fpp")):
        out["H4." + k] = (("H4.route", k), kind)
    out["H5.net_exec"] = (("H5.revision", "net_exec"), "pp")
    out["H5.net_repair"] = (("H5.revision", "net_repair"), "pp")
    return out


def rob_cell(ctx, key, src_base, rel, kind, n_path, rule=True, sens=False, twin=True):
    p = src_base + rel
    two = twin and kind in TWO_DECIMAL
    ctx.est2(key + ".original", "rescore", p + ("d0",), p + ("ci0", 0), p + ("ci0", 1), kind, "estimate under the original reference queries", two=two)
    ctx.est2(key + ".corrected", "rescore", p + ("d1",), p + ("ci1", 0), p + ("ci1", 1), kind, "estimate under the corrected reference queries", two=two)
    ctx.est2(key + ".shift", "rescore", p + ("shift",), p + ("ci_shift", 0), p + ("ci_shift", 1), kind, "corrected minus original on the same items and predictions", two=two)
    if n_path is not None:
        ctx.val(key + ".n", "rescore", n_path, "int", "items in the cell (common set)")
    if rule and ctx.has("rescore", p + ("robust",)):
        ctx.val(key + ".robust", "rescore", p + ("robust",), "str", "SAP 20.7 'robust': sign unchanged and original/corrected intervals overlap")
        ctx.val(key + ".stable", "rescore", p + ("stable",), "str", "SAP 20.7 'stable': robust and shift interval contains 0 or |shift| small")
        r, s = ctx.raw_of(ctx.prefix + key + ".robust"), ctx.raw_of(ctx.prefix + key + ".stable")
        if r is not MISSING:
            txt = "robust, stable" if (r and s) else ("robust" if r else "not robust")
            ctx.derived(key + ".rule", txt, "str", "text from .robust and .stable", "SAP 20.7 reading of this cell", ctx.paths["rescore"])
    if sens and ctx.has("rescore", p + ("sens_no_thr",)):
        ctx.est2(key + ".nothr.original", "rescore", p + ("sens_no_thr", "d0"), p + ("sens_no_thr", "ci0", 0), p + ("sens_no_thr", "ci0", 1), kind, "original, without thrombosis_prediction", two=two)
        ctx.est2(key + ".nothr.corrected", "rescore", p + ("sens_no_thr", "d1"), p + ("sens_no_thr", "ci1", 0), p + ("sens_no_thr", "ci1", 1), kind, "corrected, without thrombosis_prediction", two=two)
        ctx.val(key + ".nothr.n", "rescore", p + ("sens_no_thr", "n"), "int", "items without thrombosis_prediction")


def build_rob(ctx):
    start(ctx, "ROB.", "ROB")
    G1 = ("variants", "G1", "sets")
    specs = rob_contrast_specs(ctx)
    for set_name, pre, prim in (("S_same", "", True), ("S_all", "sall.", False)):
        base = G1 + (set_name, "analysis")
        for nm, (rel, kind) in specs.items():
            if not prim and not (nm.startswith("H1.") or nm.startswith("H2.") or nm in ("H4.dauroc", "H5.net_exec")):
                continue
            rob_cell(ctx, pre + nm, base, rel, kind, (base + (rel[0], "n")) if rel[0] != "contrasts" else (base + rel + ("n",)),
                     sens=prim and (nm.startswith("H1.") or (nm.startswith("H2.") and not nm.startswith("H2b"))))
        ctx.val(pre + "n_common", "rescore", base + ("common_n",), "int", "items scored under both golds (%s)" % set_name)
    for g in ("G2", "G3", "G4"):
        base = ("variants", g, "sets", "S_same", "analysis")
        for nm, (rel, kind) in specs.items():
            if nm.startswith("H1.") or nm.startswith("H2.") or nm in ("H4.dauroc", "H5.net_exec"):
                if nm.startswith("H2b"):
                    continue
                rob_cell(ctx, g.lower() + "." + nm, base, rel, kind, (base + (rel[0], "n")) if rel[0] != "contrasts" else (base + rel + ("n",)))
        ctx.val(g.lower() + ".n_common", "rescore", base + ("common_n",), "int", "items scored under both golds (%s, S_same)" % g)
    # AUROC levels and counts of revisions
    base = G1 + ("S_same", "analysis")
    for nm in ("auroc_route_ca", "auroc_v3"):
        ctx.val("H4.%s.original" % nm, "rescore", base + ("H4.route", nm, "o"), "x3", "AUROC under original gold")
        ctx.val("H4.%s.corrected" % nm, "rescore", base + ("H4.route", nm, "c"), "x3", "AUROC under corrected gold")
    for c in ("counts_orig", "counts_corr"):
        for f in ctx.keys("rescore", base + ("H5.revision", c)):
            ctx.val("H5.revision.%s.%s" % ("orig" if c == "counts_orig" else "corr", f), "rescore", base + ("H5.revision", c, f), "int",
                    "executable-root revisions of full_fixed (plus), %s gold: %s" % ("original" if c == "counts_orig" else "corrected", f))
    ctx.val("H5.revision.p_orig", "rescore", base + ("H5.revision", "sign_test_p_orig"), "p", "exact sign test, original gold")
    ctx.val("H5.revision.p_corr", "rescore", base + ("H5.revision", "sign_test_p_corr"), "p", "exact sign test, corrected gold")
    # levels
    for run in ctx.keys("rescore", base + ("levels",)):
        lp = base + ("levels", run)
        k = "level." + run
        ctx.est2(k + ".original", "rescore", lp + ("ex0",), lp + ("ci0", 0), lp + ("ci0", 1), "pctv", "EX (%) under original gold, S_same")
        ctx.est2(k + ".corrected", "rescore", lp + ("ex1",), lp + ("ci1", 0), lp + ("ci1", 1), "pctv", "EX (%) under corrected gold, S_same")
        ctx.est2(k + ".shift", "rescore", lp + ("shift",), lp + ("ci_shift", 0), lp + ("ci_shift", 1), "pp", "EX shift, pp")
        ctx.val(k + ".n", "rescore", lp + ("n",), "int", "items")
    # pair audit, rank
    for b in ctx.keys("rescore", base + ("pair_audit",)):
        nm = {"q27b": "27b"}.get(b, b)
        pa = base + ("pair_audit", b)
        ctx.val("audit.%s.pairs_differ" % nm, "rescore", pa + ("pairs_with_differing_sql",), "int", "items where gated and fixed return different SQL")
        ctx.val("audit.%s.fixed_only_right_orig" % nm, "rescore", pa + ("second_only_right_orig",), "int", "items where only the fixed loop is right, original gold")
        ctx.val("audit.%s.fixed_only_right_corr" % nm, "rescore", pa + ("second_only_right_corr",), "int", "items where only the fixed loop is right, corrected gold")
    ctx.val("rank.spearman", "rescore", base + ("rank_plus", "spearman"), "x2", "Spearman of the nine plus systems' EX, original vs corrected")
    ctx.val("rank.reversed_pairs", "rescore", base + ("rank_plus", "reversed_pairs"), "int", "system pairs whose order reverses")
    ctx.val("rank.n_pairs", "rescore", base + ("rank_plus", "n_pairs"), "int", "system pairs")
    # sets
    sb = G1 + ("S_same",)
    ctx.val("n_same", "rescore", sb + ("analysis", "common_n"), "int", "S_same items scored under both golds (primary set)")
    ctx.val("n_all", "rescore", G1 + ("S_all", "analysis", "common_n"), "int", "S_all items scored under both golds")
    ctx.val("n_eff_change", "rescore", sb + ("effective_gold_change",), "int", "S_same items whose gold result changed")
    ctx.val("n_eff_change_all", "rescore", G1 + ("S_all", "effective_gold_change"), "int", "S_all items whose gold result changed")
    ctx.val("n_text_changed", "rescore", sb + ("sql_text_changed",), "int", "S_same items whose gold SQL text changed")
    ctx.val("n_text_same_result", "rescore", sb + ("text_changed_but_same_result",), "int", "S_same items with changed text and unchanged result")
    ctx.val("n_db_discordant", "rescore", ("variants", "G1", "n_db_discordant"), "int", "items excluded: corrected gold differs between dev and Mini-Dev database files")
    ctx.val("n_corrected_items", "overlap", ("variants", "G1", "n_items"), "int", "dev items with a corrected counterpart (G1)")
    ctx.val("n_s_same_raw", "overlap", ("variants", "G1", "n_S_same"), "int", "G1 items with question, evidence and database unchanged (before exclusions)")
    ctx.val("n_dev", "overlap", ("_meta", "n_dev"), "int", "development questions")
    nsame, ndev = ctx.raw_of("ROB.n_same"), ctx.raw_of("ROB.n_dev")
    if nsame is not MISSING and ndev is not MISSING:
        ctx.derived("share_dev", float(nsame) / float(ndev), "pct", "n_same / n_dev", "S_same as a share of the development set (%)", ctx.paths["rescore"] + " + " + ctx.paths["overlap"])
    # verdicts per conclusion
    for cn in ctx.keys("rescore", ("headline", "S_same")):
        hp = ("headline", "S_same", cn)
        ctx.val("%s.unchanged" % cn, "rescore", hp + ("unchanged",), "str", "all contrasts of %s robust under SAP 20.7" % cn)
        ctx.val("%s.unchanged_and_stable" % cn, "rescore", hp + ("unchanged_and_stable",), "str", "all contrasts of %s robust and stable" % cn)
        nr = ctx.get("rescore", hp + ("not_robust",))
        if nr is not MISSING:
            ctx.derived("%s.n_not_robust" % cn, len(nr), "int", "len(not_robust)", "contrasts of %s that are not robust" % cn, ctx.paths["rescore"])
    # flips (fixed-rule case list sizes)
    cf = ctx.keys("rescore", ("cases_fixed_rule",))
    for k in cf:
        if k.endswith("|n_total"):
            run, direction = k.split("|")[0], k.split("|")[1]
            ctx.val("flips.%s.%s" % (run, direction), "rescore", ("cases_fixed_rule", k), "int", "items that flip %s for %s under corrected gold (S_same)" % (direction, run))
    # E10 L6
    for s_name, pre in (("S_all", "e10_l6."), ("S_same", "e10_l6.same.")):
        ep = ("e10_l6", "G1", s_name, "eff")
        for grp, nm in (("L6", "L6"), ("non_L6", "nonL6")):
            ctx.val(pre + nm + ".k", "rescore", ep + (grp, "k"), "int", "E10 sheet items in the set with a changed gold result, label %s" % nm)
            ctx.val(pre + nm + ".n", "rescore", ep + (grp, "n"), "int", "E10 sheet items in the set, label %s" % nm)
            ctx.est(pre + nm, "rescore", ep + (grp, "share"), ep + (grp, "wilson"), "pct", "share with a changed gold result, label %s (Wilson)" % nm)
        ctx.est("%sdiff" % pre, "rescore", ep + ("diff_L6_minus_nonL6", "diff"), ep + ("diff_L6_minus_nonL6", "newcombe"), "fpp", "L6 share minus other labels' share (Newcombe)")
        ctx.val("%sdiff.corroborated" % pre, "rescore", ep + ("diff_L6_minus_nonL6", "corroborated"), "str", "interval of the difference excludes 0")
    # E10 L6, changed reference TEXT instead of changed reference RESULT (S_all; the text variant of the same comparison; stored under e10_l6.G1.S_all.text)
    tp = ("e10_l6", "G1", "S_all", "text")
    for grp, nm in (("L6", "L6"), ("non_L6", "nonL6")):
        ctx.val("e10_l6.text." + nm + ".k", "rescore", tp + (grp, "k"), "int", "E10 sheet items in S_all with a changed reference query text, label %s" % nm)
        ctx.val("e10_l6.text." + nm + ".n", "rescore", tp + (grp, "n"), "int", "E10 sheet items in S_all, label %s" % nm)
        ctx.est("e10_l6.text." + nm, "rescore", tp + (grp, "share"), tp + (grp, "wilson"), "pct", "share with a changed reference query text, label %s (Wilson)" % nm)
    ctx.est("e10_l6.text.diff", "rescore", tp + ("diff_L6_minus_nonL6", "diff"), tp + ("diff_L6_minus_nonL6", "newcombe"), "fpp", "L6 share minus other labels' share, changed reference TEXT variant (Newcombe)")
    ctx.val("e10_l6.text.diff.corroborated", "rescore", tp + ("diff_L6_minus_nonL6", "corroborated"), "str", "interval of the difference excludes 0 (text variant)")
    ctx.val("e10_l6.n_in_set", "rescore", ("e10_l6", "G1", "S_all", "n_in_set"), "int", "E10 sheet items inside S_all")
    ctx.val("e10_l6.n_sheet", "rescore", ("e10_l6", "n_sheet_main"), "int", "E10 sheet size")
    ctx.val("e10_l6.n_L6", "rescore", ("e10_l6", "n_L6"), "int", "L6 items on the sheet")


# ------------------------------------------------------------------------------------------ DIAG
def build_diag(ctx):
    start(ctx, "DIAG.", "DIAG")
    cats = ctx.keys("diag", ("per_category",))
    groups = [(c, ("per_category", c)) for c in cats] + [("all", ("pooled", "all"))]
    for c, base in groups:
        ctx.val(c + ".n_inj", "diag", base + ("n_pert",), "int", "injected queries")
        ctx.val(c + ".n_gold", "diag", base + ("n_gold",), "int", "gold controls")
        ctx.rate(c + ".detect", "diag", base + ("detect_triggered",), "detection: Critic Score below the pipeline's revision trigger, injected queries (%)")
        ctx.rate(c + ".detect_mustfix", "diag", base + ("detect_must_fix",), "detection at the must-fix level (%)")
        ctx.rate(c + ".detect_issues", "diag", base + ("detect_issues_listed",), "injected queries with an issue listed (%)")
        ctx.rate(c + ".fpr", "diag", base + ("fp_triggered_gold",), "false-positive rate: gold controls below the revision trigger (%)")
        ctx.rate(c + ".fpr_mustfix", "diag", base + ("fp_must_fix_gold",), "gold controls at the must-fix level (%)")
        ctx.rate(c + ".repair1", "diag", base + ("repair_round1_all",), "injected query repaired in one round, all injected (%)")
        ctx.rate(c + ".repair1_given_det", "diag", base + ("repair_round1_given_triggered",), "repaired in one round, given detected (%)")
        ctx.rate(c + ".final_ex", "diag", base + ("official_ex_set_final_all",), "final answer matches gold by set EX, all injected (%)")
        ctx.rate(c + ".gold_keep", "diag", base + ("gold_final_match_gold",), "gold control still matching gold after the loop (%)")
        ctx.rate(c + ".gated_revises", "diag", base + ("gated_revises",), "injected queries the gated policy revises (%)")
        ctx.val(c + ".score_inj", "diag", base + ("mean_score_pert",), "x2", "mean Critic Score, injected")
        ctx.val(c + ".score_gold", "diag", base + ("mean_score_gold",), "x2", "mean Critic Score, gold controls")
        ctx.est2(c + ".auroc", "diag", base + ("auroc_paired_controls", "auroc"), base + ("auroc_paired_controls", "lo"), base + ("auroc_paired_controls", "hi"), "x3",
                 "AUROC injected vs paired gold controls", n=base + ("auroc_paired_controls", "n_pert"))
        ctx.est2(c + ".auroc_sem", "diag", base + ("auroc_semantic_paired", "auroc"), base + ("auroc_semantic_paired", "lo"), base + ("auroc_semantic_paired", "hi"), "x3",
                 "AUROC of the semantic sub-score, paired")
        # harm = 1 - gold_keep (derived)
        gk, lo, hi = ctx.raw_of("DIAG." + c + ".gold_keep"), ctx.raw_of("DIAG." + c + ".gold_keep.ci_lo"), ctx.raw_of("DIAG." + c + ".gold_keep.ci_hi")
        if gk is not MISSING:
            src = ctx.paths["diag"]
            ctx.derived(c + ".harm", 1.0 - gk, "pct", "1 - gold_keep", "gold controls whose loop result no longer matches gold (%)", src)
            ctx.derived(c + ".harm.ci", (1.0 - hi, 1.0 - lo), "pct", "1 - gold_keep.ci (ends swapped)", "interval of harm", src)
            ctx.derived(c + ".harm.ci_lo", 1.0 - hi, "pct", "1 - gold_keep.ci_hi", "lower end of harm.ci", src)
            ctx.derived(c + ".harm.ci_hi", 1.0 - lo, "pct", "1 - gold_keep.ci_lo", "upper end of harm.ci", src)
    ctx.est2("all.auroc_all", "diag", ("pooled", "all", "auroc_all_pert_vs_all_gold", "auroc"), ("pooled", "all", "auroc_all_pert_vs_all_gold", "lo"),
             ("pooled", "all", "auroc_all_pert_vs_all_gold", "hi"), "x3", "AUROC all injected vs all gold controls", n=("pooled", "all", "auroc_all_pert_vs_all_gold", "n_pert"))
    ctx.val("n_categories", "diag", ("_meta", "n_records"), "int", "injected records")
    ctx.val("n_dbs", "diag", ("pooled", "all", "n_dbs_pert"), "int", "databases")


# ------------------------------------------------------------------------------------------ FBK
def build_fbk(ctx):
    start(ctx, "FBK.", "FBK")
    ctx.val("verdict", "fbk", ("reading_rule", "verdict"), "str", "reading-rule verdict (SAP 17.6)")
    ctx.val("rule_text", "fbk", ("reading_rule", "text"), "str", "reading-rule sentence")
    ctx.val("hard_text", "fbk", ("reading_rule", "hard_categories_text"), "str", "which hard categories nominally improve")
    imp = ctx.get("fbk", ("reading_rule", "improved_at_n"))
    if imp is not MISSING:
        ctx.derived("n_improved_budgets", len(imp), "int", "len(improved_at_n)", "budgets at which a learner reaches 'improved'", ctx.paths["fbk"])
    ctx.val("n_items", "fbk", ("_meta", "n_items"), "int", "labelled items (injected + gold controls)")
    ctx.val("n_inj", "fbk", ("_meta", "n_injected"), "int", "injected items")
    ctx.val("n_ctrl", "fbk", ("_meta", "n_controls"), "int", "gold controls")
    ctx.val("n_dbs", "fbk", ("_meta", "n_databases"), "int", "databases")
    ctx.val("seeds", "fbk", ("_meta", "seeds"), "int", "random draws of the labelled items")
    ctx.val("alpha0", "fbk", ("_meta", "alpha0"), "pct", "uncalibrated false-positive rate (target of Rule FP, %)")
    ctx.val("d0", "fbk", ("_meta", "d0"), "pct", "uncalibrated detection (target of Rule DET, %)")
    ctx.val("margin_auroc", "fbk", ("_meta", "margins", "auroc"), "x3", "AUROC margin for 'improved'")
    for learner in ctx.keys("fbk", ("pooled",)):
        for rule in ("fp", "det"):
            for n in ctx.keys("fbk", ("pooled", learner, rule)):
                b = ("pooled", learner, rule, n)
                for met, mname in (("detection", "det"), ("fpr", "fpr")):
                    k = "%s.n%s.%s.%s" % (learner, n, rule, mname)
                    ctx.est(k, "fbk", b + (met, "mean"), b + (met, "ci"), "pct", "pooled %s under Rule %s, learner %s, n = %s labelled items (mean over seeds)" % (met, rule.upper(), learner, n))
                    ctx.est("%s.n%s.%s.d_%s" % (learner, n, rule, mname), "fbk", b + (met, "delta_vs_n0", "mean"), b + (met, "delta_vs_n0", "ci"), "fpp",
                            "difference to n = 0 of the above")
        for n in ctx.keys("fbk", ("auroc", learner)):
            b = ("auroc", learner, n)
            ctx.est("%s.n%s.auroc" % (learner, n), "fbk", b + ("mean",), b + ("ci",), "x3", "pooled out-of-fold AUROC, learner %s, n = %s" % (learner, n))
            ctx.est("%s.n%s.d_auroc" % (learner, n), "fbk", b + ("delta_vs_n0", "mean"), b + ("delta_vs_n0", "ci"), "x3s", "AUROC minus AUROC at n = 0")
    for cat in ("distinct_drop", "count_distinct"):
        for learner in ctx.keys("fbk", ("per_category", cat)):
            for rule in ("fp", "det"):
                for n in ctx.keys("fbk", ("per_category", cat, learner, rule)):
                    b = ("per_category", cat, learner, rule, n)
                    met, mname = ("detection", "det") if rule == "fp" else ("fpr", "fpr")
                    ctx.est("cat.%s.%s.n%s.%s.%s" % (cat, learner, n, rule, mname), "fbk", b + (met, "mean"), b + (met, "ci"), "pct", "%s, Rule %s, %s, n = %s" % (cat, rule.upper(), learner, n))
                    ctx.est("cat.%s.%s.n%s.%s.d_%s" % (cat, learner, n, rule, mname), "fbk", b + (met, "delta_vs_n0", "mean"), b + (met, "delta_vs_n0", "ci"), "fpp", "difference to n = 0")
        for learner in ctx.keys("fbk", ("per_category_auroc", cat)):
            for n in ctx.keys("fbk", ("per_category_auroc", cat, learner)):
                ctx.val("cat.%s.%s.n%s.auroc" % (cat, learner, n), "fbk", ("per_category_auroc", cat, learner, n, "mean"), "x3", "paired AUROC %s, %s, n = %s" % (cat, learner, n))
    for cat in ctx.keys("fbk", ("addendum_descriptive", "component_means_by_category")):
        for grp in ("injected", "gold_control"):
            for i, comp in enumerate(("syntax", "semantic", "efficiency")):
                ctx.val("comp.%s.%s.%s" % (cat, grp, comp), "fbk", ("addendum_descriptive", "component_means_by_category", cat, grp, "mean_syntax_semantic_efficiency", i), "x2",
                        "mean %s sub-score, %s, %s" % (comp, cat, grp))


# ------------------------------------------------------------------------------------------ VAD
def build_vad(ctx):
    start(ctx, "VAD.", "VAD")
    vs = ctx.keys("v2", ("variants",))
    n_inj = ("_meta", "n_injected")
    ctx.val("n_inj", "v2", n_inj, "int", "injected queries")
    ctx.val("n_gold", "v2", ("_meta", "n_gold_controls"), "int", "gold controls")
    ctx.val("operating_point", "v2", ("_meta", "operating_point"), "str", "trigger of VA/VB versus V0")
    metrics = (("detection", "detection", "pct"), ("fpr", "fpr", "pct"), ("harm", "harm", "pct"), ("repair1", "repair_round1_all", "pct"), ("repair_final", "repair_final_all", "pct"), ("auroc", "auroc", "x3"))
    for v in vs:
        if v == "V0_semantic_only":
            ctx.val("V0sem.auroc", "v2", ("variants", v, "point", "auroc"), "x3", "AUROC of the V0 semantic sub-score alone")
            continue
        for k, f, kind in metrics:
            ctx.est2("%s.%s" % (v, k), "v2", ("variants", v, "point", f), ("variants", v, "ci", f, "lo"), ("variants", v, "ci", f, "hi"), kind, "%s of variant %s (%s)" % (k, v, f))
        if ctx.has("v2", ("variants", v, "ci", "detection_delta")):
            for k, f, kind in metrics:
                dk = "fpp" if kind == "pct" else "x3s"
                ctx.ci2("%s.d_%s" % (v, k), "v2", ("variants", v, "ci", f + "_delta", "lo"), ("variants", v, "ci", f + "_delta", "hi"), dk, "difference to V0, variant %s" % v, two=(dk == "fpp"))
                if k in ("detection", "fpr"):
                    ctx.val("%s.d_%s" % (v, k), "v2", ("decision", v, "delta_" + f), dk, "difference to V0 (point), variant %s" % v, two=(dk == "fpp"))
        if ctx.has("v2", ("decision", v)):
            ctx.val("%s.rule_met" % v, "v2", ("decision", v, "improves_rule_18_7"), "str", "SAP 18.7 rule met")
            ctx.val("%s.strict_met" % v, "v2", ("decision", v, "improves_strict_reading"), "str", "strict reading of the rule met")
        if ctx.has("v2", ("calls", v)):
            ctx.val("%s.calls" % v, "v2", ("calls", v, "calls_per_query"), "x2", "model calls per query (594 queries)")
            ctx.val("%s.extra_calls_vs_VA" % v, "v2", ("calls", v, "extra_calls_per_query_vs_VA"), "x2s", "extra calls per query versus VA")
            ctx.val("%s.extra_calls_vs_V0" % v, "v2", ("calls", v, "extra_calls_per_query_vs_V0"), "x2s", "extra calls per query versus V0")
            ctx.val("%s.n_flagged_roots" % v, "v2", ("calls", v, "n_flagged_roots"), "int", "roots flagged")
        for c in ctx.keys("v2", ("per_category",)):
            for k, f, kind in metrics[:3]:
                ctx.val("cat.%s.%s.%s" % (c, v, k), "v2", ("per_category", c, v, f), "pct", "%s, category %s, variant %s" % (k, c, v))
    for v in ctx.keys("v2", ("stages",)):
        for grp in ("pert", "gold"):
            for f in ("n", "critic_flagged", "after_final"):
                ctx.val("stage.%s.%s.%s" % (v, grp, f), "v2", ("stages", v, grp, f), "int", "%s %s: %s" % (v, grp, f))
    for grp in ("pert", "gold"):
        for f in ctx.keys("v2", ("appeal", grp, "issues")):
            ctx.val("appeal.%s.%s" % (grp, f), "v2", ("appeal", grp, "issues", f), "int", "VD appeal, %s: issues %s" % (grp, f))
        for f in ctx.keys("v2", ("appeal", grp, "roots")):
            ctx.val("appeal.%s.roots.%s" % (grp, f), "v2", ("appeal", grp, "roots", f), "int", "VD appeal, %s: roots %s" % (grp, f))


# ------------------------------------------------------------------------------------------ VE
def qlab(q):
    return "q" + q.replace(".", "")


def pair_lab(p):
    return p.replace(" - ", "_vs_").replace(" ", "")


def build_ve(ctx):
    start(ctx, "VE.", "VE")
    for k in ctx.keys("ve", ("units_count",)):
        ctx.val("units." + k.replace("-", "_"), "ve", ("units_count", k), "int", "evaluation units of class %s" % k)
    ctx.val("n_units", "ve", ("_meta", "units", "n_units"), "int", "units (roots of the stored VA run)")
    qv = ctx.get("ve", ("_meta", "q_rule"))
    if qv is not MISSING:
        ctx.derived("q_rule", qv, "x2", "_meta.q_rule", "user accuracy q of the decision rule", ctx.paths["ve"])
    am = (("firm_all", "firmness_all"), ("resp_all", "responsiveness_all"), ("firm_correct", "firmness_F_claim_correct"), ("resp_wrong", "responsiveness_F_claim_wrong"),
          ("resp_inj", "responsiveness_F_claim_wrong_injected"), ("resp_gold", "responsiveness_F_gold_flag"), ("firm_pass", "firmness_P_gold"), ("resp_missed", "responsiveness_P_injected"))
    for arm in ctx.keys("ve", ("arbitration_quality",)):
        b = ("arbitration_quality", arm)
        for k, f in am:
            if ctx.data.get("ve") and ctx.get_quiet("ve", b + ("point", f, "rate")) is None:
                ctx.na.append("%s: %s (arm %s reviews flags only)" % (ctx.paths["ve"], "/".join(b + ("point", f)), arm))
                continue
            ctx.est2("%s.%s" % (arm, k), "ve", b + ("point", f, "rate"), b + ("ci", f, "lo"), b + ("ci", f, "hi"), "pct", "%s of arm %s (%%)" % (f, arm), n=b + ("point", f, "n"))
        ctx.est2("%s.balacc" % arm, "ve", b + ("point", "balanced_accuracy"), b + ("ci", "balanced_accuracy", "lo"), b + ("ci", "balanced_accuracy", "hi"), "pct", "balanced accuracy of the arbiter, arm %s (%%)" % arm)
        ctx.val("%s.hold_all" % arm, "ve", ("validity", arm, "hold_rate_all"), "pct", "share of all arbitrated units held, arm %s (%%)" % arm)
        ctx.val("%s.invalid" % arm, "ve", ("validity", arm, "invalid_or_unparsable"), "int", "unparsable or evidence-less outputs, arm %s" % arm)
        ctx.val("%s.n_units" % arm, "ve", ("validity", arm, "n"), "int", "units arbitrated, arm %s" % arm)
    mets = (("det", "DET", "pct", "fpp"), ("udet", "UDET", "pct", "fpp"), ("fpr", "FPR", "pct", "fpp"), ("ccr", "CCR", "pct", "fpp"), ("wcr", "WCR", "pct", "fpp"),
            ("mer", "MER", "pct", "fpp"), ("tnr", "TNR", "pct", "fpp"), ("harm_lb", "HARM_LB", "pct", "fpp"), ("j", "J", "pct", "fpp"))
    for q in ctx.keys("ve", ("by_q",)):
        ql = qlab(q)
        base = ("by_q", q)
        for pol in ctx.keys("ve", base + ("point",)):
            for k, f, kind, dk in mets:
                if ctx.has("ve", base + ("ci", pol, f)):
                    ctx.est2("%s.%s.%s" % (ql, pol, k), "ve", base + ("point", pol, f), base + ("ci", pol, f, "lo"), base + ("ci", pol, f, "hi"), "pct", "%s, policy %s, q = %s (%%)" % (f, pol, q))
                else:
                    ctx.val("%s.%s.%s" % (ql, pol, k), "ve", base + ("point", pol, f), "pct", "%s, policy %s, q = %s (%%); no interval in the file" % (f, pol, q))
        for pr in ctx.keys("ve", base + ("delta_point",)):
            for k, f, kind, dk in mets:
                nm = "%s.%s.%s" % (ql, pair_lab(pr), k)
                ctx.est2(nm, "ve", base + ("delta_point", pr, f), base + ("delta_ci", pr, f, "lo"), base + ("delta_ci", pr, f, "hi"), "fpp", "%s difference %s, q = %s (pp)" % (f, pr, q), two=False)
        for arm in ctx.keys("ve", base + ("calls_per_query",)):
            ctx.val("%s.calls.%s" % (ql, arm), "ve", base + ("calls_per_query", arm), "x2", "extra LLM calls per query, arm %s, q = %s" % (arm, q))
            ctx.val("%s.disputes.%s" % (ql, arm), "ve", base + ("expected_disputes", arm), "int", "expected disputed decisions, arm %s, q = %s" % (arm, q))
        if ctx.has("ve", base + ("decision_rule_19_5",)):
            for arm in ctx.keys("ve", base + ("decision_rule_19_5",)):
                rp = base + ("decision_rule_19_5", arm)
                for f in ("clause1a_J_beats_obey", "clause1b_CCR_beats_obey", "clause2_no_loss_vs_none", "clause3_J_beats_none_19_9", "useful", "useful_19_5_as_committed", "useful_strict_reading"):
                    ctx.val("%s.rule.%s.%s" % (ql, arm, f), "ve", rp + (f,), "str", "SAP 19.5/19.9 %s, arm %s, q = %s" % (f, arm, q))
                for f in ("delta_J_vs_obey", "delta_CCR_vs_obey", "delta_CCR_vs_none", "delta_J_vs_none"):
                    ctx.val("%s.rule.%s.%s" % (ql, arm, f), "ve", rp + (f,), "fpp", "%s, arm %s, q = %s (pp)" % (f, arm, q))
                c1a = ctx.raw_of("VE.%s.rule.%s.clause1a_J_beats_obey" % (ql, arm))
                c1b = ctx.raw_of("VE.%s.rule.%s.clause1b_CCR_beats_obey" % (ql, arm))
                us = ctx.raw_of("VE.%s.rule.%s.useful" % (ql, arm))
                if MISSING not in (c1a, c1b, us):
                    txt = "not met" if not us else ("met on J" if c1a else "met on retention only")
                    ctx.derived("%s.rule.%s.text" % (ql, arm), txt, "str", "from clause1a, clause1b, useful", "reading of the rule for arm %s" % arm, ctx.paths["ve"])
    # break-even q of an arm versus obeying the user, linear interpolation between the computed q values (file values only)
    for arm in ("A", "B"):
        pts = []
        for q in ctx.keys("ve", ("by_q",)):
            v = ctx.get("ve", ("by_q", q, "delta_point", "%s - obey" % arm, "J"))
            if v is not MISSING:
                pts.append((float(q), v))
        pts.sort()
        be = None
        for (q0, d0), (q1, d1) in zip(pts, pts[1:]):
            if d0 * d1 < 0:
                be = q0 + (q1 - q0) * (0 - d0) / (d1 - d0)
        if be is not None:
            ctx.derived("breakeven_q.%s" % arm, be, "x2", "linear interpolation of J(arm - obey) = 0 between the computed q", "q at which arm %s equals obeying the user on J" % arm, ctx.paths["ve"])


# ------------------------------------------------------------------------------------------ EXT1
EXT_INT = {"n", "n_dbs", "b", "c", "B", "seed_boot", "api_fail", "sys_error", "unpriced_rows", "final_correct", "final_executes", "gate_fired", "rootonly_exec_fail"}
EXT_PCT = {"same_sign_share", "gate_fired_share", "revised_share", "exec_ok", "ex_plain", "ex_ts", "ex_ts_ci_lo", "ex_ts_ci_hi"}
EXT_STR = {"status", "text", "verdict", "line"}


def ext1_kind(key, v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "wilson" if (len(v) == 2 and key.endswith("_wilson")) else None
    if isinstance(v, dict) or v is None:
        return None
    last = key.split(".")[-1]
    if last in EXT_STR:
        return "str"
    if last in EXT_INT:
        return "int"
    if re.search(r"(^|_)p(_|$)", last) or last == "mcnemar_p_two":
        return "p"
    if "calls_ratio" in key or last == "calls_per_q":
        return "x2"
    if last in EXT_PCT:
        return "pct"
    if last in ("prompt_tok_per_q", "completion_tok_per_q"):
        return "tok"
    if last == "latency_mean_s":
        return "x1"
    if last in ("cost_cny_per_q", "cost_cny_per_correct"):
        return "cny4"
    if last in ("cost_cny_total", "embed_cny_total"):
        return "cny2"
    if last in ("delta_ex", "item_ci_lo", "item_ci_hi", "loo_min", "loo_max", "ci_lo", "ci_hi") or ".per_backbone." in key:
        return "pp"
    return None


def build_ext1(ctx):
    start(ctx, "EXT1.", "EXT1")
    d = ctx.data.get("ext1")
    if d is None:
        ctx.get("ext1", ("EXT1",))
        return
    src = ctx.paths["ext1"]
    unclassified = []
    for key in d:
        if key.startswith("_"):
            continue
        v = d[key]
        kind = ext1_kind(key, v)
        if kind is None:
            if v is None:
                ctx.na.append(src + ": " + key)
                continue
            if isinstance(v, (list, dict)):
                continue                       # arms_complete, bird_reference: structure, not a value
            unclassified.append(key)
            continue
        if kind == "bool":
            ctx._put(key, v, "str", src, key, "flag")
        elif kind == "wilson":
            ctx._put(key[:-len("_wilson")] + ".wilson", (v[0], v[1]), "pct", src, key, "Wilson interval")
        else:
            if kind == "str" and key.endswith(".line"):
                v = v.replace("_", " ")        # reading phrase for a table cell: words, not a code identifier
            ctx._put(key, v, kind, src, key, "see results/ext1/ext1_summary.json")
    for k in unclassified:
        ctx.errors.append("%s: no format rule for key %s (add it to ext1_kind)" % (src, k))
    # intervals
    for key in list(ctx.entries):
        e = ctx.entries[key]
        pairs = ((".ci_lo", ".ci_hi", ".ci"), (".item_ci_lo", ".item_ci_hi", ".item_ci"), (".loo_min", ".loo_max", ".loo"), ("_ci_lo", "_ci_hi", ".ci"))
        for lo, hi, new in pairs:
            if key.endswith(lo):
                base = key[:-len(lo)]
                hk = base + hi
                if hk in ctx.entries and (base + new) not in ctx.entries:
                    ctx._put(base + new, (e.raw, ctx.entries[hk].raw), e.kind, src, e.field + " , " + ctx.entries[hk].field, "interval formed from %s and %s" % (key, hk))
                if new == ".ci":
                    for suffix, other in ((".ci_lo", e), (".ci_hi", ctx.entries[hk])):
                        if base + suffix not in ctx.entries:
                            ctx._put(base + suffix, other.raw, other.kind, src, other.field, "end of " + base + ".ci")
                break
    # derived: per-arm cost per 1000 questions, pooled gate firing
    for key in list(ctx.entries):
        if key.endswith(".cost_cny_per_q"):
            ctx._put(key[:-len(".cost_cny_per_q")] + ".cost_k", ctx.entries[key].raw * 1000.0, "cny2", src, "derived: 1000 * " + key, "CNY per 1000 questions")
    ndb = d.get("EXT1.XB1.n_dbs")
    if ndb is not None:
        ctx._put("EXT1.n_dbs", ndb, "int", src, "derived: EXT1.XB1.n_dbs", "databases of the Spider-Realistic evaluation set")
    fired, nn = 0, 0
    bks = sorted(set(k.split(".")[2] for k in d if k.startswith("EXT1.gate.") and k.endswith(".gate_fired")))
    for b in bks:
        fired += d["EXT1.gate.%s.gate_fired" % b]
        nn += d["EXT1.gate.%s.n" % b]
    if bks:
        ctx._put("EXT1.gate.pooled.fired", fired, "int", src, "derived: sum over backbones of gate.<b>.gate_fired", "gate firings, all backbones")
        ctx._put("EXT1.gate.pooled.n", nn, "int", src, "derived: sum over backbones of gate.<b>.n", "items x backbones")
        ctx._put("EXT1.gate.pooled.share", float(fired) / float(nn), "pct", src, "derived: fired / n", "share of items on which the gate fired (%)")


# ------------------------------------------------------------------------------------------ E10
def build_e10(ctx):
    start(ctx, "E10.", "E10")
    pl = ("primary_label",)
    ctx.est2("agreement.kappa", "e10agree", pl + ("kappa",), pl + ("kappa_ci95", 0), pl + ("kappa_ci95", 1), "x3", "Cohen kappa of the two annotation runs, 8 labels (before adjudication)", n=pl + ("n",))
    ctx.est2("agreement.ac1", "e10agree", pl + ("ac1",), pl + ("ac1_ci95", 0), pl + ("ac1_ci95", 1), "x3", "Gwet AC1 of the two annotation runs")
    ctx.val("agreement.pct_agree", "e10agree", pl + ("pct_agree",), "pct", "percent agreement of the two runs (%)")
    l5 = ("L5_vs_rest",)
    ctx.est2("agreement.kappa_l5", "e10agree", l5 + ("kappa",), l5 + ("kappa_ci95", 0), l5 + ("kappa_ci95", 1), "x3", "kappa for L5 versus the rest")
    ctx.val("n_items", "e10agree", ("n_items",), "int", "sheet items annotated twice")
    ctx.val("n_disagree", "e10agree", ("n_disagree",), "int", "items on which the runs differ (adjudicated)")
    ctx.val("n_adj_equals_A1", "e10agree", ("n_adjudicated_equals_A1",), "int", "adjudicated label equals run A1")
    ctx.val("n_adj_equals_A2", "e10agree", ("n_adjudicated_equals_A2",), "int", "adjudicated label equals run A2")
    for sysn in ctx.keys("e10agree", ("per_system",)):
        for f, kind in (("n", "int"), ("kappa", "x3"), ("pct_agree", "pct"), ("ac1", "x3")):
            ctx.val("system.%s.%s" % (sysn, f), "e10agree", ("per_system", sysn, f), kind, "agreement, system %s: %s" % (sysn, f))
    for run in ("A1", "A2"):
        for lab in ctx.keys("e10agree", ("counts_" + run,)):
            ctx.val("counts.%s.%s" % (run, lab), "e10agree", ("counts_" + run, lab), "int", "sheet items labelled %s by run %s" % (lab, run))
    names = (("revision_effect_all_dev_items", "revision"), ("revision_effect_discordant_items", "revision.discordant"),
             ("repair_of_nonexecutable_root_all_dev_items", "repair"), ("repair_of_nonexecutable_root_discordant_items", "repair.discordant"))
    for blk, nm in names:
        for sysn in ctx.keys("e10disc", (blk,)):
            sn = sysn.replace("full_", "")
            for f, g in (("n_changed", "n"), ("helped", "helped"), ("hurt", "hurt"), ("neutral_both_correct_or_both_wrong", "neutral")):
                ctx.val("%s.%s.%s" % (nm, sn, g), "e10disc", (blk, sysn, f), "int", "%s, %s: %s (E10 revision effect)" % (blk, sysn, g))
        if nm == "revision":
            for f, g in (("n_changed", "n"), ("helped", "helped"), ("hurt", "hurt"), ("neutral_both_correct_or_both_wrong", "neutral")):
                ctx.val("revision." + g, "e10disc", (blk, "full_fixed", f), "int", "alias of revision.fixed.%s (qwen3.7-plus, fixed loop, all dev items)" % g)
    for f in ("n", "fixed_correct_gated_wrong", "fixed_wrong_gated_correct", "n_in_sheet_as_full_prediction", "n_in_sheet_as_zs_prediction"):
        ctx.val("discordant." + f, "e10disc", ("discordant", f), "int", "gated/fixed discordant items: %s" % f)


# ------------------------------------------------------------------------------------------ PF (exploratory-posthoc, simulated feedback)
PF_ARM_TEXT = {"zs": "single call", "full_fixed": "fixed loop", "full_gated": "gated loop", "route_ca": "route_ca",
               "pf_n0": "patent Critic, no feedback (n=0)", "pf_n250_q100": "patent Critic, 250 feedback items, q=1.0",
               "pf_n1000_q100": "patent Critic, 1000 feedback items, q=1.0", "pf_n1000_q085": "patent Critic, 1000 feedback items, q=0.85"}
PF_CURVE_COPY = ("calls_per_q", "cost_per_q", "cost_k", "detection", "false_flag", "flag_rate", "helped", "hurt", "net", "revisions")


def _pf_count(ctx, key, rate_full, n_full, pct=True):
    """k = rate * n (rate in %); the source stores rates only, k must come out integral or no key is written"""
    r, n = ctx.raw_of(rate_full), ctx.raw_of(n_full)
    if r is MISSING or n is MISSING:
        return
    k = r * n / (100.0 if pct else 1.0)
    if abs(k - round(k)) > 1e-6:
        ctx.errors.append("PF: %s * %s is not an integer count (%r)" % (rate_full, n_full, k))
        return
    ctx._put(key + ".k", int(round(k)), "int", ctx.entries[rate_full].src, "derived: %s x %s / 100" % (rate_full, n_full), "count behind " + key)


def build_pf(ctx):
    start(ctx, "PF.", "PF")
    if "pf" in ctx.absent:
        return
    S = "pf"
    sc = ctx.paths[S]

    def top(f):
        return (f,)

    def opt(f):
        return ctx.has(S, top(f))

    # --- meta
    ctx.val("n_items", S, ("_meta", "n_items"), "int", "dev items scored in every PF arm")
    ctx.val("n_dbs", S, ("_meta", "n_dbs"), "int", "databases (bootstrap clusters)")
    ctx.val("boot_B", S, ("_meta", "B"), "int", "bootstrap resamples")
    ctx.val("cost_total", S, ("_meta", "spent_cny_reported"), "cny2", "CNY spent on the whole PF block (tuning, smoke test, four dev arms), nominal price")
    ctx.val("feedback_note", S, ("_meta", "feedback"), "str", "what the feedback is (simulated oracle user)")
    # --- arms
    arms = sorted(set(k.split(".")[1] for k in ctx.keys(S, ()) if k.startswith("arm.")), key=lambda a: list(PF_ARM_TEXT).index(a) if a in PF_ARM_TEXT else len(PF_ARM_TEXT))
    for a in arms:
        P = "arm.%s." % a
        T = PF_ARM_TEXT.get(a, a)
        k = "arm.%s." % a
        mm = re.match(r"pf_n(\d+)(?:_q(\d+))?$", a)
        if mm:
            ctx.derived(k + "fb", int(mm.group(1)), "int", "number in the arm id '%s'" % a, "feedback items available to the Critic, %s" % T, sc)
            if mm.group(2):
                ctx.derived(k + "q", int(mm.group(2)) / 100.0, "x2", "number in the arm id '%s' (digits / 100)" % a, "simulated user accuracy q, %s" % T, sc)
        ctx.est2(k + "ex", S, top(P + "ex_pp"), top(P + "ex_lo"), top(P + "ex_hi"), "pctv", "EX (%%) on dev, %s" % T, two=True, n=top(P + "n"))
        _pf_count(ctx, k + "ex", "PF." + k + "ex", "PF." + k + "ex.n")
        ctx.val(k + "ex_nothr", S, top(P + "ex_no_thrombosis_pp"), "pctv", "EX (%%) without thrombosis_prediction, %s" % T, two=True)
        ctx.val(k + "calls_per_q", S, top(P + "calls_per_q"), "x2", "LLM calls per query, %s" % T)
        ctx.val(k + "embed_per_q", S, top(P + "embed_calls_per_q"), "x2", "embedding calls per query (not in the call count), %s" % T)
        ctx.val(k + "rerank_per_q", S, top(P + "rerank_calls_per_q"), "x2", "rerank calls per query (not in the call count), %s" % T)
        ctx.val(k + "prompt_tok_per_q", S, top(P + "prompt_tokens_per_q"), "tok", "prompt tokens per query, %s" % T)
        ctx.val(k + "completion_tok_per_q", S, top(P + "completion_tokens_per_q"), "tok", "completion tokens per query, %s" % T)
        if ctx.val(k + "cost_per_q", S, top(P + "cost_cny_per_q"), "cny4", "CNY per query (nominal price), %s" % T) is not None:
            ctx.derived(k + "cost_k", ctx.raw_of("PF." + k + "cost_per_q") * 1000.0, "cny2", "1000 x " + P + "cost_cny_per_q", "CNY per 1000 questions, %s" % T, sc)
        ctx.val(k + "cost_total", S, top(P + "cost_cny_total"), "cny2", "CNY for all dev queries of the arm, %s" % T)
        ctx.val(k + "cost_per_correct", S, top(P + "cost_cny_per_correct"), "cny4", "CNY per correct answer, %s" % T)
        # revision behaviour
        ctx.val(k + "revisions", S, top(P + "items_with_revision"), "int", "items whose final SQL differs from the root SQL, %s" % T)
        ctx.val(k + "revision_share", S, top(P + "revision_share_pp"), "pctv", "share of items revised (%%), %s" % T, two=True)
        ctx.derived(k + "revision_share.n", ctx.raw_of("PF." + k + "ex.n"), "int", "same n as " + k + "ex", "denominator of " + k + "revision_share", sc)
        ctx.val(k + "helped", S, top(P + "helped"), "int", "revisions that turned a wrong root into a right answer, %s" % T)
        ctx.val(k + "hurt", S, top(P + "hurt"), "int", "revisions that turned a right root into a wrong answer, %s" % T)
        ctx.est2(k + "net", S, top(P + "net_helped_minus_hurt_pp"), top(P + "net_lo"), top(P + "net_hi"), "pp", "helped minus hurt (pp of dev items), %s" % T, two=True)
        # Critic behaviour (arms with a Critic)
        if opt(P + "critic_flag_rate_pp"):
            ctx.val(k + "flag_rate", S, top(P + "critic_flag_rate_pp"), "pctv", "share of items the Critic flagged (%%), %s" % T, two=True)
            ctx.derived(k + "flag_rate.n", ctx.raw_of("PF." + k + "ex.n"), "int", "same n as " + k + "ex", "denominator of " + k + "flag_rate", sc)
            _pf_count(ctx, k + "flag_rate", "PF." + k + "flag_rate", "PF." + k + "flag_rate.n")
        if opt(P + "critic_detection_pp"):
            ctx.val(k + "detection", S, top(P + "critic_detection_pp"), "pctv", "share of wrong roots the Critic flagged (%%), %s" % T, two=True, )
            ctx.val(k + "detection.n", S, top(P + "n_wrong_roots"), "int", "wrong root SQL (denominator of detection)")
            _pf_count(ctx, k + "detection", "PF." + k + "detection", "PF." + k + "detection.n")
            ctx.val(k + "false_flag", S, top(P + "critic_false_flag_pp"), "pctv", "share of right roots the Critic flagged (%%), %s" % T, two=True)
            ctx.val(k + "false_flag.n", S, top(P + "n_right_roots"), "int", "right root SQL (denominator of false_flag)")
            _pf_count(ctx, k + "false_flag", "PF." + k + "false_flag", "PF." + k + "false_flag.n")
        if opt(P + "parse_fail_items"):
            ctx.val(k + "parse_fail", S, top(P + "parse_fail_items"), "int", "items whose Critic output could not be parsed (counted as pass), %s" % T)
            ctx.val(k + "root_identical", S, top(P + "root_sql_identical_to_full_fixed_pp"), "pctv", "share of items whose root SQL equals the fixed loop's (%%), %s" % T, two=True)
            ctx.val(k + "rejected_share", S, top(P + "share_items_with_rejected_case_pp"), "pctv", "share of items whose rubric holds a rejected case (%%), %s" % T, two=True)
            ctx.derived(k + "rejected_share.n", ctx.raw_of("PF." + k + "ex.n"), "int", "same n as " + k + "ex", "denominator of " + k + "rejected_share", sc)
            _pf_count(ctx, k + "rejected_share", "PF." + k + "rejected_share", "PF." + k + "rejected_share.n")
            n_with = ctx.raw_of("PF." + k + "rejected_share.k")
            for nm, f, nn, txt in (("flag_rate_rejected", "flag_rate_with_rejected_case_pp", n_with, "items with a rejected case in the rubric"),
                                    ("flag_rate_no_rejected", "flag_rate_without_rejected_case_pp", None if n_with is MISSING else ctx.raw_of("PF." + k + "ex.n") - n_with, "items without one")):
                if opt(P + f):
                    ctx.val(k + nm, S, top(P + f), "pctv", "Critic flag rate (%%) on %s, %s" % (txt, T), two=True)
                    ctx.derived(k + nm + ".n", nn, "int", "share x n / 100 (or n minus it)", "number of " + txt, sc)
                    _pf_count(ctx, k + nm, "PF." + k + nm, "PF." + k + nm + ".n")
                else:
                    ctx.na.append("%s: %s (no such items)" % (sc, P + f))
    # --- learning curve: q = 1.0 from the file's curve block, q = 0.85 from the arm
    curve_arm = {("q100", "n0"): "pf_n0", ("q100", "n250"): "pf_n250_q100", ("q100", "n1000"): "pf_n1000_q100",
                 ("q085", "n0"): "pf_n0", ("q085", "n1000"): "pf_n1000_q085"}
    for (q, nk), arm in curve_arm.items():
        if "PF.arm.%s.ex" % arm not in ctx.entries:
            continue
        base = "curve.%s.%s." % (q, nk)
        m = re.match(r"n(\d+)$", nk)
        ctx.derived(base + "fb", int(m.group(1)), "int", "number in the file's curve key '%s'" % nk, "feedback items available to the Critic", sc)
        if q == "q100":
            ctx.est2(base + "ex", S, ("curve." + nk + ".ex_pp",), ("curve." + nk + ".lo",), ("curve." + nk + ".hi",), "pctv", "EX (%%) on dev at %s feedback items, accurate simulated user" % m.group(1), two=True)
            ctx.derived(base + "ex.n", ctx.raw_of("PF.arm.%s.ex.n" % arm), "int", "same n as arm.%s.ex" % arm, "dev items", sc)
            _pf_count(ctx, base + "ex", "PF." + base + "ex", "PF." + base + "ex.n")
            suffixes = PF_CURVE_COPY
        else:
            suffixes = ("ex",) + PF_CURVE_COPY
        for sfx in suffixes:
            for tail in ("", ".ci", ".ci_lo", ".ci_hi", ".n", ".k"):
                src_key = "PF.arm.%s.%s%s" % (arm, sfx, tail)
                if src_key in ctx.entries:
                    e = ctx.entries[src_key]
                    ctx._put(base + sfx + tail, e.raw, e.kind, e.src, e.field, "same value as arm.%s.%s%s (curve point q=%s, n=%s)" % (arm, sfx, tail, q[1:], m.group(1)), two=e.two)
    ctx.val("PF4.complete", S, ("PF4.complete",), "str", "PF4 curve has all three points")
    ctx.val("PF4.monotone", S, ("PF4.monotone_nondecreasing",), "str", "curve monotone non-decreasing (the PF1 pre-specified criterion also needs n=1000 > n=0)")
    ctx.val("PF4.expectation", S, ("PF4.expectation",), "str", "the file's reading of the expected gain")
    pts = ctx.get_quiet(S, ("PF4.points_present",))
    if pts is not MISSING and pts is not None:
        ctx.derived("PF4.points", ", ".join(str(x) for x in pts), "str", "PF4.points_present", "feedback counts present on the curve", sc)
    # --- paired contrasts
    names = sorted(set(k.split(".")[0] for k in ctx.keys(S, ()) if k.endswith(".d_pp")))
    for nm in names:
        key = "ref." + nm[len("ref_"):] if nm.startswith("ref_") else nm
        P = nm + "."
        a, b = ctx.get_quiet(S, (P + "a",)), ctx.get_quiet(S, (P + "b",))
        txt = "%s minus %s" % (a, b)
        ctx.est2(key + ".delta", S, top(P + "d_pp"), top(P + "lo_pp"), top(P + "hi_pp"), "pp", "paired EX difference (pp), " + txt, two=True, n=top(P + "n"))
        ctx.est2(key + ".delta_nothr", S, top(P + "d_pp_no_thrombosis"), top(P + "lo_pp_no_thrombosis"), top(P + "hi_pp_no_thrombosis"), "pp", "same without thrombosis_prediction, " + txt, two=True)
        ctx._put(key + ".delta.loo", (ctx.get(S, top(P + "loo_min_pp")), ctx.get(S, top(P + "loo_max_pp"))), "pp", sc, P + "loo_min_pp , loo_max_pp", "range of the difference when one database is left out, " + txt, two=True)
        ctx.val(key + ".a_only", S, top(P + "a_only"), "int", "items only %s answers right" % a)
        ctx.val(key + ".b_only", S, top(P + "b_only"), "int", "items only %s answers right" % b)
        ctx.val(key + ".reading", S, top(P + "reading"), "str", "SAP 22 reading rule word, " + txt)
        ctx.val(key + ".a", S, top(P + "a"), "str", "arm a of the contrast")
        ctx.val(key + ".b", S, top(P + "b"), "str", "arm b of the contrast")
        if opt(P + "share_dbs_same_sign"):
            ctx.val(key + ".share_dbs_same_sign", S, top(P + "share_dbs_same_sign"), "pct", "share of databases whose difference has the same sign, " + txt)
        else:
            ctx.na.append("%s: %s (undefined: difference is exactly 0)" % (sc, P + "share_dbs_same_sign"))
    # --- PF3: calls and cost ratios against the gated loop
    for nm in sorted(set(k.split(".")[0] for k in ctx.keys(S, ()) if k.startswith("PF3") and k.endswith(".cost_ratio"))):
        P = nm + "."
        pair = ctx.get_quiet(S, (nm + "a" + ("_q100" if nm.endswith("_q100") else "") + ".a",))
        who = "%s / full_gated" % (pair if pair is not MISSING else "patent arm")
        ctx.est2(nm + ".calls_ratio", S, top(P + "calls_ratio"), top(P + "lo"), top(P + "hi"), "x2", "calls per query, " + who)
        ctx.est2(nm + ".cost_ratio", S, top(P + "cost_ratio"), top(P + "cost_lo"), top(P + "cost_hi"), "x2", "cost per query, " + who)
        ctx.val(nm + ".accuracy_reading", S, top(P + "accuracy_reading"), "str", "reading of the accuracy difference, " + who)
        ctx.val(nm + ".verdict", S, top(P + "verdict"), "str", "PF3 verdict word, " + who)
    # --- calls against the fixed loop
    for a in sorted(set(k.split(".")[1] for k in ctx.keys(S, ()) if k.startswith("calls_vs_fixed."))):
        P = "calls_vs_fixed.%s." % a
        ctx.est2("calls_vs_fixed." + a, S, top(P + "calls_ratio_vs_fixed"), top(P + "lo"), top(P + "hi"), "x2", "calls per query, %s / full_fixed" % a)


# ------------------------------------------------------------------------------------------ RV2 (posthoc-secondary, SAP Amendment 10 = section 23)
RV2_B = 10000   # bootstrap draws fixed by SAP 22/23 (a protocol constant, not a result)


def _emit_summary(ctx, src, expected, label, api_calls_ok=False):
    """shared writer of the RV2-style summary files: {"_meta": {...}, "keys": {key: {"v", "kind", "ci", "item_ci", "n", "meaning"}}}; strict against the expected key list."""
    doc = ctx.data[src]
    meta = doc.get("_meta") or {}
    path = ctx.paths[src]
    if api_calls_ok:
        if not isinstance(meta.get("api_calls"), int) or meta.get("api_calls") < 0:
            ctx.errors.append("%s: _meta.api_calls = %r, must be a non-negative integer (billed calls of the new runs)" % (path, meta.get("api_calls")))
    elif meta.get("api_calls") != 0:
        ctx.errors.append("%s: _meta.api_calls = %r, must be 0" % (path, meta.get("api_calls")))
    if meta.get("missing_keys"):
        ctx.errors.append("%s: _meta.missing_keys not empty (%d)" % (path, len(meta["missing_keys"])))
    if meta.get("B") != RV2_B:
        ctx.errors.append("%s: _meta.B = %r, must be 10000" % (path, meta.get("B")))
    keys = doc.get("keys") or {}
    exp_kinds = dict(expected)
    for k in keys:
        if k not in exp_kinds:
            ctx.errors.append("%s: key not in the grammar: %s" % (path, k))
    for k, kind in expected:
        if k not in keys:
            ctx.errors.append("%s: missing key %s" % (path, k))
            continue
        e = keys[k]
        v = e.get("v")
        if v is None:
            ctx.errors.append("%s: null at %s" % (path, k))
            continue
        if e.get("kind") != kind:
            ctx.errors.append("%s: kind of %s is %r, grammar says %r" % (path, k, e.get("kind"), kind))
            continue
        meaning = e.get("meaning") or (label + " " + k)
        if isinstance(v, bool):
            v = "yes" if v else "no"
        if isinstance(v, str):
            ctx._put(k, v, "str", path, "keys/%s/v" % k, meaning)
        elif isinstance(v, list):
            ctx._put(k, (v[0], v[1]), kind, path, "keys/%s/v" % k, meaning)
            ctx._put(k + "_lo", v[0], kind, path, "keys/%s/v[0]" % k, "lower end of " + k)
            ctx._put(k + "_hi", v[1], kind, path, "keys/%s/v[1]" % k, "upper end of " + k)
        else:
            ctx._put(k, v, kind, path, "keys/%s/v" % k, meaning)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            ci = e.get("ci")
            if ci is not None:
                if ci[0] is None or ci[1] is None:
                    ctx.errors.append("%s: interval has null end at %s" % (path, k))
                else:
                    ctx._put(k + ".ci", (ci[0], ci[1]), kind, path, "keys/%s/ci" % k, "95% cluster interval of " + k)
                    ctx._put(k + ".ci_lo", ci[0], kind, path, "keys/%s/ci[0]" % k, "lower end of " + k + ".ci")
                    ctx._put(k + ".ci_hi", ci[1], kind, path, "keys/%s/ci[1]" % k, "upper end of " + k + ".ci")
            ic = e.get("item_ci")
            if ic is not None:
                if ic[0] is None or ic[1] is None:
                    ctx.errors.append("%s: item interval has null end at %s" % (path, k))
                else:
                    ctx._put(k + ".item_ci", (ic[0], ic[1]), kind, path, "keys/%s/item_ci" % k, "95% item-level interval of " + k)
                    ctx._put(k + ".item_ci_lo", ic[0], kind, path, "keys/%s/item_ci[0]" % k, "lower end of " + k + ".item_ci")
                    ctx._put(k + ".item_ci_hi", ic[1], kind, path, "keys/%s/item_ci[1]" % k, "upper end of " + k + ".item_ci")
            if e.get("n") is not None:
                ctx._put(k + ".n", e["n"], "int", path, "keys/%s/n" % k, "sample size of " + k)
    return keys


def build_rv3(ctx):
    """results/r3/rv3_summary.json (round-3 post hoc analyses, merged by code/tools/r3_merge.py; grammar code/tools/rv3_keys.py = rv3_grammar.json; prefix RV3.): same format and
    strict gate as build_rv2, except that _meta.api_calls counts the billed calls of the new runs (seeds, same-examples control) instead of being 0."""
    start(ctx, "RV3.", "RV3")
    src = "rv3"
    if src in ctx.absent:
        ctx.errors.append("source file missing: %s" % ctx.paths[src])
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rv3_keys
    _emit_summary(ctx, src, rv3_keys.expected_keys(), "RV3", api_calls_ok=True)


def build_rv4(ctx):
    """results/r4/rv4_summary.json (round-4 post hoc statistics, merged by code/tools/r4_merge.py from r4_stats.py; grammar code/tools/rv4_keys.py = rv4_grammar.json; prefix RV4.): same
    format and strict gate as build_rv2 (_meta.api_calls must be 0: no model call)."""
    start(ctx, "RV4.", "RV4")
    src = "rv4"
    if src in ctx.absent:
        ctx.errors.append("source file missing: %s" % ctx.paths[src])
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rv4_keys
    _emit_summary(ctx, src, rv4_keys.expected_keys(), "RV4")


def build_rv2s(ctx):
    """results/posthoc2/rv2s_summary.json (statistics add-on S2, grammar code/tools/rv2s_keys.py, prefix RV2.small|deff|margin|cost.nodisc|gate): same format and strict gate as build_rv2."""
    start(ctx, "RV2.", "RV2")
    src = "rv2s"
    if src in ctx.absent:
        ctx.errors.append("source file missing: %s" % ctx.paths[src])
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rv2s_keys
    keys = _emit_summary(ctx, src, rv2s_keys.expected_keys(), "RV2S")
    # intervals of the existing RV2.loop.<b>.margin_over_eff_<in|btw> (the margin divided by the FIXED-loop effect): .ci (cluster) and .item_ci where both ends are finite;
    # the open-ended ones (effect interval reaches 0) are only in the text keys RV2.margin.<b>.<d>.fixed.*_text
    for b in rv2s_keys.PB:
        for d in ("in", "btw"):
            base = "loop.%s.margin_over_eff_%s" % (b, d)
            if "RV2." + base not in ctx.entries:
                continue
            for t, suf in (("cl", ".ci"), ("item", ".item_ci")):
                lo, hi = keys["margin.%s.%s.fixed.lo_%s" % (b, d, t)]["v"], keys["margin.%s.%s.fixed.hi_%s" % (b, d, t)]["v"]
                if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                    f = "keys/margin.%s.%s.fixed.lo_%s,hi_%s" % (b, d, t, t)
                    ctx._put(base + suf, (lo, hi), "x2", ctx.paths[src], f, "95% interval of " + base + " (" + ("cluster" if t == "cl" else "item") + "): 1.5 pp divided by the interval of the fixed-loop effect")
                    ctx._put(base + suf + "_lo", lo, "x2", ctx.paths[src], f, "lower end of " + base + suf)
                    ctx._put(base + suf + "_hi", hi, "x2", ctx.paths[src], f, "upper end of " + base + suf)


def build_rv2(ctx):
    """results/posthoc2/rv2_summary.json: {"_meta": {...}, "keys": {key: {"v", "kind", "ci", "item_ci", "n", "meaning"}}}. The expected key set is the grammar of
    code/tools/rv2_keys.py; a missing, extra or null key, a kind that differs from the grammar, api_calls != 0 or an unfinished run fails the strict gate."""
    start(ctx, "RV2.", "RV2")
    src = "rv2"
    if src in ctx.absent:
        ctx.errors.append("source file missing: %s" % ctx.paths[src])
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rv2_keys
    doc = ctx.data[src]
    meta = doc.get("_meta") or {}
    path = ctx.paths[src]
    if meta.get("api_calls") != 0:
        ctx.errors.append("%s: _meta.api_calls = %r, must be 0" % (path, meta.get("api_calls")))
    if meta.get("missing_keys"):
        ctx.errors.append("%s: _meta.missing_keys not empty (%d)" % (path, len(meta["missing_keys"])))
    if meta.get("B") != RV2_B:
        ctx.errors.append("%s: _meta.B = %r, must be 10000" % (path, meta.get("B")))
    keys = doc.get("keys") or {}
    expected = rv2_keys.expected_keys()
    exp_kinds = dict(expected)
    for k in keys:
        if k not in exp_kinds:
            ctx.errors.append("%s: key not in the grammar: %s" % (path, k))
    for k, kind in expected:
        if k not in keys:
            ctx.errors.append("%s: missing key %s" % (path, k))
            continue
        e = keys[k]
        v = e.get("v")
        if v is None:
            ctx.errors.append("%s: null at %s" % (path, k))
            continue
        if e.get("kind") != kind:
            ctx.errors.append("%s: kind of %s is %r, grammar says %r" % (path, k, e.get("kind"), kind))
            continue
        meaning = e.get("meaning") or ("RV2 " + k)
        if isinstance(v, bool):
            v = "yes" if v else "no"
        if isinstance(v, str):
            ctx._put(k, v, "str", path, "keys/%s/v" % k, meaning)
        elif isinstance(v, list):
            ctx._put(k, (v[0], v[1]), kind, path, "keys/%s/v" % k, meaning)
            for suf in (".ci", ".item_ci", ".t_ci"):
                if k.endswith(suf):
                    ctx._put(k + "_lo", v[0], kind, path, "keys/%s/v[0]" % k, "lower end of " + k)
                    ctx._put(k + "_hi", v[1], kind, path, "keys/%s/v[1]" % k, "upper end of " + k)
        else:
            ctx._put(k, v, kind, path, "keys/%s/v" % k, meaning)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            ci = e.get("ci")
            if ci is not None:
                if ci[0] is None or ci[1] is None:
                    ctx.errors.append("%s: interval has null end at %s" % (path, k))
                else:
                    ctx._put(k + ".ci", (ci[0], ci[1]), kind, path, "keys/%s/ci" % k, "95% cluster interval of " + k)
                    ctx._put(k + ".ci_lo", ci[0], kind, path, "keys/%s/ci[0]" % k, "lower end of " + k + ".ci")
                    ctx._put(k + ".ci_hi", ci[1], kind, path, "keys/%s/ci[1]" % k, "upper end of " + k + ".ci")
            ic = e.get("item_ci")
            if ic is not None:
                if ic[0] is None or ic[1] is None:
                    ctx.errors.append("%s: item interval has null end at %s" % (path, k))
                else:
                    ctx._put(k + ".item_ci", (ic[0], ic[1]), kind, path, "keys/%s/item_ci" % k, "95% item-level interval of " + k)
                    ctx._put(k + ".item_ci_lo", ic[0], kind, path, "keys/%s/item_ci[0]" % k, "lower end of " + k + ".item_ci")
                    ctx._put(k + ".item_ci_hi", ic[1], kind, path, "keys/%s/item_ci[1]" % k, "upper end of " + k + ".item_ci")
            if e.get("n") is not None:
                ctx._put(k + ".n", e["n"], "int", path, "keys/%s/n" % k, "sample size of " + k)
    # derived counts over the dual-interval rows (design effect below 1; rows whose reading changes between the two interval types)
    dk = [k for k, _ in expected if k.startswith("dual.") and k.endswith(".deff") and k in keys and isinstance(keys[k].get("v"), (int, float))]
    if dk:
        ctx.derived("dual.n_rows", len(dk), "int", "number of dual.<id>.deff keys", "contrasts with both interval types", path)
        ctx.derived("dual.deff_lt1.n", len(dk), "int", "number of dual.<id>.deff keys", "contrasts with an observed design effect (denominator of deff_lt1.k)", path)
        ctx.derived("dual.deff_lt1.k", sum(1 for k in dk if keys[k]["v"] < 1), "int", "count of dual.<id>.deff < 1", "contrasts whose observed design effect is below 1", path)
    rc = [k for k, _ in expected if k.startswith("dual.") and k.endswith(".reading_changes") and k in keys]
    if rc:
        ctx.derived("dual.n_reading_changes", sum(1 for k in rc if keys[k]["v"] in (True, "yes")), "int", "count of dual.<id>.reading_changes = yes", "contrasts whose reading changes between the two interval types", path)


def build_rv4t0(ctx):
    """results/r4_t0/rv4t0_summary.json (temperature-0 non-determinism check, written by code/tools/r4_t0.py; grammar code/tools/rv4t0_keys.py = rv4t0_grammar.json; prefix RV4T0.): same format
    and strict gate as build_rv3 (_meta.api_calls counts the billed calls of the rerun)."""
    start(ctx, "RV4T0.", "RV4T0")
    src = "rv4t0"
    if src in ctx.absent:
        ctx.errors.append("source file missing: %s" % ctx.paths[src])
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rv4t0_keys
    _emit_summary(ctx, src, rv4t0_keys.expected_keys(), "RV4T0", api_calls_ok=True)


BUILDERS = [build_rob, build_diag, build_fbk, build_vad, build_ve, build_ext1, build_e10, build_pf, build_rv2, build_rv2s, build_rv3, build_rv4, build_rv4t0]


def build_all(results_dir, allow_missing=False):
    ctx = Ctx(results_dir, allow_missing)
    for b in BUILDERS:
        b(ctx)
    # K.fmt for every estimate with an interval; K.d2 twins
    for key in list(ctx.entries):
        if key.endswith(".ci") and key[:-3] in ctx.entries:
            e, c = ctx.entries[key[:-3]], ctx.entries[key]
            if not isinstance(e.raw, tuple) and not isinstance(e.raw, (str, bool)):
                ctx.entries[key[:-3] + ".fmt"] = Entry(key[:-3] + ".fmt", (e, c), "fmt", e.src, e.field, "estimate and interval, text form", e.label)
    for key in list(ctx.entries):
        e = ctx.entries[key]
        if e.two and e.kind in TWO_DECIMAL and not key.endswith(".fmt"):
            ctx.entries[key + ".d2"] = Entry(key + ".d2", e.raw, TWO_DECIMAL[e.kind], e.src, e.field, e.meaning + " (two decimals)", e.label)
    ctx.n_aliases = add_aliases(ctx)
    return ctx


def value_text(e, entries):
    if e.kind == "fmt":
        est, ci = e.raw
        return "%s %s" % (est.text(), ci.text())
    return e.text()


# ------------------------------------------------------------------------------------------ output
def write_macros(ctx, path):
    lines = ["% GENERATED by code/tools/make_xmacros.py -- do not edit. Exploratory, robustness-posthoc and external-validity numbers; scope labels in internal note XMACROS.",
             "%% Used only through \\xres{KEY}; never mixed with \\res (confirmatory). %s" % time.strftime("generated %Y-%m-%d %H:%M:%S")]
    for p, h in sorted(ctx.shas.items()):
        lines.append("%% %s sha256 %s" % (p, h))
    lines += ["\\makeatletter", "\\def\\xres#1{\\@ifundefined{xresv@#1}{\\resfallback{#1}}{\\csname xresv@#1\\endcsname}}"]
    for key in sorted(ctx.entries):
        e = ctx.entries[key]
        lines.append("\\expandafter\\gdef\\csname xresv@%s\\endcsname{%s}" % (key, value_text(e, ctx.entries)))
    lines.append("\\makeatother")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_macros(path):
    out = {}
    rx = re.compile(r"^\\expandafter\\gdef\\csname xresv@(.+?)\\endcsname\{(.*)\}$")
    for ln in open(path, encoding="utf-8"):
        m = rx.match(ln.rstrip("\n"))
        if m:
            out[m.group(1)] = m.group(2)
    return out


# ------------------------------------------------------------------------------------------ legacy-name aliases
def alias_pairs(ctx):
    """(alias key, existing key) pairs for names that the supplement and drafts use: the result files' own spelling of the same cell.
    ROB.<source contrast>.{d0,d1,shift}; DIAG.pooled.{detection,fp}; VAD.<v>.{delta_fpr,delta_detection}; VE.q<q as in the file, e.g. 0.85>.<policy>.<METRIC upper case>.
    An alias is a copy of the target (same text), listed in the catalogue with 'alias of'."""
    out = []
    for nm, (rel, kind) in rob_contrast_specs(ctx).items():
        if rel[0] != "contrasts":
            continue
        src_name = rel[1]
        for new, old in (("d0", "original"), ("d1", "corrected"), ("shift", "shift")):
            out.append(("ROB.%s.%s" % (src_name, new), "ROB.%s.%s" % (nm, old)))
    out += [("DIAG.pooled.detection", "DIAG.all.detect"), ("DIAG.pooled.fp", "DIAG.all.fpr")]
    for v in ctx.keys("v2", ("decision",)):
        out += [("VAD.%s.delta_fpr" % v, "VAD.%s.d_fpr" % v), ("VAD.%s.delta_detection" % v, "VAD.%s.d_detection" % v)]
    for q in ctx.keys("ve", ("by_q",)):
        for pol in ctx.keys("ve", ("by_q", q, "point")):
            for m in ctx.keys("ve", ("by_q", q, "point", pol)):
                out.append(("VE.q%s.%s.%s" % (q, pol, m), "VE.%s.%s.%s" % (qlab(q), pol, m.lower())))
    return out


def add_aliases(ctx):
    n = 0
    for new, old in alias_pairs(ctx):
        for suf in ("", ".ci", ".ci_lo", ".ci_hi", ".fmt", ".n", ".k"):
            if old + suf in ctx.entries and new + suf not in ctx.entries:
                e = ctx.entries[old + suf]
                ctx.entries[new + suf] = Entry(new + suf, e.raw, e.kind, e.src, e.field, "alias of %s%s: %s" % (old, suf, e.meaning), e.label)
                n += 1
    return n


# ------------------------------------------------------------------------------------------ global key check
RES_USE = re.compile(r"\\(res|xres|tsres)\{([^{}]+)\}")


def strip_tex_comments(tex):
    return "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in tex.splitlines())


def paper_sources(paper_dir):
    import glob
    fs = [os.path.join(paper_dir, "main.tex"), os.path.join(paper_dir, "supplement.tex")]
    for sub in ("sections", "tables", "supp_tables"):
        fs += sorted(glob.glob(os.path.join(paper_dir, sub, "*.tex")))
    return [f for f in fs if os.path.exists(f)]


def defined_keys(paper_dir, xkeys):
    def grab(name, pat):
        p = os.path.join(paper_dir, name)
        return set(re.findall(pat, open(p, encoding="utf-8", errors="replace").read())) if os.path.exists(p) else set()
    return {"res": grab("results_macros.tex", r"csname resv@(.+?)\\endcsname"), "tsres": grab("results_macros_ts.tex", r"csname tsv@(.+?)\\endcsname"), "xres": set(xkeys)}


def check_paper(paper_dir, xkeys):
    """-> {macro: {key: [files]}} of keys used in main.tex, sections, tables, supplement.tex, supp_tables that no generated macro file defines"""
    defs = defined_keys(paper_dir, xkeys)
    gaps = collections.OrderedDict((m, {}) for m in ("res", "xres", "tsres"))
    used_n = collections.defaultdict(set)
    for f in paper_sources(paper_dir):
        txt = strip_tex_comments(open(f, encoding="utf-8", errors="replace").read())
        for m in RES_USE.finditer(txt):
            k = m.group(2).strip()
            if "#" in k or k == "KEY":
                continue
            used_n[m.group(1)].add(k)
            if k not in defs[m.group(1)]:
                gaps[m.group(1)].setdefault(k, set()).add(os.path.relpath(f, paper_dir))
    return gaps, used_n


# ------------------------------------------------------------------------------------------ catalogue (internal note XMACROS)
CLAIMS = [   # (row, claim, keys); keys are checked against the generated set
    ("A5 / 20-21", "Corrected-gold weakening, S_same: n, gated minus fixed pooled and per backbone", ["ROB.n_same", "ROB.H1.pooled.corrected", "ROB.H1.pooled.corrected.ci", "ROB.H1.pooled.original", "ROB.H1.pooled.original.ci", "ROB.H1.pooled.shift", "ROB.H1.pooled.shift.ci", "ROB.H1.pooled.rule", "ROB.H1.flash.corrected", "ROB.H1.flash.corrected.ci", "ROB.H1.27b.corrected", "ROB.H1.27b.corrected.ci", "ROB.H1.plus.corrected", "ROB.H1.plus.corrected.ci", "ROB.H1.plus.rule", "ROB.C1.n_not_robust"]),
    ("20", "Strong backbone unchanged: routing AUROC, revisions", ["ROB.H4.dauroc.original", "ROB.H4.dauroc.corrected", "ROB.H4.dauroc.corrected.ci", "ROB.H5.revision.corr.helped", "ROB.H5.revision.corr.hurt", "ROB.H5.revision.orig.helped", "ROB.H5.revision.orig.hurt", "ROB.H5.net_exec.corrected", "ROB.H5.net_exec.corrected.ci"]),
    ("22", "Pipeline gain shrinks (W1 under corrected gold)", ["ROB.H2.pooled.original", "ROB.H2.pooled.corrected", "ROB.H2.pooled.corrected.ci", "ROB.H2.27b.corrected", "ROB.H2.27b.corrected.ci", "ROB.H2.flash.corrected", "ROB.H2.flash.corrected.ci", "ROB.H2.plus.corrected", "ROB.H2.plus.corrected.ci"]),
    ("23", "L6 corroborated (24 of 29 vs 30 of 65)", ["ROB.e10_l6.L6.k", "ROB.e10_l6.L6.n", "ROB.e10_l6.L6", "ROB.e10_l6.L6.ci", "ROB.e10_l6.nonL6.k", "ROB.e10_l6.nonL6.n", "ROB.e10_l6.nonL6", "ROB.e10_l6.diff", "ROB.e10_l6.diff.ci"]),
    ("4.4 why", "Pair audit on flash", ["ROB.audit.flash.pairs_differ", "ROB.audit.flash.fixed_only_right_orig", "ROB.audit.flash.fixed_only_right_corr", "ROB.n_eff_change", "ROB.share_dev", "ROB.n_corrected_items"]),
    ("14 / A6", "Semantic revisions net zero (qwen3.7-plus, fixed loop, all dev items)", ["E10.revision.n", "E10.revision.helped", "E10.revision.hurt", "E10.revision.neutral", "E10.revision.gated.n", "E10.discordant.n"]),
    ("15", "E10 agreement between the two runs (headline kappa and its interval stay `\\res{E10.kappa}`, `\\res{E10.kappa.ci}`; this is the agreement.json bootstrap)", ["E10.agreement.kappa", "E10.agreement.kappa.ci", "E10.agreement.ac1", "E10.n_items", "E10.n_disagree"]),
    ("16 / A8", "Critic sensitive, not specific (pooled and DISTINCT)", ["DIAG.all.detect", "DIAG.all.detect.ci", "DIAG.all.fpr", "DIAG.all.fpr.ci", "DIAG.all.auroc", "DIAG.all.auroc_all", "DIAG.all.harm", "DIAG.distinct_drop.detect", "DIAG.distinct_drop.fpr", "DIAG.distinct_drop.auroc", "DIAG.order_flip.fpr", "DIAG.all.gated_revises.k", "DIAG.all.repair1_given_det"]),
    ("17", "Feedback calibration gives no gain", ["FBK.verdict", "FBK.logreg.n400.auroc", "FBK.logreg.n400.d_auroc", "FBK.logreg.n400.d_auroc.ci", "FBK.threshold.n400.fp.d_det", "FBK.logreg.n400.fp.d_det", "FBK.logreg.n400.det.d_fpr", "FBK.cat.distinct_drop.logreg.n400.fp.det", "FBK.cat.distinct_drop.logreg.n400.auroc"]),
    ("18", "Better-specified Critics fail the rule", ["VAD.VC.d_fpr", "VAD.VC.d_fpr.ci", "VAD.VC.d_detection", "VAD.VC.d_detection.ci", "VAD.VC.extra_calls_vs_VA", "VAD.VC.rule_met", "VAD.VD.rule_met", "VAD.V0.fpr", "VAD.V0.detection"]),
    ("19", "Critic stubborn, not sycophantic (simulated user)", ["VE.A.firm_correct", "VE.A.firm_correct.ci", "VE.A.resp_all", "VE.A.resp_all.ci", "VE.q085.A.ccr", "VE.q085.obey.ccr", "VE.q085.A_vs_obey.j", "VE.q085.A_vs_obey.j.ci", "VE.q085.A_vs_A0.j", "VE.q085.A_vs_A0.j.ci", "VE.breakeven_q.A", "VE.q085.calls.A", "VE.C.firm_correct", "VE.q085.rule.A.text"]),
    ("24", "Cost replicates on Spider-Realistic (XB3)", ["EXT1.XB3.calls_ratio", "EXT1.XB3.calls_ratio.ci", "EXT1.XB3.delta_ex", "EXT1.XB3.delta_ex.ci", "EXT1.XB3.p_adj", "EXT1.XB3.verdict", "EXT1.XB3.n"]),
    ("25", "Weak-backbone gain not established (XB1), gate", ["EXT1.XB1.delta_ex", "EXT1.XB1.delta_ex.ci", "EXT1.XB1.p_adj", "EXT1.XB1.verdict", "EXT1.XB1_loop.delta_ex", "EXT1.XB1_loop.delta_ex.ci", "EXT1.gate.pooled.fired", "EXT1.gate.pooled.n"]),
    ("28 / PF", "Feedback-updated Critic: no gain with use (simulated oracle feedback)", ["PF.PF1.delta", "PF.PF1.delta.ci", "PF.PF1.reading", "PF.PF2.delta", "PF.PF2.delta.ci", "PF.PF2_n0.delta", "PF.PF2_n0.delta.ci", "PF.PF3.calls_ratio", "PF.PF3.cost_ratio", "PF.PF3.verdict", "PF.PF3a.delta", "PF.PF3a.delta.ci", "PF.PF4.monotone", "PF.curve.q100.n0.ex", "PF.curve.q100.n250.ex", "PF.curve.q100.n1000.ex", "PF.curve.q085.n1000.ex", "PF.arm.pf_n0.detection", "PF.arm.pf_n250_q100.detection", "PF.arm.pf_n1000_q100.detection", "PF.arm.pf_n1000_q100.helped", "PF.arm.pf_n1000_q100.hurt", "PF.arm.pf_n1000_q085.cost_k", "PF.cost_total"]),
    ("26", "Strong backbone (XB2), world_1 and scorer sensitivity", ["EXT1.XB2.delta_ex", "EXT1.XB2.delta_ex.ci", "EXT1.XB2.text", "EXT1.sens_no_world_1.XB2.delta_ex", "EXT1.sens_no_world_1.XB2.item_ci", "EXT1.sens_no_world_1.XB2.verdict", "EXT1.arm.plus.gated.ex_plain", "EXT1.arm.plus.zs.ex_plain"]),
]
NOT_EXPOSED = [
    "ROB: item-level intervals (`item_ci*`), per-database tables (`by_db`), `disc_orig/disc_corr` counts, flip transitions, gold-failure counts, validation-gate lists, case lists (`cases_fixed_rule` items); G2-G4 and S_all expose only the headline cells (H1, H2, H4.dauroc, H5.net_exec).",
    "DIAG: strata (`strata_descriptive`), score distributions and curves (`critic_diag_fig.json`), raw cases; `auroc_score_vs_gold_all_controls` is null in the file.",
    "FBK: seed ranges, the other seven categories, LR coefficients (`addendum_descriptive.mean_lr_coefficients`); only the two hard categories are exposed per budget.",
    "VAD: `critic_v2_audit_summary.json` (schema-pruning audit) and issue-level text; per-category cells carry point estimates only (the file has no per-category intervals).",
    "VE: seeded realisation (`realisation`), expected disputes by class, `q = 1.00` decision rule (absent in the file).",
    "EXT1: `bird_reference` (copied by hand inside the file; use `\\res{W1.*}` / `\\res{P1prime.*}` for the BIRD side), `arms_complete` list.",
    "E10: label prevalence and the 8-label taxonomy stay under `\\res{E10.*}` (results/E10.json, confirmatory file).",
    "PF: `pf_cases.json` (PF1/PF2 flip cases), `pf_arms.csv` / `pf_contrasts.csv` / `pf_curve.csv` (copies of the summary; the summary is the only source read), `tuning/`; `ex_nothr` contrasts exist for each contrast but no per-category cells; the file has no interval for Critic detection / false-flag / flag rates (point + n + k only).",
]


def write_doc(ctx, path):
    keys = sorted(ctx.entries)
    byp = collections.Counter(k.split(".")[0] for k in keys)
    base = [k for k in keys if not re.search(r"\.(ci_lo|ci_hi|fmt|d2)$", k)]
    missing_claim = [k for _, _, ks in CLAIMS for k in ks if k not in ctx.entries]
    tsv = os.path.splitext(path)[0] + "_keys.tsv"
    with open(tsv, "w", encoding="utf-8") as f:
        f.write("key\tvalue_as_typeset\tkind\tscope_label\tsource_file\tsource_field\tmeaning\n")
        for k in keys:
            e = ctx.entries[k]
            f.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (k, value_text(e, ctx.entries).replace("\t", " "), e.kind, e.label, e.src, e.field, e.meaning.replace("\t", " ")))
    L = []
    w = L.append
    w("# Handoff XMACROS: `\\xres{KEY}` macros and tables for exploratory, robustness and external-validity numbers")
    w("")
    w("Generated by `code/tools/make_xmacros.py --doc internal note XMACROS` from `results/{robust,diag,ext1,e10}/*.json` (this file is regenerated; do not edit by hand). Macro file: `paper/results_macros_x.tex`. Tables: `paper/tables/tab_robust.tex`, `tab_external.tex`, `tab_critic_diag.tex` (`code/tools/make_xtables.py`). Complete key list with the typeset value, source file and field: `internal note XMACROS_keys` (%d keys)." % len(keys))
    w("")
    w("## 1. Rules for the writers")
    w("")
    w("* Use `\\xres{KEY}` for every number below; never `\\res{}` (confirmatory files only). A missing key prints a bold `[KEY]`.")
    w("* Formatting is fixed in the generator (STYLE_SPEC section 7): differences in pp with sign and one decimal, EX and rates in percent with one decimal, ratios two decimals, AUROC three decimals, p with three decimals or `<0.001`; the minus is `\\ensuremath{-}`; an interval is `[lo, hi]`.")
    w("* Every estimate `K` with an interval has `K.ci` (text `[lo, hi]`), `K.ci_lo`, `K.ci_hi`, and `K.fmt` (`est [lo, hi]`). A sample size is `K.n`, a count `K.k`. Cells that tables print with two decimals have a twin `K.d2` (and `K.ci.d2`).")
    w("* Scope labels (put the word in the sentence, STYLE_SPEC 7): %s." % "; ".join("`%s.` = %s" % (k, v) for k, v in sorted(LABELS.items())))
    w("* The abstract skeleton (R3.3) writes `\\res{ROB.n_same}`, `\\res{ROB.H1.pooled.corrected}` and `\\res{E10.revision.helped}`: change `ROB.*` to `\\xres`; `E10.revision.*` exists under both (the `\\res` copy is results/E10.json, the `\\xres` copy is read from results/e10/discordant_summary.json; the tests assert they agree).")
    w("* Intervals: ROB, DIAG, FBK, VAD, VE, EXT1 are 95% database-clustered percentile bootstrap intervals (B in each file's `_meta`); DIAG `*.fpr`, `*.detect` etc. use the database-clustered bootstrap of the rate; E10 kappa is a bootstrap over sheet items; L6 shares are Wilson, their difference Newcombe.")
    w("")
    w("## 2. Key count by prefix")
    w("")
    w("| Prefix | Keys (all) | Base keys (without .ci_lo, .ci_hi, .fmt, .d2) | Scope label | Sources |")
    w("|---|---|---|---|---|")
    srcs = collections.defaultdict(set)
    for k in keys:
        srcs[k.split(".")[0]].add(ctx.entries[k].src)
    for p in sorted(byp):
        w("| `%s.` | %d | %d | %s | %s |" % (p, byp[p], len([k for k in base if k.split(".")[0] == p]), LABELS.get(p, ""), ", ".join("`%s`" % x for x in sorted(set(y.strip() for z in srcs[p] for y in z.split(" + "))))))
    w("| total | %d | %d | | |" % (len(keys), len(base)))
    w("")
    w("## 3. Claim table (STORYLINE R3.10) and abstract rows -> keys")
    w("")
    w("| Row | Claim | Keys (value as typeset) |")
    w("|---|---|---|")
    for row, claim, ks in CLAIMS:
        cell = "<br>".join("`%s` = %s" % (k, value_text(ctx.entries[k], ctx.entries)) if k in ctx.entries else "`%s` **MISSING**" % k for k in ks)
        w("| %s | %s | %s |" % (row, claim, cell))
    w("")
    w("Rows 1-13 and 27 are confirmatory/tuning rows (`\\res`, `\\tsres`); row 28 is `PF` (see section 4, `PF.`).")
    if missing_claim:
        w("")
        w("**Claim keys missing from this run:** " + ", ".join("`%s`" % k for k in missing_claim))
    w("")
    w("## 4. Key grammar by prefix (every base key is in the .tsv)")
    w("")
    w("**`ROB.`** (results/robust/rescore.json; primary = variant G1, set S_same; scope ROB). Cell `ROB.<contrast>.<field>` with contrasts `H1.{plus,flash,27b,pooled}` (gated minus fixed; `pooled` = P1'), `H2.{flash,27b,pooled,plus}` (gated minus single call; `pooled` = W1 on the weak backbones, `plus` = control), `H2b.<b>.{loop,stack}` (loop over root-only; root-only minus single call), `H3.<pair>` (descriptive baselines), `H4.{dauroc,daurc,dacc60}` (routing, route_ca minus V3), `H5.{net_exec,net_repair}`. Fields: `original`, `corrected`, `shift` (corrected minus original, same items), each with `.ci`; `n`; `robust`, `stable`, `rule` (SAP 20.7 text: `robust, stable` / `robust` / `not robust`); `nothr.*` = without thrombosis_prediction (H1, H2). Other prefixes: `sall.<contrast>.*` (set S_all), `g2.`, `g3.`, `g4.` (variants, S_same), `level.<system>.{original,corrected,shift}` (EX in %), `audit.<b>.*`, `rank.*`, `flips.<run>.{gain,loss}`, `e10_l6.*`, `C1..C4.{unchanged,unchanged_and_stable,n_not_robust}`, set sizes `n_same n_all n_eff_change n_text_changed n_db_discordant n_corrected_items share_dev`.")
    w("")
    w("**`DIAG.`** (critic_diag_summary.json): `DIAG.<category|all>.<metric>` with `n_inj n_gold detect detect_mustfix detect_issues fpr fpr_mustfix repair1 repair1_given_det final_ex gold_keep harm gated_revises score_inj score_gold auroc auroc_sem`; rate metrics have `.ci`, `.n`, `.k`; `harm` = 1 - `gold_keep` (derived). Categories: %s." % ", ".join("`%s`" % c for c in ctx.keys("diag", ("per_category",))))
    w("")
    w("**`FBK.`** (feedback_calibration.json): `FBK.verdict` (`no_help`), `FBK.<learner>.n<N>.<rule>.<metric>` with learner `threshold|logreg`, N the number of labelled items, rule `fp` (max detection at the uncalibrated false-positive rate) or `det` (min false-positive rate at the uncalibrated detection), metric `det|fpr` (realised value) or `d_det|d_fpr` (difference to N = 0, in pp); `FBK.<learner>.n<N>.auroc` and `.d_auroc`; hard categories `FBK.cat.<c>.<learner>.n<N>...`; sub-score means `FBK.comp.<c>.<injected|gold_control>.<syntax|semantic|efficiency>`.")
    w("")
    w("**`VAD.`** (critic_v2_summary.json): `VAD.<V0|VA|VB|VC|VD>.<detection|fpr|harm|repair1|repair_final|auroc>` (+`.ci`), differences to V0 `VAD.<v>.d_<metric>` (+`.ci`), `rule_met`, `strict_met` (SAP 18.7), `calls`, `extra_calls_vs_VA`, `extra_calls_vs_V0`; `VAD.V0sem.auroc`; `VAD.cat.<category>.<v>.<detection|fpr|harm>`; `VAD.stage.*`, `VAD.appeal.*` counts.")
    w("")
    w("**`VE.`** (critic_ve_summary.json; simulated user): arbitration quality, independent of q: `VE.<arm>.<firm_correct|firm_pass|firm_all|resp_all|resp_wrong|resp_inj|resp_gold|resp_missed|balacc>` (+`.ci`, `.n`) for arms A (same model, neutral prompt), A0 (no user statement), B (other family), C (old prompt, flags only; pass cells are null in the file and skipped); `hold_all`, `invalid`, `n_units`. Outcomes by user accuracy q: `VE.q<100|085|070>.<policy>.<det|udet|fpr|ccr|wcr|mer|tnr|harm_lb|j>` for policies `none obey oracle_arbiter oracle_all A A0 B C`; paired differences `VE.q085.<X>_vs_<Y>.<metric>` = X minus Y in pp (`A_vs_obey`, `A_vs_none`, `A_vs_A0`, `B_vs_obey`, ...); `VE.q085.calls.<arm>` (extra LLM calls per query), `.disputes.<arm>`; `VE.q085.rule.<arm>.*` (SAP 19.5/19.9 clauses, `.text`); `VE.breakeven_q.<A|B>` (derived: linear interpolation of J(arm - obey) = 0 between the computed q); `VE.units.*`.")
    w("")
    w("**`PF.`** (results/pf/pf_summary.json; scope exploratory-posthoc, SAP Amendment 9 = Section 22; feedback is a SIMULATED ORACLE user, so every value is an upper bound for a real user; `PF.n_items`, `PF.n_dbs`, `PF.boot_B`, `PF.cost_total` = CNY spent on the whole PF block, `PF.feedback_note`). Arms: `PF.arm.<id>.<metric>` with ids `zs full_fixed full_gated route_ca pf_n0 pf_n250_q100 pf_n1000_q100 pf_n1000_q085` (`pf_n<N>_q<QQ>` = N feedback items, q = QQ/100; `.fb` and `.q` are read off the id). Metrics: `ex` (% , `.ci`, `.fmt`, `.d2`, `.n`, `.k`), `ex_nothr`, `calls_per_q`, `embed_per_q`, `rerank_per_q` (not in the call count), `prompt_tok_per_q`, `completion_tok_per_q`, `cost_per_q` (CNY), `cost_k` (CNY per 1 000 questions, derived), `cost_total`, `cost_per_correct`, `revisions`, `revision_share`, `helped`, `hurt`, `net` (helped minus hurt, pp, with `.ci`), and for arms with a Critic `flag_rate`, `detection` (share of wrong root SQL flagged; `.n` = wrong roots, `.k`), `false_flag` (right roots flagged; `.n`, `.k`), `parse_fail`, `root_identical`, `rejected_share` (items whose rubric holds a rejected case), `flag_rate_rejected` / `flag_rate_no_rejected` (flag rate with and without one, `.n`, `.k`; absent for pf_n0, which has no cases). Learning curve: `PF.curve.q100.{n0,n250,n1000}.*` and `PF.curve.q085.{n0,n1000}.*` (`q085.n0` is the no-feedback arm, shared) with `ex` (+ci, n), `fb`, and the arm keys `calls_per_q cost_k detection false_flag flag_rate helped hurt net revisions` copied; `PF.PF4.{complete,monotone,expectation,points}`. Paired contrasts `PF.<c>.{delta,delta_nothr}` (pp, `.ci`, `.fmt`, `.d2`, `.n`; `delta.loo` = leave-one-database-out range), `.a_only`, `.b_only`, `.reading` (SAP 22 reading word), `.a`, `.b`, `.share_dbs_same_sign`, for `c` in `PF1` (n=1000 q=1 vs n=0), `PF2` (n=1000 q=0.85 vs full_fixed), `PF2_q100`, `PF2_n0` (n=0 vs full_fixed), `PF3a` (n=1000 q=0.85 vs full_gated), `PF3a_q100`, `PF4_250_vs_0`, `PF4_1000_vs_250`, `PF4_noise_cost`, and `ref.<x>` (`ref.fixed_vs_gated`, `ref.pf_n0_vs_zs`, `ref.pf_n250_q100_vs_zs`, ...). Cost and calls against the gated loop: `PF.PF3.{calls_ratio,cost_ratio}` (+ci, q=0.85 arm) and `PF.PF3_q100.*` with `.accuracy_reading` and `.verdict`; `PF.calls_vs_fixed.<arm>` (calls ratio to the fixed loop, with interval).")
    w("**`RV2.`** (results/posthoc2/rv2_summary.json; scope posthoc-secondary, SAP Amendment 10 = Section 23; computed on existing runs, zero API calls, never `\\res`, never in the P* family; the grammar of the 2688 base keys is `code/tools/rv2_keys.py`, described in `internal note RV2`): `RV2.{loop,roots,cost,pareto,dual,route,het,rev}.<...>`. A base key with an interval carries `.ci` (cluster), `.item_ci`, `.ci_lo`, `.ci_hi`, `.item_ci_lo`, `.item_ci_hi`, `.n`; the base keys `dual.<id>.{ci,item_ci,t_ci}` are interval texts and carry `_lo`/`_hi`. Tables: `tab_loop.tex`, `tab_cost.tex`, `tab_dual.tex`, `tab_dual_full.tex`. **`RV2.` statistics add-on S2** (results/posthoc2/rv2s_summary.json, grammar `code/tools/rv2s_keys.py`, described in `internal note S2`; post hoc, robustness, zero API calls): `RV2.small.<id>.*` (small-cluster readings of every `RV2.dual.<id>` contrast and of the loop effects `loop.<b>.eff_<in|btw>.<arm>`, `loop.<b>.stack`: intervals `{cl,item,t,tdb,sf,sfu,wide,wideu,old}_ci`, p-values `p_{t,tdb,sf,sfu}`, reading words `class_*`, `reading_changes`), `RV2.small.pstar.*` (family P* re-tested), `RV2.deff.*` (design-effect audit), `RV2.margin.*` (1.5 pp margin divided by the loop effect, with intervals), `RV2.cost.nodisc.*` (cost ratios with and without the cache discount, front, route cost per correct answer, cached-token audit), `RV2.gate.*` (gate-fire share, helped / hurt with sign test). Kind `p4` prints four decimals.")
    w("")
    w("**`RV3.`** (results/r3/rv3_summary.json, merged by `code/tools/r3_merge.py`; grammar `code/tools/rv3_grammar.json` via `code/tools/rv3_keys.py`, described in `internal note R3EXP`; scope posthoc-secondary, SAP section 26; new runs: the T 0.7 replication (seeds 101, 202, 303) of every decomposition arm and the same-examples control, so `_meta.api_calls` is the number of billed calls, not 0; never `\\res`, never in the P* family): `RV3.rep.*` (E-S: the replication of three runs at T 0.7, kept apart from the headline run at T 0 and never averaged with it; arms, paired contrasts averaged over the three T 0.7 runs with cluster, item-level, between-run and widest intervals, the headline run's value and reading beside them with an agrees flag, P1' margin, W1, calls ratio, stack versus loop, root-regeneration audit), `RV3.samex.*` (E-C, same layout), `RV3.ret.*`, `RV3.route.*`, `RV3.p5.*` (E-R), `RV3.q360.*` (E-Q), and the author addendum `RV3.inc.*` (cost per additional correct answer, `undefined_reason`), `RV3.mde.*`, `RV3.mult.*`, `RV3.critic.*`, `RV3.stack.pooled.*`, `RV3.lat.*`, plus `RV3.cost.*` (cost of the round-3 runs). Kinds `x3s` (signed, three decimals) and `pp2` (signed, two decimals) keep a bound such as -0.049 from printing as 0.0.")
    w("**`RV4.`** (results/r4/rv4_summary.json, merged by `code/tools/r4_merge.py` from `code/tools/r4_stats.py`; grammar `code/tools/rv4_grammar.json` via `code/tools/rv4_keys.py`, described in `internal note R4A`; scope posthoc-secondary, SAP section 27; no model call, `_meta.api_calls` = 0; never `\\res`, never in the P* family): `RV4.wide.*` (one widest-interval definition for the locked run and the replication: hull of the clustered, item-level, cluster-robust t and sign-flip intervals; readings and the agrees flag per contrast, counts), `RV4.regen.*` (root regeneration between gated and fixed arms and the net questions it carries), `RV4.ccost.*` (cost per extra correct answer of the Critic increment, fixed loop over gated loop, per backbone), `RV4.rmargin.*` (P1' margin ratio and share kept in the replication, locked run beside), `RV4.mde4.*` (minimum detectable effects at question and database level), `RV4.p5u.*` (P5' with one unit per distinct question), `RV4.silent.*` (silent-error rate and the share of wrong answers the gate can reach), `RV4.mult.*` (breakdown of the post hoc contrast count).")
    w("")
    w("**`RV4T0.`** (results/r4_t0/rv4t0_summary.json, written by `code/tools/r4_t0.py`; grammar `code/tools/rv4t0_grammar.json` via `code/tools/rv4t0_keys.py`, described in `internal note R4T0`; scope posthoc-secondary, SAP section 27; new runs, so `_meta.api_calls` is the number of billed calls of the rerun; never `\\res`, never in the P* family): the temperature-0 non-determinism check, reduced to the reruns that finished (`RV4T0.nd.<b>.<arm>.*`: locked run s0 against one rerun r1, shares of changed final SQL and changed correctness, b and c counts, exact McNemar p, EX difference with cluster, item-level and widest interval; `RV4T0.nd.complete.*`; `RV4T0.pair.*`; `RV4T0.lock.*` the paper's key contrasts in the locked run; `RV4T0.cmp.*`; `RV4T0.req.*`; `RV4T0.cost.*`).")
    w("")
    w("**`EXT1.`** (ext1_summary.json; keys are the file's own): `EXT1.XB1|XB2|XB3.{delta_ex,n,n_dbs,p_adj,verdict,text,per_backbone.<b>,member.<b>.*}`, `.ci`, `.item_ci`, `.loo`; `EXT1.XB1_loop`, `EXT1.XB1_stack`; `EXT1.sens_no_world_1.*` (same cells without world_1); `EXT1.desc.<b>.<contrast>.*`; `EXT1.arm.<b>.<zs|fixed|gated|rootonly>.{ex_ts,ex_plain,calls_per_q,cost_k,...}` (`cost_k` = CNY per 1000 questions, derived); `EXT1.gate.<b>.*` and `EXT1.gate.pooled.{fired,n,share}` (derived sums).")
    w("")
    w("**`E10.`** (results/e10/agreement.json, discordant_summary.json): agreement `E10.agreement.{kappa,ac1,pct_agree,kappa_l5}`, `E10.{n_items,n_disagree,system.<s>.*,counts.<run>.<label>}`; revision effect `E10.revision.{n,helped,hurt,neutral}` (full_fixed, all dev items), `E10.revision.{fixed,gated}.*`, `E10.revision.discordant.*`, `E10.repair.*`, `E10.discordant.*`.")
    w("")
    w("## 5. Not exposed, not sourced")
    w("")
    for t in NOT_EXPOSED:
        w("* " + t)
    if ctx.na:
        w("* Null by design in the sources (skipped, %d fields): %s" % (len(ctx.na), "; ".join(sorted(ctx.na)[:6]) + (" ..." if len(ctx.na) > 6 else "")))
    w("")

    w("## 5b. Legacy-name aliases (names used by the supplement and drafts)")
    w("")
    w("The result files spell some cells differently from the grammar above. Aliases give the same text under the file's own spelling (each alias row in the .tsv says `alias of <key>`): `ROB.<source contrast>.{d0,d1,shift}` (`d0` = original, `d1` = corrected; source contrasts `H1.plus`, `H1.pooled3`, `H2.pooled_weak`, `H2.q27b`, ...), `DIAG.pooled.{detection,fp}` (= `DIAG.all.detect`, `DIAG.all.fpr`), `VAD.<v>.{delta_fpr,delta_detection}` (= `VAD.<v>.d_fpr`, `d_detection`), `VE.q<q as in the file>.<policy>.<METRIC>` (e.g. `VE.q0.85.A.CCR` = `VE.q085.A.ccr`). Each alias has the same suffixes as its target (`.ci`, `.ci_lo`, `.ci_hi`, `.fmt`, `.n`, `.k` where the target has them). New code should use the grammar names of section 4; %d alias keys are generated." % ctx.n_aliases)
    w("")
    w("## 5c. Numbers that two files report: which source each key uses")
    w("")
    w("The V0 (pipeline Critic) cells exist in both `critic_diag_summary.json` (DIAG; rate bootstrap, B = 5 000) and `critic_v2_summary.json` (VAD; the V0 row of the variant comparison, one resample shared by all variants so that paired differences are exact). The table compares the two spellings of the same quantity; 'Same point' says whether the estimates are identical and 'Same text' whether the typeset estimate and interval are. Rule: `DIAG.*` (and its alias `DIAG.pooled.*`) always read the diag summary, `VAD.V0.*` always the v2 summary; `tab_critic_diag` uses `VAD.*` for all variant rows (including V0) so that every row of that table shares one resample. Current values:")
    w("")
    w("| Quantity | `DIAG` key | DIAG text | `VAD` key | VAD text | Same point | Same text |")
    w("|---|---|---|---|---|---|---|")
    for dk, vk in (("DIAG.all.detect", "VAD.V0.detection"), ("DIAG.all.fpr", "VAD.V0.fpr"), ("DIAG.all.harm", "VAD.V0.harm"), ("DIAG.all.repair1", "VAD.V0.repair1"), ("DIAG.all.auroc_all", "VAD.V0.auroc")):
        if dk + ".fmt" in ctx.entries and vk + ".fmt" in ctx.entries:
            a_, b_ = value_text(ctx.entries[dk + ".fmt"], ctx.entries), value_text(ctx.entries[vk + ".fmt"], ctx.entries)
            same_pt = abs(float(ctx.entries[dk].raw) - float(ctx.entries[vk].raw)) < 1e-9
            w("| %s | `%s` | %s | `%s` | %s | %s | %s |" % (dk.split(".")[-1], dk, a_.replace("\\ensuremath{-}", "-"), vk, b_.replace("\\ensuremath{-}", "-"), "yes" if same_pt else "**no**", "yes" if a_ == b_ else "**no**"))
    w("")
    w("`VAD.V0.auroc` is the all-injected-versus-all-gold AUROC, i.e. `DIAG.all.auroc_all` (not the paired `DIAG.all.auroc`). Both files are computed from the same stored records; intervals differ where the bootstraps draw different database resamples (the pooled false-positive upper bound is the case that was reported).")
    w("")
    w("## 5d. Cost per 1 000 questions")
    w("")
    w("`EXT1.arm.<b>.<arm>.cost_k` = 1 000 x `cost_cny_per_q` (CNY per 1 000 questions, two decimals); one exists for every `cost_cny_per_q` in the EXT1 summary (%d keys). The other per-question CNY keys (`E1/E2/E11 ... cost_per_q`, `cost_per_correct`) are confirmatory keys of `results_macros.tex`, written by `make_macros.py`, which this task does not edit. Patch for its owner: in `FORMAT_RULES` add `(r\"\\.(cost_per_q)(\\.mean)?$\", \"cny4\")` as now, and in `derive()` add for every key ending `.cost_per_q.mean` (or `.cost_per_q`) a twin `<base>.cost_k` = 1000 x value formatted `cny2` (and `.cost_k.ci` from `cost_per_q.ci_lo/ci_hi` x 1000). Until then, tables of confirmatory cost cannot print CNY per 1 000 questions from a macro." % len([k for k in ctx.entries if k.endswith(".cost_k")]))
    w("")
    w("## 5e. Global key check")
    w("")
    w("`python3 code/tools/make_xmacros.py --check-paper paper` lists every `\\\\res`, `\\\\xres` and `\\\\tsres` key used in `paper/main.tex`, `sections/*.tex`, `tables/*.tex`, `supplement.tex` and `supp_tables/*.tex` (comments removed) that its macro file does not define (`results_macros.tex`, `results_macros_x.tex`, `results_macros_ts.tex`); exit code 6 if any is undefined. Run it after every writer batch.")
    w("")
    w("## 6. Preamble patch (the preamble has other uncommitted edits, so it is not touched here)")
    w("")
    w("Insert after the `\\tsres` block of `paper/preamble.tex` (both builds, typeset normally):")
    w("")
    w("```latex")
    w("% XMACROS: \\xres{KEY} = exploratory / robustness / external-validity numbers (internal note XMACROS), generated by code/tools/make_xmacros.py; loaded in BOTH builds.")
    w("\\IfFileExists{results_macros_x.tex}{\\input{results_macros_x}}{\\providecommand{\\xres}[1]{\\resfallback{#1}}}")
    w("```")
    w("")
    w("`\\resfallback` is defined above that line, so a missing key or a missing file prints bold `[KEY]` as `\\res` does. Table files use `\\xres` only, and `\\tabinput{tab_robust}` etc. include them.")
    w("")
    w("## 7. Reproduce and tests")
    w("")
    w("```bash")
    w("python3 code/tools/make_xmacros.py --doc internal note XMACROS      # strict; --allow-missing for drafts")
    w("python3 code/tools/make_xtables.py")
    w("cd code && python3 -m pytest -q tests/test_make_xmacros.py tests/test_make_xtables.py   # host python 3.6 runs them as scripts; a worker machine: copy tools, tests and the needed results files to a local dir, then pytest")
    w("```")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    return tsv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=os.path.join(REPO, "results"))
    ap.add_argument("--out", default=os.path.join(REPO, "paper", "results_macros_x.tex"))
    ap.add_argument("--report", default=None)
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--check-paper", nargs="?", const=os.path.join(REPO, "paper"), default=None, metavar="PAPER_DIR",
                    help="after generating, list every \\res/\\xres/\\tsres key used by main.tex, sections, tables, supplement.tex, supp_tables that is not defined (exit 6 on a gap)")
    ap.add_argument("--doc", default=None, help="also write the key catalogue (markdown) to this path")
    a = ap.parse_args(argv)
    ctx = build_all(a.results, a.allow_missing)
    report = a.report or os.path.join(a.results, "_reports", "missing_xkeys.txt")
    if ctx.scope_errors:
        sys.stderr.write("make_xmacros: WRONG SCOPE\n  " + "\n  ".join(ctx.scope_errors) + "\n")
        return 4
    errs = sorted(set(ctx.errors))
    if errs:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(report)), exist_ok=True)
            with open(report, "w") as f:
                f.write("# make_xmacros: sources that could not be read (%s)\n" % time.strftime("%Y-%m-%d %H:%M"))
                f.write("\n".join(errs) + "\n")
        except OSError:
            pass
        sys.stderr.write("make_xmacros: %d missing sources (first 20):\n  %s\n" % (len(errs), "\n  ".join(errs[:20])))
        if not a.allow_missing:
            sys.stderr.write("make_xmacros: FAILING (strict mode); use --allow-missing for drafts\n")
            return 3
    write_macros(ctx, a.out)
    by = collections.Counter(k.split(".")[0] for k in ctx.entries)
    print("make_xmacros: %d keys -> %s  (%s)" % (len(ctx.entries), a.out, ", ".join("%s %d" % kv for kv in sorted(by.items()))))
    if "pf" in ctx.absent:
        print("make_xmacros: PF not sourced (results/pf/pf_summary.json absent)")
    if a.check_paper:
        gaps, used_n = check_paper(a.check_paper, ctx.entries.keys())
        total = sum(len(v) for v in gaps.values())
        for m in gaps:
            print("make_xmacros: check-paper \\%s: %d distinct keys used, %d undefined" % (m, len(used_n[m]), len(gaps[m])))
            for k, fs in sorted(gaps[m].items()):
                print("    \\%s{%s}  in %s" % (m, k, ", ".join(sorted(fs))))
        if total:
            rc_check = 6
        else:
            rc_check = 0
    else:
        rc_check = 0
    if a.doc:
        write_doc(ctx, a.doc)
        print("make_xmacros: catalogue -> %s (+ .tsv)" % a.doc)
    return rc_check


if __name__ == "__main__":
    sys.exit(main())
