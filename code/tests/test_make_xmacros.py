"""Tests of code/tools/make_xmacros.py (\\xres macros for exploratory / robustness / external-validity numbers).
Plain asserts; runs with pytest or `python tests/test_make_xmacros.py`. Tests that need the real result files use $XMACROS_RESULTS (default <repo>/results) and skip when absent."""
import ast
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOLS)
import make_xmacros as X  # noqa: E402

REPO = X.REPO
RESULTS = os.environ.get("XMACROS_RESULTS", os.path.join(REPO, "results"))


def have_real():
    return all(os.path.exists(os.path.join(RESULTS, rel)) for rel, _ in [v for v in X.SOURCES.values() if "pf/" not in v[0]])


def skip_unless_real():
    if not have_real():
        try:
            import pytest
            pytest.skip("real result files not available (set XMACROS_RESULTS)")
        except ImportError:
            return False
    return True


_CTX = {}


def real_ctx():
    if "c" not in _CTX:
        _CTX["c"] = X.build_all(RESULTS)
    return _CTX["c"]


# ------------------------------------------------------------------------------------------ formatting
def test_formats():
    assert X.fnum(-0.55, "pp") == "-0.6" or X.fnum(-0.55, "pp") == "-0.5"      # float rounding of .55 either way, never a different length
    assert X.fnum(2.364, "pp") == "+2.4"
    assert X.fnum(2.364, "pp2") == "+2.36"
    assert X.fnum(0.02364, "fpp") == "+2.4"
    assert X.fnum(0.8535, "pct") == "85.4" or X.fnum(0.8535, "pct") == "85.3"
    assert X.fnum(72.22, "pctv") == "72.2"
    assert X.fnum(0.0702, "x3s") == "+0.070"
    assert X.fnum(0.8571, "x3") == "0.857"
    assert X.fnum(0.5804, "x2") == "0.58"
    assert X.fnum(0.9865, "x2s") == "+0.99"
    assert X.fnum(237.5, "int") in ("238", "237")
    assert X.fnum(0.0004, "p") == "<0.001" and X.fnum(0.04839, "p") == "0.048"
    assert X.fnum(4542.4, "tok") == "4,542"
    assert X.fnum(0.0041426, "cny4") == "0.0041"


def test_no_negative_or_positive_zero():
    for kind in ("pp", "pp2", "fpp", "fpp2", "x3s", "x2s"):
        assert X.fnum(0.0, kind).lstrip("+-") == X.fnum(0.0, kind)
        assert not X.fnum(-1e-9, kind).startswith("-"), kind
        assert not X.fnum(1e-9, kind).startswith("+"), kind


def test_render_minus_and_interval():
    assert X.render_raw(-1.94, "pp") == "\\ensuremath{-}1.9"
    assert X.render_raw((-3.22, -0.79), "pp") == "[\\ensuremath{-}3.2, \\ensuremath{-}0.8]"
    assert X.render_raw((0.5, 0.7), "x3") == "[0.500, 0.700]"
    assert X.render_raw(True, "str") == "yes" and X.render_raw(False, "str") == "no"
    assert X.render_raw("no_help", "str") == "no\\_help"


def test_same_rules_as_make_macros():
    spec = importlib.util.spec_from_file_location("make_macros", os.path.join(TOOLS, "make_macros.py"))
    try:
        M = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(M)
    except Exception:
        return
    for v in (-3.614, -0.26, 1.234, 2.93, 12.5):
        assert X.fnum(v, "pp") == M._num(v, "pp"), v
        assert X.fnum(v / 100.0, "fpp") == M._num(v / 100.0, "ppf"), v
    for v in (0.8535, 0.2727, 0.3162):
        assert X.fnum(v, "pct") == M._num(v, "pct")
    for v in (0.0702, -0.036):
        assert X.fnum(v, "x3s") == M._num(v, "x3s")
    for v in (0.5804, 0.64, 3.0768):
        assert X.fnum(v, "x2") == M._num(v, "x2")
    for v in (0.8571, 0.735):
        assert X.fnum(v, "x3") == M._num(v, "x3")
    for v in (0.0004, 0.0484, 0.096):
        assert X.fnum(v, "p") == M._num(v, "p")


# ------------------------------------------------------------------------------------------ strict mode
def test_strict_fails_without_sources_and_writes_nothing():
    d = tempfile.mkdtemp()
    try:
        out = os.path.join(d, "x.tex")
        rc = X.main(["--results", os.path.join(d, "empty"), "--out", out, "--report", os.path.join(d, "r.txt")])
        assert rc == 3
        assert not os.path.exists(out)
        assert "source file missing" in open(os.path.join(d, "r.txt")).read()
    finally:
        shutil.rmtree(d)


def test_allow_missing_writes_and_reports():
    d = tempfile.mkdtemp()
    try:
        out = os.path.join(d, "x.tex")
        rc = X.main(["--results", os.path.join(d, "empty"), "--out", out, "--report", os.path.join(d, "r.txt"), "--allow-missing"])
        assert rc == 0 and os.path.exists(out)
        assert "xres" in open(out).read()
        assert "results/robust/rescore.json" in open(os.path.join(d, "r.txt")).read()
    finally:
        shutil.rmtree(d)


def test_wrong_scope_is_fatal():
    d = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(d, "robust"))
        json.dump({"_meta": {"scope": "confirmatory"}}, open(os.path.join(d, "robust", "rescore.json"), "w"))
        rc = X.main(["--results", d, "--out", os.path.join(d, "x.tex"), "--report", os.path.join(d, "r.txt"), "--allow-missing"])
        assert rc == 4
    finally:
        shutil.rmtree(d)


def test_missing_field_in_one_file_is_reported():
    if not skip_unless_real() and not have_real():
        return
    d = tempfile.mkdtemp()
    try:
        shutil.copytree(RESULTS, os.path.join(d, "r"), ignore=shutil.ignore_patterns("raw-ignored", "*.gz", "*.csv", "chunks", "pilot", "smoke", "ts", "r1", "p2", "ref", "e4_min", "pf", "tuning*", "records*", "usage*"))
        p = os.path.join(d, "r", "diag", "critic_diag_summary.json")
        j = json.load(open(p))
        del j["pooled"]["all"]["detect_triggered"]
        json.dump(j, open(p, "w"))
        rc = X.main(["--results", os.path.join(d, "r"), "--out", os.path.join(d, "x.tex"), "--report", os.path.join(d, "rep.txt")])
        assert rc == 3
        assert "pooled/all/detect_triggered" in open(os.path.join(d, "rep.txt")).read()
        rc = X.main(["--results", os.path.join(d, "r"), "--out", os.path.join(d, "x.tex"), "--report", os.path.join(d, "rep.txt"), "--allow-missing"])
        assert rc == 0
        keys = X.parse_macros(os.path.join(d, "x.tex"))
        assert "DIAG.all.detect" not in keys and "DIAG.all.fpr" in keys
    finally:
        shutil.rmtree(d)


# ------------------------------------------------------------------------------------------ key coverage and values (real files)
ABSTRACT_KEYS = ["ROB.n_same", "ROB.H1.pooled.corrected", "E10.revision.helped", "E10.revision.hurt"]


def test_claim_and_abstract_keys_exist():
    if not skip_unless_real() and not have_real():
        return
    c = real_ctx()
    missing = [k for _, _, ks in X.CLAIMS for k in ks if k not in c.entries]
    assert not missing, missing
    for k in ABSTRACT_KEYS:
        assert k in c.entries, k


def test_every_interval_has_all_forms():
    if not have_real():
        return
    c = real_ctx()
    for k in list(c.entries):
        if k.endswith(".ci") and not k.endswith(".d2"):
            base = k[:-3]
            assert base + ".ci_lo" in c.entries and base + ".ci_hi" in c.entries, k
            if base in c.entries and not isinstance(c.entries[base].raw, (str, bool)):
                assert base + ".fmt" in c.entries, k
    assert "ROB.H1.pooled.corrected.d2" in c.entries and "ROB.H1.pooled.corrected.ci.d2" in c.entries


def test_every_category_and_variant_covered():
    if not have_real():
        return
    c = real_ctx()
    d = json.load(open(os.path.join(RESULTS, "diag", "critic_diag_summary.json")))
    for cat in d["per_category"]:
        for m in ("detect", "fpr", "auroc", "harm", "n_inj"):
            assert "DIAG.%s.%s" % (cat, m) in c.entries, (cat, m)
    v2 = json.load(open(os.path.join(RESULTS, "diag", "critic_v2_summary.json")))
    for v in v2["variants"]:
        if v != "V0_semantic_only":
            assert "VAD.%s.fpr" % v in c.entries
    ve = json.load(open(os.path.join(RESULTS, "diag", "critic_ve_summary.json")))
    for q in ve["by_q"]:
        ql = "q" + q.replace(".", "")
        for pol in ve["by_q"][q]["point"]:
            assert "VE.%s.%s.j" % (ql, pol) in c.entries, (q, pol)
    for arm in ve["arbitration_quality"]:
        assert "VE.%s.firm_correct" % arm in c.entries


def test_ext1_every_scalar_key_present():
    if not have_real():
        return
    c = real_ctx()
    d = json.load(open(os.path.join(RESULTS, "ext1", "ext1_summary.json")))
    for k, v in d.items():
        if k.startswith("_") or isinstance(v, dict) or v is None:
            continue
        if isinstance(v, list) and not k.endswith("_wilson"):
            continue
        kk = k[:-len("_wilson")] + ".wilson" if k.endswith("_wilson") else k
        assert kk in c.entries, k


def _text(c, key):
    return X.value_text(c.entries[key], c.entries)


def test_values_equal_independent_computation():
    if not have_real():
        return
    c = real_ctx()
    r = json.load(open(os.path.join(RESULTS, "robust", "rescore.json")))
    a = r["variants"]["G1"]["sets"]["S_same"]["analysis"]["contrasts"]["H1.pooled3"]
    assert _text(c, "ROB.H1.pooled.corrected") == "\\ensuremath{-}%s" % ("%.1f" % abs(a["d1"]))
    assert _text(c, "ROB.H1.pooled.corrected.d2") == "\\ensuremath{-}%s" % ("%.2f" % abs(a["d1"]))
    assert _text(c, "ROB.H1.pooled.corrected.ci") == "[\\ensuremath{-}%.1f, \\ensuremath{-}%.1f]" % (abs(a["ci1"][0]), abs(a["ci1"][1]))
    assert c.entries["ROB.n_same"].raw == r["variants"]["G1"]["sets"]["S_same"]["analysis"]["common_n"]
    dg = json.load(open(os.path.join(RESULTS, "diag", "critic_diag_summary.json")))["pooled"]["all"]["detect_triggered"]
    assert _text(c, "DIAG.all.detect") == "%.1f" % (100 * dg["rate"])
    assert _text(c, "DIAG.all.detect.n") == str(dg["n"])
    ve = json.load(open(os.path.join(RESULTS, "diag", "critic_ve_summary.json")))
    assert _text(c, "VE.q085.A.ccr") == "%.1f" % (100 * ve["by_q"]["0.85"]["point"]["A"]["CCR"])
    ex = json.load(open(os.path.join(RESULTS, "ext1", "ext1_summary.json")))
    assert c.entries["EXT1.gate.pooled.fired"].raw == sum(ex["EXT1.gate.%s.gate_fired" % b] for b in ("plus", "flash", "27b"))


def test_derived_keys():
    if not have_real():
        return
    c = real_ctx()
    gk = c.entries["DIAG.all.gold_keep"].raw
    assert abs(c.entries["DIAG.all.harm"].raw - (1.0 - gk)) < 1e-12
    lo, hi = c.entries["DIAG.all.gold_keep.ci"].raw
    assert c.entries["DIAG.all.harm.ci"].raw == (1.0 - hi, 1.0 - lo)
    be = c.entries["VE.breakeven_q.A"].raw
    assert 0.70 < be < 0.85                                  # between the computed q values by construction


def test_e10_xres_agrees_with_confirmatory_file():
    p = os.path.join(RESULTS, "E10.json")
    if not have_real() or not os.path.exists(p):
        return
    c = real_ctx()
    e = json.load(open(p))
    for k in ("n", "helped", "hurt", "neutral"):
        assert c.entries["E10.revision.%s" % k].raw == e["E10.revision.%s" % k], k
    assert abs(c.entries["E10.agreement.kappa"].raw - e["E10.kappa.mean"]) < 1e-3


def test_macro_file_roundtrip():
    if not have_real():
        return
    d = tempfile.mkdtemp()
    try:
        out = os.path.join(d, "x.tex")
        assert X.main(["--results", RESULTS, "--out", out, "--report", os.path.join(d, "r.txt")]) == 0
        keys = X.parse_macros(out)
        c = real_ctx()
        assert len(keys) == len(c.entries)
        txt = open(out).read()
        assert txt.count("{") == txt.count("}")
        assert "\\def\\xres#1" in txt
        for k in ("ROB.H1.pooled.corrected", "DIAG.all.fpr.ci"):
            assert keys[k] == _text(c, k)
    finally:
        shutil.rmtree(d)


def test_catalogue_has_no_missing_claim_key():
    if not have_real():
        return
    d = tempfile.mkdtemp()
    try:
        tsv = X.write_doc(real_ctx(), os.path.join(d, "XM.md"))
        md = open(os.path.join(d, "XM.md")).read()
        assert "MISSING" not in md
        rows = open(tsv).read().splitlines()
        assert len(rows) == len(real_ctx().entries) + 1
    finally:
        shutil.rmtree(d)


# ------------------------------------------------------------------------------------------ follow-up: aliases, cost per 1 000, global check
REQUESTED = {"DIAG.pooled.detection.fmt": "DIAG.all.detect.fmt", "DIAG.pooled.fp.fmt": "DIAG.all.fpr.fmt", "ROB.H1.plus.d1.fmt": "ROB.H1.plus.corrected.fmt",
             "ROB.H1.pooled3.d0.fmt": "ROB.H1.pooled.original.fmt", "ROB.H1.pooled3.d1.fmt": "ROB.H1.pooled.corrected.fmt",
             "ROB.H2.pooled_weak.d0.fmt": "ROB.H2.pooled.original.fmt", "ROB.H2.pooled_weak.d1.fmt": "ROB.H2.pooled.corrected.fmt",
             "VAD.VC.delta_fpr.fmt": "VAD.VC.d_fpr.fmt", "VE.q0.85.A.CCR.fmt": "VE.q085.A.ccr.fmt", "VE.q0.85.A.WCR.fmt": "VE.q085.A.wcr.fmt"}


def test_requested_alias_keys_exist_and_equal_their_targets():
    if not have_real():
        return
    c = real_ctx()
    for new, old in REQUESTED.items():
        assert new in c.entries, new
        assert _text(c, new) == _text(c, old), (new, old)
    assert _text(c, "ROB.H1.pooled3.d1.fmt") == "\\ensuremath{-}1.9 [\\ensuremath{-}3.2, \\ensuremath{-}0.8]"


def test_alias_families_follow_the_file_spelling():
    if not have_real():
        return
    c = real_ctx()
    ve = json.load(open(os.path.join(RESULTS, "diag", "critic_ve_summary.json")))
    for q in ve["by_q"]:
        for pol, mm in ve["by_q"][q]["point"].items():
            for m in mm:
                assert "VE.q%s.%s.%s" % (q, pol, m) in c.entries, (q, pol, m)
    assert "ROB.H2.q27b.d1" in c.entries and "ROB.H2.plus_control.d0.ci" in c.entries
    assert c.n_aliases > 0
    for k, e in c.entries.items():
        m = re.match(r"alias of (\S+): ", e.meaning)
        if m:
            assert m.group(1) in c.entries, (k, m.group(1))


def test_followup_ext1_keys_exist():
    if not have_real():
        return
    c = real_ctx()
    for k in ("EXT1.XB1.delta_ex.fmt", "EXT1.XB2.delta_ex.fmt", "EXT1.XB1.verdict", "EXT1.XB3.p_adj", "EXT1.XB3.delta_ex.fmt"):
        assert k in c.entries, k


def test_cost_per_1000_exists_for_every_per_question_cost():
    if not have_real():
        return
    c = real_ctx()
    per_q = [k for k in c.entries if k.endswith(".cost_cny_per_q")]
    assert len(per_q) == 12
    for k in per_q:
        kk = k[:-len(".cost_cny_per_q")] + ".cost_k"
        assert kk in c.entries, kk
        assert abs(c.entries[kk].raw - 1000.0 * c.entries[k].raw) < 1e-9


def test_check_paper_finds_undefined_keys_and_ignores_comments():
    d = tempfile.mkdtemp()
    try:
        for sub in ("sections", "tables", "supp_tables"):
            os.makedirs(os.path.join(d, sub))
        open(os.path.join(d, "results_macros.tex"), "w").write("\\expandafter\\gdef\\csname resv@A.b\\endcsname{1}\n")
        open(os.path.join(d, "results_macros_ts.tex"), "w").write("\\expandafter\\gdef\\csname tsv@TS.a\\endcsname{1}\n")
        open(os.path.join(d, "main.tex"), "w").write("\\res{A.b} \\res{A.missing} % \\res{in.comment}\n\\xres{X.ok} \\tsres{TS.a}\n")
        open(os.path.join(d, "supplement.tex"), "w").write("\\xres{X.gap} \\xres{X.ok} \\tsres{TS.gap} 50\\% \\res{#1}\n")
        open(os.path.join(d, "supp_tables", "t.tex"), "w").write("\\xres{X.gap2}\n")
        gaps, used = X.check_paper(d, ["X.ok"])
        assert sorted(gaps["res"]) == ["A.missing"]
        assert sorted(gaps["xres"]) == ["X.gap", "X.gap2"]
        assert sorted(gaps["tsres"]) == ["TS.gap"]
        assert "supp_tables/t.tex" in list(gaps["xres"]["X.gap2"])[0].replace(os.sep, "/")
        assert len(used["xres"]) == 3
        rc = X.main(["--results", os.path.join(d, "none"), "--out", os.path.join(d, "x.tex"), "--report", os.path.join(d, "r.txt"), "--allow-missing", "--check-paper", d])
        assert rc == 6
    finally:
        shutil.rmtree(d)


def test_current_paper_keys_are_all_defined_when_paper_is_present():
    """informational guard for the repository checkout: skipped when the paper tree is absent; run `make_xmacros.py --check-paper paper` for the listing"""
    paper = os.path.join(REPO, "paper")
    if not have_real() or not os.path.exists(os.path.join(paper, "supplement.tex")) or not os.path.exists(os.path.join(paper, "results_macros.tex")):
        return
    gaps, _ = X.check_paper(paper, real_ctx().entries.keys())
    assert not gaps["xres"], sorted(gaps["xres"])[:10]


# ------------------------------------------------------------------------------------------ PF
def pf_present():
    return os.path.exists(os.path.join(RESULTS, "pf", "pf_summary.json"))


def _csv(name):
    import csv
    with open(os.path.join(RESULTS, "pf", name), newline="") as f:
        return list(csv.DictReader(f))


def test_pf_key_families_exist():
    if not have_real() or not pf_present():
        return
    c = real_ctx()
    arms = ("zs", "full_fixed", "full_gated", "route_ca", "pf_n0", "pf_n250_q100", "pf_n1000_q100", "pf_n1000_q085")
    for a in arms:
        for s in ("ex", "ex.ci", "ex.fmt", "ex.d2", "ex.n", "ex.k", "calls_per_q", "cost_per_q", "cost_k", "helped", "hurt", "net.fmt"):
            assert "PF.arm.%s.%s" % (a, s) in c.entries, (a, s)
    for a in ("full_fixed", "pf_n0", "pf_n250_q100", "pf_n1000_q100", "pf_n1000_q085"):
        for s in ("detection", "detection.n", "detection.k", "false_flag", "flag_rate", "flag_rate.k"):
            assert "PF.arm.%s.%s" % (a, s) in c.entries, (a, s)
    for a in ("pf_n250_q100", "pf_n1000_q100", "pf_n1000_q085"):
        for s in ("flag_rate_rejected", "flag_rate_rejected.n", "flag_rate_rejected.k", "flag_rate_no_rejected", "rejected_share", "fb", "q"):
            assert "PF.arm.%s.%s" % (a, s) in c.entries, (a, s)
    for q, ns in (("q100", ("n0", "n250", "n1000")), ("q085", ("n0", "n1000"))):
        for n in ns:
            for s in ("ex.fmt", "ex.d2", "ex.n", "calls_per_q", "cost_k", "detection", "helped", "hurt", "fb"):
                assert "PF.curve.%s.%s.%s" % (q, n, s) in c.entries, (q, n, s)
    for ct in ("PF1", "PF2", "PF2_n0", "PF2_q100", "PF3a", "PF3a_q100", "PF4_250_vs_0", "PF4_1000_vs_250", "PF4_noise_cost", "ref.fixed_vs_gated", "ref.pf_n0_vs_zs"):
        for s in ("delta", "delta.ci", "delta.ci_lo", "delta.ci_hi", "delta.fmt", "delta.d2", "delta.n", "delta.loo", "a_only", "b_only", "reading"):
            assert "PF.%s.%s" % (ct, s) in c.entries, (ct, s)
    for ct in ("PF3", "PF3_q100"):
        for s in ("calls_ratio.fmt", "cost_ratio.fmt", "calls_ratio.ci", "cost_ratio.ci", "verdict", "accuracy_reading"):
            assert "PF.%s.%s" % (ct, s) in c.entries, (ct, s)
    assert "PF.calls_vs_fixed.pf_n0.fmt" in c.entries and "PF.n_items" in c.entries and "PF.PF4.monotone" in c.entries
    assert all(e.label == X.LABELS["PF"] for k, e in c.entries.items() if k.startswith("PF."))


def test_pf_values_equal_the_csv_files():
    if not have_real() or not pf_present():
        return
    c = real_ctx()
    for r in _csv("pf_arms.csv"):
        a = r["label"]
        assert c.entries["PF.arm.%s.ex" % a].text() == X.fnum(r["ex_pp"], "pctv"), a
        assert _text(c, "PF.arm.%s.ex.d2" % a) == X.fnum(r["ex_pp"], "pctv2"), a
        assert c.entries["PF.arm.%s.calls_per_q" % a].text() == X.fnum(r["calls_per_q"], "x2")
        assert c.entries["PF.arm.%s.cost_k" % a].text() == X.fnum(1000.0 * float(r["cost_cny_per_q"]), "cny2")
        assert c.entries["PF.arm.%s.helped" % a].text() == "%d" % int(r["helped"])
        if r["critic_detection_pp"]:
            assert c.entries["PF.arm.%s.detection" % a].text() == X.fnum(r["critic_detection_pp"], "pctv")
            assert c.entries["PF.arm.%s.detection.n" % a].text() == "%d" % int(r["n_wrong_roots"])
    names = {"PF1": "PF1", "PF2": "PF2", "PF2_n0": "PF2_n0", "PF2_q100": "PF2_q100", "PF3a": "PF3a", "PF3a_q100": "PF3a_q100"}
    for r in _csv("pf_contrasts.csv"):
        nm = r["name"]
        key = "PF." + ("ref." + nm[len("ref_"):] if nm.startswith("ref_") else nm)
        assert c.entries[key + ".delta"].text() == X._minus(X.fnum(r["d_pp"], "pp")), nm
        assert _text(c, key + ".delta.ci.d2") == "[%s, %s]" % tuple(X._minus(X.fnum(r[f], "pp2")) for f in ("lo_pp", "hi_pp")), nm
        assert c.entries[key + ".a_only"].text() == "%d" % int(r["a_only"])
    for r in _csv("pf_curve.csv"):
        n = r["n"]
        assert c.entries["PF.curve.q100.n%s.ex" % n].text() == X.fnum(r["ex_pp"], "pctv"), n
        assert _text(c, "PF.curve.q100.n%s.ex.ci_lo.d2" % n) == X.fnum(r["lo"], "pctv2")


def test_pf_cost_k_counts_and_curve_copies_are_consistent():
    if not have_real() or not pf_present():
        return
    c = real_ctx()
    per_q = [k for k in c.entries if k.startswith("PF.arm.") and k.endswith(".cost_per_q")]
    assert len(per_q) == 8
    for k in per_q:
        kk = k[:-len("cost_per_q")] + "cost_k"
        assert abs(c.entries[kk].raw - 1000.0 * c.entries[k].raw) < 1e-9, kk
    for a in ("full_fixed", "pf_n0", "pf_n250_q100", "pf_n1000_q100", "pf_n1000_q085"):
        for m in ("detection", "false_flag", "flag_rate"):
            r, n, k = (c.entries["PF.arm.%s.%s%s" % (a, m, s)].raw for s in ("", ".n", ".k"))
            assert abs(r * n / 100.0 - k) < 1e-6, (a, m)
        assert c.entries["PF.arm.%s.flag_rate.k" % a].raw == c.entries["PF.arm.%s.revisions" % a].raw
    for q, n, arm in (("q100", "n250", "pf_n250_q100"), ("q085", "n1000", "pf_n1000_q085"), ("q085", "n0", "pf_n0")):
        for m in ("cost_k", "detection", "calls_per_q"):
            assert _text(c, "PF.curve.%s.%s.%s" % (q, n, m)) == _text(c, "PF.arm.%s.%s" % (arm, m))
    assert _text(c, "PF.curve.q100.n250.ex.fmt") == _text(c, "PF.arm.pf_n250_q100.ex.fmt")
    assert c.entries["PF.arm.pf_n1000_q085.q"].text() == "0.85" and c.entries["PF.arm.pf_n1000_q085.fb"].text() == "1000"


def test_pf_absent_is_silent_and_wrong_scope_is_refused():
    if not have_real() or not pf_present():
        return
    d = tempfile.mkdtemp()
    try:
        r = os.path.join(d, "res")
        shutil.copytree(RESULTS, r, ignore=shutil.ignore_patterns("pf", "raw-ignored", "tuning"))
        c = X.build_all(r)
        assert not [k for k in c.entries if k.startswith("PF.")] and not c.errors
        os.makedirs(os.path.join(r, "pf"))
        d0 = json.load(open(os.path.join(RESULTS, "pf", "pf_summary.json")))
        d0["_meta"]["scope"] = "confirmatory"
        json.dump(d0, open(os.path.join(r, "pf", "pf_summary.json"), "w"))
        c = X.build_all(r)
        assert c.scope_errors
        assert X.main(["--results", r, "--out", os.path.join(d, "x.tex"), "--report", os.path.join(d, "rep.txt")]) == 4
    finally:
        shutil.rmtree(d)


def test_followup_ts_key_is_not_an_xres_key():
    if not have_real():
        return
    assert not [k for k in real_ctx().entries if k.startswith("TS.")]


# ------------------------------------------------------------------------------------------ no hand-typed numbers
ALLOWED_NUMBERS = {0, 1, 2, 3, -1, 1.0, 1000.0, 100.0, 0.0}        # small ints are tuple indices and slice lengths; floats only unit conversions


def test_no_typed_numbers_in_builders():
    src = open(os.path.join(TOOLS, "make_xmacros.py")).read()
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and (node.name.startswith("build_") or node.name.startswith("rob_")):
            for n in ast.walk(node):
                if isinstance(n, ast.Num) and not isinstance(n.n, bool) and n.n not in ALLOWED_NUMBERS:
                    bad.append((node.name, n.n))
    assert not bad, bad


def test_no_result_like_numbers_in_builder_strings():
    """labels and meanings inside the builders may name SAP sections (16-22.x) but must not contain decimal numbers that look like results"""
    src = open(os.path.join(TOOLS, "make_xmacros.py")).read()
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and (node.name.startswith("build_") or node.name.startswith("rob_")):
            for n in ast.walk(node):
                if isinstance(n, ast.Str):
                    for m in re.finditer(r"(?<![\w.%])(\d+\.\d+)(?![\w])", n.s):
                        # 1.5 pp is the pre-specified non-inferiority margin (SAP design constant, not a result)
                        if m.group(1) == "1.5" and "1.5 pp" in n.s:
                            continue
                        if not re.fullmatch(r"(1[6-9]|2[0-2])\.\d+", m.group(1)):
                            bad.append((n.s[:50], m.group(1)))
    assert not bad, bad


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e)[:300])
    sys.exit(1 if fails else 0)
