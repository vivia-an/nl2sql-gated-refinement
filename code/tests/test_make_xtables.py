"""Tests of code/tools/make_xtables.py (tab_robust, tab_external, tab_critic_diag).
Plain asserts; runs with pytest or `python tests/test_make_xtables.py`. Needs the real result files ($XMACROS_RESULTS, default <repo>/results); skips otherwise."""
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOLS)
import make_xmacros as X  # noqa: E402
import make_xtables as T  # noqa: E402

RESULTS = os.environ.get("XMACROS_RESULTS", os.path.join(X.REPO, "results"))
BASE = ("tab_robust.tex", "tab_external.tex", "tab_critic_diag.tex")


def pf_present():
    return os.path.exists(os.path.join(RESULTS, "pf", "pf_summary.json"))


NAMES = BASE + (("tab_pf.tex",) if pf_present() else ())


def have_real():
    return all(os.path.exists(os.path.join(RESULTS, rel)) for rel, _ in [v for v in X.SOURCES.values() if "pf/" not in v[0]])


_CACHE = {}


def built():
    if "d" not in _CACHE:
        d = tempfile.mkdtemp()
        rc = T.main(["--results", RESULTS, "--out", d, "--paper", os.path.join(d, "nopaper")])
        assert rc == 0
        _CACHE["d"] = d
        _CACHE["ctx"] = X.build_all(RESULTS)
    return _CACHE["d"], _CACHE["ctx"]


def body(fn):
    d, _ = built()
    return "\n".join(l for l in open(os.path.join(d, fn)).read().splitlines() if not l.startswith("%"))


def test_tables_written_and_every_xres_key_exists():
    if not have_real():
        return
    d, ctx = built()
    for fn in NAMES:
        assert os.path.exists(os.path.join(d, fn)), fn
        keys, _ = T.used_keys(open(os.path.join(d, fn)).read())
        assert keys
        miss = [k for k in keys if k not in ctx.entries]
        assert not miss, (fn, miss[:5])


def test_strict_refuses_without_sources():
    d = tempfile.mkdtemp()
    try:
        rc = T.main(["--results", os.path.join(d, "empty"), "--out", os.path.join(d, "t"), "--paper", d])
        assert rc == 3
        assert not os.path.exists(os.path.join(d, "t", "tab_robust.tex"))
    finally:
        shutil.rmtree(d)


def test_no_hand_typed_numbers_in_cells():
    if not have_real():
        return
    allowed = [r"95\\%", r"1\\,000", r"SAP (Amendment )?\d+(\.\d+)?", r"Amendment \d+", r"Section \d+", r"(?i:qwen)\d\.\d(-\w+)?", r"\bXB\d\b", r"\bP1\$'\$", r"\bW1\b", r"\bV[0-9A-D]\b", r"\bA0\b",
               r"world\\_1", r"\$q\$", r"\bq\b", r"\\label\{[^}]*\}", r"tab:\w+", r"results/[\w/\\_.]+", r"ext1\\_summary", r"\\texttt\{[^}]*\}", r"\bM\d\b", r"\bC[1-4]\b", r"\bH[1-5]\b", r"60\\%", r"contains 0", r"\b27[bB]\b", r"\\vspace\{[\d.]+pt\}"]       # model names in display form (Qwen3.7-Plus) or key form (qwen3.7-plus); 60% = the pre-stated coverage level of the routing metric; 27[bB] = the model name qwen3.5-27b (key form or display form "27B"), not a result
    for fn in NAMES:
        txt = body(fn)
        txt = re.sub(r"\\x?res\{[^{}]*\}", "", txt)
        txt = re.sub(r"\\multicolumn\{\d+\}", "", txt)
        txt = re.sub(r"\\begin\{tabularx\}\{\\linewidth\}\{.*\}\s*$", "", txt, flags=re.M)      # column spec (relative widths)
        for pat in allowed:
            txt = re.sub(pat, "", txt)
        digits = re.findall(r"(?<![\\\w])\d+(?:\.\d+)?", txt)
        assert not digits, (fn, digits[:10])


def test_two_decimal_cells_use_d2_keys():
    if not have_real():
        return
    txt = body("tab_robust.tex")
    assert "ROB.H1.pooled.corrected.d2" in txt and "ROB.H1.pooled.corrected.ci.d2" in txt
    # AUROC cells are three decimals (no d2 twin)
    assert "ROB.H4.dauroc.corrected}" in txt


def test_rows_have_consistent_column_counts():
    if not have_real():
        return
    for fn in NAMES:
        txt = body(fn)
        for tab in re.findall(r"\\begin\{tabularx\}\{\\linewidth\}\{([^\n]*)\}\n(.*?)\\end\{tabularx\}", txt, re.S):
            spec, content = tab
            ncols = len(re.findall(r"[YZC]\{", spec))
            for row in content.split("\\\\"):
                row = row.strip()
                if not row or row.startswith("\\toprule") or row.startswith("\\bottomrule"):
                    continue
                row = re.sub(r"^\\(midrule|toprule)\s*", "", row).strip()
                if not row or row.startswith("\\bottomrule"):
                    continue
                if row.startswith("\\multicolumn"):
                    assert row.startswith("\\multicolumn{%d}" % ncols), (fn, row[:60])
                else:
                    assert row.count(" & ") == ncols - 1, (fn, ncols, row[:80])


def test_captions_carry_scope_label_and_interval_method():
    if not have_real():
        return
    c = body("tab_robust.tex")
    assert "robustness check, post hoc" in c and "database-clustered" in c and "\\xres{ROB.n_same}" in c
    c = body("tab_critic_diag.tex")
    assert "Exploratory, post hoc, simulated" in c and "percentile" in c
    c = body("tab_external.tex")
    assert "external validity" in c and "database-clustered" in c


def test_tab_pf_caption_columns_and_keys():
    if not have_real() or not pf_present():
        return
    c = body("tab_pf.tex")
    assert "\\caption{Exploratory, post hoc, simulated feedback" in c
    assert "database-clustered" in c and "Amendment 9 (Section 22)" in c
    for k in ("PF.PF1.delta.d2", "PF.PF2.delta.d2", "PF.PF3a.delta.d2", "PF.PF3.calls_ratio", "PF.PF3.cost_ratio", "PF.arm.pf_n0.ex.d2", "PF.arm.pf_n1000_q085.cost_k"):
        assert "\\xres{%s}" % k in c, k
    assert c.count("\\begin{tabularx}") == 2


def test_tab_pf_skipped_when_pf_absent():
    d = tempfile.mkdtemp()
    try:
        import shutil as sh
        r = os.path.join(d, "res")
        sh.copytree(RESULTS, r, ignore=sh.ignore_patterns("pf", "raw-ignored", "tuning"))
        if not have_real():
            return
        rc = T.main(["--results", r, "--out", os.path.join(d, "t"), "--paper", os.path.join(d, "nopaper")])
        assert rc == 0 and not os.path.exists(os.path.join(d, "t", "tab_pf.tex"))
    finally:
        shutil.rmtree(d)


def test_only_header_is_bold_and_no_dagger_words():
    if not have_real():
        return
    for fn in NAMES:
        txt = body(fn)
        assert "\\textbf" not in txt and "\\bf" not in txt
        assert not re.search(r"\b(significant|robust estimate|state-of-the-art|outperform)", txt, re.I)


def test_bird_columns_use_confirmatory_keys_that_exist():
    p = os.path.join(X.REPO, "paper", "results_macros.tex")
    if not have_real() or not os.path.exists(p):
        return
    have = set(re.findall(r"csname resv@(.+?)\\endcsname", open(p, encoding="utf-8", errors="replace").read()))
    d, _ = built()
    _, rk = T.used_keys(open(os.path.join(d, "tab_external.tex")).read())
    assert rk
    assert not [k for k in rk if k not in have], [k for k in rk if k not in have]


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
