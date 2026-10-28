"""Tests of the RV2 tables of code/tools/make_xtables.py (tab_loop, tab_cost, tab_dual, tab_dual_full; SAP Amendment 10 / section 23). Stdlib only (host python 3.6).
Plain asserts; runs with pytest or `python tests/test_make_xtables_rv2.py`. Needs results/posthoc2/rv2_summary.json (skips when absent)."""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import make_xmacros as X  # noqa: E402
import make_xtables as T  # noqa: E402

RESULTS = os.environ.get("XMACROS_RESULTS", os.path.join(X.REPO, "results"))
NAMES = ("tab_loop.tex", "tab_cost.tex", "tab_dual.tex", "tab_dual_full.tex")
_C = {}


def ctx():
    if "c" not in _C:
        _C["c"] = X.build_all(RESULTS) if os.path.exists(os.path.join(RESULTS, "posthoc2", "rv2_summary.json")) else None
    return _C["c"]


def tex(name):
    c = ctx()
    if c is None:
        return None
    return dict(T.TABLES)[name](c.entries)


def test_tables_are_registered():
    assert [n for n, _ in T.TABLES if n in NAMES] == list(NAMES)


def test_every_xres_key_of_the_rv2_tables_exists():
    c = ctx()
    if c is None:
        return
    for n in NAMES:
        xk, rk = T.used_keys(tex(n))
        assert xk and not rk, n
        assert not [k for k in xk if k not in c.entries], (n, [k for k in xk if k not in c.entries][:3])


def test_captions_start_with_the_marker_and_use_booktabs():
    if ctx() is None:
        return
    for n in NAMES:
        t = tex(n)
        # a table taller than one page (round 3): one caption (\captionof), blocks of rows as page material, no float and no "Continued." caption; the others are one float
        mk = "Pooled over" if n == "tab_loop.tex" else "Post hoc,"               # tab_loop opens with its finding, the others with the marker
        assert t.count("\\caption{" + mk) + t.count("\\captionof{table}{" + mk) == 1, n
        assert "\\ContinuedFloat" not in t and "Continued." not in t, n
        assert t.count("\\begin{table}") == t.count("\\caption{" + mk), n
        assert "\\toprule" in t and "\\midrule" in t and "\\bottomrule" in t, n
        assert "\\hline" not in t, n


def test_labels_are_unique_and_split_tables_carry_one_caption():
    if ctx() is None:
        return
    labels = []
    for n in NAMES:
        labels += re.findall(r"\\label\{([^}]+)\}", tex(n))
    assert len(labels) == len(set(labels)) and {"tab:loop", "tab:cost", "tab:dual", "tab:dual_full"} <= set(labels)
    full = tex("tab_dual_full.tex")
    assert full.count("\\begin{table}") == 0 and full.count("\\captionof{table}") == 1 and full.count("\\begin{longtable}") == 1 and "\\endhead" in full and full.count("\\addtocounter{table}{-1}") == 1       # one caption, one longtable that repeats its header, the counter step of the environment undone
    for n in ("tab_cost.tex", "tab_dual.tex", "tab_small.tex"):
        assert tex(n).count("\\captionof{table}") == 1 and "\\begin{table}" not in tex(n), n
        assert tex(n).count("\\begin{longtable}") == tex(n).count("\\addtocounter{table}{-1}"), n          # every longtable undoes its own counter step: one number per table
        assert "ContinuedFloat" not in tex(n) and "Continued." not in tex(n), n


def test_every_dual_row_of_the_grammar_is_in_the_full_table():
    c = ctx()
    if c is None:
        return
    import rv2_keys
    full = tex("tab_dual_full.tex")
    for i in rv2_keys.DUAL_IDS:
        assert "\\xres{RV2.dual.%s.est}" % i in full, i


def test_headline_rows_are_a_subset_of_the_full_table():
    if ctx() is None:
        return
    head, full = tex("tab_dual.tex"), tex("tab_dual_full.tex")
    ks = set(re.findall(r"\\xres\{(RV2\.dual\.[^}]+\.est)\}", head))
    assert ks and ks <= set(re.findall(r"\\xres\{(RV2\.dual\.[^}]+\.est)\}", full))


def test_no_value_is_typed_into_a_cell():
    """a row is `label & \\xres{..} ...`; outside the label and the fixed words there is no decimal number"""
    if ctx() is None:
        return
    for n in ("tab_loop.tex", "tab_cost.tex", "tab_dual.tex"):
        for line in tex(n).splitlines():
            if line.startswith("%") or "&" not in line or "\\xres" not in line:
                continue
            cells = [x.strip() for x in line.rstrip("\\ ").split("&")][1:]
            for cell in cells:
                rest = re.sub(r"\\xres\{[^}]*\}", "", cell)
                assert not re.search(r"\d", rest), (n, cell)


def test_tab_cost_names_the_cache_assumption_in_caption_and_every_cost_column():
    """V3 M-3 (round 3): the discount is an assumption applied to recorded cached-token counts; every cost or ratio column says which pricing it uses."""
    if ctx() is None:
        return
    t = tex("tab_cost.tex")
    cap = t.split("\\label{tab:cost}")[0]
    assert "actually" not in cap and "assum" in cap and "cached prompt tokens" in cap and "0.2 of the input price" in cap
    headers = [l for l in t.splitlines() if l.startswith("Configuration &") or l.startswith("Qwen3.7-Plus single call against &") or l.startswith("Score and coverage &")]
    assert len(headers) == 6 and len(set(headers)) == 3          # three blocks; a longtable states its header twice (first page, continuation pages)
    for h in headers:
        for col in h.rstrip("\\ ").split(" & "):
            if re.search(r"CNY|[Cc]ost ratio|Dominance|Pareto", col):
                assert "cache discount assumed" in col or "full input price" in col, col


def test_tab_loop_is_three_blocks_with_no_nine_column_block():
    """R3 N-m7 (round 3): the lower block had nine columns and wrapped its interval cells; it is now two blocks of at most seven columns, and the ratio column is renamed."""
    if ctx() is None:
        return
    t = tex("tab_loop.tex")
    specs = re.findall(r"\\begin\{tabularx\}\{[^}]*\}\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", t)
    assert len(specs) == 3 and max(len(re.findall(r"[YZ]\{", sp)) for sp in specs) <= 7
    assert "Margin / fixed-loop effect" in t and "1.5\\,pp / fixed effect" not in t
    assert t.count("\\begin{table}") == 1 and "\\label{tab:loop}" in t


if __name__ == "__main__":
    bad = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                bad += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if bad else 0)
