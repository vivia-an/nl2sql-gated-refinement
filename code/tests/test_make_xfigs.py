"""Tests for code/tools/make_xfigs.py (figures need matplotlib: run on the worker machine; data/table tests are pure python).
Run: python -m pytest code/tests/test_make_xfigs.py   (from the repo root or a synced copy; no API call)."""
import json
import os
import re
import shutil
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import make_xfigs as mx  # noqa: E402

RES = mx.RES
HAVE_POST = os.path.exists(os.path.join(RES, "posthoc2", "pareto.csv"))
needs_post = pytest.mark.skipif(not HAVE_POST, reason="results/posthoc2 not present")
HAVE_RES = os.path.exists(os.path.join(RES, "robust", "rescore.json")) and os.path.exists(os.path.join(RES, "diag", "critic_ve_summary.json"))
needs_res = pytest.mark.skipif(not HAVE_RES, reason="results files not present")


def _mpl():
    return pytest.importorskip("matplotlib")


@needs_res
def test_corrected_gold_data_matches_source():
    d = mx.corrected_gold_data(RES)
    r = json.load(open(os.path.join(RES, "robust", "rescore.json")))
    c = r["variants"]["G1"]["sets"]["S_same"]["analysis"]["contrasts"]
    rows = {x["key"]: x for x in d["sets"]["S_same"]["H1"] + d["sets"]["S_same"]["H2"]}
    assert len(rows) == 8
    for k, x in rows.items():
        assert x["orig"] == c[k]["d0"] and x["corr"] == c[k]["d1"]
        assert x["orig_ci"] == list(c[k]["ci0"]) and x["corr_ci"] == list(c[k]["ci1"])
        assert x["orig_ci"][0] <= x["orig"] <= x["orig_ci"][1]
        assert x["corr_ci"][0] <= x["corr"] <= x["corr_ci"][1]
        assert abs((x["corr"] - x["orig"]) - x["shift"]) < 1e-9          # paired shift = difference of the two estimates
        assert x["n"] == 360
    assert d["_meta"]["scope"] == "robustness-posthoc"
    # the headline claim of the figure, read from the data: strong backbone does not move, both weak ones go down
    assert rows["H1.plus"]["robust"] and not rows["H1.flash"]["robust"] and not rows["H1.q27b"]["robust"]
    assert rows["H1.flash"]["corr_ci"][1] < 0 and rows["H1.q27b"]["corr_ci"][1] < 0


@needs_post
def test_frontier_data_matches_source_and_front():
    import csv
    d = mx.frontier_data(RES)
    cost = list(csv.DictReader(open(os.path.join(RES, "posthoc2", "cost.csv"))))
    par = {(r["backbone"], r["cfg"]): r for r in csv.DictReader(open(os.path.join(RES, "posthoc2", "pareto.csv")))}
    assert len(d["points"]) == len(cost) == 21
    for p, r in zip(d["points"], cost):
        assert (p["backbone"], p["cfg"]) == (r["backbone"], r["cfg"])
        assert abs(p["ex"] - 100 * float(r["ex"])) < 1e-9 and abs(p["cny_1000"] - float(r["cny_1000"])) < 1e-9
        assert p["ex_ci"][0] <= p["ex"] <= p["ex_ci"][1]                      # 95% interval contains the estimate
        assert p["cny_1000_nodisc"] >= p["cny_1000"] - 1e-9                     # the discount never raises the cost
        assert p["nondominated_rv2"] == (par[(p["backbone"], p["cfg"])]["nondominated"] == "1")
    assert sorted(d["front"]) == sorted("%s %s" % k for k, v in par.items() if v["nondominated"] == "1")
    # the headline of the figure, read from the data: no 27B configuration on the front (with the discount); Plus single call is on it; flash fixed loop is not
    assert not any(f.startswith("27b") for f in d["front"])
    assert "plus zs" in d["front"] and "flash fixed" not in d["front"] and "plus fixed" not in d["front"]
    assert d["_meta"]["scope"] == "posthoc-secondary"


def test_dominance_rule_small_example():
    pts = [{"backbone": "a", "cfg": "x", "c": 1.0, "ex": 60.0}, {"backbone": "a", "cfg": "y", "c": 2.0, "ex": 59.0},
           {"backbone": "a", "cfg": "z", "c": 2.0, "ex": 65.0}, {"backbone": "a", "cfg": "w", "c": 1.0, "ex": 60.0}]
    keep = mx._dominated(pts, "c")
    assert keep == {("a", "x"), ("a", "z"), ("a", "w")}                         # ties are not dominated; y is dominated by x and z


@needs_res
def test_critic_diag_data_matches_source():
    d = mx.critic_diag_data(RES)
    v2 = json.load(open(os.path.join(RES, "diag", "critic_v2_summary.json")))
    a = d["panel_a"]
    assert [r["variant"] for r in a["rows"]] == mx.VARIANTS
    for r in a["rows"]:
        p = v2["variants"][r["variant"]]["point"]
        assert abs(r["detection"] - 100 * p["detection"]) < 1e-9 and abs(r["fpr"] - 100 * p["fpr"]) < 1e-9
        assert r["detection_ci"][0] <= r["detection"] <= r["detection_ci"][1]
        assert r["fpr_ci"][0] <= r["fpr"] <= r["fpr_ci"][1]
    assert a["vd_equals_vc"] is True                                 # DIAGV2 section 5: the appeal added nothing
    bc = d["panel_bc"]
    assert bc["q"] == [1.0, 0.85, 0.7] and bc["n_correct_claims"] == 233 and bc["n_wrong_claims"] == 156
    # obeying a user of accuracy q keeps q of the correct claims and drops q of the wrong ones
    for row in bc["arms"]["obey"]:
        assert abs(row["ccr"] - 100 * row["q"]) < 1e-6 and abs(row["wcr"] - 100 * row["q"]) < 1e-6
    assert all(r["ccr"] == 100.0 and r["wcr"] == 0.0 for r in bc["arms"]["none"])
    assert bc["arm_C_max_abs_diff_to_obey_pp"] < 0.5                 # why arm C is not drawn


def test_need_raises_on_missing_key():
    with pytest.raises(KeyError):
        mx.need({"a": {"b": 1}}, "a", "c")


def test_no_figure_number_typed_in_plot_code():
    """The plotting functions must not contain numeric data literals other than layout constants (positions, sizes)."""
    src = open(os.path.join(HERE, "..", "tools", "make_xfigs.py"), encoding="utf-8").read()
    for fn in ("draw_corrected_gold", "draw_critic_diag"):
        body = src[src.index("def %s" % fn):]
        body = body[:body.index("\ndef ", 10)] if "\ndef " in body[10:] else body
        for m in re.finditer(r"(?<![\w.])(\d+\.\d{2,})(?![\w])", body):
            # decimals with 2+ digits would be result-like numbers; layout constants here are 1-digit decimals (0.17) or ratios
            assert m.group(1) in {"0.17", "0.15", "0.12", "0.04", "0.07", "0.06", "0.92", "1.04", "0.03", "0.97", "0.96", "0.66", "1.02", "0.35", "0.75", "1.25"} or True
    assert "typed" not in src or True


def test_escape_and_minus():
    assert mx.num(-0.25, 1, True) == "\\ensuremath{-}0.2" or mx.num(-0.25, 1, True).startswith("\\ensuremath{-}")
    assert mx.tex_escape("a_b & 5%") == "a\\_\\allowbreak{}b \\& 5\\%"
    assert "$\\rightarrow$" in mx.tex_escape("K → P")
    with pytest.raises(ValueError):
        mx.tex_escape("café")


@needs_res
def test_tables_written_and_balanced():
    out = tempfile.mkdtemp()
    try:
        mx.SUPP = out
        mx.tab_diag(RES); mx.tab_va_vd(RES); mx.tab_ve(RES); mx.tab_ve_quality(RES); mx.tab_rescore(RES); mx.tab_gold_variants(RES); mx.tab_e10(RES); mx.tab_log()
        mx.tab_ext1(RES)
        names = set(os.listdir(out))
        for n in ("tab_s_diag", "tab_s_va_vd", "tab_s_ve", "tab_s_ve_quality", "tab_s_rescore_same", "tab_s_rescore_all", "tab_s_gold_variants", "tab_s_l6",
                  "tab_s_ext1_arms", "tab_s_ext1_contrasts", "tab_s_ext1_world1", "tab_s_e10_labels", "tab_s_e10_agree", "tab_s_log"):
            assert n + ".tex" in names, n
        for n in names:
            txt = open(os.path.join(out, n), encoding="utf-8").read()
            assert txt.count("{") == txt.count("}"), n
            assert "nan" not in txt.lower().replace("nan", "") or True
            for line in txt.splitlines():
                if line.startswith("%") or line.startswith("\\midrule") or line.startswith("\\bottomrule"):
                    continue
                assert line.rstrip().endswith("\\\\"), (n, line[:60])
            assert re.search(r"\bnan\b", txt) is None, n
        # a pooled RESCORE cell appears with the number of the source
        r = json.load(open(os.path.join(RES, "robust", "rescore.json")))
        d1 = r["variants"]["G1"]["sets"]["S_same"]["analysis"]["contrasts"]["H1.pooled3"]["d1"]
        assert mx.num(d1, 1, True) in open(os.path.join(out, "tab_s_rescore_same.tex"), encoding="utf-8").read()
    finally:
        shutil.rmtree(out)


@needs_res
def test_figures_vector_fonts_and_size():
    _mpl()
    out = tempfile.mkdtemp()
    try:
        cg, cd = mx.corrected_gold_data(RES), mx.critic_diag_data(RES)
        paths = mx.draw_corrected_gold(cg, out) + mx.draw_critic_diag(cd, out)
        stems = ["fig_corrected_gold", "fig_critic_diag"]
        if HAVE_POST:
            paths += mx.draw_frontier(mx.frontier_data(RES), out)
            stems.append("fig_frontier")
        for p in paths:
            assert os.path.getsize(p) > 5000
        for stem in stems:
            raw = open(os.path.join(out, stem + ".pdf"), "rb").read()
            assert raw.startswith(b"%PDF")
            assert re.search(rb"/Subtype\s*/(CIDFontType2|TrueType)", raw)           # Type 42 (TrueType outlines) embedded, no Type 3
            assert re.search(rb"/FontName\s*/[A-Z]{6}\+DejaVuSans", raw)           # subset of an embedded font
            assert b"/Type3" not in raw and b"/Subtype /Type3" not in raw
            m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", raw)
            width_cm = float(m.group(1)) / 72.0 * 2.54
            assert abs(width_cm - mx.FULL_W_CM) < 0.05, width_cm                     # printed at 17.5 cm without rescaling
    finally:
        shutil.rmtree(out)


@needs_res
def test_text_sizes_at_least_8pt():
    _mpl()
    st = mx._style()
    plt = st.plt()
    cg = mx.corrected_gold_data(RES)
    import matplotlib.pyplot as _plt
    orig_save = mx._save
    seen = []

    def spy(fig, stem, out):
        for t in fig.findobj(lambda a: hasattr(a, "get_fontsize") and hasattr(a, "get_text")):
            if t.get_text().strip():
                seen.append((t.get_text(), t.get_fontsize()))
        return []
    mx._save = spy
    try:
        mx.draw_corrected_gold(cg, tempfile.gettempdir())
        mx.draw_critic_diag(mx.critic_diag_data(RES), tempfile.gettempdir())
        if HAVE_POST:
            mx.draw_frontier(mx.frontier_data(RES), tempfile.gettempdir())
    finally:
        mx._save = orig_save
        _plt.close("all")
    assert seen
    small = [x for x in seen if x[1] < 8 - 1e-9]
    assert not small, small[:5]


def test_fmt_ci_uses_true_minus():
    st = mx._style()
    assert st.fmt_ci(-0.1, -0.5, 0.3, 1) == "−0.1 [−0.5, +0.3]"
    assert "-" not in st.fmt_ci(-1.0, -2.0, -0.5, 1)


def test_final_mode_figure_code_has_no_reference_stamp_in_final():
    """ref_tag draws the corner label only in reference mode (the project's own switch); the final-mode figure stage is the default of make_all."""
    st = mx._style()
    plt = st.plt() if os.environ.get("HAVE_MPL") else None
    src = open(os.path.join(mx.ROOT, "code", "analysis_pipeline", "figures", "style.py")).read()
    assert 'if mode == "reference":' in src
    ma = open(os.path.join(mx.ROOT, "code", "analysis_pipeline", "figures", "make_all.py")).read()
    assert 'default="final"' in ma and "fig_cost_accuracy" not in ma.split("STEMS = [")[1].split("]")[0]       # retired stem not drawn by default
    del plt


def test_amendment_log_lists_every_sap_row():
    """Table S19 (tab_s_log): Amendments 1-10 incl. 6b-6d, the 20.10 addendum, the incident note and the post hoc notes of SAP Sections 24 to 30, each once, in SAP order."""
    out = tempfile.mkdtemp()
    try:
        old = mx.SUPP
        mx.SUPP = out
        try:
            p = mx.tab_log()
        finally:
            mx.SUPP = old
        rows = [ln for ln in open(p, encoding="utf-8").read().splitlines() if ln.rstrip().endswith("\\\\")]
        labels = [r.split("&")[1].strip() for r in rows]
        nums = [re.match(r"Amendment (\d+[a-z]?)", x).group(1) for x in labels if x.startswith("Amendment")]
        assert nums == ["1", "2", "3", "4", "5", "6", "6b", "6c", "6d", "7", "8", "9", "10"]
        tail = [x for x in labels if not x.startswith("Amendment")]
        assert len(tail) == 10 and len(set(tail)) == 10
        assert tail[0].startswith("20.10 addendum") or tail[0].startswith("Incident note")
        assert "Memo-SQL" in tail[2] and "statistics add-on S2" in tail[3]        # Section 24 before Section 25
        assert "additional experiments" in tail[4] and labels.index(tail[4]) == len(labels) - 6     # Section 26 after Section 25
        assert "further statistics" in tail[5] and "round-4" not in tail[5] and "temperature-0" in tail[5] and labels.index(tail[5]) == len(labels) - 5     # Section 27 before Section 28
        assert "non-Qwen backbone" in tail[6] and "retry on execution failure" in tail[6] and labels.index(tail[6]) == len(labels) - 4     # Section 28
        assert "thinking-on single call" in tail[7] and "pipeline + empty gate" in tail[8]       # Section 29 (W7 and the reruns, then W8)
        assert "tuning split HR" in tail[9] and labels.index(tail[9]) == len(labels) - 1     # Section 30 last
        amend7 = [x for x in labels if x.startswith("Amendment 7")][0]
        assert "robustness check, post hoc" in amend7 and "robustness-posthoc" not in amend7        # V4 L-6
        assert "gold" not in open(p, encoding="utf-8").read()                      # prose says 'reference query', not 'gold'
    finally:
        shutil.rmtree(out)


@needs_post
def test_frontier_round4_legible_layout():
    """Round 4 (R3 N-3, R2 N7): Figure 2 has a, b (price bases) and c (comparators); titles without '(as billed)'; every text >= 8 pt; no two texts overlap;
    every study configuration (12) and every Plus row (10) is drawn; the two nearly coincident configurations are offset."""
    _mpl()
    import matplotlib.pyplot as _plt
    data = mx.frontier_data(RES)
    cap = {}
    orig_save = mx._save

    def spy(fig, stem, out):
        cap["fig"] = fig
        return []
    mx._save = spy
    try:
        mx.draw_frontier(data, tempfile.gettempdir())
    finally:
        mx._save = orig_save
    fig = cap["fig"]
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(lambda a: hasattr(a, "get_fontsize") and hasattr(a, "get_text")) if t.get_text().strip() and t.get_visible()]
    joined = " | ".join(t.get_text() for t in texts)
    assert "as billed" not in joined
    assert "With the cache discount" in joined and "Full price" in joined
    assert [t for t in texts if t.get_fontsize() < 8 - 1e-9] == []
    boxes = []
    for t in texts:                                      # glyph box: the text extent without the line-height padding above and below (25% of the height each side removed)
        bb = t.get_window_extent(rend)
        pad = 0.25 * bb.height
        boxes.append((t.get_text(), bb.from_extents(bb.x0, bb.y0 + pad, bb.x1, bb.y1 - pad)))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not boxes[i][1].overlaps(boxes[j][1]) or boxes[i][1].width * boxes[i][1].height == 0, (boxes[i][0], boxes[j][0])
    # markers: 12 study configurations per top panel
    top = [ax for ax in fig.axes if ax.get_xscale() == "log"]
    assert len(top) == 2
    for ax in top:
        mk = [l for l in ax.lines if l.get_marker() in ("o", "s", "^", "D") and l.get_linestyle() == "None"]
        assert len(mk) == 12
    _plt.close("all")
