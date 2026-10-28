"""Round-3 figure checks (group F): Figure 1 without the bar panel, Figure 3 split (panel b moved to the supplement figure fig_quoting), text >= 8 pt at the printed width.
Needs matplotlib (run on the worker machine): python -m pytest code/tests/test_figures_round3.py ; figures are redrawn from figs/data/figdata_final.json, nothing is written to figs/."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
DATA = os.path.join(ROOT, "figs", "data", "figdata_final.json")
needs_data = pytest.mark.skipif(not os.path.exists(DATA), reason="figs/data/figdata_final.json not present")


def _mods():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from analysis_pipeline.figures import make_all, style
    return make_all, style


def _fig(stem):
    make_all, style = _mods()
    D = json.load(open(DATA))
    fig = make_all.build_figure(stem, D, "final", False)
    return fig, style


def _texts(fig):
    return [(t.get_text(), t.get_fontsize()) for t in fig.findobj(lambda a: hasattr(a, "get_fontsize") and hasattr(a, "get_text")) if t.get_text().strip()]


def test_stems_and_print_widths_declared():
    make_all, style = _mods()
    assert "fig_quoting" in make_all.STEMS
    for s in ("fig1_architecture", "fig_repair", "fig_errors", "fig_pollution"):
        assert style.PRINT_W_CM[s] <= 17.5
    assert style.PRINT_W_CM["fig_quoting"] <= 17.5


@needs_data
def test_fig1_has_no_bar_panel():
    fig, _ = _fig("fig1_architecture")
    txt = " ".join(t for t, _ in _texts(fig))
    assert "Worked example" in txt
    assert "calls per question" not in txt and "CNY per 1000" not in txt          # the cost bars of the former panel (b) are gone
    assert "(c)" not in txt and "tuning set" in txt
    src = open(os.path.join(ROOT, "code", "analysis_pipeline", "figures", "draw_arch.py")).read()
    assert "_headline(" not in src


@needs_data
def test_fig3_main_has_two_panels_and_supplement_figure_holds_tuning_panel():
    fig, _ = _fig("fig_repair")
    assert len([a for a in fig.axes if a.get_visible()]) == 2
    assert not any("tuning set" in t for t, _ in _texts(fig))                      # the tuning-set panel left the main figure
    fq, _ = _fig("fig_quoting")
    txt = " ".join(t for t, _ in _texts(fq))
    assert "tuning set" in txt and "quoting" in txt.lower()


@needs_data
@pytest.mark.parametrize("stem", ["fig1_architecture", "fig_repair", "fig_quoting", "fig_pollution", "fig_errors"])
def test_text_at_least_8pt_and_no_shrinking_at_print_width(stem):
    fig, style = _fig(stem)
    small = [x for x in _texts(fig) if x[1] < 8 - 1e-9]
    assert not small, small[:5]
    w_cm = style.fit_width(fig, style.PRINT_W_CM[stem])
    assert w_cm <= style.PRINT_W_CM[stem] + 0.01, w_cm            # the PDF is printed at scale >= 1, so the 8 pt text stays >= 8 pt


@needs_data
def test_figdata_sources_match_result_files():
    """_meta.sources hashes equal the sha256 of the files in results/ (regenerated after the MEMOFIX changes)."""
    import hashlib
    D = json.load(open(DATA))
    src = D["_meta"]["sources"]
    assert src
    bad = []
    for name, h in src.items():
        for cand in (os.path.join(ROOT, "results", name), os.path.join(ROOT, "results", "posthoc2", name), os.path.join(ROOT, "figs", "data", name)):
            if os.path.exists(cand):
                if hashlib.sha256(open(cand, "rb").read()).hexdigest() != h:
                    bad.append(name)
                break
    assert not bad, bad
