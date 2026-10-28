"""TS.sel.rule3.hr.delta_ex: the selection rule's own HR figure (gated minus fixed) is a tuning-set key sourced from exp/selection_decisions.json"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "code", "tools"))
SEL = os.path.join(REPO, "exp", "selection_decisions.json")
TSTEX = os.path.join(REPO, "paper", "results_macros_ts.tex")


def _sel():
    return json.load(open(SEL))["rule3_detail"]["hr"]


def test_key_values_come_from_the_selection_file():
    if not os.path.exists(SEL) or not os.path.exists(os.path.join(REPO, "results", "x5", "x5_tuning_summary.json")):
        return                                    # build() needs all tuning summaries
    import make_macros_ts as T
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        R = T.build()
    finally:
        os.chdir(cwd)
    h = _sel()
    assert R.v["TS.sel.rule3.hr.delta_ex.ci_lo"] == h["diff_pp_ci95"][0]
    assert R.v["TS.sel.rule3.hr.delta_ex.ci_hi"] == h["diff_pp_ci95"][1]
    assert R.v["TS.sel.rule3.hr.delta_ex.mean"] == h["diff_pp"]
    # the rule failed condition (i): the bound is below the margin of -1.5 pp
    assert h["diff_pp_ci95"][0] < -1.5


def test_generated_macro_is_defined_and_typeset_with_three_decimals():
    if not os.path.exists(TSTEX) or not os.path.exists(SEL):
        return
    txt = open(TSTEX, encoding="utf-8").read()
    m = re.search(r"csname tsv@TS\.sel\.rule3\.hr\.delta_ex\.ci_lo\\endcsname\{(.*)\}", txt)
    assert m, "TS.sel.rule3.hr.delta_ex.ci_lo is not defined in results_macros_ts.tex"
    lo = _sel()["diff_pp_ci95"][0]
    assert m.group(1) == "\\ensuremath{-}%.3f" % abs(lo)


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            f()
            print("ok  ", n)
