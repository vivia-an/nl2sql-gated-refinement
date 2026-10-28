"""Tests of the RV3 builder of code/tools/make_xmacros.py (summary results/r3/rv3_summary.json, grammar rv3_keys.py / rv3_grammar.json; prefix RV3.;
scope posthoc-secondary). Stdlib only, runs on the host. Plain asserts; runs with pytest or `python tests/test_make_xmacros_rv2s.py`.
Tests that need the real file use $XMACROS_RESULTS (default <repo>/results) and skip when absent."""
import copy
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOLS)
import make_xmacros as X  # noqa: E402
import rv3_keys as K3  # noqa: E402

RESULTS = os.environ.get("XMACROS_RESULTS", os.path.join(X.REPO, "results"))
SRC = os.path.join(RESULTS, "r3", "rv3_summary.json")
_DOC = {}


def doc():
    if "d" not in _DOC:
        _DOC["d"] = json.load(open(SRC)) if os.path.exists(SRC) else None
    return _DOC["d"]


def _skip():
    if doc() is None:
        try:
            import pytest
            pytest.skip("results/posthoc2/rv3_summary.json absent")
        except ImportError:
            return True
    return False


def run_builder(mut=None):
    d = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(d, "r3"))
        j = copy.deepcopy(doc())
        if mut:
            mut(j)
        json.dump(j, open(os.path.join(d, "r3", "rv3_summary.json"), "w"))
        ctx = X.Ctx(d)
        X.build_rv3(ctx)
        return ctx, list(ctx.errors), list(ctx.scope_errors)
    finally:
        shutil.rmtree(d)


def test_source_is_registered_with_the_posthoc_secondary_scope_only():
    assert X.SOURCES["rv3"] == ("r3/rv3_summary.json", ("posthoc-secondary",))
    assert "build_rv3" in [b.__name__ for b in X.BUILDERS]
    assert "RV3" in X.LABELS


def test_grammar_file_is_wellformed():
    g = K3.expected_keys()
    keys = [k for k, _ in g]
    assert len(keys) == len(set(keys)) and len(keys) > 500
    assert all(kind in K3.KINDS for _, kind in g)
    for fam in ("rep.arm.", "rep.stack.pooled3.", "rep.g_minus_f.pooled3.", "rep.w1.weak2.", "samex.stack.", "ret.", "route.", "p5.", "q360.", "inc.", "mde.", "mult.", "critic.", "stack.pooled.", "lat.", "cost."):
        assert any(k.startswith(fam) for k in keys), fam


def test_signed_formats_keep_small_bounds_visible():
    assert X.fnum(-0.04873, "x3s") == "-0.049" and X.fnum(-0.04873, "pp2") == "-0.05" and X.fnum(-0.04873, "pp") == "0.0"          # why the pooled stack bound uses pp2 / x3s


def test_clean_file_has_no_errors_and_every_expected_key():
    if _skip():
        return
    ctx, err, sc = run_builder()
    assert not err and not sc, err[:3]
    for k, _ in K3.expected_keys():
        assert "RV3." + k in ctx.entries, k


def test_api_calls_may_be_positive_but_not_negative_or_missing():
    if _skip():
        return

    def pos(j):
        j["_meta"]["api_calls"] = 20000

    def neg(j):
        j["_meta"]["api_calls"] = -1

    def gone(j):
        del j["_meta"]["api_calls"]
    assert not run_builder(pos)[1]
    for mut in (neg, gone):
        _, err, _ = run_builder(mut)
        assert any("api_calls" in e for e in err)


def test_wrong_kind_missing_key_extra_key_null_and_b_are_errors():
    if _skip():
        return
    first = K3.expected_keys()[0][0]

    def wrong_kind(j):
        j["keys"][first]["kind"] = "str" if j["keys"][first]["kind"] != "str" else "int"

    def missing(j):
        del j["keys"][first]

    def extra(j):
        j["keys"]["bogus.key"] = {"v": 1, "kind": "int"}

    def null(j):
        j["keys"][first]["v"] = None

    def badb(j):
        j["_meta"]["B"] = 2000
    for mut, frag in ((wrong_kind, "kind of"), (missing, "missing key"), (extra, "not in the grammar"), (null, "null at"), (badb, "_meta.B")):
        _, err, _ = run_builder(mut)
        assert any(frag in e for e in err), (frag, err[:2])


def test_scope_must_be_posthoc_secondary():
    if _skip():
        return
    d = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(d, "r3"))
        j = copy.deepcopy(doc())
        j["_meta"]["scope"] = "confirmatory"
        json.dump(j, open(os.path.join(d, "r3", "rv3_summary.json"), "w"))
        assert X.Ctx(d).scope_errors
    finally:
        shutil.rmtree(d)


def test_pair_values_get_lo_hi_and_intervals_get_ci_keys():
    if _skip():
        return
    ctx, _, _ = run_builder()
    k = "RV3.rep.stack.pooled3.est"
    assert k + ".ci_lo" in ctx.entries and k + ".item_ci_hi" in ctx.entries
    k = "RV3.rep.stack.pooled3.wide"
    assert isinstance(ctx.entries[k].raw, tuple) and k + "_lo" in ctx.entries and k + "_hi" in ctx.entries


def test_values_equal_the_summary_file_and_no_duplicate_with_other_prefixes():
    if _skip():
        return
    ctx, _, _ = run_builder()
    for k in ("rep.g_minus_f.pooled3.est", "mde.item_min", "mult.n_total", "cost.total_cny", "stack.pooled.wide_lo_x3"):
        assert ctx.entries["RV3." + k].raw == doc()["keys"][k]["v"], k
    if os.path.exists(os.path.join(RESULTS, "posthoc2", "rv2_summary.json")):
        ctx2 = X.build_all(RESULTS)                   # raises RuntimeError on a duplicate key between prefixes
        assert "RV3.rep.stack.pooled3.est" in ctx2.entries and "RV2.dual.W1.pooled.est" in ctx2.entries


def test_pooled_stack_bound_is_not_rounded_to_zero():
    if _skip():
        return
    ctx, _, _ = run_builder()
    lo = ctx.entries["RV3.stack.pooled.wide_lo_x3"]
    assert lo.raw < 0 and "0.0" != lo.text() and lo.text().replace("\\ensuremath{-}", "-") == "-0.049"


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
