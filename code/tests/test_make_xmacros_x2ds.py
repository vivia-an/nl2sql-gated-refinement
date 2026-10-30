"""Tests of the X2DS builder of code/tools/make_xmacros.py (key file results/x2ds/x2ds_keys.json written by code/tools/x2ds_keys.py; grammar x2ds_grammar.json; prefix X2DS.;
scope posthoc-secondary; non-Qwen backbone and the retry-only control). Stdlib only, runs on the host. Plain asserts; runs with pytest or `python tests/test_make_xmacros_x2ds.py`.
Tests that need the real files use $XMACROS_RESULTS (default <repo>/results) and skip when absent."""
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
import x2ds_keys as K  # noqa: E402

RESULTS = os.environ.get("XMACROS_RESULTS", os.path.join(X.REPO, "results"))
SRC = os.path.join(RESULTS, "x2ds", "x2ds_keys.json")
_DOC = {}


def doc():
    if "d" not in _DOC:
        _DOC["d"] = json.load(open(SRC)) if os.path.exists(SRC) else None
    return _DOC["d"]


def _skip():
    if doc() is None:
        try:
            import pytest
            pytest.skip("results/x2ds/x2ds_keys.json absent")
        except ImportError:
            return True
    return False


def run_builder(mut=None):
    d = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(d, "x2ds"))
        j = copy.deepcopy(doc())
        if mut:
            mut(j)
        json.dump(j, open(os.path.join(d, "x2ds", "x2ds_keys.json"), "w"))
        ctx = X.Ctx(d)
        X.build_x2ds(ctx)
        return ctx, list(ctx.errors), list(ctx.scope_errors)
    finally:
        shutil.rmtree(d)


def test_source_is_registered_with_the_posthoc_secondary_scope_only():
    assert X.SOURCES["x2ds"] == ("x2ds/x2ds_keys.json", ("posthoc-secondary",))
    assert "build_x2ds" in [b.__name__ for b in X.BUILDERS]
    assert "X2DS" in X.LABELS


def test_grammar_file_is_wellformed():
    g = K.expected_keys()
    assert len(g) > 600 and len(set(k for k, _ in g)) == len(g)
    assert all(kind in K.KINDS for _, kind in g)


def test_reading_convention_end_printing_as_zero_contains_zero():
    assert K.reading(0.004, 1.0) == "contains 0"
    assert K.reading(0.006, 1.0) == "positive"
    assert K.reading(-2.0, -0.004) == "contains 0"
    assert K.reading(-2.0, -0.006) == "negative"


def test_key_file_equals_what_the_inputs_give():
    if _skip():
        return
    again = json.loads(json.dumps(K.build(os.path.join(RESULTS, "x2ds"))))
    assert again == doc()
    assert [k for k, _ in K.expected_keys()] == list(doc()["keys"])


def test_builder_is_clean_on_the_real_file_and_defines_the_numbers_of_the_paper():
    if _skip():
        return
    ctx, errors, scope_errors = run_builder()
    assert errors == [] and scope_errors == []
    for k in ("X2DS.ds.arm.gated.ex", "X2DS.flash.ctr.zsretry_minus_zs", "X2DS.flash.ctr.zsretry_minus_zs.wide", "X2DS.pool.w2.share_retry", "X2DS.retry.gain_min", "X2DS.ds.calls_ratio_gf"):
        assert k in ctx.entries, k


def test_the_numbers_stated_in_the_task_are_in_the_key_file():
    if _skip():
        return
    k = doc()["keys"]
    r2 = lambda x: round(x, 2)
    assert [r2(k["ds.arm.%s.ex" % a]["v"]) for a in ("zs", "fixed", "gated", "rootonly")] == [62.39, 64.28, 64.28, 64.02]
    assert r2(k["ds.ctr.gated_minus_zs"]["v"]) == 1.89 and r2(k["ds.ctr.gated_minus_zs"]["ci"][0]) == 0.43
    assert r2(k["ds.ctr.gated_minus_fixed"]["v"]) == 0.0 and (k["ds.ctr.gated_minus_fixed.b"]["v"], k["ds.ctr.gated_minus_fixed.c"]["v"]) == (34, 34)
    assert r2(k["ds.calls_ratio_gf"]["v"]) == 0.56
    for b, est in (("plus", 0.26), ("flash", 1.50), ("27b", 0.85), ("ds", 0.33)):
        assert r2(k["%s.ctr.zsretry_minus_zs" % b]["v"]) == est, b
    for b, est in (("plus", 0.39), ("flash", 0.91), ("27b", 2.61), ("ds", 1.56)):
        assert r2(k["%s.ctr.gated_minus_zsretry" % b]["v"]) == est, b
    assert [r2(k["%s.arm.zsretry.cny_list" % b]["v"]) for b in ("plus", "flash", "27b", "ds")] == [6.11, 0.64, 1.97, 5.77]


def test_widest_interval_contains_the_other_three_and_is_their_hull():
    if _skip():
        return
    keys = doc()["keys"]
    for key in keys:
        if key.endswith(".wide") and ".ctr." in key:
            base = key[:-5]
            lo = min(keys[base]["ci"][0], keys[base]["item_ci"][0], keys[base + ".t"]["v"][0], keys[base + ".sf"]["v"][0])
            hi = max(keys[base]["ci"][1], keys[base]["item_ci"][1], keys[base + ".t"]["v"][1], keys[base + ".sf"]["v"][1])
            assert keys[key]["v"] == [lo, hi], key


def test_builder_refuses_a_missing_key_a_null_and_a_wrong_kind():
    if _skip():
        return

    def drop(j):
        del j["keys"]["ds.arm.zs.ex"]

    def null(j):
        j["keys"]["ds.arm.zs.ex"]["v"] = None

    def kind(j):
        j["keys"]["ds.arm.zs.ex"]["kind"] = "int"
    for m in (drop, null, kind):
        _, errors, _ = run_builder(m)
        assert errors, m.__name__


def test_builder_refuses_an_unknown_key_and_a_negative_api_count():
    if _skip():
        return

    def extra(j):
        j["keys"]["ds.extra"] = {"v": 1, "kind": "int", "meaning": "x"}

    def api(j):
        j["_meta"]["api_calls"] = -1
    for m in (extra, api):
        _, errors, _ = run_builder(m)
        assert errors, m.__name__


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
    print("ok: %d tests" % n)
