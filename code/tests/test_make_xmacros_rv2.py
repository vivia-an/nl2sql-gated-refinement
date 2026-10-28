"""Tests of the RV2 builder of code/tools/make_xmacros.py (prefix RV2., scope posthoc-secondary, SAP Amendment 10 / section 23). Stdlib only, runs on the host.
Plain asserts; runs with pytest or `python tests/test_make_xmacros_rv2.py`. Tests that need the real file use $XMACROS_RESULTS (default <repo>/results) and skip when absent."""
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
import rv2_keys as K  # noqa: E402

RESULTS = os.environ.get("XMACROS_RESULTS", os.path.join(X.REPO, "results"))
SRC = os.path.join(RESULTS, "posthoc2", "rv2_summary.json")
_DOC = {}


def doc():
    if "d" not in _DOC:
        _DOC["d"] = json.load(open(SRC)) if os.path.exists(SRC) else None
    return _DOC["d"]


def _skip():
    if doc() is None:
        try:
            import pytest
            pytest.skip("results/posthoc2/rv2_summary.json absent")
        except ImportError:
            return True
    return False


def run_builder(mut=None):
    """write a (mutated) copy of the summary into a temp results dir and run build_rv2 on it -> (ctx, errors, scope_errors)"""
    d = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(d, "posthoc2"))
        j = copy.deepcopy(doc())
        if mut:
            mut(j)
        json.dump(j, open(os.path.join(d, "posthoc2", "rv2_summary.json"), "w"))
        ctx = X.Ctx(d)
        X.build_rv2(ctx)
        return ctx, list(ctx.errors), list(ctx.scope_errors)
    finally:
        shutil.rmtree(d)


def test_source_is_registered_with_the_posthoc_secondary_scope_only():
    assert X.SOURCES["rv2"] == ("posthoc2/rv2_summary.json", ("posthoc-secondary",))
    assert "RV2" in X.LABELS and "build_rv2" in [b.__name__ for b in X.BUILDERS]


def test_clean_file_has_no_errors_and_every_expected_key():
    if _skip():
        return
    ctx, err, sc = run_builder()
    assert not err and not sc, err[:3]
    for k, _ in K.expected_keys():
        assert "RV2." + k in ctx.entries, k


def test_values_equal_the_summary_file():
    if _skip():
        return
    ctx, _, _ = run_builder()
    for k in ("loop.pooled.eff_in.gated", "dual.P1p.pooled.est", "dual.W1.flash.p_mcnemar", "cost.plus.zs.cny_1000", "route.route_ca.cov60.err", "rev.net"):
        e = doc()["keys"][k]
        assert ctx.entries["RV2." + k].raw == e["v"] or ctx.entries["RV2." + k].raw == tuple(e["v"]), k
    e = doc()["keys"]["loop.flash.eff_in.gated"]
    assert ctx.entries["RV2.loop.flash.eff_in.gated.ci"].raw == tuple(e["ci"]) and ctx.entries["RV2.loop.flash.eff_in.gated.item_ci"].raw == tuple(e["item_ci"])


def test_interval_valued_base_keys_get_lo_and_hi():
    if _skip():
        return
    ctx, _, _ = run_builder()
    for k in ("dual.P1p.plus.ci", "dual.P1p.plus.item_ci", "dual.W1.flash.t_ci"):
        e = doc()["keys"][k]["v"]
        assert ctx.entries["RV2." + k + "_lo"].raw == e[0] and ctx.entries["RV2." + k + "_hi"].raw == e[1]


def test_strict_gate_fails_for_each_kind_of_defect():
    if _skip():
        return
    def drop(j):
        del j["keys"]["loop.plus.stack"]

    def null(j):
        j["keys"]["loop.plus.stack"]["v"] = None

    def extra(j):
        j["keys"]["loop.nonsense"] = {"v": 1, "kind": "pp"}

    def kind(j):
        j["keys"]["loop.plus.stack"]["kind"] = "pct"

    def api(j):
        j["_meta"]["api_calls"] = 3

    def miss(j):
        j["_meta"]["missing_keys"] = ["x"]

    def boot(j):
        j["_meta"]["B"] = 2000

    def interval(j):
        j["keys"]["loop.flash.eff_in.gated"]["ci"] = [None, 1.0]

    for f, needle in ((drop, "missing key"), (null, "null at"), (extra, "not in the grammar"), (kind, "grammar says"), (api, "api_calls"),
                      (miss, "missing_keys"), (boot, "_meta.B"), (interval, "null end")):
        _, err, _ = run_builder(f)
        assert any(needle in e for e in err), (f.__name__, err[:3])


def test_wrong_scope_is_reported():
    if _skip():
        return
    def scope(j):
        j["_meta"]["scope"] = "confirmatory"
    _, _, sc = run_builder(scope)
    assert sc and "posthoc-secondary" in sc[0]


def test_absent_file_is_an_error_for_rv2():
    d = tempfile.mkdtemp()
    try:
        ctx = X.Ctx(d)
        X.build_rv2(ctx)
        assert any("posthoc2/rv2_summary.json" in e for e in ctx.errors)
    finally:
        shutil.rmtree(d)


def test_build_all_and_check_paper_are_clean_with_the_real_results():
    if _skip() or not all(os.path.exists(os.path.join(RESULTS, rel)) for rel, _ in [v for v in X.SOURCES.values() if "pf/" not in v[0]]):
        return
    ctx = X.build_all(RESULTS)
    assert not [e for e in ctx.errors if "posthoc2" in e]
    assert "RV2.loop.pooled.eff_in.gated.fmt" in ctx.entries
    assert all(k.startswith("RV2.") for k in ctx.entries if k.startswith("RV2")) and not any(k.startswith("RV2.") and k.endswith(".fmt.fmt") for k in ctx.entries)


def test_builder_has_no_result_numbers_typed_in():
    import ast
    tree = ast.parse(open(os.path.join(TOOLS, "make_xmacros.py")).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_rv2":
            nums = [n.n for n in ast.walk(node) if isinstance(n, ast.Num) and not isinstance(n.n, bool)]
            assert set(nums) <= {0, 1, 2, 3}, nums


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
