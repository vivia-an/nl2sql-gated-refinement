"""Tests of the RV2S builder of code/tools/make_xmacros.py (summary results/posthoc2/rv2s_summary.json, grammar rv2s_keys.py; prefix RV2.small|deff|margin|cost.nodisc|gate;
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
import rv2s_keys as KS  # noqa: E402

RESULTS = os.environ.get("XMACROS_RESULTS", os.path.join(X.REPO, "results"))
SRC = os.path.join(RESULTS, "posthoc2", "rv2s_summary.json")
_DOC = {}


def doc():
    if "d" not in _DOC:
        _DOC["d"] = json.load(open(SRC)) if os.path.exists(SRC) else None
    return _DOC["d"]


def _skip():
    if doc() is None:
        try:
            import pytest
            pytest.skip("results/posthoc2/rv2s_summary.json absent")
        except ImportError:
            return True
    return False


def run_builder(mut=None):
    d = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(d, "posthoc2"))
        j = copy.deepcopy(doc())
        if mut:
            mut(j)
        json.dump(j, open(os.path.join(d, "posthoc2", "rv2s_summary.json"), "w"))
        ctx = X.Ctx(d)
        X.build_rv2s(ctx)
        return ctx, list(ctx.errors), list(ctx.scope_errors)
    finally:
        shutil.rmtree(d)


def test_source_is_registered_with_the_posthoc_secondary_scope_only():
    assert X.SOURCES["rv2s"] == ("posthoc2/rv2s_summary.json", ("posthoc-secondary",))
    assert "build_rv2s" in [b.__name__ for b in X.BUILDERS]


def test_p4_format():
    assert X.fnum(0.00048828125, "p4") == "0.0005" and X.fnum(0.00002, "p4") == "<0.0001" and X.fnum(0.0293, "p4") == "0.0293"


def test_clean_file_has_no_errors_and_every_expected_key():
    if _skip():
        return
    ctx, err, sc = run_builder()
    assert not err and not sc, err[:3]
    for k, _ in KS.expected_keys():
        assert "RV2." + k in ctx.entries, k


def test_interval_keys_get_lo_hi_and_ratio_keys_get_both_interval_types():
    if _skip():
        return
    ctx, _, _ = run_builder()
    for t in KS.CI_TYPES:
        k = "RV2.small.P1p.pooled.%s_ci" % t
        assert isinstance(ctx.entries[k].raw, tuple) and k + "_lo" in ctx.entries and k + "_hi" in ctx.entries
    k = "RV2.cost.nodisc.ratio.plus.gated.vs_plus_zs.nodisc"
    assert k + ".ci" in ctx.entries and k + ".item_ci_lo" in ctx.entries


def test_wrong_kind_missing_key_extra_key_and_api_calls_are_errors():
    if _skip():
        return

    def wrong_kind(j):
        j["keys"]["small.P1p.pooled.est"]["kind"] = "pct"

    def missing(j):
        del j["keys"]["deff.all.min"]

    def extra(j):
        j["keys"]["deff.bogus"] = {"v": 1, "kind": "int", "ci": None, "item_ci": None, "n": None, "meaning": ""}

    def api(j):
        j["_meta"]["api_calls"] = 3

    def null(j):
        j["keys"]["deff.all.max"]["v"] = None
    for mut, frag in ((wrong_kind, "kind of"), (missing, "missing key"), (extra, "not in the grammar"), (api, "api_calls"), (null, "null at")):
        _, err, _ = run_builder(mut)
        assert any(frag in e for e in err), (frag, err[:2])


def test_scope_must_be_posthoc_secondary():
    if _skip():
        return
    d = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(d, "posthoc2"))
        j = copy.deepcopy(doc())
        j["_meta"]["scope"] = "confirmatory"
        json.dump(j, open(os.path.join(d, "posthoc2", "rv2s_summary.json"), "w"))
        ctx = X.Ctx(d)
        assert ctx.scope_errors
    finally:
        shutil.rmtree(d)


def test_values_equal_the_summary_file_and_no_duplicate_with_rv2():
    if _skip():
        return
    ctx, _, _ = run_builder()
    for k in ("small.W1.pooled.est", "deff.all.max", "margin.pooled.in.fixed.ratio", "gate.bird.plus.gated.fired", "small.pstar.P1p.padj_sf"):
        e = doc()["keys"][k]
        assert ctx.entries["RV2." + k].raw == e["v"], k
    if os.path.exists(os.path.join(RESULTS, "posthoc2", "rv2_summary.json")):
        ctx2 = X.build_all(RESULTS)                   # raises RuntimeError on a duplicate key between RV2 and RV2S
        assert "RV2.small.W1.pooled.est" in ctx2.entries and "RV2.dual.W1.pooled.est" in ctx2.entries


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
