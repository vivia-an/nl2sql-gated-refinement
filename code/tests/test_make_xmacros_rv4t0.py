"""Tests of the RV4T0 builder of code/tools/make_xmacros.py (summary results/r4_t0/rv4t0_summary.json, grammar rv4t0_keys.py / rv4t0_grammar.json; prefix RV4T0.;
scope posthoc-secondary; temperature-0 non-determinism check, reduced scope). Stdlib only, runs on the host. Plain asserts; runs with pytest or `python tests/test_make_xmacros_rv4t0.py`.
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
import rv4t0_keys as K  # noqa: E402

RESULTS = os.environ.get("XMACROS_RESULTS", os.path.join(X.REPO, "results"))
SRC = os.path.join(RESULTS, "r4_t0", "rv4t0_summary.json")
_DOC = {}


def doc():
    if "d" not in _DOC:
        _DOC["d"] = json.load(open(SRC)) if os.path.exists(SRC) else None
    return _DOC["d"]


def _skip():
    if doc() is None:
        try:
            import pytest
            pytest.skip("results/r4_t0/rv4t0_summary.json absent")
        except ImportError:
            return True
    return False


def run_builder(mut=None):
    d = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(d, "r4_t0"))
        j = copy.deepcopy(doc())
        if mut:
            mut(j)
        json.dump(j, open(os.path.join(d, "r4_t0", "rv4t0_summary.json"), "w"))
        ctx = X.Ctx(d)
        X.build_rv4t0(ctx)
        return ctx, list(ctx.errors), list(ctx.scope_errors)
    finally:
        shutil.rmtree(d)


def test_source_is_registered_with_the_posthoc_secondary_scope_only():
    assert X.SOURCES["rv4t0"] == ("r4_t0/rv4t0_summary.json", ("posthoc-secondary",))
    assert "build_rv4t0" in [b.__name__ for b in X.BUILDERS]
    assert "RV4T0" in X.LABELS


def test_grammar_file_is_wellformed():
    g = K.expected_keys()
    keys = [k for k, _ in g]
    assert len(keys) == len(set(keys)) and len(keys) > 150
    assert all(kind in K.KINDS for _, kind in g)
    for fam in ("nd.plus.fixed.", "nd.flash.zs.", "nd.27b.zs.", "nd.27b.fixed.", "nd.flash.fixed.", "nd.complete.", "pair.f_minus_zs.", "lock.", "cmp.", "cost.", "lock_file."):
        assert any(k.startswith(fam) for k in keys), fam


def test_clean_file_has_no_errors_and_every_expected_key():
    if _skip():
        return
    ctx, err, sc = run_builder()
    assert not err and not sc, err[:3]
    for k, _ in K.expected_keys():
        assert "RV4T0." + k in ctx.entries, k


def test_api_calls_must_be_a_nonnegative_int():
    if _skip():
        return

    def neg(j):
        j["_meta"]["api_calls"] = -1

    def gone(j):
        del j["_meta"]["api_calls"]
    for mut in (neg, gone):
        _, err, _ = run_builder(mut)
        assert any("api_calls" in e for e in err)


def test_wrong_kind_missing_key_extra_key_null_and_b_are_errors():
    if _skip():
        return
    first = K.expected_keys()[0][0]

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
        os.makedirs(os.path.join(d, "r4_t0"))
        j = copy.deepcopy(doc())
        j["_meta"]["scope"] = "confirmatory"
        json.dump(j, open(os.path.join(d, "r4_t0", "rv4t0_summary.json"), "w"))
        assert X.Ctx(d).scope_errors
    finally:
        shutil.rmtree(d)


def test_values_equal_the_summary_file_and_no_duplicate_with_other_prefixes():
    if _skip():
        return
    ctx, _, _ = run_builder()
    for k in ("nd.27b.zs.sql_diff", "nd.plus.fixed.b", "nd.plus.fixed.c", "nd.complete.sql_diff_mean", "cost.total_cny"):
        if k in doc()["keys"]:
            assert ctx.entries["RV4T0." + k].raw == doc()["keys"][k]["v"], k
    if os.path.exists(os.path.join(RESULTS, "r4", "rv4_summary.json")):
        X.build_all(RESULTS)                       # raises RuntimeError on a duplicate key between prefixes


def test_all_twelve_arms_complete_and_b_plus_c_matches_changed_correctness():
    """round 5: every arm has a complete r1 (1534 questions); b + c is the share of changed correctness; the contrasts and the noise scale are present."""
    if _skip():
        return
    k = doc()["keys"]
    if k["nd.complete.n_arms"]["v"] < 12:      # round-4 reduced file: the old partial labelling
        assert k["nd.flash.fixed.complete"]["v"] == "no"
        return
    assert doc()["_meta"]["mode"] == "r1" and doc()["_meta"]["complete"] is True
    for b in ("plus", "flash", "27b"):
        for a in ("zs", "rootonly", "gated", "fixed"):
            n = int(k["nd.%s.%s.n" % (b, a)]["v"])
            assert n == 1534 and k["nd.%s.%s.complete" % (b, a)]["v"] == "yes"
            bc = int(k["nd.%s.%s.b" % (b, a)]["v"]) + int(k["nd.%s.%s.c" % (b, a)]["v"])
            assert abs(100.0 * bc / n - k["nd.%s.%s.ex_changed" % (b, a)]["v"]) < 1e-6
            assert abs(100.0 * bc ** 0.5 / n - k["nd.%s.%s.sd_net" % (b, a)]["v"]) < 1e-6
    assert int(k["ctr.n_total"]["v"]) == 25 and "ctr.g_minus_f.pooled3.change" in k and "ctr.loop_g.pooled3.sd_indep" in k and "p1p.r1_est" in k


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
