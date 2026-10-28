"""Tests of code/tools/r3_merge.py (round-3 merge into results/r3/rv3_summary.json). Stdlib only, runs on the host. Tests that need the merged real files use
$R3_DIR (default <repo>/results/r3) and skip when absent."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOLS)
import r3_merge as M  # noqa: E402
import rv3_keys as K3  # noqa: E402

R3 = os.environ.get("R3_DIR", os.path.join(M.REPO, "results", "r3"))


def _have():
    return all(os.path.exists(os.path.join(R3, n)) for n in ("rep.json", "samex.json", "retrieval.json", "routing.json", "p5.json", "q360.json", "addendum.json", "cost.json"))


def test_reading_words_follow_the_sup0_rule():
    assert M.reading(0.1, 2.0) == "positive" and M.reading(-2.0, -0.1) == "negative" and M.reading(-0.1, 2.0) == "contains 0" and M.reading(0.0, 1.0) == "contains 0"
    assert M.ptxt([-0.0487, 2.6585]) == "[-0.05, +2.66]"


def test_out_rejects_duplicate_and_null_keys():
    o = M.Out()
    o.put("a.b", 1.0, "x2")
    for bad in (lambda: o.put("a.b", 2.0, "x2"), lambda: o.put("c", None, "x2")):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_fmt_entry_uses_the_macro_formats():
    assert M.fmt_entry({"v": -0.04873, "kind": "x3s"}) == "-0.049"
    assert M.fmt_entry({"v": 0.456, "kind": "pp2", "ci": [-1.234, 2.256]}) == "+0.46 cl[-1.23, +2.26]"
    assert M.fmt_entry({"v": [1.0, 2.5], "kind": "pp2"}) == "[+1.00, +2.50]"
    assert M.fmt_entry({"v": "yes", "kind": "str"}) == "yes"


def test_merged_summary_matches_the_committed_grammar():
    if not _have():
        return
    doc, order = M.build(R3)
    g = K3.expected_keys()
    assert [k for k, _ in g] == order, "rv3_grammar.json is stale: run r3_merge.py --write-grammar"
    assert all(doc["keys"][k]["kind"] == kind for k, kind in g)
    for k in order:
        assert doc["keys"][k]["v"] is not None


def test_every_model_run_is_complete_in_the_inputs():
    if not _have():
        return
    S = json.load(open(os.path.join(R3, "rep.json")))
    assert S["notes"]["n_qids"] == 1534
    assert all(v == "ok" for v in S["notes"]["run_notes"].values()), [k for k, v in S["notes"]["run_notes"].items() if v != "ok"]
    assert all(e["n_runs"] == 3 and e["s0"] is not None for e in S["arms"].values())


def test_headline_run_is_never_averaged_into_the_replication():
    """author decision 2026-10-28: the replication mean is the mean of the three T 0.7 runs only; seed 0 is reported apart."""
    if not _have():
        return
    S = json.load(open(os.path.join(R3, "rep.json")))
    for k, e in S["arms"].items():
        assert sorted(e["per"]) == ["101", "202", "303"], k
        assert abs(e["mean"] - sum(e["per"].values()) / 3.0) < 1e-9, k
    for k, e in S["contrasts"].items():
        assert sorted(e["per"]) == ["101", "202", "303"], k
        assert abs(e["est"] - sum(e["per"].values()) / 3.0) < 1e-6, k
        assert e["agrees"]["cluster"] in (True, False) and "s0" in e
    doc, order = M.build(R3)
    assert not any(k.startswith("seed.") for k in order), "old mixed-temperature keys must not be emitted"


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
