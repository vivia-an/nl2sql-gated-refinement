"""Tests of code/tools/r4_merge.py (round-4 merge into results/r4/rv4_summary.json). Stdlib only, runs on the host. Tests that need the real result files use $R4_DIR
(default <repo>/results/r4) and skip when absent."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOLS)
import r4_merge as M  # noqa: E402
import rv4_keys as K4  # noqa: E402

R4 = os.environ.get("R4_DIR", os.path.join(M.REPO, "results", "r4"))
R3 = os.path.join(M.REPO, "results", "r3")


def _have():
    return all(os.path.exists(os.path.join(R4, n)) for n in ("widest.json", "regen.json", "critic_cost.json", "margin_rep.json", "mde_cluster.json", "p5_units.json", "silent.json")) \
        and os.path.exists(os.path.join(R3, "addendum.json"))


def _build():
    """M.build needs the git history for the lock versions (A15); a checkout or worker machine copy without .git skips these tests."""
    try:
        return M.build(R4, R3)
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ == "CalledProcessError" or isinstance(e, OSError):
            return None
        raise


class Any(dict):
    def __missing__(self, key):
        return self.template


def _cell(rep_wide, s0_wide, old):
    return {"est": 0.5 * (rep_wide[0] + rep_wide[1]), "per": {"101": 0.0, "202": 0.3, "303": 0.6},
            "rep4": {"wide": rep_wide, "t": [rep_wide[0] + 0.1, rep_wide[1] - 0.1], "sf": [rep_wide[0], rep_wide[1]]}, "s0_4": {"wide": s0_wide}, "agrees_old": old}


def test_text_helpers():
    assert M.money(3.366) == "3.37" and M.iv_money(0.8249, None) == "[0.82, unbounded]" and M.iv_money(0.056, 0.2) == "[0.06, 0.20]"
    assert M.yes(True) == "yes" and M.yes(False) == "no"


def test_widest_counts_agreement_by_reading_word_under_one_definition():
    o = M.Out()
    W = {"contrasts": Any(), "samex": Any()}
    W["contrasts"].template = _cell([-1.0, 2.0], [0.5, 2.0], True)      # rep contains 0, locked positive: disagree under the single definition, agreed under the old flag
    W["samex"].template = _cell([0.5, 2.0], [0.5, 2.0], False)          # same word, old flag false
    M.emit_wide(o, W)
    k = {key: e["v"] for key, e in o.keys.items()}
    assert k["wide.rep.n_contrasts"] == 25 and k["wide.rep.n_agree"] == 0 and k["wide.rep.n_agree_prev"] == 25
    assert k["wide.samex.n_contrasts"] == 12 and k["wide.samex.n_agree"] == 12 and k["wide.samex.n_agree_prev"] == 0
    assert k["wide.rep.stack.plus.rep_reading"] == "contains 0" and k["wide.rep.stack.plus.s0_reading"] == "positive" and k["wide.rep.stack.plus.agrees"] == "no"
    assert k["wide.rep.w1.weak2.rep_wide"] == [-1.0, 2.0]
    # round 7 (revision list 7 D3, D9): unrounded lower end and the replication allowance of P1'
    assert k["wide.rep.stack.plus.rep_lo_raw"] == "-1.0e+00"
    est, lo, sd = 0.5, -1.0, 0.3                          # toy cell: per-run estimates 0.0, 0.3, 0.6 -> SD 0.3
    assert abs(k["wide.rep.g_minus_f.pooled3.rep_run_sd"] - sd) < 1e-12
    want = est - ((est - lo) ** 2 + (1.96 * sd) ** 2) ** 0.5
    assert abs(k["wide.rep.g_minus_f.pooled3.rep_allow_lo"] - want) < 1e-12
    assert abs(k["wide.rep.g_minus_f.pooled3.rep_allow_clearance"] - (want + 1.5)) < 1e-12
    assert "wide.rep.stack.plus.rep_allow_lo" not in k and "wide.samex.stack.plus.rep_lo_raw" not in k


def test_replication_allowance_values_from_the_result_file():
    if not _have():
        return
    import math
    with open(os.path.join(R4, "widest.json")) as f:
        W = json.load(f)
    e = W["contrasts"]["g_minus_f|pooled3"]
    per = list(e["per"].values())
    mu = sum(per) / 3.0
    sd = math.sqrt(sum((x - mu) ** 2 for x in per) / 2.0)
    lo = e["rep4"]["wide"][0]
    want = e["est"] - math.sqrt((e["est"] - lo) ** 2 + (1.96 * sd) ** 2)
    o = M.Out()
    M.emit_wide(o, W)
    assert abs(o.keys["wide.rep.g_minus_f.pooled3.rep_allow_lo"]["v"] - want) < 1e-12
    assert want < lo and want > -1.5                      # widening lowers the end, and the margin is still kept (the supplement sentence depends on both)
    # the Plus gated-over-root-only replication: a lower end of -1.9e-09 is zero up to the search tolerance and reads "contains 0"
    raw = float(o.keys["wide.rep.loop_g.plus.rep_lo_raw"]["v"])
    assert abs(raw) < 1e-6 and o.keys["wide.rep.loop_g.plus.rep_reading"]["v"] == "contains 0"


COUNTS = {"rep": 25, "samex": 12, "route": 7, "p5": 2, "q360": 18, "ret": 6, "critic": 3, "spider": 6}


def test_mult_total_is_the_sum_of_its_parts():
    o = M.Out()
    M.emit_mult(o, {"mult": {"n_dual": 110, "n_loop": 20}}, COUNTS)
    k = {key: e["v"] for key, e in o.keys.items()}
    assert k["mult.n_round2"] == 130 and k["mult.n_total"] == 209 and k["mult.n_round3"] == 203
    assert abs(k["mult.expected_false_pos"] - 10.45) < 1e-9
    assert sum(k["mult.n_" + p] for p in ("s32", "loop", "rep", "samex", "route", "p5", "q360", "ret", "critic", "spider")) == k["mult.n_total"]
    # round 5 (A5): the four stack contrasts of the same-examples control are those of the replication, counted twice
    assert k["mult.n_dup"] == 4 and k["mult.n_distinct"] == 205


def test_mult_counts_are_read_from_the_result_files():
    """round 5 (V5 L-7): the counts that are not table rows come from the files that hold them, not from typed constants"""
    res = os.path.join(M.REPO, "results")
    if not os.path.exists(os.path.join(R3, "rep_contrasts.csv")):
        return
    assert M.mult_counts(R3, res) == COUNTS


def test_lock_versions_hash_the_committed_text_and_detect_a_changed_first_entry():
    base = [{"system": "a", "config_sha256": "x"}, {"system": "b", "config_sha256": "y"}]

    def mk(extra, first=None):
        e = [dict(x) for x in base]
        if first:
            e[0]["config_sha256"] = first
        return json.dumps({"entries": e + [{"system": "c%d" % i} for i in range(extra)]}, sort_keys=True)
    texts = {"2d14769": mk(0), "09a14f0": mk(2), "fe59b33": mk(5)}
    L = M.lock_versions(show=lambda rev: texts[rev])
    assert (len(L["orig"]["entries"]), len(L["primary"]["entries"]), len(L["r3"]["entries"])) == (2, 4, 7) and L["orig_unchanged"] is True
    assert len(L["orig"]["hash"]) == 12 and L["orig"]["hash"] != L["primary"]["hash"]
    texts["fe59b33"] = mk(5, first="CHANGED")
    assert M.lock_versions(show=lambda rev: texts[rev])["orig_unchanged"] is False
    o = M.Out()
    M.emit_lock(o, L)
    assert o.keys["lock.n_r3"]["v"] == 7 and o.keys["lock.orig_unchanged"]["v"] == "yes"


def test_committed_lock_versions_match_the_published_hashes():
    try:
        L = M.lock_versions()
    except Exception:       # no git history in this checkout
        return
    assert L["primary"]["hash"] == "d97296311b75" and len(L["primary"]["entries"]) == 20
    assert L["r3"]["hash"] == "5fa2aca433b9" and len(L["r3"]["entries"]) == 28 and L["orig_unchanged"] is True and len(L["orig"]["entries"]) == 16


def test_merged_summary_matches_the_committed_grammar():
    if not _have():
        return
    built = _build()
    if built is None:
        return
    doc, order = built
    g = K4.expected_keys()
    assert [k for k, _ in g] == order, "rv4_grammar.json is stale: run r4_merge.py --write-grammar"
    assert all(doc["keys"][k]["kind"] == kind for k, kind in g)
    assert all(doc["keys"][k]["v"] is not None for k in order)
    assert doc["_meta"]["api_calls"] == 0 and doc["_meta"]["B"] == 10000


def test_undefined_icer_where_the_increment_is_not_positive():
    if not _have():
        return
    built = _build()
    if built is None:
        return
    K = built[0]["keys"]
    assert K["ccost.27b.defined"]["v"] == "no" and K["ccost.27b.icer_disc"]["v"] == "undefined"
    assert K["ccost.plus.defined"]["v"] == "yes" and K["ccost.plus.icer_disc_iv"]["v"].endswith("unbounded]")


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


def test_zero_end_convention_changes_the_replication_count_by_one():
    # round 8 (V8 M-1): an end at zero within 1e-6 read as positive lowers the agreeing count by one in the replication (20 -> 19)
    o = M.Out()
    W = {"contrasts": Any(), "samex": Any()}
    W["contrasts"].template = _cell([-1e-9, 2.0], [-0.5, 2.0], True)     # rep end at -1e-9 (zero), locked run contains 0: agree under the rule, disagree if read positive
    W["samex"].template = _cell([0.5, 2.0], [0.5, 2.0], False)
    M.emit_wide(o, W)
    k = {key: e["v"] for key, e in o.keys.items()}
    assert k["wide.rep.n_agree"] == 25 and k["wide.rep.n_agree_zero_positive"] == 0
    assert k["wide.samex.n_agree"] == 12 and k["wide.samex.n_agree_zero_positive"] == 12


def test_replication_zero_convention_count_from_the_result_file():
    if not _have():
        return
    with open(os.path.join(R4, "widest.json")) as f:
        W = json.load(f)
    o = M.Out()
    M.emit_wide(o, W)
    assert o.keys["wide.rep.n_agree"]["v"] == 20 and o.keys["wide.rep.n_agree_zero_positive"]["v"] == 19
