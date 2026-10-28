"""RESCORE (SAP Amendment 7): set construction, scoring against a replaced gold, reading rule, contrasts. Offline sqlite only; numpy needed
for the contrast tests (worker machine venv)."""
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import rescore_corrected as RC  # noqa: E402

np = pytest.importorskip("numpy")


def dev_row(i, db="d1", q="how many?", e="e", sql="SELECT 1"):
    return {"question_id": i, "db_id": db, "question": q, "evidence": e, "SQL": sql}


# ------------------------------------------------------------------------------------------------------ sets
def test_build_sets_rules():
    dev = {i: dev_row(i) for i in range(1, 7)}
    var = {
        1: dev_row(1),                                             # nothing changed
        2: dev_row(2, sql="SELECT  1 "),                           # whitespace only: SQL text NOT changed
        3: dev_row(3, sql="SELECT 2"),                             # gold changed, same wording
        4: dev_row(4, q="how many rows?", sql="SELECT 2"),         # question changed: not in S_same
        5: dev_row(5, e=" e "),                                    # evidence differs only by strip: unchanged
        6: dev_row(6, e="other"),                                  # evidence changed
        99: dev_row(99),                                           # not a dev id: ignored
    }
    st = RC.build_sets(dev, var)
    q = lambda i: "bird_dev:d1:%d" % i
    assert st["all"] == [q(i) for i in range(1, 7)]
    assert st["same"] == [q(1), q(2), q(3), q(5)]
    assert st["sql_text_changed"][q(2)] is False and st["sql_text_changed"][q(3)] is True and st["sql_text_changed"][q(4)] is True
    assert st["q_changed"][q(4)] and not st["q_changed"][q(5)] and st["e_changed"][q(6)]
    st2 = RC.build_sets(dev, var, discordant={q(3)})
    assert q(3) not in st2["all"] and st2["db_discordant"] == [q(3)]


def test_kw_diff_and_same_result():
    assert RC.kw_diff("SELECT a FROM t", "select a from t") == ""
    assert RC.kw_diff("SELECT a FROM t", "SELECT DISTINCT a FROM t ORDER BY a LIMIT 1") == "DISTINCT|LIMIT|ORDER BY"
    assert "LEFT JOIN" in RC.kw_diff("SELECT 1", "SELECT 1 FROM a LEFT JOIN b ON 1")
    ok = {"ok": True, "digest": "3:abc"}
    assert RC.same_result(ok, {"ok": True, "digest": "3:abc"}) is True
    assert RC.same_result(ok, {"ok": True, "digest": "3:abd"}) is False
    assert RC.same_result(ok, {"ok": False, "digest": None}) is False
    assert RC.same_result({"ok": False, "digest": None}, {"ok": False, "digest": None}) is True
    assert RC.same_result(None, ok) is None


# ------------------------------------------------------------------------------------------------------ reading rule
def test_classify_type_s():
    r = RC.classify("S", 3.0, (1.0, 5.0), 2.0, (0.5, 3.5), -1.0, (-2.0, 0.2))
    assert r["robust"] and r["stable"]                                   # CI of the shift contains 0
    r = RC.classify("S", 3.0, (1.0, 5.0), -0.5, (-2.0, 1.0), -3.5, (-5.0, -2.0))
    assert not r["robust"] and not r["sign_same"]                        # sign flipped
    r = RC.classify("S", 3.0, (2.5, 3.5), 5.0, (4.0, 6.0), 2.0, (1.0, 3.0))
    assert not r["robust"] and not r["overlap"]                          # disjoint intervals
    r = RC.classify("S", 3.0, (1.0, 5.0), 2.0, (0.5, 3.5), -2.5, (-3.0, -2.0))
    assert r["robust"] and not r["stable"]                               # moved, significantly, by > 1 pp
    assert not RC.classify("S", 0.0, (-1, 1), 0.5, (-1, 2), 0.5, (0, 1))["robust"]   # d0 = 0 has no sign


def test_classify_type_n():
    assert RC.classify("N", -0.1, (-1.0, 0.8), 0.4, (-0.6, 1.4), 0.5, (-0.2, 1.1))["robust"]
    assert RC.classify("N", -0.1, (-1.0, 0.8), 0.4, (-0.6, 1.4), 0.5, (-0.2, 1.1))["stable"]
    assert not RC.classify("N", -0.1, (-1.0, 0.8), 1.8, (0.8, 2.8), 1.9, (1.0, 2.8))["robust"]   # |d1| > 1.5 pp
    assert RC.classify("N", 0.0, (-1.0, 1.0), -0.2, (-1.2, 0.8), -0.2, (-0.5, 0.1), "auc")["robust"]


def test_newcombe_and_revision_class():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from analysis_pipeline import stats
    lo, hi = RC.newcombe_diff(8, 10, 2, 10, stats.wilson)
    assert lo > 0 and hi < 1 and lo < 0.6 < hi
    assert RC.newcombe_diff(1, 0, 1, 5, stats.wilson) is None
    assert RC.revision_class(0, 1, True, True) == "helped"
    assert RC.revision_class(1, 0, True, True) == "hurt"
    assert RC.revision_class(1, 1, True, True) == "neutral"
    assert RC.revision_class(0, 1, False, True) == "repair_helped"
    assert RC.revision_class(0, 1, True, False) is None


def test_route_features_matches_primary_formula():
    import math
    coef = {"features": ["v3", "consistency", "nonempty"], "coef": [0.5, 2.0, 1.0], "intercept": -8.0}
    rows = [{"critic": {"score": 9.0, "semantic": 9, "syntax": 10}, "cons_bits": [1, 1, 0, 1, 0], "final_status": "ok_nonempty"},
            {"critic": {"score": 9.0, "semantic": None}, "cons_bits": [1], "final_status": "ok_nonempty"}]
    v3, ca = RC.route_features(rows, coef, 4)
    z = -8.0 + 0.5 * 9.0 + 2.0 * 0.75 + 1.0 * 1.0
    assert abs(ca[0] - 10.0 / (1 + math.exp(-z))) < 1e-12 and v3[0] == 9.0
    assert v3[1] is None and ca[1] is None


# ------------------------------------------------------------------------------------------------------ scoring against a replaced gold
@pytest.fixture()
def dbp(tmp_path):
    p = str(tmp_path / "t.sqlite")
    c = sqlite3.connect(p)
    c.execute("create table a (id integer primary key, name text)")
    c.executemany("insert into a values (?,?)", [(i, "n%d" % (i % 3)) for i in range(1, 7)])
    c.commit()
    c.close()
    return p


def test_evaluator_with_replaced_gold(dbp, tmp_path, monkeypatch):
    from nl2sql import paths
    from nl2sql.eval import score as SC
    monkeypatch.setattr(paths, "CACHE", tmp_path / "gc")
    monkeypatch.setattr(SC, "_GC", None)
    it = RC._item("bird_dev:t:1", "t", dbp)
    ev_o = RC.make_evaluator({"bird_dev:t:1": "select count(*) from a"})                   # original gold: 6
    ev_c = RC.make_evaluator({"bird_dev:t:1": "select count(distinct name) from a"})       # corrected gold: 3
    pred = "select count(*) from a where id <= 3"                                          # returns 3
    assert ev_o.score_sql(it, pred)["ex"] == 0
    assert ev_c.score_sql(it, pred)["ex"] == 1
    from nl2sql.db import execute_robust
    pr = execute_robust(dbp, pred, 30.0, None)
    assert ev_c.score_sql(it, pred, pred_res=pr)["ex"] == 1                                # passing the executed result is equivalent
    ev_bad = RC.make_evaluator({"bird_dev:t:1": "select nope from a"})                     # failing gold: EX 0, flagged
    out = ev_bad.score_sql(it, pred)
    assert out["ex"] == 0 and out["gold_fail"] is True
    g = ev_o.gold_exec(it)
    from nl2sql.eval.score import rows_digest
    assert rows_digest(g["rows"]) == rows_digest([[6]])
    assert RC.same_result({"ok": True, "digest": rows_digest(g["rows"])}, {"ok": True, "digest": rows_digest([(6,)])}) is True


# ------------------------------------------------------------------------------------------------------ contrasts on planted data
def _planted(n_per_db=60, dbs=("a", "b", "c", "d")):
    """items for labels X (gated) and Y (fixed): corrected gold flips 10 % of X's right answers to wrong."""
    rng = np.random.RandomState(7)
    items = {}
    qids = []
    for d in dbs:
        for i in range(n_per_db):
            qids.append("bird_dev:%s:%d" % (d, i))
    ex_y = (rng.rand(len(qids)) < 0.6).astype(int)
    ex_x = ex_y.copy()
    flip = rng.rand(len(qids)) < 0.1
    ex_x[flip] = 1 - ex_x[flip]
    x_c = ex_x.copy()
    right = np.where(ex_x == 1)[0]
    x_c[right[:8]] = 0                      # corrected gold: 8 of X's right answers become wrong
    for lab, o, c in (("X", ex_x, x_c), ("Y", ex_y, ex_y)):
        items[("final", lab)] = {q: {"usable": "1", "orig": str(int(o[k])), "G1": str(int(c[k])), "ex_stored": str(int(o[k])), "sha": "s", "root_status": "",
                                     "v3": "", "route_ca": "", "db": q.split(":")[1]} for k, q in enumerate(qids)}
    return items, qids, ex_x, x_c, ex_y


def test_paired_contrast_planted():
    from analysis_pipeline import stats as S
    items, qids, ex_x, x_c, ex_y = _planted()
    ctx = RC.Ctx(S, items, qids, 500)
    r = RC.paired_contrast(ctx, [("X", "Y")], "G1")
    assert r["n"] == len(qids) and r["n_dbs"] == 4
    assert abs(r["d0"] - 100 * (ex_x - ex_y).mean()) < 1e-9
    assert abs(r["d1"] - 100 * (x_c - ex_y).mean()) < 1e-9
    assert abs(r["shift"] - (r["d1"] - r["d0"])) < 1e-9 and r["shift"] < 0
    assert r["ci0"][0] <= r["d0"] <= r["ci0"][1]
    assert r["disc_orig"]["first_only"] + r["disc_orig"]["second_only"] >= 1
    # pooling: the mean of two identical pairs equals the single pair
    rp = RC.paired_contrast(ctx, [("X", "Y"), ("X", "Y")], "G1", with_item=False)
    assert abs(rp["d1"] - r["d1"]) < 1e-9
    # one resample is shared: the interval of the shift is not wider than the sum of the two interval widths
    w = lambda c: c[1] - c[0]
    assert w(r["ci_shift"]) <= w(r["ci0"]) + w(r["ci1"]) + 1e-9
    # usable filter drops items
    items[("final", "X")][qids[0]]["usable"] = "0"
    ctx2 = RC.Ctx(S, items, qids, 200)
    assert RC.paired_contrast(ctx2, [("X", "Y")], "G1", with_item=False)["n"] == len(qids) - 1


def test_headline_logic():
    def cell(rb, st):
        return {"conclusion": "C1", "robust": rb, "stable": st}
    res = {"variants": {"G1": {"sets": {
        "S_same": {"analysis": {"contrasts": {"a": cell(True, True), "b": cell(True, False)}, "H4.route": {"dauroc": dict(cell(True, True), conclusion="C3")},
                                "H5.revision": {"net_exec": dict(cell(False, False), conclusion="C4")}}},
        "S_all": {"analysis": {"contrasts": {"a": cell(True, True), "b": cell(False, False)}, "H4.route": {"dauroc": dict(cell(True, True), conclusion="C3")},
                               "H5.revision": {"net_exec": dict(cell(False, False), conclusion="C4")}}}}}}}
    h = RC.headline(res)["S_same"]
    assert h["C1"]["unchanged"] and not h["C1"]["unchanged_and_stable"] and h["C1"]["wording_sensitive"]
    assert h["C3"]["unchanged"] and not h["C3"]["wording_sensitive"]
    assert not h["C4"]["unchanged"] and h["C4"]["not_robust"] == ["H5.net_exec"]


def test_pair_audit_and_cases():
    from analysis_pipeline import stats as S
    items, qids, ex_x, x_c, ex_y = _planted()
    for lab, tag in (("X", "a"), ("Y", "b")):
        for k, q in enumerate(qids):
            items[("final", lab)][q]["sha"] = "same" if k % 2 == 0 else tag
    ctx = RC.Ctx(S, items, qids, 100)
    eff = {q: True for q in qids}
    pa = RC.pair_audit(ctx, "X", "Y", "G1", eff)
    assert pa["n"] == len(qids) and pa["pairs_with_differing_sql"] == len(qids) // 2
    assert sum(pa["transitions"].values()) == int((ex_x != x_c).sum())      # only X moves; every move is a transition of the pair
    assert pa["second_only_right_corr"] >= pa["second_only_right_orig"]
    ids = (30, 4, 12, 7, 50, 9, 2)
    rows = [["G1", "plus.gated", "bird_dev:d:%d" % i, "d", "gain", int(i % 2), 1, 1, 3, 3, 1, 1, "LIMIT"] for i in ids]
    out = RC.case_list(rows, per_cell=5)
    got = out["plus.gated|gain"]
    assert out["plus.gated|gain|n_total"] == 7
    assert [c["qid"].split(":")[2] for c in got] == ["7", "9", "2", "4", "12"]       # S_same first, ascending numeric id inside each block
    assert [c["in_S_same"] for c in got] == [True, True, False, False, False]
    assert out["plus.gated|loss"] == [] and out["flash.zs|gain"] == []
