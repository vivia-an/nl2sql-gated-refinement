import json
import math
import random

import numpy as np
import pytest

from nl2sql.analysis import critic_v5 as cv
from nl2sql.systems.ours import critic_features as cf
from nl2sql.systems.ours.learned_critic import LearnedCritic


def synth(n_db=4, n_q=30, seed=3):
    """candidates whose correctness depends on the semantic score and on the exec status (a learnable signal)."""
    rnd = random.Random(seed)
    recs = []
    for d in range(n_db):
        for q in range(n_q + d):
            qid = "bird_train:db%d:%d" % (d, q)
            prev = None
            for k in range(3):
                sem = rnd.choice([3.0, 5.0, 7.0, 9.0, 10.0])
                empty = rnd.random() < 0.2
                p = 0.1 + 0.08 * sem - (0.25 if empty else 0.0)
                y = int(rnd.random() < min(max(p, 0.02), 0.97))
                syn = 8.0 if empty else 10.0
                score = round(0.3 * syn + 0.5 * sem + 0.2 * 10.0, 4)
                recs.append({"qid": qid, "gid": "c0|" + qid, "db_id": "db%d" % d, "cand_idx": k, "sql": "SELECT a FROM t%d WHERE b = %d" % (d, q + k),
                             "syntax": syn, "semantic": sem, "efficiency": 10.0, "score": score,
                             "status": "ok_empty" if empty else "ok_nonempty", "n_rows": 0 if empty else rnd.randint(1, 20), "n_cols": 1,
                             "top_sim": rnd.random(), "mean_ws": 6.3, "k": k, "tier": 0, "action": "root" if k == 0 else "revise",
                             "prev_score": prev, "changed": None if k == 0 else True, "y": y})
                prev = score
    return recs


def test_sql_features_known_query():
    f = cf.sql_features("SELECT T1.a, COUNT(*) FROM x AS T1 INNER JOIN y AS T2 ON T1.id = T2.id WHERE T2.n LIKE 'a%' AND T1.v > 3 "
                        "GROUP BY T1.a ORDER BY 2 DESC LIMIT 1")
    assert f["n_joins"] == 1 and f["n_tables_sql"] == 2 and f["n_agg"] == 1 and f["has_group"] == 1 and f["has_order"] == 1
    assert f["has_limit"] == 1 and f["has_like"] == 1 and f["n_str_lit"] == 1 and f["n_num_lit"] >= 1 and f["n_bool_conn"] == 1
    assert f["n_subq"] == 0
    assert cf.sql_features("SELECT (SELECT max(v) FROM z) FROM y")["n_subq"] == 1
    assert cf.sql_features("")["n_joins"] == 0                       # empty sql does not raise


def test_record_features_and_names_consistent():
    r = synth(1, 2)[0]
    x = cf.record_features(r)
    names = cf.feature_names()
    assert set(names) <= set(x) | set(cf.SET_FEATURES) and len(set(names)) == len(names)
    assert len(cf.vectorize(x, names)) == len(names)
    assert math.isnan(x["plan_scans"])                                 # not recorded -> NaN, not 0
    assert x["st_ok_empty"] + x["st_ok_nonempty"] == 1 and x["is_empty"] in (0.0, 1.0)


def test_snake_folds_rule():
    counts = {"a": 50, "b": 40, "c": 30, "d": 20, "e": 10, "f": 9, "g": 8, "h": 7}
    f = cv.snake_folds(counts, 4)
    assert [f[d] for d in "abcdefgh"] == [0, 1, 2, 3, 3, 2, 1, 0]
    assert sorted(set(f.values())) == [0, 1, 2, 3]


def test_ece_perfect_and_biased():
    rnd = np.random.RandomState(0)
    p = rnd.rand(6000)
    y = (rnd.rand(6000) < p).astype(int)
    e, rel = cv.ece_equal_mass(p, y)
    assert e < 0.04 and len(rel) == 15
    e2, _ = cv.ece_equal_mass(np.clip(p + 0.2, 0, 1), y)
    assert e2 > 0.12


def test_grouped_cv_learns_signal_and_never_mixes_dbs(tmp_path):
    recs = synth()
    rep = cv.run(recs, str(tmp_path), models=("lr", "hgb"), feature_sets=("live", "set"), k=4, seed=0, boot=60)
    assert rep["n_dbs"] == 4 and sorted(rep["folds"]) == [0, 1, 2, 3]
    assert all(len(v) == 1 for v in rep["fold_dbs"].values())            # one db per fold -> held-out db never seen in training
    for key in ("live/lr", "live/hgb", "set/hgb"):
        m = rep["models"][key]
        assert m["auroc"] > 0.6 and 0 <= m["ece_raw"] <= 1 and "ece_cal" in m and m["auroc_ci"][0] is not None
        assert m["selection"]["top1"] <= m["selection"]["oracle"]
    assert rep["models"]["v3_score"]["auroc"] > 0.6
    rows = [json.loads(l) for l in open(tmp_path / "oof.jsonl")]
    assert len(rows) == len(recs) and "live/hgb+iso" in rows[0]


def test_export_roundtrip_and_live_feature_parity(tmp_path):
    recs = synth()
    out = tmp_path / "critic.pkl"
    cv.run(recs, str(tmp_path), models=("hgb",), feature_sets=("live",), k=4, boot=20, export="hgb", export_path=str(out))
    lc = LearnedCritic.load(str(out))
    assert lc.b["model_name"] == "hgb" and lc.names == cf.feature_names()
    p = lc.p_correct(recs[0])
    assert 0.0 <= p <= 1.0
    X, _ = cv.build_matrix(recs[:5], lc.names, False)
    assert np.allclose(lc.predict_matrix(X)[0], p)
    # live path: a record built from pipeline-like objects equals the offline record feature-wise
    class Ex:
        WS = 6.3
    class Ctx:
        tables = ["t0"]; ddl = "CREATE TABLE t0 (a int)"; ex_sims = [0.4, 0.2]; examples = [Ex(), Ex()]; full_schema = False
    class Item:
        question = "how many rows are there in the table"; evidence = "e"
    res = {"ok": True, "rows": [(1,)] * 7, "cols": ["c"]}
    crit = {"syntax": 10.0, "semantic": 8.0, "efficiency": 10.0, "score": 9.0, "status": "ok_nonempty"}
    qf = {"q_tokens": 0.1, "n_objects": 0.2, "has_time": 0.0, "n_tables": 0.05, "n_columns": 0.1, "has_evidence": 1.0}
    live = cf.live_record(Item, "SELECT a FROM t0", res, crit, Ctx, ["SCAN t0", "SEARCH t1 USING INDEX i (x=?)"],
                          {"prev_score": 6.0, "changed": True, "k": 1, "tier": 0, "action": "revise", "qf": qf})
    fx = cf.record_features(live)
    assert fx["plan_scans"] == 1 and fx["plan_search"] == 1 and fx["d_score_prev"] == 3.0 and fx["act_revise"] == 1.0
    assert fx["top_sim"] == 0.4 and abs(fx["mean_sim"] - 0.3) < 1e-9 and fx["qf_n_tables"] == 0.05 and fx["log_rows"] == pytest.approx(math.log1p(7))
    assert 0.0 <= lc.p_correct(live) <= 1.0


def test_trees_loader_maps_state(tmp_path):
    state = [0.0] * 27
    state[10] = math.log1p(9) / math.log1p(1000)
    state[11] = 0.3
    state[16], state[17], state[18], state[19], state[22] = 0.5, 0.4, 0.63, 0.2, 0.15
    tree = {"qid": "bird_train:d:1", "db_id": "d", "fold": 2, "nodes": [
        {"id": 0, "parent": None, "action": "root", "sql": "select 1", "tier": 0, "k": 0, "state": state, "ex": 1, "score": 9.0,
         "syntax": 10.0, "semantic": 9.0, "efficiency": 10.0, "status": "ok_nonempty"},
        {"id": 1, "parent": 0, "action": "revise", "sql": "select 2", "tier": 0, "k": 1, "state": state, "ex": 0, "score": 7.0,
         "syntax": 10.0, "semantic": 5.0, "efficiency": 10.0, "status": "ok_nonempty"}]}
    recs = cv.recs_from_trees([tree])
    assert len(recs) == 2 and recs[0]["fold"] == 2 and abs(recs[0]["n_rows"] - 9) < 1e-6 and recs[0]["n_cols"] == pytest.approx(3.0)
    assert recs[1]["prev_score"] == 9.0 and recs[0]["prev_score"] is None and recs[1]["y"] == 0
    assert recs[0]["mean_ws"] == pytest.approx(6.3)


def test_m5_critic_v5_branch(tmp_path):
    """Critic(variant v5) returns 10*p and keeps score_v3; V3 path unchanged (no p_correct)."""
    from nl2sql import config as C
    from nl2sql.systems.ours.defaults import DEFAULTS
    from nl2sql.systems.ours.m5 import Critic
    out = tmp_path / "c.pkl"
    cv.run(synth(), str(tmp_path), models=("lr",), feature_sets=("live",), boot=10, export="lr", export_path=str(out))

    class FakeLLM:
        def chat(self, *a, **k):
            return {"text": '{"semantic": 8, "issues": [], "suggestions": []}', "prompt_tokens": 1, "completion_tokens": 1}

    class Ctx:
        tables = ["t"]; ddl = "CREATE TABLE t (a int)"; lineage = ""; ex_sims = [0.3]; examples = []; full_schema = True
        def sections_core(self):
            return "# Schema\n" + self.ddl

    class Item:
        question = "q"; evidence = ""; db_path = str(tmp_path / "none.sqlite")

    cfg = C.deep_update(DEFAULTS, {"m5": {"variant": "v5", "learned": {"path": str(out)}}})
    res = {"ok": True, "rows": [(1,)], "cols": ["a"], "error": None, "truncated": False, "elapsed_s": 0.0}
    c5 = Critic(FakeLLM(), "m", cfg).score(Item, "SELECT a FROM t", Ctx(), None, res, 0, {}, meta={"k": 0, "action": "root"})
    assert 0 <= c5["score"] <= 10 and abs(c5["score"] - 10 * c5["p_correct"]) < 1e-3 and "score_v3" in c5
    c3 = Critic(FakeLLM(), "m", C.deep_update(DEFAULTS, {})).score(Item, "SELECT a FROM t", Ctx(), None, res, 0, {})
    assert "p_correct" not in c3 and abs(c3["score"] - (0.3 * 10 + 0.5 * 8 + 0.2 * 10)) < 1e-6
