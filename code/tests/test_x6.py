"""X6 tests (offline: scripted fake LLM, toy sqlite databases, synthetic preds; no API call, no dev / Spider data).
Covers: baselines honour ddl.quote_keywords (legacy byte-identical when off), full_gated_blind (P5 control), stream --checkpoints,
tools/selection_rules.py (SAP 13.6 rules 3 and 4), the min_* configs."""
import json
import os
import sqlite3
import sys

import numpy as np
import pytest

from nl2sql import config as C, data, paths
from nl2sql.data import Item
from nl2sql.systems import get_system
from nl2sql.systems.baselines import _common as BC, mac_sql as MAC, dail_sql as DAIL, memo_sql as MEMO
from nl2sql.systems.ours import m5 as M5, pipeline as PL
from test_x5 import BAD, GOOD, ScriptLLM, make_sys, mk_item, toydb  # noqa: F401  (toydb is a fixture)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import selection_rules as SR  # noqa: E402

CONFIGS = paths.EXP_DIR / "configs"


# ------------------------------------------------------------------------------------------------ 1. baselines + ddl.quote_keywords
class RecLLM:
    """Records every chat request; answers with a valid SQL block (n > 1 -> list)."""

    def __init__(self):
        self.calls = []

    def chat(self, model, messages, *, temperature=0.0, seed=None, max_tokens=1024, n=1, tag="", **kw):
        self.calls.append({"tag": tag, "text": "\n".join(m["content"] for m in messages)})
        t = '```sql\nSELECT count(*) FROM "transaction"\n```'
        return {"text": t if n == 1 else [t] * n, "prompt_tokens": 1, "completion_tokens": 1, "latency_s": 0.0}

    def embed(self, texts, model):  # pragma: no cover - not used by the systems under test
        raise AssertionError("embedding call")

    def prompts(self):
        return [c["text"] for c in self.calls]


@pytest.fixture()
def schema_patch(monkeypatch, toydb):
    sch = data._build_schema(toydb, "toy", "t")
    monkeypatch.setattr(BC, "get_schema_cached", lambda item: sch)
    MAC._DBINFO.clear()
    yield sch
    MAC._DBINFO.clear()


SYSTEMS = [("zero_shot", {}), ("bon_exec_vote", {"n": 2}), ("self_debug", {"max_rounds": 2, "feedback": "simple"}), ("ac_sql", {"max_regen": 1}),
           ("mac_sql", {"max_refine": 1})]


def _run(name, cfg, toydb):
    llm = RecLLM()
    system = get_system(name)(llm, "m", dict(cfg))
    row = system.run(mk_item(toydb), seed=0)
    return llm, row


@pytest.mark.parametrize("name,cfg", SYSTEMS)
def test_baselines_show_quoted_identifiers_only_when_the_flag_is_on(name, cfg, toydb, schema_patch):
    if name == "mac_sql":
        pytest.importorskip("nl2sql.systems.baselines._vendor.macsql_const")      # not redistributed: python code/tools/fetch_macsql_const.py
    off, _ = _run(name, cfg, toydb)
    off2, _ = _run(name, dict(cfg, ddl={"quote_keywords": False}), toydb)
    on, _ = _run(name, dict(cfg, ddl={"quote_keywords": True}), toydb)
    assert off.calls and on.calls
    assert off.prompts() == off2.prompts()                                  # flag false = flag absent = legacy prompts, byte for byte
    schema_prompts_off = [p for p in off.prompts() if "transaction" in p and "CREATE TABLE" in p or "# Table:" in p]
    assert schema_prompts_off, name
    for p in schema_prompts_off:
        assert '# Table: "transaction"' not in p and 'CREATE TABLE "transaction"' not in p
    first_off, first_on = off.prompts()[0], on.prompts()[0]
    assert first_on != first_off
    if name == "mac_sql":
        assert "# Table: transaction" in first_off and '# Table: "transaction"' in first_on and '("order",' in first_on and "(order," in first_off
    else:
        assert "CREATE TABLE transaction (" in first_off and 'CREATE TABLE "transaction" (' in first_on
        assert '"order" TEXT' in first_on and ' order TEXT' in first_off
    # nothing but the quoting differs: strip the added quotes from the quoted prompt and the legacy prompt comes back
    back = first_on
    for w in ("transaction", "order"):
        back = back.replace('"%s"' % w, w)
    assert back == first_off


def test_schema_text_legacy_is_the_stored_ddl_and_quoted_is_a_pure_rerender(toydb, schema_patch):
    it = mk_item(toydb)
    assert BC.schema_text(it) == schema_patch["ddl"] == BC.schema_text(it, quote_keywords=False)
    q = BC.schema_text(it, quote_keywords=True)
    assert q == data.render_ddl(schema_patch, quote_keywords=True) and q != schema_patch["ddl"]
    assert BC.raw_ddl(it) == data.render_ddl(schema_patch, samples=False, descriptions=False)
    assert 'CREATE TABLE "transaction"' in BC.raw_ddl(it, quote_keywords=True)
    con = sqlite3.connect(":memory:")                                       # the quoted schema text is valid SQLite
    con.executescript(BC.raw_ddl(it, quote_keywords=True))
    assert BC.quote_flag(None) is False and BC.quote_flag({}) is False and BC.quote_flag({"ddl": {"quote_keywords": True}}) is True
    assert BC.qname("transaction", True) == '"transaction"' and BC.qname("transaction", False) == "transaction" and BC.qname("amt", True) == "amt"


def test_dail_and_memo_pass_the_flag(toydb, schema_patch, monkeypatch):
    it = mk_item(toydb)
    assert 'CREATE TABLE "transaction"' in DAIL.ddl_of(it, "rich", True) and 'CREATE TABLE transaction' in DAIL.ddl_of(it, "rich")
    assert 'CREATE TABLE "transaction"' in DAIL.ddl_of(it, "plain", True) and 'CREATE TABLE transaction' in DAIL.ddl_of(it, "plain")
    ex = Item(qid="toy:t:2", db_id="t", question="q", evidence="", gold_sql="SELECT 1", db_path=toydb, split="toy", meta={})
    assert 'CREATE TABLE "transaction"' in DAIL.example_block(ex, "full", "rich", True) and 'CREATE TABLE transaction' in DAIL.example_block(ex, "full", "rich")
    seen = []
    real = MEMO.schema_text

    def spy(item, **kw):
        seen.append(kw.get("quote_keywords", False))
        return real(item, **kw)
    monkeypatch.setattr(MEMO, "schema_text", spy)
    for flag in (False, True):
        assert MEMO.MemoSQL(RecLLM(), "m", {"ddl": {"quote_keywords": flag}, "preset": "light"}).qk is flag
    s = MEMO.MemoSQL(RecLLM(), "m", {"ddl": {"quote_keywords": True}, "preset": "light"})
    try:
        s.solve(mk_item(toydb), 0, BC.Ctx(s, mk_item(toydb), 0))
    except Exception:
        pass                                                               # memory file / embeddings are not available here; only the schema call matters
    assert seen and seen[-1] is True


def test_min_baseline_configs_turn_quoting_on_and_zero_config_keeps_it_off():
    base = ["min_zs", "min_sdebug", "min_acsql", "min_macsql", "min_bon_vote", "min_bon_vote_n3", "min_memosql_light"]
    base += sorted(p.stem for p in CONFIGS.glob("min_tune_*.yaml"))
    assert len(base) == 7 + 11
    for n in base:
        cfg = C.load_config(CONFIGS / (n + ".yaml"))
        assert (cfg.get("ddl") or {}).get("quote_keywords") is True, n
    for n in ("zs",):                                                      # pre-existing (locked-to-be) config: untouched
        assert "ddl" not in C.load_config(CONFIGS / (n + ".yaml"))
    # a baseline built from a min_* config really reads the flag
    for n, sysname in (("min_sdebug", "sdebug"), ("min_acsql", "acsql_s"), ("min_macsql", "macsql"), ("min_bon_vote", "bon_vote")):
        assert get_system(sysname)(RecLLM(), "m", C.load_config(CONFIGS / (n + ".yaml"))).qk is True


# ------------------------------------------------------------------------------------------------ 2. full_gated_blind (P5)
GATED_CFG = {"ddl": {"quote_keywords": True}, "route": {"score": "exec", "theta_h": 7.5}}


def test_blind_variant_withholds_the_error_text_but_keeps_gate_and_call_count(monkeypatch, toydb):
    item = mk_item(toydb)
    la, lb = ScriptLLM(gen=BAD, rev=GOOD), ScriptLLM(gen=BAD, rev=GOOD)
    oa = make_sys(monkeypatch, PL.FullGated, la, dict(GATED_CFG)).run(item)
    ob = make_sys(monkeypatch, PL.FullGatedBlind, lb, dict(GATED_CFG)).run(item)
    assert la.tags() == lb.tags() == ["gen", "rev"]                        # identical gate / call count
    assert oa["actions"] == ob["actions"] and oa["final_sql"] == ob["final_sql"] == GOOD
    ra, rb = la.calls[1]["user"], lb.calls[1]["user"]
    assert 'near "transaction": syntax error' in ra and "Execution error:" in ra       # control: the normal variant shows the error
    assert "syntax error" not in rb and "Execution error" not in rb and "near " not in rb and "transaction\n" not in rb.split("# Execution")[1]
    assert M5.BLIND_FEEDBACK in rb.split("# Execution feedback")[1] and M5.BLIND_PREVIEW in rb
    assert "Reviewer feedback" not in rb
    # everything before "# Execution" (schema, question, previous SQL) is identical: only the execution blocks differ
    assert ra.split("# Execution\n")[0] == rb.split("# Execution\n")[0]
    assert BAD in rb                                                       # the previous SQL is still shown


def test_blind_variant_is_blind_in_every_round_and_default_stays_unchanged(monkeypatch, toydb):
    item = mk_item(toydb)
    llm = ScriptLLM(gen=BAD, rev=lambda u, n, k: "SELECT nope%d FROM missing_table" % k)      # keeps failing until t_max
    out = make_sys(monkeypatch, PL.FullGatedBlind, llm, dict(GATED_CFG)).run(item)
    revs = [c["user"] for c in llm.calls if c["tag"] == "rev"]
    assert len(revs) == 3 and out["actions"].count("revise") == 3
    assert all("no such table" not in r and "syntax error" not in r and M5.BLIND_FEEDBACK in r for r in revs)
    # default feedback mode of every config / system is "exec" (existing prompts unchanged)
    assert PL.resolved_cfg({})["m5"]["revise_feedback"] == "exec"
    assert make_sys(monkeypatch, PL.FullGated, ScriptLLM(), dict(GATED_CFG)).pipe.blind is False
    assert PL.FullGated.cfg_override == {"m5": {"policy": "gated"}}


def test_blind_variant_validation_registry_and_config(monkeypatch):
    with pytest.raises(ValueError, match="revise_feedback"):
        make_sys(monkeypatch, PL.FullGated, ScriptLLM(), {"m5": {"revise_feedback": "nope"}, "route": {"score": "exec"}})
    with pytest.raises(ValueError, match="gated"):
        make_sys(monkeypatch, PL.FullFixed, ScriptLLM(), {"m5": {"revise_feedback": "blind"}})
    assert get_system("full_gated_blind") is PL.FullGatedBlind and PL.FullGatedBlind.force_p0
    s = make_sys(monkeypatch, PL.FullGatedBlind, ScriptLLM(), C.load_config(CONFIGS / "min_gated_blind.yaml"))
    assert s.pipe.gated and s.pipe.blind and s.pipe.qk
    # min_gated_blind == min_full_gated except the revise_feedback key
    a, b = C.load_config(CONFIGS / "min_full_gated.yaml"), C.load_config(CONFIGS / "min_gated_blind.yaml")
    assert b["m5"].pop("revise_feedback") == "blind" and a == b
    # the blind system alone (no config key) is blind as well
    s2 = make_sys(monkeypatch, PL.FullGatedBlind, ScriptLLM(), C.load_config(CONFIGS / "min_full_gated.yaml"))
    assert s2.pipe.blind


# ------------------------------------------------------------------------------------------------ 3. stream --checkpoints
def test_parse_checkpoints():
    from nl2sql import stream
    assert stream.parse_checkpoints(None) == (0, 40, 100) == stream.DEFAULT_CHECKPOINTS
    assert stream.parse_checkpoints("0,40,100") == (0, 40, 100) and stream.parse_checkpoints("100, 0") == (0, 100)
    assert stream.parse_checkpoints("0,25,50,100") == (0, 25, 50, 100) and stream.parse_checkpoints("0,12.5,100") == (0, 12.5, 100)
    for bad in ("0,40", "40,100", "", "0,x,100", "0,120,100", "-5,0,100"):
        with pytest.raises(ValueError):
            stream.parse_checkpoints(bad)


def _toy_stream_items(toydb, n_stream=10, n_probe=4):
    items = []
    for i in range(3 + n_stream + n_probe):
        role = "seed" if i < 3 else ("stream" if i < 3 + n_stream else "probe")
        items.append(Item(qid="toy:t0:%d" % i, db_id="t0", question="q", evidence="", gold_sql="select 1", db_path=toydb, split="x", meta={"role": role}))
    return items


def test_stream_unit_probes_follow_checkpoints_and_default_is_unchanged(tmp_path, monkeypatch, toydb):
    from nl2sql import stream
    import test_x4_runner as T4
    T4._patch_unit(monkeypatch, _toy_stream_items(toydb))
    cfg = {"m6": {"params": {}}}

    def go(name, **kw):
        T4._FakeSys.boom_at, T4._FakeSys.calls = None, 0
        assert stream.run_unit(None, cfg, T4.MODEL, "t0", "ws_full", 0, 1, str(tmp_path / name), {}, workers=1, **kw)[1] == "ok"
        d = tmp_path / name / "ws_full-p00-perm1-t0"
        evs = [json.loads(l) for l in open(d / "events.jsonl")]
        return [e["after"] for e in evs if e["type"] == "probe"], json.load(open(d / "pollution.json")), len([e for e in evs if e["type"] == "stream"])
    after, meta, n_st = go("default")
    assert after == [0, 4, 10] and n_st == 10 and meta["checkpoints_pct"] == [0, 40, 100]          # round(0.4 * 10) = 4: the registered protocol
    assert go("explicit", checkpoints=(0, 40, 100))[0] == [0, 4, 10]
    after2, meta2, _ = go("two", checkpoints=stream.parse_checkpoints("0,100"))
    assert after2 == [0, 10] and meta2["checkpoints_pct"] == [0, 100]
    assert go("four", checkpoints=(0, 25, 50, 100))[0] == [0, 2, 5, 10]                           # round(2.5) = 2 (banker's rounding, as in the default)


def test_summarize_trial_counts_probe_runs_by_number_of_checkpoints(tmp_path):
    from nl2sql import stream
    for ck, n_runs in (([0, 4, 10], 3), ([0, 10], 2)):
        d = tmp_path / ("c%d" % len(ck)) / "ws_full-p00-perm1-t0"
        os.makedirs(d)
        with open(d / "events.jsonl", "w") as f:
            for i in range(10):
                f.write(json.dumps({"type": "stream", "i": i, "ex": 1, "abstained": False, "calls": 2}) + "\n")
            for a in ck:
                f.write(json.dumps({"type": "probe", "after": a, "items": [{"ex_all": 1, "calls": 3}] * 4, "diag": {}}) + "\n")
        metric, per_db, n = stream.summarize_trial(str(tmp_path / ("c%d" % len(ck))), ["t0"], "ws_full", 0, 1)
        assert metric["calls_per_q"] == round((10 * 2 + n_runs * 4 * 3) / (10 + n_runs * 4), 3), ck


def test_stream_cli_has_checkpoints_option():
    src = open(os.path.join(os.path.dirname(stream_file()), "stream.py")).read()
    assert '"--checkpoints"' in src and "must include 0 and 100" in src


def stream_file():
    from nl2sql import stream
    return stream.__file__


# ------------------------------------------------------------------------------------------------ 4. selection_rules.py
def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _paired_rows(dbs, n_per_db, p_ex, flip_gated, calls_fixed, calls_gated, seed, tag):
    """-> (fixed_rows, gated_rows) on the same qids. gated differs from fixed on a `flip_gated` share of items: -1 = right->wrong, +1 = wrong->right."""
    rng = np.random.RandomState(seed)
    fx, gt = [], []
    for d in dbs:
        for i in range(n_per_db):
            ex = int(rng.rand() < p_ex)
            ex_g = ex
            u = rng.rand()
            if flip_gated < 0 and ex == 1 and u < -flip_gated:
                ex_g = 0
            if flip_gated > 0 and ex == 0 and u < flip_gated:
                ex_g = 1
            q = "bird:%s:%s%d" % (d, tag, i)
            fx.append({"qid": q, "db_id": d, "ex": ex, "calls": calls_fixed + int(rng.rand() < 0.2)})
            gt.append({"qid": q, "db_id": d, "ex": ex_g, "calls": calls_gated + int(rng.rand() < 0.2)})
    return fx, gt


def _scenario(tmp_path, name, hr_flip, hr_calls, ht_flip, ht_calls, **kw):
    d = tmp_path / name
    hf, hg = _paired_rows(["a", "b", "c", "d", "e", "f"], 60, 0.65, hr_flip, hr_calls[0], hr_calls[1], 1, "hr")
    tf, tg = _paired_rows(["w", "x", "y", "z"], 75, 0.65, ht_flip, ht_calls[0], ht_calls[1], 2, "ht")
    paths_ = {}
    for nm, rows in (("hr_fixed", hf), ("hr_gated", hg), ("ht_fixed", tf), ("ht_gated", tg)):
        _write(str(d / nm / "preds.jsonl"), rows)
        paths_[nm] = str(d / nm)
    return paths_


def test_rule3_gated_when_cheaper_and_not_worse(tmp_path):
    p = _scenario(tmp_path, "ok", +0.01, (3, 2), 0.0, (3, 2))
    res = SR.decide(p["hr_fixed"], p["hr_gated"], p["ht_fixed"], p["ht_gated"], B=2000)
    assert res["rule3"] == "gated" and res["full"] == "full_gated" and all(res["rule3_detail"]["conditions"].values())
    assert res["rule3_detail"]["hr"]["ratio"] < 0.8 and res["rule4_p3"] is None and res["route_ca_adopted"] is None
    assert abs(res["b_ref_calls"] - res["b_ref_calls_gated"]) < 1e-9 and res["b_ref_calls"] < res["b_ref_calls_fixed"]
    assert abs(res["p2_calls_ceiling"] - 1.15 * res["b_ref_calls"]) < 2e-4 and res["m_family"] == 5


def test_rule3_each_condition_can_fail_alone(tmp_path):
    # (i) accuracy loss: 8 % of right fixed answers become wrong -> HR lower bound far below -1.5 pp
    p = _scenario(tmp_path, "acc", -0.08, (3, 2), 0.0, (3, 2))
    r = SR.decide(p["hr_fixed"], p["hr_gated"], p["ht_fixed"], p["ht_gated"], B=2000)
    assert r["rule3"] == "fixed" and r["rule3_detail"]["conditions"] == {"i_hr_dEX_lower95_gt_-1.5pp": False, "ii_hr_ratio_upper95_lt_0.80": True,
                                                                         "iii_ht_point_dEX_ge_-2.0pp_and_ratio_lt_0.80": True}
    assert r["full"] == "full_fixed" and r["b_ref_calls"] == r["b_ref_calls_fixed"] and r["fallback_note"]
    # (ii) calls ratio 0.9 on HR (3 -> 2.7 is not enough): R upper bound >= 0.80
    p = _scenario(tmp_path, "ratio", 0.0, (10, 9), 0.0, (3, 2))
    r = SR.decide(p["hr_fixed"], p["hr_gated"], p["ht_fixed"], p["ht_gated"], B=2000)
    c = r["rule3_detail"]["conditions"]
    assert r["rule3"] == "fixed" and c["i_hr_dEX_lower95_gt_-1.5pp"] and not c["ii_hr_ratio_upper95_lt_0.80"] and c["iii_ht_point_dEX_ge_-2.0pp_and_ratio_lt_0.80"]
    # (iii) HT: 4 % of the right answers lost on HT only (point estimate below -2 pp) while HR is fine
    p = _scenario(tmp_path, "ht", 0.0, (3, 2), -0.05, (3, 2))
    r = SR.decide(p["hr_fixed"], p["hr_gated"], p["ht_fixed"], p["ht_gated"], B=2000)
    c = r["rule3_detail"]["conditions"]
    assert r["rule3"] == "fixed" and c["i_hr_dEX_lower95_gt_-1.5pp"] and c["ii_hr_ratio_upper95_lt_0.80"] and not c["iii_ht_point_dEX_ge_-2.0pp_and_ratio_lt_0.80"]
    assert r["rule3_detail"]["ht"]["diff_pp"] < -2.0


def test_rule3_bootstrap_matches_paired_compare_and_handles_nulls(tmp_path):
    import paired_compare as PC
    p = _scenario(tmp_path, "pc", +0.02, (3, 2), 0.0, (3, 2))
    rows_f = SR.load_rows(p["hr_fixed"])
    rows_g = SR.load_rows(p["hr_gated"])
    q0 = sorted(rows_f)[0]
    rows_g[q0]["ex"] = None                                               # a null in either run drops the pair
    pr, drop = SR.pair(rows_f, rows_g)
    assert drop["ex_null"] == 1 and q0 not in pr["qids"] and len(pr["qids"]) == len(rows_f) - 1
    b = SR.boot_ex_and_ratio(pr, 4000)
    ref = PC.cluster_ci(pr["db"], pr["ex_g"] - pr["ex_f"], 4000)           # same quantity, same resampling scheme (different RNG draw order)
    assert abs(b["diff_pp_ci95"][0] - 100 * ref[0]) < 0.6 and abs(b["diff_pp_ci95"][1] - 100 * ref[1]) < 0.6
    pt = SR.point(pr)
    assert abs(pt["diff_pp"] - 100 * float((pr["ex_g"] - pr["ex_f"]).mean())) < 1e-12 and abs(pt["ratio"] - pr["c_g"].sum() / pr["c_f"].sum()) < 1e-12
    # ratio CI brackets the point estimate; one-sided bounds lie inside the two-sided interval
    assert b["ratio_ci95"][0] <= pt["ratio"] <= b["ratio_ci95"][1] and b["ratio_ci95"][0] <= b["ratio_hi_onesided95"] <= b["ratio_ci95"][1]


def _cons_rows(dbs, n_per_db, informative, seed, tag):
    rng = np.random.RandomState(seed)
    rows = []
    for d in dbs:
        for i in range(n_per_db):
            ex = int(rng.rand() < 0.65)
            v3 = float(np.clip(rng.normal(7.0 + 1.2 * ex, 1.8), 0, 10))                  # weakly informative
            if informative:
                bits = [int(rng.rand() < (0.92 if ex else 0.35)) for _ in range(4)]
            else:
                bits = [int(rng.rand() < 0.6) for _ in range(4)]
            rows.append({"qid": "bird:%s:%s%d" % (d, tag, i), "db_id": d, "ex": ex, "calls": 3, "critic": {"score": v3, "semantic": 8, "syntax": 10},
                         "consistency": float(np.mean(bits)), "cons_bits": bits, "final_status": "ok_nonempty" if rng.rand() < 0.95 else "ok_empty"})
    return rows


def test_rule4_keeps_a_useful_consistency_score_and_drops_noise(tmp_path):
    p = _scenario(tmp_path, "r3", 0.0, (3, 2), 0.0, (3, 2))
    for name, informative, expect in (("good", True, "keep"), ("noise", False, "drop")):
        d = tmp_path / ("cons_" + name)
        _write(str(d / "hr" / "preds.jsonl"), _cons_rows(list("abcdef"), 90, informative, 3, "hr"))
        _write(str(d / "ht" / "preds.jsonl"), _cons_rows(list("wxyz"), 75, informative, 4, "ht"))
        res = SR.decide(p["hr_fixed"], p["hr_gated"], p["ht_fixed"], p["ht_gated"], str(d / "hr"), str(d / "ht"), k=2, B=1500)
        assert res["rule4_p3"] == expect, res["rule4_detail"]
        h = res["rule4_detail"]["hr"]
        assert (res["route_ca_adopted"] is True) == (expect == "keep") and res["m_family"] == (5 if expect == "keep" else 4)
        assert "ht_heldout_info" in res["rule4_detail"] and h["n_databases"] == 6
        if expect == "keep":
            assert h["dAUROC"] >= 0.03 and h["dAUROC_ci95"][0] > 0 and h["dAcc60_pp"] >= 2.0
        else:
            assert not (h["dAUROC"] >= 0.03 and h["dAUROC_ci95"][0] > 0)


def test_rule4_uses_only_the_first_k_consistency_bits_and_each_condition_matters(tmp_path):
    rows = _cons_rows(list("abcdef"), 90, True, 5, "hr")
    # make bits beyond the first two pure noise: k=2 sees the signal, and a row set where the first two are noise but later ones informative gives drop at k=2
    rng = np.random.RandomState(9)
    noisy_first = []
    for r in rows:
        b = list(r["cons_bits"])
        noisy_first.append(dict(r, cons_bits=[int(rng.rand() < 0.6), int(rng.rand() < 0.6)] + b[2:], consistency=None))
    d = tmp_path / "k"
    _write(str(d / "a" / "preds.jsonl"), rows)
    _write(str(d / "b" / "preds.jsonl"), noisy_first)
    p = _scenario(tmp_path, "r3k", 0.0, (3, 2), 0.0, (3, 2))
    args = (p["hr_fixed"], p["hr_gated"], p["ht_fixed"], p["ht_gated"])
    assert SR.decide(*args, hr_cons=str(d / "a"), k=2, B=1000)["rule4_p3"] == "keep"
    assert SR.decide(*args, hr_cons=str(d / "b"), k=2, B=1000)["rule4_p3"] == "drop"
    assert SR.decide(*args, hr_cons=str(d / "b"), k=4, B=1000)["rule4_detail"]["k"] == 4
    # too few rows -> refuses instead of guessing
    _write(str(d / "tiny" / "preds.jsonl"), rows[:10])
    with pytest.raises(SystemExit):
        SR.decide(*args, hr_cons=str(d / "tiny"), k=2, B=100)


def test_selection_rules_cli_writes_the_decisions_file(tmp_path):
    p = _scenario(tmp_path, "cli", +0.01, (3, 2), 0.0, (3, 2))
    d = tmp_path / "cli_cons"
    _write(str(d / "hr" / "preds.jsonl"), _cons_rows(list("abcdef"), 90, True, 3, "hr"))
    out = tmp_path / "exp" / "selection_decisions.json"
    res = SR.main(["--hr-fixed", p["hr_fixed"], "--hr-gated", p["hr_gated"], "--ht-fixed", p["ht_fixed"], "--ht-gated", p["ht_gated"],
                   "--hr-cons", str(d / "hr"), "--boot", "1000", "--out", str(out)])
    j = json.load(open(out))
    for k in ("rule3", "rule4_p3", "b_ref_calls", "decided", "full", "route_ca_adopted", "m_family"):      # the freeze checklist (not released) step 2 + analysis_pipeline.runs.load_decisions
        assert k in j, k
    assert j["rule3"] in ("gated", "fixed") and j["rule4_p3"] in ("keep", "drop") and j["full"] == "full_" + j["rule3"]
    assert j["rule3"] == res["rule3"] and isinstance(j["b_ref_calls"], float) and len(j["decided"]) == 10
    assert {i["name"] for i in j["inputs"]} == {"hr_fixed", "hr_gated", "ht_fixed", "ht_gated", "hr_cons"} and all(len(i["sha256"]) == 64 for i in j["inputs"])
    # lock_gate --require only checks that the file exists; the file loads like the analysis pipeline expects
    assert json.load(open(out))["rule3"] == j["rule3"]
    # a preds.jsonl path works as well as a directory
    assert SR.preds_path(p["hr_fixed"]).endswith("preds.jsonl") and SR.preds_path(os.path.join(p["hr_fixed"], "preds.jsonl")).endswith("preds.jsonl")
