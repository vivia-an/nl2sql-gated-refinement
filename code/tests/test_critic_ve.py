"""Offline tests of the user-dispute / appeal diagnostic (tools/critic_ve.py, SAP Amendment 6d / section 19).  No LLM call, no database, no network."""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import critic_ve as E  # noqa: E402


def rec(rid, cat="where_drop", db="d1", pert_trig=True, pert_fix_ok=True, gold=True, gold_trig=False, gold_fix_bag=0):
    def crit(trig):
        return {"score": 4.0 if trig else 9.0, "semantic_raw": 4.0 if trig else 9.0,
                "issues": [{"question_phrase": "most", "sql_fragment": "ORDER BY x", "description": "wrong order (question: \"most\"; SQL: \"ORDER BY x\")"}] if trig else [],
                "style_issues": ([] if trig else [{"description": "alias style"}])}
    ck = "ex_bag" if cat == "distinct_drop" else "ex_set"
    v = {"pert": {"sql": "SELECT 1", "triggered": pert_trig, "critic": crit(pert_trig), "root_res": {"ex_set": 0, "ex_bag": 0},
                  "rounds": [{"sql": "SELECT 2", "noop": False, "score_res": {"ex_set": int(pert_fix_ok), "ex_bag": int(pert_fix_ok)}}] if pert_trig else []}}
    if gold:
        v["gold"] = {"sql": "SELECT 3", "triggered": gold_trig, "critic": crit(gold_trig), "root_res": {"ex_set": 1, "ex_bag": 1},
                     "rounds": [{"sql": "SELECT 4", "noop": False, "score_res": {"ex_set": gold_fix_bag, "ex_bag": gold_fix_bag}}] if gold_trig else []}
    return {"id": rid, "cat": cat, "qid": "q" + rid, "db_id": db, "variants": v}


ITEMS = {r: {"id": r, "question": "Q " + r, "evidence": "E"} for r in ("a", "b", "c", "d")}


def synth():
    recs = {"a": rec("a", pert_trig=True, pert_fix_ok=True, gold_trig=True, db="d1"),       # F-inj right, F-gold wrong
            "b": rec("b", pert_trig=True, pert_fix_ok=False, gold_trig=False, db="d2"),     # F-inj wrong, P-gold
            "c": rec("c", pert_trig=False, gold=False, db="d3"),                            # P-inj
            "d": rec("d", pert_trig=False, gold_trig=False, db="d4")}                       # P-inj, P-gold
    return E.build_units(recs, ITEMS)


def test_build_units_labels():
    us = {u["uid"]: u for u in synth()}
    assert len(us) == 7
    assert us["a:pert"]["kind"] == "F" and us["a:pert"]["critic_right"] is True and us["a:pert"]["claims"][0]["description"] == "wrong order"
    assert us["b:pert"]["kind"] == "F" and us["b:pert"]["critic_right"] is False
    assert us["a:gold"]["kind"] == "F" and us["a:gold"]["critic_right"] is False                  # a flagged gold control is a wrong claim by definition
    assert us["c:pert"]["kind"] == "P" and us["c:pert"]["critic_right"] is False and us["c:pert"]["sql_wrong"] is True
    assert us["d:gold"]["kind"] == "P" and us["d:gold"]["critic_right"] is True
    assert us["d:gold"]["remarks"] == ["alias style"]


def test_build_units_asserts_injected_root_wrong():
    r = rec("a", pert_trig=False, gold=False)
    r["variants"]["pert"]["root_res"] = {"ex_set": 1, "ex_bag": 1}
    try:
        E.build_units({"a": r}, ITEMS)
        assert False
    except AssertionError:
        pass


def test_p_disputed_and_obey_none():
    assert E.p_disputed(True, 0.85) == 1 - 0.85 and E.p_disputed(False, 0.85) == 0.85
    us = synth()
    none = E.policy_flags(us, "none", 0.85)
    assert none == [1.0 if u["kind"] == "F" else 0.0 for u in us]
    # q = 1: obey the user = oracle for every unit
    assert E.policy_flags(us, "obey", 1.0) == E.policy_flags(us, "oracle_all", 1.0)
    # q = 0.5 obey: every unit flagged with prob 0.5
    assert all(abs(f - 0.5) < 1e-12 for f in E.policy_flags(us, "obey", 0.5))


def test_oracle_arbiter_and_arm_policy():
    us = synth()
    q = 0.85
    dec_hold = {u["uid"]: True for u in us}           # an arbiter that always holds == no appeal
    assert E.policy_flags(us, "A", q, dec_hold) == E.policy_flags(us, "none", q)
    dec_yield = {u["uid"]: False for u in us}         # always yields == obey
    for x, y in zip(E.policy_flags(us, "A", q, dec_yield), E.policy_flags(us, "obey", q)):
        assert abs(x - y) < 1e-12
    dec_ok = {u["uid"]: u["critic_right"] for u in us}     # a perfect arbiter
    for x, y in zip(E.policy_flags(us, "A", q, dec_ok), E.policy_flags(us, "oracle_arbiter", q)):
        assert abs(x - y) < 1e-12
    # oracle_all flags exactly: F with right claim, and wrong passes
    of = E.policy_flags(us, "oracle_all", q)
    for u, f in zip(us, of):
        assert f == (1.0 if (u["kind"] == "F" and u["critic_right"]) or (u["kind"] == "P" and not u["critic_right"]) else 0.0)
    # arm not defined for a unit (C on passes) -> no appeal
    dc = {u["uid"]: False for u in us if u["kind"] == "F"}
    pc = E.policy_flags(us, "C", q, dc)
    assert all(pc[i] == 0.0 for i, u in enumerate(us) if u["kind"] == "P")


def test_metrics_and_groups():
    us = synth()
    m = E.group_masks(us)
    assert sum(m["inj"]) == 4 and sum(m["gold"]) == 3 and sum(m["F_inj_right"]) == 1 and sum(m["F_wrong"]) == 2 and sum(m["P_inj"]) == 2 and sum(m["P_gold"]) == 2
    v = E.metric_values(E.policy_flags(us, "none", 0.85), m)
    assert abs(v["DET"] - 2 / 4) < 1e-12 and abs(v["UDET"] - 1 / 4) < 1e-12 and abs(v["FPR"] - 1 / 3) < 1e-12
    assert abs(v["J"] - (1 / 4 - 1 / 3)) < 1e-12 and v["CCR"] == 1.0 and v["WCR"] == 0.0 and v["MER"] == 0.0 and v["TNR"] == 1.0
    vo = E.metric_values(E.policy_flags(us, "oracle_all", 0.85), m)
    assert vo["CCR"] == 1.0 and vo["WCR"] == 1.0 and vo["MER"] == 1.0 and vo["TNR"] == 1.0 and vo["FPR"] == 0.0


def test_bootstrap_shapes_and_pairing():
    us = synth()
    m = E.group_masks(us)
    flags = {"none": E.policy_flags(us, "none", 0.85), "obey": E.policy_flags(us, "obey", 0.85)}
    s = E.boot_metric_samples(us, flags, m, 200, 1)
    assert len(s["none"]["J"]) == 200
    # the difference of an identical policy is exactly 0 in every resample
    flags2 = {"x": flags["none"], "y": flags["none"]}
    s2 = E.boot_metric_samples(us, flags2, m, 50, 1)
    d = s2["x"]["CCR"] - s2["y"]["CCR"]
    assert abs(d[~(d != d)]).max() == 0


def test_hold_yield_rates_and_realisation():
    us = synth()
    dec = {u["uid"]: u["critic_right"] for u in us}      # perfect arbiter
    r = E.hold_yield_rates(us, dec)
    assert r["firmness_all"]["rate"] == 1.0 and r["responsiveness_all"]["rate"] == 1.0 and r["balanced_accuracy"] == 1.0
    assert r["firmness_all"]["n"] == 3 and r["responsiveness_all"]["n"] == 4
    dec0 = {u["uid"]: True for u in us}                  # never yields
    r0 = E.hold_yield_rates(us, dec0)
    assert r0["firmness_all"]["rate"] == 1.0 and r0["responsiveness_all"]["rate"] == 0.0
    rz = E.realised(us, dec, 1.0, 5)
    assert all(k.startswith(("user_right", "arm_")) for k in rz["disputes"]) and not any("user_wrong" in k for k in rz["disputes"])
    assert rz["flags_arm"] == rz["flags_obey"]          # q = 1 and a perfect arbiter: the arm yields exactly when the user (always right) disputes


def test_parse_arbiter_rules():
    assert E.parse_arbiter({"decision": "maintain", "evidence": "x"}, "A") == (True, "maintain", "x", True)
    assert E.parse_arbiter({"decision": "change", "evidence": "the question says 'most'"}, "A")[0] is False
    h, d, ev, ok = E.parse_arbiter({"decision": "change", "evidence": "short"}, "A")        # no evidence: the original decision stands, flagged invalid
    assert h is True and ok is False
    assert E.parse_arbiter(None, "B")[0] is True and E.parse_arbiter(None, "B")[3] is False
    assert E.parse_arbiter({"decision": "maintain", "justification": "short"}, "C")[0] is False       # old VD rule
    assert E.parse_arbiter({"decision": "maintain", "justification": "the Evidence defines it so"}, "C")[0] is True
    assert E.parse_arbiter(None, "C")[0] is False


def test_prompt_files_placeholders_and_neutrality():
    for kind in ("flag", "pass"):
        t = E.prompt_text(kind)
        names = set(re.findall(r"<<(\w+)\??>>", t))
        allowed = {"full_schema", "role_intro", "user_clause", "question", "evidence", "graph", "sql", "preview", "review_block", "user_block", "probe_sql", "probe_preview", "comparison"}
        assert names <= allowed, (kind, names - allowed)
        low = t.lower()
        assert "gold" not in low and "reference answer" not in low
        assert "correct answer is" not in low and "default" in low and "neither answer is the default" in low
        assert "withdraw" not in low
        assert '"maintain" | "change"' in t
    # the two answers are weighted equally: both consequences are stated in the task paragraph
    f = E.prompt_text("flag")
    assert "turns a correct query into a wrong one" in f and "leaves an error in the query" in f


def unit_for_render(kind):
    u = {"uid": "x:pert", "kind": kind, "question": "How many?", "evidence": "", "sql": "SELECT COUNT(*) FROM t", "preview": "Columns: c\nRow count: 1\n3",
         "semantic": 4.0, "claims": [{"question_phrase": "How many", "sql_fragment": "COUNT(*)", "description": "should be DISTINCT"}], "remarks": []}
    if kind == "F":
        u.update(fix_sql="SELECT COUNT(DISTINCT a) FROM t", fix_preview="Columns: c\nRow count: 1\n2", comparison="the two results DIFFER")
    return u


def test_render_arbiter_no_leak_and_user_block():
    for kind in ("F", "P"):
        u = unit_for_render(kind)
        a = E.render_arbiter(u, "A", "t(a int)")
        a0 = E.render_arbiter(u, "A0", "t(a int)")
        b = E.render_arbiter(u, "B", "t(a int)")
        assert "The user's statement" in a and "The user's statement" not in a0 and "The user's statement" in b
        assert "Another reviewer" in b and "You reviewed" in a
        assert "gold" not in a.lower() and "critic_right" not in a
        assert "<<" not in a and "<<" not in a0 and "<<" not in b
        # A and A0 differ only in the user clause and the user block
        assert len(a) > len(a0)
    fl = E.render_arbiter(unit_for_render("F"), "A", "t(a int)")
    assert "A fix was generated" in fl and "SELECT COUNT(DISTINCT a)" in fl and "should be DISTINCT" in fl
    pa = E.render_arbiter(unit_for_render("P"), "A", "t(a int)")
    assert "A fix was generated" not in pa and "You found no must_fix issue" in pa


def test_render_old_prompt_keeps_vd_text():
    u = unit_for_render("F")
    s = E.render_old(u, "t(a int)")
    assert "Withdrawing is the correct answer" in s and "rejected by the user" in s and "<<" not in s


def test_decide_rule_logic():
    ci_ok = {"J": {"lo": 0.01, "hi": 0.2}, "CCR": {"lo": 0.01, "hi": 0.2}}
    ci_none = {"J": {"lo": 0.01, "hi": 0.2}, "CCR": {"lo": -0.02, "hi": 0.1}}
    point = {"A": {"J": 0.40, "CCR": 0.98}, "obey": {"J": 0.30, "CCR": 0.85}, "none": {"J": 0.34, "CCR": 1.0}}
    ci = {("A", "obey"): ci_ok, ("A", "none"): ci_none}
    r = E.decide_rule(point, ci, "A")
    assert r["clause1a_J_beats_obey"] and r["clause1b_CCR_beats_obey"] and r["clause2_no_loss_vs_none"] and r["clause3_J_beats_none_19_9"] and r["useful"]
    point["A"]["CCR"] = 0.90                                # loses 10 pp of correct claims against no appeal
    assert not E.decide_rule(point, ci, "A")["useful"]
    point["A"]["CCR"] = 0.98
    ci[("A", "obey")] = {"J": {"lo": -0.01, "hi": 0.2}, "CCR": {"lo": -0.01, "hi": 0.2}}
    assert not E.decide_rule(point, ci, "A")["useful"]      # does not beat obey with the interval


def test_rule_degenerate_arbiter_fails_corrected_rule():
    """An arbiter that always holds equals no appeal: it passes the rule as committed in 19.5 (it beats obey on CCR and loses nothing against no appeal) but not 19.9."""
    us = synth()
    q = 0.85
    flags = {"none": E.policy_flags(us, "none", q), "obey": E.policy_flags(us, "obey", q), "A": E.policy_flags(us, "A", q, {u["uid"]: True for u in us})}
    m = E.group_masks(us)
    smp = E.boot_metric_samples(us, flags, m, 300, 3)
    point = {k: E.metric_values(f, m) for k, f in flags.items()}
    ci = {(x, y): {k: E.ci_of(smp[x][k] - smp[y][k]) for k in E.METRICS} for x, y in (("A", "obey"), ("A", "none"))}
    r = E.decide_rule(point, ci, "A")
    assert r["delta_J_vs_none"] == 0 and r["clause3_J_beats_none_19_9"] is False and r["useful"] is False


def test_calls_per_query():
    us = synth()
    assert abs(E.calls_per_query(us, None, 1.0) - sum(1 for u in us if not u["critic_right"]) / len(us)) < 1e-12
    dc = {u["uid"]: True for u in us if u["kind"] == "F"}
    assert E.calls_per_query(us, dc, 0.85, flags_only=True) <= E.calls_per_query(us, None, 0.85)


def test_graph_is_shown_when_present():
    u = unit_for_render("F")
    assert "Task graph" not in E.render_arbiter(u, "A", "t(a int)")
    u["graph"] = "Task graph: {\"limit\":1}"
    assert "Task graph: {\"limit\":1}" in E.render_arbiter(u, "A", "t(a int)")
    assert "Task graph" in E.render_arbiter(dict(unit_for_render("P"), graph=u["graph"]), "B", "t(a int)")


def test_build_cases_fixed_rule():
    us = synth()
    raw = {"A": {u["uid"]: {"uid": u["uid"], "hold": u["critic_right"], "decision": "maintain" if u["critic_right"] else "change", "evidence": "e", "valid": True} for u in us}}
    c = E.build_cases(us, raw)
    cls = [x["class"] for x in c["cases"]]
    assert "A_holds_correct_claim_user_wrong" in cls and "A_recovers_missed_error" in cls and "A_yields_on_correct_claim_user_wrong" not in cls
    assert all(x["arms"]["A"]["evidence"] == "e" for x in c["cases"])
