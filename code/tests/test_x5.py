"""X5 tests (offline: scripted fake LLM, toy sqlite databases, no API call, no dev / Spider data).
Covers: identifier hygiene (ddl.quote_keywords), execution-gated policy, consistency sub-score, routing score + fit tool,
arbitration, M3 fallback config, legacy byte-identity, config / lock plumbing."""
import hashlib
import json
import os
import re
import sqlite3
import sys

import numpy as np
import pytest

from nl2sql import config as C, data, db, paths, prompts_util as P
from nl2sql.data import Item, SQLITE_KEYWORDS, _q, needs_quote, render_ddl
from nl2sql.systems.ours import m3 as M3, m5 as M5, pipeline as PL
from nl2sql.systems.ours.defaults import DEFAULTS
from nl2sql.systems.ours.route import RouteScorer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import arbitrate_probe as AP  # noqa: E402
import fit_route_score as FR  # noqa: E402

CONFIGS = paths.EXP_DIR / "configs"


# ------------------------------------------------------------------------------------------------ fixtures / fakes
@pytest.fixture()
def toydb(tmp_path):
    p = str(tmp_path / "toy.sqlite")
    con = sqlite3.connect(p)
    con.execute('CREATE TABLE "transaction" (id INTEGER PRIMARY KEY, amt REAL, "order" TEXT)')
    con.executemany('INSERT INTO "transaction" VALUES (?,?,?)', [(1, 5.0, "a"), (2, 7.5, "b"), (3, 1.0, "c")])
    con.execute("CREATE TABLE empty_t (x INTEGER)")
    con.commit()
    con.close()
    return p


def mk_item(db_path, qid="toy:t:1"):
    return Item(qid=qid, db_id="t", question="how many transactions are there?", evidence="", gold_sql="", db_path=db_path, split="toy", meta={})


class ScriptLLM:
    """Scripted chat: handlers per tag; records every call. n > 1 returns a list of texts."""

    def __init__(self, gen=None, rev=None, crit=None, cons=None, arb=None):
        self.h = {"gen": gen, "rev": rev, "crit": crit, "cons": cons, "arb": arb}
        self.calls = []

    def chat(self, model, messages, *, temperature=0.0, seed=None, max_tokens=2048, n=1, tag="", **kw):
        user = messages[-1]["content"]
        self.calls.append({"tag": tag, "T": temperature, "seed": seed, "n": n, "user": user})
        h = self.h.get(tag)
        if h is None:
            raise AssertionError("unexpected LLM call with tag %r" % tag)
        text = h(user, n, len(self.calls)) if callable(h) else h
        return {"text": text, "prompt_tokens": 1, "completion_tokens": 1, "latency_s": 0.0, "cached": False}

    def tags(self):
        return [c["tag"] for c in self.calls]


CRIT9 = '{"semantic": 9, "issues": [], "suggestions": []}'


class StubStore:
    schema = {"tables": {"transaction": [{"name": "id"}, {"name": "amt"}, {"name": "order"}]}, "row_counts": {"transaction": 3}}


class StubRes:
    def db_store(self, item):
        return StubStore()

    def corpus_for(self, item):
        return None

    def ensure_terms(self, item):
        pass


def fake_retrieve(res, item, graph, cfg, corpus, **kw):
    qk = bool((cfg.get("ddl") or {}).get("quote_keywords"))
    sc = {"tables": {"transaction": [{"name": "id", "type": "INTEGER", "desc": "", "examples": [], "pk": 1},
                                     {"name": "amt", "type": "REAL", "desc": "", "examples": []},
                                     {"name": "order", "type": "TEXT", "desc": "", "examples": []}]}, "fks": []}
    return M3.Context(tables=["transaction"], ddl=render_ddl(sc, quote_keywords=qk), full_schema=True)


def make_sys(monkeypatch, cls, llm, cfg):
    monkeypatch.setattr(PL.M3, "retrieve", fake_retrieve)
    cfg = C.deep_update({"m1": {"enabled": False}}, cfg)
    s = cls(llm, "m", cfg)
    s.pipe.res = StubRes()
    return s


BAD = 'SELECT count(*) FROM transaction'
GOOD = 'SELECT count(*) FROM "transaction"'
GATED = {"m5": {"policy": "gated"}, "route": {"score": "exec", "theta_h": 7.5}, "ddl": {"quote_keywords": True}}


# ------------------------------------------------------------------------------------------------ 1. identifier hygiene
def test_keyword_list_is_complete_and_quoting_is_valid_sqlite():
    assert len(SQLITE_KEYWORDS) == 147 and all(k == k.upper() for k in SQLITE_KEYWORDS)
    con = sqlite3.connect(":memory:")
    for i, kw in enumerate(sorted(SQLITE_KEYWORDS)):
        con.execute('CREATE TABLE "%s" ("%s" INTEGER)' % (kw, kw))              # every keyword is valid once quoted
        con.execute('SELECT "%s" FROM %s' % (kw, _q(kw, True)))
    # every word that is NOT usable bare as a table name must be in the list (probe: list + common column-like words)
    probe = set(k.lower() for k in SQLITE_KEYWORDS) | {"date", "time", "year", "name", "type", "value", "values", "count", "index", "group",
                                                        "order", "limit", "match", "current", "transaction", "key", "desc", "rank", "level"}
    for w in sorted(probe):
        try:
            sqlite3.connect(":memory:").execute("CREATE TABLE %s (a)" % w)
            bare_ok = True
        except sqlite3.OperationalError:
            bare_ok = False
        if not bare_ok:
            assert w.upper() in SQLITE_KEYWORDS, w
            assert needs_quote(w, True) and _q(w, True) == '"%s"' % w
    # names that need quoting for other reasons are quoted in both modes
    for n in ("my col", "1abc", "a-b", 'we"ird', "Größe"):
        assert needs_quote(n, False) and _q(n, False) == _q(n, True)
    assert _q('we"ird', True) == '"we""ird"'
    for n in ("amount", "OrderId", "_x1"):
        assert _q(n, True) == n and _q(n, False) == n
    assert _q("Transaction", False) == "Transaction" and _q("Transaction", True) == '"Transaction"'      # case-insensitive keyword match


def _legacy_q(name):                                  # verbatim copy of data._q before X5
    return '"%s"' % name.replace('"', '""') if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name) else name


def _legacy_render_ddl(schema, tables=None, samples=True, descriptions=True, flag=None):     # verbatim copy of data.render_ddl before X5
    names = list(tables) if tables is not None else list(schema["tables"].keys())
    out = []
    for t in names:
        cols = schema["tables"][t]
        pk_cols = [c["name"] for c in cols if c.get("pk")]
        fk_here = [f for f in schema["fks"] if f["table"] == t and (tables is None or f["ref_table"] in names)]
        items = []
        for c in cols:
            s = "  %s %s" % (_legacy_q(c["name"]), c["type"] or "")
            s = s.rstrip()
            cm = []
            if descriptions and c.get("desc"):
                cm.append(c["desc"])
            if samples and c.get("examples"):
                cm.append("samples: " + ", ".join(c["examples"]))
            items.append((s, (" | ".join(cm) if cm else None)))
        tail = []
        if pk_cols:
            tail.append("  PRIMARY KEY (%s)" % ", ".join(_legacy_q(c) for c in pk_cols))
        for f in fk_here:
            tail.append("  FOREIGN KEY (%s) REFERENCES %s(%s)" % (_legacy_q(f["column"]), _legacy_q(f["ref_table"]), _legacy_q(f["ref_column"] or "")))
        total = len(items) + len(tail)
        body = []
        for k, (s, cm) in enumerate(items):
            comma = "," if (k < total - 1) else ""
            body.append(s + comma + ("  -- " + cm if cm else ""))
        for k, s in enumerate(tail):
            comma = "," if (len(items) + k < total - 1) else ""
            body.append(s + comma)
        head = "CREATE TABLE %s (" % _legacy_q(t)
        if flag and t in flag:
            head = "-- likely relevant\n" + head
        out.append(head + "\n" + "\n".join(body) + "\n);")
    return "\n".join(out)


def kw_schema():
    return {"tables": {"transaction": [{"name": "id", "type": "INTEGER", "desc": "key", "examples": [], "pk": 1},
                                       {"name": "order", "type": "TEXT", "desc": "", "examples": ["a", "b"]},
                                       {"name": "my col", "type": "TEXT", "desc": "x", "examples": ["q"]},
                                       {"name": "group", "type": "", "desc": "", "examples": []}],
                       "country": [{"name": "Match", "type": "INT", "desc": "", "examples": [], "pk": 1}]},
            "fks": [{"table": "transaction", "column": "id", "ref_table": "country", "ref_column": "Match"}]}


def test_render_ddl_legacy_is_byte_identical_and_quoting_is_a_pure_prompt_change():
    sc = kw_schema()
    for kw in ({}, {"tables": ["country"]}, {"samples": False}, {"flag": {"country"}}, {"descriptions": False, "flag": {"transaction"}}):
        assert render_ddl(sc, **kw) == _legacy_render_ddl(sc, **kw)                  # default = legacy, byte for byte
        assert render_ddl(sc, quote_keywords=False, **kw) == _legacy_render_ddl(sc, **kw)
    legacy, quoted = render_ddl(sc), render_ddl(sc, quote_keywords=True)
    assert "CREATE TABLE transaction (" in legacy and 'CREATE TABLE "transaction" (' in quoted
    assert '"order" TEXT' in quoted and '"group"' in quoted and '"Match" INT' in quoted
    assert "FOREIGN KEY (id) REFERENCES country(\"Match\")" in quoted and "REFERENCES country(Match)" in legacy
    # quoted DDL is valid SQLite (comments included) and creates the tables with their ORIGINAL names; legacy DDL is not
    con = sqlite3.connect(":memory:")
    con.executescript(quoted)
    assert {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")} == {"transaction", "country"}
    with pytest.raises(sqlite3.OperationalError, match='near "transaction"'):
        sqlite3.connect(":memory:").executescript(legacy)
    # only the text of the prompt differs: after removing the added quotes around keyword names both renderings are equal
    assert quoted.replace('"transaction"', "transaction").replace('"order"', "order").replace('"group"', "group").replace('"Match"', "Match") == legacy


OLD_TEMPLATES = {      # prompts/{gen,gen_zs,rev}.user.txt as they were before X5 (the placeholder <<quote_rule?>> must render to nothing)
    "gen.user": "<<sections>>\n\n# Task\nWrite one SQLite query that answers the request below.\nRules: use only tables and columns that appear in the schema; quote identifiers that contain spaces or special characters with double quotes; join tables only through the listed foreign keys or join hints; follow the evidence literally when it defines a term or formula; return exactly the columns the question asks for, in the order asked, with no extra columns.\nReturn only the SQL.\n\n# Request\nQuestion: <<question>>\nEvidence: <<evidence>>\n<<graph>>",
    "gen_zs.user": "# Schema\n<<schema>>\n\n# Task\nWrite one SQLite query that answers the request below.\nRules: use only tables and columns that appear in the schema; quote identifiers that contain spaces or special characters with double quotes; follow the evidence literally when it defines a term or formula; return exactly the columns the question asks for, in the order asked, with no extra columns.\nReturn only the SQL.\n\n# Request\nQuestion: <<question>>\nEvidence: <<evidence>>\n",
    "rev.user": "<<sections>>\n\n# Task\nFix the SQLite query below so that it answers the question. You see the previous query, its execution result and a reviewer's feedback. Use only tables and columns that appear in the schema. Keep what is correct and change only what the feedback shows to be wrong.\nReturn only the corrected SQL.\n\n# Request\nQuestion: <<question>>\nEvidence: <<evidence>>\n<<graph>>\n\n# Previous SQL\n<<sql>>\n\n# Execution\n<<preview>>\n\n# Reviewer feedback (score <<score>>/10; <<mode>>)\n<<feedback>>\n",
}


def _old_render(name, **kw):
    return re.sub(r"<<(\w+)>>", lambda m: str(kw[m.group(1)]), OLD_TEMPLATES[name])


def test_prompt_templates_legacy_render_identical_and_rule_only_when_asked():
    args = {"gen.user": dict(sections="S", question="Q", evidence="E", graph="G"),
            "gen_zs.user": dict(schema="DDL", question="Q", evidence="E"),
            "rev.user": dict(sections="S", question="Q", evidence="E", graph="G", sql="SELECT 1", preview="P", score="5.0", mode="must fix", feedback="F")}
    for name, kw in args.items():
        tpl = (paths.PROMPTS_DIR / (name + ".txt")).read_text()
        assert tpl.replace("<<quote_rule?>>", "").replace("<<value_hints?>>", "").rstrip("\n") == OLD_TEMPLATES[name].rstrip("\n")       # only the optional placeholders (quote_rule X5, value_hints X9) were added
        assert P.render(name, **kw).rstrip("\n") == _old_render(name, **kw).rstrip("\n")
        assert "reserved" not in P.render(name, **kw)
        withrule = P.render(name, quote_rule=P.QUOTE_RULE, **kw)
        assert "Quote SQLite reserved words" in withrule and withrule.count("\n") == P.render(name, **kw).count("\n")
    with pytest.raises(KeyError):
        P.render("gen.user", sections="S", question="Q", evidence="E")          # required placeholders still required


def test_pipeline_prompts_quote_rule_follows_config(monkeypatch, toydb):
    item = mk_item(toydb)
    for flag in (False, True):
        llm = ScriptLLM(gen=GOOD, crit=CRIT9)
        s = make_sys(monkeypatch, PL.FullFixed, llm, {"ddl": {"quote_keywords": flag}})
        s.run(item)
        user = llm.calls[0]["user"]
        assert ("Quote SQLite reserved words" in user) == flag
        assert ('CREATE TABLE "transaction" (' in user) == flag and ("CREATE TABLE transaction (" in user) == (not flag)


def test_zero_shot_prompt_follows_config(monkeypatch, toydb):
    item = mk_item(toydb)
    sc = {"tables": {"transaction": [{"name": "id", "type": "INTEGER", "desc": "", "examples": [], "pk": 1}]}, "fks": [], "pks": [], "row_counts": {}}
    sc["ddl"] = _legacy_render_ddl(sc)
    monkeypatch.setattr(PL.data, "get_schema", lambda it: sc)
    off = PL.ZeroShot(None, "m", {})._req(item, 0)[1]["content"]
    on = PL.ZeroShot(None, "m", {"ddl": {"quote_keywords": True}})._req(item, 0)[1]["content"]
    assert off == _old_render("gen_zs.user", schema=sc["ddl"], question=item.question, evidence="(none)")
    assert 'CREATE TABLE "transaction"' in on and "Quote SQLite reserved words" in on and "CREATE TABLE transaction" in off


def test_lineage_hints_quote_keyword_tables():
    from nl2sql.systems.ours.knowledge import DbStore
    st = DbStore.__new__(DbStore)
    st.edges = [{"a": "transaction", "ac": "id", "b": "country", "bc": "Match", "kind": "fk", "overlap": 1.0}]
    assert st.lineage_text(["transaction", "country"]) == "transaction.id = country.Match  (FK)"
    assert st.lineage_text(["transaction", "country"], quote_keywords=True) == '"transaction".id = country."Match"  (FK)'


def test_beer_factory_gold_sql_runs_with_quoted_ddl_names_unchanged():
    """Real BIRD-train data (skipped where it is not mounted): the database keeps its names; only the prompt text is quoted."""
    try:
        items = [i for i in data.load_split("bird_train_HR", with_gold=True) if i.db_id == "beer_factory"]
    except Exception as e:                                  # data not available on this machine
        pytest.skip("BIRD train data not available: %s" % e)
    if not items:
        pytest.skip("beer_factory not in the HR split here")
    sc = data.get_schema(items[0])
    con = sqlite3.connect("file:%s?mode=ro&immutable=1" % items[0].db_path, uri=True)
    real_tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    con.close()
    assert "transaction" in real_tables and list(sc["tables"]) == real_tables           # data / stored schema untouched
    quoted = render_ddl(sc, quote_keywords=True)
    assert 'CREATE TABLE "transaction" (' in quoted and "CREATE TABLE transaction (" in render_ddl(sc) and "CREATE TABLE transaction (" in sc["ddl"]
    mem = sqlite3.connect(":memory:")
    mem.executescript(render_ddl(sc, quote_keywords=True, samples=False, descriptions=False))      # prompt DDL is itself valid SQLite, original names
    assert {r[0] for r in mem.execute("SELECT name FROM sqlite_master WHERE type='table'")} == set(real_tables)
    bad = []
    for it in items:                                         # every gold query of the db still executes
        r = db.execute(it.db_path, it.gold_sql, timeout_s=60, max_rows=1)
        if not r["ok"]:
            bad.append((it.qid, r["error"]))
    assert not bad, bad[:3]
    assert any("transaction" in it.gold_sql for it in items)


# ------------------------------------------------------------------------------------------------ 2. gated policy
def test_gate_trigger_and_feedback_units():
    for st in ("err_syntax", "err_identifier", "timeout"):
        assert M5.gate_trigger(st) and M5.gate_trigger(st, True)
    assert not M5.gate_trigger("ok_nonempty", True) and not M5.gate_trigger("ok_empty") and M5.gate_trigger("ok_empty", True)
    assert M5.exec_feedback_text({"ok": False, "error": 'near "transaction": syntax error'}, "err_syntax") == 'Execution error: near "transaction": syntax error'
    assert "no rows" in M5.exec_feedback_text({"ok": True, "rows": []}, "ok_empty")


def test_gated_repairs_execution_error_without_semantic_critic(monkeypatch, toydb):
    item = mk_item(toydb)
    llm = ScriptLLM(gen=BAD, rev=GOOD)
    out = make_sys(monkeypatch, PL.FullGated, llm, {"ddl": {"quote_keywords": True}, "route": {"score": "exec", "theta_h": 7.5}}).run(item)
    assert llm.tags() == ["gen", "rev"]                                     # no critic call at all
    assert out["actions"] == ["revise", "accept"] and out["final_sql"] == GOOD and out["iterations"] == 1
    rev = llm.calls[1]["user"]
    assert 'Execution error: near "transaction": syntax error' in rev.split("# Execution feedback")[1]       # feedback = the actual error text
    assert "Reviewer feedback" not in rev and "# Execution feedback" in rev and "Quote SQLite reserved words" in rev
    assert out["critic"]["semantic"] is None and out["critic"]["score"] == 10.0 and out["final_status"] == "ok_nonempty"
    assert out["abstained"] is False and out["route_kind"] == "exec" and out["route_score"] == 10.0 and out["confidence"] == 10.0
    assert [c["score_kind"] for c in out["candidates"]] == ["exec", "exec"] and out["candidates"][0]["status"] == "err_syntax"


def test_gated_accepts_executable_root_with_one_call(monkeypatch, toydb):
    llm = ScriptLLM(gen=GOOD)
    out = make_sys(monkeypatch, PL.FullGated, llm, GATED).run(mk_item(toydb))
    assert llm.tags() == ["gen"] and out["actions"] == ["accept"] and out["iterations"] == 0 and not out["abstained"]


def test_gated_empty_result_is_configurable(monkeypatch, toydb):
    empty = 'SELECT id FROM "transaction" WHERE amt > 100'
    for gate_empty, tags in ((False, ["gen"]), (True, ["gen", "rev"])):
        llm = ScriptLLM(gen=empty, rev=GOOD)
        out = make_sys(monkeypatch, PL.FullGated, llm, C.deep_update(GATED, {"m5": {"gate_empty": gate_empty}})).run(mk_item(toydb))
        assert llm.tags() == tags
        assert out["final_sql"] == (GOOD if gate_empty else empty)
        if gate_empty:
            assert "returned no rows" in llm.calls[1]["user"]


def test_gated_never_replaces_executable_root_by_failing_revision_and_stops_at_tmax(monkeypatch, toydb):
    empty = 'SELECT id FROM "transaction" WHERE amt > 100'
    llm = ScriptLLM(gen=empty, rev="SELECT nope FROM nowhere")
    out = make_sys(monkeypatch, PL.FullGated, llm, C.deep_update(GATED, {"m5": {"gate_empty": True, "t_max": 2}})).run(mk_item(toydb))
    assert out["final_sql"] == empty and out["final_status"] == "ok_empty"            # empty (executable) beats an error
    assert out["actions"] == ["revise", "revise", "revise_noop", "stop_best"] and llm.tags() == ["gen", "rev", "rev"]     # error node, then identical sql -> stop
    llm = ScriptLLM(gen=BAD, rev=BAD)                                                  # still failing: revise_noop stops, abstain via exec route
    out = make_sys(monkeypatch, PL.FullGated, llm, GATED).run(mk_item(toydb))
    assert out["actions"] == ["revise", "revise_noop", "stop_best"] and out["abstained"] is True and out["critic"]["score"] == 2.0


def test_gated_score_for_routing_is_one_final_only_critic_call(monkeypatch, toydb):
    cfg = {"m5": {"policy": "gated", "score_for_routing": True}, "route": {"score": "v3"}, "ddl": {"quote_keywords": True}}
    llm = ScriptLLM(gen=BAD, rev=GOOD, crit='{"semantic": 4, "issues": [], "suggestions": []}')
    out = make_sys(monkeypatch, PL.FullGated, llm, cfg).run(mk_item(toydb))
    assert llm.tags() == ["gen", "rev", "crit"]                                        # the semantic critic runs once, after the loop
    assert out["critic"]["semantic"] == 4.0 and out["critic"]["score"] == pytest.approx(0.3 * 10 + 0.5 * 4 + 0.2 * 10)
    assert out["route_kind"] == "v3" and out["route_score"] == out["critic"]["score"] == 7.0
    assert out["abstained"] is True                                                    # 7.0 < theta_h 7.5 (patent routing rule)
    llm = ScriptLLM(gen=GOOD, crit=CRIT9)
    out = make_sys(monkeypatch, PL.FullGated, llm, cfg).run(mk_item(toydb))
    assert llm.tags() == ["gen", "crit"] and out["abstained"] is False and out["route_score"] == pytest.approx(9.5)


def test_config_validation(monkeypatch):
    llm = ScriptLLM()
    with pytest.raises(ValueError, match="score_for_routing"):
        make_sys(monkeypatch, PL.FullGated, llm, {"route": {"score": "v3"}})            # v3 routing needs the semantic call
    with pytest.raises(ValueError, match="consistency.k"):
        make_sys(monkeypatch, PL.FullGated, llm, {"route": {"score": "exec+consistency"}})
    with pytest.raises(ValueError):
        make_sys(monkeypatch, PL.FullGated, llm, {"route": {"score": "bogus"}})
    with pytest.raises(ValueError, match="policy"):
        make_sys(monkeypatch, PL.FullSystem, llm, {"m5": {"policy": "nope"}})
    with pytest.raises(ValueError, match="v5"):
        make_sys(monkeypatch, PL.FullGated, llm, {"m5": {"variant": "v5"}, "route": {"score": "exec"}})


def test_legacy_p0_rows_have_no_x5_keys_and_unchanged_flow(monkeypatch, toydb):
    llm = ScriptLLM(gen=GOOD, crit=CRIT9)
    out = make_sys(monkeypatch, PL.FullFixed, llm, {}).run(mk_item(toydb))
    assert llm.tags() == ["gen", "crit"] and out["actions"] == ["accept"] and out["confidence"] == out["critic"]["score"] == 9.5
    new_keys = {"consistency", "n_samples", "n_agree", "cons_bits", "route_score", "route_kind", "arbitrated", "arb_choice"}
    assert not (new_keys & set(out))
    assert 'CREATE TABLE transaction (' in llm.calls[0]["user"]                         # legacy DDL unquoted (HEADROOM row 1 artefact kept reproducible)
    # patent loop still revises on a low semantic score (feedback = critic text, legacy rev template)
    llm = ScriptLLM(gen=GOOD, rev='SELECT sum(amt) FROM "transaction"',
                    crit=lambda u, n, k: '{"semantic": 3, "issues": [{"type": "semantic", "description": "wrong"}], "suggestions": []}' if "count(*)" in u.split("# SQL")[1] else CRIT9)
    out = make_sys(monkeypatch, PL.FullFixed, llm, {}).run(mk_item(toydb))
    assert llm.tags() == ["gen", "crit", "rev", "crit"] and "Reviewer feedback (score" in llm.calls[2]["user"]
    assert out["actions"] == ["revise", "accept"] and out["final_sql"].startswith("SELECT sum")


# ------------------------------------------------------------------------------------------------ 3. consistency + routing score
def cons_handler(final_ok_sql, samples_by_call):
    def h(user, n, k):
        outs = samples_by_call.pop(0)
        assert len(outs) == n
        return outs if n > 1 else outs[0]
    return h


def test_consistency_samples_by_n_and_result_agreement(monkeypatch, toydb):
    samples = [[GOOD, 'SELECT count(id) FROM "transaction"', "SELECT 1", "SELEC bad"], [GOOD, "SELECT count(*) FROM empty_t UNION SELECT 3"]]
    cfg = {"m5": {"policy": "gated", "score_for_routing": True}, "ddl": {"quote_keywords": True},
           "route": {"score": "v3+consistency", "consistency": {"k": 6, "temperature": 0.7, "n_per_call": 4}}}
    llm = ScriptLLM(gen=GOOD, crit=CRIT9, cons=cons_handler(GOOD, samples))
    out = make_sys(monkeypatch, PL.FullGated, llm, cfg).run(mk_item(toydb), seed=100)
    cons = [c for c in llm.calls if c["tag"] == "cons"]
    assert [c["n"] for c in cons] == [4, 2] and [c["seed"] for c in cons] == [101, 102] and all(c["T"] == 0.7 for c in cons)
    assert cons[0]["user"] == llm.calls[0]["user"]                                      # same prompt as the initial generation
    assert out["cons_bits"] == [1, 1, 0, 0, 1, 0] and out["n_samples"] == 6 and out["n_agree"] == 3 and out["consistency"] == 0.5
    # {(0,),(3,)} from the UNION differs from {(3,)}: set equality of result rows, not "contains"
    assert out["route_kind"] == "v3+consistency" and out["route_score"] == pytest.approx((9.5 + 5.0) / 2, abs=1e-3) == out["confidence"]
    assert out["critic"]["score"] == pytest.approx(9.5)                                 # raw V3 stays available for post-hoc fits


def test_consistency_failed_final_never_agrees_and_single_sample(monkeypatch, toydb):
    cfg = {"m5": {"policy": "gated", "t_max": 0}, "ddl": {"quote_keywords": True}, "route": {"score": "consistency", "consistency": {"k": 2}}}
    llm = ScriptLLM(gen=BAD, cons=cons_handler(None, [[BAD, BAD]]))
    out = make_sys(monkeypatch, PL.FullGated, llm, cfg).run(mk_item(toydb))
    assert out["consistency"] == 0.0 and out["n_samples"] == 2 and out["route_score"] == 0.0
    cfg["route"]["consistency"] = {"k": 1}
    llm = ScriptLLM(gen=GOOD, cons=cons_handler(None, [[GOOD]]))
    out = make_sys(monkeypatch, PL.FullGated, llm, cfg).run(mk_item(toydb))
    assert out["consistency"] == 1.0 and out["n_samples"] == 1 and llm.calls[1]["n"] == 1 and out["route_score"] == 10.0


def test_consistency_recorded_without_changing_v3_routing(monkeypatch, toydb):
    cfg = {"route": {"score": "v3", "consistency": {"k": 4}}}
    llm = ScriptLLM(gen=GOOD, crit=CRIT9, cons=cons_handler(None, [[GOOD] * 4]))
    out = make_sys(monkeypatch, PL.FullFixed, llm, cfg).run(mk_item(toydb))
    assert out["consistency"] == 1.0 and out["confidence"] == out["critic"]["score"] and "route_score" not in out and out["abstained"] is False


def test_route_scorer_modes(tmp_path):
    f = dict(v3=8.0, exec=10.0, consistency=0.5, nonempty=1.0)
    assert RouteScorer({"score": "v3"}).legacy and RouteScorer({"score": "v3"}).score(f) == 8.0
    assert RouteScorer({"score": "exec"}).score(f) == 10.0
    assert RouteScorer({"score": "consistency", "consistency": {"k": 2}}).score(f) == 5.0
    lin = RouteScorer({"score": "v3+consistency", "consistency": {"k": 2}})
    assert lin.score(f) == pytest.approx(6.5) and not lin.legacy
    w = RouteScorer({"score": "v3+consistency", "weights": {"v3": 3, "consistency": 1}, "consistency": {"k": 2}})
    assert w.score(f) == pytest.approx((3 * 8 + 5) / 4.0)
    coef = {"features": ["v3", "consistency", "nonempty"], "coef": [0.5, 2.0, 1.0], "intercept": -6.0}
    z = -6.0 + 0.5 * 8 + 2.0 * 0.5 + 1.0
    lr = RouteScorer({"score": "v3+consistency", "coef": coef, "consistency": {"k": 2}})
    assert lr.score(f) == pytest.approx(10.0 / (1 + np.exp(-z)), abs=1e-3)
    p = tmp_path / "c.json"
    json.dump(coef, open(p, "w"))
    assert RouteScorer({"score": "v3+consistency", "coef_path": str(p), "consistency": {"k": 2}}).score(f) == lr.score(f)
    with pytest.raises(ValueError):
        lr.score(dict(f, consistency=None))
    with pytest.raises(ValueError):
        RouteScorer({"score": "v3+consistency", "coef": {"features": ["zzz"], "coef": [1.0]}, "consistency": {"k": 2}})


# ------------------------------------------------------------------------------------------------ 4. arbitration
def judge_prefers(substr):
    """Fake judge: picks whichever of Candidate A / B contains `substr` (so it works for either A/B order)."""
    def h(user, n, k):
        a = user.split("# Candidate A")[1].split("# Candidate B")[0]
        return json.dumps({"choice": "A" if substr in a else "B", "reason": "x"})
    return h


def semantic_by_sql(user, n, k):
    sql_part = user.split("# SQL")[1]
    return '{"semantic": 9, "issues": [], "suggestions": []}' if "sum(" in sql_part else '{"semantic": 3, "issues": [], "suggestions": []}'


def test_arbitration_picks_between_root_and_revision_in_either_order(monkeypatch, toydb):
    flips = set()
    for i in range(1, 11):
        qid, prefer, expect = "toy:t:%d" % i, ("count(" if i % 2 else "sum("), ("root" if i % 2 else "final")
        llm = ScriptLLM(gen=GOOD, rev='SELECT sum(amt) FROM "transaction"', crit=semantic_by_sql, arb=judge_prefers(prefer))
        out = make_sys(monkeypatch, PL.FullFixed, llm, {"m5": {"arbitrate": True}}).run(mk_item(toydb, qid))
        assert llm.tags().count("arb") == 1 and out["arbitrated"] == 1 and out["arb_choice"] == expect and out["arb_parse_ok"]
        flips.add(out["arb_flip"])
        assert ("count(" in out["final_sql"]) == (expect == "root") and out["actions"][-1] == "arbitrate_" + expect
        both = llm.calls[llm.tags().index("arb")]["user"]
        assert "Execution of A:" in both and "Execution of B:" in both and "Row count" in both            # both result previews
        if expect == "root":      # recomputed from the chosen node: V3 6.5 < theta_h 7.5 -> route to human
            assert out["abstained"] is True and out["confidence"] == out["critic"]["score"] == pytest.approx(6.5)
        else:
            assert out["abstained"] is False
    assert flips == {0, 1}                                                                          # both A/B orders exercised


def test_arbitration_not_called_when_results_equal_errors_or_no_revision(monkeypatch, toydb):
    # revision with the same result set (and the higher score, so it is the final answer) -> no judge
    def sem(user, n, k):
        part = user.split("# SQL")[1]
        return '{"semantic": 9, "issues": [], "suggestions": []}' if "count(id)" in part else '{"semantic": 3, "issues": [], "suggestions": []}'
    llm = ScriptLLM(gen=GOOD, rev='SELECT count(id) FROM "transaction"', crit=sem, arb=lambda *a: pytest.fail("judge called"))
    out = make_sys(monkeypatch, PL.FullFixed, llm, {"m5": {"arbitrate": True, "t_max": 1}}).run(mk_item(toydb))
    assert "arb" not in llm.tags() and out["arbitrated"] == 0 and "count(id)" in out["final_sql"]
    # root failed to execute: the executable revision wins, no judge
    llm = ScriptLLM(gen=BAD, rev=GOOD, crit=CRIT9, arb=lambda *a: pytest.fail("judge called"))
    out = make_sys(monkeypatch, PL.FullFixed, llm, {"m5": {"arbitrate": True}}).run(mk_item(toydb))
    assert out["arbitrated"] == 0 and out["final_sql"] == GOOD
    # no revision at all
    llm = ScriptLLM(gen=GOOD, crit=CRIT9)
    out = make_sys(monkeypatch, PL.FullFixed, llm, {"m5": {"arbitrate": True}}).run(mk_item(toydb))
    assert "arb" not in llm.tags() and out["arbitrated"] == 0


def test_arbitration_unparsable_answer_keeps_final_and_parser(monkeypatch, toydb):
    assert PL.parse_arbitration('{"choice": "b", "reason": "r"}') == "B"
    assert PL.parse_arbitration('{"choice": "A", "reas') == "A" and PL.parse_arbitration("no idea") is None
    llm = ScriptLLM(gen=GOOD, rev='SELECT sum(amt) FROM "transaction"', crit=semantic_by_sql, arb="I cannot decide")
    out = make_sys(monkeypatch, PL.FullFixed, llm, {"m5": {"arbitrate": True}}).run(mk_item(toydb))
    assert out["arb_choice"] == "final" and out["arb_parse_ok"] is False and out["final_sql"].startswith("SELECT sum")


def test_arbitration_with_gated_empty_root(monkeypatch, toydb):
    empty = 'SELECT id FROM "transaction" WHERE amt > 100'
    llm = ScriptLLM(gen=empty, rev=GOOD, arb=judge_prefers("amt > 100"))
    cfg = {"m5": {"policy": "gated", "gate_empty": True, "arbitrate": True}, "route": {"score": "exec"}}
    out = make_sys(monkeypatch, PL.FullGated, llm, cfg).run(mk_item(toydb))
    assert llm.tags() == ["gen", "rev", "arb"] and out["arb_choice"] == "root" and out["final_sql"] == empty      # judge may keep the empty root


# ------------------------------------------------------------------------------------------------ 5. M3 fallback option + configs
class _St:
    schema = {"tables": {"a": [{"name": "x", "type": "", "desc": "", "examples": []}], "b": [{"name": "y", "type": "", "desc": "", "examples": []}]},
              "fks": [], "pks": [], "row_counts": {}}
    full_tokens = 500

    def retrieve_tables(self, *a, **k):
        return [("a", 1.0)], 0

    def expand_fk_paths(self, tables):
        return tables

    def lineage_text(self, tables, quote_keywords=False):
        return ""


class _R:
    llm = None
    terms = None

    def db_store(self, item):
        return _St()

    def embed_one(self, t):
        return np.zeros(3)


def test_schema_fallback_tokens_is_an_m3_config_option(toydb):
    item = mk_item(toydb)
    base = {"m1": {"enabled": False}, "m2": {"history": False, "terms": False, "lineage": False}}
    ctx6000 = M3.retrieve(_R(), item, None, PL.resolved_cfg(base), None)
    assert ctx6000.full_schema and ctx6000.tables == ["a", "b"]                         # default 6000: tiny schema -> full-schema fallback
    ctx0 = M3.retrieve(_R(), item, None, PL.resolved_cfg(C.deep_update(base, {"m3": {"schema_fallback_tokens": 0}})), None)
    assert not ctx0.full_schema and ctx0.tables == ["a"]                                # fallback 0: M3 routing result is used
    cfg = C.load_config(CONFIGS / "full_m3force.yaml")
    assert cfg["m3"] == {"schema_fallback_tokens": 0}
    assert _flat(PL.resolved_cfg(cfg)) == dict(_flat(PL.resolved_cfg(C.load_config(CONFIGS / "full_fixed.yaml"))), **{"m3.schema_fallback_tokens": 0})


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update(_flat(v, pre + k + "."))
        else:
            out[pre + k] = v
    return out


def test_existing_configs_resolve_to_legacy_behaviour_and_new_configs_load(monkeypatch):
    new = {"full_gated", "full_gated_route", "full_gated_cons", "full_gated_cons_lr", "full_quote", "full_arb", "full_gated_arb", "full_m3force"}
    for f in sorted(CONFIGS.glob("*.yaml")):
        cfg = PL.resolved_cfg(C.load_config(f))
        if f.stem in new or f.stem.startswith(("min_", "full_gated_probe", "full_gated_empty", "full_gated_adaptive", "full_gated_vg", "full_fixed_vg")):      # min_* = MIN-plan templates (F0/X6), checked in test_x6.py; full_gated_{probe,empty,adaptive}* = X7 (checked in test_x7.py); *_vg = X9 (test_x9.py)
            continue
        assert cfg["ddl"]["quote_keywords"] is False and cfg["m5"]["policy"] == "p0" and cfg["route"]["score"] == "v3" and not cfg["m5"]["arbitrate"], f
        assert cfg["route"]["consistency"]["k"] == 0 and not cfg["m5"]["score_for_routing"], f
    for name in sorted(new - {"full_gated_cons_lr"}):
        s = make_sys(monkeypatch, PL.FullSystem, ScriptLLM(), C.load_config(CONFIGS / (name + ".yaml")))
        p = s.pipe
        assert p.qk == (name != "full_m3force"), name
        assert p.gated == name.startswith("full_gated"), name
    g = PL.resolved_cfg(C.load_config(CONFIGS / "full_gated.yaml"))
    assert g["m5"]["policy"] == "gated" and not g["m5"]["gate_empty"] and not g["m5"]["score_for_routing"] and g["route"]["score"] == "exec"
    gr = PL.resolved_cfg(C.load_config(CONFIGS / "full_gated_route.yaml"))
    assert gr["m5"]["score_for_routing"] and gr["route"]["score"] == "v3"
    gc = PL.resolved_cfg(C.load_config(CONFIGS / "full_gated_cons.yaml"))
    assert gc["route"]["consistency"] == {"k": 4, "temperature": 0.7, "n_per_call": 4} and gc["route"]["score"] == "v3+consistency"
    lr_cfg = C.load_config(CONFIGS / "full_gated_cons_lr.yaml")
    from nl2sql import paths as _paths
    coef = _paths.REPO / lr_cfg["route"]["coef_path"]
    if coef.exists():                                                                   # F1 froze exp/frozen/route_coef_hr.json: the config now loads
        make_sys(monkeypatch, PL.FullSystem, ScriptLLM(), lr_cfg)
    else:
        with pytest.raises((OSError, IOError)):                                         # coefficient file must exist before the run
            make_sys(monkeypatch, PL.FullSystem, ScriptLLM(), lr_cfg)


def test_lock_hashes_route_coefficients(tmp_path):
    from nl2sql import lock
    cfg = {"route": {"coef_path": "exp/frozen/definitely_missing_route.json"}}
    assert lock._artefact_hashes(cfg)["route_sha256"].startswith("MISSING:")
    assert lock._artefact_hashes({})["route_sha256"] is None
    e = lock.compute_entry("full_fixed", str(CONFIGS / "full_fixed.yaml"))
    assert e["route_sha256"] is None and lock.verify("full_fixed", "bird_train_HT", 0, str(CONFIGS / "full_fixed.yaml"))[0]


def test_run_py_passes_x5_fields_through(tmp_path):
    from nl2sql import run

    class Sys:
        def run(self, item, seed=0):
            return {"final_sql": "select 1", "abstained": True, "critic": {"score": 7.0}, "consistency": 0.5, "n_samples": 4, "n_agree": 2,
                    "cons_bits": [1, 0, 1, 0], "route_score": 6.0, "route_kind": "v3+consistency", "final_status": "ok_nonempty",
                    "arbitrated": 1, "arb_choice": "root", "arb_parse_ok": True, "arb_flip": 1, "confidence": 6.0}

    class Ev:
        gold = {}

        def score_sql(self, item, sql):
            return {"ex": 1, "exec_ok": True}

    it = Item(qid="x:d:1", db_id="d", question="q", evidence="", gold_sql="", db_path="", split="s", meta={})
    row, _ = run.process_item(Sys(), Ev(), it, 0, {}, {"system": "s", "model": "m"})
    for k in ("consistency", "n_samples", "n_agree", "cons_bits", "route_score", "route_kind", "final_status", "arbitrated", "arb_choice", "confidence"):
        assert k in row, k
    json.dumps(row)


# ------------------------------------------------------------------------------------------------ 6. fit tool
def synth_preds(path, dbs=6, n=120, seed=0, with_cons=True):
    rng = np.random.RandomState(seed)
    rows = []
    for d in range(dbs):
        for i in range(n):
            y = int(rng.rand() < 0.6)
            v3 = float(np.clip(rng.normal(9.0 if y else 8.4, 1.4), 0, 10))                 # weak signal
            bits = [int(rng.rand() < (0.9 if y else 0.35)) for _ in range(4)]                  # strong signal
            r = {"qid": "bird_train:db%d:%d" % (d, i), "db_id": "db%d" % d, "ex": y, "critic": {"score": v3, "semantic": 8, "syntax": 10.0},
                 "final_status": "ok_nonempty" if rng.rand() < 0.9 else "ok_empty", "exec_ok": True}
            if with_cons:
                r.update({"consistency": sum(bits) / 4.0, "n_samples": 4, "cons_bits": bits})
            rows.append(r)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_fit_logistic_matches_sklearn_and_auroc():
    rng = np.random.RandomState(1)
    X = rng.randn(400, 3) * [1.0, 3.0, 0.5] + [0.0, 5.0, 1.0]
    y = (X @ [0.8, 0.2, -1.0] + rng.randn(400) > 0.0).astype(int)
    m = FR.fit_logistic(X, y, C=1.0)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError:
        pytest.skip("sklearn missing")
    Z = (X - X.mean(0)) / X.std(0)
    sk = LogisticRegression(C=1.0, tol=1e-10, max_iter=1000).fit(Z, y)
    assert np.allclose(m["w_std"], sk.coef_[0], atol=2e-3) and abs(m["b_std"] - sk.intercept_[0]) < 2e-3
    p = FR.predict(m, X)
    assert np.allclose(p, sk.predict_proba(Z)[:, 1], atol=2e-3)                     # raw-scale coefficients reproduce the standardised model
    for s in (p, X[:, 0], np.round(X[:, 1])):                                          # ties included
        assert FR.auroc(s, y) == pytest.approx(roc_auc_score(y, s), abs=1e-9)


def test_fit_tool_end_to_end_lodo_never_trains_on_heldout_db(tmp_path):
    hr, ht, out = tmp_path / "hr.jsonl", tmp_path / "ht.jsonl", tmp_path / "coef.json"
    synth_preds(hr, seed=0)
    synth_preds(ht, dbs=3, seed=7)
    res = FR.main(["--preds", str(hr), "--eval-preds", str(ht), "--out", str(out), "--boot", "100"])
    saved = json.load(open(out))
    assert saved["features"] == ["v3", "consistency", "nonempty"] and saved["link"] == "logistic" and saved["fit"]["n"] == 720
    assert saved["coef"][1] > 0 and saved["coef"][0] > 0 and saved["inputs"][0]["sha256"]
    name = [k for k in saved["lodo"] if k.startswith("lr_lodo")][0]
    m = saved["lodo"][name]
    assert m["auroc"] > saved["lodo"]["v3"]["auroc"] + 0.05 and m["auroc_gain_vs_v3"]["ci"][0] > 0 and m["aurc"] < saved["lodo"]["v3"]["aurc"]
    assert saved["success_rule_met"] is True and saved["heldout"]["success_rule_met"] is True
    assert saved["heldout"]["metrics"]["lr_fitted_on_train"]["auroc"] > saved["heldout"]["metrics"]["v3"]["auroc"]
    # LODO prediction of a db equals a model fitted without that db
    X, y, dbs, _, _ = FR.load_rows([str(hr)], ["v3", "consistency", "nonempty"])
    p = FR.lodo_predict(X, y, dbs)
    te = dbs == "db2"
    ref = FR.predict(FR.fit_logistic(X[~te], y[~te]), X[te])
    assert np.allclose(p[te], ref)
    # the JSON is directly usable as route.coef_path and reproduces the sigmoid of the fitted model
    rs = RouteScorer({"score": "v3+consistency", "coef_path": str(out), "consistency": {"k": 4}})
    f = {"v3": 9.0, "consistency": 1.0, "nonempty": 1.0}
    assert rs.score(f) == pytest.approx(10.0 * FR.predict(FR.fit_logistic(X, y), np.array([[9.0, 1.0, 1.0]]))[0], abs=2e-3)
    # --k uses the first k sample bits
    X2, _, _, _, _ = FR.load_rows([str(hr)], ["consistency"], k=2)
    assert set(np.unique(X2)) <= {0.0, 0.5, 1.0}
    # rows without the needed feature are dropped and counted
    bad = tmp_path / "nocons.jsonl"
    synth_preds(bad, with_cons=False)
    with pytest.raises(SystemExit):
        FR.main(["--preds", str(bad), "--out", str(tmp_path / "x.json"), "--boot", "10"])


# ------------------------------------------------------------------------------------------------ 7. arbitration probe tool
def test_arbitrate_probe_on_toy_trees(monkeypatch, toydb):
    item = mk_item(toydb, "bird_train:t:1")
    llm = ScriptLLM(arb=judge_prefers("sum("))
    pipe = PL.Pipeline(llm, "m", {})
    nodes = [{"id": 0, "parent": None, "action": "root", "sql": GOOD, "ex": 0},
             {"id": 1, "parent": 0, "action": "re_retrieve", "sql": 'SELECT count(id) FROM "transaction"', "ex": 0},     # same result as root: no call
             {"id": 2, "parent": 0, "action": "revise", "sql": 'SELECT sum(amt) FROM "transaction"', "ex": 1},
             {"id": 3, "parent": 2, "action": "revise", "sql": "SELECT 2", "ex": 0}]                                    # not depth 1
    acts = ["revise", "re_retrieve"]
    rec = AP.probe_question(pipe, item, {"qid": item.qid, "db_id": "t", "nodes": nodes}, acts, "# Schema\nCREATE TABLE x (a);")
    assert rec["calls"] == 1 and rec["judged_ex"] == 1 and rec["root_ex"] == 0 and rec["revise_always_ex"] == 1 and rec["picks"][0]["pick"] == "cand"
    wrong = ScriptLLM(arb=judge_prefers("count("))
    rec2 = AP.probe_question(PL.Pipeline(wrong, "m", {}), item, {"qid": item.qid, "db_id": "t", "nodes": nodes}, acts, "# Schema\n")
    assert rec2["judged_ex"] == 0 and rec2["picks"][0]["pick"] == "champ"
    recs = [dict(rec, db="a", qid="1"), dict(rec2, db="b", qid="2"), dict(rec, db="b", qid="3")]
    s = AP.summarize(recs, boot=50)
    assert s["n_mixed"] == 3 and s["judge_acc_on_mixed"] == pytest.approx(2 / 3.0) and s["revise_always_acc_on_mixed"] == 1.0
    assert s["ex_judged_ex"] == pytest.approx(2 / 3.0) and s["judge_calls"] == 3


# ------------------------------------------------------------------------------------------------ 8. n-sample plumbing, timeout gate
def test_llm_n_samples_one_request_list_text_and_cache(tmp_path):
    import httpx
    from nl2sql.llm import LLMClient
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        n = body.get("n", 1)
        return httpx.Response(200, json={"choices": [{"message": {"content": "SELECT %d" % i}} for i in range(n)],
                                         "usage": {"prompt_tokens": 100, "completion_tokens": 5 * n}})

    kf = tmp_path / "k.env"
    kf.write_text("QWEN_API_KEY=sk-TEST\n")
    c = LLMClient(str(tmp_path / "cache"), str(tmp_path / "usage.jsonl"), key_file=str(kf))
    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    msgs = [{"role": "user", "content": "q"}]
    a = c.chat("qwen3.7-plus-2026-05-26", msgs, temperature=0.7, seed=1, n=4, tag="cons")
    b = c.chat("qwen3.7-plus-2026-05-26", msgs, temperature=0.7, seed=1, n=4, tag="cons")
    one = c.chat("qwen3.7-plus-2026-05-26", msgs, temperature=0.7, seed=1, tag="cons")
    assert len(bodies) == 2 and bodies[0]["n"] == 4 and "n" not in bodies[1]                 # ONE request for four samples
    assert a["text"] == ["SELECT 0", "SELECT 1", "SELECT 2", "SELECT 3"] and b["cached"] and b["text"] == a["text"] and one["text"] == "SELECT 0"
    assert a["completion_tokens"] == 20 and a["prompt_tokens"] == 100                            # prompt billed once, completions x4
    # the n = 1 cache key is exactly the pre-X5 key (no "n" in params), so no existing cache row is orphaned
    assert "n" not in LLMClient._params(0.7, 1, 2048, False, 1) and LLMClient._params(0.7, 1, 2048, False, 4)["n"] == 4


def test_gated_timeout_triggers_revision(monkeypatch, toydb):
    real = db.execute

    def fake_exec(path, sql, timeout_s=30, max_rows=10000):
        if sql == BAD:
            return {"ok": False, "rows": [], "cols": [], "error": "timeout: exceeded 30s", "elapsed_s": 30.0, "truncated": False}
        return real(path, sql, timeout_s=timeout_s, max_rows=max_rows)

    monkeypatch.setattr(PL.db, "execute", fake_exec)
    llm = ScriptLLM(gen=BAD, rev=GOOD)
    out = make_sys(monkeypatch, PL.FullGated, llm, GATED).run(mk_item(toydb))
    assert out["candidates"][0]["status"] == "timeout" and out["actions"] == ["revise", "accept"] and llm.tags() == ["gen", "rev"]
    assert "Execution error: timeout: exceeded 30s" in llm.calls[1]["user"]


class HashEmbedLLM(ScriptLLM):
    """ScriptLLM + deterministic hash embeddings / word-overlap rerank, so the REAL Resources / DbStore / M3 path runs offline."""

    def embed_np(self, texts, model):
        out = []
        for t in texts:
            rs = np.random.RandomState(int(hashlib.md5(t.encode()).hexdigest()[:8], 16))
            out.append(rs.randn(32).astype(np.float32))
        return np.stack(out)

    def rerank(self, q, docs, model):
        qs = set(q.lower().split())
        return [len(qs & set(d.lower().split())) / (len(qs) + 1.0) for d in docs]


def test_real_m3_path_beer_factory_quoting_and_gated_repair(tmp_path, monkeypatch):
    """End to end on the real BIRD-train beer_factory db (HEADROOM row 1: unquoted `transaction` -> syntax error): real Resources, DbStore, M3.
    Legacy prompt shows `CREATE TABLE transaction`; the quoted prompt shows the quoted name (full-schema fallback AND routed context)."""
    monkeypatch.setattr(paths, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(data, "_SCHEMA_MEM", {})
    try:
        items = [i for i in data.load_split("bird_train_HR") if i.db_id == "beer_factory"]
    except Exception as e:
        pytest.skip("BIRD train data not available: %s" % e)
    if not items:
        pytest.skip("beer_factory not in the HR split here")
    item = items[0]
    base = {"m1": {"enabled": False}, "m2": {"history": False, "terms": False}}
    ok_sql = 'SELECT count(*) FROM "transaction"'
    for fallback, qk in ((6000, False), (6000, True), (0, True)):
        cfg = C.deep_update(base, {"ddl": {"quote_keywords": qk}, "m3": {"schema_fallback_tokens": fallback}, "m5": {"policy": "gated"},
                                   "route": {"score": "exec"}})
        llm = HashEmbedLLM(gen='SELECT count(*) FROM transaction', rev=ok_sql)
        out = FullGatedFactory(llm, cfg).run(item)
        user = llm.calls[0]["user"]
        assert out["ctx_full_schema"] == (fallback == 6000)
        if fallback == 6000:                                                  # full schema: the table is certainly shown
            assert (('CREATE TABLE "transaction" (' in user) == qk) and (("CREATE TABLE transaction (" in user) == (not qk))
        elif 'transaction' in user:
            assert 'CREATE TABLE "transaction" (' in user
        assert ("Quote SQLite reserved words" in user) == qk
        # the model copies the unquoted name from the prompt (as in the HR run): execution error -> one revision with the error text
        assert llm.tags() == ["gen", "rev"] and out["actions"] == ["revise", "accept"] and out["final_status"] == "ok_nonempty"
        assert 'near "transaction": syntax error' in llm.calls[1]["user"]
        assert out["critic"]["semantic"] is None and "crit" not in llm.tags()


def FullGatedFactory(llm, cfg):
    return PL.FullGated(llm, "m", cfg)


def test_paired_compare_tool(tmp_path):
    import paired_compare as PC
    for nm, exs, st in (("a", [0, 0, 1, 1, 0, 1], ["err_syntax", "err_syntax", "ok_nonempty", "ok_nonempty", "ok_empty", "ok_nonempty"]),
                        ("b", [1, 1, 1, 1, 0, 0], ["ok_nonempty"] * 5 + ["ok_empty"])):
        d = tmp_path / nm
        d.mkdir()
        with open(d / "preds.jsonl", "w") as f, open(d / "cands.jsonl", "w") as g:
            for i, (e, s) in enumerate(zip(exs, st)):
                qid = "bird_train:%s:%d" % ("d1" if i < 3 else "d2", i)
                f.write(json.dumps({"qid": qid, "db_id": "d1" if i < 3 else "d2", "ex": e, "calls": 3 if nm == "a" else 2, "exec_ok": s.startswith("ok"),
                                    "abstained": False}) + "\n")
                g.write(json.dumps({"qid": qid, "cand_idx": 0, "status": s, "ex": e}) + "\n")
    r = PC.compare(str(tmp_path / "a" / "preds.jsonl"), str(tmp_path / "b" / "preds.jsonl"), boot=50, label_a="p0", label_b="gated")
    assert r["n_pairs"] == 6 and r["p0"]["ex"] == 0.5 and r["gated"]["ex"] == pytest.approx(4 / 6.0)
    assert r["p0"]["root"]["exec_error_or_timeout"] == pytest.approx(2 / 6.0) and r["gated"]["root"]["exec_error_or_timeout"] == 0.0
    assert r["diff_pp"]["b_right_a_wrong"] == 2 and r["diff_pp"]["a_right_b_wrong"] == 1 and r["diff_pp"]["b_minus_a"] == pytest.approx(100 / 6.0)
    assert r["p0"]["calls_per_q"] == 3 and r["gated"]["calls_per_q"] == 2
    only = PC.compare(str(tmp_path / "a" / "preds.jsonl"), str(tmp_path / "b" / "preds.jsonl"), {"d1"}, boot=10)
    assert only["n_pairs"] == 3


def test_registry_has_full_gated():
    from nl2sql.systems import get_system
    assert get_system("full_gated") is PL.FullGated and get_system("full_fixed") is PL.FullFixed
    assert PL.FullGated.cfg_override == {"m5": {"policy": "gated"}} and PL.FullGated.force_p0
