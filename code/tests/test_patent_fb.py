"""Offline tests of patent_fb (SAP Amendment 9, section 22): scripted fake LLM, toy sqlite database, no API call, no dev data.
Covers: (1) nothing existing changed (pinned source / prompt / models hashes, no `pf` key anywhere in existing configs, registry, rows),
(2) the n = 0 prompt is the frozen VA prompt, (3) the feedback store (nested subsets, noise, leakage assertions), (4) the Critic call and score,
(5) the P0 loop with the new Critic."""
import hashlib
import json
import os
import re
import sqlite3
import sys

import pytest

from nl2sql import config as C, paths, prompts_util as P
from nl2sql.data import Item, render_ddl
from nl2sql.systems import get_system
from nl2sql.systems.ours import m3 as M3, m5 as M5, pipeline as PL, patent_fb as PF
from nl2sql.systems.ours.defaults import DEFAULTS

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "nl2sql", "systems", "ours")

# sha256 of the existing source files at the commit before patent_fb (they must stay byte-identical)
PINNED = {
    "pipeline.py": "7094fdb7377169816ab4952fdb6292cd4dfcaa92700f86684bcb31c63dc72145",
    "m5.py": "bafdaaa1df79e79fe2898bfae33b006dc927336f19cd3c57b5eb22af14bc7ba7",
    "defaults.py": "f090368bc5799c46bf3d95878447e5e4619f2c7e6b645f205ec0a82dfe7f4446",
    "corpus.py": "2535afa50f75ae469b8abdf6f298b8034a245e7b40e877cb49b499e106bb182c",
    "m3.py": "fff02bdd49cd3043c366d22f24d07be18e5d2e0bce737d5f37202d251f6372e8",
    "m1.py": "ae8ed7cd50ac3fec83e9fe1ca21d65ddfabb2ca3a1f844f98f067af6b5c63609",
}
PROMPTS_SHA = "a61f8211c4372dd6d1bbca6848ce52949dfce0c87fc50e82905fe89566c357e0"
MODELS_SHA = "5131de632eb8ae2f35f4c9c5a1f33d1d9d943a7796ecf02aeb5afe0145d6704b"
# canonical-json sha256 of the 16 dev-lock entries that existed before Amendment 9 (they must stay byte-identical after `lock add`)
LOCK16 = ["197dca6c0f7dcbda1b72527bca2b9f10eb372f8a016dc3c4a959b67cb24de5eb", "dd6bc846fca4c03fc2cf182075f5980c47e6c3437aa1106f1f6b04f1cf5fbd60",
          "ccf801482fd0d2c854f0c9d53ad1b82a9557353016957816fec78b6ed1dee0fe", "3772723c3531218ef142b191777b04cbd9c1e3059c788928493991198e528d51",
          "7a92320856df29db9b9d59bf182ef8751c36670e0920317d832fed09e9c96f63", "8715c1ce12f9d5260f09d8ba262797867fa087fe67e804d00cd1b6005c3c8562",
          "838f8df1021dd33a44b862c38f8e9358175a8e6839ba4d699550b4c517731e9e", "ed4d3ab9c13348802f24b1da7fd4c082b526c5c00d396785f4330143bd0e4201",
          "9965dc85df616b9c2b993284add225c68f7a958ef1ff6a0e91b217f0d29901a4", "0810a07308dc79ae6f9d91b7bf9a0cd2b6d15ccd79ceaa04cfd8de6c1e52f947",
          "52f8b787579045f930198deee46d7ab5c8ad042d15402d3afc5c70feeef77100", "60e255f4f9682fb3100bfe566e366f0d79d44427d310228272df270e1d8437b2",
          "778dfd9c28148725cc29b9ce097bc81818220f892899f100cffe89da6a01573c", "f10bfb91785c2a013db26ef0c398f5145f29c757979f14392248ad11d36ba4d1",
          "8de53d2d1a306e7b305b26bf648a1b289a627477b78b2320615645f2bad4d793", "7ac79bbf783d49cab0f8d23899556da1792f150801ccef37ab199a820bc1895f"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ------------------------------------------------------------------------------------------------ fakes (as in test_x5)
@pytest.fixture()
def toydb(tmp_path):
    p = str(tmp_path / "toy.sqlite")
    con = sqlite3.connect(p)
    con.execute('CREATE TABLE "transaction" (id INTEGER PRIMARY KEY, amt REAL, "order" TEXT)')
    con.executemany('INSERT INTO "transaction" VALUES (?,?,?)', [(1, 5.0, "a"), (2, 7.5, "b"), (3, 1.0, "c")])
    con.commit()
    con.close()
    return p


def mk_item(db_path, qid="bird_dev:t:1"):
    return Item(qid=qid, db_id="t", question="how many transactions are there?", evidence="", gold_sql="SELECT 'GOLDMARK'", db_path=db_path, split="toy", meta={})


class ScriptLLM:
    def __init__(self, gen=None, rev=None, crit=None):
        self.h = {"gen": gen, "rev": rev, "crit": crit}
        self.calls = []

    def chat(self, model, messages, *, temperature=0.0, seed=None, max_tokens=2048, n=1, tag="", **kw):
        user = messages[-1]["content"]
        self.calls.append({"tag": tag, "max_tokens": max_tokens, "user": user, "messages": messages})
        h = self.h.get(tag)
        if h is None:
            raise AssertionError("unexpected LLM call with tag %r" % tag)
        text = h(user, len(self.calls)) if callable(h) else h
        return {"text": text, "prompt_tokens": 1, "completion_tokens": 1, "latency_s": 0.0, "cached": False}

    def tags(self):
        return [c["tag"] for c in self.calls]


class StubStore:
    schema = {"tables": {"transaction": [{"name": "id", "type": "INTEGER"}, {"name": "amt", "type": "REAL"}, {"name": "order", "type": "TEXT"}]},
              "row_counts": {"transaction": 3}}


class StubRes:
    def db_store(self, item):
        return StubStore()

    def corpus_for(self, item):
        return None

    def ensure_terms(self, item):
        pass

    def embed_one(self, text):
        return [0.0]


def fake_retrieve(res, item, graph, cfg, corpus, **kw):
    sc = {"tables": {"transaction": [{"name": "id", "type": "INTEGER", "desc": "", "examples": [], "pk": 1},
                                     {"name": "amt", "type": "REAL", "desc": "", "examples": []},
                                     {"name": "order", "type": "TEXT", "desc": "", "examples": []}]}, "fks": []}
    return M3.Context(tables=["transaction"], ddl=render_ddl(sc, quote_keywords=True), full_schema=True)


class FakeEntry:
    def __init__(self, qid, q="q", ev="e", sql="SELECT 1"):
        self.id, self.kind, self.origin, self.question, self.evidence, self.sql = "bi:" + qid, "exp", "burnin", q + " " + qid, ev, sql


class FakeCorpus:
    """order / entries / recall like ExperienceCorpus; recall returns the first k entries that are not excluded (so exclusion is observable)."""

    def __init__(self, qids, extra=()):
        self.entries = {}
        for x in qids:
            e = FakeEntry(x)
            self.entries[e.id] = e
        for x in extra:
            self.entries[x] = type("H", (), {"id": x, "kind": "history", "origin": "pool", "question": "h", "evidence": "", "sql": "gold"})()
        self.order = list(self.entries)
        self.last_exclude = None

    def recall(self, rerank_model, question, qv, k, exclude=(), rerank=True, hybrid=True):
        self.last_exclude = set(exclude)
        return [self.entries[i] for i in self.order if i not in exclude][:k], 1


QIDS = ["bird_train:db%d:%d" % (i % 7, i) for i in range(1218)]
ORACLE = {x: (1 if i % 6 else 0) for i, x in enumerate(QIDS)}


def pf_cfg(**over):
    pf = {"n": 0, "q": 1.0, "seed": 20261128}
    pf.update(over)
    return {"m1": {"enabled": False}, "ddl": {"quote_keywords": True}, "pf": pf}


def make_sys(monkeypatch, llm, **over):
    monkeypatch.setattr(PL.M3, "retrieve", fake_retrieve)
    s = PF.PatentFB(llm, "m", pf_cfg(**over))
    s.pipe.res = StubRes()
    return s


# ------------------------------------------------------------------------------------------------ 1. nothing existing changed
def test_existing_sources_prompts_models_unchanged():
    for f, h in PINNED.items():
        assert sha(os.path.join(SRC, f)) == h, f
    assert C.prompts_sha() == PROMPTS_SHA
    assert C.models_sha() == MODELS_SHA


def test_defaults_registry_and_configs_untouched():
    assert "pf" not in DEFAULTS and "pf" not in DEFAULTS["m5"]
    import glob
    n = 0
    for f in glob.glob(str(paths.EXP_DIR / "configs" / "**" / "*.yaml"), recursive=True):
        if os.sep + "pf" + os.sep in f:
            continue
        cfg = C.load_config(f)
        assert "pf" not in cfg, f
        n += 1
    assert n > 30
    for name, cls in {"full": PL.FullSystem, "full_fixed": PL.FullFixed, "full_gated": PL.FullGated, "full_gated_blind": PL.FullGatedBlind,
                      "full_rl": PL.FullRL, "full_x": PL.FullX}.items():
        assert get_system(name) is cls
    assert get_system("patent_fb") is PF.PatentFB and issubclass(PF.PatentFB, PL.FullFixed)


def test_dev_lock_original_16_entries_byte_identical():
    d = json.load(open(str(paths.DEV_LOCK)))["entries"]
    got = [hashlib.sha256(json.dumps(e, sort_keys=True).encode()).hexdigest() for e in d[:16]]
    assert got == LOCK16
    assert [e["system"] for e in d[16:20]] == ["patent_fb"] * 4          # the four patent_fb entries follow them
    for e in d[20:]:                                   # later entries are the lock entries of the additional experiments (2026-10-28: replication seeds, controls)
        assert e["date"] >= "2026-10-28"


def test_full_fixed_has_stock_critic_and_no_pf_key(monkeypatch, toydb):
    llm = ScriptLLM(gen='SELECT count(*) FROM "transaction"', crit='{"semantic": 9, "issues": [], "suggestions": []}')
    monkeypatch.setattr(PL.M3, "retrieve", fake_retrieve)
    s = PL.FullFixed(llm, "m", {"m1": {"enabled": False}, "ddl": {"quote_keywords": True}})
    s.pipe.res = StubRes()
    assert type(s.pipe.critic) is M5.Critic
    out = s.run(mk_item(toydb))
    assert "pf" not in out
    assert "max_tokens" in llm.calls[1] and llm.calls[1]["max_tokens"] == 512          # stock critic budget unchanged


def test_run_py_row_copies_pf_only_when_present():
    src = open(os.path.join(HERE, "..", "nl2sql", "run.py")).read()
    assert src.count('"pf"):') == 1 and "if k in res:" in src


# ------------------------------------------------------------------------------------------------ 2. prompt
def test_n0_prompt_equals_frozen_va_prompt():
    va = open(str(paths.EXP_DIR / "diag_prompts" / "va_grounded.user.txt")).read()
    tpl = open(str(paths.EXP_DIR / "pf_prompts" / "pf_critic.user.txt")).read()
    kw = dict(sections="S", full_schema="T(a int)", question="Q?", evidence="E", graph="G", sql="SELECT 1", preview="P")
    assert PF.render_text(tpl, standard="", **kw) == PF.render_text(va, **kw)
    names = set(re.findall(r"<<(\w+)\??>>", tpl))
    assert names <= {"sections", "full_schema", "standard", "question", "evidence", "graph", "sql", "preview"}


def test_standard_section_content_and_no_gold_marker():
    std = open(str(paths.EXP_DIR / "pf_prompts" / "pf_standard.txt")).read()
    assert set(re.findall(r"<<(\w+)\??>>", std)) == {"cases"}
    low = std.lower()
    assert "gold" not in low and "reference answer" not in low and "simulated" not in low


# ------------------------------------------------------------------------------------------------ 3. feedback store
def test_store_nested_subsets_deterministic_and_noise():
    a = PF.FeedbackStore({x: None for x in QIDS}, ORACLE, 250, 1.0, 7)
    b = PF.FeedbackStore({x: None for x in QIDS}, ORACLE, 1000, 1.0, 7)
    c = PF.FeedbackStore({x: None for x in QIDS}, ORACLE, 1000, 1.0, 7)
    assert set(a.qids) < set(b.qids) and b.qids == c.qids and len(a.qids) == 250 and len(b.qids) == 1000
    assert all(b.verdict[x] == ORACLE[x] for x in b.qids)                      # q = 1: the oracle
    d = PF.FeedbackStore({x: None for x in QIDS}, ORACLE, 1000, 0.85, 7)
    assert d.qids == b.qids
    flips = sum(d.verdict[x] != ORACLE[x] for x in d.qids)
    assert 110 < flips < 190                                                    # about 15 % of 1000
    assert d.stats()["flipped"] == flips
    e = PF.FeedbackStore({x: None for x in QIDS}, ORACLE, 250, 0.85, 7)         # the verdict of an entry does not depend on n
    assert all(e.verdict[x] == d.verdict[x] for x in e.qids)
    other = PF.FeedbackStore({x: None for x in QIDS}, ORACLE, 250, 1.0, 8)
    assert other.qids != a.qids                                                  # the seed changes the draw
    with pytest.raises(ValueError):
        PF.FeedbackStore({x: None for x in QIDS}, ORACLE, 2000, 1.0, 7)


def test_build_store_asserts_no_dev_diag_or_tuning_qid(tmp_path):
    corp = FakeCorpus(QIDS[:50], extra=["bird_train:hist:1"])
    pf = dict(PF.PF_DEFAULTS, n=10)
    st, size = PF.build_store(corp, ORACLE, pf)
    assert size == 50 and len(st.qids) == 10 and all(("bi:" + x) in corp.entries for x in st.qids)
    # history (gold) entries are never in the store
    assert not any(i.startswith("bird_train:hist") for i in st.ids)
    # a forbidden qid (DIAG / tuning file) inside the store is an error
    f = tmp_path / "forb.json"
    f.write_text(json.dumps({"items": [{"qid": QIDS[3]}, {"qid": "bird_train:zzz:9"}]}))
    with pytest.raises(AssertionError):
        PF.build_store(corp, ORACLE, dict(pf, forbid_files=[str(f)]))
    (tmp_path / "ok.json").write_text(json.dumps({"qids": ["bird_train:other:1"]}))
    PF.build_store(corp, ORACLE, dict(pf, forbid_files=[str(tmp_path / "ok.json")]))
    # a dev qid in the store is an error even without a forbid file
    bad = FakeCorpus(QIDS[:5] + ["bird_dev:x:1"])
    with pytest.raises(AssertionError):
        PF.build_store(bad, dict(ORACLE, **{"bird_dev:x:1": 1}), pf)


def test_memory_cases_exclude_non_store_and_own_qid(monkeypatch, toydb):
    corp = FakeCorpus(QIDS[:30], extra=["bird_train:hist:1", "bird_train:hist:2"])
    st, _ = PF.build_store(corp, ORACLE, dict(PF.PF_DEFAULTS, n=8, q=0.85))
    s = make_sys(monkeypatch, ScriptLLM())
    cr = PF.FBCritic(s.pipe.critic, s.pipe, dict(PF.PF_DEFAULTS), st, corp, "T", "S")
    item = mk_item(toydb, qid="bird_train:dbX:77")
    cases = cr.cases(item)
    assert len(cases) == 4 and all(c["id"] in st.ids for c in cases)
    assert all(i not in corp.last_exclude for i in st.ids if i != "bi:" + item.qid)
    assert "bird_train:hist:1" in corp.last_exclude and "bi:" + item.qid in corp.last_exclude and item.qid in corp.last_exclude
    assert all(c["verdict"] == st.verdict[c["qid"]] and c["oracle"] == ORACLE[c["qid"]] for c in cases)
    assert cr.cases(item) is cases                                              # cached per qid
    cr0 = PF.FBCritic(s.pipe.critic, s.pipe, dict(PF.PF_DEFAULTS), None, None, "T", "S")
    assert cr0.cases(item) == []


# ------------------------------------------------------------------------------------------------ 4. critic call and score
CRIT_BAD = json.dumps({"semantic": 4, "issues": [{"severity": "must_fix", "question_phrase": "how many", "sql_fragment": "count(*)", "description": "wrong"},
                                                   {"severity": "style", "question_phrase": "", "sql_fragment": "", "description": "alias"}],
                       "suggestions": ["use count(distinct id)"]})
CRIT_OK = '{"semantic": 10, "issues": [], "suggestions": []}'


def test_score_is_stock_weighted_sum_and_prompt_has_no_standard_at_n0(monkeypatch, toydb):
    llm = ScriptLLM(crit=CRIT_BAD)
    s = make_sys(monkeypatch, llm)
    item = mk_item(toydb)
    res = s.pipe._exec(item, 'SELECT count(*) FROM "transaction"')
    ctx = fake_retrieve(None, item, None, s.rcfg, None)
    out = s.pipe.critic.score(item, 'SELECT count(*) FROM "transaction"', ctx, None, res, 0, {"transaction": 3})
    assert out["syntax"] == 10.0 and out["semantic"] == 4.0
    assert out["score"] == round(0.3 * 10 + 0.5 * 4 + 0.2 * out["efficiency"], 4)
    assert len(out["issues"]) == 1 and out["issues"][0]["sql_fragment"] == "count(*)" and len(out["style_issues"]) == 1
    assert out["suggestions"] == ["use count(distinct id)"]
    user = llm.calls[0]["user"]
    assert "Scoring standard" not in user and "ACCEPTED" not in user and "Complete list of tables" in user
    assert llm.calls[0]["max_tokens"] == 900 and llm.calls[0]["tag"] == "crit"
    assert "transaction(id INTEGER, amt REAL, order TEXT)" in user                    # complete listing from the full schema


def test_unparseable_critic_output_retries_once_then_neutral(monkeypatch, toydb):
    llm = ScriptLLM(crit="not json at all")
    s = make_sys(monkeypatch, llm)
    item = mk_item(toydb)
    res = s.pipe._exec(item, 'SELECT count(*) FROM "transaction"')
    out = s.pipe.critic.score(item, 'SELECT count(*) FROM "transaction"', fake_retrieve(None, item, None, s.rcfg, None), None, res, 0, {})
    assert llm.tags() == ["crit", "crit"] and out["semantic"] == 5.0 and out["parse_ok"] is False and out["issues"] == []
    assert s.pipe.critic._pfail == {item.qid: 1}


def test_prompt_with_standard_shows_verdicts_and_never_gold(monkeypatch, toydb):
    corp = FakeCorpus(QIDS[:30])
    st, _ = PF.build_store(corp, ORACLE, dict(PF.PF_DEFAULTS, n=12))
    llm = ScriptLLM(crit=CRIT_OK)
    s = make_sys(monkeypatch, llm)
    cr = PF.FBCritic(s.pipe.critic, s.pipe, dict(PF.PF_DEFAULTS), st, corp, *PF.load_prompts(dict(PF.PF_DEFAULTS))[:2])
    item = mk_item(toydb, qid="bird_train:dbX:77")
    res = s.pipe._exec(item, 'SELECT count(*) FROM "transaction"')
    cr.score(item, 'SELECT count(*) FROM "transaction"', fake_retrieve(None, item, None, s.rcfg, None), None, res, 0, {})
    user = llm.calls[0]["user"]
    assert "# Scoring standard from user feedback" in user and user.count("Case ") == 4
    assert user.count("User verdict: ACCEPTED") + user.count("User verdict: REJECTED") == 4
    assert "GOLDMARK" not in user and "SELECT 1" in user                              # the fake entries' SQL, never the item's gold
    assert user.index("Scoring standard") < user.index("# Task")


# ------------------------------------------------------------------------------------------------ 5. the P0 loop with the new Critic
def test_p0_loop_unchanged_thresholds_and_reviser_gets_must_fix_only(monkeypatch, toydb):
    gens = []

    def crit(user, i):
        return CRIT_BAD if "count(*) FROM transaction" in user else CRIT_OK

    llm = ScriptLLM(gen="SELECT count(*) FROM transaction", crit=crit, rev=lambda u, i: 'SELECT count(*) FROM "transaction"')
    s = make_sys(monkeypatch, llm)
    out = s.run(mk_item(toydb))
    # root fails to execute (unquoted keyword): syntax 2 (identifier error -> 3) -> score < 8.5 -> revise; the revision scores 10 -> accept
    assert out["actions"][0] == "revise" and out["actions"][-1] in ("accept", "stop_best") and out["iterations"] == 1
    assert out["final_sql"] == 'SELECT count(*) FROM "transaction"'
    assert out["pf"] == {"n": 0, "q": 1.0, "parse_fail": 0, "cases": []}
    rev_user = [c for c in llm.calls if c["tag"] == "rev"][0]["user"]
    assert "use count(distinct id)" in rev_user and "alias" not in rev_user and "wrong (question:" in rev_user       # must_fix + suggestions, no style
    assert s.rcfg["m5"]["theta_acc"] == 8.5 and s.rcfg["m5"]["t_max"] == 3 and s.rcfg["m5"]["weights"] == [0.3, 0.5, 0.2]


def test_patent_fb_requires_p0_and_matching_hashes(monkeypatch):
    monkeypatch.setattr(PL.M3, "retrieve", fake_retrieve)
    with pytest.raises(ValueError):
        PF.PatentFB(ScriptLLM(), "m", dict(pf_cfg(), m5={"policy": "gated"}))
    with pytest.raises(ValueError):
        PF.PatentFB(ScriptLLM(), "m", pf_cfg(verdicts_sha256="0" * 64))
    d = os.path.join(str(paths.EXP_DIR), "pf_data", "burnin_verdicts.jsonl")
    rows = PF.load_verdicts(d)
    assert len(rows) == 1479 and all(q.startswith("bird_train:") for q in rows)


def test_prompt_files_match_frozen_json_when_present():
    fz = paths.EXP_DIR / "pf_prompts" / "FROZEN.json"
    if not fz.exists():
        pytest.skip("not frozen yet")
    want = json.load(open(str(fz)))["prompts"]
    for f, h in want.items():
        assert sha(str(paths.EXP_DIR / "pf_prompts" / f)) == h, f
    PF.load_prompts(dict(PF.PF_DEFAULTS))                                       # raises on a mismatch


def test_all_pf_configs_resolve_and_point_to_frozen_artefacts():
    import glob
    fs = sorted(glob.glob(str(paths.EXP_DIR / "configs" / "pf" / "patent_fb_*.yaml")))
    for f in fs:
        cfg = C.load_config(f)
        pf = cfg["pf"]
        assert cfg["m6"]["path"] == "exp/frozen/corpus_burnin_39e2477c.sqlite" and cfg["m6"]["update"] is False
        assert pf["n"] in (0, 250, 1000) and pf["q"] in (1.0, 0.85) and pf["k"] == 4
        assert pf["verdicts_sha256"] == sha(str(paths.REPO / pf["verdicts"]))
        assert cfg.get("ddl") == {"quote_keywords": True} and cfg.get("temperature") == 0.0 and "m5" not in cfg
        assert pf.get("forbid_files")
