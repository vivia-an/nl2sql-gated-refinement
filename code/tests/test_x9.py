"""X9 tests (offline: scripted fake LLM, toy sqlite databases, synthetic preds; no API call, no dev / Spider data).
Covers: value index (limits, cache, thread-safety), normalisation, matching tiers, ranking / caps, prompt block + optional placeholder (legacy
byte-identity with the switch off), M3 integration, pipeline logging, config / defaults, tools/vg_eval.py."""
import gzip
import json
import os
import re
import sqlite3
import sys
import threading

import pytest

from nl2sql import config as C, data, db, paths, prompts_util as P
from nl2sql.data import Item
from nl2sql.systems.ours import m3 as M3, pipeline as PL, values as VG
from nl2sql.systems.ours.defaults import DEFAULTS
from test_x5 import CRIT9, ScriptLLM, StubStore  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

CONFIGS = paths.EXP_DIR / "configs"


# ------------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(autouse=True)
def private_cache(tmp_path, monkeypatch):
    """Every test gets its own value-index cache directory and an empty in-memory index cache."""
    monkeypatch.setattr(paths, "CACHE", tmp_path / "cache")
    VG.clear_memory()
    yield
    VG.clear_memory()


@pytest.fixture()
def vdb(tmp_path):
    p = str(tmp_path / "v.sqlite")
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE person (id INTEGER PRIMARY KEY, full_name TEXT, city TEXT, income_group TEXT, notes TEXT, age INTEGER, pic BLOB, born DATE)")
    rows = [("Vicky Hartzler", "NOBLE", "High income", "x" * 100, 30, b"ab", "1960-01-01"),
            ("Roy Blunt", "Noble", "High income: OECD", "y", 40, b"cd", "1950-01-01"),
            ("Claire McCaskill", "New York", "Low income", "y", 50, None, "1953-01-01"),
            ("Mary Bishopville", "Bishopville", "Lower middle income", "z", 20, None, "1970-01-01"),
            ("O'Brien Smith", "St. Louis", "High income", "z", 21, None, "1971-01-01")]
    con.executemany("INSERT INTO person(full_name,city,income_group,notes,age,pic,born) VALUES (?,?,?,?,?,?,?)", rows)
    con.execute('CREATE TABLE "transaction" (id INTEGER PRIMARY KEY, "order" TEXT, status VARCHAR(10))')
    con.executemany('INSERT INTO "transaction"("order",status) VALUES (?,?)', [("Ice Cream", "Paid"), ("ice-cream", "Open"), ("Hot Dog", "Paid")])
    con.execute("CREATE TABLE empty_t (x TEXT)")
    con.commit()
    con.close()
    return p


def schema_of(p):
    return data._build_schema(p, "toy", "t")


def P_(**kw):
    q = {k: v for k, v in kw.items()}
    return VG.params({"m2": {"vg": q}}) if q else VG.params()


def lits(hs):
    return [(h["table"], h["column"], h["literal"]) for h in hs]


# ------------------------------------------------------------------------------------------------ 1. normalisation
def test_normalisation_case_punctuation_accents_stemming():
    assert VG.norm_tokens("NOBLE") == VG.norm_tokens("Noble") == ("noble",)
    assert VG.compact(VG.norm_tokens("E-mail")) == VG.compact(VG.norm_tokens("email ")) == "email"
    assert VG.compact(VG.norm_tokens("O'Brien")) == VG.compact(VG.norm_tokens("OBrien")) == "obrien"
    assert VG.norm_tokens("Café Zürich") == ("cafe", "zurich")
    assert VG.norm_tokens("Community Post Office ") == VG.norm_tokens("community post office")
    for a, b in (("Cities", "city"), ("boxes", "box"), ("customers", "customer"), ("teams", "team"), ("Playing", "play"), ("Watched", "watch")):
        assert VG.norm_word(a) == VG.norm_word(b), (a, b)
    assert VG.norm_word("glass") == "glass" and VG.norm_word("bus") == "bus" and VG.norm_word("2013s") == "2013s"       # ss / us / non-alphabetic words stay
    assert VG.norm_tokens("") == () and VG.norm_tokens(None) == ()


def test_sql_quote_and_literal_in_sql():
    assert VG.sql_quote("O'Brien") == "'O''Brien'"
    assert VG.literal_in_sql("Noble", "SELECT 1 FROM t WHERE c = 'Noble'")
    assert not VG.literal_in_sql("Noble", "SELECT 1 FROM t WHERE c = 'NOBLE'")                 # exact, case-sensitive: the stored spelling is the point
    assert VG.literal_in_sql("Noble", "SELECT 1 FROM t WHERE c LIKE '%Noble%'")
    assert VG.literal_in_sql("O'Brien", "SELECT 1 FROM t WHERE c = 'O''Brien'")
    assert not VG.literal_in_sql("Noble", "") and not VG.literal_in_sql("Noble", None)


# ------------------------------------------------------------------------------------------------ 2. index build and limits
def test_build_skips_non_text_declared_types(vdb):
    cols, _ = VG.build_columns(vdb, schema_of(vdb), P_())
    by = {(c["table"], c["column"]): c for c in cols}
    assert ("person", "age") not in by and ("person", "pic") not in by and ("person", "born") not in by and ("person", "id") not in by
    assert by[("person", "city")]["values"] == ["NOBLE", "Noble", "New York", "Bishopville", "St. Louis"] or set(by[("person", "city")]["values"]) == \
        {"NOBLE", "Noble", "New York", "Bishopville", "St. Louis"}
    assert by[("transaction", "order")]["status"] == "ok" and by[("transaction", "status")]["values"]       # keyword table / column, VARCHAR(10)
    assert by[("empty_t", "x")]["status"] == "empty" and by[("empty_t", "x")]["values"] == []


def test_build_drops_long_values_and_respects_max_len(vdb):
    cols, _ = VG.build_columns(vdb, schema_of(vdb), P_())
    notes = next(c for c in cols if c["column"] == "notes")
    assert "x" * 100 not in notes["values"] and set(notes["values"]) == {"y", "z"}                    # 100 characters > max_len 60
    cols, _ = VG.build_columns(vdb, schema_of(vdb), P_(max_len=12))
    assert "High income: OECD" not in next(c for c in cols if c["column"] == "income_group")["values"]   # 17 characters


def test_build_caps_distinct_per_column_and_total(tmp_path):
    p = str(tmp_path / "big.sqlite")
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE t (a TEXT, b TEXT)")
    con.executemany("INSERT INTO t VALUES (?,?)", [("a%d" % i, "b%d" % (i % 5)) for i in range(500)])
    con.commit()
    con.close()
    sc = data._build_schema(p, "toy", "t")
    cols, meta = VG.build_columns(p, sc, P_(max_distinct=50))
    by = {c["column"]: c for c in cols}
    assert by["a"]["status"] == "truncated" and len(by["a"]["values"]) == 50 and by["b"]["status"] == "ok" and len(by["b"]["values"]) == 5
    assert meta["status"] == {"truncated": 1, "ok": 1} and meta["n_values"] == 55
    cols, meta = VG.build_columns(p, sc, P_(max_total=20))
    assert sum(len(c["values"]) for c in cols) <= 20 and "total_cap" in meta["status"] or "truncated" in meta["status"]
    assert meta["n_values"] <= 20


def test_build_scans_only_scan_rows_of_a_table(tmp_path):
    p = str(tmp_path / "scan.sqlite")
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE t (a TEXT)")
    con.executemany("INSERT INTO t VALUES (?)", [("v%d" % i,) for i in range(1000)])
    con.commit()
    con.close()
    cols, _ = VG.build_columns(p, data._build_schema(p, "toy", "t"), P_(scan_rows=100))
    assert set(cols[0]["values"]) == {"v%d" % i for i in range(100)} and cols[0]["scanned"] == 100         # first rows only, documented bias


def test_build_time_budget_and_statement_timeout(vdb, monkeypatch):
    clock = {"t": 0.0}

    def now():
        clock["t"] += 10.0                                  # every look at the clock costs 10 virtual seconds
        return clock["t"]
    cols, meta = VG.build_columns(vdb, schema_of(vdb), P_(budget_s=25.0), now=now)
    st = [c["status"] for c in cols]
    assert "budget" in st and "ok" in st and meta["status"].get("budget", 0) >= 1                      # columns after the budget are skipped, earlier ones kept
    real = db.execute

    def slow(db_path, sql, timeout_s=30, max_rows=10000):
        if "person" in sql and "city" in sql:
            return {"ok": False, "rows": [], "cols": [], "error": "timeout: interrupted", "elapsed_s": 9.0, "truncated": False}
        return real(db_path, sql, timeout_s=timeout_s, max_rows=max_rows)
    monkeypatch.setattr(VG.db, "execute", slow)
    cols, meta = VG.build_columns(vdb, schema_of(vdb), P_())
    by = {c["column"]: c for c in cols}
    assert by["city"]["status"] == "timeout" and by["city"]["values"] == [] and by["full_name"]["status"] == "ok"      # one timeout does not stop the build


def test_small_tables_are_indexed_before_large_ones(tmp_path):
    p = str(tmp_path / "order.sqlite")
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE big (a TEXT)")
    con.executemany("INSERT INTO big VALUES (?)", [("x%d" % i,) for i in range(200)])
    con.execute("CREATE TABLE small (b TEXT)")
    con.executemany("INSERT INTO small VALUES (?)", [("s%d" % i,) for i in range(3)])
    con.commit()
    con.close()
    cols, _ = VG.build_columns(p, data._build_schema(p, "toy", "t"), P_(max_total=10))
    assert [c["table"] for c in cols] == ["small", "big"] and cols[0]["status"] == "ok" and len(cols[0]["values"]) == 3


def test_blob_stored_in_text_column_is_not_indexed(tmp_path):
    p = str(tmp_path / "blob.sqlite")
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE t (a TEXT)")
    con.executemany("INSERT INTO t VALUES (?)", [(b"\x00\x01",), ("ok",), (5,)])
    con.commit()
    con.close()
    cols, _ = VG.build_columns(p, data._build_schema(p, "toy", "t"), P_())
    assert cols[0]["values"] == ["ok", "5"]                                                         # typeof(col) = 'text' only: the BLOB is skipped, 5 became '5' by TEXT affinity


# ------------------------------------------------------------------------------------------------ 3. cache + thread safety
def test_index_is_cached_on_disk_and_reloaded(vdb, monkeypatch):
    calls = []
    real = VG.build_columns
    monkeypatch.setattr(VG, "build_columns", lambda *a, **k: calls.append(1) or real(*a, **k))
    sc = schema_of(vdb)
    i1 = VG.get_index(vdb, sc, P_())
    assert len(calls) == 1 and i1.meta["n_values"] > 0
    files = list((paths.CACHE / "vstore").glob("vidx_*.json.gz"))
    assert len(files) == 1
    d = json.load(gzip.open(files[0], "rt"))
    assert d["db_path"] == vdb and d["meta"]["status"]["ok"] >= 3 and d["cols"][0]["status"] in ("ok", "empty")
    assert VG.get_index(vdb, sc, P_()) is i1                                                         # in-memory hit
    VG.clear_memory()                                                                                # "new process": reads the file, no rebuild
    i2 = VG.get_index(vdb, sc, P_())
    assert len(calls) == 1 and i2 is not i1 and i2.hints("people in noble", None, P_())
    j = VG.get_index(vdb, sc, P_(max_len=30))                                                        # other build parameters -> other file
    assert len(calls) == 2 and len(list((paths.CACHE / "vstore").glob("vidx_*.json.gz"))) == 2
    # a changed database file (size / mtime) is a different key
    os.utime(vdb, (1, 1))
    VG.get_index(vdb, sc, P_())
    assert len(calls) == 3


def test_concurrent_first_use_builds_once(vdb, monkeypatch):
    n = []
    real = VG.build_columns

    def slow(*a, **k):
        n.append(1)
        import time
        time.sleep(0.2)
        return real(*a, **k)
    monkeypatch.setattr(VG, "build_columns", slow)
    sc = schema_of(vdb)
    out = []
    ts = [threading.Thread(target=lambda: out.append(VG.get_index(vdb, sc, P_()))) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(n) == 1 and len(out) == 8 and all(o is out[0] for o in out)
    assert not list((paths.CACHE / "vstore").glob("*.tmp"))                                          # atomic write left no temp file


def test_corrupt_cache_file_is_rebuilt(vdb):
    sc = schema_of(vdb)
    VG.get_index(vdb, sc, P_())
    f = next((paths.CACHE / "vstore").glob("vidx_*.json.gz"))
    f.write_bytes(b"not gzip")
    VG.clear_memory()
    assert VG.get_index(vdb, sc, P_()).hints("who is Hartzler", None, P_(loose_single=True))


# ------------------------------------------------------------------------------------------------ 4. matching
@pytest.fixture()
def idx(vdb):
    return VG.get_index(vdb, schema_of(vdb), P_())


def test_exact_match_is_case_punctuation_and_plural_insensitive(idx):
    h = idx.hints("How many people live in noble?", None, P_())
    assert {(x["column"], x["literal"]) for x in h} == {("city", "NOBLE"), ("city", "Noble")} and {x["kind"] for x in h} == {"exact"}   # both stored spellings
    assert [x["literal"] for x in idx.hints("people from New-York", None, P_())] == ["New York"]
    assert [x["literal"] for x in idx.hints("people from new york city", None, P_())][:1] == ["New York"]
    assert idx.hints("order of ice cream", None, P_())[0]["literal"] in ("Ice Cream", "ice-cream")
    assert {x["literal"] for x in idx.hints("two ice cream orders", None, P_())} == {"Ice Cream", "ice-cream"}      # ... and the plural
    assert [x["literal"] for x in idx.hints("born in St Louis", None, P_())] == ["St. Louis"]                      # punctuation
    assert [x["literal"] for x in idx.hints("who is OBrien Smith", None, P_())] == ["O'Brien Smith"]               # apostrophe


def test_siblings_and_contains(idx):
    h = idx.hints("People with High income", None, P_())
    assert [x["literal"] for x in h] == ["High income", "High income: OECD"] and h[0]["identical"] is True and h[1]["identical"] is False
    h = idx.hints("People with High income", None, P_(siblings=False))
    assert [x["literal"] for x in h] == ["High income"]
    h = idx.hints("Which party does Hartzler belong to?", None, P_())
    assert [(x["literal"], x["kind"]) for x in h] == [("Vicky Hartzler", "contains")]                              # the stored value is the full name
    assert idx.hints("Which party does Hartzler belong to?", None, P_(contains=False)) == []


def test_single_lowercase_word_matches_only_exactly(idx):
    assert idx.hints("which income group", None, P_()) == []                                                    # 'income' is generic: contained in 4 values
    assert idx.hints("people named hartzler", None, P_()) == []                                                 # lower-case single word: no contains
    assert idx.hints("people named hartzler", None, P_(loose_single=True))[0]["literal"] == "Vicky Hartzler"
    assert idx.hints("people with noble city", None, P_())[0]["literal"] in ("NOBLE", "Noble")                  # exact needs no capital letter ('female' = 'Female')


def test_bare_lowercase_word_matches_categorical_columns_only():
    ix = VG.ValueIndex.from_values({("t", "city"): ["Rain"] + ["N%d" % i for i in range(60)], ("t", "events"): ["Rain", "Fog"]})
    assert [(h["column"], h["literal"]) for h in ix.hints("days with rain", None, P_())] == [("events", "Rain")]            # 61 distinct > cat_max 50: a name-like column
    assert {h["column"] for h in ix.hints("days with Rain", None, P_())} == {"city", "events"}                           # spelled as a name: any column
    assert {h["column"] for h in ix.hints("days with rain", None, P_(cat_max=100))} == {"city", "events"}
    cols = [{"table": "t", "column": "events", "values": ["Rain", "Fog"], "status": "truncated", "scanned": 9}]
    assert VG.ValueIndex(cols).hints("days with rain", None, P_()) == []                                                  # a truncated column is never 'categorical'
    assert VG.ValueIndex(cols).hints("days with Rain", None, P_())[0]["literal"] == "Rain"


def test_generic_contains_is_dropped(vdb, tmp_path):
    words = "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta".split()
    ix = VG.ValueIndex.from_values({("t", "c"): ["Zephyr of %s" % w for w in words] + ["Zephyr"]})
    assert [h["literal"] for h in ix.hints("Which Zephyr", None, P_(siblings=False))] == ["Zephyr"]               # exact
    ix = VG.ValueIndex.from_values({("t", "c"): ["Zephyr of %s" % w for w in words]})
    assert ix.hints("Which Zephyr", None, P_()) == []                                                           # 'Zephyr' is in 8 > 6 values: generic
    assert len(ix.hints("Which Zephyr", None, P_(contains_max_values=10))) == 3                                  # allowed now; max_variants 3 per column
    assert len(ix.hints("Which Zephyr", None, P_(contains_max_values=10, max_variants=8, max_hints=20))) == 8
    # a short capitalised word that covers little of the value is not a hint; a lower-case n-gram needs to cover 50 % of the value
    assert ix.hints("Which Zep", None, P_()) == [] and ix.hints("which zephyr", None, P_(contains_max_values=10)) == []
    short = VG.ValueIndex.from_values({("t", "c"): ["Lake Placid Club Resort Hotel"]})
    assert short.hints("near lake placid", None, P_()) == [] and short.hints("near Lake Placid", None, P_())[0]["literal"] == "Lake Placid Club Resort Hotel"      # capitalised and >= 6 compact characters


def test_fuzzy_match_and_threshold(idx):
    h = idx.hints("Which city is Bishopvile", None, P_())
    assert [(x["literal"], x["kind"]) for x in h] == [("Bishopville", "fuzzy")] and 0.85 <= h[0]["sim"] < 1
    assert idx.hints("Which city is Bishopvile", None, P_(fuzzy=False)) == []
    assert idx.hints("Which city is Bishopvile", None, P_(fuzzy_threshold=0.99)) == []
    assert idx.hints("Which city is Zzzzzzzzz", None, P_()) == []                                              # nothing similar
    assert idx.hints("Which city is bishopvile", None, P_()) == []                                              # lower-case single word: no fuzzy (loose_single off)


def test_stop_words_numbers_and_short_tokens_do_not_match(vdb):
    ix = VG.ValueIndex.from_values({("t", "c"): ["The", "How", "Total", "2013", "1", "ab", "US", "Number", "List", "Id"]})
    assert ix.hints("How many of the total number? list id", None, P_()) == []
    assert ix.hints("in 2013 or 1", None, P_()) == [] and ix.hints("in 2013 or 1", None, P_(numbers=True))[0]["literal"] == "2013"      # digits >= 2 only
    assert [h["literal"] for h in ix.hints("people in the US", None, P_())] == ["US"]                          # 2-letter all-capital token
    assert ix.hints("people in us", None, P_()) == []


def test_shown_tables_restrict_the_hints(idx):
    assert idx.hints("Ice Cream and Noble", ["person"], P_()) and all(h["table"] == "person" for h in idx.hints("Ice Cream and Noble", ["person"], P_()))
    assert idx.hints("Ice Cream and Noble", ["transaction"], P_())[0]["table"] == "transaction"
    assert idx.hints("Ice Cream and Noble", [], P_()) == []
    assert {h["table"] for h in idx.hints("Ice Cream and Noble", None, P_())} == {"person", "transaction"}
    assert idx.hints("Ice Cream", ["TRANSACTION"], P_())                                                         # table names compare case-insensitively


def test_longer_span_wins_and_no_overlapping_hints(idx):
    ix = VG.ValueIndex.from_values({("t", "name"): ["Los Angeles"], ("t", "first"): ["Los"], ("t", "second"): ["Angeles"]})
    h = ix.hints("trips to Los Angeles", None, P_())
    assert [(x["column"], x["literal"]) for x in h] == [("name", "Los Angeles")]                                # the 2-gram covers the 1-grams


def test_caps_max_hints_cols_and_variants():
    vals = {("t", "c%d" % i): ["Alpha"] for i in range(6)}
    ix = VG.ValueIndex.from_values(vals)
    assert len(ix.hints("Alpha", None, P_())) == 2                                                              # max_cols_per_value (default 2)
    assert len(ix.hints("Alpha", None, P_(max_cols_per_value=5))) == 5
    assert len(ix.hints("Alpha", None, P_(max_cols_per_value=5, max_hints=2))) == 2
    ix = VG.ValueIndex.from_values({("t", "c"): ["ALPHA", "Alpha", "alpha", "aLPHA"]})
    assert len(ix.hints("Alpha", None, P_())) == 3 and len(ix.hints("Alpha", None, P_(max_variants=2))) == 2
    assert ix.hints("Alpha", None, P_())[0]["literal"] == "Alpha"                                               # the verbatim spelling first


def test_column_whose_name_shares_a_word_with_the_question_comes_first():
    ix = VG.ValueIndex.from_values({("t", "other"): ["Smith"], ("t", "author_name"): ["Smith"]})
    assert ix.hints("Which author name is Smith", None, P_())[0]["column"] == "author_name"
    assert ix.hints("Smith", None, P_())[0]["column"] == "other"                                                # no name hint: smaller / earlier column


def test_camel_case_column_names_rank_columns_and_siblings_reach_other_columns():
    ix = VG.ValueIndex.from_values({("Country", "LongName"): ["High income"], ("Country", "ShortName"): ["High income"],
                                    ("Country", "IncomeGroup"): ["High income: OECD", "High income: nonOECD"]})
    h = ix.hints("Which high income group countries are from Asia?", None, P_())
    assert h[0]["column"] == "IncomeGroup" and {x["literal"] for x in h if x["column"] == "IncomeGroup"} == {"High income: OECD", "High income: nonOECD"}
    assert len({x["column"] for x in h}) == 2                                                                  # max_cols_per_value; the exact column is still there
    ix._prepare()
    assert ix.col_words[ix.cols.index(next(c for c in ix.cols if c["column"] == "IncomeGroup"))] == {"income", "group"}


def test_schema_names_are_not_values():
    sc = {"tables": {"Customers": [{"name": "UnitCost"}, {"name": "price"}, {"name": "zip_code"}]}}
    ix = VG.ValueIndex.from_values({("t", "c"): ["Price", "Zip Code", "Unit Cost", "Customers", "Smith"]})
    q = "the Price and the zip code of Unit Cost for Customers named Smith"
    assert {h["literal"] for h in ix.hints(q, None, P_())} == {"Price", "Zip Code", "Unit Cost", "Customers", "Smith"}
    ix.set_names(sc)
    assert {h["literal"] for h in ix.hints(q, None, P_())} == {"Smith"}                                         # n-grams that spell a table / column are schema references


def test_index_is_prepared_at_build_time(vdb):
    assert VG.get_index(vdb, schema_of(vdb), P_())._ready is True


def test_identical_modes():
    ix = VG.ValueIndex.from_values({("t", "a"): ["Noble"], ("t", "b"): ["NOBLE"], ("t", "c"): ["Zebra"]})
    q = "people from Noble and Zebra"
    assert {(h["column"]) for h in ix.hints(q, None, P_(identical="keep"))} == {"a", "b", "c"}
    assert {(h["column"]) for h in ix.hints(q, None, P_(identical="drop"))} == {"b"}                            # only the non-verbatim spelling survives
    h = ix.hints(q, None, P_(identical="multi"))
    assert {x["column"] for x in h} == {"a", "b"}                                                               # 'Noble' sits in 2 columns -> disambiguation kept; 'Other' (1) dropped
    with pytest.raises(ValueError):
        P_(identical="x")


def test_hints_are_deterministic_and_pure(idx):
    q = "People with High income living in noble or new york and named Hartzler"
    a = idx.hints(q, None, P_())
    assert a == idx.hints(q, None, P_()) and idx.hints("", None, P_()) == [] and idx.hints(None, None, P_()) == []


def test_params_validation():
    with pytest.raises(ValueError, match="unknown"):
        VG.params({"m2": {"vg": {"nope": 1}}})
    with pytest.raises(ValueError):
        VG.params({"m2": {"vg": {"max_hints": 0}}})
    with pytest.raises(ValueError):
        VG.params({"m2": {"vg": {"fuzzy_threshold": 0.2}}})
    assert VG.params({"m2": {"vg": {"max_hints": 3}}})["max_hints"] == 3 and VG.params({})["max_hints"] == 10 and VG.params(None) == VG.VG_DEFAULTS


# ------------------------------------------------------------------------------------------------ 5. prompt block + placeholder
def H(table, column, lit):
    return {"table": table, "column": column, "literal": lit, "kind": "exact", "surface": lit, "sim": 1.0, "identical": True}


def test_render_block_format_and_quoting():
    assert VG.render_block([]) == ""
    b = VG.render_block([H("person", "city", "NOBLE"), H("person", "city", "Noble"), H("transaction", "order", "O'Brien")], quote_keywords=True)
    assert b.startswith("\n\n# Value hints\n") and b.endswith("- person.city: 'NOBLE', 'Noble'\n- \"transaction\".\"order\": 'O''Brien'")
    assert VG.render_block([H("transaction", "order", "x")], quote_keywords=False).endswith("- transaction.order: 'x'")
    assert VG.render_block([H("my t", "c", "x"), H("my t", "c", "x")]).count("'x'") == 1                         # duplicates collapse; special names always quoted
    assert '"my t".c' in VG.render_block([H("my t", "c", "x")])


LEGACY_HEAD = {n: P._read(n).replace("<<sections>><<value_hints?>>", "<<sections>>", 1) for n in ("gen.user", "rev.user", "rev_exec.user", "rev_exec_probe.user")}


def test_templates_only_gained_the_optional_placeholder():
    for n, legacy in LEGACY_HEAD.items():
        cur = P._read(n)
        assert cur.startswith("<<sections>><<value_hints?>>\n\n# Task") and cur.count("value_hints") == 1 and legacy.startswith("<<sections>>\n\n# Task"), n
        kw = dict(sections="S", question="Q", evidence="E", graph="G", quote_rule="", sql="x", preview="p", feedback="f", score="1", mode="m", probe="pr")
        legacy_rendered = re.sub(r"<<(\w+)(\?)?>>", lambda m: str(kw[m.group(1)]) if m.group(1) in kw else "", legacy)
        assert P.render(n, **kw) == legacy_rendered, n                                                          # absent placeholder: byte-identical
        assert P.render(n, value_hints="", **kw) == legacy_rendered, n
        with_h = P.render(n, value_hints="\n\n# Value hints\n- a.b: 'c'", **kw)
        assert with_h == legacy_rendered.replace("S\n\n# Task", "S\n\n# Value hints\n- a.b: 'c'\n\n# Task", 1), n


# ------------------------------------------------------------------------------------------------ 6. M3 integration + pipeline
class RealSchemaStore:
    def __init__(self, schema):
        self.schema = schema
        self.full_tokens = 10

    def retrieve_tables(self, qtext, qvec, rq, k_t, mode="full", n_domains=2, rerank=True):
        return [(t, 1.0) for t in list(self.schema["tables"])[:k_t]], 0

    def expand_fk_paths(self, tables):
        return list(tables)

    def lineage_text(self, tables, quote_keywords=False):
        return ""


class RealRes:
    terms = None

    def __init__(self, schema):
        self.s = RealSchemaStore(schema)
        self.llm = None

    def db_store(self, item):
        return self.s

    def embed_one(self, text):
        import numpy as np
        return np.ones(4, dtype="float32")

    def corpus_for(self, item):
        return None

    def ensure_terms(self, item):
        pass


def vitem(p, question="how many people live in noble?", evidence="", qid="toy:t:1"):
    return Item(qid=qid, db_id="t", question=question, evidence=evidence, gold_sql="", db_path=p, split="toy", meta={})


def cfg_of(**kw):
    return PL.resolved_cfg(C.deep_update({"m1": {"enabled": False}, "m3": {"schema_fallback_tokens": 0, "k_t": 6}, "m2": {"history": False, "terms": False, "lineage": False}}, kw))


def test_m3_retrieve_attaches_hints_only_when_the_switch_is_on(vdb):
    res = RealRes(schema_of(vdb))
    off = M3.retrieve(res, vitem(vdb), None, cfg_of(), None)
    assert off.value_hints == [] and off.vh_text == "" and off.vh_info == {} and not (paths.CACHE / "vstore").exists()      # nothing built, nothing logged
    on = M3.retrieve(res, vitem(vdb), None, cfg_of(m2={"value_grounding": True}), None)
    assert {(h["column"], h["literal"]) for h in on.value_hints} == {("city", "NOBLE"), ("city", "Noble")}
    assert on.vh_text.startswith("\n\n# Value hints") and "- person.city: 'NOBLE', 'Noble'" in on.vh_text and on.vh_info["tokens"] > 0
    assert on.ddl == off.ddl and on.tables == off.tables and on.sections() == off.sections()                      # nothing else in the context changed
    none = M3.retrieve(res, vitem(vdb, "what is the average age?"), None, cfg_of(m2={"value_grounding": True}), None)
    assert none.value_hints == [] and none.vh_text == ""
    cfg = cfg_of(m2={"value_grounding": True, "vg": {"max_hints": 1}})
    assert len(M3.retrieve(res, vitem(vdb), None, cfg, None).value_hints) == 1


def test_hints_respect_the_tables_shown_in_the_prompt(vdb):
    res = RealRes(schema_of(vdb))
    cfg = cfg_of(m2={"value_grounding": True}, m3={"k_t": 1, "schema_fallback_tokens": 0})
    ctx = M3.retrieve(res, vitem(vdb, "people from Noble who ordered Ice Cream"), None, cfg, None)
    assert ctx.tables == ["person"] and {h["table"] for h in ctx.value_hints} == {"person"}                       # 'transaction' is not shown -> no hint for it


def test_hint_failure_never_fails_the_question(vdb, monkeypatch):
    res = RealRes(schema_of(vdb))
    monkeypatch.setattr(VG, "get_index", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    ctx = M3.retrieve(res, vitem(vdb), None, cfg_of(m2={"value_grounding": True}), None)
    assert ctx.value_hints == [] and ctx.vh_text == "" and ctx.vh_info == {"error": "OSError"}


def make_vg_sys(monkeypatch, cls, llm, cfg, vdb):
    """A pipeline whose retrieval is the REAL M3.retrieve over a stub store serving the toy database's real schema."""
    cfg = C.deep_update({"m1": {"enabled": False}, "m3": {"schema_fallback_tokens": 0}, "m2": {"history": False, "terms": False, "lineage": False}}, cfg)
    s = cls(llm, "m", cfg)
    s.pipe.res = RealRes(schema_of(vdb))
    return s


VGC = {"m2": {"value_grounding": True}, "ddl": {"quote_keywords": True}, "m5": {"policy": "gated"}, "route": {"score": "exec", "theta_h": 7.5}}
SQL_NOBLE = "SELECT count(*) FROM person WHERE city = 'Noble'"
SQL_WRONG = "SELECT count(*) FROM person WHERE city = 'noble'"


def test_generation_prompt_carries_the_block_and_the_row_is_logged(monkeypatch, vdb):
    llm = ScriptLLM(gen=SQL_NOBLE)
    out = make_vg_sys(monkeypatch, PL.FullGated, llm, VGC, vdb).run(vitem(vdb))
    u = llm.calls[0]["user"]
    assert "# Value hints" in u and "- person.city: 'NOBLE', 'Noble'" in u and u.index("# Value hints") < u.index("# Task") and u.index("CREATE TABLE") < u.index("# Value hints")
    vh = out["value_hints"]
    assert vh["n"] == 2 and vh["cols"] == ["person.city"] and vh["lits"] == ["NOBLE", "Noble"] and vh["used"] == 1 and vh["used_root"] == 1 and vh["tokens"] > 0
    assert llm.tags() == ["gen"] and out["final_sql"] == SQL_NOBLE


def test_logged_use_flags_the_literal_actually_written(monkeypatch, vdb):
    out = make_vg_sys(monkeypatch, PL.FullGated, ScriptLLM(gen=SQL_WRONG), VGC, vdb).run(vitem(vdb))
    assert out["value_hints"]["n"] == 2 and out["value_hints"]["used"] == 0 and out["value_hints"]["used_root"] == 0       # 'noble' is not a stored spelling


def test_revision_prompts_keep_the_hints(monkeypatch, vdb):
    bad = "SELECT count(*) FROM person WHERE citty = 1"
    llm = ScriptLLM(gen=bad, rev=SQL_NOBLE)
    out = make_vg_sys(monkeypatch, PL.FullGated, llm, VGC, vdb).run(vitem(vdb))
    assert llm.tags() == ["gen", "rev"] and "- person.city: 'NOBLE', 'Noble'" in llm.calls[1]["user"] and "Execution error" in llm.calls[1]["user"]
    assert out["value_hints"]["used"] == 1 and out["value_hints"]["used_root"] == 0 and out["actions"][0] == "revise"
    llm = ScriptLLM(gen=SQL_WRONG, rev=SQL_NOBLE, crit='{"semantic": 3, "issues": ["x"], "suggestions": []}')     # P0 loop: rev.user
    cfg = {"m2": {"value_grounding": True}, "ddl": {"quote_keywords": True}}
    out = make_vg_sys(monkeypatch, PL.FullFixed, llm, cfg, vdb).run(vitem(vdb))
    assert "rev" in llm.tags() and all("# Value hints" in c["user"] for c in llm.calls if c["tag"] in ("gen", "rev"))


def test_empty_result_probe_template_also_gets_the_block(monkeypatch, vdb):
    empty = "SELECT count(*), max(age) FROM person WHERE city = 'zzz'"
    cfg = C.deep_update(VGC, {"m5": {"gate_empty": True, "probe_values": True}})
    sql_empty = "SELECT age FROM person WHERE city = 'noble'"
    llm = ScriptLLM(gen=sql_empty, rev="SELECT age FROM person WHERE city = 'Noble'")
    make_vg_sys(monkeypatch, PL.FullGated, llm, cfg, vdb).run(vitem(vdb))
    rev = llm.calls[1]["user"]
    assert "# Values stored in the filtered columns" in rev and "# Value hints" in rev                               # X7 probe block and X9 hints coexist


def test_questions_without_hints_give_the_legacy_prompt(monkeypatch, vdb):
    q = vitem(vdb, "what is the average age?")
    llm_on, llm_off = ScriptLLM(gen="SELECT avg(age) FROM person"), ScriptLLM(gen="SELECT avg(age) FROM person")
    out_on = make_vg_sys(monkeypatch, PL.FullGated, llm_on, VGC, vdb).run(q)
    off_cfg = C.deep_update(VGC, {"m2": {"value_grounding": False}})
    out_off = make_vg_sys(monkeypatch, PL.FullGated, llm_off, off_cfg, vdb).run(q)
    assert llm_on.calls[0]["user"] == llm_off.calls[0]["user"] and "Value hints" not in llm_on.calls[0]["user"]       # same prompt = same cache key
    assert out_on["value_hints"]["n"] == 0 and out_on["value_hints"]["cols"] == [] and "value_hints" not in out_off


def test_switch_off_is_byte_identical_and_adds_nothing(monkeypatch, vdb):
    assert DEFAULTS["m2"]["value_grounding"] is False and DEFAULTS["m2"]["vg"] == {}
    cfg_off = C.deep_update(VGC, {"m2": {"value_grounding": False}})
    llm = ScriptLLM(gen=SQL_NOBLE)
    s = make_vg_sys(monkeypatch, PL.FullGated, llm, cfg_off, vdb)
    out = s.run(vitem(vdb))
    ctx = M3.retrieve(s.pipe.res, vitem(vdb), None, s.pipe.cfg, None)
    expected = P.render("gen.user", sections=ctx.sections(), question="how many people live in noble?", evidence="(none)", graph="", quote_rule=P.QUOTE_RULE)
    assert llm.calls[0]["user"] == expected and "value_hints" not in out and "Value hints" not in expected
    assert all("value_hints" not in c for c in out["candidates"]) and not (paths.CACHE / "vstore").exists()


def test_warm_builds_the_indexes_up_front(monkeypatch, vdb):
    s = make_vg_sys(monkeypatch, PL.FullGated, ScriptLLM(), VGC, vdb)
    s.pipe.res.corpus_for = lambda it: None
    s.warm([vitem(vdb), vitem(vdb, qid="toy:t:2")])
    assert len(list((paths.CACHE / "vstore").glob("vidx_*.json.gz"))) == 1
    s2 = make_vg_sys(monkeypatch, PL.FullGated, ScriptLLM(), C.deep_update(VGC, {"m2": {"value_grounding": False}}), vdb)
    VG.clear_memory()
    for f in (paths.CACHE / "vstore").glob("*"):
        f.unlink()
    s2.pipe.res.corpus_for = lambda it: None
    s2.warm([vitem(vdb)])
    assert not list((paths.CACHE / "vstore").glob("vidx_*.json.gz"))


# ------------------------------------------------------------------------------------------------ 7. config / run.py
def test_configs_and_defaults():
    g, f = C.load_config(str(CONFIGS / "full_gated_vg.yaml")), C.load_config(str(CONFIGS / "full_fixed_vg.yaml"))
    base_g, base_f = C.load_config(str(CONFIGS / "full_gated.yaml")), C.load_config(str(CONFIGS / "full_quote.yaml"))
    assert g["m2"] == {"value_grounding": True} and f["m2"] == {"value_grounding": True}
    assert {k: v for k, v in g.items() if k != "m2"} == base_g and {k: v for k, v in f.items() if k != "m2"} == base_f      # the ONLY difference is the switch
    for name in ("full_gated_vg", "full_fixed_vg"):
        cfg = PL.resolved_cfg(C.load_config(str(CONFIGS / (name + ".yaml"))))
        pipe = PL.Pipeline(object(), "m", C.load_config(str(CONFIGS / (name + ".yaml"))), resources=object())
        assert pipe.vg_on and pipe.vg_params == VG.VG_DEFAULTS and cfg["ddl"]["quote_keywords"] is True
    for p in sorted(CONFIGS.glob("*.yaml")):                                                                    # no other config enables it
        if p.stem not in ("full_gated_vg", "full_fixed_vg"):
            assert not (C.load_config(str(p)).get("m2") or {}).get("value_grounding"), p.name
    with pytest.raises(ValueError, match="unknown"):
        PL.Pipeline(object(), "m", {"m2": {"value_grounding": True, "vg": {"typo": 1}}}, resources=object())
    PL.Pipeline(object(), "m", {"m2": {"vg": {"typo": 1}}}, resources=object())                                 # parameters are only validated when the switch is on


def test_run_py_passes_value_hints_through():
    from nl2sql import run

    class Sys:
        def run(self, item, seed=0):
            return {"final_sql": "select 1", "value_hints": {"n": 2, "cols": ["a.b"], "lits": ["x"], "used": 1, "used_root": 0, "tokens": 30, "ms": 1.0}}

    class Ev:
        gold = {}

        def score_sql(self, item, sql):
            return {"ex": 1, "exec_ok": True}

    it = Item(qid="q", db_id="d", question="?", evidence="", gold_sql="", db_path="x", split="s")
    row, _ = run.process_item(Sys(), Ev(), it, 0, {}, {"system": "s", "model": "m"})
    assert row["value_hints"]["n"] == 2 and row["value_hints"]["used"] == 1


# ------------------------------------------------------------------------------------------------ 8. tools/vg_eval.py
import vg_eval as VE  # noqa: E402


def test_vg_eval_recall_and_noise_on_synthetic_rows(vdb):
    sc = schema_of(vdb)
    ix = VG.get_index(vdb, sc, P_())
    index_vals = {("t", c["table"] + "." + c["column"]): set(c["values"]) for c in ix.cols}
    rows = [{"qid": "q1", "db_id": "t", "level": "approx", "gold_literals": [
                {"lit": "Noble", "present": False, "text_class": "case", "exists_in_gold_col": True, "gold_col": "person.city"}]},
            {"qid": "q2", "db_id": "t", "level": "absent", "gold_literals": [
                {"lit": "Bishopville", "present": False, "text_class": "absent", "exists_in_gold_col": True, "gold_col": "person.city"},
                {"lit": "Low income", "present": False, "text_class": "absent", "exists_in_gold_col": True, "gold_col": "person.income_group"}]},
            {"qid": "q3", "db_id": "t", "level": "absent", "gold_literals": [
                {"lit": "Nowhere", "present": False, "text_class": "absent", "exists_in_gold_col": False, "gold_col": "person.city"}]}]
    pop = VE.population(rows)
    assert [r["qid"] for r in pop["reachable"]] == ["q1", "q2"] and [r["qid"] for r in pop["nonverbatim"]] == ["q1", "q2", "q3"]
    hints_of = {"q1": ix.hints("people in noble", None, P_()), "q2": ix.hints("people in Bishopville", None, P_()), "q3": []}
    rec, det = VE.answer_recall(pop["reachable"], hints_of, index_vals)
    assert rec["answers"] == 2 and rec["literals"] == 3 and rec["col_recall"] == pytest.approx(2 / 3) and rec["answer_any_col"] == 2 and rec["answer_all_lit"] == 1
    assert rec["in_index"] == 1.0 and rec["by_text_class"]["case"]["col_recall"] == 1.0 and rec["by_text_class"]["absent"]["col_recall"] == 0.5
    assert det[0]["qid"] == "q1" and det[0]["hit_col"] == [True]


def test_vg_eval_noise_counts_distractors_and_conflicts(vdb):
    sc = schema_of(vdb)
    ix = VG.get_index(vdb, sc, P_())
    items = {"a": Item(qid="a", db_id="t", question="people in noble", evidence="", gold_sql="SELECT count(*) FROM person WHERE city = 'Noble'", db_path=vdb, split="s"),
             "b": Item(qid="b", db_id="t", question="people in noble", evidence="", gold_sql="SELECT count(*) FROM person WHERE city = 'New York'", db_path=vdb, split="s"),
             "c": Item(qid="c", db_id="t", question="average age", evidence="", gold_sql="SELECT avg(age) FROM person", db_path=vdb, split="s")}
    preds = {"a": {"ex": 0, "final_sql": "SELECT count(*) FROM person WHERE city = 'noble'"},
             "b": {"ex": 1, "final_sql": "SELECT count(*) FROM person WHERE city = 'New York'"},
             "c": {"ex": 1, "final_sql": "SELECT avg(age) FROM person"}}
    hints_of = {q: ix.hints(items[q].question, None, P_()) for q in items}
    blocks = {q: {"tokens": 10 if hints_of[q] else 0} for q in items}
    z = VE.noise(preds, items, hints_of, blocks, {vdb: sc})
    assert z["questions"] == 3 and z["with_hint"] == 2 and z["hints_total"] == 4 and z["tokens_total"] == 20
    assert z["gold_hit"] == 1 and z["new_useful_hints"] == 1                                                    # 'Noble' is a gold literal the wrong SQL did not contain
    assert z["distractor"] == 3 and z["correct_with_hint"] == 1 and z["correct_with_conflicting_hint"] == 1       # b's correct SQL filters city with another literal
    assert z["wrong_with_hint"] == 1 and z["kinds"] == {"exact": 4} and z["tokens_per_q_with_hint"] == 10


def test_vg_eval_rejects_build_parameter_overrides_in_variants(monkeypatch, tmp_path):
    import argparse
    a = argparse.Namespace(run=[], variant=['x={"max_len": 10}'], build=None, rows_cache=None, no_db=True, out=None)
    with pytest.raises(SystemExit, match="build parameter"):
        VE.run(a)


# ------------------------------------------------------------------------------------------------ 9. queue file
def test_x9_queue_is_consistent():
    q = paths.EXP_DIR / "queues" / "X9_live.jsonl"
    jobs = [json.loads(l) for l in q.read_text().splitlines() if l.strip() and not l.startswith("#")]
    ids = [j["id"] for j in jobs]
    assert len(ids) == len(set(ids)) == 6 and all(a in ids for j in jobs for a in j.get("after", []))
    llm = [j for j in jobs if j.get("needs_key")]
    assert {j["id"] for j in llm} == {"x9-vg-ht", "x9-vg-hr"} and all(j["conc"] == 4 and "--conc 4" in j["command"] for j in llm)
    for j in llm:
        cfg = re.search(r"--config (\S+)", j["command"]).group(1)
        assert cfg == "exp/configs/full_gated_vg.yaml" and (paths.REPO / cfg).exists() and "--split bird_train_H" in j["command"] and "bird_dev" not in j["command"]
        assert j["command"].count("cache_export.py import") == 1                                               # ONE archive per job (X7 pitfall)
        assert re.search(r"import \$R/runs/X5/1b-h[rt]-gated &&", j["command"])
    assert sum(j["est_calls"] for j in jobs) == 972 and all("limit 0" not in j["command"] for j in jobs)
    assert any("x7_adopt.py" in j["command"] for j in jobs) and sum("paired_compare.py" in j["command"] for j in jobs) == 3
