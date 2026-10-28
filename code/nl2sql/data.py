"""Datasets, splits, schemas.

Split names (INTERFACES.md): bird_dev, bird_train_HT, bird_train_HR, bird_train_pool, bird_minidev, spider_dev,
spider_syn, spider_realistic, spider_dk, ambiqt, livesql_lite. Extras: bird_train_HT_full (all questions of the
HT dbs), spider_train_pool.
 * bird_train_HT  := HT-300 (the only set used for prompt/hyper-parameter search; EXPERIMENT_SPEC 1.2 step 6),
   returned in a fixed shuffled order so `--limit N` is a stratified-ish sample.
 * bird_train_HR  := the N_db = min(n_q,150) questions drawn per HR db (all HR questions), Item.meta["role"] in
   {seed, probe, stream}, shuffled order.
 * qid = "<dataset>:<db_id>:<question_id>" with dataset in {bird_dev, bird_train, spider_dev, ...}, so the same
   question has the same qid in every split that contains it.
Locked splits (dev lock, CONTRACT section 3): load_split() returns items with gold_sql="" unless the caller is the
evaluator (`load_gold`), which logs to exp/dev_access.log."""
import csv
import json
import os
import re
import tempfile
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import paths
from .util import tmp_suffix
from .db import row_counts, _run

UNLOCKED = {"bird_train_HT", "bird_train_HT_full", "bird_train_HR", "bird_train_pool", "spider_train_pool"}
ALL_SPLITS = ["bird_dev", "bird_train_HT", "bird_train_HT_full", "bird_train_HR", "bird_train_pool", "bird_minidev",
              "spider_dev", "spider_syn", "spider_realistic", "spider_dk", "ambiqt", "ambiqt_mix", "livesql_lite", "spider_train_pool"]


@dataclass
class Item:
    qid: str
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    db_path: str
    split: str
    meta: dict = field(default_factory=dict)


def is_locked(split):
    return split not in UNLOCKED


def dataset_of(split):
    if split.startswith("bird_train"):
        return "bird_train"
    return split


# --------------------------------------------------------------------------- paths
def _bird_dirs():
    d = paths.DATA / "bird"
    return {
        "dev": d / "dev" / "dev_20240627",
        "train": d / "train" / "train",
        "minidev": d / "minidev" / "minidev" / "MINIDEV",
    }


def db_path(dataset, db_id):
    b = _bird_dirs()
    if dataset in ("bird_dev",):
        return str(b["dev"] / "dev_databases" / db_id / (db_id + ".sqlite"))
    if dataset in ("bird_train",):
        return str(b["train"] / "train_databases" / db_id / (db_id + ".sqlite"))
    if dataset == "bird_minidev":
        return str(b["minidev"] / "dev_databases" / db_id / (db_id + ".sqlite"))
    if dataset in ("spider_dev", "spider_syn", "spider_realistic", "spider_train", "spider_train_pool"):
        return str(paths.DATA / "spider" / "spider_data" / "database" / db_id / (db_id + ".sqlite"))
    if dataset == "spider_dk":
        p = paths.DATA / "spider_dk" / "database" / db_id / (db_id + ".sqlite")
        if p.exists():
            return str(p)
        return str(paths.DATA / "spider" / "spider_data" / "database" / db_id / (db_id + ".sqlite"))
    if dataset == "ambiqt":
        return str(paths.DATA / "ambiqt_db" / "db-content" / "database" / db_id / (db_id + ".sqlite"))
    if dataset == "livesql_lite":
        return str(paths.DATA / "livesqlbench_base_lite_sqlite" / db_id / (db_id + "_template.sqlite"))
    raise KeyError(dataset)


def _desc_dir(dataset, db_id):
    b = _bird_dirs()
    if dataset == "bird_dev":
        return b["dev"] / "dev_databases" / db_id / "database_description"
    if dataset == "bird_train":
        return b["train"] / "train_databases" / db_id / "database_description"
    if dataset == "bird_minidev":
        return b["minidev"] / "dev_databases" / db_id / "database_description"
    return None


def _jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- raw loaders
def _bird_dev_rows():
    return _jload(_bird_dirs()["dev"] / "dev.json")


def _bird_train_rows():
    return _jload(_bird_dirs()["train"] / "train.json")


def _bird_item(ds, split, idx, r, with_gold):
    return Item(qid="%s:%s:%s" % (ds, r["db_id"], r.get("question_id", idx)), db_id=r["db_id"],
                question=r["question"], evidence=(r.get("evidence") or "").strip(),
                gold_sql=(r["SQL"] if with_gold else ""), db_path=db_path(ds, r["db_id"]), split=split,
                meta={"difficulty": r.get("difficulty"), "idx": idx})


def _splits_json():
    if not paths.SPLITS_JSON.exists():
        raise FileNotFoundError("exp/splits/train_holdout.json missing: run python exp/splits/make_splits.py")
    return _jload(paths.SPLITS_JSON)


def load_split(name, with_gold=None):
    """List[Item]. with_gold=None -> gold only for unlocked splits."""
    if with_gold is None:
        with_gold = not is_locked(name)
    if name == "bird_dev":
        return [_bird_item("bird_dev", name, i, r, with_gold) for i, r in enumerate(_bird_dev_rows())]
    if name == "bird_minidev":
        # D0: the 2024-06 zip has duplicate question_id 137/138 (498 distinct); the HF 2025-07 version has 500 distinct ids -> use HF
        rows = _jload(paths.DATA / "bird_mini_dev_hf" / "data" / "mini_dev_sqlite-00000-of-00001.json")
        out = []
        for i, r in enumerate(rows):
            out.append(Item(qid="bird_minidev:%s:%s" % (r["db_id"], r.get("question_id", i)), db_id=r["db_id"],
                            question=r["question"], evidence=(r.get("evidence") or "").strip(),
                            gold_sql=(r["SQL"] if with_gold else ""), db_path=db_path("bird_minidev", r["db_id"]),
                            split=name, meta={"difficulty": r.get("difficulty")}))
        return out
    if name.startswith("bird_train"):
        rows = _bird_train_rows()
        sp = _splits_json()
        if name == "bird_train_pool":
            pool = set(sp["pool_dbs"])
            return [_bird_item("bird_train", name, i, r, with_gold) for i, r in enumerate(rows) if r["db_id"] in pool]
        if name == "bird_train_HT_full":
            ht = set(sp["ht_dbs"])
            return [_bird_item("bird_train", name, i, r, with_gold) for i, r in enumerate(rows) if r["db_id"] in ht]
        by_qid = {"bird_train:%s:%d" % (r["db_id"], i): (i, r) for i, r in enumerate(rows)}
        if name == "bird_train_HT":
            items = []
            for q in sp["ht300_qids"]:
                i, r = by_qid[q]
                it = _bird_item("bird_train", name, i, r, with_gold)
                it.meta.update(sp.get("ht300_meta", {}).get(q, {}))
                items.append(it)
            return items
        if name == "bird_train_HR":
            role_of = {}
            for db, roles in sp["hr_roles"].items():
                for role, qs in roles.items():
                    for q in qs:
                        role_of[q] = role
            items = []
            for q in sp["hr_order"]:
                i, r = by_qid[q]
                it = _bird_item("bird_train", name, i, r, with_gold)
                it.meta["role"] = role_of[q]
                it.meta["fold"] = sp["cv_fold_of_db"][r["db_id"]]
                items.append(it)
            return items
        raise KeyError(name)
    if name in ("spider_dev", "spider_syn", "spider_realistic", "spider_dk", "spider_train_pool"):
        S = paths.DATA / "spider"
        if name == "spider_dev":
            rows = _jload(S / "spider_data" / "dev.json")
            qf = lambda r: r["question"]
        elif name == "spider_train_pool":
            rows = _jload(S / "spider_data" / "train_spider.json")
            qf = lambda r: r["question"]
        elif name == "spider_syn":
            rows = _jload(paths.DATA / "spider_syn" / "Spider-Syn" / "dev.json")
            qf = lambda r: r["SpiderSynQuestion"]
        elif name == "spider_realistic":
            rows = _jload(paths.DATA / "spider_realistic" / "spider-realistic.json")
            qf = lambda r: r["question"]
        else:
            rows = _jload(paths.DATA / "repos" / "Spider-DK" / "Spider-DK.json")
            qf = lambda r: r["question"]
        out = []
        for i, r in enumerate(rows):
            m = {}
            if name == "spider_syn":
                m["original_question"] = r["SpiderQuestion"]
            if name == "spider_dk":
                m["dk_type"] = r.get("type")
            out.append(Item(qid="%s:%s:%d" % (name, r["db_id"], i), db_id=r["db_id"], question=qf(r), evidence="",
                            gold_sql=(r["query"] if with_gold else ""), db_path=db_path(name, r["db_id"]),
                            split=name, meta=m))
        return out
    if name == "ambiqt":
        return _load_ambiqt(with_gold)
    if name == "ambiqt_mix":
        d = _jload(paths.EXP_DIR / "splits" / "ambiqt_mix.json")
        return [Item(qid=r["qid"], db_id=r["db_id"], question=r["question"], evidence=r["evidence"],
                     gold_sql=(r["gold_sql"] if with_gold else ""), db_path=r["db_path"], split=name,
                     meta={"ambiguous": r["ambiguous"], "amb_type": r["amb_type"], "gold_sql_2": (r["gold_sql_2"] if with_gold else ""),
                           "ambiguous_schema": r["ambiguous_schema"]}) for r in d["items"]]
    if name == "livesql_lite":
        out = []
        for line in open(paths.DATA / "livesqlbench_base_lite" / "livesqlbench_data.jsonl", encoding="utf-8"):
            r = json.loads(line)
            ek = r.get("external_knowledge")
            out.append(Item(qid="livesql_lite:%s:%s" % (r["selected_database"], r["instance_id"]), db_id=r["selected_database"],
                            question=r["query"], evidence=" ".join(map(str, ek)) if isinstance(ek, list) else str(ek or ""),
                            gold_sql="", db_path=db_path("livesql_lite", r["selected_database"]), split=name,
                            meta={"gold_public": False, "category": r.get("category"), "tier": r.get("difficulty_tier"),
                                  "dialect": "postgresql_origin_sqlite_template"}))
        return out
    raise KeyError("unknown split %s" % name)


def _load_ambiqt(with_gold):
    """AmbiQT validation sets. Each item has two valid SQLs (query1/query2) over a rewritten (ambiguous) schema.
    The shipped databases are the original Spider ones; the ambiguous schema is given as text in
    meta['ambiguous_schema']. Execution against the rewritten schema is NOT available (see the data notes of the benchmark)."""
    out = []
    base = paths.DATA / "repos" / "AmbiQT" / "benchmark"
    for typ, short in (("col-synonyms", "C"), ("tbl-synonyms", "T"), ("tbl-split", "J"), ("tbl-agg", "P")):
        rows = _jload(base / typ / "validation.json")
        for i, r in enumerate(rows):
            out.append(Item(qid="ambiqt:%s:%s-%d" % (r["db_id"], short, i), db_id=r["db_id"], question=r["question"],
                            evidence="", gold_sql=(r["query1"] if with_gold else ""), db_path=db_path("ambiqt", r["db_id"]),
                            split="ambiqt", meta={"ambiguous": 1, "amb_type": short,
                                                  "gold_sql_2": r["query2"] if with_gold else "",
                                                  "ambiguous_schema": r.get("schema_without_content", "")}))
    return out


def load_gold(split, run_id=""):
    """Evaluator-only: {qid: gold_sql}. Every access to a locked split is logged."""
    items = load_split(split, with_gold=True)
    if is_locked(split):
        paths.DEV_ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(paths.DEV_ACCESS_LOG, "a") as f:
            f.write("%s\tsplit=%s\tn=%d\trun=%s\tpid=%d\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), split, len(items), run_id, os.getpid()))
    return {it.qid: it.gold_sql for it in items}


# --------------------------------------------------------------------------- schema
# Full SQLite keyword list (https://www.sqlite.org/lang_keywords.html, 3.45+; 147 words). Some are "fallback" tokens the parser
# still accepts as identifiers (e.g. KEY, MATCH, CURRENT), others are hard reserved (TRANSACTION, ORDER, GROUP, INDEX ...).
# Quoting any of them is always safe, so the list is not split. TRUE/FALSE are not keywords but SQLite resolves them to 1/0
# unless a column of that name exists; quoting is harmless.
SQLITE_KEYWORDS = frozenset("""
ABORT ACTION ADD AFTER ALL ALTER ALWAYS ANALYZE AND AS ASC ATTACH AUTOINCREMENT BEFORE BEGIN BETWEEN BY CASCADE CASE CAST CHECK
COLLATE COLUMN COMMIT CONFLICT CONSTRAINT CREATE CROSS CURRENT CURRENT_DATE CURRENT_TIME CURRENT_TIMESTAMP DATABASE DEFAULT
DEFERRABLE DEFERRED DELETE DESC DETACH DISTINCT DO DROP EACH ELSE END ESCAPE EXCEPT EXCLUDE EXCLUSIVE EXISTS EXPLAIN FAIL FILTER
FIRST FOLLOWING FOR FOREIGN FROM FULL GENERATED GLOB GROUP GROUPS HAVING IF IGNORE IMMEDIATE IN INDEX INDEXED INITIALLY INNER INSERT
INSTEAD INTERSECT INTO IS ISNULL JOIN KEY LAST LEFT LIKE LIMIT MATCH MATERIALIZED NATURAL NO NOT NOTHING NOTNULL NULL NULLS OF
OFFSET ON OR ORDER OTHERS OUTER OVER PARTITION PLAN PRAGMA PRECEDING PRIMARY QUERY RAISE RANGE RECURSIVE REFERENCES REGEXP REINDEX
RELEASE RENAME REPLACE RESTRICT RETURNING RIGHT ROLLBACK ROW ROWS SAVEPOINT SELECT SET TABLE TEMP TEMPORARY THEN TIES TO TRANSACTION
TRIGGER UNBOUNDED UNION UNIQUE UPDATE USING VACUUM VALUES VIEW VIRTUAL WHEN WHERE WINDOW WITH WITHOUT
""".split())
_ALWAYS_QUOTE = frozenset(["TRUE", "FALSE"])
_PLAIN_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def needs_quote(name, quote_keywords=True):
    """True when `name` cannot be written bare in SQLite: not a plain identifier, or (quote_keywords) a reserved word."""
    if not _PLAIN_IDENT.match(name):
        return True
    return bool(quote_keywords) and (name.upper() in SQLITE_KEYWORDS or name.upper() in _ALWAYS_QUOTE)


def _q(name, quote_keywords=False):
    """Identifier as shown in prompts. quote_keywords=False is the legacy behaviour (only names with special characters are
    quoted; keeps old prompts / cache keys reproducible); True also quotes SQLite reserved words (config ddl.quote_keywords)."""
    return '"%s"' % name.replace('"', '""') if needs_quote(name, quote_keywords) else name


def _read_desc_csv(p):
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            with open(p, newline="", encoding=enc) as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
        except Exception:
            return []
    return []


def _build_schema(db_file, dataset, db_id):
    tables = {}
    fks, pks = [], []
    r = _run(db_file, "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid", 30, None)
    tnames = [x[0] for x in r["rows"]]
    desc = {}
    dd = _desc_dir(dataset, db_id)
    if dd and dd.exists():
        for f in dd.glob("*.csv"):
            tname = f.stem
            rows = _read_desc_csv(f)
            d = {}
            for row in rows:
                row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
                oc = row.get("original_column_name", "")
                if oc:
                    d[oc.lower()] = row
            desc[tname.lower()] = d
    rc = row_counts(db_file)
    for t in tnames:
        info = _run(db_file, 'PRAGMA table_info("%s")' % t.replace('"', '""'), 30, None)["rows"]
        cols = []
        for cid, cname, ctype, notnull, dflt, pk in info:
            dtext = ""
            row = desc.get(t.lower(), {}).get(cname.lower())
            if row:
                parts = []
                if row.get("column_name") and row["column_name"].lower() != cname.lower():
                    parts.append(row["column_name"])
                if row.get("column_description") and row["column_description"].lower() not in (cname.lower(), (row.get("column_name") or "").lower()):
                    parts.append(row["column_description"])
                if row.get("value_description"):
                    parts.append("values: " + re.sub(r"\s+", " ", row["value_description"])[:200])
                dtext = "; ".join(parts)
            ex = []
            if (ctype or "").upper() in ("TEXT", "") or "CHAR" in (ctype or "").upper() or "CLOB" in (ctype or "").upper():
                er = _run(db_file, 'SELECT DISTINCT "%s" FROM "%s" WHERE "%s" IS NOT NULL AND length("%s") <= 40 LIMIT 3'
                          % (cname.replace('"', '""'), t.replace('"', '""'), cname.replace('"', '""'), cname.replace('"', '""')), 10, 3)
                if er["ok"]:
                    ex = [str(x[0]) for x in er["rows"]]
            cols.append({"name": cname, "type": ctype or "", "desc": dtext, "examples": ex, "notnull": bool(notnull),
                         "pk": int(pk)})
            if pk:
                pks.append({"table": t, "column": cname, "pos": int(pk)})
        tables[t] = cols
        for (fid, seq, ref_t, frm, to, *_rest) in _run(db_file, 'PRAGMA foreign_key_list("%s")' % t.replace('"', '""'), 30, None)["rows"]:
            fks.append({"table": t, "column": frm, "ref_table": ref_t, "ref_column": to if to else None, "id": fid, "seq": seq})
    # fill missing ref columns with the target's primary key
    for fk in fks:
        if fk["ref_column"] is None:
            cand = [p["column"] for p in pks if p["table"] == fk["ref_table"]]
            fk["ref_column"] = cand[0] if cand else None
    schema = {"db_id": db_id, "tables": tables, "fks": fks, "pks": pks, "row_counts": rc}
    schema["ddl"] = render_ddl(schema)
    return schema


def render_ddl(schema, tables=None, samples=True, descriptions=True, flag=None, quote_keywords=False):
    """Spec schema block: CREATE TABLE with FK lines + `-- col: description | samples: a, b, c` comments.
    tables: iterable restricting/ordering tables; flag: set of tables marked `-- likely relevant`.
    quote_keywords: also double-quote SQLite reserved words used as table/column names (config `ddl.quote_keywords`; default False =
    legacy output, byte-identical to earlier runs). Only the prompt text changes, never the database or the stored schema."""
    qk = bool(quote_keywords)
    names = list(tables) if tables is not None else list(schema["tables"].keys())
    out = []
    for t in names:
        cols = schema["tables"][t]
        lines = []
        pk_cols = [c["name"] for c in cols if c.get("pk")]
        n = len(cols)
        fk_here = [f for f in schema["fks"] if f["table"] == t and (tables is None or f["ref_table"] in names)]
        items = []
        for c in cols:
            s = "  %s %s" % (_q(c["name"], qk), c["type"] or "")
            s = s.rstrip()
            cm = []
            if descriptions and c.get("desc"):
                cm.append(c["desc"])
            if samples and c.get("examples"):
                cm.append("samples: " + ", ".join(c["examples"]))
            items.append((s, (" | ".join(cm) if cm else None)))
        tail = []
        if pk_cols:
            tail.append("  PRIMARY KEY (%s)" % ", ".join(_q(c, qk) for c in pk_cols))
        for f in fk_here:
            tail.append("  FOREIGN KEY (%s) REFERENCES %s(%s)" % (_q(f["column"], qk), _q(f["ref_table"], qk), _q(f["ref_column"] or "", qk)))
        total = len(items) + len(tail)
        body = []
        for k, (s, cm) in enumerate(items):
            comma = "," if (k < total - 1) else ""
            body.append(s + comma + ("  -- " + cm if cm else ""))
        for k, s in enumerate(tail):
            comma = "," if (len(items) + k < total - 1) else ""
            body.append(s + comma)
        head = "CREATE TABLE %s (" % _q(t, qk)
        if flag and t in flag:
            head = "-- likely relevant\n" + head
        out.append(head + "\n" + "\n".join(body) + "\n);")
    return "\n".join(out)


import threading
_SCHEMA_MEM = {}
_BUILD_LOCK = threading.Lock()


def get_schema(item_or_db, split=None):
    """{"ddl","tables":{t:[{name,type,desc,examples,...}]},"fks":[{table,column,ref_table,ref_column}],"pks":[{table,column}],
    "row_counts":{t:n}}. Cached on disk under NL2SQL_CACHE/schema."""
    if isinstance(item_or_db, Item):
        dbf, db_id = item_or_db.db_path, item_or_db.db_id
        ds = item_or_db.qid.split(":")[0]
    else:
        db_id = item_or_db
        ds = dataset_of(split or "bird_dev")
        dbf = db_path(ds, db_id)
    key = (dbf,)
    if key in _SCHEMA_MEM:
        return _SCHEMA_MEM[key]
    st = os.stat(dbf)
    cdir = paths.CACHE / "schema"
    cdir.mkdir(parents=True, exist_ok=True)
    cf = cdir / ("%s__%s__%d_%d.json" % (ds, db_id, st.st_size, int(st.st_mtime)))
    if cf.exists():
        try:
            sc = json.loads(cf.read_text())
            _SCHEMA_MEM[key] = sc
            return sc
        except Exception:
            pass
    with _BUILD_LOCK:
        if key in _SCHEMA_MEM:
            return _SCHEMA_MEM[key]
        if cf.exists():
            try:
                sc = json.loads(cf.read_text())
                _SCHEMA_MEM[key] = sc
                return sc
            except Exception:
                pass
        sc = _build_schema(dbf, ds, db_id)
    # unique temp file per call (mkstemp: O_EXCL, never shared between threads/processes), then atomic rename
    fd, tmp = tempfile.mkstemp(prefix=cf.name + ".", suffix=".tmp", dir=str(cdir))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(sc, f)
        os.replace(tmp, cf)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    _SCHEMA_MEM[key] = sc
    return sc
