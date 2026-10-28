"""Minimal local stand-ins for X2's core modules (llm.py, db.py, data.py, systems/base.py).

Used ONLY for B1's own tests while X2's code is absent (see _common.py: it imports the real
modules first and falls back here).  Interfaces follow docs/INTERFACES.md.  Not part of the
evaluation pipeline.  Split names are emulated: the real HT/HR/pool splits come from
exp/splits/train_holdout.json (X1/X2); here HT = 4 deterministic BIRD-train dbs, pool = the rest.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sqlite3
import threading
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field

BIRD_TRAIN = os.environ.get("B1_BIRD_TRAIN", os.path.join(os.environ.get("NL2SQL_ROOT", os.path.expanduser("~/nl2sql_work")), "data", "bird", "train", "train"))
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# --------------------------------------------------------------------------- llm
class LLMClient:
    def __init__(self, cache_path, usage_path, key_file=os.path.expanduser("~/.nl2sql/qwen.env"), max_concurrency_per_model=8):
        self.key = None
        for line in open(key_file):
            if line.startswith("QWEN_API_KEY="):
                self.key = line.strip().split("=", 1)[1]
        assert self.key, "no key"
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        self.cache_path, self.usage_path = cache_path, usage_path
        self._lock = threading.Lock()
        self._sem = {}
        self._maxc = max_concurrency_per_model
        con = sqlite3.connect(cache_path)
        con.execute("create table if not exists c (k text primary key, v text)")
        con.commit()
        con.close()

    def _semaphore(self, model):
        with self._lock:
            if model not in self._sem:
                self._sem[model] = threading.Semaphore(self._maxc)
            return self._sem[model]

    def _post(self, path, payload):
        req = urllib.request.Request(BASE_URL + path, data=json.dumps(payload).encode(),
                                     headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        last = None
        for a in range(6):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                body = e.read().decode()[:300]
                last = "HTTP %s %s" % (e.code, body)
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(min(60, 2 ** (a + 1)))
                    continue
                raise RuntimeError(last)
            except Exception as e:  # timeout etc.
                last = repr(e)
                time.sleep(min(60, 2 ** (a + 1)))
        raise RuntimeError("api_fail " + str(last))

    def chat(self, model, messages, *, temperature=0.0, seed=None, max_tokens=2048, enable_thinking=False, n=1, tag=""):
        params = dict(temperature=temperature, seed=seed, top_p=1.0, max_tokens=max_tokens, n=n,
                      extra_body={"enable_thinking": enable_thinking})
        key = hashlib.sha256((model + "|" + json.dumps(params, sort_keys=True) + "|" + json.dumps(messages, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()
        with self._lock:
            con = sqlite3.connect(self.cache_path)
            row = con.execute("select v from c where k=?", (key,)).fetchone()
            con.close()
        if row:
            d = json.loads(row[0])
            d["cached"] = True
            return d
        payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                   "top_p": 1.0, "enable_thinking": enable_thinking}
        if seed is not None:
            payload["seed"] = int(seed)
        if n > 1:
            payload["n"] = n
        t0 = time.time()
        with self._semaphore(model):
            r = self._post("/chat/completions", payload)
        lat = time.time() - t0
        texts = [c["message"]["content"] or "" for c in r["choices"]]
        d = {"text": texts if n > 1 else texts[0], "prompt_tokens": r["usage"]["prompt_tokens"],
             "completion_tokens": r["usage"]["completion_tokens"], "latency_s": lat, "cached": False}
        with self._lock:
            con = sqlite3.connect(self.cache_path)
            con.execute("insert or replace into c values (?,?)", (key, json.dumps(d)))
            con.commit()
            con.close()
            with open(self.usage_path, "a") as f:
                f.write(json.dumps({"model": model, "tag": tag, "pt": d["prompt_tokens"], "ct": d["completion_tokens"], "lat": lat, "t": time.time()}) + "\n")
        return d

    def embed(self, texts, model):
        out = []
        for i in range(0, len(texts), 10):
            batch = texts[i:i + 10]
            key = hashlib.sha256((model + "|emb|" + json.dumps(batch, ensure_ascii=False)).encode()).hexdigest()
            with self._lock:
                con = sqlite3.connect(self.cache_path)
                row = con.execute("select v from c where k=?", (key,)).fetchone()
                con.close()
            if row:
                out.extend(json.loads(row[0]))
                continue
            with self._semaphore(model):
                r = self._post("/embeddings", {"model": model, "input": batch, "encoding_format": "float"})
            vecs = [x["embedding"] for x in sorted(r["data"], key=lambda x: x["index"])]
            with self._lock:
                con = sqlite3.connect(self.cache_path)
                con.execute("insert or replace into c values (?,?)", (key, json.dumps(vecs)))
                con.commit()
                con.close()
            out.extend(vecs)
        return out


# --------------------------------------------------------------------------- db
class db:  # namespace, mirrors nl2sql.db
    @staticmethod
    def execute(db_path, sql, timeout_s=30, max_rows=10000):
        t0 = time.time()
        try:
            con = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True, timeout=5)
            con.text_factory = lambda b: b.decode("utf-8", "replace")
            deadline = t0 + timeout_s
            con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10000)
            cur = con.cursor()
            cur.execute(sql)
            rows = cur.fetchmany(max_rows + 1)
            trunc = len(rows) > max_rows
            rows = rows[:max_rows]
            con.close()
            return {"ok": True, "rows": rows, "error": None, "elapsed_s": time.time() - t0, "truncated": trunc}
        except Exception as e:
            return {"ok": False, "rows": [], "error": str(e), "elapsed_s": time.time() - t0, "truncated": False}

    @staticmethod
    def ex_match(pred_rows, gold_rows):
        return int(set(pred_rows) == set(gold_rows))

    @staticmethod
    def explain(db_path, sql):
        r = db.execute(db_path, "EXPLAIN QUERY PLAN " + sql)
        return [str(x[-1]) for x in r["rows"]] if r["ok"] else []


# --------------------------------------------------------------------------- data
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


_HT_DBS = None


def _all_items():
    data = json.load(open(os.path.join(BIRD_TRAIN, "train.json")))
    items = []
    for i, d in enumerate(data):
        dbp = os.path.join(BIRD_TRAIN, "train_databases", d["db_id"], d["db_id"] + ".sqlite")
        items.append(Item("bird_train:%s:%d" % (d["db_id"], i), d["db_id"], d["question"], d.get("evidence", "") or "",
                          d.get("SQL", ""), dbp, "", {}))
    return items


def load_split(name):
    items = _all_items()
    dbs = sorted({x.db_id for x in items})
    rnd = random.Random(20261025)
    rnd.shuffle(dbs)
    ht = set(dbs[:4])
    hr = set(dbs[4:12])
    out = []
    for x in items:
        if name == "bird_train_HT" and x.db_id in ht:
            out.append(x)
        elif name == "bird_train_HR" and x.db_id in hr:
            out.append(x)
        elif name == "bird_train_pool" and x.db_id not in ht and x.db_id not in hr:
            out.append(x)
    for x in out:
        x.split = name
    return out


def get_schema(item_or_db, split=None):
    db_path = item_or_db.db_path if hasattr(item_or_db, "db_path") else None
    db_id = item_or_db.db_id if hasattr(item_or_db, "db_id") else item_or_db
    if db_path is None:
        db_path = os.path.join(BIRD_TRAIN, "train_databases", db_id, db_id + ".sqlite")
    con = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    tabs = [r[0] for r in con.execute("select name from sqlite_master where type='table' and name!='sqlite_sequence'")]
    descs = {}
    ddir = os.path.join(os.path.dirname(db_path), "database_description")
    if os.path.isdir(ddir):
        for fn in os.listdir(ddir):
            if not fn.endswith(".csv"):
                continue
            for enc in ("utf-8-sig", "latin-1"):
                try:
                    rows = list(csv.DictReader(open(os.path.join(ddir, fn), encoding=enc)))
                    break
                except Exception:
                    rows = []
            for r in rows:
                oc = (r.get("original_column_name") or "").strip().lower()
                descs[(fn[:-4].lower(), oc)] = (r.get("column_description") or r.get("column_name") or "").strip()
    ddl, tables, fks, pks = [], {}, [], []
    for t in tabs:
        ddl.append(con.execute("select sql from sqlite_master where name=?", (t,)).fetchone()[0])
        cols = []
        for _, name, typ, _, _, pk in con.execute("pragma table_info(`%s`)" % t):
            ex = []
            if (typ or "").upper() != "BLOB":
                try:
                    ex = [str(r[0])[:40] for r in con.execute("select distinct `%s` from `%s` where `%s` is not null limit 3" % (name, t, name))]
                except Exception:
                    ex = []
            cols.append({"name": name, "type": typ, "desc": descs.get((t.lower(), name.lower()), ""), "examples": ex})
            if pk:
                pks.append((t, name))
        tables[t] = cols
        for r in con.execute("pragma foreign_key_list(`%s`)" % t):
            fks.append((t, r[3], r[2], r[4]))
    con.close()
    return {"ddl": "\n\n".join(ddl), "tables": tables, "fks": fks, "pks": pks}


# --------------------------------------------------------------------------- base
class System:
    name = "base"

    def __init__(self, llm, model, cfg):
        self.llm, self.model, self.cfg = llm, model, cfg or {}

    def run(self, item, seed=0):
        raise NotImplementedError
