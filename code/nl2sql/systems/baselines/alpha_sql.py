"""Alpha-SQL (Li et al., ICML 2025, arXiv 2502.17248): zero-shot text-to-SQL by Monte Carlo Tree Search.  EXPENSIVE.

Source: https://github.com/HKUSTDial/Alpha-SQL, commit 216e61fc86467591fa1dc5b7031ca536577b4d21 (2026-06-10), MIT licence.
Ported (same algorithm and constants; prompt templates are byte-identical copies in `_vendor/alphasql_templates/`):
  * action space and node-type transition table (rephrase / schema selection / identify column values / identify column
    functions / SQL generation / SQL revision / end), eager expansion of every valid action, UCT selection with
    exploration constant 1.414, random simulation to a terminal node, back-propagation of the self-consistency reward
    (share of the largest execution-result group among 5 samples at T=0.8), 24 rollouts by default;
  * `n=3` samples for the reasoning actions, `n=5` samples + execution-consistency vote for SQL generation / revision,
    `list(set(...))` de-duplication, schema-selection table/column filter with PK/FK closure, `SQL_VALIDATION_MAX_TRIES=15`;
  * schema context = `build_table_ddl_statement` (column meaning, description, value description, value examples);
  * final answer = majority over the execution results of all END nodes (valid results only), ties broken by faster execution;
  * question-specific value retrieval of the preprocessor: keyword extraction prompt, character-3-gram Jaccard >= 0.5 candidates
    (top 20), edit-similarity >= 0.3, embedding cosine >= 0.6, top-3 values replace the column's examples.
Adapted (our adaptation):
  * LLM client: OpenAI `chat.completions` (vLLM, Qwen2.5-Coder-32B / gpt-4o-mini for keywords) -> `LLMClient.chat`; sampled
    completions use the `n` argument of one request (DashScope allows n <= 4, so n=5 is sent as 4+1), each request counts as ONE call
    in `calls` (`ctx.samples` holds the completions);
    every request gets its own seed so that the cache cannot replay an identical sample;
  * embeddings: configured OpenAI-style model -> `qwen3.7-text-embedding`; keyword extraction runs on the system's own model;
  * approximate MinHash-LSH (datasketch) -> exact character-3-gram Jaccard over an in-memory inverted index (values per text
    column capped at `max_values_per_col`); execution-time tie-break uses `time_repeat` repeats (original 20);
  * unbounded `while` loops in the original (schema selection / SQL generation until a parse succeeds) are bounded by
    `max_attempts`; a node whose generation never parses is dropped;
  * `max_tokens` 4096 -> 2048: with 4096 a few degenerate n=4 requests exceeded the client's 120 s read timeout and were retried
    until failure on the primary model; an API failure after the client's retries drops that child node (`extra.api_fail` counts them);
  * rollouts, n, and sampling settings are configuration (`rollouts`, `n`, `sql_n`); defaults are the paper's/repo's values
    (24, 3, 5, T=0.8, top_p=0.8; max_tokens 2048 instead of 4096, see below).
Because of the cost (see docs/BASELINES.md) EXPERIMENT_SPEC section 4 runs it on a 500-question dev subset when calls/q > 60.
"""
from __future__ import annotations

import ast
import csv
import json
import glob
import math
import os
import random
import re
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from ._common import ApiFail, BaselineSystem, DEFAULT_EMBED_MODEL, result_key
from ._vendor import __path__ as _VENDOR_PATH

_TPL_DIR = os.path.join(_VENDOR_PATH[0], "alphasql_templates")
_TPL = {os.path.basename(p)[:-4]: open(p, encoding="utf-8").read() for p in glob.glob(os.path.join(_TPL_DIR, "*.txt"))}


def get_prompt(name, **kw):
    return _TPL[name].format(**kw)


# ----------------------------------------------------------------------------- schema (database/schema.py, utils.py)
@dataclass
class Col:
    name: str
    column_type: str = ""
    expanded_column_name: str = ""
    column_description: str = ""
    value_description: str = ""
    value_examples: List[str] = field(default_factory=list)
    primary_key: bool = False
    foreign_keys: list = field(default_factory=list)
    referenced_by: list = field(default_factory=list)


@dataclass
class Tab:
    name: str
    columns: Dict[str, Col] = field(default_factory=dict)


def _norm_desc(d):
    d = d.replace("\r", "").replace("\n", "").replace("commonsense evidence:", "").strip()
    while "  " in d:
        d = d.replace("  ", " ")
    return d


def _read_csv(path):
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    return []


def load_db_description(db_path):
    ddir = os.path.join(os.path.dirname(db_path), "database_description")
    out = {}
    if not os.path.isdir(ddir):
        return out
    for fn in glob.glob(os.path.join(ddir, "*.csv")):
        t = os.path.basename(fn)[:-4].lower().strip()
        d = {}
        for row in _read_csv(fn):
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            oc = row.get("original_column_name", "").lower()
            if not oc:
                continue
            vd = _norm_desc(row.get("value_description", ""))
            if vd.lower().startswith("not useful"):
                vd = vd[len("not useful"):].strip()
            d[oc] = {"expanded": row.get("column_name", ""), "cdesc": _norm_desc(row.get("column_description", "")), "vdesc": vd}
        out[t] = d
    return out


def load_schema(db_path):
    """Port of database/utils.py load_database_schema_dict (sqlite, read-only)."""
    con = sqlite3.connect("file:%s?mode=ro&immutable=1" % db_path, uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    desc = load_db_description(db_path)
    tabs: Dict[str, Tab] = {}
    names = [r[0].strip() for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'")]
    fk_all = []
    pk_of = {}
    for t in names:
        info = con.execute("PRAGMA table_info(`%s`)" % t).fetchall()
        pk_of[t] = [r[1].strip() for r in info if r[5] != 0]
    for t in names:
        for r in con.execute("PRAGMA foreign_key_list(`%s`)" % t).fetchall():
            tc = r[4].strip() if r[4] is not None else (pk_of.get(r[2].strip(), [None]) or [None])[0]
            if tc is not None:
                fk_all.append((t, r[3].strip(), r[2].strip(), tc))
    for t in names:
        tab = Tab(t)
        for r in con.execute("PRAGMA table_info(`%s`)" % t).fetchall():
            cn, ct = r[1].strip(), (r[2] or "").strip()
            c = Col(cn, ct, primary_key=r[5] != 0)
            for (st, sc, tt, tc) in fk_all:
                if st.lower() == t.lower() and sc.lower() == cn.lower():
                    c.foreign_keys.append((tt, tc))
                if tt.lower() == t.lower() and tc.lower() == cn.lower():
                    c.referenced_by.append((st, sc))
            dd = desc.get(t.lower(), {}).get(cn.lower())
            if dd:
                c.expanded_column_name, c.column_description, c.value_description = dd["expanded"], dd["cdesc"], dd["vdesc"]
            if ct.upper() != "BLOB":
                try:
                    c.value_examples = [x[0] for x in con.execute(
                        "SELECT DISTINCT `%s` FROM `%s` WHERE `%s` IS NOT NULL AND `%s` != '' AND length(cast(`%s` as text)) <= 100 LIMIT 3"
                        % (cn, t, cn, cn, cn))]
                except sqlite3.Error:
                    c.value_examples = []
            tab.columns[cn] = c
        tabs[t] = tab
    con.close()
    return tabs


def ddl_statement(tab: Tab):
    """Port of build_table_ddl_statement with all four add_* flags True."""
    st = "CREATE TABLE `%s` (\n" % tab.name
    fks, pks = [], []
    for cn, c in tab.columns.items():
        line = "\t`%s` %s," % (cn, c.column_type)
        parts = []
        if c.expanded_column_name.strip() != "" and c.expanded_column_name.strip().lower() != cn.lower():
            parts.append("Column Meaning: %s" % c.expanded_column_name)
        if (c.column_description.strip() != "" and c.column_description.strip().lower() != cn.lower()
                and c.column_description.strip().lower() != c.expanded_column_name.strip().lower()):
            parts.append("Column Description: %s" % c.column_description)
        if c.value_description.strip() != "":
            parts.append("Value Description: %s" % c.value_description)
        if len(c.value_examples) > 0 and c.column_type.upper() == "TEXT":
            parts.append("Value Examples: %s" % ", ".join("`%s`" % v for v in c.value_examples))
        if parts:
            line += " -- " + " | ".join(parts)
        st += line + "\n"
        if c.primary_key:
            pks.append(cn)
        for f in c.foreign_keys:
            fks.append((cn, *f))
    st += "\tPRIMARY KEY (" + ", ".join("`%s`" % p for p in pks) + "),\n"
    for sc, tt, tc in fks:
        st += "\tFOREIGN KEY (`%s`) REFERENCES `%s`(`%s`),\n" % (sc, tt, tc)
    if st[-2:] == ",\n":
        st = st[:-2]
    return st + "\n);"


def schema_ctx(tabs, shuffle_rng=None):
    stmts = [ddl_statement(t) for t in tabs.values()]
    if shuffle_rng is not None:
        shuffle_rng.shuffle(stmts)
    return "\n".join(stmts)


def subset_schema(tabs: Dict[str, Tab], sel: Dict[str, List[str]]):
    """schema selection filter with PK/FK closure (SchemaSelectionAction.select_schema)."""
    low = [t.lower() for t in sel]
    out = {}
    for stn, scols in sel.items():
        for otn, ot in tabs.items():
            if stn.lower() != otn.lower():
                continue
            nt = Tab(otn)
            for ocn, oc in ot.columns.items():
                keep = any(str(sc).lower() == ocn.lower() for sc in scols) or oc.primary_key
                keep = keep or any(tt.lower() in low for tt, _ in oc.foreign_keys)
                keep = keep or any(st.lower() in low for st, _ in oc.referenced_by)
                if keep:
                    nc = Col(**{k: getattr(oc, k) for k in ("name", "column_type", "expanded_column_name", "column_description",
                                                            "value_description", "primary_key")})
                    nc.value_examples = list(oc.value_examples)
                    nc.foreign_keys = [x for x in oc.foreign_keys if x[0].lower() in low]
                    nc.referenced_by = [x for x in oc.referenced_by if x[0].lower() in low]
                    nt.columns[ocn] = nc
            out[otn] = nt
    return out


# ----------------------------------------------------------------------------- value retrieval (preprocessor + LSH)
_SCHEMA = {}
_VIDX = {}
_LOCK = threading.Lock()


def get_tabs(db_path):
    with _LOCK:
        if db_path not in _SCHEMA:
            _SCHEMA[db_path] = load_schema(db_path)
        return _SCHEMA[db_path]


def _grams(s, n=3):
    return {s[i:i + n] for i in range(len(s) - n + 1)}


class ValueIndex:
    def __init__(self, db_path, tabs, n=3, cap=50000):
        self.n = n
        self.vals = []        # (table, column, value)
        self.grams = []
        self.post = defaultdict(list)
        con = sqlite3.connect("file:%s?mode=ro&immutable=1" % db_path, uri=True)
        con.text_factory = lambda b: b.decode("utf-8", "replace")
        for t, tab in tabs.items():
            for cn, c in tab.columns.items():
                if c.primary_key or c.column_type.lower() != "text":
                    continue
                try:
                    rows = con.execute("SELECT DISTINCT `%s` FROM `%s` WHERE `%s` IS NOT NULL AND trim(cast(`%s` as text)) != '' LIMIT %d"
                                       % (cn, t, cn, cn, cap)).fetchall()
                except sqlite3.Error:
                    continue
                for (v,) in rows:
                    v = str(v)
                    g = _grams(v, n)
                    if not g:
                        continue
                    vid = len(self.vals)
                    self.vals.append((t, cn, v))
                    self.grams.append(g)
                    for x in g:
                        self.post[x].append(vid)
        con.close()

    def query(self, q, thr=0.5, top_k=20):
        qg = _grams(q, self.n)
        if not qg:
            return []
        cnt = Counter()
        for x in qg:
            for vid in self.post.get(x, ()):
                cnt[vid] += 1
        need = math.ceil(thr * len(qg) - 1e-9)
        res = []
        for vid, c in cnt.items():
            if c < need:
                continue
            g = self.grams[vid]
            j = c / (len(g) + len(qg) - c)
            if j >= thr:
                res.append((j, vid))
        res.sort(key=lambda x: -x[0])
        return [{"query": q, "lsh_score": j, "table_name": self.vals[v][0], "column_name": self.vals[v][1], "value": self.vals[v][2]}
                for j, v in res[:top_k]]


def get_vindex(db_path, tabs, cap):
    with _LOCK:
        if db_path in _VIDX:
            return _VIDX[db_path]
    idx = ValueIndex(db_path, tabs, cap=cap)
    with _LOCK:
        _VIDX[db_path] = idx
    return idx


def extract_keywords(ctx, question, hint, cfg):
    prompt = get_prompt("keywords_extraction", QUESTION=question, HINT=hint)
    for attempt in range(int(cfg.get("max_attempts", 3))):
        text = ctx.user(prompt, temperature=0.2, seed=(ctx.seed * 7919 + attempt) % (2 ** 31), max_tokens=512, tag="alpha.keywords")
        m = re.search(r"```python\s*\[(.*?)\]\s*```", text, re.DOTALL)
        if not m:
            continue
        try:
            raw = ast.literal_eval("[%s]" % m.group(1))
        except Exception:
            continue
        kws = []
        for k in raw:
            k = str(k)
            kws.append(k.strip())
            kws.append(k.replace("/", "-").strip("\"").strip("'"))
            for base in (k, k.strip("'"), k.strip("\"")):
                kws.extend(base.replace("=", " ").replace("(", " ").replace(")", " ").replace("_", " ").split(" "))
        return sorted({x.strip() for x in kws if x.strip() != ""})
    return []


def relevant_values(ctx, item, tabs, cfg):
    kws = extract_keywords(ctx, item.question, item.evidence or "", cfg)
    idx = get_vindex(item.db_path, tabs, int(cfg.get("max_values_per_col", 50000)))
    cands = []
    for k in kws:
        cands.extend(idx.query(k, float(cfg.get("lsh_threshold", 0.5)), int(cfg.get("lsh_top_k", 20))))
    cands = [c for c in cands if SequenceMatcher(None, c["value"], c["query"]).ratio() >= float(cfg.get("edit_threshold", 0.3))]
    cands = [c for c in cands if str(c["value"]).strip() and str(c["query"]).strip()]
    if not cands:
        return {}, kws
    texts = sorted({str(c["value"]) for c in cands} | {str(c["query"]) for c in cands})
    emb = ctx.llm.embed_np(texts, cfg["embedding_model"]) if hasattr(ctx.llm, "embed_np") else np.asarray(ctx.llm.embed(texts, cfg["embedding_model"]))
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    pos = {t: i for i, t in enumerate(texts)}
    groups = defaultdict(dict)
    for c in cands:
        s = float(emb[pos[str(c["value"])]] @ emb[pos[str(c["query"])]])
        if s >= float(cfg.get("embedding_threshold", 0.6)):
            key = (c["table_name"], c["column_name"])
            groups[key][c["value"]] = max(s, groups[key].get(c["value"], -1.0))
    return {k: sorted(v, key=lambda x: -v[x]) for k, v in groups.items()}, kws


# ----------------------------------------------------------------------------- execution helpers
def valid_result(res):
    return res["ok"] and any(any(c is not None for c in row) for row in res["rows"])


def fmt_exec(res, row_limit=3, val_limit=100):
    if not res["ok"]:
        return res.get("error") or "error"
    cols = list(res.get("cols") or [])
    rows = res["rows"][:row_limit]
    ncol = len(rows[0]) if rows else len(cols)
    if len(cols) != ncol:
        cols = ["col%d" % i for i in range(ncol)]
    if len(cols) != len(set(cols)):
        cols = [c + "_" + str(i) for i, c in enumerate(cols)]
    cells = [[(str(v)[:val_limit] + "..." if isinstance(v, str) and len(v) > val_limit else str(v)) for v in r] for r in rows]
    w = [max([len(cols[i])] + [len(r[i]) for r in cells]) for i in range(len(cols))]
    sep = "+" + "+".join("-" * (x + 2) for x in w) + "+"
    line = lambda r: "|" + "|".join(" " + r[i].center(w[i]) + " " for i in range(len(r))) + "|"
    return "\n".join([sep, line(cols), sep] + [line(r) for r in cells] + [sep])


def normalize_sql(sql):
    sql = sql.strip()
    if sql.startswith("```sql") and sql.endswith("```"):
        sql = sql[6:-3]
    try:
        import sqlglot
        return sqlglot.parse_one(sql, dialect="sqlite").sql(dialect="sqlite", normalize=True, pretty=False, comments=False)
    except Exception:
        s = sql.replace(";", "").replace("\n", " ").replace("\t", " ")
        while "  " in s:
            s = s.replace("  ", " ")
        return s.strip()


def extract_tag_sql(text):
    m = re.search(r"<sql>(.*)</sql>", text or "", flags=re.DOTALL)
    if not m:
        return None
    return normalize_sql(m.group(1).strip())


# ----------------------------------------------------------------------------- MCTS (algorithm/mcts)
class T(Enum):
    ROOT = "root"
    REPHRASE = "rephrase_question"
    SCHEMA = "schema_selection"
    VALUES = "identify_column_values"
    FUNCS = "identify_column_functions"
    REVISION = "sql_revision"
    GENERATION = "sql_generation"
    END = "end"


A_REPHRASE, A_SCHEMA, A_VALUES, A_FUNCS, A_GEN, A_REV, A_END = "rephrase", "schema", "values", "funcs", "gen", "rev", "end"

VALID = {
    T.ROOT: [A_REPHRASE, A_VALUES, A_FUNCS, A_SCHEMA, A_GEN],
    T.REPHRASE: [A_VALUES, A_FUNCS, A_SCHEMA, A_GEN],
    T.SCHEMA: [A_VALUES, A_FUNCS, A_GEN],
    T.VALUES: [A_FUNCS, A_SCHEMA, A_GEN],
    T.FUNCS: [A_SCHEMA, A_VALUES, A_GEN],
    T.GENERATION: [A_END, A_REV],
    T.REVISION: [A_END],
    T.END: [],
}


class Node:
    __slots__ = ("type", "parent", "action", "depth", "children", "path", "rephrased", "sel_schema", "sel_ctx", "values", "funcs",
                 "sql", "rev_sql", "final_sql", "cons", "is_valid", "N", "Q")

    def __init__(self, type_, parent=None, action=None):
        self.type, self.parent, self.action = type_, parent, action
        self.depth = 0 if parent is None else parent.depth + 1
        self.children = []
        self.path = []
        self.rephrased = self.sel_schema = self.sel_ctx = self.values = self.funcs = None
        self.sql = self.rev_sql = self.final_sql = self.cons = self.is_valid = None
        self.N, self.Q = 0, 0.0
        if parent is not None:      # inherit state (original: copy.deepcopy(node))
            for k in ("rephrased", "sel_schema", "sel_ctx", "values", "funcs", "sql", "rev_sql", "final_sql", "cons", "is_valid"):
                setattr(self, k, getattr(parent, k))
            self.path = parent.path + [self]

    def terminal(self):
        return self.type == T.END


class AlphaSearch:
    def __init__(self, ctx, item, tabs, cfg, rng):
        self.ctx, self.item, self.cfg, self.rng = ctx, item, cfg, rng
        self.tabs = tabs
        self.question, self.hint = item.question, item.evidence or ""
        self.ctx_full = schema_ctx(tabs)
        self.n = int(cfg["n"])
        self.sql_n = int(cfg["sql_n"])
        self.temp = float(cfg["sample_temperature"])
        self.max_attempts = int(cfg["max_attempts"])
        self._seed_ctr = 0
        self.cache_exec = {}
        self.generated = []
        self.api_fail = 0

    # ---- LLM sampling
    def _next_seed(self):
        self._seed_ctr += 1
        return (self.ctx.seed * 1000003 + self._seed_ctr * 7919 + 12345) % (2 ** 31)

    def sample(self, prompt, n, seed):
        kw = dict(temperature=self.temp, max_tokens=int(self.cfg["max_tokens"]), tag="alpha.mcts")
        if self.cfg.get("top_p") is not None:
            kw["top_p"] = float(self.cfg["top_p"])
        if n == 1:
            return [self.ctx.user(prompt, seed=seed, **kw)]
        # DashScope accepts n <= 4 per request (400 "Range of n should be [1, 4]"): split larger groups into several requests.
        mx = int(self.cfg.get("max_n_per_request", 4))
        out, left, k = [], n, 0
        while left > 0:
            m = min(mx, left)
            r = self.ctx.user(prompt, seed=seed + 7 * k, n=m, **kw) if m > 1 else self.ctx.user(prompt, seed=seed + 7 * k, **kw)
            out.extend(r if isinstance(r, list) else [r])
            left -= m
            k += 1
        return out

    def execute(self, sql):
        if sql not in self.cache_exec:
            self.cache_exec[sql] = self.ctx.execute(sql)
        return self.cache_exec[sql]

    # ---- context helpers
    def _thoughts(self, node, with_sql=False):
        parts = ""
        for p in node.path:
            if p.action == A_VALUES:
                parts += "Identify column values: %s\n" % p.values
            elif p.action == A_FUNCS:
                parts += "Identify column functions: %s\n" % p.funcs
            elif with_sql and p.action == A_GEN:
                res = self.execute(p.sql) if p.sql else {"ok": False, "error": "no sql", "rows": []}
                parts += "SQL generation: %s\nSQL execution result:\n%s\n" % (p.sql, fmt_exec(res))
        return parts

    def _q(self, node):
        return node.rephrased if node.rephrased else self.question

    def _ctx(self, node):
        return node.sel_ctx if node.sel_ctx else self.ctx_full

    def _hint(self, node, with_sql=False):
        th = self._thoughts(node, with_sql)
        return self.hint + ("\n\nHere are my previous thoughts:\n%s" % th if th else "")

    # ---- actions -> children
    def children_of(self, node, action, seed, rng):
        if action == A_REPHRASE:
            rs = list(set(self.sample(get_prompt("raphrase_question", QUESTION=self.question, HINT=self.hint), self.n, seed)))
            out = []
            for r in rs:
                c = Node(T.REPHRASE, node, action)
                c.rephrased = r
                out.append(c)
            return out
        if action in (A_VALUES, A_FUNCS):
            name = "identify_column_values" if action == A_VALUES else "identify_column_functions"
            prompt = get_prompt(name, QUESTION=self._q(node), HINT=self._hint(node), SCHEMA_CONTEXT=self._ctx(node))
            out = []
            for r in list(set(self.sample(prompt, self.n, seed))):
                c = Node(T.VALUES if action == A_VALUES else T.FUNCS, node, action)
                if action == A_VALUES:
                    c.values = r
                else:
                    c.funcs = r
                out.append(c)
            return out
        if action == A_SCHEMA:
            return self._schema_children(node, seed, rng)
        if action == A_GEN:
            return self._sql_child(node, seed, "sql_generation", T.GENERATION, with_sql=False)
        if action == A_REV:
            return self._sql_child(node, seed, "sql_revision", T.REVISION, with_sql=True)
        if action == A_END:
            c = Node(T.END, node, action)
            c.final_sql = node.sql if node.type == T.GENERATION else node.rev_sql
            return [c]
        raise ValueError(action)

    def _schema_children(self, node, seed, rng):
        # the original passes the full schema here, not the already-selected one
        prompt = get_prompt("schema_selection", QUESTION=self._q(node), HINT=self._hint(node), SCHEMA_CONTEXT=self.ctx_full)
        nodes, seen = [], set()
        for attempt in range(self.max_attempts):
            if len(nodes) >= self.n:
                break
            for resp in self.sample(prompt, self.n - len(nodes), seed + attempt * 31):
                try:
                    sel = json.loads(re.search(r"```json\n(.*)```", resp, flags=re.DOTALL).group(1))
                except Exception:
                    continue
                if not isinstance(sel, dict):
                    continue
                sub = subset_schema(self.tabs, {k: (v if isinstance(v, list) else []) for k, v in sel.items()})
                if not sub:
                    continue
                key = frozenset((t.lower(), str(c).lower()) for t, cs in sel.items() if isinstance(cs, list) for c in cs)
                c = Node(T.SCHEMA, node, A_SCHEMA)
                c.sel_schema = sub
                c.sel_ctx = schema_ctx(sub, rng)
                if key in seen:
                    continue
                seen.add(key)
                nodes.append(c)
        return nodes[: self.n]

    def _sql_child(self, node, seed, tpl, ntype, with_sql):
        prompt = get_prompt(tpl, QUESTION=self._q(node), HINT=self._hint(node, with_sql=with_sql), SCHEMA_CONTEXT=self._ctx(node))
        sql, score, valid = self._most_consistent(prompt, seed)
        if not sql:
            return []
        c = Node(ntype, node, A_GEN if ntype == T.GENERATION else A_REV)
        if ntype == T.GENERATION:
            c.sql = sql
            self.generated.append(sql)
        else:
            c.rev_sql = sql
        c.cons, c.is_valid = score, valid
        return [c]

    def _most_consistent(self, prompt, seed):
        allq, groups, tries, spare = [], defaultdict(list), 0, []
        maxtries = int(self.cfg["sql_validation_max_tries"])
        k = 0
        while len(allq) < self.sql_n:
            if tries >= maxtries and allq:
                break
            if k >= self.max_attempts:
                break
            for resp in self.sample(prompt, self.sql_n - len(allq), seed + k * 17):
                sql = extract_tag_sql(resp)
                if sql is None:
                    continue
                res = self.execute(sql)
                if valid_result(res) or tries >= maxtries:
                    allq.append(sql)
                    if valid_result(res):
                        groups[result_key(res)].append(sql)
                else:
                    tries += 1
                    spare.append(sql)
            k += 1
        if not allq and spare:           # bounded retries ran out: accept an executed-but-invalid SQL like the original after 15 tries
            allq = spare[:1]
        if not groups and allq:
            return self.rng.choice(allq), 0.0, False
        if not groups:
            return None, 0.0, False
        best, bs, tot = None, 0, 0
        for r, qs in groups.items():
            tot += len(qs)
            if len(qs) > bs:
                best, bs = self.rng.choice(qs), len(qs)
        return best, bs / tot, True

    # ---- MCTS
    def valid_actions(self, node):
        used = {p.action for p in node.path if p.action is not None}
        return [a for a in VALID[node.type] if a not in used]

    def _safe_children(self, nd, a, sd, rg):
        """An API failure (after the client's own retries) drops that child instead of aborting the whole search."""
        try:
            return self.children_of(nd, a, sd, rg)
        except ApiFail:
            self.api_fail += 1
            return []

    def expand(self, node):
        acts = self.valid_actions(node)
        jobs = [(node, a, self._next_seed(), random.Random(self.rng.random())) for a in acts]
        outs = self.ctx.pmap(lambda nd, a, sd, rg: self._safe_children(nd, a, sd, rg), jobs,
                             workers=int(self.cfg.get("parallel", 8)))
        for ch in outs:
            node.children.extend(ch)
        self.rng.shuffle(node.children)

    def select(self, node):
        cur = node
        c = float(self.cfg["exploration_constant"])
        while cur.children and not cur.terminal():
            unvisited = [ch for ch in cur.children if ch.N == 0]
            if unvisited:
                return unvisited[0]
            cur = max(cur.children, key=lambda ch: ch.Q / ch.N + c * math.sqrt(math.log(cur.N) / ch.N))
        return cur

    def simulate(self, node):
        cur = node
        while not cur.terminal():
            self.expand(cur)
            if not cur.children:
                return None
            cur = self.rng.choice(cur.children)
        return cur

    @staticmethod
    def backprop(node):
        cur = node
        reward = (cur.cons or 0.0) if cur.N == 0 else cur.Q / cur.N
        while cur is not None:
            cur.N += 1
            cur.Q += reward
            cur = cur.parent

    def solve(self):
        root = Node(T.ROOT)
        root.path = [root]
        for _ in range(int(self.cfg["rollouts"])):
            leaf = self.select(root)
            if leaf.terminal():
                self.backprop(leaf)
                continue
            self.expand(leaf)
            if not leaf.children:
                leaf.N += 1          # nothing to explore below this node: avoid re-selecting it forever
                continue
            leaf = self.rng.choice(leaf.children)
            end = self.simulate(leaf)
            if end is not None:
                self.backprop(end)
        ends = []
        stack = [root]
        while stack:
            n = stack.pop()
            if n.type == T.END:
                ends.append(n)
            stack.extend(n.children)
        return ends


def select_final(ctx, ends, repeat, fallback):
    groups, groups_inv = defaultdict(list), defaultdict(list)
    sqls = [e.final_sql for e in ends if e.final_sql]
    for i, s in enumerate(sqls):
        res = ctx.execute(s)
        if res["ok"]:
            k = result_key(res)
            if valid_result(res):
                groups[k].append(i)
            groups_inv[k].append(i)
    g = groups if groups else groups_inv
    if not g:
        return (fallback[0] if fallback else ""), {"selection": "fallback"}
    tot = sum(len(v) for v in g.values())
    scored = []
    for k, idx in g.items():
        t = []
        for _ in range(repeat):
            t0 = time.time()
            r = ctx.execute(sqls[idx[0]])
            if r["ok"]:
                t.append(r.get("elapsed_s", time.time() - t0))
        scored.append((len(idx) / tot, -(sum(t) / len(t) if t else 1e9), idx[0]))
    scored.sort(reverse=True)
    best = scored[0]
    return sqls[best[2]], {"selection": "vote", "n_end": len(sqls), "n_groups": len(g), "top_share": round(best[0], 3)}


class AlphaSQL(BaselineSystem):
    name = "alpha_sql"
    expensive = True
    default_cfg = {"rollouts": 24, "exploration_constant": 1.414, "n": 3, "sql_n": 5, "sample_temperature": 0.8, "top_p": 0.8,
                   "max_tokens": 2048, "sql_validation_max_tries": 15, "max_attempts": 3, "value_retrieval": True,
                   "embedding_model": DEFAULT_EMBED_MODEL, "max_values_per_col": 50000, "time_repeat": 3, "parallel": 8, "max_n_per_request": 4}

    def solve(self, item, seed, ctx):
        tabs = get_tabs(item.db_path)
        # question-specific value examples (preprocessor)
        import copy
        tabs = copy.deepcopy(tabs)
        n_vals = 0
        if self.cfg.get("value_retrieval", True):
            rel, _kws = relevant_values(ctx, item, tabs, self.cfg)
            for (t, c), vals in rel.items():
                if t in tabs and c in tabs[t].columns:
                    tabs[t].columns[c].value_examples = vals[:3]
                    n_vals += 1
        rng = random.Random((hash(item.qid) & 0xFFFFFFFF) ^ (seed * 2654435761 & 0xFFFFFFFF))
        search = AlphaSearch(ctx, item, tabs, self.cfg, rng)
        ends = search.solve()
        sql, info = select_final(ctx, ends, int(self.cfg["time_repeat"]), search.generated)
        info.update({"rollouts": int(self.cfg["rollouts"]), "value_columns": n_vals, "api_fail": search.api_fail, "samples": ctx.samples})
        return sql, {"iterations": int(self.cfg["rollouts"]), "actions": ["mcts x%d" % int(self.cfg["rollouts"]), "vote"],
                     "error_class": None if sql else "no_sql", "extra": info}
