#!/usr/bin/env python3
"""Controlled critic diagnostic ("injected semantic errors"), SAP Amendment 4 (docs/SAP.md section 16).  scope = diagnostic-injected.

What it does (worker machine side, Python 3.12 + sqlglot; nothing here touches code/prompts/** or exp/models.yaml):
  build    rule-based, seeded error injection into the GOLD SQL of BIRD-train POOL questions (no LLM, no model output is read).  Writes the frozen
           item list results/diag/items.json (+ items.csv).  Offline.
  run      for every item: M1 intent + M3 context exactly as Pipeline.root() builds them (frozen full_fixed config, qwen37plus self-critic), then the
           UNCHANGED M5 critic (v3, weights 0.3/0.5/0.2) on the injected SQL and on the gold control, and the UNCHANGED P0 revision loop
           (Pipeline.act / FullSystem._run_p0 logic: revise while score < theta_acc, t_max 3, final = best node) starting from that SQL.
  analyze  offline: detection / false-positive / AUROC / repair with database-clustered bootstrap intervals -> results/diag/*.json + cases.

The only generation that is skipped is the root SQL generation: the injected SQL (or the gold control) takes the place of the root candidate.
Gold SQL is used offline to build faults and to score repairs; it is never shown to the critic or to the reviser.

Usage (cwd = code dir, python = worker machine venv):
  python tools/critic_diag.py build   --out ../results/diag [--explore]
  python tools/critic_diag.py run     --items ../results/diag/items.json --out $R/runs/DIAG/run1 [--conc 4] [--limit N]
  python tools/critic_diag.py analyze --items items.json --run run1/records.jsonl --out ../results/diag
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import random
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)

SEED = 20261030
N_PER_CAT = 44                 # perturbed items per category (SAP 16.2)
N_CTRL_PER_CAT = 22            # gold controls per category (the first accepted items of the category)
CATEGORIES = [                 # fixed order = assignment order (a question belongs to at most one category)
    "distinct_drop", "limit_change", "order_flip", "count_distinct", "agg_maxmin", "agg_sumavg", "cmp_flip", "where_drop", "col_swap",
]
MAX_GOLD_ROWS = 1000           # gold result must be a complete set of at most this many rows
EXEC_TIMEOUT_S = 10.0          # gold and injected SQL must finish within this
SCOPE = "diagnostic-injected"


# ======================================================================================================== tokens / splicing
def _tokens(sql):
    import sqlglot
    return sqlglot.Dialect.get_or_raise("sqlite").tokenize(sql)


def _splice(sql, edits):
    """edits = [(start, end_exclusive, replacement)], non-overlapping."""
    out, pos = [], 0
    for s, e, r in sorted(edits):
        out.append(sql[pos:s])
        out.append(r)
        pos = e
    out.append(sql[pos:])
    return "".join(out)


def _same_case(orig, new):
    return new.upper() if orig.isupper() else (new.lower() if orig.islower() else new.capitalize() if orig[:1].isupper() else new)


def _kw(t, word):
    return t.text.upper() == word and t.token_type.name not in ("STRING", "IDENTIFIER")


def _is_name(t):
    return t.token_type.name in ("VAR", "IDENTIFIER")


def _depths(toks):
    d, out = 0, []
    for t in toks:
        if t.token_type.name == "L_PAREN":
            out.append(d)
            d += 1
        elif t.token_type.name == "R_PAREN":
            d -= 1
            out.append(d)
        else:
            out.append(d)
    return out


# ======================================================================================================== edit generators
# every generator returns a list of {"edits": [(s, e, repl)], "info": {...}}; the engine shuffles them with a per-item seed and takes the first valid one

def gen_swap_function(sql, toks, pair):
    out = []
    for i, t in enumerate(toks[:-1]):
        if t.text.upper() in pair and t.token_type.name not in ("STRING", "IDENTIFIER") and toks[i + 1].token_type.name == "L_PAREN":
            new = _same_case(sql[t.start:t.end + 1], pair[t.text.upper()])
            out.append({"edits": [(t.start, t.end + 1, new)], "info": {"from": t.text.upper(), "to": pair[t.text.upper()]}})
    return out


def gen_agg_maxmin(sql, toks, schema=None):
    return gen_swap_function(sql, toks, {"MAX": "MIN", "MIN": "MAX"})


def gen_agg_sumavg(sql, toks, schema=None):
    return gen_swap_function(sql, toks, {"SUM": "AVG", "AVG": "SUM"})


def gen_count_distinct(sql, toks, schema=None):
    out = []
    for i, t in enumerate(toks[:-2]):
        if not (_kw(t, "COUNT") and toks[i + 1].token_type.name == "L_PAREN"):
            continue
        a = toks[i + 2]
        if _kw(a, "DISTINCT"):
            nxt = toks[i + 3] if i + 3 < len(toks) else None
            end = nxt.start if nxt is not None else a.end + 1
            out.append({"edits": [(a.start, end, "")], "info": {"from": "COUNT(DISTINCT x)", "to": "COUNT(x)"}})
        elif a.token_type.name not in ("STAR", "NUMBER", "R_PAREN"):
            out.append({"edits": [(a.start, a.start, "DISTINCT ")], "info": {"from": "COUNT(x)", "to": "COUNT(DISTINCT x)"}})
    return out


def gen_cmp_flip(sql, toks, schema=None):
    flip = {">": "<", "<": ">", ">=": "<=", "<=": ">="}
    out = []
    for t in toks:
        if t.token_type.name in ("GT", "LT", "GTE", "LTE") and t.text in flip:
            out.append({"edits": [(t.start, t.end + 1, flip[t.text])], "info": {"from": t.text, "to": flip[t.text]}})
    return out


def gen_limit_change(sql, toks, schema=None):
    out = []
    for i, t in enumerate(toks[:-1]):
        n = toks[i + 1]
        after = toks[i + 2] if i + 2 < len(toks) else None
        if _kw(t, "LIMIT") and n.token_type.name == "NUMBER" and n.text == "1" and not (after is not None and after.token_type.name == "COMMA"):
            out.append({"edits": [(n.start, n.end + 1, "2")], "info": {"from": "LIMIT 1", "to": "LIMIT 2"}})
    return out


def gen_distinct_drop(sql, toks, schema=None):
    out = []
    for i, t in enumerate(toks[1:], 1):
        if _kw(t, "DISTINCT") and _kw(toks[i - 1], "SELECT"):
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            end = nxt.start if nxt is not None else t.end + 1
            out.append({"edits": [(t.start, end, "")], "info": {"from": "SELECT DISTINCT", "to": "SELECT"}})
    return out


_CLAUSE_END = ("GROUP BY", "ORDER BY", "LIMIT", "HAVING", "UNION", "INTERSECT", "EXCEPT", "WINDOW")


def gen_order_flip(sql, toks, schema=None):
    """First key of each ORDER BY: ASC <-> DESC (a missing direction counts as ASC and gets DESC appended)."""
    dep = _depths(toks)
    out = []
    for i, t in enumerate(toks[:-1]):
        if not _kw(t, "ORDER BY"):          # sqlglot tokenizes "ORDER BY" as ONE token
            continue
        d0 = dep[i]
        j = i + 1
        last = None
        while j < len(toks):
            u = toks[j]
            if dep[j] < d0 or (dep[j] == d0 and (u.token_type.name in ("COMMA", "SEMICOLON") or any(_kw(u, w) for w in ("LIMIT", "UNION", "INTERSECT", "EXCEPT")))):
                break
            if dep[j] == d0 and u.token_type.name == "R_PAREN":
                break
            last = j
            j += 1
        if last is None or last < i + 1:
            continue
        lt = toks[last]
        if _kw(lt, "DESC"):
            out.append({"edits": [(lt.start, lt.end + 1, _same_case(sql[lt.start:lt.end + 1], "ASC"))], "info": {"from": "DESC", "to": "ASC"}})
        elif _kw(lt, "ASC"):
            out.append({"edits": [(lt.start, lt.end + 1, _same_case(sql[lt.start:lt.end + 1], "DESC"))], "info": {"from": "ASC", "to": "DESC"}})
        else:
            out.append({"edits": [(lt.end + 1, lt.end + 1, " DESC")], "info": {"from": "ASC (implicit)", "to": "DESC"}})
    return out


def where_clauses(toks):
    """-> list of dict(where=tok index, end=index after last tok, conj=[(first, last)], ok=no top-level OR) for every WHERE at its own depth."""
    dep = _depths(toks)
    res = []
    for i, t in enumerate(toks):
        if not _kw(t, "WHERE"):
            continue
        d0 = dep[i]
        j = i + 1
        conj, first, between, has_or = [], i + 1, False, False
        while j < len(toks):
            u = toks[j]
            if dep[j] < d0 or (u.token_type.name == "R_PAREN" and dep[j] == d0 - 1) or (dep[j] == d0 and (u.token_type.name == "SEMICOLON" or any(_kw(u, w) for w in _CLAUSE_END))):
                break
            if dep[j] == d0:
                if _kw(u, "BETWEEN"):
                    between = True
                elif _kw(u, "AND"):
                    if between:
                        between = False
                    else:
                        conj.append((first, j - 1, j))
                        first = j + 1
                elif _kw(u, "OR"):
                    has_or = True
            j += 1
        if first <= j - 1:
            conj.append((first, j - 1, None))
        res.append({"where": i, "end": j, "conj": conj, "ok": not has_or and bool(conj)})
    return res


def _trim_left(sql, s):
    while s > 0 and sql[s - 1] == " ":
        s -= 1
    return s


def gen_where_drop(sql, toks, schema=None):
    out = []
    for wc in where_clauses(toks):
        if not wc["ok"]:
            continue
        conj = wc["conj"]
        n = len(conj)
        for k, (a, b, and_idx) in enumerate(conj):
            if n == 1:
                s, e = _trim_left(sql, toks[wc["where"]].start), toks[b].end + 1
                edits = [(s, e, "")]
            elif k < n - 1:
                edits = [(toks[a].start, toks[conj[k + 1][0]].start, "")]
            else:
                prev_and = conj[k - 1][2]
                edits = [(_trim_left(sql, toks[prev_and].start), toks[b].end + 1, "")]
            out.append({"edits": edits, "info": {"n_conjuncts": n, "dropped": sql[toks[a].start:toks[b].end + 1][:120]}})
    return out


def _decl_norm(t):
    return re.sub(r"\s+", " ", (t or "").strip().upper())


def gen_col_swap(sql, toks, schema=None):
    """Replace one column reference by a different column of the SAME table with the same declared type.  Needs the schema
    ({"tables": {name: [{"name","type",...}]}}) and sqlglot to resolve aliases."""
    import sqlglot
    from sqlglot import exp
    if not schema:
        return []
    try:
        ast = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return []
    tbl_cols = {t.lower(): cols for t, cols in schema["tables"].items()}
    tbl_real = {t.lower(): t for t in schema["tables"]}
    alias2tbl = {}
    for tb in ast.find_all(exp.Table):
        nm = (tb.name or "").lower()
        if nm in tbl_cols:
            alias2tbl[(tb.alias or tb.name).lower()] = nm
            alias2tbl[nm] = nm
    col_alias = set((a.alias or "").lower() for a in ast.find_all(exp.Alias))
    in_query = set(alias2tbl.values())
    name_tables = collections.defaultdict(set)
    for tn in in_query:
        for c in tbl_cols[tn]:
            name_tables[c["name"].lower()].add(tn)
    taken = set(alias2tbl)
    clause_kw = {"SELECT": "select", "FROM": "from", "ON": "join_on", "WHERE": "where", "GROUP BY": "group", "ORDER BY": "order", "HAVING": "having"}
    out = []
    loc = "select"
    for i, t in enumerate(toks):
        if t.token_type.name not in ("DOT", "COMMA") and t.text.upper() in clause_kw and t.token_type.name not in ("STRING", "IDENTIFIER"):
            loc = clause_kw[t.text.upper()]
        if not _is_name(t):
            continue
        prev = toks[i - 1] if i > 0 else None
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        tbl = None
        if prev is not None and prev.token_type.name == "DOT":
            q = toks[i - 2] if i >= 2 else None
            before = toks[i - 3] if i >= 3 else None
            if q is None or not _is_name(q) or (before is not None and before.token_type.name == "DOT"):
                continue
            tbl = alias2tbl.get(q.text.lower())
            if tbl is None or (nxt is not None and nxt.token_type.name in ("DOT", "L_PAREN")):
                continue
        else:
            if nxt is not None and nxt.token_type.name in ("DOT", "L_PAREN"):
                continue
            if prev is not None and _kw(prev, "AS"):
                continue
            nm = t.text.lower()
            if nm in col_alias or nm in taken or nm in tbl_real:
                continue
            ts = name_tables.get(nm)
            if not ts or len(ts) != 1:
                continue
            tbl = list(ts)[0]
        cols = tbl_cols[tbl]
        orig = [c for c in cols if c["name"].lower() == t.text.lower()]
        if not orig:
            continue
        o = orig[0]
        sibs = [c for c in cols if c["name"].lower() != o["name"].lower() and _decl_norm(c["type"]) == _decl_norm(o["type"]) and _decl_norm(o["type"]) != ""]
        raw = sql[t.start:t.end + 1]
        for c in sibs:
            nm = c["name"]
            if t.token_type.name == "IDENTIFIER" and raw[:1] in ("`", '"', "["):
                q0 = raw[0]
                new = q0 + nm + ("]" if q0 == "[" else q0)
            elif re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", nm):
                new = nm
            else:
                new = "`" + nm + "`"
            out.append({"edits": [(t.start, t.end + 1, new)], "info": {"from": o["name"], "to": nm, "table": tbl_real[tbl], "loc": loc, "type": o["type"]}})
    return out


GENERATORS = {
    "distinct_drop": gen_distinct_drop, "limit_change": gen_limit_change, "order_flip": gen_order_flip, "count_distinct": gen_count_distinct,
    "agg_maxmin": gen_agg_maxmin, "agg_sumavg": gen_agg_sumavg, "cmp_flip": gen_cmp_flip, "where_drop": gen_where_drop, "col_swap": gen_col_swap,
}


# ======================================================================================================== validity
def _rows_bag(rows):
    return collections.Counter(tuple(r) for r in rows)


def _comparator(cat):
    return "bag" if cat == "distinct_drop" else "set"


def differs(cat, gold_rows, pert_rows):
    """True iff the injected result differs from gold under the category's comparator (set for all, bag for distinct_drop)."""
    if cat == "distinct_drop":
        return _rows_bag(gold_rows) != _rows_bag(pert_rows)
    from nl2sql import db
    return db.ex_match(pert_rows, gold_rows) == 0


def exec_gold(item):
    from nl2sql import db
    from nl2sql.systems.ours import m5 as M5
    r = db.execute(item.db_path, item.gold_sql, timeout_s=EXEC_TIMEOUT_S, max_rows=MAX_GOLD_ROWS + 1)
    if not r["ok"] or r["truncated"] or r["elapsed_s"] > EXEC_TIMEOUT_S or M5.exec_status(r) != "ok_nonempty":
        return None
    return r


def valid_injection(cat, item, gres, sql):
    """executes (no error), non-empty and not an all-NULL row (exec status ok_nonempty), and differs from gold under the comparator."""
    from nl2sql import db
    from nl2sql.systems.ours import m5 as M5
    if sql.strip() == item.gold_sql.strip():
        return None
    r = db.execute(item.db_path, sql, timeout_s=EXEC_TIMEOUT_S, max_rows=MAX_GOLD_ROWS + 1)
    if not r["ok"] or M5.exec_status(r) != "ok_nonempty":
        return None
    if r["truncated"]:
        return r                                   # > MAX_GOLD_ROWS rows against a gold of <= MAX_GOLD_ROWS rows: differs for sure
    return r if differs(cat, gres["rows"], r["rows"]) else None


# ======================================================================================================== build
def burnin_qids(path):
    s = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                s.add(json.loads(line)["qid"])
    return s


def pool_items():
    from nl2sql import data
    return data.load_split("bird_train_pool", with_gold=True)


def schema_of(item):
    from nl2sql import data
    return data.get_schema(item)


def draw_category(cat, items_by_db, used, excluded, n, seed=SEED, stats=None):
    """Round-robin over databases (seeded order), one candidate per database per round, seeded order inside a database; a candidate is accepted
    iff the first valid edit (seeded order) exists.  Returns the accepted records in draw order."""
    rng = random.Random("%s:%s" % (seed, cat))
    dbs = sorted(items_by_db)
    rng.shuffle(dbs)
    queues = {}
    for d in dbs:
        its = sorted(items_by_db[d], key=lambda i: i.qid)
        rng.shuffle(its)
        queues[d] = its
    gen = GENERATORS[cat]
    accepted = []
    st = stats if stats is not None else {}
    st.update({"examined": 0, "no_edit": 0, "no_valid": 0, "gold_bad": 0})
    while len(accepted) < n and any(queues.values()):
        for d in dbs:
            if len(accepted) >= n:
                break
            q = queues[d]
            while q:
                it = q.pop()
                if it.qid in used or it.qid in excluded:
                    continue
                st["examined"] += 1
                toks = _tokens(it.gold_sql) if it.gold_sql.strip() else []
                sc = schema_of(it) if cat == "col_swap" else None
                cands = gen(it.gold_sql, toks, sc) if toks else []
                if not cands:
                    st["no_edit"] += 1
                    continue
                gres = exec_gold(it)
                if gres is None:
                    st["gold_bad"] += 1
                    continue
                erng = random.Random("%s:%s:%s" % (seed, cat, it.qid))
                erng.shuffle(cands)
                got = None
                for c in cands[:12]:                       # at most 12 tries per question (bounds the execution time on large databases)
                    sql = _splice(it.gold_sql, c["edits"])
                    r = valid_injection(cat, it, gres, sql)
                    if r is not None:
                        got = (c, sql, r)
                        break
                if got is None:
                    st["no_valid"] += 1
                    continue
                c, sql, r = got
                used.add(it.qid)
                accepted.append({"cat": cat, "qid": it.qid, "db_id": it.db_id, "question": it.question, "evidence": it.evidence,
                                 "gold_sql": it.gold_sql, "pert_sql": sql, "edit": c["info"], "n_edit_candidates": len(cands),
                                 "gold_n_rows": len(gres["rows"]), "pert_n_rows": len(r["rows"]), "pert_truncated": bool(r["truncated"]),
                                 "comparator": _comparator(cat)})
                break
    return accepted


def cmd_build(a):
    from nl2sql import paths
    out = a.out
    os.makedirs(out, exist_ok=True)
    pool = pool_items()
    sp = json.load(open(str(paths.SPLITS_JSON)))
    held = set(sp["held_out_dbs"])
    assert not (held & set(i.db_id for i in pool)), "pool must be disjoint from the HT/HR (held-out) databases"
    bl = os.path.join(str(paths.REPO), "exp", "frozen", "corpus_burnin.sqlite.log.jsonl") if not a.burnin_log else a.burnin_log
    if not os.path.exists(bl):
        bl = os.path.join(str(paths.ROOT), "exp", "frozen", "corpus_burnin.sqlite.log.jsonl")
    excl = burnin_qids(bl)
    burn = set(excl)
    extra = set()                                  # SAP Amendment 6: extra qids kept out of a new draw (the evaluation items); empty = Amendment 4 behaviour
    if getattr(a, "exclude_items", ""):
        extra = set(x["qid"] for x in json.load(open(a.exclude_items))["items"])
        excl = burn | extra
    elig = [i for i in pool if i.qid not in excl]
    by_db = collections.defaultdict(list)
    for i in elig:
        by_db[i.db_id].append(i)
    print("pool %d items, %d dbs; burn-in %d qids (%s); eligible %d items in %d dbs" % (len(pool), len(set(i.db_id for i in pool)), len(excl), bl, len(elig), len(by_db)))
    n = a.n if a.n else N_PER_CAT
    seed = int(getattr(a, "seed", 0) or SEED)
    n_ctrl = int(getattr(a, "ctrl", 0) or N_CTRL_PER_CAT)
    idp = getattr(a, "id_prefix", "") or "D"
    used, items, stats = set(), [], {}
    for cat in CATEGORIES:
        st = {}
        t0 = time.time()
        acc = draw_category(cat, by_db, used, set(), n, seed=seed, stats=st)
        stats[cat] = dict(st, accepted=len(acc), dbs=len(set(x["db_id"] for x in acc)), secs=round(time.time() - t0, 1))
        print("%-15s accepted %3d in %2d dbs  %s" % (cat, len(acc), stats[cat]["dbs"], {k: v for k, v in st.items()}), flush=True)
        for k, x in enumerate(acc):
            x["rank"] = k
            x["control"] = k < n_ctrl
        items.extend(acc)
    for k, x in enumerate(items):
        x["id"] = "%s%03d" % (idp, k)
    body = {"_meta": {"scope": SCOPE, "seed": seed, "n_per_cat": n, "n_ctrl_per_cat": n_ctrl, "categories": CATEGORIES,
                      "pool_items": len(pool), "pool_dbs": len(set(i.db_id for i in pool)), "burnin_excluded": len(burn),
                      "eligible_items": len(elig), "burnin_log": os.path.basename(bl), "build_stats": stats,
                      "rule": "docs/SAP.md section 16"},
            "items": items}
    if extra:
        body["_meta"]["extra_excluded"] = len(extra)
        body["_meta"]["rule"] = "docs/SAP.md section 18.3 (tuning draw; same generator as section 16)"
    p = os.path.join(out, "items.json")
    json.dump(body, open(p, "w"), indent=1, ensure_ascii=False, sort_keys=True)
    with open(os.path.join(out, "items.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "cat", "qid", "db_id", "control", "comparator", "edit", "gold_n_rows", "pert_n_rows", "gold_sql", "pert_sql", "question"])
        for x in items:
            w.writerow([x["id"], x["cat"], x["qid"], x["db_id"], int(x["control"]), x["comparator"], json.dumps(x["edit"], ensure_ascii=False),
                        x["gold_n_rows"], x["pert_n_rows"], x["gold_sql"], x["pert_sql"], x["question"]])
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    print("items %d (controls %d) dbs %d  sha256 %s" % (len(items), sum(1 for x in items if x["control"]), len(set(x["db_id"] for x in items)), h))


# ======================================================================================================== run
def _score_sql(item, sql, gold_rows):
    from nl2sql import db
    r = db.execute(item.db_path, sql, timeout_s=30, max_rows=20000)
    if not r["ok"] or r["truncated"]:
        return {"ok": bool(r["ok"]), "ex_set": 0, "ex_bag": 0, "n_rows": len(r["rows"])}
    return {"ok": True, "ex_set": db.ex_match(r["rows"], gold_rows), "ex_bag": int(_rows_bag(r["rows"]) == _rows_bag(gold_rows)), "n_rows": len(r["rows"])}


def _crit_brief(c):
    return {k: c.get(k) for k in ("syntax", "semantic", "efficiency", "score", "status", "parse_ok", "issues", "suggestions")}


def eval_variant(pipe, cfg, item, node0, gold_rows, theta_acc, t_max, seed):
    """FullSystem._run_p0 on a given root node: revise while the critic score is below theta_acc (max t_max revisions), final = best node
    by (score, earlier).  Returns the per-round log and the pipeline's final choice."""
    from nl2sql.systems.ours import m5 as M5
    nodes, cur, actions = [node0], node0, []
    rounds = []
    while True:
        s = cur["critic"]["score"]
        if s >= theta_acc:
            actions.append("accept")
            break
        if cur["k"] >= t_max:
            break
        child = pipe.act(item, cur, "revise", T=cfg["temperature"], seed=seed, corpus=None)
        actions.append("revise")
        rounds.append({"sql": child["sql"], "noop": child["sql"].strip() == cur["sql"].strip(), "critic": _crit_brief(child["critic"]),
                       "score_res": _score_sql(item, child["sql"], gold_rows)})
        if child["sql"].strip() == cur["sql"].strip():
            actions.append("revise_noop")
            nodes.append(child)
            break
        nodes.append(child)
        cur = child
    best = max(nodes, key=lambda n: (n["critic"]["score"], -n["k"]))
    if actions and actions[-1] != "accept":
        actions.append("stop_best")
    return {"actions": actions, "rounds": rounds, "final_sql": best["sql"], "final_k": best["k"], "final_score": best["critic"]["score"],
            "final_res": _score_sql(item, best["sql"], gold_rows)}


def cmd_run(a):
    import yaml
    from nl2sql import config as C
    from nl2sql import db, paths
    from nl2sql.llm import LLMClient
    from nl2sql.systems.ours import m1 as M1
    from nl2sql.systems.ours import m3 as M3
    from nl2sql.systems.ours import m5 as M5
    from nl2sql.systems.ours import pipeline as PL
    from nl2sql.rl import features as F
    from concurrent.futures import ThreadPoolExecutor

    body = json.load(open(a.items))
    items = body["items"]
    if a.limit:
        items = items[: a.limit]
    os.makedirs(a.out, exist_ok=True)
    cfg_path = os.path.join(str(paths.REPO), "exp", "configs", "frozen", "full_fixed.yaml")
    cfg = PL.resolved_cfg(C.load_config(cfg_path))
    assert cfg["m5"]["policy"] == "p0" and cfg["m5"]["variant"] == "v3" and list(cfg["m5"]["weights"]) == [0.3, 0.5, 0.2]
    model = C.resolve_model("qwen37plus")
    llm = LLMClient(cache_path=str(paths.CACHE / "llm"), usage_path=os.path.join(a.out, "usage.jsonl"), key_file=paths.KEY_FILE, max_concurrency_per_model=a.conc)
    pipe = PL.Pipeline(llm, model, cfg)
    pool = {i.qid: i for i in pool_items()}
    theta_acc, theta_fb, t_max = cfg["m5"]["theta_acc"], cfg["m5"]["theta_fb"], cfg["m5"]["t_max"]
    gated_cfg = PL.resolved_cfg(C.load_config(os.path.join(str(paths.REPO), "exp", "configs", "frozen", "full_gated.yaml")))["m5"]
    rec_path = os.path.join(a.out, "records.jsonl")
    done = set()
    if os.path.exists(rec_path):
        for line in open(rec_path):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    lock = threading.Lock()
    fout = open(rec_path, "a")
    seed = 0

    def one(x):
        if x["id"] in done:
            return None
        it = pool[x["qid"]]
        assert it.gold_sql == x["gold_sql"], x["id"]
        t0 = time.time()
        gres = db.execute(it.db_path, it.gold_sql, timeout_s=30, max_rows=None)
        gold_rows = gres["rows"]
        graph, m1_ok = M1.intent(llm, model, it, seed, cfg) if cfg["m1"]["enabled"] else (None, None)
        pipe.res.ensure_terms(it)
        corpus = pipe.res.corpus_for(it)
        ctx = M3.retrieve(pipe.res, it, graph, cfg, corpus, exclude_ids=())
        store = pipe.res.db_store(it)
        qf0 = F.question_features(it, graph, len(ctx.tables), sum(len(store.schema["tables"][t]) for t in ctx.tables if t in store.schema["tables"]))
        rec = {"id": x["id"], "cat": x["cat"], "qid": x["qid"], "db_id": x["db_id"], "m1_ok": m1_ok, "exemplar_ids": list(ctx.exemplar_ids), "variants": {}}
        todo = [("pert", x["pert_sql"])] + ([("gold", x["gold_sql"])] if x["control"] else [])
        for kind, sql in todo:
            node = pipe._evaluate(it, sql, ctx, graph, seed, 0, qf=qf0, action="root")
            node.update({"usage": {}, "action": "root", "conf": None, "n_rr": 0, "k": 0, "calls_used": 0, "m1_ok": m1_ok, "qf": qf0})
            crit = node["critic"]
            v = {"sql": sql, "critic": _crit_brief(crit), "triggered": bool(crit["score"] < theta_acc), "must_fix": bool(crit["score"] < theta_fb),
                 "abstain_theta_h": bool(crit["score"] < cfg["m5"]["theta_h"]),
                 "gated_would_revise": bool(M5.gate_node(node, gated_cfg.get("gate_empty", False), gated_cfg.get("gate_zero", False))),
                 "root_res": _score_sql(it, sql, gold_rows)}
            v.update(eval_variant(pipe, cfg, it, node, gold_rows, theta_acc, t_max, seed))
            rec["variants"][kind] = v
        rec["secs"] = round(time.time() - t0, 1)
        with lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
        return rec["id"]

    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=a.conc) as ex:
        for r in ex.map(one, items):
            if r:
                n_done += 1
                if n_done % 10 == 0:
                    print("[diag] %d/%d %.0fs stats=%s" % (n_done, len(items) - len(done), time.time() - t0, json.dumps(llm.stats)), flush=True)
    print("[diag] done %d records in %.0fs stats=%s" % (n_done, time.time() - t0, json.dumps(llm.stats)), flush=True)


# ======================================================================================================== analyze
def _auroc(pos_scores, neg_scores):
    """P(score_pos < score_neg) + 0.5 P(=): AUROC of 'lower critic score = injected error' (tie-aware)."""
    from analysis_pipeline import stats as S
    import numpy as np
    s = np.concatenate([np.asarray(pos_scores, float), np.asarray(neg_scores, float)])
    y = np.concatenate([np.ones(len(pos_scores), int), np.zeros(len(neg_scores), int)])
    return S.auroc(-s, y)


def _boot_rate(vals, dbs, B, seed):
    from analysis_pipeline import stats as S
    import numpy as np
    if len(vals) == 0:
        return {"n": 0, "rate": None, "lo": None, "hi": None}
    cb = S.ClusterBoot(dbs, B=B, seed=seed)
    lo, hi = S.pctl_ci(cb.means(np.asarray(vals, float)))
    return {"n": int(len(vals)), "k": int(sum(vals)), "rate": float(np.mean(vals)), "lo": lo, "hi": hi, "n_dbs": cb.k}


def _boot_auroc(pert, ctrl, B, seed):
    """pert, ctrl: lists of (db, score).  Cluster bootstrap over databases (shared resample of both groups)."""
    import numpy as np
    from analysis_pipeline import stats as S
    if not pert or not ctrl:
        return {"auroc": None, "lo": None, "hi": None}
    dbs = sorted(set(d for d, _ in pert) | set(d for d, _ in ctrl))
    pi = {d: [s for dd, s in pert if dd == d] for d in dbs}
    ci = {d: [s for dd, s in ctrl if dd == d] for d in dbs}
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(B):
        pick = rng.randint(0, len(dbs), size=len(dbs))
        p = [s for j in pick for s in pi[dbs[j]]]
        c = [s for j in pick for s in ci[dbs[j]]]
        if p and c:
            vals.append(_auroc(p, c))
    lo, hi = S.pctl_ci(vals)
    return {"auroc": _auroc([s for _, s in pert], [s for _, s in ctrl]), "lo": lo, "hi": hi, "n_pert": len(pert), "n_ctrl": len(ctrl)}


def cmd_analyze(a):
    from analysis_pipeline import stats as S
    body = json.load(open(a.items))
    recs = [json.loads(l) for l in open(a.run) if l.strip()]
    by_id = {x["id"]: x for x in body["items"]}
    B = a.B
    seed = S.SEED_BOOT
    cats = [c for c in CATEGORIES if any(r["cat"] == c for r in recs)]

    def rows(cat=None, kind="pert"):
        return [r for r in recs if (cat is None or r["cat"] == cat) and kind in r["variants"]]

    def metric(rs, kind, fn):
        vals = [fn(r["variants"][kind]) for r in rs]
        return _boot_rate([int(v) for v in vals], [r["db_id"] for r in rs], B, seed)

    def rep_fn(kind_cmp):
        # repair success = the answer equals gold under the category comparator (set; bag for distinct_drop)
        return kind_cmp

    out = {"_meta": {"scope": SCOPE, "block": "controlled critic diagnostic (SAP Amendment 4, section 16)", "n_records": len(recs), "B": B,
                     "ci": "95%% percentile, database-clustered bootstrap (seed %d)" % seed,
                     "reading": "Critic sensitivity to injected obvious errors; NOT evidence of EX improvement on BIRD; injected errors are more detectable than natural ones.",
                     "decision_rule": "detected = Critic Score < theta_acc (8.5) = the P0 revision trigger of the pipeline; must_fix = score < theta_fb (7.5)"},
           "per_category": {}, "pooled": {}}

    def cmp_key(r, v):
        return "ex_bag" if r["cat"] == "distinct_drop" else "ex_set"

    for cat in cats + [None]:
        rp = rows(cat, "pert")
        rg = [r for r in rows(cat, "gold")]
        d = {}
        d["n_pert"], d["n_gold"] = len(rp), len(rg)
        d["n_dbs_pert"] = len(set(r["db_id"] for r in rp))
        d["detect_triggered"] = metric(rp, "pert", lambda v: v["triggered"])
        d["detect_must_fix"] = metric(rp, "pert", lambda v: v["must_fix"])
        d["detect_issues_listed"] = metric(rp, "pert", lambda v: len(v["critic"].get("issues") or []) > 0)
        d["detect_semantic_le5"] = metric(rp, "pert", lambda v: (v["critic"].get("semantic") or 0) <= 5)
        d["gated_revises"] = metric(rp, "pert", lambda v: v["gated_would_revise"])
        d["fp_triggered_gold"] = metric(rg, "gold", lambda v: v["triggered"]) if rg else None
        d["fp_must_fix_gold"] = metric(rg, "gold", lambda v: v["must_fix"]) if rg else None
        d["mean_score_pert"] = float(sum(r["variants"]["pert"]["critic"]["score"] for r in rp) / max(1, len(rp)))
        d["mean_score_gold"] = float(sum(r["variants"]["gold"]["critic"]["score"] for r in rg) / max(1, len(rg))) if rg else None
        d["auroc_score_vs_gold_all_controls"] = None
        # repair: one revision round (first revision, only where a revision was made), pipeline final
        ok_root = [int(r["variants"]["pert"]["root_res"][cmp_key(r, None)]) for r in rp]
        d["root_match_gold"] = _boot_rate(ok_root, [r["db_id"] for r in rp], B, seed)
        d["repair_round1_all"] = _boot_rate([int(bool(r["variants"]["pert"]["rounds"]) and r["variants"]["pert"]["rounds"][0]["score_res"][cmp_key(r, None)]) for r in rp],
                                            [r["db_id"] for r in rp], B, seed)
        trig = [r for r in rp if r["variants"]["pert"]["triggered"]]
        d["repair_round1_given_triggered"] = _boot_rate([int(r["variants"]["pert"]["rounds"][0]["score_res"][cmp_key(r, None)]) for r in trig if r["variants"]["pert"]["rounds"]],
                                                         [r["db_id"] for r in trig if r["variants"]["pert"]["rounds"]], B, seed)
        d["repair_final_all"] = metric(rp, "pert", lambda v: 0)  # placeholder replaced below
        d["repair_final_all"] = _boot_rate([int(r["variants"]["pert"]["final_res"][cmp_key(r, None)]) for r in rp], [r["db_id"] for r in rp], B, seed)
        d["repair_final_given_triggered"] = _boot_rate([int(r["variants"]["pert"]["final_res"][cmp_key(r, None)]) for r in trig], [r["db_id"] for r in trig], B, seed)
        d["official_ex_set_final_all"] = _boot_rate([int(r["variants"]["pert"]["final_res"]["ex_set"]) for r in rp], [r["db_id"] for r in rp], B, seed)
        d["revision_changed_sql_given_triggered"] = _boot_rate([int(any(not x["noop"] for x in r["variants"]["pert"]["rounds"])) for r in trig], [r["db_id"] for r in trig], B, seed)
        d["gated_repair_all"] = {"note": "gated policy revises only on execution error/timeout; every injected SQL executes, so the gated final answer equals the injected SQL", "rate": d["root_match_gold"]["rate"]}
        # gold controls: harm of the P0 loop on correct SQL
        if rg:
            d["gold_final_match_gold"] = _boot_rate([int(r["variants"]["gold"]["final_res"]["ex_bag"]) for r in rg], [r["db_id"] for r in rg], B, seed)
            d["gold_final_ex_set"] = _boot_rate([int(r["variants"]["gold"]["final_res"]["ex_set"]) for r in rg], [r["db_id"] for r in rg], B, seed)
        # paired AUROC on the questions that have both variants
        both = [r for r in recs if (cat is None or r["cat"] == cat) and "gold" in r["variants"]]
        if both:
            d["auroc_paired_controls"] = _boot_auroc([(r["db_id"], r["variants"]["pert"]["critic"]["score"]) for r in both],
                                                     [(r["db_id"], r["variants"]["gold"]["critic"]["score"]) for r in both], min(B, 2000), seed)
            d["auroc_semantic_paired"] = _boot_auroc([(r["db_id"], r["variants"]["pert"]["critic"]["semantic"] or 0) for r in both],
                                                     [(r["db_id"], r["variants"]["gold"]["critic"]["semantic"] or 0) for r in both], min(B, 2000), seed)
        if cat is None:
            allg = [r for r in recs if "gold" in r["variants"]]
            d["auroc_all_pert_vs_all_gold"] = _boot_auroc([(r["db_id"], r["variants"]["pert"]["critic"]["score"]) for r in recs],
                                                          [(r["db_id"], r["variants"]["gold"]["critic"]["score"]) for r in allg], min(B, 2000), seed)
        out["pooled" if cat is None else "per_category"][cat or "all"] = d
    # descriptive strata inside categories (not pre-registered as tests; sizes are small, read as listings)
    strata = {}
    def stratum(name, cat, keyfn):
        groups = collections.defaultdict(list)
        for r in rows(cat, "pert"):
            groups[str(keyfn(by_id[r["id"]]["edit"]))].append(r)
        strata[name] = {}
        for g, rs in sorted(groups.items()):
            strata[name][g] = {"n": len(rs), "detect_triggered": sum(int(r["variants"]["pert"]["triggered"]) for r in rs),
                               "repair_final": sum(int(r["variants"]["pert"]["final_res"]["ex_bag" if cat == "distinct_drop" else "ex_set"]) for r in rs)}
    stratum("col_swap_by_clause", "col_swap", lambda e: e.get("loc"))
    stratum("where_drop_by_n_conjuncts", "where_drop", lambda e: "1" if e.get("n_conjuncts") == 1 else ">=2")
    stratum("count_distinct_by_direction", "count_distinct", lambda e: e.get("from"))
    out["strata_descriptive"] = strata
    # small figure-ready json
    fig = {"_meta": dict(out["_meta"], kind="figure data"), "categories": cats,
           "detect": {c: out["per_category"][c]["detect_triggered"] for c in cats},
           "repair_round1_all": {c: out["per_category"][c]["repair_round1_all"] for c in cats},
           "repair_final_all": {c: out["per_category"][c]["repair_final_all"] for c in cats},
           "fp_gold_pooled": out["pooled"]["all"]["fp_triggered_gold"], "auroc_pooled": out["pooled"]["all"].get("auroc_all_pert_vs_all_gold"),
           "score_pert": {c: sorted(r["variants"]["pert"]["critic"]["score"] for r in rows(c, "pert")) for c in cats},
           "score_gold": sorted(r["variants"]["gold"]["critic"]["score"] for r in recs if "gold" in r["variants"])}
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "critic_diag_summary.json"), "w"), indent=1, ensure_ascii=False)
    json.dump(fig, open(os.path.join(a.out, "critic_diag_fig.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(a.out, "critic_diag_items_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "cat", "qid", "db_id", "variant", "score", "syntax", "semantic", "efficiency", "triggered", "must_fix", "n_issues", "root_match", "rounds",
                    "round1_match", "final_match", "final_changed", "gated_would_revise"])
        for r in recs:
            for kind, v in r["variants"].items():
                ck = "ex_bag" if r["cat"] == "distinct_drop" else "ex_set"
                w.writerow([r["id"], r["cat"], r["qid"], r["db_id"], kind, v["critic"]["score"], v["critic"]["syntax"], v["critic"]["semantic"], v["critic"]["efficiency"],
                            int(v["triggered"]), int(v["must_fix"]), len(v["critic"].get("issues") or []), v["root_res"][ck], len(v["rounds"]),
                            (v["rounds"][0]["score_res"][ck] if v["rounds"] else ""), v["final_res"][ck], int(v["final_sql"].strip() != v["sql"].strip()),
                            int(v["gated_would_revise"])])
    cases = build_cases(recs, by_id)
    json.dump(cases, open(os.path.join(a.out, "critic_diag_cases.json"), "w"), indent=1, ensure_ascii=False)
    print("wrote summary/fig/csv/cases to", a.out)


def build_cases(recs, by_id, per_class=2):
    """Illustrative real cases by a FIXED rule (not by size of effect): four outcome classes, in each the `per_class` lowest-id records of different
    categories.  A = injected, triggered, pipeline final matches gold; B = injected, triggered, final does not match; C = injected, NOT triggered
    (missed); D = gold control flagged (false positive)."""
    def ck(r):
        return "ex_bag" if r["cat"] == "distinct_drop" else "ex_set"

    classes = [("A_detected_and_repaired", lambda r: "pert" in r["variants"] and r["variants"]["pert"]["triggered"] and r["variants"]["pert"]["final_res"][ck(r)] == 1, "pert"),
               ("B_detected_not_repaired", lambda r: "pert" in r["variants"] and r["variants"]["pert"]["triggered"] and r["variants"]["pert"]["final_res"][ck(r)] == 0, "pert"),
               ("C_missed", lambda r: "pert" in r["variants"] and not r["variants"]["pert"]["triggered"], "pert"),
               ("D_gold_false_positive", lambda r: "gold" in r["variants"] and r["variants"]["gold"]["triggered"], "gold")]
    out = {"_meta": {"scope": SCOPE, "selection": "fixed rule: per outcome class the %d lowest-id records of different categories; illustrative, not representative; gold_control_* shows whether the unperturbed gold of the same question was flagged too (BIRD gold is not always right)" % per_class},
           "cases": []}
    for name, pred, kind in classes:
        seen = set()
        for r in sorted(recs, key=lambda r: r["id"]):
            if len(seen) >= per_class:
                break
            if r["cat"] in seen or not pred(r):
                continue
            seen.add(r["cat"])
            x = by_id.get(r["id"], {})
            v = r["variants"][kind]
            out["cases"].append({"class": name, "id": r["id"], "category": r["cat"], "source_qid": r["qid"], "db_id": r["db_id"], "question": x.get("question"),
                                 "evidence": x.get("evidence"), "gold_sql": x.get("gold_sql"), "injected_sql": x.get("pert_sql"), "edit": x.get("edit"),
                                 "evaluated_variant": kind, "evaluated_sql": v["sql"],
                                 "critic": {k: v["critic"].get(k) for k in ("score", "syntax", "semantic", "efficiency", "issues", "suggestions")},
                                 "triggered": v["triggered"], "actions": v["actions"],
                                 "gold_control_triggered": (r["variants"]["gold"]["triggered"] if "gold" in r["variants"] else None),
                                 "gold_control_score": (r["variants"]["gold"]["critic"]["score"] if "gold" in r["variants"] else None),
                                 "revisions": [{"sql": q["sql"], "same_as_previous": q["noop"], "critic_score": (q["critic"] or {}).get("score"),
                                                "matches_gold": q["score_res"][ck(r)]} for q in v["rounds"]],
                                 "final_sql": v["final_sql"], "final_matches_gold": v["final_res"][ck(r)]})
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--n", type=int, default=0)
    b.add_argument("--burnin-log", default="")
    b.add_argument("--seed", type=int, default=0, help="Amendment 6 tuning draw; 0 = the frozen Amendment 4 seed")
    b.add_argument("--ctrl", type=int, default=0, help="gold controls per category; 0 = 22")
    b.add_argument("--exclude-items", default="", help="items.json whose qids are excluded from the draw (Amendment 6)")
    b.add_argument("--id-prefix", default="", help="item id prefix; default D")
    r = sub.add_parser("run")
    r.add_argument("--items", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--conc", type=int, default=4)
    r.add_argument("--limit", type=int, default=0)
    z = sub.add_parser("analyze")
    z.add_argument("--items", required=True)
    z.add_argument("--run", required=True)
    z.add_argument("--out", required=True)
    z.add_argument("--B", type=int, default=5000)
    a = ap.parse_args()
    {"build": cmd_build, "run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
