"""E8 linking recall: share of the gold SQL's tables / columns that are present in the context shown to the generator.
Context = the M3 output of the root generation (preds.jsonl: ctx_tables, ctx_full_schema, ctx_tokens). The DDL shows every column of an included
table, so a column is 'present' iff at least one table that owns it (by schema lookup) is in the context. A full-schema fallback counts as recall 1
(report ctx_tokens next to it). Zero-shot rows (ctx_tables None) are full-schema by construction."""
from __future__ import annotations

from .. import data


def gold_refs(sql: str, schema_tables: dict):
    """-> (set of gold table names, list of (column, frozenset of candidate tables)). Lower-cased. Parse failure -> (None, None)."""
    try:
        import sqlglot
        from sqlglot import exp
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None, None
    alias = {}
    tabs = set()
    for t in tree.find_all(exp.Table):
        if not t.name:
            continue
        tabs.add(t.name.lower())
        alias[(t.alias or t.name).lower()] = t.name.lower()
    own = {}
    for t, cols in schema_tables.items():
        for c in cols:
            own.setdefault(c["name"].lower(), set()).add(t.lower())
    cols = []
    for c in tree.find_all(exp.Column):
        if not c.name or c.name == "*":
            continue
        q = (c.table or "").lower()
        if q and q in alias:
            cand = {alias[q]}
        else:
            cand = (own.get(c.name.lower(), set()) & tabs) or own.get(c.name.lower(), set())
        if cand:
            cols.append((c.name.lower(), frozenset(cand)))
    return tabs, cols


def linking_recall(gold_sql: str, schema_tables: dict, ctx_tables, full_schema: bool):
    """-> dict(table_recall, column_recall) or None when the gold cannot be parsed. Empty reference sets give recall 1.0."""
    tabs, cols = gold_refs(gold_sql, schema_tables)
    if tabs is None:
        return None
    if full_schema or ctx_tables is None:
        return {"table_recall": 1.0, "column_recall": 1.0}
    have = {t.lower() for t in ctx_tables}
    tr = sum(1 for t in tabs if t in have) / len(tabs) if tabs else 1.0
    cr = sum(1 for _, cand in cols if cand & have) / len(cols) if cols else 1.0
    return {"table_recall": tr, "column_recall": cr}


def summarize_linking(preds, items_by_qid: dict, schema_of):
    """preds: iterable of preds.jsonl rows; items_by_qid with gold; schema_of(item) -> schema['tables'] dict. -> mean recalls + ctx tokens."""
    tr, cr, tok, n = 0.0, 0.0, 0.0, 0
    for r in preds:
        it = items_by_qid.get(r["qid"])
        if it is None or not it.gold_sql:
            continue
        lr = linking_recall(it.gold_sql, schema_of(it), r.get("ctx_tables"), bool(r.get("ctx_full_schema")))
        if lr is None:
            continue
        tr += lr["table_recall"]; cr += lr["column_recall"]; tok += r.get("ctx_tokens") or 0; n += 1
    return {"n": n, "table_recall": tr / n if n else None, "column_recall": cr / n if n else None, "ctx_tokens_mean": tok / n if n else None}
