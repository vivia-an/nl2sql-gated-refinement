"""X7 value probing (HEADROOM row 1 follow-up; config `m5.probe_values`).

When a SQL executes but returns an empty / degenerate result (no rows, or one row of NULLs), the most common cause on BIRD-like data is a
filter literal that does not match the stored value (case, format, code vs. word). The model only sees 3 sample values per column in the
schema block. This module collects, READ-ONLY and time-guarded, up to `n` distinct stored values of every short text column that the
failed SQL compares with a literal in WHERE / JOIN ... ON / HAVING, so that the revision prompt can show them.

  literal_columns(sql, schema)                 -> [{table, column, type, literals}]   (sqlglot; [] when the SQL does not parse)
  probe_column(db_path, table, column, ...)    -> {values, complete} | None            (one SELECT DISTINCT ... LIMIT n+1, db.execute)
  probe_failed_sql(db_path, sql, schema, ...)  -> [{table, column, literals, values, complete}]
  probe_text(probes, n)                        -> the text of the `# Values stored ...` prompt block

No value of the gold SQL, no other table, no write: the only statements issued are `SELECT DISTINCT "col" FROM "tbl" WHERE ... LIMIT n+1`
on identifiers that exist in the stored schema, on the read-only (`mode=ro&immutable=1`) database connection of `nl2sql.db`."""
import re

from ... import db

_TEXT_RE = re.compile(r"CHAR|TEXT|CLOB", re.I)
CMP_SIDES = ("EQ", "NEQ", "In", "Like", "ILike", "Glob")


def is_text_type(ctype):
    """Declared type of a text-like column: TEXT / VARCHAR / CHAR / CLOB, or no declared type (the rule `data._build_schema` uses for sample values)."""
    t = (ctype or "").strip()
    return t == "" or bool(_TEXT_RE.search(t))


def _qid(name):
    """Identifier for display (double quotes, as in the schema block)."""
    return '"%s"' % str(name).replace('"', '""')


def _bq(name):
    """Identifier for the probe statement: backticks, because SQLite silently turns an unknown "double-quoted" identifier into a string literal."""
    return "`%s`" % str(name).replace("`", "``")


def _scope_tables(select, exp):
    """alias(lower) / table name(lower) -> table name for the tables directly in the FROM / JOIN list of one SELECT."""
    out = {}
    items = []
    frm = select.args.get("from")
    if frm is not None:
        items.append(frm.this)
    for j in select.args.get("joins") or []:
        items.append(j.this)
    for t in items:
        if isinstance(t, exp.Table) and t.name:
            out[t.name.lower()] = t.name
            if t.alias:
                out[t.alias.lower()] = t.name
    return out


def _lit_text(lit):
    s = lit.this
    return s if isinstance(s, str) else str(s)


def literal_columns(sql, schema, max_literals=3, text_only=True):
    """Columns of `schema` that `sql` compares with a string / number literal (=, <>, IN, LIKE, GLOB) in a WHERE, JOIN ... ON or HAVING clause,
    in order of first appearance, text-like columns only. -> [{"table", "column", "type", "literals": [str, ...]}]. Never raises: an unparsable SQL
    gives []. text_only=False (analysis tools) also returns numeric / other columns. Unqualified columns are resolved against the tables of the enclosing SELECT (then outer SELECTs); an ambiguous column is attributed
    to every candidate table of its scope."""
    try:
        import sqlglot
        from sqlglot import exp
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return []
    if tree is None:
        return []
    tabs = {t.lower(): t for t in (schema.get("tables") or {})}
    cols_of = {t.lower(): {c["name"].lower(): c for c in cs} for t, cs in (schema.get("tables") or {}).items()}
    found = {}
    order = []
    sides = tuple(getattr(exp, n) for n in CMP_SIDES if hasattr(exp, n))
    for node in tree.find_all(*sides, bfs=False):          # depth-first = textual order
        if node.find_ancestor(exp.Where, exp.Join, exp.Having) is None:
            continue
        if isinstance(node, exp.In):
            colside, lits = node.this, list(node.expressions or [])
            if node.args.get("query") is not None:
                continue
        else:
            a, b = node.this, node.expression
            if isinstance(b, exp.Literal) and not isinstance(a, exp.Literal):
                colside, lits = a, [b]
            elif isinstance(a, exp.Literal) and not isinstance(b, exp.Literal):
                colside, lits = b, [a]
            else:
                continue
        lits = [l for l in lits if isinstance(l, exp.Literal)]
        if not lits:
            continue
        col = colside if isinstance(colside, exp.Column) else None
        if col is None:                         # lower(col) = 'x', CAST(col AS TEXT) = '1', col || '' ...: accept a single column inside
            inner = list(colside.find_all(exp.Column))
            if len(inner) != 1:
                continue
            col = inner[0]
        cname = (col.name or "").lower()
        if not cname:
            continue
        # resolve through the enclosing SELECT scopes
        cand = []
        sel = col.find_ancestor(exp.Select)
        while sel is not None and not cand:
            scope = _scope_tables(sel, exp)
            if col.table:
                t = scope.get(col.table.lower())
                if t and t.lower() in cols_of and cname in cols_of[t.lower()]:
                    cand = [t.lower()]
            else:
                cand = sorted({t.lower() for t in scope.values() if t.lower() in cols_of and cname in cols_of[t.lower()]})
            sel = sel.find_ancestor(exp.Select)
        for tl in cand:
            c = cols_of[tl][cname]
            if text_only and not is_text_type(c.get("type")):
                continue
            key = (tl, cname)
            if key not in found:
                found[key] = {"table": tabs[tl], "column": c["name"], "type": c.get("type") or "", "literals": []}
                order.append(key)
            for l in lits:
                t = _lit_text(l)
                if t not in found[key]["literals"] and len(found[key]["literals"]) < max_literals:
                    found[key]["literals"].append(t)
    return [found[k] for k in order]


def probe_column(db_path, table, column, n=8, max_len=40, timeout_s=5.0):
    """{"values": [str,...] (<= n, each <= max_len characters), "complete": bool (no further distinct value exists)} or None (error, timeout or
    no short non-NULL value). One read-only statement; `timeout_s` guards it (db.execute interrupts the query)."""
    q, c = _bq(table), _bq(column)
    sql = "SELECT DISTINCT %s FROM %s WHERE %s IS NOT NULL AND length(%s) <= %d LIMIT %d" % (c, q, c, c, int(max_len), int(n) + 1)
    r = db.execute(db_path, sql, timeout_s=timeout_s, max_rows=int(n) + 1)
    if not r["ok"] or not r["rows"]:
        return None
    vals = [str(x[0]) for x in r["rows"]]
    return {"values": vals[: int(n)], "complete": len(vals) <= int(n)}


_TOK = re.compile(r"[A-Za-z0-9]+")


def near_tokens(literals):
    """Longest alphanumeric token (>= 3 characters, LIKE wildcards ignored) of each literal; [] when none."""
    out = []
    for l in literals:
        toks = [t for t in _TOK.findall(str(l)) if len(t) >= 3]
        if toks:
            t = max(toks, key=len).lower()
            if t not in out:
                out.append(t)
    return out


def probe_near(db_path, table, column, literals, n=8, max_len=40, timeout_s=5.0):
    """Values of the column that contain (case-insensitively) the longest alphanumeric token of a failed literal: the stored spelling of what the
    query tried to match ('bishopville' -> 'Bishopville SC'). One read-only statement per token (tokens are [A-Za-z0-9] only). -> list of str."""
    vals = []
    for tok in near_tokens(literals):
        c, q = _bq(column), _bq(table)
        sql = "SELECT DISTINCT %s FROM %s WHERE %s IS NOT NULL AND length(%s) <= %d AND lower(%s) LIKE '%%%s%%' LIMIT %d" % (c, q, c, c, int(max_len), c, tok, int(n))
        r = db.execute(db_path, sql, timeout_s=timeout_s, max_rows=int(n))
        if r["ok"]:
            for x in r["rows"]:
                v = str(x[0])
                if v not in vals:
                    vals.append(v)
    return vals[: int(n)]


def probe_failed_sql(db_path, sql, schema, n=8, max_len=40, max_cols=4, timeout_s=5.0, near=False):
    """-> [{"table", "column", "literals", "values", "complete", "near"}] for at most `max_cols` columns of `sql` (see literal_columns).
    near=True: values resembling the failed literal(s) (probe_near) come first, then the first distinct values up to `n`; `near` = how many were near."""
    out = []
    for c in literal_columns(sql, schema):
        if len(out) >= max_cols:
            break
        p = probe_column(db_path, c["table"], c["column"], n=n, max_len=max_len, timeout_s=timeout_s)
        if not p:
            continue
        vals, k = p["values"], 0
        if near and not p["complete"]:
            nv = probe_near(db_path, c["table"], c["column"], c["literals"], n=n, max_len=max_len, timeout_s=timeout_s)
            vals = nv + [v for v in p["values"] if v not in nv]
            vals, k = vals[: int(n)], len(nv[: int(n)])
        out.append({"table": c["table"], "column": c["column"], "literals": c["literals"], "values": vals, "complete": p["complete"], "near": k})
    return out


def _sq(s):
    return "'" + str(s).replace("'", "''") + "'"


def probe_text(probes, n=8):
    """Body of the `# Values stored in the filtered columns` block (template rev_exec_probe.user): one line per column."""
    lines = []
    for p in probes:
        vals = ", ".join(_sq(v) for v in p["values"])
        tail = "" if p["complete"] else ", ..."
        lines.append("- %s.%s (the query compares it with %s%s): %s%s" % (_qid(p["table"]), _qid(p["column"]), ", ".join(_sq(l) for l in p["literals"]),
                                                                           "; the first %d values resemble it" % p["near"] if p.get("near") else "", vals, tail))
    return "\n".join(lines)


def probe_cols(probes):
    return ["%s.%s" % (p["table"], p["column"]) for p in probes]
