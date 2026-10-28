"""E4 pollution injection hook: replace a fraction p of seed-set SQLs by a *wrong executable* SQL whose result differs
from gold. 50 % natural (an actual failure of zs on that question), 50 % programmatic sqlglot mutation of gold
(comparison flip, aggregate swap, drop one WHERE conjunct, same-type sibling column, join key change). Every mutation
is verified to execute and to differ from gold in result (<=10 tries, else natural failure, else skip + log)."""
import random

from ... import db


def _mutations(sql, schema, rnd):
    import sqlglot
    from sqlglot import exp
    try:
        base = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return
    kinds = ["cmp", "agg", "drop_where", "sibling_col", "join_key"]
    rnd.shuffle(kinds)
    for kind in kinds:
        t = base.copy()
        ok = False
        if kind == "cmp":
            nodes = [n for n in t.find_all((exp.GT, exp.GTE, exp.LT, exp.LTE, exp.EQ, exp.NEQ))]
            if nodes:
                n = rnd.choice(nodes)
                flip = {exp.GT: exp.LT, exp.GTE: exp.LTE, exp.LT: exp.GT, exp.LTE: exp.GTE, exp.EQ: exp.NEQ, exp.NEQ: exp.EQ}[type(n)]
                n.replace(flip(this=n.this, expression=n.expression))
                ok = True
        elif kind == "agg":
            swap = {exp.Sum: exp.Avg, exp.Avg: exp.Sum, exp.Max: exp.Min, exp.Min: exp.Max, exp.Count: exp.Sum}
            nodes = [n for n in t.find_all(tuple(swap.keys()))]
            if nodes:
                n = rnd.choice(nodes)
                n.replace(swap[type(n)](this=n.this))
                ok = True
        elif kind == "drop_where":
            w = t.args.get("where")
            if w is not None:
                conj = list(w.this.flatten()) if isinstance(w.this, exp.And) else [w.this]
                if len(conj) >= 2:
                    drop = rnd.randrange(len(conj))
                    rest = [c for i, c in enumerate(conj) if i != drop]
                    new = rest[0]
                    for c in rest[1:]:
                        new = exp.And(this=new, expression=c)
                    w.set("this", new)
                    ok = True
                elif len(conj) == 1:
                    t.set("where", None)
                    ok = True
        elif kind == "sibling_col":
            cols = [c for c in t.find_all(exp.Column) if c.name]
            if cols:
                c = rnd.choice(cols)
                tabs = schema["tables"]
                for tname, tcols in tabs.items():
                    names = {x["name"].lower(): x for x in tcols}
                    if c.name.lower() in names:
                        typ = names[c.name.lower()]["type"]
                        sib = [x["name"] for x in tcols if x["type"] == typ and x["name"].lower() != c.name.lower()]
                        if sib:
                            c.set("this", exp.to_identifier(rnd.choice(sib)))
                            ok = True
                        break
        elif kind == "join_key":
            ons = [j.args.get("on") for j in t.find_all(exp.Join) if j.args.get("on") is not None]
            if ons:
                on = rnd.choice(ons)
                eqs = list(on.find_all(exp.EQ))
                if eqs:
                    e = rnd.choice(eqs)
                    cols = list(e.find_all(exp.Column))
                    if len(cols) == 2:
                        tabs = schema["tables"]
                        alt = [x["name"] for tcols in tabs.values() for x in tcols if x["name"].lower() != cols[1].name.lower()]
                        if alt:
                            cols[1].set("this", exp.to_identifier(rnd.choice(alt)))
                            ok = True
        if ok:
            try:
                yield t.sql(dialect="sqlite")
            except Exception:
                continue


def pollute_entries(seed_items, p, rng_seed, schema, natural_failures, db_path):
    """seed_items: list of Items (gold SQL available). natural_failures: {qid: wrong_sql} (zs failures on those questions).
    Returns (polluted: {qid: sql}, log: {qid: how}); entries not in `polluted` keep their gold SQL."""
    n = int(round(p * len(seed_items)))
    rnd = random.Random(rng_seed + int(round(p * 100)))
    chosen = rnd.sample(seed_items, n) if n else []
    polluted, log = {}, {}
    for k, it in enumerate(chosen):
        want_natural = (k % 2 == 0)
        gold = db.execute(db_path, it.gold_sql, max_rows=None)
        sql, how = None, None
        order = ["natural", "mutation"] if want_natural else ["mutation", "natural"]
        for kind in order:
            if kind == "natural":
                w = natural_failures.get(it.qid)
                if w:
                    r = db.execute(db_path, w, max_rows=None)
                    if r["ok"] and gold["ok"] and not db.ex_match(r["rows"], gold["rows"]):
                        sql, how = w, "natural"
                        break
            else:
                tries = 0
                for cand in _mutations(it.gold_sql, schema, rnd):
                    tries += 1
                    if tries > 10:
                        break
                    r = db.execute(db_path, cand, max_rows=None)
                    if r["ok"] and gold["ok"] and not db.ex_match(r["rows"], gold["rows"]):
                        sql, how = cand, "mutation"
                        break
                if sql:
                    break
        if sql:
            polluted[it.qid] = sql
            log[it.qid] = how
        else:
            log[it.qid] = "skipped"
    return polluted, log
