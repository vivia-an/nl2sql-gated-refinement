"""E7 AmbiQT mix: 500 ambiguous AmbiQT validation items stratified over the ambiguity types that can be executed + 500 unambiguous
Spider-dev questions (seed 20261027). AmbiQT ships rewritten *schemas* but only the original Spider databases, so for the types
C (col-synonyms), T (tbl-synonyms), J (tbl-split) we materialise a per-item sqlite database with the rewritten schema
(columns/tables copied from the original) so that both valid SQLs (query1/query2) execute. Type P (tbl-agg: precomputed
aggregates) needs query-specific aggregation tables and is NOT built (declared; listed in the manifest).
Output: exp/splits/ambiqt_mix.json (items) + derived dbs under NL2SQL_CACHE/ambiqt_dbs/."""
import json
import os
import random
import re
import shutil
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code"))
from nl2sql import data, paths  # noqa: E402

SEED = 20261027


def q(n):
    return '"%s"' % n.replace('"', '""')


def build_db(src, dst, typ, rec):
    if os.path.exists(dst):
        return True
    shutil.copyfile(src, dst)
    c = sqlite3.connect(dst)
    try:
        if typ == "C":
            for t, cols in rec["extra_map"].items():
                info = c.execute("PRAGMA table_info(%s)" % q(t)).fetchall()
                names = [r[1] for r in info]
                sel, defs = [], []
                for r in info:
                    if r[1] in cols:
                        for syn in cols[r[1]]:
                            sel.append("%s AS %s" % (q(r[1]), q(syn)))
                    else:
                        sel.append(q(r[1]))
                c.execute("CREATE TABLE %s AS SELECT %s FROM %s" % (q(t + "__new"), ", ".join(sel), q(t)))
                c.execute("DROP TABLE %s" % q(t))
                c.execute("ALTER TABLE %s RENAME TO %s" % (q(t + "__new"), q(t)))
        elif typ == "T":
            for t, news in rec["extra_table_map"].items():
                for n in news:
                    c.execute("CREATE TABLE %s AS SELECT * FROM %s" % (q(n), q(t)))
                c.execute("DROP TABLE %s" % q(t))
        elif typ == "J":
            for t, col in rec["split_map"].items():
                pk = rec["primary_key"][t]
                newt = t + "_" + col
                c.execute("CREATE TABLE %s AS SELECT %s, %s FROM %s" % (q(newt), q(pk), q(col), q(t)))
                info = c.execute("PRAGMA table_info(%s)" % q(t)).fetchall()
                keep = [q(r[1]) for r in info if r[1] != col]
                c.execute("CREATE TABLE %s AS SELECT %s FROM %s" % (q(t + "__new"), ", ".join(keep), q(t)))
                c.execute("DROP TABLE %s" % q(t))
                c.execute("ALTER TABLE %s RENAME TO %s" % (q(t + "__new"), q(t)))
        c.commit()
        # both gold queries must execute
        for k in ("query1", "query2"):
            c.execute(rec[k]).fetchmany(1)
        return True
    except Exception as e:
        c.close()
        os.remove(dst)
        return False
    finally:
        try:
            c.close()
        except Exception:
            pass


def main():
    rnd = random.Random(SEED)
    base = paths.DATA / "repos" / "AmbiQT" / "benchmark"
    outdir = paths.CACHE / "ambiqt_dbs"
    outdir.mkdir(parents=True, exist_ok=True)
    types = {"C": "col-synonyms", "T": "tbl-synonyms", "J": "tbl-split"}
    per = {"C": 167, "T": 167, "J": 166}
    items = []
    stats = {}
    for typ, dname in types.items():
        rows = json.load(open(base / dname / "validation.json"))
        idx = list(range(len(rows)))
        rnd.shuffle(idx)
        got = 0
        for i in idx:
            r = rows[i]
            src = data.db_path("ambiqt", r["db_id"])
            dst = str(outdir / ("%s_%d.sqlite" % (typ, i)))
            if build_db(src, dst, typ, r):
                items.append({"qid": "ambiqt_mix:%s:%s-%d" % (r["db_id"], typ, i), "db_id": r["db_id"], "question": r["question"], "evidence": "",
                              "gold_sql": r["query1"], "gold_sql_2": r["query2"], "db_path": dst, "ambiguous": 1, "amb_type": typ,
                              "ambiguous_schema": r.get("schema_without_content", "")})
                got += 1
                if got == per[typ]:
                    break
        stats[typ] = got
    sp = data.load_split("spider_dev", with_gold=True)
    for it in rnd.sample(sp, 500):
        items.append({"qid": it.qid.replace("spider_dev", "ambiqt_mix"), "db_id": it.db_id, "question": it.question, "evidence": "",
                      "gold_sql": it.gold_sql, "gold_sql_2": "", "db_path": it.db_path, "ambiguous": 0, "amb_type": None, "ambiguous_schema": ""})
    json.dump({"seed": SEED, "n": len(items), "ambiguous_by_type": stats, "type_P_tbl_agg": "not built (needs query-specific aggregation tables)",
               "items": items}, open(HERE / "ambiqt_mix.json", "w"), indent=0)
    print(len(items), stats)


if __name__ == "__main__":
    main()
