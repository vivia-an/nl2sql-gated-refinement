"""Cross-check eval/spider_ts.py (worker pool around the official exec_eval) against the OFFICIAL evaluation.py CLI
(`--etype exec`, no --plug_value, DISTINCT removed) on Spider dev:
  (1) gold-vs-gold: our ex_ts must be 1 for every item (and equal to the official accuracy, 1.000);
  (2) N synthetic "smoke" predictions made by seeded mutations of the gold SQL (no LLM, no dev tuning involved): kinds
      same / wrap_limit1 / drop_where / toggle_distinct / flip_order / loosen_cmp / col_swap / other_gold / broken.
      Compared in 4 chunks: count of exec-correct per chunk, ours vs the official CLI; also plain EX (db.execute + set equality on the
      original Spider db) to show where the test suite differs from plain EX.
Usage: python tests/crosscheck_spider_ts.py out.json [N=200] [GOLD_LIMIT=1034] [WORKERS=4]"""
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nl2sql import data, db, paths  # noqa: E402
from nl2sql.eval import spider_ts  # noqa: E402

TS = paths.DATA / "eval_tools" / "test-suite-sql-eval"
DB_DIR = paths.DATA / "spider_testsuite_dbs" / "database"


def flat(s):
    return " ".join(s.split())


def mutate(g, kind, rnd, others):
    if kind == "same":
        return g
    if kind == "wrap_limit1":
        return "SELECT * FROM (%s) LIMIT 1" % g.rstrip(";")
    if kind == "drop_where":
        m = re.search(r"\bWHERE\b.*?(?=\bGROUP BY\b|\bORDER BY\b|\bLIMIT\b|\bINTERSECT\b|\bUNION\b|\bEXCEPT\b|\)|$)", g, re.I | re.S)
        return (g[:m.start()] + g[m.end():]) if m else g + " LIMIT 1"
    if kind == "toggle_distinct":
        return re.sub(r"(?i)\bSELECT\s+DISTINCT\b", "SELECT", g, 1) if re.search(r"(?i)\bSELECT\s+DISTINCT\b", g) else re.sub(r"(?i)\bSELECT\b", "SELECT DISTINCT", g, 1)
    if kind == "flip_order":
        if re.search(r"(?i)\bDESC\b", g):
            return re.sub(r"(?i)\bDESC\b", "ASC", g)
        return re.sub(r"(?i)\bASC\b", "DESC", g) if re.search(r"(?i)\bASC\b", g) else g + " ORDER BY 1 DESC"
    if kind == "loosen_cmp":
        for a, b in ((" > ", " >= "), (" < ", " <= "), (" = ", " >= ")):
            if a in g:
                return g.replace(a, b, 1)
        return g
    if kind == "col_swap":
        m = re.match(r"(?is)^(\s*SELECT\s+(?:DISTINCT\s+)?)(.*?)(\s+FROM\s+.*)$", g)
        if m and "(" not in m.group(2) and m.group(2).count(",") >= 1:
            cols = [c.strip() for c in m.group(2).split(",")]
            cols[0], cols[1] = cols[1], cols[0]
            return m.group(1) + ", ".join(cols) + m.group(3)
        return g
    if kind == "other_gold":
        return rnd.choice(others)
    if kind == "broken":
        return g.replace("SELECT", "SELEC", 1)
    raise ValueError(kind)


def official(gold_lines, pred_lines, tag):
    d = tempfile.mkdtemp(prefix="tscheck_")
    gp, pp = os.path.join(d, "g.txt"), os.path.join(d, "p.txt")
    open(gp, "w").write("\n".join(gold_lines) + "\n")
    open(pp, "w").write("\n".join(pred_lines) + "\n")
    env = dict(os.environ, NLTK_DATA=str(paths.ROOT / "nltk_data"))
    r = subprocess.run([sys.executable, "evaluation.py", "--gold", gp, "--pred", pp, "--db", str(DB_DIR), "--table", str(TS / "tables.json"),
                        "--etype", "exec"], cwd=str(TS), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out = r.stdout
    cnt = exe = None
    for line in out.splitlines():
        t = line.split()
        if t and t[0] == "count" and cnt is None:
            cnt = int(t[-1])
        if t and t[0] == "execution":
            exe = float(t[-1])
    if cnt is None or exe is None:
        raise RuntimeError("cannot parse official output (%s): %s %s" % (tag, out[-300:], r.stderr[-300:]))
    return cnt, int(round(exe * cnt))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else None
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    GL = int(sys.argv[3]) if len(sys.argv) > 3 else 1034
    W = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    items = data.load_split("spider_dev", with_gold=True)
    by_db = {}
    for it in items:
        by_db.setdefault(it.db_id, []).append(it.gold_sql)
    rep = {"workers": W, "t0": time.strftime("%F %T")}
    pool = spider_ts.TSPool(W, timeout_s=600)
    # (1) gold vs gold
    import threading
    gl = items[:GL]
    res = [None] * len(gl)

    def work(ix):
        for i in ix:
            res[i] = pool.match(gl[i].db_id, gl[i].gold_sql, gl[i].gold_sql)
    ths = [threading.Thread(target=work, args=(range(k, len(gl), W),)) for k in range(W)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    ours1 = sum(1 for r in res if r["ex_ts"] == 1)
    rep["gold_vs_gold"] = {"n": len(gl), "ours_ex_ts_1": ours1, "ours_errors": [(gl[i].qid, r["error"]) for i, r in enumerate(res) if r["error"]][:5]}
    cnt, ok = official([flat(i.gold_sql) + "\t" + i.db_id for i in gl], [flat(i.gold_sql) for i in gl], "gold")
    rep["gold_vs_gold"].update({"official_n": cnt, "official_exec_correct": ok, "agree": bool(ok == ours1 and cnt == len(gl))})
    # (2) mutated predictions
    rnd = random.Random(20261027)
    kinds = ["same", "wrap_limit1", "drop_where", "toggle_distinct", "flip_order", "loosen_cmp", "col_swap", "other_gold", "broken"]
    pick = rnd.sample(items, N)
    rows = []
    for k, it in enumerate(pick):
        kind = kinds[k % len(kinds)]
        p = mutate(it.gold_sql, kind, rnd, [g for g in by_db[it.db_id] if g != it.gold_sql] or [it.gold_sql])
        rows.append({"qid": it.qid, "db_id": it.db_id, "kind": kind, "gold": it.gold_sql, "pred": flat(p), "db_path": it.db_path})
    res2 = [None] * len(rows)

    def work2(ix):
        for i in ix:
            res2[i] = pool.match(rows[i]["db_id"], rows[i]["pred"], rows[i]["gold"])
    ths = [threading.Thread(target=work2, args=(range(k, len(rows), W),)) for k in range(W)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    for r, m in zip(rows, res2):
        r["ex_ts"], r["ts_error"] = m["ex_ts"], m["error"]
        g, p = db.execute(r["db_path"], r["gold"], 30, None), db.execute(r["db_path"], r["pred"], 30, None)
        r["ex_plain"] = db.ex_match(p["rows"], g["rows"]) if (g["ok"] and p["ok"]) else 0
    chunks = []
    for c in range(4):
        sl = rows[c * N // 4:(c + 1) * N // 4]
        cnt, ok = official([flat(r["gold"]) + "\t" + r["db_id"] for r in sl], [r["pred"] for r in sl], "chunk%d" % c)
        chunks.append({"n": len(sl), "official": ok, "ours": sum(1 for r in sl if r["ex_ts"] == 1), "official_n": cnt})
    rep["mutations"] = {"n": N, "chunks": chunks, "agree_all_chunks": all(c["official"] == c["ours"] and c["official_n"] == c["n"] for c in chunks),
                        "ours_ex_ts": sum(1 for r in rows if r["ex_ts"] == 1), "plain_ex": sum(r["ex_plain"] for r in rows),
                        "ts_errors": sum(1 for r in rows if r["ts_error"])}
    kinds_tab = {}
    for r in rows:
        d = kinds_tab.setdefault(r["kind"], {"n": 0, "plain_ex": 0, "ex_ts": 0})
        d["n"] += 1
        d["plain_ex"] += r["ex_plain"]
        d["ex_ts"] += int(r["ex_ts"] == 1)
    rep["mutations"]["by_kind"] = kinds_tab
    rep["mutations"]["plain_vs_ts_disagree"] = sum(1 for r in rows if r["ex_plain"] != int(r["ex_ts"] == 1))
    rep["rows"] = [{k: v for k, v in r.items() if k not in ("db_path",)} for r in rows]
    rep["t1"] = time.strftime("%F %T")
    pool.close()
    if out:
        json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1))
    assert rep["gold_vs_gold"]["agree"] and rep["mutations"]["agree_all_chunks"]


if __name__ == "__main__":
    main()
