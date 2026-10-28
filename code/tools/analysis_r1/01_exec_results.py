"""Execute every stored candidate SQL (HR trees, HR/HT cands, HT zs/zs_ours/full_fixed finals) and the gold SQL against the
read-only BIRD-train sqlite files (NO LLM, NO API). Output out/exec_results.jsonl: one row per (db_id, sql) with result hash,
n_rows, n_cols, status. Used for execution-consistency voting and for EX recomputation (sanity check vs stored ex).
Resumable. Usage: python 01_exec_results.py [workers=3]"""
import sys, os, json, hashlib, sqlite3, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from multiprocessing import Pool

DBDIR = R + "/data/bird/train/train/train_databases/"
OUTF = os.path.join(OUT, "exec_results.jsonl")


def run_one(args):
    db_id, sql = args
    path = DBDIR + "%s/%s.sqlite" % (db_id, db_id)
    t0 = time.time()
    conn = None
    try:
        conn = sqlite3.connect("file:%s?mode=ro&immutable=1" % path, uri=True)
        dl = t0 + 30
        conn.set_progress_handler(lambda: 1 if time.time() > dl else 0, 20000)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchmany(200001)
        trunc = len(rows) > 200000
        rows = rows[:200000]
        s = set(map(tuple, rows))
        try:
            h = hashlib.sha1(("\n".join(sorted(map(repr, s)))).encode("utf8", "replace")).hexdigest()
        except Exception:
            h = "unhashable"
        empty = (not rows) or (len(rows) == 1 and all(c is None for c in rows[0]))
        ncol = len(cur.description or [])
        return {"db_id": db_id, "sql": sql, "ok": True, "h": h, "n_rows": len(rows), "n_set": len(s), "n_cols": ncol,
                "empty": bool(empty), "trunc": trunc, "t": round(time.time() - t0, 3)}
    except Exception as e:
        msg = str(e)
        return {"db_id": db_id, "sql": sql, "ok": False, "err": msg[:120], "timeout": "interrupt" in msg.lower() or (time.time() - t0) >= 29.5,
                "t": round(time.time() - t0, 3)}
    finally:
        if conn is not None:
            try: conn.close()
            except Exception: pass


def collect():
    tr = jl(TREES)
    train = json.load(open(TRAIN))
    S = set()
    gold = {}
    for t in tr:
        for n in t["nodes"]:
            S.add((t["db_id"], n["sql"]))
    for q in {t["qid"] for t in tr}:
        _, db, idx = q.split(":"); S.add((db, train[int(idx)]["SQL"]))
    for run in (HR_RUN, HT_RUN):
        for c in jl(run + "/cands.jsonl"):
            S.add((c["db_id"], c["sql"]))
        for p in jl(run + "/preds.jsonl"):
            S.add((p["db_id"], p["final_sql"]))
    for run in (HT_ZS, HT_ZS_B1):
        for p in jl(run + "/preds.jsonl"):
            S.add((p["db_id"], p["final_sql"]))
    sp = json.load(open(SPLIT))
    for q in sp["ht300_qids"]:
        _, db, idx = q.split(":"); S.add((db, train[int(idx)]["SQL"]))
    return sorted(S)


if __name__ == "__main__":
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    S = collect()
    done = set()
    if os.path.exists(OUTF):
        for l in open(OUTF):
            try:
                r = json.loads(l); done.add((r["db_id"], r["sql"]))
            except Exception:
                pass
    todo = [x for x in S if x not in done]
    print("unique (db,sql):", len(S), "todo", len(todo), flush=True)
    t0 = time.time(); n = 0
    with open(OUTF, "a") as f, Pool(W) as pool:
        for r in pool.imap(run_one, todo, chunksize=4):
            f.write(json.dumps(r) + "\n"); n += 1
            if n % 500 == 0:
                f.flush(); print(n, "/", len(todo), "%.0fs" % (time.time() - t0), flush=True)
    print("done", flush=True)
