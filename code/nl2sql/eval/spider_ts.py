"""Spider test-suite execution accuracy (Zhong et al. 2020; official taoyds/test-suite-sql-eval @ e97acc54).

The OFFICIAL code is used, not re-implemented: `exec_eval.eval_exec_match` from
  $NL2SQL_DATA/eval_tools/test-suite-sql-eval/   (override with env NL2SQL_TS_DIR)
run against the distilled test-suite databases in
  $NL2SQL_DATA/spider_testsuite_dbs/database/<db_id>/*.sqlite   (override with env NL2SQL_TS_DBS)
exactly as the official `evaluation.py` does for `--etype exec` (prediction `value` -> `1` replacement, DISTINCT
removed unless keep_distinct, order-sensitive iff gold has ORDER BY, all sqlite files of the db directory must agree).
We report the "with values" variant (plug_value=False): our systems predict literals themselves.

Why subprocess workers: the official function can block inside sqlite (its asyncio timeout cannot interrupt C code) and
prints to stdout. Each worker is a fresh `python -m nl2sql.eval.spider_ts --worker` process speaking JSON lines;
a call that exceeds `timeout_s` kills the worker (respawned) and scores 0 with error="timeout".
A gold query that errors raises AssertionError in the official code; here it is returned as error="gold_error:..."
and ex_ts = 0 (item stays in the denominator, like gold_fail for BIRD).

CLI (post-hoc scoring of a preds.jsonl or gold-vs-gold):
  python -m nl2sql.eval.spider_ts --split spider_dev --preds <run>/preds.jsonl --out <run>/ts.jsonl [--workers 4]
  python -m nl2sql.eval.spider_ts --split spider_dev --gold-as-pred --limit 200 --out gold_ts.jsonl"""
import argparse
import json
import os
import queue
import select
import subprocess
import sys
import threading
import time

from .. import paths

TS_SPLITS = {"spider_dev", "spider_syn", "spider_realistic", "spider_dk"}


def ts_dir():
    return os.environ.get("NL2SQL_TS_DIR", str(paths.DATA / "eval_tools" / "test-suite-sql-eval"))


def ts_dbs():
    return os.environ.get("NL2SQL_TS_DBS", str(paths.DATA / "spider_testsuite_dbs" / "database"))


def has_suite(db_id):
    return os.path.exists(os.path.join(ts_dbs(), db_id, db_id + ".sqlite"))


# ------------------------------------------------------------------ worker side
def _worker_main():
    out = os.fdopen(os.dup(1), "w")          # official code prints to stdout: keep our channel separate
    sys.stdout = sys.stderr
    sys.path.insert(0, ts_dir())
    import exec_eval                           # noqa: official module
    for line in sys.stdin:
        try:
            req = json.loads(line)
        except Exception:
            continue
        res = {"id": req.get("id"), "ex_ts": 0, "error": None}
        try:
            db = os.path.join(ts_dbs(), req["db_id"], req["db_id"] + ".sqlite")
            p = req["pred"]
            if req.get("official_value_replace", True):
                p = p.replace("value", "1")           # official evaluation.py does this before eval_exec_match
            res["ex_ts"] = int(exec_eval.eval_exec_match(db=db, p_str=p, g_str=req["gold"], plug_value=bool(req.get("plug_value")),
                                                         keep_distinct=bool(req.get("keep_distinct")),
                                                         progress_bar_for_each_datapoint=False))
        except AssertionError as e:
            res["error"] = "gold_error: " + str(e)[:200]
        except Exception as e:
            res["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        out.write(json.dumps(res) + "\n")
        out.flush()


# ------------------------------------------------------------------ client side
class _Worker:
    def __init__(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(paths.CODE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        self.p = subprocess.Popen([sys.executable, "-m", "nl2sql.eval.spider_ts", "--worker"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env,
                                  cwd=str(paths.CODE_DIR))

    def call(self, req, timeout_s):
        self.p.stdin.write(json.dumps(req) + "\n")
        self.p.stdin.flush()
        r, _, _ = select.select([self.p.stdout], [], [], timeout_s)
        if not r:
            return None
        line = self.p.stdout.readline()
        return json.loads(line) if line else None

    def kill(self):
        try:
            self.p.kill()
            self.p.wait(5)
        except Exception:
            pass


class TSPool:
    """Thread-safe pool of test-suite workers. match(db_id, pred, gold) -> {"ex_ts": 0|1, "error": str|None}."""

    def __init__(self, n=2, timeout_s=300.0):
        self.timeout_s = timeout_s
        self._q = queue.Queue()
        self._all = []
        self._n = 0
        for _ in range(n):
            w = _Worker()
            self._all.append(w)
            self._q.put(w)

    def match(self, db_id, pred, gold, plug_value=False, keep_distinct=False):
        if not has_suite(db_id):
            return {"ex_ts": None, "error": "no_suite_db"}
        if not pred or not pred.strip():
            return {"ex_ts": 0, "error": "empty_pred"}
        w = self._q.get()
        self._n += 1
        try:
            res = w.call({"id": self._n, "db_id": db_id, "pred": pred, "gold": gold, "plug_value": plug_value,
                          "keep_distinct": keep_distinct}, self.timeout_s)
            if res is None:                    # timeout or worker died
                w.kill()
                w = _Worker()
                res = {"ex_ts": 0, "error": "timeout"}
            return {"ex_ts": res["ex_ts"], "error": res.get("error")}
        finally:
            self._q.put(w)

    def close(self):
        for w in list(self._q.queue):
            w.kill()


_POOL = None
_POOL_LOCK = threading.Lock()


def shared_pool(n=None):
    """Process-wide pool used by Evaluator (size env NL2SQL_TS_WORKERS, default 2)."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = TSPool(n or int(os.environ.get("NL2SQL_TS_WORKERS", "2")))
        return _POOL


# ------------------------------------------------------------------ CLI
def _cli(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--split", default="spider_dev")
    ap.add_argument("--preds", default=None)
    ap.add_argument("--gold-as-pred", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args(argv)
    if args.worker:
        return _worker_main()
    from .. import data
    gold = data.load_gold(args.split, run_id="spider_ts_cli")
    items = {i.qid: i for i in data.load_split(args.split)}
    if args.gold_as_pred:
        rows = [{"qid": q, "db_id": items[q].db_id, "final_sql": g, "ex": 1} for q, g in gold.items() if q in items]
    else:
        rows = [json.loads(l) for l in open(args.preds) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    pool = TSPool(args.workers, args.timeout)
    res = {}
    lock = threading.Lock()
    qq = queue.Queue()
    for r in rows:
        qq.put(r)

    def work():
        while True:
            try:
                r = qq.get_nowait()
            except queue.Empty:
                return
            g = gold.get(r["qid"], "")
            m = pool.match(r["db_id"], r.get("final_sql") or "", g) if g else {"ex_ts": None, "error": "no_gold"}
            with lock:
                res[r["qid"]] = {"qid": r["qid"], "db_id": r["db_id"], "ex": r.get("ex"), "ex_ts": m["ex_ts"], "error": m["error"]}

    t0 = time.time()
    th = [threading.Thread(target=work) for _ in range(args.workers)]
    [t.start() for t in th]
    [t.join() for t in th]
    pool.close()
    out = [res[r["qid"]] for r in rows if r["qid"] in res]
    if args.out:
        with open(args.out, "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
    sc = [r for r in out if r["ex_ts"] is not None]
    S = {"n": len(out), "n_scored": len(sc), "ex_ts": sum(r["ex_ts"] for r in sc) / max(1, len(sc)),
         "ex_plain": sum((r["ex"] or 0) for r in sc) / max(1, len(sc)), "errors": sum(1 for r in out if r["error"]),
         "disagree_ts_vs_plain": sum(1 for r in sc if r["ex"] is not None and r["ex"] != r["ex_ts"]), "wall_s": round(time.time() - t0, 1)}
    print(json.dumps(S))


if __name__ == "__main__":
    _cli()
