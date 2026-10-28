"""B1 smoke runner: python -m tests_b1.smoke_b1 --system NAME [--n 10] [--model ID] [--cfg k=v ...]
Uses the real nl2sql core when present, else the shim.  Writes JSONL to --out."""
import argparse, json, os, sys, time, contextvars
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nl2sql.systems.baselines import BASELINES
from nl2sql.systems.baselines import _common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--split", default="bird_train_HT")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--model", default="qwen3.7-plus-2026-05-26")
    ap.add_argument("--cfg", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = {}
    for kv in a.cfg:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except Exception:
            pass
        cfg[k] = v
    root = os.environ.get("NL2SQL_ROOT", os.path.expanduser("~/nl2sql_work"))
    from nl2sql.llm import LLMClient
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    llm = LLMClient(os.path.join(root, "cache", "b1_smoke"), a.out + ".usage.jsonl")
    if "emb_cache_dir" not in cfg:
        cfg["emb_cache_dir"] = os.path.join(root, "cache", "b1_emb")
    if "cache_dir" not in cfg:
        cfg["cache_dir"] = os.path.join(root, "cache", "b1_dbinfo")
    items = C.datamod.load_split(a.split)
    # spread over dbs deterministically
    items = items[a.offset::max(1, len(items) // max(a.n, 1))][:a.n]
    sysobj = BASELINES[a.system](llm, a.model, cfg)
    from nl2sql import meter as M  # type: ignore
    def one(it):
        with M.Meter() as m:
            row = sysobj.run(it, seed=a.seed)
        row["meter"] = m.summary()
        gold = C.dbmod.execute(it.db_path, it.gold_sql)
        pred = C.dbmod.execute(it.db_path, row["final_sql"]) if row["final_sql"] else {"ok": False, "rows": []}
        row["ex"] = int(pred["ok"] and gold["ok"] and C.dbmod.ex_match(pred["rows"], gold["rows"]))
        return row
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda it: contextvars.copy_context().run(one, it), items))
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    n = len(rows)
    s = lambda k: sum(r[k] for r in rows)
    print(json.dumps({"system": a.system, "n": n, "ex": s("ex") / n, "exec_ok": sum(r["exec_ok"] for r in rows) / n,
                      "calls/q": s("calls") / n, "prompt_tok/q": s("prompt_tokens") / n, "compl_tok/q": s("completion_tokens") / n,
                      "meter_calls/q": sum(r["meter"]["calls"] for r in rows) / n, "wall_s": round(time.time() - t0, 1),
                      "errors": [r["error_class"] for r in rows if r["error_class"]]}))

if __name__ == "__main__":
    main()
