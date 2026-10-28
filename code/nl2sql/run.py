"""Resumable runner.

python -m nl2sql.run --system <name> --split <split> --model <key|id> --seed <s> --config exp/configs/<x>.yaml \
       --out <runs>/<exp>/<run_id> [--limit N] [--workers K] [--exp E1] [--temperature T]

* preds.jsonl (CONTRACT section 4) appended per item and flushed; restart skips finished qids.
* locked splits: exp/dev_lock.json must contain a matching entry (nl2sql/lock.py) or the run is refused.
* the pipeline sees only question/evidence/db_id on locked splits (gold stripped); the Evaluator loads gold.
* api_fail / sys_error items are re-queued once after the main pass; the final report prints api_fail AND sys_error (and cache-health)
  counts - a sys_error row is an infrastructure failure, never a silent EX = 0 (scoring failures have ex = null).
* the LLM cache is machine-local (paths.CACHE); at the end the call-cache rows used by the run are archived read-only to
  <out>/llm_cache.jsonl.gz (nl2sql.cache_archive, tools/cache_export.py)."""
import argparse
import json
import os
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from . import config as C
from . import cache_archive, data, lock, paths, sqlcache
from .eval.score import Evaluator
from .llm import LLMClient, LLMError
from .meter import Meter
from .systems import get_system
from .util import json_default


def make_run_id(exp, system, model_key, seed, cfg_sha):
    return "%s-%s-%s-s%d-%s" % (exp or "x", system, model_key, seed, cfg_sha[:8])


def load_done(preds_path):
    done = {}
    if os.path.exists(preds_path):
        with open(preds_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["qid"]] = r
                except Exception:
                    pass
    return done


def process_item(system, evaluator, item, seed, cfg, run_info):
    """Run one question under a Meter, evaluate, return (row, cand_rows)."""
    m = Meter()
    err = None
    res = {}
    t0 = time.time()
    with m:
        try:
            res = system.run(item, seed=seed) or {}
        except LLMError as e:
            err = ("api_fail", str(e)[:200])
        except Exception as e:
            err = ("sys_error:" + type(e).__name__, (str(e) + " | " + traceback.format_exc()[-600:])[:900])
    if not err and res.get("error_class") == "system_error":
        # BaselineSystem.run() swallows exceptions into error_class "system_error" (message in extra.error); before MEMOFIX such rows were
        # neither counted as sys_error nor re-queued, so a run with 500 failed items looked "done" (MIN/M2/memosql-s0).
        ex_ = res.get("extra") or {}
        err = ("sys_error:baseline", (str(ex_.get("error", "")) + " | " + str(ex_.get("trace", ""))[-500:])[:900])
    wall = time.time() - t0
    s = m.summary()
    sql = res.get("final_sql") or ""
    sc = None
    if not err:
        try:
            sc = evaluator.score_sql(item, sql)
        except Exception as e:      # scoring is infrastructure: an exception here must neither drop the result loop nor become EX = 0
            err = ("sys_error:score:" + type(e).__name__, (str(e) + " | " + traceback.format_exc()[-600:])[:900])
            sc = {"ex": None, "exec_ok": False, "soft_f1": None, "jaccard": None, "overlap": None, "auto_class": None}
    elif err:
        sc = {"ex": 0 if evaluator.gold.get(item.qid) else None, "exec_ok": False,
              "soft_f1": 0.0, "jaccard": 0.0, "overlap": 0.0, "auto_class": None}
    row = {"qid": item.qid, "db_id": item.db_id, "split": item.split, "system": run_info["system"], "model": run_info["model"],
           "seed": seed, "final_sql": sql, "exec_ok": bool(sc.get("exec_ok")), "ex": sc.get("ex"),
           "abstained": bool(res.get("abstained", False)), "critic": res.get("critic"),
           "iterations": res.get("iterations", 0), "actions": res.get("actions", []),
           "calls": res.get("calls", s["calls"]) if s["calls"] == 0 else s["calls"],
           "prompt_tokens": s["prompt_tokens"], "completion_tokens": s["completion_tokens"], "cached_tokens": s["cached_tokens"],
           "embed_calls": s["embed_calls"], "rerank_calls": s["rerank_calls"],
           "latency_s": s["latency_s"], "exec_s": s["exec_s"], "wall_s": round(wall, 3),
           "error_class": res.get("error_class"), "stages": s["stages"],
           "soft_f1": sc.get("soft_f1"), "jaccard": sc.get("jaccard"), "overlap": sc.get("overlap"),
           "auto_class": sc.get("auto_class"), "pred_error": sc.get("pred_error")}
    for k in ("ex_ts", "ts_error"):             # Spider test-suite EX (spider_* splits only; absent elsewhere)
        if k in sc:
            row[k] = sc[k]
    for k in ("exemplar_ids", "confidence", "verbalized_confidence", "n_candidates", "trace", "m1_ok", "ctx_tables", "ctx_full_schema", "ctx_tokens",
              # X5 (additive): routing / consistency / arbitration fields of the ours-systems
              "final_status", "consistency", "n_samples", "n_agree", "cons_bits", "route_score", "route_kind", "arbitrated", "arb_choice", "arb_parse_ok", "arb_flip",
              # X7 (additive): value probing, adaptive path
              "probed", "probe_cols", "adaptive_path", "adaptive_cons", "adaptive_n", "adaptive_choice", "adaptive_sizes", "adaptive_tie", "adaptive_v3",
              # X9 (additive): value grounding {n, cols, lits, used, used_root, tokens, ms}
              "value_hints",
              # patent_fb (additive, absent for every other system): feedback-store parameters and the cases shown to the Critic
              "pf"):
        if k in res:
            row[k] = res[k]
    if err:
        row["error_class"], row["err_msg"] = err
    cands = []
    for ci, c in enumerate(res.get("candidates") or []):
        try:
            cs = evaluator.score_sql(item, c.get("sql", "")) if c.get("sql") else {"ex": 0}
        except Exception:
            cs = {"ex": None, "score_error": True}
        c = dict(c)
        c.update({"qid": item.qid, "db_id": item.db_id, "cand_idx": ci, "ex": cs.get("ex")})
        cands.append(c)
    return row, cands


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", default="primary_actor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--exp", default="x")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--conc", type=int, default=8, help="per-model concurrency cap of this process")
    ap.add_argument("--qids-file", default=None, help="restrict to qids listed (one per line)")
    ap.add_argument("--no-tuning-log", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cache", default=None)
    args = ap.parse_args(argv)

    cfg_path = args.config if os.path.isabs(args.config) else str(paths.REPO / args.config)
    cfg = C.load_config(cfg_path)
    model = C.resolve_model(args.model)
    mkey = C.model_key(model)
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    ok, msg = lock.verify(args.system, args.split, args.seed, cfg_path)
    if not ok:
        print("REFUSED: " + msg, file=sys.stderr)
        sys.exit(3)
    cfg_sha = C.config_sha(cfg_path)
    run_id = make_run_id(args.exp, args.system, mkey, args.seed, cfg_sha)
    out = args.out or str(paths.RUNS / args.exp / run_id)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.yaml"), "w") as f:
        yaml.safe_dump({"resolved": cfg, "argv": {k: v for k, v in vars(args).items()}, "run_id": run_id}, f, allow_unicode=True)
    meta = {"run_id": run_id, "system": args.system, "split": args.split, "model": model, "seed": args.seed,
            "config_sha256": cfg_sha, "models_yaml_sha256": C.models_sha(), "prompts_sha256": C.prompts_sha(),
            "git_hash": C.git_hash(), "lock": msg, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=1)

    items = data.load_split(args.split)
    if args.qids_file:
        keep = {l.strip() for l in open(args.qids_file) if l.strip()}
        items = [i for i in items if i.qid in keep]
    if args.limit:
        items = items[: args.limit]
    preds_path = os.path.join(out, "preds.jsonl")
    cands_path = os.path.join(out, "cands.jsonl")
    done = load_done(preds_path)
    todo = [i for i in items if i.qid not in done]
    print("[run] %s split=%s n=%d todo=%d model=%s workers=%d out=%s" % (run_id, args.split, len(items), len(todo), model, args.workers, out), flush=True)

    cache_dir = args.cache or str(paths.CACHE / "llm")
    print("[run] cache=%s (machine-local; archive -> %s/llm_cache.jsonl.gz)" % (cache_dir, out), flush=True)
    llm = LLMClient(cache_path=cache_dir, usage_path=os.path.join(out, "usage.jsonl"),
                    key_file=paths.KEY_FILE, max_concurrency_per_model=args.conc)
    system = get_system(args.system)(llm, model, cfg)
    evaluator = Evaluator(args.split, run_id=run_id)
    if not any(evaluator.gold.values()):
        print("[run] WARNING: split %s has no gold SQL (e.g. livesql_lite: gold locked by the BIRD team); EX will be null, predictions only" % args.split, flush=True)
    info = {"system": args.system, "model": model}
    lock_w = threading.Lock()
    t_start = time.time()

    def write(row, cands):
        with lock_w:
            with open(preds_path, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
            if cands:
                with open(cands_path, "a") as f:
                    for c in cands:
                        f.write(json.dumps(c, ensure_ascii=False, default=json_default) + "\n")

    def pass_(batch, label):
        n_ok = n_bad = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_item, system, evaluator, it, args.seed, cfg, info): it for it in batch}
            for k, fu in enumerate(as_completed(futs), 1):
                try:
                    row, cands = fu.result()
                    write(row, cands)
                except Exception as e:      # one failing item must never end this loop (the executor would drain the queue and drop results)
                    it = futs[fu]
                    row, cands = {"qid": it.qid, "db_id": it.db_id, "split": it.split, "system": info["system"], "model": info["model"],
                                  "seed": args.seed, "final_sql": "", "exec_ok": False, "ex": None, "abstained": False, "critic": None,
                                  "iterations": 0, "actions": [], "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                  "latency_s": 0.0, "error_class": "sys_error:runner:" + type(e).__name__,
                                  "err_msg": (str(e) + " | " + traceback.format_exc()[-500:])[:900]}, []
                    try:
                        write(row, cands)
                    except Exception:
                        print("[run] cannot write error row for %s" % it.qid, flush=True)
                bad = row["error_class"] in ("api_fail",) or str(row["error_class"]).startswith("sys_error")
                n_bad += int(bad)
                n_ok += int(not bad)
                if k % 25 == 0 or k == len(batch):
                    print("[run] %s %d/%d bad=%d elapsed=%.0fs 429=%d" % (label, k, len(batch), n_bad, time.time() - t_start, llm.stats["n429"]), flush=True)
        return n_bad

    todo.sort(key=lambda i: i.db_id)          # same-db questions adjacent: better implicit context-cache hits (shared schema prefix)
    if hasattr(system, "warm"):
        system.warm(todo)
    pass_(todo, "pass1")
    # re-queue api_fail / sys_error once (spec: rerun the shard once)
    done = load_done(preds_path)
    redo = [i for i in items if i.qid in done and (done[i.qid].get("error_class") == "api_fail" or str(done[i.qid].get("error_class")).startswith("sys_error"))]
    if redo:
        # drop failed rows so the retry replaces them (keep a note)
        keep = [r for q, r in done.items() if q not in {i.qid for i in redo}]
        with open(preds_path + ".failed1", "a") as f:
            for i in redo:
                f.write(json.dumps(done[i.qid], ensure_ascii=False) + "\n")
        with open(preds_path, "w") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("[run] re-queue %d failed items" % len(redo), flush=True)
        pass_(redo, "pass2")

    from .analysis.summarize import summarize_run
    summ = summarize_run(preds_path)
    summ["client_stats"] = llm.snapshot_stats()
    summ["wall_s"] = round(time.time() - t_start, 1)
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    keys = ("n", "ex", "calls_per_q", "prompt_tokens_per_q", "completion_tokens_per_q", "api_fail", "sys_error", "wall_s", "client_stats")
    print("[run] done", json.dumps({k: summ.get(k) for k in keys}), flush=True)
    if summ.get("sys_error") or summ.get("api_fail"):
        print("[run] WARNING: %s sys_error + %s api_fail rows remain; their EX is not a measurement. Inspect error_class/err_msg in preds.jsonl "
              "and rerun the same command to retry them." % (summ.get("sys_error"), summ.get("api_fail")), flush=True)
    if any(summ["client_stats"].get("cache_" + k) for k in ("db_errors", "quarantined", "dropped_writes", "miss_on_error", "bad_rows")):
        print("[run] WARNING: cache trouble (treated as cache misses, results unaffected): %s" % json.dumps(sqlcache.stats()), flush=True)
    cache_archive.finish_run(out, store=llm.cache)
    if args.split == "bird_train_HT" and not args.no_tuning_log:
        try:
            with open(paths.EXP_DIR / "tuning_log.jsonl", "a") as f:
                f.write(json.dumps({"system": args.system, "config_sha": cfg_sha, "split": args.split, "n_q": summ["n"], "metric": {"ex": summ["ex"]},
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke": args.smoke, "run_id": run_id}) + "\n")
        except Exception:
            pass
    llm.close()


if __name__ == "__main__":
    main()
