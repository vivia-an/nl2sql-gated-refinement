#!/usr/bin/env python3
"""Offline dry run of a config against archived call caches: which LLM calls of a live run would be cache hits and which would be new (X7). NO network,
NO API key is read, BIRD-train tuning splits only (worker machine venv).

  NL2SQL_CACHE=<NL2SQL_CACHE> python tools/cache_dryrun.py --system full_gated --config exp/configs/full_gated_adaptive.yaml --split bird_train_HT
        --import-run $R/runs/X5/2-ht-cons [--import-run ...] [--emb-from <NL2SQL_CACHE>/llm/emb.sqlite] [--kstore-from <NL2SQL_CACHE>/kstore]
        [--limit N] [--workers 4] --out $R/x7_work/dry/adaptive-ht

1. The call-cache archives (`llm_cache.jsonl.gz`) of the given runs are imported into the PRIVATE cache `$NL2SQL_CACHE/llm` (refused when that is the
   default worker machine cache or on /volume); optional read-only copies of an embedding / rerank cache (sqlite backup API) and of the knowledge store.
2. The real system runs with `LLMClient(offline=True)`. A chat call that is not in the cache is COUNTED (by stage tag, with the estimated prompt tokens)
   and answered with a dummy text instead of failing; the rest of the pipeline continues on the dummy, so counts behind a miss are a lower bound
   (a revision that would be needed after a real, different answer is not seen). An embedding / rerank miss cannot be faked: that question is
   reported in `embed_rerank_miss` (its chat calls are not counted).
3. Output: `<out>/dryrun_summary.json` (+ preds.jsonl / cands.jsonl): chat calls total / cached / new per tag, new calls per question, questions with all
   calls cached, estimated prompt tokens and a price estimate for the new calls (prompt tokens x list price + `--assumed-completion` tokens x output price).
   EX in this summary is NOT a measurement when new calls were answered by the dummy; for fully cached questions it is exact."""
import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from collections import Counter

from x7_common import HERE, TUNING_SPLITS  # noqa: F401  (sys.path set there)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--system", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", default="qwen37plus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--import-run", action="append", default=[])
    ap.add_argument("--emb-from", default=None)
    ap.add_argument("--kstore-from", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--qids-file", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--assumed-completion", type=int, default=60, help="completion tokens assumed per new call, for the price estimate only")
    ap.add_argument("--dump-misses", default=None, help="write every missed chat request (tag, params, messages) to this JSONL, for diagnosing key differences")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.split not in TUNING_SPLITS:
        raise SystemExit("cache_dryrun works on tuning splits only (%s)" % ", ".join(TUNING_SPLITS))

    from nl2sql import cache_archive as CA, config as C, data, paths
    from nl2sql.eval.score import Evaluator
    from nl2sql.llm import LLMClient, LLMError
    from nl2sql.run import process_item
    from nl2sql.systems import get_system
    from nl2sql.util import est_tokens

    cache = str(paths.CACHE)
    if os.environ.get("NL2SQL_CACHE") is None or os.path.abspath(cache) == os.path.abspath(paths.DEFAULT_CACHE) or cache.startswith("/volume"):
        raise SystemExit("refusing: set NL2SQL_CACHE to a PRIVATE machine-local directory (not the default worker machine cache, not /volume); got %s" % cache)
    llm_dir = os.path.join(cache, "llm")
    os.makedirs(llm_dir, exist_ok=True)
    imp = CA.import_archives([os.path.abspath(p) for p in a.import_run], llm_dir) if a.import_run else {}
    print("[dryrun] imported", json.dumps(imp), flush=True)
    if a.emb_from and not os.path.exists(os.path.join(llm_dir, "emb.sqlite")):
        src = sqlite3.connect("file:%s?mode=ro" % a.emb_from, uri=True)
        dst = sqlite3.connect(os.path.join(llm_dir, "emb.sqlite"))
        src.backup(dst)
        dst.close()
        src.close()
    if a.kstore_from:
        kd = os.path.join(cache, "kstore")
        os.makedirs(kd, exist_ok=True)
        for f in os.listdir(a.kstore_from):
            if not os.path.exists(os.path.join(kd, f)):
                shutil.copy2(os.path.join(a.kstore_from, f), os.path.join(kd, f))

    import contextvars
    misses = Counter()
    miss_tokens = Counter()
    tally = contextvars.ContextVar("x7_dryrun_new_calls", default=[0])

    class DryClient(LLMClient):
        def chat(self, model, messages, *, temperature=0.0, seed=None, max_tokens=2048, enable_thinking=False, n=1, tag="", top_p=1.0):
            try:
                return super().chat(model, messages, temperature=temperature, seed=seed, max_tokens=max_tokens, enable_thinking=enable_thinking,
                                    n=n, tag=tag, top_p=top_p)
            except LLMError as e:
                if "offline cache miss" not in str(e):
                    raise
                misses[tag] += 1
                if a.dump_misses:
                    with open(a.dump_misses, "a") as df:
                        df.write(json.dumps({"tag": tag, "model": model, "temperature": temperature, "seed": seed, "n": n, "max_tokens": max_tokens, "top_p": top_p,
                                             "enable_thinking": enable_thinking, "messages": messages}, ensure_ascii=False) + "\n")
                tally.get()[0] += 1
                miss_tokens[tag] += est_tokens("".join(m["content"] for m in messages))
                text = {"m1": "{}", "crit": '{"semantic": 10, "issues": [], "suggestions": []}'}.get(tag, "SELECT 1")
                return {"text": [text] * n if n != 1 else text, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "latency_s": 0.0, "cached": False}

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    cfg_path = a.config if os.path.isabs(a.config) else str(paths.REPO / a.config)
    cfg = C.load_config(cfg_path)
    model = C.resolve_model(a.model)
    llm = DryClient(cache_path=llm_dir, usage_path=os.path.join(out, "usage.jsonl"), key_file="/nonexistent/no-key", offline=True)
    system = get_system(a.system)(llm, model, cfg)
    items = data.load_split(a.split)
    if a.qids_file:
        keep = {l.strip() for l in open(a.qids_file) if l.strip()}
        items = [i for i in items if i.qid in keep]
    if a.limit:
        items = items[: a.limit]
    evaluator = Evaluator(a.split, run_id="x7-dryrun")
    info = {"system": a.system, "model": model}
    items.sort(key=lambda i: i.db_id)
    if hasattr(system, "warm"):
        try:
            system.warm(items)
        except LLMError as e:
            raise SystemExit("warm-up needs embeddings that are not in the private cache (%s): pass --emb-from / --kstore-from from a worker machine cache that holds them" % e)
    rows, cands_all, new_per_q = [], [], {}
    from concurrent.futures import ThreadPoolExecutor

    def one(it):
        tally.set([0])
        row, cands = process_item(system, evaluator, it, a.seed, cfg, info)
        new_per_q[it.qid] = tally.get()[0]
        return it.qid, row, cands

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for qid, row, cands in ex.map(one, items):
            rows.append(row)
            cands_all.extend(cands)
    with open(os.path.join(out, "preds.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    with open(os.path.join(out, "cands.jsonl"), "w") as f:
        for c in cands_all:
            f.write(json.dumps(c, ensure_ascii=False, default=str) + "\n")
    stages = Counter()
    for r in rows:
        for t, v in (r.get("stages") or {}).items():
            stages[t] += v.get("calls", 0)
    emb_fail = [r["qid"] for r in rows if str(r.get("error_class") or "").startswith(("api_fail", "sys_error"))]
    n_new = sum(misses.values())
    prices = {"in": 2.0, "out": 8.0}
    est_yuan = (sum(miss_tokens.values()) * prices["in"] + n_new * a.assumed_completion * prices["out"]) / 1e6
    # questions whose every chat call was cached = questions without a failed / dummy-answered request: derive from the usage log (cached flag)
    summ = {"system": a.system, "config": a.config, "split": a.split, "n_questions": len(rows), "n_questions_fully_cached": sum(1 for q, v in new_per_q.items() if v == 0), "imported": imp,
            "chat_calls_total": int(sum(stages.values())), "chat_calls_by_tag": dict(stages), "new_calls_by_tag": dict(misses), "new_calls_total": n_new,
            "new_calls_per_question": n_new / float(max(len(rows), 1)), "calls_per_question_live": sum(stages.values()) / float(max(len(rows), 1)),
            "est_new_prompt_tokens": int(sum(miss_tokens.values())), "est_new_yuan_list_price": round(est_yuan, 4),
            "assumed_completion_tokens_per_new_call": a.assumed_completion, "rows_with_error": emb_fail[:20], "n_rows_with_error": len(emb_fail),
            "ex_note": "exact only for questions with no new call (dummy answers otherwise)", "wall_s": round(time.time() - t_start, 1)}
    json.dump(summ, open(os.path.join(out, "dryrun_summary.json"), "w"), indent=1)
    print("[dryrun] " + json.dumps({k: summ[k] for k in ("n_questions", "n_questions_fully_cached", "chat_calls_total", "new_calls_total", "new_calls_by_tag", "new_calls_per_question",
                                                          "est_new_yuan_list_price", "n_rows_with_error")}), flush=True)
    llm.close()
    return summ


if __name__ == "__main__":
    main()
