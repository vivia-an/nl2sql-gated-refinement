"""Sustainable-concurrency probe. For each concurrency level fires N zero-shot generation requests (unique seeds -> no cache hits) and reports
calls/s, latency p50/p95, 429/5xx/timeouts, failures. python tools/throughput_probe.py --model qwen37plus --levels 8,16,32,48 --n 120"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nl2sql import config as C, data, paths  # noqa: E402
from nl2sql import prompts_util as P  # noqa: E402
from nl2sql.llm import LLMClient, LLMError  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen37plus")
    ap.add_argument("--levels", default="8,16,32,48")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default=str(paths.RUNS / "throughput"))
    a = ap.parse_args()
    model = C.resolve_model(a.model)
    items = data.load_split("bird_train_HT")
    msgs = []
    for it in items[:60]:
        sc = data.get_schema(it)
        msgs.append([{"role": "system", "content": P.render("common.system")},
                     {"role": "user", "content": P.render("gen_zs.user", schema=sc["ddl"], question=it.question, evidence=it.evidence or "(none)")}])
    res = []
    seed0 = int(time.time()) % 100000 * 1000
    for lvl in [int(x) for x in a.levels.split(",")]:
        llm = LLMClient(os.path.join(a.out, "probe_cache.sqlite"), os.path.join(a.out, "usage.jsonl"), paths.KEY_FILE, max_concurrency_per_model=lvl)
        lats, fails, toks = [], 0, 0

        def one(i):
            t = time.time()
            try:
                r = llm.chat(model, msgs[i % len(msgs)], temperature=0.0, seed=seed0 + lvl * 1000 + i, max_tokens=256, tag="probe")
                return time.time() - t, r["prompt_tokens"] + r["completion_tokens"], 0
            except LLMError:
                return time.time() - t, 0, 1

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=lvl) as ex:
            out = list(ex.map(one, range(a.n)))
        wall = time.time() - t0
        lats = sorted(o[0] for o in out)
        rec = {"model": model, "concurrency": lvl, "n": a.n, "wall_s": round(wall, 1), "calls_per_s": round(a.n / wall, 2),
               "tokens_per_s": round(sum(o[1] for o in out) / wall, 0), "lat_p50": round(lats[len(lats) // 2], 2),
               "lat_p95": round(lats[int(0.95 * len(lats))], 2), "fails": sum(o[2] for o in out),
               "n429": llm.stats["n429"], "n5xx": llm.stats["n5xx"], "ntimeout": llm.stats["ntimeout"], "retries": llm.stats["retries"]}
        res.append(rec)
        print(json.dumps(rec), flush=True)
    json.dump(res, open(os.path.join(a.out, "throughput_%s.json" % a.model), "w"), indent=1)


if __name__ == "__main__":
    main()
