"""Run the first-stage requests of a system through the DashScope Batch API (50 % price, 24h window, no context-cache discount)
and store the results in the call cache; the normal runner then finishes with cache hits.
  python tools/batch_prefill.py --system zs_ours --split bird_train_HT --model qwen37plus --config exp/configs/zs.yaml [--limit N]
  python tools/batch_prefill.py --probe --model qwen37plus      # 3-request probe: does this model accept batch?
Only for non-interactive first stages (zero-shot / initial generation, critic scoring of fixed candidates, E2 sweeps)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nl2sql import config as C, data, paths  # noqa: E402
from nl2sql.llm import LLMClient  # noqa: E402
from nl2sql.systems import get_system  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system")
    ap.add_argument("--split")
    ap.add_argument("--model", required=True)
    ap.add_argument("--config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--poll", type=float, default=20)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=str(paths.RUNS / "batch"))
    a = ap.parse_args()
    model = C.resolve_model(a.model)
    llm = LLMClient(a.cache or str(paths.CACHE / "llm"), os.path.join(a.out, "usage_batch.jsonl"), paths.KEY_FILE)
    if a.probe:
        reqs = [{"messages": [{"role": "user", "content": "Reply with the single word OK. #%d" % i}], "max_tokens": 8, "tag": "probe"} for i in range(3)]
    else:
        cfg = C.load_config(a.config if os.path.isabs(a.config) else str(paths.REPO / a.config))
        sysm = get_system(a.system)(llm, model, cfg)
        items = data.load_split(a.split)
        if a.limit:
            items = items[: a.limit]
        reqs = []
        for it in items:
            r = sysm.plan_requests(it, a.seed)
            if r is None:
                sys.exit("system %s has no batchable first stage" % a.system)
            reqs += r
    res = llm.batch_prefill(model, reqs, poll_s=a.poll, state_dir=a.out)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
