#!/usr/bin/env python3
"""R3 pre-batch probe: ONE tiny chat request per model key given (max_tokens 1, throwaway cache, key read from the key file, never printed).
Exit 0 = every response ok; exit 9 = a response says Arrearage / quota / insufficient balance / access denied (stop ALL runs); exit 2 = other failure.
    python tools/r3_probe.py qwen37plus qwen37flash qwen3527b"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nl2sql import config as C, paths  # noqa: E402
from nl2sql.llm import LLMClient, LLMError  # noqa: E402

STOP = ("arrearage", "quota", "insufficient", "overdue", "accessdenied", "balance")


def probe(keys):
    rc = 0
    with tempfile.TemporaryDirectory() as d:
        llm = LLMClient(cache_path=os.path.join(d, "c"), usage_path=None, key_file=paths.KEY_FILE, max_concurrency_per_model=1, max_attempts=2)
        for k in keys:
            m = C.resolve_model(k)
            try:
                llm.chat(m, [{"role": "user", "content": "ok"}], temperature=0.0, seed=0, max_tokens=1, tag="r3probe")
                print("[r3_probe] %s ok" % k, flush=True)
            except LLMError as e:
                msg = str(e).lower()
                if any(s in msg for s in STOP):
                    print("[r3_probe] %s STOP: %s" % (k, str(e)[:160]), flush=True)
                    return 9
                print("[r3_probe] %s failed: %s" % (k, str(e)[:160]), flush=True)
                rc = 2
        llm.close()
    return rc


if __name__ == "__main__":
    sys.exit(probe(sys.argv[1:] or ["qwen37plus"]))
