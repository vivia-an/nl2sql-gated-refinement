"""Offline phase of Memo-SQL: python -m tests_b1.build_memo_memory --out PATH [--n 400] [--model ID]
Uses the pool split only.  Reads the API key through LLMClient (never printed)."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nl2sql.llm import LLMClient
from nl2sql.systems.baselines.memo_sql import build_memory

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--cand", type=int, default=3)
ap.add_argument("--model", default="qwen3.7-plus-2026-05-26")
a = ap.parse_args()
root = os.environ.get("NL2SQL_ROOT", os.path.expanduser("~/nl2sql_work"))
llm = LLMClient(os.path.join(root, "cache", "b1_smoke"), a.out + ".usage.jsonl")
t0 = time.time()
n, usage = build_memory(llm, a.model, a.out, n_questions=a.n, n_cand=a.cand)
print(json.dumps({"entries": n, "questions": a.n, "usage": usage, "wall_s": round(time.time() - t0, 1)}))
