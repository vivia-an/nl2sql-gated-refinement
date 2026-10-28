"""Markdown table of smoke outputs: python -m tests_b1.summarize_b1 OUT_DIR  (reads *.jsonl rows written by smoke_b1)."""
import glob, json, os, sys
d = sys.argv[1]
print("| run | n | EX | exec_ok | calls/q | prompt tok/q | completion tok/q | wall s/q (cache-affected) | errors |")
print("|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
    if f.endswith(".usage.jsonl"):
        continue
    rows = [json.loads(l) for l in open(f)]
    if not rows:
        continue
    n = len(rows)
    sysn = os.path.basename(f)[:-6]
    avg = lambda k: sum(r[k] for r in rows) / n
    print("| %s | %d | %.2f | %.2f | %.1f | %.0f | %.0f | %.1f | %s |" % (
        sysn, n, sum(r.get("ex", 0) for r in rows) / n, sum(1 for r in rows if r["exec_ok"]) / n, avg("calls"),
        avg("prompt_tokens"), avg("completion_tokens"), avg("latency_s"),
        ",".join(sorted({r["error_class"] for r in rows if r["error_class"]})) or "-"))
