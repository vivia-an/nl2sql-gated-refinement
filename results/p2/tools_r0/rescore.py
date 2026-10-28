"""Re-score preds.jsonl of HT runs against a FRESH gold cache (NL2SQL_CACHE must point to an empty/local dir); report disagreements."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nl2sql import data
from nl2sql.eval.score import Evaluator
split = sys.argv[1]
items = {i.qid: i for i in data.load_split(split)}
ev = Evaluator(split, run_id="r0_rescore")
tot = bad = 0
for d in sys.argv[2:]:
    rows = [json.loads(l) for l in open(d + "/preds.jsonl")]
    dis = 0
    for r in rows:
        try:
            s = ev.score_sql(items[r["qid"]], r["final_sql"])
        except Exception as e:
            print("EXC", r["qid"], type(e).__name__, str(e)[:80], "stored ex", r["ex"], "stored class", r.get("auto_class"), flush=True)
            continue
        tot += 1
        if s["ex"] != r["ex"]:
            dis += 1
            print("DIFF", r["qid"], "stored", r["ex"], r.get("auto_class"), "now", s["ex"], s.get("auto_class"), flush=True)
    bad += dis
    print(os.path.basename(d)[-40:], len(rows), "disagree", dis, flush=True)
print("TOTAL", tot, "disagree", bad)
