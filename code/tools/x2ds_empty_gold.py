"""Post hoc: BIRD development questions whose reference result is empty (no rows) or one row of NULLs, the population on which an
empty-result gate can only harm, and the EX of stored arms on them. Gold SQL executed through the scorer's gold cache; no model call.
    PYTHONPATH=code python3 code/tools/x2ds_empty_gold.py name=preds.jsonl ... (writes empty_gold.json, or the file named by $EMPTY_GOLD_OUT)"""
import json, os, sys
from nl2sql import data
from nl2sql.eval.score import Evaluator
ev = Evaluator("bird_dev")
items = data.load_split("bird_dev", with_gold=True)
emp, nullrow, fail = [], [], []
for it in items:
    g = ev.gold_exec(it)
    if not g or not g.get("ok"):
        fail.append(it.qid); continue
    rows = g.get("rows") or []
    if g.get("big"):
        continue
    if not rows:
        emp.append(it.qid)
    elif len(rows) == 1 and all(c is None for c in rows[0]):
        nullrow.append(it.qid)
out = {"n": len(items), "gold_empty": len(emp), "gold_one_null_row": len(nullrow), "gold_fail": len(fail), "qids_empty": emp, "qids_null": nullrow}
arms = {}
for a in sys.argv[1:]:
    name, p = a.split("=", 1)
    pr = {}
    for l in open(p):
        r = json.loads(l); pr[r["qid"]] = r
    arms[name] = {"empty_ex": [pr[q]["ex"] for q in emp + nullrow if q in pr]}
    arms[name]["sum"] = sum(x or 0 for x in arms[name]["empty_ex"])
out["arms"] = {k: v["sum"] for k, v in arms.items()}
print(json.dumps({k: v for k, v in out.items() if not k.startswith("qids")}, indent=1))
json.dump(out, open(os.environ.get("EMPTY_GOLD_OUT", "empty_gold.json"), "w"), indent=1)
