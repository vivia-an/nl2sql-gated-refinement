"""Q1 addendum: execution-error roots by db and by error text (needs exec_results.jsonl only for text; trees have status)."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
from trees import *
Qs = load_trees()
ER = {}
for r in jl(OUT + "/exec_results.jsonl"):
    ER[(r["db_id"], r["sql"])] = r
res = {}
errs = collections.Counter(); ex_by_db = collections.Counter()
for q in Qs:
    r = ER.get((q.db, q.root["sql"]))
    if r is None or r["ok"]: continue
    msg = r["err"]
    key = (q.db, msg[:60])
    errs[key] += 1
res["root_errors_top"] = [(list(k), v) for k, v in errs.most_common(25)]
# by root status: replay gain / harm
rp = [(q, q.p0()) for q in Qs]
tab = collections.defaultdict(lambda: collections.Counter())
for q, r in rp:
    st = "exec_error" if (q.root["status"].startswith("err") or q.root["status"] == "timeout") else q.root["status"]
    tab[st]["n"] += 1
    tab[st]["root_right"] += q.root["ex"]
    tab[st]["final_right"] += r["ex"]
    tab[st]["gain"] += int(q.root["ex"] == 0 and r["ex"] == 1)
    tab[st]["harm"] += int(q.root["ex"] == 1 and r["ex"] == 0)
    tab[st]["revised"] += int(len(r["nodes"]) > 1)
    tab[st]["calls"] += r["calls"]
res["P0_replay_by_root_status"] = {k: dict(v) for k, v in tab.items()}
tab2 = collections.defaultdict(lambda: collections.Counter())
for q, r in rp:
    st = "exec_error" if (q.root["status"].startswith("err") or q.root["status"] == "timeout") else q.root["status"]
    d = q.db if q.db == "beer_factory" else "others"
    key = d + "/" + st
    tab2[key]["n"] += 1; tab2[key]["gain"] += int(q.root["ex"] == 0 and r["ex"] == 1); tab2[key]["harm"] += int(q.root["ex"] == 1 and r["ex"] == 0)
res["P0_replay_gain_harm_beer_vs_others"] = {k: dict(v) for k, v in tab2.items()}
dump("q1_errs.json", res)
print(json.dumps(res, indent=1))
