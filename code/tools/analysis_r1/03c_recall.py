"""Q2 addendum: what the V3 critic flags (score < 8.5 = revise) by root status; confusion of flag vs correctness. HR trees roots, HT roots."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
from trees import *
Qs = load_trees()
htc = [c for c in jl(HT_RUN + "/cands.jsonl") if c["cand_idx"] == 0]
res = {}
for nm, rows in (("HR", [(q.root["status"], q.root["score"], q.root["semantic"], q.root["ex"], q.db) for q in Qs]), ("HT", [(c["status"], c["score"], c["semantic"], c["ex"], c["db_id"]) for c in htc])):
    tab = collections.defaultdict(collections.Counter)
    for st, sc, sem, ex, db in rows:
        g = "exec_error" if st.startswith("err") or st == "timeout" else st
        flag = sc < 8.5
        tab[g]["n"] += 1; tab[g]["wrong"] += int(ex == 0); tab[g]["flagged"] += int(flag); tab[g]["wrong_flagged"] += int(ex == 0 and flag); tab[g]["right_flagged"] += int(ex == 1 and flag)
        tab[g]["sem<7_flag"] += int(sem is not None and sem < 7)
        tab[g]["wrong_sem10"] += int(ex == 0 and sem == 10)
    res[nm] = {k: dict(v) for k, v in tab.items()}
    tot = collections.Counter()
    for v in tab.values(): tot.update(v)
    res[nm]["ALL"] = dict(tot)
    # semantic-only (executable roots): recall/precision of flagging wrong ones
    ex_ok = [(sem, ex) for st, sc, sem, ex, db in rows if st in ("ok_nonempty", "ok_empty")]
    w = [s for s, e in ex_ok if e == 0]; r = [s for s, e in ex_ok if e == 1]
    res[nm]["executable_roots_semantic_flag(<7)"] = {"wrong_n": len(w), "wrong_flagged": sum(s < 7 for s in w), "recall": sum(s < 7 for s in w) / len(w), "right_n": len(r), "right_flagged": sum(s < 7 for s in r),
                                                     "precision": sum(s < 7 for s in w) / max(1, sum(s < 7 for s in w) + sum(s < 7 for s in r))}
dump("q2c_recall.json", res)
print(json.dumps(res, indent=1))
