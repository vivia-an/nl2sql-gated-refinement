"""Post hoc (SAP Section 30): the harm of the empty-result trigger on the tuning split HR, where some reference results are empty. No model call.
Needs the nl2sql package and the BIRD training databases (gold execution of HR):
    python3 code/tools/x2ds_w9.py --runs <WORK_DIR>/runs/W9 --out <OUT>/w9_harm.json
Per backbone: single call (zs), single call + retry (zsretry, imports zs) and empty-gate retry (zsretryempty, imports zs and zsretry), so that the last two differ only by
the revisions an empty result triggers. Empty reference = no rows, or one row of NULLs (the gate treats it as empty). Harm = right under zsretry, wrong under
zsretryempty, among the questions with an empty reference. EX differences with a 95% database-clustered percentile bootstrap interval (2,000 draws, seed 0)."""
import argparse
import json
import random

from nl2sql import data
from nl2sql.db import execute_robust

BK = ("plus", "flash", "27b", "dsv41flash")


def preds(p):
    return {json.loads(l)["qid"]: json.loads(l) for l in open(p)}


def boot(items, diff, n=2000, seed=0):
    by = {}
    for q, db in items:
        by.setdefault(db, []).append(diff[q])
    dbs = sorted(by)
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        s = [x for _ in dbs for x in by[rnd.choice(dbs)]]
        out.append(100.0 * sum(s) / len(s))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qids", required=True, help="exp/posthoc/w9_qids.txt: HR without bike_share_1")
    ap.add_argument("--bk", default=",".join(BK), help="comma-separated backbones (default: all four)")
    a = ap.parse_args()
    keep = set(open(a.qids).read().split())
    items = [it for it in data.load_split("bird_train_HR", with_gold=True) if it.qid in keep]
    empty, nullrow, fail = set(), set(), []
    for it in items:
        g = execute_robust(it.db_path, it.gold_sql, timeout_s=120, max_rows=2)   # two rows decide empty / one NULL row; no full result in memory
        if not g or not g.get("ok"):
            fail.append(it.qid)
            continue
        rows = g.get("rows") or []
        if not rows:
            empty.add(it.qid)
        elif len(rows) == 1 and all(c is None for c in rows[0]):
            nullrow.add(it.qid)
    E0 = empty | nullrow
    qdb = [(it.qid, it.db_id) for it in items]
    out = {"_meta": {"scope": "posthoc", "sap": "Section 30", "split": "bird_train_HR"}, "n": len(items), "n_db": len({d for _, d in qdb}),
           "gold_empty": len(empty), "gold_null_row": len(nullrow), "gold_fail": len(fail), "backbones": {}}
    for b in a.bk.split(","):
        Z, R, E = (preds("%s/%s-%s-s0/preds.jsonl" % (a.runs, x, b)) for x in ("zs", "zsretry", "zsretryempty"))
        qs = [q for q, _ in qdb]
        assert all(q in Z and q in R and q in E for q in qs), b
        ex = lambda P: 100.0 * sum(P[q]["ex"] for q in qs) / len(qs)
        d_er = {q: E[q]["ex"] - R[q]["ex"] for q in qs}
        d_ez = {q: E[q]["ex"] - Z[q]["ex"] for q in qs}
        fired = [q for q in qs if E[q]["final_sql"] != R[q]["final_sql"] or len(E[q].get("actions") or []) != len(R[q].get("actions") or [])]
        r = {"ex_zs": ex(Z), "ex_zsretry": ex(R), "ex_zsretryempty": ex(E),
             "dex_er": sum(d_er.values()) * 100.0 / len(qs), "dex_er_ci": boot(qdb, d_er), "dex_ez": sum(d_ez.values()) * 100.0 / len(qs), "dex_ez_ci": boot(qdb, d_ez),
             "up_er": sum(1 for q in qs if d_er[q] > 0), "down_er": sum(1 for q in qs if d_er[q] < 0),
             "up_ez": sum(1 for q in qs if d_ez[q] > 0), "down_ez": sum(1 for q in qs if d_ez[q] < 0),
             "n_empty_ref": len(E0), "right_r_empty_ref": sum(R[q]["ex"] for q in E0), "right_e_empty_ref": sum(E[q]["ex"] for q in E0),
             "harm": sum(1 for q in E0 if R[q]["ex"] > E[q]["ex"]),   # as scored; x2ds_w9_iv.py splits it into rewritten queries (harm) and scoring flips "repair_empty_ref": sum(1 for q in E0 if E[q]["ex"] > R[q]["ex"]),
             "harm_vs_zs": sum(1 for q in E0 if Z[q]["ex"] > E[q]["ex"]),
             "down_er_outside_empty_ref": sum(1 for q in qs if d_er[q] < 0 and q not in E0),
             "n_differ": len(fired), "harm_qids": sorted(q for q in E0 if R[q]["ex"] > E[q]["ex"])}
        # where the trigger can act: the final answer of the plain retry is empty (otherwise the two arms follow the same path, so any difference is
        # the scoring of the same query on two executions, or a timeout of the root on the large databases)
        F = [q for q in qs if R[q].get("final_status") == "ok_empty"]
        O = [q for q in qs if R[q].get("final_status") != "ok_empty"]
        r.update({"n_fired": len(F), "up_fired": sum(1 for q in F if d_er[q] > 0), "down_fired": sum(1 for q in F if d_er[q] < 0),
                  "up_out": sum(1 for q in O if d_er[q] > 0), "down_out": sum(1 for q in O if d_er[q] < 0),
                  "flips_same_sql_out": sum(1 for q in O if d_er[q] != 0 and E[q]["final_sql"] == R[q]["final_sql"])})
        r["net_fired_pp"] = 100.0 * (r["up_fired"] - r["down_fired"]) / len(qs)
        # how the model answered the revision request on a correct empty answer: kept the query (no harm) or changed it
        ce = [q for q in E0 if R[q]["ex"] == 1]
        r["correct_empty_kept"] = sum(1 for q in ce if E[q]["final_sql"] == R[q]["final_sql"])
        r["harm_per_100_empty_ref"] = 100.0 * r["harm"] / len(E0) if E0 else None
        r["harm_per_100_correct_empty"] = 100.0 * r["harm"] / len(ce) if ce else None
        out["backbones"][b] = r
        print(b, {k: v for k, v in r.items() if not k.endswith("qids")}, flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
