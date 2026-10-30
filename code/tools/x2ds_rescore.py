#!/usr/bin/env python3
"""Post hoc: re-score the retry arms of the X2DS / W7 / W8 blocks against the corrected reference queries, on the subset of the
robustness check (S_same of the primary variant G1, 360 questions). No model call.

    PYTHONPATH=code python3 code/tools/x2ds_rescore.py --w <WORK_DIR> --variant <arcwise_plat_sql_only_with_diff.json> --rescore results/robust/rescore.json \
        --gold-info <results/robust/gold_info.json> --out <OUT>/x2ds_rescore.json

The subset is rebuilt with `rescore_corrected.build_sets` (dev rows, corrected-release rows, db-discordant list of results/robust/rescore.json), as in
r3_q360.py. A prediction counts as correct when the order- and duplicate-insensitive digest of its result set (`rows_digest`, executed on the development
database with the 30 s limit of the scorer) equals the digest of the reference result stored by rescore_corrected.py (results/robust/gold_info.json, dev
database); a failing reference counts as wrong. The stored-run columns of results/robust/items_ex.csv.gz are used to check this scoring on the arms they hold.
Intervals: database-clustered percentile bootstrap (B = 10 000, seed 20261101), as in the robustness check."""
import argparse
import csv
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import rescore_corrected as RC  # noqa: E402

BK = ("plus", "flash", "27b", "dsv41flash")
CONTRASTS = (("zsretry", "zs"), ("zsretryempty", "zs"), ("gated", "zs"), ("gatedempty", "gated"), ("gatedempty", "zsretryempty"),
             ("gated", "zsretryempty"), ("zsretryempty", "zsretry"))


def arm_paths(W, bk):
    M, X = W + "/analysis_runs/MIN", W + "/runs"
    if bk == "dsv41flash":
        p = {"zs": X + "/X2DS/zs-dsv41flash-s0", "gated": X + "/X2DS/gated-dsv41flash-s0"}
    elif bk == "plus":
        p = {"zs": M + "/M1/zs-s0", "gated": M + "/M1/gated-s0"}
    else:
        p = {"zs": "%s/M5/zs-%s-s0" % (M, bk), "gated": "%s/M5/gated-%s-s0" % (M, bk)}
    p["zsretry"] = "%s/X2DSR/zsretry-%s-s0" % (X, bk)
    p["zsretryempty"] = "%s/W7/zsretryempty-%s-s0" % (X, bk)
    p["gatedempty"] = "%s/W8/gatedempty-%s-s0" % (X, bk)
    return p


def boot(dbs, d, B, seed=20261101):
    u = np.unique(dbs)
    mem = {x: np.where(dbs == x)[0] for x in u}
    rng = np.random.RandomState(seed)
    v = np.empty(B)
    for b in range(B):
        idx = np.concatenate([mem[x] for x in rng.choice(u, len(u), replace=True)])
        v[b] = d[idx].mean()
    return [100.0 * float(np.percentile(v, 2.5)), 100.0 * float(np.percentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--rescore", required=True)
    ap.add_argument("--gold-info", required=True)
    ap.add_argument("--items-ex", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=10000)
    a = ap.parse_args()
    from nl2sql.db import execute_robust
    from nl2sql.eval.score import rows_digest
    dev = RC.jload(os.path.join(a.w, "data", RC.DEV_JSON))
    dev_by_id = {int(r["question_id"]): r for r in dev}
    var = {int(r["question_id"]): r for r in RC.jload(a.variant)}
    discord = set(RC.jload(a.rescore)["variants"]["G1"]["db_discordant"])
    st = RC.build_sets(dev_by_id, var, discord)
    qs = sorted(st["same"])
    gi = RC.jload(a.gold_info)
    gold = {g: {q: (gi["%s|%s" % (g, q)]["dev"]["digest"] if gi["%s|%s" % (g, q)]["dev"]["ok"] else None) for q in qs} for g in ("orig", "G1")}
    db_of = {q: q.split(":")[1] for q in qs}
    dbs = np.array([db_of[q] for q in qs])
    dig_cache = {}

    def score(q, sql):
        k = (q, sql)
        if k not in dig_cache:
            p = os.path.join(a.w, "data", RC.DEV_DB_DIR, db_of[q], db_of[q] + ".sqlite")
            r = execute_robust(p, sql or "", timeout_s=30.0, max_rows=None) if (sql or "").strip() else {"ok": False}
            dig_cache[k] = rows_digest(r["rows"]) if r.get("ok") else None
        d = dig_cache[k]
        return {g: (1.0 if (d is not None and gold[g][q] is not None and d == gold[g][q]) else 0.0) for g in gold}

    out = {"_meta": {"scope": "posthoc", "n_S_same": len(qs), "n_dbs": int(len(np.unique(dbs))), "B": a.B, "variant": "G1", "db": "dev"}, "backbones": {}}
    check = {}
    if a.items_ex:
        lab = {"plus": "plus", "flash": "flash", "27b": "q27b"}
        rows = list(csv.DictReader(gzip.open(a.items_ex, "rt")))
        for r in rows:
            if r["kind"] == "final":
                check[(r["label"], r["qid"])] = (r["orig"], r["G1"])
    for bk in BK:
        P = arm_paths(a.w, bk)
        ex = {}
        for arm, d in P.items():
            fp = os.path.join(d, "preds.jsonl")
            if not os.path.exists(fp):
                continue
            pr = {}
            for line in open(fp):
                r = json.loads(line)
                pr[r["qid"]] = r
            sc = [score(q, pr[q].get("final_sql")) for q in qs]
            ex[arm] = {g: np.array([s[g] for s in sc]) for g in gold}
        res = {"arms": {k: {g: 100.0 * float(v[g].mean()) for g in gold} for k, v in ex.items()}, "contrasts": {}}
        if check and bk in ("plus", "flash", "27b"):
            lb = {"plus": "plus", "flash": "flash", "27b": "q27b"}[bk]
            agree = {}
            for arm in ("zs", "gated"):
                m = [(int(float(check[(lb + "." + arm, q)][0])) == int(ex[arm]["orig"][i]), int(float(check[(lb + "." + arm, q)][1])) == int(ex[arm]["G1"][i]))
                     for i, q in enumerate(qs) if (lb + "." + arm, q) in check and check[(lb + "." + arm, q)][0] not in ("", None)]
                agree[arm] = {"n": len(m), "orig_agree": sum(x for x, _ in m), "G1_agree": sum(y for _, y in m)}
            res["check_against_items_ex"] = agree
        for x, y in CONTRASTS:
            if x not in ex or y not in ex:
                continue
            d0, d1 = ex[x]["orig"] - ex[y]["orig"], ex[x]["G1"] - ex[y]["G1"]
            res["contrasts"]["%s_minus_%s" % (x, y)] = {"orig": 100.0 * float(d0.mean()), "orig_ci": boot(dbs, d0, a.B),
                                                        "corrected": 100.0 * float(d1.mean()), "corrected_ci": boot(dbs, d1, a.B),
                                                        "shift": 100.0 * float((d1 - d0).mean()), "shift_ci": boot(dbs, d1 - d0, a.B)}
        out["backbones"][bk] = res
        print(bk, json.dumps({k: "%+.2f -> %+.2f" % (v["orig"], v["corrected"]) for k, v in res["contrasts"].items()}), res.get("check_against_items_ex"), flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
