#!/usr/bin/env python3
"""Paired comparison of two runs on the same questions (X5 live tests; numpy only).

  paired_compare.py A/preds.jsonl B/preds.jsonl [--db beer_factory[,db2]] [--boot 2000] [--json out.json] [--label-a P0] [--label-b gated]

Pairs on qid (rows with ex null or a missing partner are dropped and counted). Reports per run: EX, calls/q, share of final answers that do
not execute, EX of the accepted answers (abstentions excluded) and coverage, and - when `cands.jsonl` sits next to preds.jsonl - the status mix
and EX of the ROOT candidate (cand_idx 0; this is the number HEADROOM row 1a is about: execution errors of the first generation).
Then the paired difference B - A in EX (pp) with a 95 % database-clustered bootstrap CI (databases resampled with replacement), and the
discordant counts (B right / A wrong, A right / B wrong). A difference below ~2 pp on ~1 100 questions is noise (HEADROOM section 1)."""
import argparse
import json
import os
import sys

import numpy as np


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["qid"]] = r
    return rows


def load_roots(preds_path):
    p = os.path.join(os.path.dirname(os.path.abspath(preds_path)), "cands.jsonl")
    out = {}
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    if c.get("cand_idx") == 0:
                        out[c["qid"]] = c
    return out


def run_stats(rows, roots, qids):
    rs = [rows[q] for q in qids]
    n = max(len(rs), 1)
    acc = [r for r in rs if not r.get("abstained")]
    out = {"n": len(rs), "ex": float(np.mean([r["ex"] for r in rs])) if rs else None,
           "calls_per_q": float(np.mean([r.get("calls", 0) for r in rs])) if rs else None,
           "final_not_executing": sum(1 for r in rs if not r.get("exec_ok")) / float(n),
           "coverage": len(acc) / float(n), "ex_accepted": float(np.mean([r["ex"] for r in acc])) if acc else None}
    if roots:
        rr = [roots[q] for q in qids if q in roots]
        if rr:
            st = {}
            for c in rr:
                st[c.get("status")] = st.get(c.get("status"), 0) + 1
            exs = [c["ex"] for c in rr if c.get("ex") is not None]
            out["root"] = {"n": len(rr), "status": st, "exec_error_or_timeout": sum(v for k, v in st.items() if k and (k.startswith("err") or k == "timeout")) / float(len(rr)),
                           "ex": float(np.mean(exs)) if exs else None}
    return out


def cluster_ci(dbs, d, B, seed=20261101):
    rng = np.random.RandomState(seed)
    u = np.unique(dbs)
    mem = {x: np.where(dbs == x)[0] for x in u}
    vals = []
    for _ in range(B):
        idx = np.concatenate([mem[x] for x in rng.choice(u, len(u), replace=True)])
        vals.append(d[idx].mean())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def compare(a_path, b_path, dbs_only=None, boot=2000, label_a="A", label_b="B"):
    A, B = load(a_path), load(b_path)
    qids = [q for q in A if q in B and A[q].get("ex") is not None and B[q].get("ex") is not None]
    if dbs_only:
        qids = [q for q in qids if A[q].get("db_id") in dbs_only]
    res = {"n_pairs": len(qids), "dropped": {"only_in_a": len(set(A) - set(B)), "only_in_b": len(set(B) - set(A)),
                                             "ex_null": sum(1 for q in A if q in B and (A[q].get("ex") is None or B[q].get("ex") is None))},
           label_a: run_stats(A, load_roots(a_path), qids), label_b: run_stats(B, load_roots(b_path), qids)}
    if qids:
        ea = np.array([A[q]["ex"] for q in qids], float)
        eb = np.array([B[q]["ex"] for q in qids], float)
        dbs = np.array([A[q].get("db_id") or q.split(":")[1] for q in qids])
        d = eb - ea
        res["diff_pp"] = {"b_minus_a": 100 * float(d.mean()), "ci95": [100 * x for x in cluster_ci(dbs, d, boot)] if len(np.unique(dbs)) > 1 else None,
                          "b_right_a_wrong": int(((eb == 1) & (ea == 0)).sum()), "a_right_b_wrong": int(((ea == 1) & (eb == 0)).sum()),
                          "n_databases": int(len(np.unique(dbs)))}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--db", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = compare(a.a, a.b, set(a.db.split(",")) if a.db else None, a.boot, a.label_a, a.label_b)
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    return res


if __name__ == "__main__":
    main()
