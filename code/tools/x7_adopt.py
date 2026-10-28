#!/usr/bin/env python3
"""Mechanical adoption check of the X7 candidates (value probing, adaptive path) against full_gated on HR and HT-300 (numpy only; tuning data).

  x7_adopt.py --ref-hr R/runs/X5/1b-hr-gated --ref-ht R/runs/X5/1b-ht-gated
              --cand probe=R/runs/X7/probe-hr,R/runs/X7/probe-ht --cand adaptive=R/runs/X7/adaptive-hr,R/runs/X7/adaptive-ht
              [--boot 10000] [--out exp/x7_adoption.json]

Rule (X7 brief): ADOPT a candidate iff
  (HT)  paired ΔEX = EX(candidate) - EX(full_gated) on HT-300: lower 95 % bound of the database-clustered percentile bootstrap (B = 10 000) > 0, AND
  (HR)  consistent sign: ΔEX point estimate > 0 and no more HR databases with a negative difference than with a positive one, at most one negative.
The cost is reported next to the verdict, never part of it: calls ratio R = sum calls(candidate) / sum calls(full_gated) on each split (the adaptive path
costs about 1.5 x the gated loop by construction, so it can never satisfy SAP P1b's R < 0.80; if it passes the EX rule it is a different, more accurate
and more expensive system, not a replacement of `full`).  Pairing is on qid; rows with ex null in either run are dropped (counted).  With 4 databases
in HT and ~300 questions the HT bound can only be positive for an effect of a few points: a miss is not evidence of no effect (power note in the output)."""
import argparse
import json
import os

import numpy as np

from x7_common import load_preds, paired


def per_db(a, b, qids, dbs):
    out = {}
    for d in sorted(set(dbs)):
        ix = [i for i, x in enumerate(dbs) if x == d]
        da = np.array([a[qids[i]] for i in ix], float)
        db_ = np.array([b[qids[i]] for i in ix], float)
        out[d] = {"n": len(ix), "diff_pp": 100 * float((db_ - da).mean())}
    return out


def compare(ref_dir, cand_dir, boot):
    A, B = load_preds(ref_dir), load_preds(cand_dir)
    qids = sorted(q for q in A if q in B and A[q].get("ex") is not None and B[q].get("ex") is not None)
    dbs = [A[q]["db_id"] for q in qids]
    ea = {q: A[q]["ex"] for q in qids}
    eb = {q: B[q]["ex"] for q in qids}
    res = paired(dbs, [ea[q] for q in qids], [eb[q] for q in qids], boot)
    res["dropped"] = {"only_ref": len(set(A) - set(B)), "only_cand": len(set(B) - set(A)), "ex_null": sum(1 for q in A if q in B and (A[q].get("ex") is None or B[q].get("ex") is None))}
    res["per_db"] = per_db(ea, eb, qids, dbs)
    ca, cb = sum(A[q].get("calls", 0) for q in qids), sum(B[q].get("calls", 0) for q in qids)
    res["calls_per_q_ref"], res["calls_per_q_cand"] = ca / float(max(len(qids), 1)), cb / float(max(len(qids), 1))
    res["calls_ratio"] = cb / float(ca) if ca else None
    return res


def verdict(hr, ht):
    pos = sum(1 for v in hr["per_db"].values() if v["diff_pp"] > 1e-9)
    neg = sum(1 for v in hr["per_db"].values() if v["diff_pp"] < -1e-9)
    ht_ok = bool(ht["ci95_pp"] and ht["ci95_pp"][0] > 0)
    hr_ok = bool(hr["diff_pp"] > 0 and neg <= 1 and neg <= pos)
    return {"ht_ci_lower_gt_0": ht_ok, "hr_sign_consistent": hr_ok, "hr_dbs_positive": pos, "hr_dbs_negative": neg, "adopt": ht_ok and hr_ok}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref-hr", required=True)
    ap.add_argument("--ref-ht", required=True)
    ap.add_argument("--cand", action="append", required=True, help="NAME=HR_RUN,HT_RUN")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = {}
    for spec in a.cand:
        name, paths_ = spec.split("=", 1)
        hr_dir, ht_dir = paths_.split(",")
        hr, ht = compare(a.ref_hr, hr_dir, a.boot), compare(a.ref_ht, ht_dir, a.boot)
        v = verdict(hr, ht)
        out[name] = {"HR": hr, "HT": ht, "verdict": v, "inputs": {"ref_hr": a.ref_hr, "ref_ht": a.ref_ht, "cand_hr": hr_dir, "cand_ht": ht_dir}}
        print("%-10s HT dEX %+.2f pp CI [%s] (W/L %d/%d, calls ratio %.2f) | HR dEX %+.2f pp CI [%s] (W/L %d/%d, dbs +%d/-%d, calls ratio %.2f) -> %s"
              % (name, ht["diff_pp"], ", ".join("%+.2f" % x for x in ht["ci95_pp"]) if ht["ci95_pp"] else "n/a", ht["b_right_a_wrong"], ht["a_right_b_wrong"], ht["calls_ratio"] or float("nan"),
                 hr["diff_pp"], ", ".join("%+.2f" % x for x in hr["ci95_pp"]) if hr["ci95_pp"] else "n/a", hr["b_right_a_wrong"], hr["a_right_b_wrong"],
                 v["hr_dbs_positive"], v["hr_dbs_negative"], hr["calls_ratio"] or float("nan"), "ADOPT" if v["adopt"] else "do not adopt"))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
    return out


if __name__ == "__main__":
    main()
