#!/usr/bin/env python3
"""Mechanical choice of the baseline configuration among the HT-300 second-pass trials (the freeze checklist (not released) step 7). Stdlib only, Python >= 3.6.

  python3 code/tools/pick_trial.py --runs $R/runs/MINT --b-ref 2.02 [--write exp/baseline_decisions.json]

Reads <runs>/<baseline>-t<k>/summary.json (ex, calls_per_q). Rule written BEFORE the trials are looked at (SPEC section 4 compute matching + equal effort):
  window = [0.85, 1.15] * B_ref  (B_ref = live HR mean calls/q of the primary system `full`: 2.02 for full_gated, 3.29 for full_fixed; measured in X5)
  choose the trial with the highest HT-300 EX among those whose measured calls/q lies in the window; ties -> fewer calls -> lower trial number;
  if no trial lies in the window the baseline is flagged OUT_OF_WINDOW and the trial closest in calls is taken (it is then reported as 'not budget-matched'
  and is not eligible for B*, SAP 13.3 P2).
Prints the --baseline-trials argument of code/tools/freeze_configs.py."""
import argparse
import glob
import json
import os
import re
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--b-ref", type=float, required=True)
    ap.add_argument("--lo", type=float, default=0.85)
    ap.add_argument("--hi", type=float, default=1.15)
    ap.add_argument("--write", default=None)
    a = ap.parse_args()
    lo, hi = a.lo * a.b_ref, a.hi * a.b_ref
    trials = {}
    for d in sorted(glob.glob(os.path.join(a.runs, "*-t[0-9]*"))):
        m = re.match(r"^(.+)-t(\d+)$", os.path.basename(d))
        f = os.path.join(d, "summary.json")
        if not m or not os.path.exists(f):
            continue
        s = json.load(open(f))
        if s.get("sys_error") or s.get("api_fail"):
            print("WARNING: %s has sys_error=%s api_fail=%s: rerun before choosing" % (d, s.get("sys_error"), s.get("api_fail")))
        trials.setdefault(m.group(1), []).append({"t": int(m.group(2)), "ex": s["ex"], "calls": s["calls_per_q"], "tokens": s.get("prompt_tokens_per_q"), "n": s["n"]})
    dec, arg = {}, []
    print("window = [%.2f, %.2f] calls/q (B_ref %.2f)" % (lo, hi, a.b_ref))
    for b, ts in sorted(trials.items()):
        print("\n%s" % b)
        for t in sorted(ts, key=lambda x: x["t"]):
            t["in_window"] = lo <= t["calls"] <= hi
            print("  t%d  n=%d  EX %.4f  calls/q %.3f  prompt tok/q %s  %s" % (t["t"], t["n"], t["ex"], t["calls"], t["tokens"], "in window" if t["in_window"] else "outside"))
        inside = [t for t in ts if t["in_window"]]
        if inside:
            best = sorted(inside, key=lambda x: (-x["ex"], x["calls"], x["t"]))[0]
            flag = "ok"
        else:
            best = sorted(ts, key=lambda x: (abs(x["calls"] - a.b_ref), x["t"]))[0]
            flag = "OUT_OF_WINDOW"
        print("  -> t%d (%s)" % (best["t"], flag))
        dec[b] = {"trial": best["t"], "ex": best["ex"], "calls_per_q": best["calls"], "status": flag, "b_ref": a.b_ref, "n_trials": len(ts)}
        arg.append("%s=%d" % (b, best["t"]))
    print("\n--baseline-trials %s" % ",".join(arg))
    if a.write:
        json.dump({"date": time.strftime("%F %T"), "rule": "highest HT-300 EX inside [0.85,1.15]*B_ref; ties fewer calls", "decisions": dec}, open(a.write, "w"), indent=1)
        print("wrote", a.write)


if __name__ == "__main__":
    main()
