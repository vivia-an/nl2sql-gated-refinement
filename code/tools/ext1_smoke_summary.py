#!/usr/bin/env python3
"""EXT1 smoke disclosure (SAP Amendment 8, section 21.12): one line per smoke arm (20 items) read from the pulled run directories, plus the verbatim
`[run] done` verdict line of each run and the list-price cost (cost_report convention: rows served from our own call cache are free).
stdlib only, Python >= 3.6.   python3 code/tools/ext1_smoke_summary.py --runs results/raw-ignored/EXT1S --out results/ext1/smoke_summary.json"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = {}
    for d in sorted(glob.glob(os.path.join(a.runs, "ext1s-*"))):
        name = os.path.basename(d)
        rows = [json.loads(l) for l in open(os.path.join(d, "preds.jsonl")) if l.strip()]
        n = len(rows)
        ts = [r.get("ex_ts") for r in rows]
        done_line = None
        lp = os.path.join(d, "_queue", "log")
        if os.path.exists(lp):
            for l in open(lp):
                if l.startswith("[run] done"):
                    done_line = l.strip()
        arms[name] = {"n": n, "ex_ts": sum(t or 0 for t in ts) / float(n), "ex_ts_n_scored": sum(1 for t in ts if t is not None),
                      "ex_plain": sum((r.get("ex") or 0) for r in rows) / float(n), "exec_ok": sum(1 for r in rows if r.get("exec_ok")) / float(n),
                      "calls_per_q": sum(r.get("calls", 0) for r in rows) / float(n),
                      "n_revised": sum(1 for r in rows if "revise" in (r.get("actions") or [])),
                      "errors": sum(1 for r in rows if r.get("error_class")), "done_line": done_line}
    cost = subprocess.check_output([sys.executable, os.path.join(HERE, "cost_report.py"), a.runs, "--by", "model"]).decode()
    total = [l for l in cost.splitlines() if l.startswith("TOTAL")]
    out = {"_meta": {"scope": "external-validity", "label": "smoke (disclosed, not an analysis; items stay in the full arms)", "split": "spider_realistic",
                     "items": "exp/queues/ext1_smoke_qids.txt (20 items, random.Random(20261127).sample over the 508 file positions)",
                     "cache": "separate machine-local call cache, so the full arms are not made cheaper by smoke hits"},
           "arms": arms, "cost_report_by_model": cost.splitlines(), "total_cny": float(total[0].split("=")[1]) if total else None}
    json.dump(out, open(a.out, "w"), indent=1, sort_keys=True)
    print("wrote", a.out, "arms:", len(arms), "total CNY", out["total_cny"])


if __name__ == "__main__":
    main()
