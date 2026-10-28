#!/usr/bin/env python3
"""R3: cost of the new runs, CNY, from runs/R3*/**/usage*.jsonl with the rules of cost_report.py (rows with cached=true are not billed; implicit-cache tokens at
cache_hit_ratio of exp/prices.yaml; embedding / rerank requests are not priced, see RV2 cost.embed_rerank_included). Output results/r3/cost.json:
total (discounted and at list price), per model, per arm, number of billed chat calls, plus the throw-away probes of r3_probe (not logged: max_tokens 1)."""
import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cost_report import load_prices  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dirs", default="R3,R3smoke")
    a = ap.parse_args(argv)
    P = load_prices(a.prices)
    ratio, bratio = P["cache_hit_ratio"], P["batch_ratio"]
    tot = collections.defaultdict(lambda: collections.Counter())
    by_run = collections.defaultdict(lambda: collections.Counter())
    unpriced = collections.Counter()
    for d in a.dirs.split(","):
        for f in sorted(glob.glob(os.path.join(a.runs, d, "**", "usage*.jsonl"), recursive=True)):
            run = os.path.relpath(os.path.dirname(f), a.runs)
            for line in open(f):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("cached") or r.get("kind") != "chat":
                    if r.get("kind") != "chat" and not r.get("cached"):
                        tot["_nonchat"]["rows"] += 1
                    continue
                m = r.get("model", "?")
                pr = P["models"].get(m)
                pt, ct = r.get("pt", 0) or 0, r.get("ct", 0) or 0
                if pr is None:
                    unpriced[m] += 1
                    continue
                cached = min(r.get("cached_tokens", 0) or 0, pt)
                mult = bratio if r.get("mode") == "batch" else 1.0
                disc = mult * ((pt - cached) * pr["in"] + cached * pr["in"] * ratio + ct * pr["out"]) / 1e6
                nod = mult * (pt * pr["in"] + ct * pr["out"]) / 1e6
                for t in (tot[m], by_run[run]):
                    t["calls"] += 1
                    t["pt"] += pt
                    t["ct"] += ct
                    t["cached"] += cached
                    t["cny_disc"] += disc
                    t["cny_nodisc"] += nod
    out = {"dirs": a.dirs.split(","), "price_file": a.prices, "cache_hit_ratio": ratio, "unpriced_rows": dict(unpriced),
           "total": {"cny_disc": sum(v["cny_disc"] for k, v in tot.items() if not k.startswith("_")),
                     "cny_nodisc": sum(v["cny_nodisc"] for k, v in tot.items() if not k.startswith("_")),
                     "calls": sum(v["calls"] for k, v in tot.items() if not k.startswith("_"))},
           "by_model": {k: dict(v) for k, v in tot.items() if not k.startswith("_")},
           "by_run": {k: dict(v) for k, v in sorted(by_run.items())},
           "s303": {"cny_disc": sum(v["cny_disc"] for k, v in by_run.items() if k.endswith("-s303")), "cny_nodisc": sum(v["cny_nodisc"] for k, v in by_run.items() if k.endswith("-s303")),
                    "calls": sum(v["calls"] for k, v in by_run.items() if k.endswith("-s303")), "runs": sum(1 for k in by_run if k.endswith("-s303"))},
           "note": "embedding and rerank requests are not priced (placeholder prices, tokens not logged); the 1-token probes of r3_probe are not logged"}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "cost.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("[r3_cost] total CNY (discounted) %.2f, list %.2f, calls %d" % (out["total"]["cny_disc"], out["total"]["cny_nodisc"], out["total"]["calls"]))


if __name__ == "__main__":
    main()
