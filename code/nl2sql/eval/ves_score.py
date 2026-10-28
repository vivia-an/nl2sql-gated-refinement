"""Post-hoc VES / R-VES scorer for a finished run (SPEC section 5: run it as an isolated evaluator job, one process,
no other heavy task on the worker machine; API latency plays no role because only SQL execution is timed).

  python -m nl2sql.eval.ves_score --preds <run>/preds.jsonl --split bird_dev [--iters 5] [--timing both]
         [--trim sigma3|minmax] [--out <run>/ves.jsonl] [--gold-times-cache F.json] [--limit N] [--qids-file F]

Reads preds.jsonl (final_sql, ex), times only the EX-correct items (others contribute 0), writes one row per item to
ves.jsonl (resumable: finished qids are skipped) and <out>.summary.json. `analysis.summarize.summarize_run` picks
up a ves.jsonl lying next to preds.jsonl (keys ves, rves, ves_ci, ves_coverage)."""
import argparse
import json
import os
import sys
import time

from .. import data
from .ves import aggregate, item_rves, item_ves, measure_ratio


def load_rows(path):
    rows = []
    if os.path.exists(path):
        for line in open(path):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def summarize_ves(ves_path, preds_path=None):
    """-> dict(n, n_ex1, ves, rves, ves_ci, rves_ci, coverage, timing_failures, trim_fallbacks, load1_mean)."""
    from ..analysis.summarize import cluster_bootstrap_ci
    rows = load_rows(ves_path)
    if not rows:
        return {"n": 0}
    out = {"n": len(rows), "n_ex1": sum(1 for r in rows if r.get("ex") == 1)}
    for key in ("ves", "rves"):
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if vals:
            m, lo, hi = cluster_bootstrap_ci(vals, [r["db_id"] for r in rows if r.get(key) is not None])
            out[key], out[key + "_ci"] = m, [lo, hi]
    out["timing_failures"] = sum(1 for r in rows if r.get("ex") == 1 and (r.get("fail_official") or r.get("fail_minidev")))
    out["trim_fallbacks"] = sum(1 for r in rows if r.get("trim_fallback"))
    loads = [r["load1"] for r in rows if r.get("load1") is not None]
    out["load1_mean"] = sum(loads) / len(loads) if loads else None
    if preds_path and os.path.exists(preds_path):
        n_preds = len(load_rows(preds_path))
        out["coverage"] = len(rows) / n_preds if n_preds else None
    return out


def _load_cache(path):
    if path and os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            return {}
    return {}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--timing", default="both", choices=["both", "official", "minidev"])
    ap.add_argument("--trim", default="sigma3", choices=["sigma3", "minmax"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--gold-times-cache", default=None, help="json file: reuse gold timings across systems (SPEC: measured once)")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--qids-file", default=None)
    args = ap.parse_args(argv)

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.preds)), "ves.jsonl")
    preds = load_rows(args.preds)
    if args.qids_file:
        keep = {l.strip() for l in open(args.qids_file) if l.strip()}
        preds = [r for r in preds if r["qid"] in keep]
    if args.limit:
        preds = preds[: args.limit]
    done = {r["qid"] for r in load_rows(out_path)}
    todo = [r for r in preds if r["qid"] not in done]
    items = {i.qid: i for i in data.load_split(args.split)}
    gold = data.load_gold(args.split, run_id="ves_score:" + os.path.basename(os.path.dirname(os.path.abspath(args.preds))))
    modes = ["official", "minidev"] if args.timing == "both" else [args.timing]
    cache = _load_cache(args.gold_times_cache)
    print("[ves] n=%d todo=%d iters=%d timing=%s trim=%s load1=%.2f" % (len(preds), len(todo), args.iters, modes, args.trim,
                                                                      os.getloadavg()[0]), flush=True)
    t0 = time.time()
    with open(out_path, "a") as fo:
        for k, r in enumerate(todo, 1):
            it = items.get(r["qid"])
            g = gold.get(r["qid"], "")
            row = {"qid": r["qid"], "db_id": r["db_id"], "ex": r.get("ex"), "iters": args.iters, "trim": args.trim,
                   "load1": round(os.getloadavg()[0], 2)}
            for mode in modes:
                tag = "ves" if mode == "official" else "rves"
                ratio, fail = 0.0, None
                if r.get("ex") == 1 and it is not None and g and r.get("final_sql"):
                    ck = "|".join([it.db_path, g, mode, str(args.iters)])
                    gt = cache.get(ck)
                    m = measure_ratio(it.db_path, r["final_sql"], g, args.iters, mode, args.trim, args.timeout, gold_times=gt)
                    if gt is None and not m["fail"] and args.gold_times_cache:
                        cache[ck] = m["gold_times"]
                    ratio, fail = m["time_ratio"], m["fail"]
                    row["trim_fallback"] = bool(row.get("trim_fallback") or m["trim_fallback"])
                row[tag + "_ratio"] = ratio
                row[tag] = item_ves(ratio) if mode == "official" else item_rves(ratio)
                row["fail_" + mode] = fail
            fo.write(json.dumps(row) + "\n")
            fo.flush()
            if k % 50 == 0:
                print("[ves] %d/%d %.0fs" % (k, len(todo), time.time() - t0), flush=True)
    if args.gold_times_cache:
        tmp = args.gold_times_cache + ".tmp"
        json.dump(cache, open(tmp, "w"))
        os.replace(tmp, args.gold_times_cache)
    s = summarize_ves(out_path, args.preds)
    json.dump(s, open(out_path.replace(".jsonl", ".summary.json"), "w"), indent=1)
    print("[ves] done", json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
