#!/usr/bin/env python3
"""R3 / E-C: same-examples single-call control. `zs_ex` = the single call of `zs_ours` plus exactly the few-shot examples the root-only arm of the same
backbone and seed retrieved for the same question (nothing else). Post hoc, descriptive (SAP section 26).

    worker machine:  PYTHONPATH=<code> python code/tools/r3_samex.py --runs $R/runs --out results/r3 [--B 10000]

Per backbone and run (s0 = T 0, the headline run, reported apart; s101 / s202 / s303 = T 0.7, the replication, whose mean is the estimate; the two are never
averaged together): EX of zs, zs_ex, root-only; contrasts
    ex_minus_zs     what the examples alone add to the single call
    root_minus_ex   what the rest of the root-only pipeline (graph, hints, terms, lineage, routed context) adds on top of the examples
    stack           root-only minus single call (the E-S stack contrast)
and the share of the stack that the examples carry (defined only when the stack interval excludes 0). Estimate averaged over the three T = 0.7 runs with the
database-clustered and the item-level intervals, the between-run range and the t interval, as in r3_seeds, and the reading of the seed-0 run. No model call."""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3_common as C  # noqa: E402
from r3_common import S  # noqa: E402
import r3_seeds as RS  # noqa: E402

ARMS3 = (("zs", lambda b, s: C.run_path(b, "zs", s)), ("zsex", lambda b, s: C.EXRUN[(b, s)]), ("rootonly", lambda b, s: C.run_path(b, "rootonly", s)))
CON = (("ex_minus_zs", "zsex", "zs"), ("root_minus_ex", "rootonly", "zsex"), ("stack", "rootonly", "zs"))


def run(runs_root, B):
    rows, note = {}, {}
    for b in C.BK:
        for nm, fn in ARMS3:
            for s in C.SEEDS:
                try:
                    r = C.load_preds(runs_root, fn(b, s))
                except KeyError:
                    continue
                ok, why = C.check_run(r)
                note["%s|%s|%d" % (b, nm, s)] = why
                if ok:
                    rows[(b, nm, s)] = r
    common = None
    for r in rows.values():
        q = set(k for k, x in r.items() if C.ex_of(x) is not None)
        common = q if common is None else common & q
    qids = sorted(common)
    first = next(iter(rows.values()))
    dbs = np.array([first[q]["db_id"] for q in qids])
    ex = {k: np.array([C.ex_of(r[q]) for q in qids], float) for k, r in rows.items()}
    boot = S.ClusterBoot(dbs, B=B)
    out = {"arms": {}, "contrasts": {}, "share": {}, "notes": {"n_qids": len(qids), "run_notes": note}}
    for b in C.BK:
        for nm, _ in ARMS3:
            ss = [s for s in C.REP if (b, nm, s) in ex]
            if ss:
                v = [100.0 * ex[(b, nm, s)].mean() for s in ss]
                e = {"per": dict(zip(ss, v)), "mean": float(np.mean(v)), "n_runs": len(ss), "s0": (100.0 * ex[(b, nm, 0)].mean()) if (b, nm, 0) in ex else None}
                if len(ss) >= 2:
                    e["sd"] = float(np.std(v, ddof=1))
                out["arms"]["%s|%s" % (b, nm)] = e
    per = {}
    for cid, x, y in CON:
        for b in C.BK:
            ss = [s for s in C.SEEDS if (b, x, s) in ex and (b, y, s) in ex]
            if not [s for s in ss if s in C.REP]:
                continue
            d = {s: ex[(b, x, s)] - ex[(b, y, s)] for s in ss}
            per[(cid, b)] = d
            out["contrasts"]["%s|%s" % (cid, b)] = RS.contrast_stats(d, dbs, boot, B)
        if all((cid, b) in per for b in C.BK):
            ss = sorted(set.intersection(*[set(per[(cid, b)]) for b in C.BK]))
            if [s for s in ss if s in C.REP]:
                d = {s: np.mean([per[(cid, b)][s] for b in C.BK], axis=0) for s in ss}
                out["contrasts"]["%s|pooled3" % cid] = RS.contrast_stats(d, dbs, boot, B)
    # share of the stack carried by the examples
    for sc in ("plus", "flash", "27b", "pooled3"):
        a, st = out["contrasts"].get("ex_minus_zs|%s" % sc), out["contrasts"].get("stack|%s" % sc)
        if not a or not st:
            continue
        defined = bool(st["ci"][0] > 0 or st["ci"][1] < 0)
        e = {"defined": defined, "stack": st["est"], "examples": a["est"], "share": (a["est"] / st["est"]) if st["est"] else None}
        if st.get("s0") and a.get("s0"):
            e["s0"] = {"stack": st["s0"]["est"], "examples": a["s0"]["est"], "defined": bool(st["s0"]["ci"][0] > 0 or st["s0"]["ci"][1] < 0),
                       "share": (a["s0"]["est"] / st["s0"]["est"]) if st["s0"]["est"] else None}
        if defined:
            bs = [per[("ex_minus_zs", b)] for b in (C.BK if sc == "pooled3" else (sc,))]
            ss = sorted(s for s in set.intersection(*[set(p) for p in bs]) if s in C.REP)
            av = np.mean([np.mean([p[s] for s in ss], axis=0) for p in bs], axis=0)
            bt = [per[("stack", b)] for b in (C.BK if sc == "pooled3" else (sc,))]
            sv = np.mean([np.mean([p[s] for s in ss], axis=0) for p in bt], axis=0)
            ab, sb = boot.means(av), boot.means(sv)
            e["share_ci"] = S.pctl_ci(ab / np.where(sb == 0, np.nan, sb))
        out["share"][sc] = e
    # discordance between zs_ex and root-only (how much do the two differ per question)
    out["discord"] = {}
    for b in C.BK:
        for s in C.SEEDS:
            if (b, "zsex", s) in ex and (b, "rootonly", s) in ex:
                a_, r_ = ex[(b, "zsex", s)], ex[(b, "rootonly", s)]
                out["discord"]["%s|%d" % (b, s)] = {"rate_pct": 100.0 * float((a_ != r_).mean()), "zsex_only": int(((a_ == 1) & (r_ == 0)).sum()), "root_only": int(((a_ == 0) & (r_ == 1)).sum())}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=C.B_DEFAULT)
    a = ap.parse_args(argv)
    res = run(a.runs, a.B)

    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "samex.json"), "w") as f:
        json.dump(conv(res), f, indent=1)
    with open(os.path.join(a.out, "samex_contrasts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "scope", "n_runs", "est_s101", "est_s202", "est_s303", "mean_T07", "cl_lo", "cl_hi", "item_lo", "item_hi", "range_lo", "range_hi", "wide_lo", "wide_hi",
                    "reading_cluster", "reading_wide", "s0_est", "s0_reading_cluster", "s0_reading_wide", "agrees_with_s0_cluster", "agrees_with_s0_wide"])
        for k, e in res["contrasts"].items():
            c, sc = k.split("|")
            s0, ag = e.get("s0"), e.get("agrees")
            w.writerow([c, sc, e["n_runs"]] + [("%.3f" % e["per"][s]) if s in e["per"] else "" for s in C.REP] +
                       ["%.3f" % e["est"], "%.3f" % e["ci"][0], "%.3f" % e["ci"][1], "%.3f" % e["item_ci"][0], "%.3f" % e["item_ci"][1], "%.3f" % e["range"][0], "%.3f" % e["range"][1],
                        "%.3f" % e["wide"][0], "%.3f" % e["wide"][1], C.reading(*e["ci"]), C.reading(*e["wide"]),
                        "%.3f" % s0["est"] if s0 else "", C.reading(*s0["ci"]) if s0 else "", C.reading(*s0["wide"]) if s0 else "",
                        ("yes" if ag["cluster"] else "no") if ag else "", ("yes" if ag["wide"] else "no") if ag else ""])
    print("[r3_samex] qids %d, contrasts %d" % (res["notes"]["n_qids"], len(res["contrasts"])))


if __name__ == "__main__":
    main()
