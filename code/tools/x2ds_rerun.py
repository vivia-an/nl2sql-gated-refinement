#!/usr/bin/env python3
"""X2DSRR analysis (post hoc, W7; no model call): reading lines of the decision-ladder steps from one rerun of every arm they use.
    python code/tools/x2ds_rerun.py --w <host work tree | WORK_DIR> --out <file.json> [--B 10000]
Locked pair s0: the stored runs of x2ds_run_analysis.sh. Rerun pair r1: Qwen zs / gated / fixed = R4T0 reruns (analysis_runs/R4T0), zs_retry = X2DSRR rerun on the R4T0 zs roots;
dsv41flash zs / gated / fixed / zs_retry = X2DSRR. Per arm, sd_net = 100 sqrt(b + c) / n from the s0-versus-r1 discordant counts (r4_t0.contrast_runs); per step x over y, the reading
line = 2 sqrt(sd_net_x^2 + sd_net_y^2) (independence formula, the paper's deciding scale) and the direct line = 2 x 100 sqrt(sum (d1 - d0)^2) / n (bound, no independence across arms).
Also the step in the rerun alone (estimate, four intervals, widest reading). Needs numpy and scipy."""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
from analysis_pipeline import stats as S  # noqa: E402
sys.path.insert(0, HERE)
import r3_common as C  # noqa: E402
from r4_t0 import four  # noqa: E402
from x2ds_analyze import load  # noqa: E402

STEPS = (("zsretry", "zs"), ("gated", "zs"), ("fixed", "zs"), ("gated", "zsretry"), ("fixed", "zsretry"), ("fixed", "gated"),
         ("zsretryempty", "zs"), ("zsretryempty", "zsretry"), ("gated", "zsretryempty"),   # W7: the empty-gate arm (W7 s0, W7RR r1)
         ("gatedempty", "zs"), ("gatedempty", "gated"), ("gatedempty", "zsretryempty"), ("fixed", "gatedempty"))   # W8: the pipeline + empty gate (W8 s0 / r1)


def paths(W, bk):
    A = W + "/analysis_runs" if os.path.isdir(W + "/analysis_runs") else W + "/runs"   # host work tree, or <WORK_DIR> of the unpacked run archives (MIN, R4T0 from 1.0.0)
    M, R, X = A + "/MIN", A + "/R4T0", W + "/runs"
    if bk == "dsv41flash":
        s0 = {a: "%s/X2DS/%s-dsv41flash-s0" % (X, a) for a in ("zs", "gated", "fixed")}
        s0["zsretry"] = X + "/X2DSR/zsretry-dsv41flash-s0"
        r1 = {a: "%s/X2DSRR/%s-dsv41flash-r1" % (X, a) for a in ("zs", "gated", "fixed", "zsretry")}
        s0["zsretryempty"], r1["zsretryempty"] = X + "/W7/zsretryempty-dsv41flash-s0", X + "/W7RR/zsretryempty-dsv41flash-r1"
        s0["gatedempty"], r1["gatedempty"] = X + "/W8/gatedempty-dsv41flash-s0", X + "/W8/gatedempty-dsv41flash-r1"
    else:
        s0 = {a: ("%s/M1/%s-s0" % (M, a)) if bk == "plus" else ("%s/M5/%s-%s-s0" % (M, a, bk)) for a in ("zs", "gated", "fixed")}
        s0["zsretry"] = "%s/X2DSR/zsretry-%s-s0" % (X, bk)
        r1 = {a: "%s/%s-%s-r1" % (R, a, bk) for a in ("zs", "gated", "fixed")}
        r1["zsretry"] = "%s/X2DSRR/zsretry-%s-r1" % (X, bk)
        s0["zsretryempty"], r1["zsretryempty"] = "%s/W7/zsretryempty-%s-s0" % (X, bk), "%s/W7RR/zsretryempty-%s-r1" % (X, bk)
        s0["gatedempty"], r1["gatedempty"] = "%s/W8/gatedempty-%s-s0" % (X, bk), "%s/W8/gatedempty-%s-r1" % (X, bk)
    return s0, r1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=10000)
    a = ap.parse_args()
    res = {"B": a.B, "backbones": {}}
    for bk in ("plus", "flash", "27b", "dsv41flash"):
        s0p, r1p = paths(a.w, bk)
        R0 = {k: load(v + "/preds.jsonl") for k, v in s0p.items()}
        R1 = {k: load(v + "/preds.jsonl") for k, v in r1p.items()}
        qids = sorted(set.intersection(*[set(r) for r in list(R0.values()) + list(R1.values())]))
        n = len(qids)
        dbs = np.array([R0["zs"][q]["db_id"] for q in qids])
        boot = S.ClusterBoot(dbs, B=a.B)
        ex0 = {k: np.array([1.0 if r[q].get("ex") == 1 else 0.0 for q in qids]) for k, r in R0.items()}
        ex1 = {k: np.array([1.0 if r[q].get("ex") == 1 else 0.0 for q in qids]) for k, r in R1.items()}
        arm = {}
        for k in ex0:
            b = int(((ex0[k] == 1) & (ex1[k] == 0)).sum())
            c = int(((ex0[k] == 0) & (ex1[k] == 1)).sum())
            arm[k] = {"ex_s0": 100.0 * ex0[k].mean(), "ex_r1": 100.0 * ex1[k].mean(), "b": b, "c": c, "sd_net": 100.0 * math.sqrt(b + c) / n,
                      "sources": {"s0": s0p[k], "r1": r1p[k]}}
        step = {}
        for x, y in STEPS:
            d0, d1 = ex0[x] - ex0[y], ex1[x] - ex1[y]
            f0, f1 = four(d0, dbs, boot, a.B), four(d1, dbs, boot, a.B)
            si = math.sqrt(arm[x]["sd_net"] ** 2 + arm[y]["sd_net"] ** 2)
            sdir = 100.0 * math.sqrt(float(((d1 - d0) ** 2).sum())) / n
            step["%s_over_%s" % (x, y)] = {"s0": f0["est"], "s0_wide": f0["wide"], "r1": f1["est"], "r1_wide": f1["wide"], "r1_cl": f1["cl"],
                                           "r1_reading_wide": C.reading(round(f1["wide"][0], 2), round(f1["wide"][1], 2)),
                                           "sd_indep": si, "line": 2 * si, "sd_direct": sdir, "line_direct": 2 * sdir}
        res["backbones"][bk] = {"n": n, "arm": arm, "step": step}
        print(bk, n, {k: "%.2f" % v["sd_net"] for k, v in arm.items()}, {k: "s0 %+.2f r1 %+.2f line %.2f" % (v["s0"], v["r1"], v["line"]) for k, v in step.items()}, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
