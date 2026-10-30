#!/usr/bin/env python3
"""X2DS pooled contrasts (post hoc). Same per-question arm files as x2ds_analyze.py; pooled = mean over backbones of the per-question paired difference on one common database resample
(the paper's definition of "pooled"). No model call.
    python code/tools/x2ds_pool.py --arm plus:zs=<preds.jsonl> --arm plus:zsretry=<preds.jsonl> ... --out <dir> [--B 10000]
Pools: q3 = Plus, Flash, 27B; w2 = Flash, 27B (the backbone set of the pre-specified test W1). Contrasts: zsretry - zs, gated - zsretry, gated - zs, fixed - zsretry, rootonly - zsretry.
Intervals: r4_t0.four (cluster bootstrap, item bootstrap, CR1 t(G-1), exact sign-flip; widest = hull), reading SUP(0). Needs numpy and scipy."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import r3_common as C  # noqa: E402
from r4_t0 import four  # noqa: E402
from analysis_pipeline import stats as S  # noqa: E402

POOLS = (("q3", ("plus", "flash", "27b")), ("w2", ("flash", "27b")))
CONTRASTS = (("zsretry_minus_zs", "zsretry", "zs"), ("gated_minus_zsretry", "gated", "zsretry"), ("gated_minus_zs", "gated", "zs"),
             ("fixed_minus_zsretry", "fixed", "zsretry"), ("rootonly_minus_zsretry", "rootonly", "zsretry"))


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                rows[r["qid"]] = r
            except ValueError:
                pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="BK:ARM=path/to/preds.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=10000)
    a = ap.parse_args()
    R = {}
    for x in a.arm:
        k, p = x.split("=", 1)
        bk, arm = k.split(":")
        R[(bk, arm)] = load(p)
    qids = sorted(set.intersection(*[set(r) for r in R.values()]))
    dbs = [R[next(iter(R))][q]["db_id"] for q in qids]
    for r in R.values():
        assert [r[q]["db_id"] for q in qids] == dbs
    ex = {k: np.array([1.0 if rows[q].get("ex") == 1 else 0.0 for q in qids]) for k, rows in R.items()}
    boot = S.ClusterBoot(dbs, B=a.B)
    out = {"n_common": len(qids), "n_dbs": len(set(dbs)), "B": a.B, "pools": {}}
    for pid, bks in POOLS:
        out["pools"][pid] = {"backbones": list(bks), "contrasts": {}}
        for cid, x, y in CONTRASTS:
            v = np.mean([ex[(b, x)] - ex[(b, y)] for b in bks], axis=0)
            f4 = four(v, dbs, boot, a.B)
            out["pools"][pid]["contrasts"][cid] = {"est_pp": f4["est"], "cluster_ci_pp": list(f4["cl"]), "item_ci_pp": list(f4["item"]), "t_ci_pp": list(f4["t"]),
                                                   "signflip_ci_pp": list(f4["sf"]), "wide_ci_pp": list(f4["wide"]), "reading_wide": C.reading(*f4["wide"]),
                                                   "reading_cluster": C.reading(*f4["cl"])}
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "x2ds_pooled.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
