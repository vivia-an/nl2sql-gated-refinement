#!/usr/bin/env python3
"""Post hoc (SAP Section 30): the intervals of the W9 contrasts under the paper's reading rule. No model call, no database access (reads preds only).
Needs numpy and scipy:
    python3 code/tools/x2ds_w9_iv.py --runs <WORK_DIR>/runs/W9 --qids exp/posthoc/w9_qids.txt --harm results/x2ds/w9_harm.json --out <OUT>/w9_iv.json
Per backbone and contrast (empty-gate retry minus single call + retry; empty-gate retry minus single call; single call + retry minus single call): the estimate, the four intervals of the
Methods (database-clustered bootstrap, item-level bootstrap, cluster-robust t(G-1), exact sign-flip over the size-weighted database means; B = 10,000) and their hull (the widest), the minimum
detectable effect (80% power, two-sided 5%, clustered standard error), and the same quantities with identical final SQL text scored identically (the same-text reading of Section 5.6).
With 7 databases the clustered intervals under-cover; the hull is the paper's reading. Code of the intervals: r4_t0.four, as for every other post hoc contrast."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
from analysis_pipeline import stats as S  # noqa: E402
sys.path.insert(0, HERE)
import r3_common as C  # noqa: E402
from r4_t0 import four  # noqa: E402

BK = ("plus", "flash", "27b", "dsv41flash")
CONTRASTS = (("zsretryempty_minus_zsretry", "zsretryempty", "zsretry"), ("zsretryempty_minus_zs", "zsretryempty", "zs"), ("zsretry_minus_zs", "zsretry", "zs"))


def preds(p):
    return {r["qid"]: r for r in (json.loads(l) for l in open(p))}


def iv(f):
    return {"est_pp": f["est"], "cluster_ci_pp": list(f["cl"]), "item_ci_pp": list(f["item"]), "t_ci_pp": list(f["t"]), "signflip_ci_pp": list(f["sf"]),
            "wide_ci_pp": list(f["wide"]), "reading_wide": C.reading(*f["wide"]), "reading_cluster": C.reading(*f["cl"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--qids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--harm", default=None, help="results/x2ds/w9_harm.json (harm_qids from x2ds_w9.py)")
    a = ap.parse_args()
    keep = set(open(a.qids).read().split())
    out = {"_meta": {"scope": "posthoc", "sap": "Section 30", "split": "bird_train_HR", "B": a.B}, "backbones": {}}
    for b in BK:
        P = {x: preds("%s/%s-%s-s0/preds.jsonl" % (a.runs, x, b)) for x in ("zs", "zsretry", "zsretryempty")}
        qids = sorted(keep)
        assert all(q in P[x] for x in P for q in qids), b
        dbs = [P["zs"][q]["db_id"] for q in qids]
        boot = S.ClusterBoot(dbs, B=a.B)
        ex = {x: np.array([float(P[x][q]["ex"]) for q in qids]) for x in P}
        sql = {x: [P[x][q].get("final_sql") or "" for q in qids] for x in P}
        r = {"n": len(qids), "n_db": len(set(dbs)), "ex": {x: 100.0 * float(ex[x].mean()) for x in P}, "contrasts": {}}
        for cid, x, y in CONTRASTS:
            v = ex[x] - ex[y]
            same = np.array([sql[x][i] == sql[y][i] for i in range(len(qids))])
            vst = np.where(same, 0.0, v)
            c = iv(four(v, dbs, boot, a.B))
            se = float(np.std(boot.means(v), ddof=1))
            c.update({"mde80_pp": 100.0 * 2.8016 * se, "up": int((v > 0).sum()), "down": int((v < 0).sum()),
                      "same_text_flips": int(((v != 0) & same).sum())})
            st = iv(four(vst, dbs, boot, a.B))
            sest = float(np.std(boot.means(vst), ddof=1))
            st.update({"mde80_pp": 100.0 * 2.8016 * sest, "up": int((vst > 0).sum()), "down": int((vst < 0).sum())})
            c["same_text"] = st
            r["contrasts"][cid] = c
        # decomposition: the trigger set (the retry's final answer is empty) against the rest, final SQL changed or not. A question counted as harmed
        # must have its query rewritten: an unchanged correct empty answer scored wrong on a second execution is a scoring flip, not harm (one such question on Flash).
        H = set(json.load(open(a.harm))["backbones"][b]["harm_qids"]) if a.harm else set()
        dec = {}
        for q in qids:
            d = P["zsretryempty"][q]["ex"] - P["zsretry"][q]["ex"]
            k = ("in" if P["zsretry"][q].get("final_status") == "ok_empty" else "out") + ("_same" if (P["zsretryempty"][q].get("final_sql") or "") == (P["zsretry"][q].get("final_sql") or "") else "_chg")
            u = dec.setdefault(k, {"up": 0, "down": 0})
            u["up"] += int(d > 0)
            u["down"] += int(d < 0)
        r["decomp"] = dec
        r["harm_rewritten"] = sum(1 for q in H if (P["zsretryempty"][q].get("final_sql") or "") != (P["zsretry"][q].get("final_sql") or ""))
        r["harm_flip"] = len(H) - r["harm_rewritten"]
        out["backbones"][b] = r
        print(b, json.dumps({k: (v["est_pp"], v["wide_ci_pp"], v["same_text"]["est_pp"], v["same_text"]["wide_ci_pp"], v["same_text"]["mde80_pp"]) for k, v in r["contrasts"].items()}), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
