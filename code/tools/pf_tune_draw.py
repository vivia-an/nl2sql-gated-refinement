#!/usr/bin/env python3
"""Fresh tuning draw for task PF (SAP Amendment 9, docs/SAP.md 22.7): BIRD-train questions of the four HT databases that were used by NO earlier
block: HT_full (all 640 HT-db questions) minus HT-300 minus the E4/R0 mini-stream roles (seed/stream/probe) minus every qid in a DIAG / tuning item
file. The HT databases are disjoint from the 56 pool databases, so the feedback store (pool-database burn-in questions) never contains a tuning
question or its database. Seeded sample of size N, sorted by qid. No LLM call. Unlocked split only (gold is not read).
  python tools/pf_tune_draw.py --out ../results/pf/tuning/items_tune_pf.json [--n 160] [--seed 20261128]"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)

from nl2sql import data, paths  # noqa: E402
from nl2sql import stream  # noqa: E402
from nl2sql.systems.ours.patent_fb import collect_qids  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--seed", type=int, default=20261128)
    a = ap.parse_args()
    sp = json.load(open(str(paths.SPLITS_JSON)))
    full = data.load_split("bird_train_HT_full", with_gold=False)
    used = {"ht300": set(sp["ht300_qids"]), "ministream": set(i.qid for i in stream.ht_ministream_items(with_gold=False)), "hr": set(sp["hr_order"]) if all(isinstance(x, str) for x in sp["hr_order"]) else set()}
    files = glob.glob(os.path.join(str(paths.REPO), "results", "diag", "**", "*.json"), recursive=True)
    diag = set()
    for f in files:
        if os.path.basename(f).startswith(("items", "critic_ve_units")):
            diag |= collect_qids(json.load(open(f)))
    used["diag_files"] = diag
    banned = set().union(*used.values())
    pool = sorted(i.qid for i in full if i.qid not in banned)
    k = min(a.n, len(pool))
    pick = sorted(random.Random("%d:pf-tune" % a.seed).sample(pool, k))
    body = {"_meta": {"rule": "docs/SAP.md 22.7", "seed": a.seed, "ht_full": len(full), "excluded": {k_: len(v) for k_, v in used.items()},
                      "available": len(pool), "n": k, "dbs": sorted(set(q.split(":")[1] for q in pick))},
            "qids": pick}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    txt = json.dumps(body, indent=1, sort_keys=True)
    open(a.out, "w").write(txt)
    print(json.dumps(body["_meta"]), "sha256", hashlib.sha256(txt.encode()).hexdigest())
    open(os.path.splitext(a.out)[0] + "_qids.txt", "w").write("\n".join(pick) + "\n")


if __name__ == "__main__":
    main()
