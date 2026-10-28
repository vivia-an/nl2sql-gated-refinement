#!/usr/bin/env python3
"""Stratified-by-db, seeded subset of question ids of a split (ids only: the gold SQL is never loaded). Worker machine side (needs nl2sql).
  python tools/make_subset500.py --split bird_dev --n 500 --seed 20261027 --out $R/exp/queues/min_subset500_qids.txt
Allocation is proportional per db (largest remainder), then a seeded shuffle inside each db; output = one qid per line, the format run.py --qids-file reads.
Run it ONCE at freeze time (the freeze checklist (not released) step 6), commit the output; the memo-SQL arm and its paired comparison use exactly this list."""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nl2sql import data  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="bird_dev")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = data.load_split(a.split, with_gold=False)
    by = {}
    for it in items:
        by.setdefault(it.db_id, []).append(it.qid)
    tot = sum(len(v) for v in by.values())
    quota = {d: len(v) * a.n / float(tot) for d, v in by.items()}
    take = {d: int(q) for d, q in quota.items()}
    rest = a.n - sum(take.values())
    for d in sorted(quota, key=lambda d: (-(quota[d] - take[d]), d))[:rest]:
        take[d] += 1
    rng = random.Random(a.seed)
    keep = []
    for d in sorted(by):
        ids = sorted(by[d])
        rng.shuffle(ids)
        keep += ids[:take[d]]
    with open(a.out, "w") as f:
        for q in sorted(keep):
            f.write(q + "\n")
    print("[subset] %s: %d of %d items, %d dbs -> %s" % (a.split, len(keep), tot, len(by), a.out))


if __name__ == "__main__":
    main()
