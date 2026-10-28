"""Deterministic BIRD train held-out split (EXPERIMENT_SPEC 1.2) + data manifest.
Usage (worker machine, venv python): python exp/splits/make_splits.py   (run from the repo/worker machine root)
Writes exp/splits/train_holdout.json and exp/splits/data_manifest.json."""
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code"))
from nl2sql import paths  # noqa: E402
from nl2sql.data import _bird_dirs  # noqa: E402

SPLIT_SEED, PART_SEED = 20261025, 20261026


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def n_tables(sql):
    try:
        import sqlglot
        from sqlglot import exp
        t = sqlglot.parse_one(sql, read="sqlite")
        return len({x.name.lower() for x in t.find_all(exp.Table) if x.name})
    except Exception:
        return None


def complexity(sql):
    k = n_tables(sql)
    if k is None:   # fallback: token length (declared in the manifest)
        return len(sql.split()) / 10.0, "tokens"
    return float(k) + min(len(sql.split()), 400) / 1000.0, "tables"   # tie-break within equal table count


def main():
    b = _bird_dirs()
    train = json.load(open(b["train"] / "train.json"))
    dev = json.load(open(b["dev"] / "dev.json"))
    tr_db = sorted({r["db_id"] for r in train})
    dv_db = sorted({r["db_id"] for r in dev})
    assert not set(tr_db) & set(dv_db), "train/dev db overlap"
    inputs = {str(b["train"] / "train.json"): sha(b["train"] / "train.json"), str(b["dev"] / "dev.json"): sha(b["dev"] / "dev.json")}
    nq = Counter(r["db_id"] for r in train)
    elig = sorted([d for d in tr_db if nq[d] >= 100], key=lambda d: (nq[d], d))
    terc = [list(x) for x in np.array_split(np.array(elig, dtype=object), 3)]
    tercile_of = {d: i for i, g in enumerate(terc) for d in g}
    rs = np.random.RandomState(SPLIT_SEED)
    held = []
    for i, g in enumerate(terc):
        g = sorted(g)
        take = min(4, len(g))
        pick = list(rs.choice(g, size=take, replace=False))
        held += pick
    # fill from the next-larger tercile if short
    if len(held) < 12:
        rest = [d for d in elig if d not in held]
        extra = list(rs.choice(sorted(rest), size=12 - len(held), replace=False))
        held += extra
    held = sorted(set(held))
    assert len(held) == 12, held
    used_seed = None
    for k in range(100):
        perm = np.random.RandomState(PART_SEED + k).permutation(sorted(held))
        ht, hr = sorted(perm[:4]), sorted(perm[4:])
        ht_t = Counter(tercile_of[d] for d in ht)
        hr_t = Counter(tercile_of[d] for d in hr)
        if len(ht_t) >= 2 and all(hr_t.get(t, 0) >= 2 for t in (0, 1, 2)):
            used_seed = PART_SEED + k
            break
    assert used_seed is not None, "no acceptable HT/HR partition"
    # D0 (docs/DATA.md sec. 9): retail_world's SQLite does not match its gold SQL (330/373 gold queries error) -> never in
    # HT/HR (asserted) and removed from the exemplar pool (POOL57 -> 56 dbs; deviation logged in internal note X2).
    EXCLUDED = {"retail_world": "gold SQL does not execute on the shipped SQLite (D0: 330 of 373 questions error)"}
    assert not (set(EXCLUDED) & set(held)), "excluded db drawn into HT/HR: the seed-derived draw must be revisited"
    pool = [d for d in tr_db if d not in held and d not in EXCLUDED]
    assert len(pool) == len(tr_db) - 12 - len([d for d in EXCLUDED if d in tr_db])

    qid = lambda i, r: "bird_train:%s:%d" % (r["db_id"], i)
    by_db = defaultdict(list)
    for i, r in enumerate(train):
        by_db[r["db_id"]].append((i, r))

    # HT-300: proportional to (db, complexity-tercile) strata, seed PART_SEED
    ht_rows = [(i, r) for d in ht for (i, r) in by_db[d]]
    cx = {}
    ckind = Counter()
    for i, r in ht_rows:
        cx[i], kind = complexity(r["SQL"])
        ckind[kind] += 1
    order = sorted(ht_rows, key=lambda x: (cx[x[0]], x[0]))
    cut = np.array_split(np.arange(len(order)), 3)
    ctier = {}
    for t, idxs in enumerate(cut):
        for j in idxs:
            ctier[order[j][0]] = t
    strata = defaultdict(list)
    for i, r in ht_rows:
        strata[(r["db_id"], ctier[i])].append(i)
    N = len(ht_rows)
    assert N >= 300
    quota = {s: 300.0 * len(v) / N for s, v in strata.items()}
    alloc = {s: int(np.floor(q)) for s, q in quota.items()}
    rem = 300 - sum(alloc.values())
    for s in sorted(strata, key=lambda s: (-(quota[s] - alloc[s]), s))[:rem]:
        alloc[s] += 1
    rs2 = np.random.RandomState(PART_SEED)
    ht300 = []
    for s in sorted(strata):
        idxs = sorted(strata[s])
        pick = rs2.choice(idxs, size=min(alloc[s], len(idxs)), replace=False)
        ht300 += [int(x) for x in pick]
    assert len(ht300) == 300, len(ht300)
    ht300 = [int(x) for x in np.random.RandomState(PART_SEED).permutation(sorted(ht300))]   # fixed shuffled order
    ht300_qids = [qid(i, train[i]) for i in ht300]
    ht300_meta = {qid(i, train[i]): {"complexity_tier": ctier[i], "tercile": tercile_of[train[i]["db_id"]]} for i in ht300}

    # HR per-db roles
    rs3 = np.random.RandomState(PART_SEED)
    hr_roles = {}
    hr_all = []
    for d in sorted(hr):
        idxs = [i for i, _ in by_db[d]]
        n = min(len(idxs), 150)
        sel = [int(x) for x in rs3.permutation(idxs)[:n]]
        ns = int(round(0.15 * n + 1e-9))
        npb = int(round(0.30 * n + 1e-9))
        hr_roles[d] = {"seed": [qid(i, train[i]) for i in sel[:ns]], "probe": [qid(i, train[i]) for i in sel[ns:ns + npb]],
                       "stream": [qid(i, train[i]) for i in sel[ns + npb:]]}
        hr_all += [qid(i, train[i]) for i in sel]
    hr_order = [hr_all[j] for j in np.random.RandomState(PART_SEED).permutation(len(hr_all))]

    # grouped CV: 4 folds x 2 dbs, snake order on train n_q desc
    srt = sorted(hr, key=lambda d: (-nq[d], d))
    snake = [0, 1, 2, 3, 3, 2, 1, 0]
    folds = [[] for _ in range(4)]
    for d, f in zip(srt, snake):
        folds[f].append(d)
    cv_fold_of_db = {d: f for f, ds in enumerate(folds) for d in ds}

    out = {"excluded_dbs": EXCLUDED, "ht_dbs": ht, "hr_dbs": hr, "pool_dbs": pool, "held_out_dbs": held,
           "terciles": {"groups": terc, "tercile_of_db": tercile_of, "n_q": {d: nq[d] for d in elig}},
           "ht300_qids": ht300_qids, "ht300_meta": ht300_meta, "hr_roles": hr_roles, "hr_order": hr_order,
           "cv_folds": folds, "cv_fold_of_db": cv_fold_of_db,
           "seeds_used": {"split": SPLIT_SEED, "partition": used_seed, "ht300": PART_SEED, "hr_roles": PART_SEED,
                          "complexity_basis": dict(ckind)},
           "input_sha256": inputs}
    json.dump(out, open(HERE / "train_holdout.json", "w"), indent=1, sort_keys=True)
    print("held-out", held, "\nHT", ht, "\nHR", hr, "\nfolds", folds, "\npool", len(pool),
          "\nHT300 complexity basis", dict(ckind), "\nHR sizes", {d: [len(v) for v in r.values()] for d, r in hr_roles.items()})
    print("partition seed used:", used_seed)


if __name__ == "__main__":
    main()
