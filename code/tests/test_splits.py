import pytest
import json

from nl2sql import data, paths


def sp():
    return json.load(open(paths.SPLITS_JSON))


@pytest.mark.needs_bird
def test_partition_disjoint_and_sizes():
    s = sp()
    ht, hr, pool = set(s["ht_dbs"]), set(s["hr_dbs"]), set(s["pool_dbs"])
    assert len(ht) == 4 and len(hr) == 8 and len(pool) == 56 and 'retail_world' not in pool | ht | hr
    assert not (ht & hr) and not (ht & pool) and not (hr & pool)
    dev = {i.db_id for i in data.load_split("bird_dev")}
    assert not (dev & (ht | hr | pool))


@pytest.mark.needs_bird
def test_ht300_hr_roles():
    s = sp()
    assert len(s["ht300_qids"]) == 300 and len(set(s["ht300_qids"])) == 300
    ht = data.load_split("bird_train_HT")
    assert len(ht) == 300 and {i.db_id for i in ht} <= set(s["ht_dbs"])
    hr = data.load_split("bird_train_HR")
    assert {i.db_id for i in hr} == set(s["hr_dbs"])
    for db, roles in s["hr_roles"].items():
        n = sum(len(v) for v in roles.values())
        assert n <= 150
        assert len(roles["seed"]) == int(0.15 * n + 0.5 + 1e-9) and len(roles["probe"]) == int(0.30 * n + 0.5 + 1e-9)
    pool = data.load_split("bird_train_pool")
    assert not ({i.db_id for i in pool} & (set(s["ht_dbs"]) | set(s["hr_dbs"])))
    # no overlap of questions between HT-300 and HR
    assert not ({i.qid for i in ht} & {i.qid for i in hr})
    folds = s["cv_folds"]
    assert len(folds) == 4 and all(len(f) == 2 for f in folds)


@pytest.mark.needs_bird
def test_deterministic_regeneration(tmp_path):
    import subprocess, sys, shutil
    before = open(paths.SPLITS_JSON).read()
    r = subprocess.run([sys.executable, str(paths.EXP_DIR / "splits" / "make_splits.py")], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]
    assert open(paths.SPLITS_JSON).read() == before


@pytest.mark.needs_bird
def test_minidev_hf_unique_ids():
    items = data.load_split("bird_minidev")
    assert len(items) == 500 and len({i.qid for i in items}) == 500
