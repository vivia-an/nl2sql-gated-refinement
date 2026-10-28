import json
import os
import subprocess
import sys

import pytest

from nl2sql import lock, paths

CODE = os.path.join(os.path.dirname(__file__), "..")


def test_unlocked_splits_pass():
    for s in ("bird_train_HT", "bird_train_HR", "bird_train_pool"):
        assert lock.verify("full_fixed", s, 0, str(paths.EXP_DIR / "configs" / "full_fixed.yaml"))[0]


def test_locked_split_refused_without_entry(tmp_path):
    cfg = str(paths.EXP_DIR / "configs" / "full_fixed.yaml")
    ok, msg = lock.verify("full_fixed", "bird_dev", 0, cfg, lock_path=str(tmp_path / "none.json"))
    assert not ok and "dev_lock" in msg or "refused" in msg
    lp = tmp_path / "l.json"
    e = lock.compute_entry("full_fixed", cfg)
    json.dump({"entries": [e]}, open(lp, "w"))
    assert lock.verify("full_fixed", "bird_dev", 0, cfg, str(lp))[0]
    assert not lock.verify("full_fixed", "bird_dev", 7, cfg, str(lp))[0]               # seed not registered
    assert not lock.verify("zs_ours", "bird_dev", 0, cfg, str(lp))[0]                  # other system
    e2 = dict(e, config_sha256="0" * 64)
    json.dump({"entries": [e2]}, open(lp, "w"))
    ok, msg = lock.verify("full_fixed", "bird_dev", 0, cfg, str(lp))
    assert not ok and "config_sha256" in msg
    e3 = dict(e, datasets_allowed=["spider_*"])
    json.dump({"entries": [e3]}, open(lp, "w"))
    assert not lock.verify("full_fixed", "bird_dev", 0, cfg, str(lp))[0]
    assert lock.verify("full_fixed", "spider_dev", 0, cfg, str(lp))[0]


def test_runner_refuses_bird_dev():
    if paths.DEV_LOCK.exists():
        pytest.skip("a real dev_lock.json exists")
    r = subprocess.run([sys.executable, "-m", "nl2sql.run", "--system", "full_fixed", "--split", "bird_dev", "--model", "qwen37plus",
                        "--config", "exp/configs/full_fixed.yaml", "--out", "/tmp/should_not_exist_x2", "--limit", "1"],
                       cwd=CODE, capture_output=True, text=True, timeout=120)
    assert r.returncode == 3 and "REFUSED" in r.stderr
    assert not os.path.exists("/tmp/should_not_exist_x2")


@pytest.mark.needs_bird
def test_gold_stripped_on_locked_splits():
    from nl2sql import data
    assert all(i.gold_sql == "" for i in data.load_split("bird_dev")[:50])
    assert all(i.gold_sql for i in data.load_split("bird_train_HT")[:50])
