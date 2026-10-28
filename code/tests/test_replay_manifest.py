"""Replay selects runs from exp/replay_manifest.json, not by "newest among identical configurations" (a release defect found while replaying the
full archive: the temperature-0 re-run R4T0 repeats the configuration of the headline fixed-loop run and was picked for its role). No data needed."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
sys.path.insert(0, CODE)
from analysis_pipeline import runs as R      # noqa: E402


def _mk(root, rel, run_id, started, config_sha="c1", system="full_fixed", cfg="full_fixed", n=3, model="qwen37plus"):
    d = os.path.join(root, rel)
    os.makedirs(d)
    with open(os.path.join(d, "preds.jsonl"), "w") as f:
        for i in range(n):
            f.write(json.dumps({"qid": "q%d" % i, "system": system, "split": "bird_dev", "model": model, "seed": 0, "ex": 1}) + "\n")
    with open(os.path.join(d, "run_meta.json"), "w") as f:
        json.dump({"run_id": run_id, "system": system, "split": "bird_dev", "model": model, "seed": 0, "config_sha256": config_sha, "started": started}, f)
    with open(os.path.join(d, "config.yaml"), "w") as f:
        f.write("argv:\n  config: exp/configs/%s.yaml\n" % cfg)


MODELS = os.path.join(ROOT, "exp", "models.yaml")


def test_replay_dirs_exclude_the_later_identical_rerun(tmp_path):
    root = str(tmp_path)
    _mk(root, "MIN/M1/fixed-s0", "MIN-full_fixed-a", "2026-10-26 10:00:00")
    _mk(root, "R4T0/plus-fixed-r1", "R4T0-full_fixed-a", "2026-10-28 02:00:00")           # same configuration, started later
    reg = R.Registry(root, {}, MODELS, "bird_dev", expected_n=3)
    assert reg.get("full_fixed").run_id == "R4T0-full_fixed-a"                          # without a selection the later re-run wins: the defect
    reg = R.Registry(root, {}, MODELS, "bird_dev", expected_n=3, replay_dirs=["MIN/M1/fixed-s0"])
    assert reg.get("full_fixed").run_id == "MIN-full_fixed-a"


def test_tie_break_does_not_depend_on_file_times(tmp_path):
    root = str(tmp_path)
    _mk(root, "A/x", "A-1", "2026-10-26 10:00:00")
    _mk(root, "B/x", "B-1", "2026-10-28 10:00:00")
    os.utime(os.path.join(root, "B/x", "preds.jsonl"), (1, 1))                           # B is the newer run but has the older file
    reg = R.Registry(root, {}, MODELS, "bird_dev", expected_n=3)
    assert reg.get("full_fixed").run_id == "B-1"


def test_missing_run_directory_is_an_error_not_a_silent_substitute(tmp_path):
    root = str(tmp_path)
    _mk(root, "R4T0/plus-fixed-r1", "R4T0-full_fixed-a", "2026-10-28 02:00:00")
    with pytest.raises(R.RunError):
        R.Registry(root, {}, MODELS, "bird_dev", expected_n=3, replay_dirs=["MIN/M1/fixed-s0"])


def test_none_means_every_run(tmp_path):
    assert R.load_replay_dirs("none") is None and R.load_replay_dirs(None) is None


def test_shipped_manifest_is_consistent():
    m = json.load(open(os.path.join(ROOT, "exp", "replay_manifest.json")))
    dirs = sorted(m["runs"])
    assert len(dirs) == len(set(dirs)) and dirs
    assert not [d for d in dirs if d.split("/")[0] in ("R3", "R4T0")]
    assert R.load_replay_dirs(os.path.join(ROOT, "exp", "replay_manifest.json")) == dirs
    for role, d in m["roles"].items():
        assert d in m["runs"], role
    roles = json.load(open(os.path.join(ROOT, "results", "primary.json")))["_meta"]["roles"]
    assert set(roles) == set(m["roles"])
    for role, rid in roles.items():
        assert m["runs"][m["roles"][role]] == rid, role                                   # the run ids behind the roles of the released primary.json
    assert m["roles"]["full_fixed"].startswith("MIN/")
