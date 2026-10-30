"""The post hoc block of exp/replay_manifest.json (X2DS / X2DSR, release 1.1.0): consistent with the top-level runs and with the paths x2ds_run_analysis.sh reads. No data needed."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
M = json.load(open(os.path.join(ROOT, "exp", "replay_manifest.json")))


def test_block_present_and_apart_from_runs():
    b = M["posthoc_x2ds"]
    assert len(b["runs"]) == 48 and len(b["inputs"]) == 22
    assert not set(b["runs"]) & set(M["runs"])                 # run_all must not need the add-on archive
    for d, rid in b["inputs"].items():
        assert M["runs"].get(d, rid) == rid                     # the inputs are runs of the 1.0.0 archive


def test_block_matches_the_analysis_script():
    sh = open(os.path.join(ROOT, "code", "tools", "x2ds_run_analysis.sh")).read()
    b = M["posthoc_x2ds"]
    for bk in ("plus", "flash", "27b", "dsv41flash"):
        assert "X2DSR/zsretry-%s-s0" % bk in sh or "zsretry-$bk-s0" in sh
        assert "X2DSR/zsretry-%s-s0" % bk in b["runs"]
    for a in ("zs", "fixed", "gated", "rootonly"):
        assert "X2DS/%s-dsv41flash-s0" % a in b["runs"]
    for d in b["inputs"]:
        if d.startswith("MIN/") and "/M5/" not in d:           # the M5 paths are built from a loop in the script; R4T0 is read by x2ds_rerun.py
            assert d.split("/", 1)[1] in sh, d
    assert all(re.match(r"^(X2DSR?R?|W7(RR)?|W8|W9)-", rid) for rid in b["runs"].values())
    for g in ("W7", "W8", "W9"):
        assert "$RUNS/%s" % g in sh
