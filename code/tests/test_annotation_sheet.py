"""E10 sheet generator: pure helpers + a deterministic end-to-end run on the smoke predictions if they exist on this machine."""
import csv
import importlib.util
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "tools", "make_annotation_sheet.py")
spec = importlib.util.spec_from_file_location("make_annotation_sheet", TOOL)
mas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mas)


def test_stratum_mapping():
    assert mas.stratum_of("exec_error:syntax") == "exec_error+timeout"
    assert mas.stratum_of("timeout") == "exec_error+timeout"
    assert mas.stratum_of("column_permutation") == "shape_mismatch(+permutation)"
    assert mas.stratum_of("row_subset") == "row_superset/subset"
    assert mas.stratum_of("gold_fail") == "other"
    assert mas.stratum_of(None) == "other"


def test_allocate_full_and_redistribute():
    sizes = {"a": 100, "b": 100, "c": 3, "d": 0}
    t = mas.allocate(sizes, 60, 25)
    assert sum(t.values()) == 60 and t["c"] == 3 and t["d"] == 0
    assert all(t[k] <= sizes[k] for k in sizes)
    assert t["a"] + t["b"] == 57
    # frame smaller than request: take everything
    t2 = mas.allocate({"a": 5, "b": 4}, 300, 25)
    assert t2 == {"a": 5, "b": 4}


def test_kappa():
    assert mas.cohen_kappa(["L1", "L2", "L3"], ["L1", "L2", "L3"]) == pytest.approx(1.0)
    k = mas.cohen_kappa(["L1", "L1", "L2", "L2"], ["L1", "L2", "L2", "L1"])
    assert k == pytest.approx(0.0)
    assert mas.cohen_kappa([], []) is None


def test_fmt_rows():
    s = mas.fmt_rows({"ok": True, "cols": ["x"], "rows": [(1,), (None,)], "truncated": False}, 10)
    assert "rows: 2" in s and "NULL" in s
    assert mas.fmt_rows({"ok": False, "error": "no such table"}, 10).startswith("ERROR")


def test_adjudication_blind(tmp_path):
    cols = ["item_id", "question", "primary_label", "secondary_label", "rationale", "notes"]
    for name, labs in (("a1.csv", ["L1", "L2", "L3"]), ("a2.csv", ["L1", "L5", "L4"])):
        with open(str(tmp_path / name), "w", newline="") as f:
            w = csv.DictWriter(f, cols)
            w.writeheader()
            for i, l in enumerate(labs):
                w.writerow({"item_id": "A%03d" % i, "question": "q", "primary_label": l, "secondary_label": "", "rationale": "r" + name, "notes": ""})
    out = subprocess.check_output([sys.executable, TOOL, "--out", str(tmp_path), "--adjudicate", str(tmp_path / "a1.csv"), str(tmp_path / "a2.csv")],
                                  universal_newlines=True)
    rep = json.loads(out.strip().splitlines()[-1])
    assert rep["n_items"] == 3 and rep["n_disagree"] == 2
    rows = list(csv.DictReader(open(str(tmp_path / "adjudication.csv"))))
    assert len(rows) == 2 and "label_X" in rows[0] and "primary_label" not in rows[0]
    assert {r["label_X"] for r in rows} | {r["label_Y"] for r in rows} == {"L2", "L5", "L3", "L4"}


SMOKE = os.path.join(os.environ.get("NL2SQL_ROOT", "/nonexistent"), "runs", "smoke50", "zs_ours", "preds.jsonl")


@pytest.mark.skipif(not os.path.exists(SMOKE), reason="needs smoke50 predictions on the worker machine")
def test_end_to_end_deterministic(tmp_path):
    full = SMOKE.replace("zs_ours", "full_fixed")
    outs = []
    for k in (1, 2):
        d = tmp_path / ("o%d" % k)
        subprocess.check_call([sys.executable, TOOL, "--split", "bird_train_HT", "--preds", "zs=" + SMOKE, "full=" + full, "--out", str(d),
                               "--n", "12", "--per-stratum", "2", "--pilot", "3"], cwd=os.path.join(HERE, ".."), stdout=subprocess.DEVNULL)
        outs.append(d)
    for fn in ("annotation_A1.csv", "annotation_A2.csv", "annotation_key.csv", "pilot_A1.csv"):
        assert open(str(outs[0] / fn)).read() == open(str(outs[1] / fn)).read()
    a1 = list(csv.DictReader(open(str(outs[0] / "annotation_A1.csv"))))
    a2 = list(csv.DictReader(open(str(outs[0] / "annotation_A2.csv"))))
    assert len(a1) == len(a2) == 12
    assert {r["item_id"] for r in a1} == {r["item_id"] for r in a2}
    assert [r["item_id"] for r in a1] != [r["item_id"] for r in a2]
    assert set(a1[0]) >= {"gold_sql", "pred_sql", "gold_result", "pred_result", "primary_label", "rationale"}
    assert "system" not in a1[0] and "auto_class" not in a1[0]
    key = list(csv.DictReader(open(str(outs[0] / "annotation_key.csv"))))
    main_ids = {r["item_id"] for r in a1}
    pilot_qids = {r["qid"] for r in key if r["item_id"].startswith("P")}
    assert pilot_qids and not pilot_qids & {r["qid"] for r in key if r["item_id"] in main_ids}


def test_allocate_trims_when_quota_exceeds_n():
    t = mas.allocate({"a": 10, "b": 10, "c": 10}, 12, 5)
    assert sum(t.values()) == 12 and max(t.values()) - min(t.values()) <= 1
