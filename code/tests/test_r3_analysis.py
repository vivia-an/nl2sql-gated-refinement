"""Tests of the round-3 post hoc analysis scripts (code/tools/r3_{common,seeds,samex,rest,addendum}.py; internal note R3EXP, SAP section 26).
Synthetic runs only (1 534 questions, 11 databases, known correctness structure): no LLM call, no worker machine data. Run on a worker machine: cd code && python -m pytest tests/test_r3_analysis.py"""
import json
import os
import sys
import zlib

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, ".."))
import r3_common as C  # noqa: E402
import r3_seeds as RS  # noqa: E402
import r3_samex as RX  # noqa: E402
import r3_rest as RR  # noqa: E402
import r3_addendum as RA  # noqa: E402

N = 1534
DBS = ["db%d" % (i % 11) for i in range(N)]
QIDS = ["bird_dev:%s:%d" % (DBS[i], i) for i in range(N)]
B = 400


def write_run(root, rel, ex, calls=None, cands=None):
    d = os.path.join(str(root), rel)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "preds.jsonl"), "w") as f:
        for i, q in enumerate(QIDS):
            f.write(json.dumps({"qid": q, "db_id": DBS[i], "ex": int(ex[i]), "calls": float(calls[i]) if calls is not None else 2.0, "error_class": None}) + "\n")
    if cands is not None:
        with open(os.path.join(d, "cands.jsonl"), "w") as f:
            for r in cands:
                f.write(json.dumps(r) + "\n")


def make_all(root, seed_shift=True):
    rng = np.random.RandomState(7)
    base = rng.rand(N)
    lvl = {"zs": 0.55, "rootonly": 0.60, "gated": 0.62, "fixed": 0.63}
    for b in C.BK:
        for arm in C.ARMS:
            for s in C.SEEDS:
                noise = np.random.RandomState(zlib.crc32(('%s|%s|%d' % (b, arm, s)).encode())).rand(N) * 0.2 - 0.1
                ex = (base + noise < lvl[arm] + (0.02 if b == "plus" else 0.0)).astype(int)
                write_run(root, C.run_path(b, arm, s), ex, calls=np.full(N, 2.0 if arm != "fixed" else 3.0))


def test_reading_and_hull():
    assert C.reading(0.1, 2) == "positive" and C.reading(-2, -0.1) == "negative" and C.reading(-0.1, 2) == "contains 0" and C.reading(0.0, 2) == "contains 0"
    assert C.hull((0, 1), (-2, 0.5), (0.2, 3)) == (-2, 3)


def test_run_map_covers_every_arm_backbone_seed():
    for b in C.BK:
        for a in C.ARMS:
            for s in C.SEEDS:
                assert isinstance(C.run_path(b, a, s), str)
    assert C.run_path("plus", "zs", 101) == "MIN/M1/zs-s101" and C.run_path("flash", "fixed", 202) == "R3/fixed-flash-s202" and C.run_path("27b", "gated", 0) == "MIN/M5/gated-27b-s0"


def test_check_run_rejects_incomplete_and_api_fail():
    rows = {q: {"qid": q, "error_class": None} for q in QIDS}
    assert C.check_run(rows)[0]
    assert not C.check_run(dict(list(rows.items())[:100]))[0]
    rows[QIDS[3]]["error_class"] = "api_fail"
    assert not C.check_run(rows)[0] and not C.check_run(None)[0]


def test_replication_contrast_est_is_the_mean_of_the_three_t07_runs_and_excludes_seed_0(tmp_path):
    make_all(tmp_path)
    D, res = RS.run(str(tmp_path), B)
    assert len(D.qids) == N
    for b in C.BK:
        for a in C.ARMS:
            e = res["arms"]["%s|%s" % (b, a)]
            assert e["n_runs"] == 3 and sorted(e["per"]) == list(C.REP)
            vals = [100.0 * D.ex[(b, a, s)].mean() for s in C.REP]
            assert abs(e["mean"] - np.mean(vals)) < 1e-9 and abs(e["sd"] - np.std(vals, ddof=1)) < 1e-9
            assert abs(e["s0"] - 100.0 * D.ex[(b, a, 0)].mean()) < 1e-9             # the headline run is reported apart
    c = res["contrasts"]["stack|plus"]
    per = [100.0 * (D.ex[("plus", "rootonly", s)] - D.ex[("plus", "zs", s)]).mean() for s in C.REP]
    assert abs(c["est"] - np.mean(per)) < 1e-9 and [round(c["per"][s], 9) for s in C.REP] == [round(p, 9) for p in per]
    d0 = 100.0 * (D.ex[("plus", "rootonly", 0)] - D.ex[("plus", "zs", 0)]).mean()
    assert abs(c["s0"]["est"] - d0) < 1e-9                                                                      # seed 0 is kept apart, not in the mean
    assert set(c["agrees"]) == {"cluster", "wide"} and c["agrees"]["cluster"] == (C.reading(*c["ci"]) == C.reading(*c["s0"]["ci"]))
    assert c["range"] == (min(per), max(per)) and 0 not in c["per"]
    assert c["ci"][0] <= c["est"] <= c["ci"][1] and c["wide"][0] <= c["ci"][0] and c["wide"][1] >= c["ci"][1]
    # pooled is the mean of the three backbone differences, W1 the mean of Flash and 27B
    p = res["contrasts"]["g_minus_f|pooled3"]
    assert abs(p["est"] - np.mean([res["contrasts"]["g_minus_f|%s" % b]["est"] for b in C.BK])) < 1e-9
    w = res["contrasts"]["w1|weak2"]
    assert abs(w["est"] - np.mean([res["contrasts"]["g_minus_zs|%s" % b]["est"] for b in ("flash", "27b")])) < 1e-9
    # calls ratio: gated 2.0 / fixed 3.0 everywhere
    cr = res["p1p_calls"]
    assert abs(cr["plus"]["est"] - 2.0 / 3.0) < 1e-9 and abs(cr["pooled3"]["est"] - 2.0 / 3.0) < 1e-9
    # split: share is defined only when the total gain excludes 0
    sp = res["split"]["plus"]
    assert sp["share_defined"] == bool(res["contrasts"]["g_minus_zs|plus"]["ci"][0] > 0 or res["contrasts"]["g_minus_zs|plus"]["ci"][1] < 0)


def test_incomplete_seed_is_excluded_not_averaged(tmp_path):
    make_all(tmp_path)
    p = os.path.join(str(tmp_path), C.run_path("flash", "fixed", 202), "preds.jsonl")
    rows = open(p).read().splitlines()[:700]
    open(p, "w").write("\n".join(rows) + "\n")
    D, res = RS.run(str(tmp_path), B)
    assert res["arms"]["flash|fixed"]["n_runs"] == 2 and "incomplete" in D.note[("flash", "fixed", 202)] and 202 not in res["arms"]["flash|fixed"]["per"]
    assert res["contrasts"]["g_minus_f|flash"]["n_runs"] == 2 and res["contrasts"]["g_minus_f|pooled3"]["n_runs"] == 2
    assert res["contrasts"]["g_minus_f|plus"]["n_runs"] == 3


def test_regen_audit_counts_root_differences(tmp_path):
    make_all(tmp_path)
    rng = np.random.RandomState(3)
    for b in C.BK:
        for s in C.SEEDS:
            roots = {}
            for a in ("gated", "fixed", "rootonly"):
                roots[a] = ["SELECT %d" % (i if (a == "fixed" and i % 100 == 0) else 0) for i in range(N)]
                cands = [{"qid": QIDS[i], "cand_idx": 0, "sql": roots[a][i], "ex": 1} for i in range(N)]
                ex = np.array([json.loads(l)["ex"] for l in open(os.path.join(str(tmp_path), C.run_path(b, a, s), "preds.jsonl"))])
                write_run(tmp_path, C.run_path(b, a, s), ex, cands=cands)
    D, res = RS.run(str(tmp_path), B)
    e = res["regen"]["plus|0"]
    n_diff = len([i for i in range(N) if i % 100 == 0 and i != 0])        # i = 0 yields "SELECT 0" in all arms
    assert e["diff.gated_fixed"] == n_diff and e["diff.fixed_rootonly"] == n_diff and e["diff.gated_rootonly"] == 0
    assert e["shared.g_minus_f"]["n"] == N - n_diff


def test_samex_contrasts_add_up(tmp_path):
    make_all(tmp_path)
    rng = np.random.RandomState(11)
    for b in C.BK:
        for s in C.SEEDS:
            ro = np.array([json.loads(l)["ex"] for l in open(os.path.join(str(tmp_path), C.run_path(b, "rootonly", s), "preds.jsonl"))])
            zs = np.array([json.loads(l)["ex"] for l in open(os.path.join(str(tmp_path), C.run_path(b, "zs", s), "preds.jsonl"))])
            ex = np.where(rng.rand(N) < 0.5, ro, zs)                          # halfway between
            write_run(tmp_path, C.EXRUN[(b, s)], ex)
    res = RX.run(str(tmp_path), B)
    for b in C.BK:
        a, r, st = (res["contrasts"]["%s|%s" % (c, b)] for c in ("ex_minus_zs", "root_minus_ex", "stack"))
        assert abs(a["est"] + r["est"] - st["est"]) < 1e-9                    # the two parts add up to the stack contrast
    sh = res["share"]["plus"]
    assert sh["defined"] == bool(res["contrasts"]["stack|plus"]["ci"][0] > 0 or res["contrasts"]["stack|plus"]["ci"][1] < 0)


def test_p5_removes_exactly_the_timeout_only_roots(tmp_path):
    make_all(tmp_path)
    # failing roots: 40 per backbone; the first 5 of them are time-outs whose SQL runs in the fixed run
    for b in C.BK:
        gated, fixed = [], []
        for i in range(N):
            if i < 40:
                st = "timeout" if i < 5 else "err_syntax"
                gated.append({"qid": QIDS[i], "cand_idx": 0, "sql": "S%d" % i, "status": st})
                fixed.append({"qid": QIDS[i], "cand_idx": 0, "sql": "S%d" % i, "status": "ok_nonempty" if i < 5 else st})
            else:
                gated.append({"qid": QIDS[i], "cand_idx": 0, "sql": "S%d" % i, "status": "ok_nonempty"})
                fixed.append({"qid": QIDS[i], "cand_idx": 0, "sql": "S%d" % i, "status": "ok_nonempty"})
        for arm, cands in (("gated", gated), ("fixed", fixed)):
            ex = np.array([json.loads(l)["ex"] for l in open(os.path.join(str(tmp_path), C.run_path(b, arm, 0), "preds.jsonl"))])
            write_run(tmp_path, C.run_path(b, arm, 0), ex, cands=cands)
        blind = "MIN/M3/blind-s0" if b == "plus" else "MIN/M5/blind-%s-s0" % b
        write_run(tmp_path, blind, np.zeros(N, int))
    res = RR.part_p5(str(tmp_path), B)
    assert res["all"]["n"] == 120 and res["minus_timeout_only"]["n"] == 105 and res["minus_any_timeout"]["n"] == 105
    assert res["timeout_only_roots"] == {"plus": 5, "flash": 5, "27b": 5}
    assert res["all"]["repaired_blind"] == 0 and res["all"]["b"] == res["all"]["repaired_gated"]


def test_mde_formula_and_incremental_cost_logic():
    from scipy import stats as st
    n, pd = 1534, 0.04                                              # a contrast with 4% discordant questions and no true difference in the sd formula
    d = np.zeros(n)
    d[:int(n * pd / 2)] = 1
    d[int(n * pd / 2):int(n * pd)] = -1
    mde = (RA.Z975 + RA.Z80) * np.std(d, ddof=1) / np.sqrt(n) * 100
    assert abs(mde - 2.8016 * np.sqrt(pd / n) * 100) < 0.05                                  # = 2.80 * sqrt(p_d / n) in pp
    assert abs((st.t.ppf(0.975, 10) + st.t.ppf(0.80, 10)) - 3.107) < 0.01


def test_conv_handles_numpy_scalars():
    assert RA.conv({"a": np.float64(1.5), "b": [np.int64(2), (np.bool_(True),)]}) == {"a": 1.5, "b": [2, [True]]}
