"""Tests of code/tools/r4_t0.py (temperature-0 rerun analysis; internal note R4T0, SAP section 27). Synthetic runs only (1 534 questions, 11 databases): no LLM call, no worker machine data.
Run on a worker machine (numpy, scipy): cd code && python -m pytest tests/test_r4_t0.py"""
import json
import os
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, ".."))
import r3_common as C  # noqa: E402
import r4_t0 as T  # noqa: E402

N = 1534
DBS = ["db%d" % (i % 11) for i in range(N)]
QIDS = ["bird_dev:%s:%d" % (DBS[i], i) for i in range(N)]
B = 300
LEVEL = {"zs": 0.60, "rootonly": 0.62, "gated": 0.64, "fixed": 0.65}


def write_run(root, rel, ex, sql=None, calls=2.0, keys=None, cached=None, root_sql=None):
    d = os.path.join(str(root), rel)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "preds.jsonl"), "w") as f:
        for i, q in enumerate(QIDS):
            f.write(json.dumps({"qid": q, "db_id": DBS[i], "ex": int(ex[i]), "calls": calls, "error_class": None,
                                "final_sql": (sql[i] if sql else "SELECT   %d  FROM t" % i), "exemplar_ids": ["e%d" % (i % 7), "f%d" % (i % 5)]}) + "\n")
    with open(os.path.join(d, "usage.jsonl"), "w") as f:
        for i in range(N):
            f.write(json.dumps({"kind": "chat", "tag": "gen", "key": (keys[i] if keys else "k%s-%d" % (rel, i)), "cached": bool(cached and cached[i])}) + "\n")
    with open(os.path.join(d, "cands.jsonl"), "w") as f:
        for i, q in enumerate(QIDS):
            f.write(json.dumps({"qid": q, "cand_idx": 0, "sql": (root_sql[i] if root_sql else "SELECT %d FROM t" % i)}) + "\n")


def build(root, flips=60, seed=3):
    """s0 = base; r1 = copy of s0; r2 = s0 with `flips` questions of the fixed arm on Flash changed (EX and SQL). T 0.7 runs: independent draws."""
    rng = np.random.RandomState(seed)
    base = {}
    for b in C.BK:
        for arm in C.ARMS:
            base[(b, arm)] = (rng.rand(N) < LEVEL[arm]).astype(int)
    for b in C.BK:
        for arm in C.ARMS:
            ex = base[(b, arm)]
            for lab in T.T0:
                e2, sq = ex.copy(), None
                if lab == "r2" and (b, arm) == ("flash", "fixed"):
                    e2[:flips] = 1 - e2[:flips]
                    sq = ["SELECT %d FROM t" % i if i >= flips else "SELECT x%d" % i for i in range(N)]
                s0rel = T.run_rel(b, arm, "s0")
                keys = ["k%s-%d" % (s0rel, i) if (lab != "r2" or i % 2 == 0) else "new%d" % i for i in range(N)]
                write_run(root, T.run_rel(b, arm, lab), e2, sql=sq, keys=keys)
            for lab in T.T07:
                write_run(root, T.run_rel(b, arm, lab), (rng.rand(N) < LEVEL[arm] - 0.02).astype(int))
    return base


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    build(root)
    return str(root)


@pytest.fixture(scope="module")
def res(tree):
    return T.run(tree, B)


def test_paths():
    assert T.run_rel("plus", "zs", "s0") == "MIN/M1/zs-s0"
    assert T.run_rel("flash", "gated", "r1") == "R4T0/gated-flash-r1"
    assert T.run_rel("27b", "fixed", "s303") == "R3/fixed-27b-s303"
    assert T.norm_sql("SELECT  a\n FROM t ") == "SELECT a FROM t"


def test_complete_and_counts(res):
    assert res["notes"]["complete_T0"] and res["notes"]["complete_T07"] and res["notes"]["n_qids"] == N
    assert len(res["T0"]) == 25 and len(res["T07"]) == 25 and len(res["cmp"]) == 25


def test_nondeterminism_shares(res):
    A = res["nd"]["arm"]
    e = A["flash|fixed"]
    assert e["sql_diff"]["s0_r1"] == 0.0 and e["ex_diff"]["s0_r1"] == 0.0
    assert abs(e["sql_diff"]["s0_r2"] - 100.0 * 60 / N) < 1e-9
    assert abs(e["ex_diff"]["s0_r2"] - 100.0 * 60 / N) < 1e-9
    assert abs(e["sql_diff"]["vs_s0"] - 0.5 * 100.0 * 60 / N) < 1e-9
    assert abs(e["sql_diff"]["r1_r2"] - 100.0 * 60 / N) < 1e-9
    f = e["ex_flip"]["s0_r2"]
    assert f["right_to_wrong"] + f["wrong_to_right"] == 60
    # whitespace-only differences are not counted in the normalised share
    assert A["plus|zs"]["sql_diff"]["vs_s0"] == 0.0
    # SD over three runs: two equal, one different
    x = [e["ex"][l] for l in T.T0]
    assert abs(e["ex_sd"] - float(np.std(x, ddof=1))) < 1e-9
    lo, hi = e["ex_sd_ci"]
    assert lo < e["ex_sd"] < hi
    assert e["ex_unstable"] == pytest.approx(100.0 * 60 / N)
    assert e["root_diff"]["vs_s0"] == 0.0 and e["exemplars_differ"]["vs_s0"] == 0.0
    assert res["nd"]["group"]["all"]["n_arms"] == 12
    assert res["nd"]["group"]["type.fixed"]["n_arms"] == 3


def test_request_identity(res):
    rq = res["requests"]
    assert rq["plus|zs|r1"]["in_s0_rows"] == 100.0
    assert rq["flash|fixed|r2"]["in_s0_rows"] == pytest.approx(50.0, abs=0.1)
    assert rq["_all"]["in_s0_rows_max"] == 100.0 and rq["_all"]["in_s0_rows_min"] == pytest.approx(50.0, abs=0.1)


def test_contrast_mean_and_intervals(res, tree):
    R = T.RunSet(tree, T.T0)
    e = res["T0"]["stack|plus"]
    per = [100.0 * (R.ex[("plus", "rootonly", l)] - R.ex[("plus", "zs", l)]).mean() for l in T.T0]
    assert e["est"] == pytest.approx(np.mean(per))
    assert e["sd_runs"] == pytest.approx(np.std(per, ddof=1))
    lo, hi = e["wide"]
    for iv in (e["cl"], e["item"], e["t"], e["sf"]):
        assert lo <= iv[0] + 1e-9 and iv[1] <= hi + 1e-9
    assert e["wide_alt"][0] <= min(e["cl"][0], e["item"][0], e["range"][0], e["t2"][0]) + 1e-9
    assert e["reading"]["wide"] in ("positive", "negative", "contains 0")
    # the pooled contrast is the mean over the three backbones
    p = res["T0"]["stack|pooled3"]
    assert p["est"] == pytest.approx(np.mean([res["T0"]["stack|%s" % b]["est"] for b in C.BK]))
    w1 = res["T0"]["w1|weak2"]
    assert w1["est"] == pytest.approx(np.mean([res["T0"]["g_minus_zs|%s" % b]["est"] for b in ("flash", "27b")]))


def test_agreement_and_flip(res):
    # runs r1 and s0 are identical, so on every contrast that does not involve flash fixed the triple reading cannot differ from s0 more than through r2 alone
    for k, e in res["T0"].items():
        assert e["single"]["r1"]["est"] == pytest.approx(e["single"]["s0"]["est"])
        assert e["agrees"]["wide"] in (True, False)
    # contrasts without flash-fixed are identical in all three runs: est equals s0 est, readings agree
    e = res["T0"]["stack|plus"]
    assert e["est"] == pytest.approx(e["single"]["s0"]["est"]) and e["agrees"]["wide"] and e["agrees"]["cluster"]
    f = res["T0"]["f_minus_zs|flash"]
    assert abs(f["per"]["r2"] - f["per"]["s0"]) > 0.5


def test_p1p_and_keys(res):
    from rv3_keys import KINDS
    K = T.build_keys(res, {"total": {"cny_disc": 12.345, "cny_nodisc": 20.0, "calls": 1000}, "by_run": {"fixed-plus-r1": {"cny_disc": 1.0}, "fixed-plus-r2": {"cny_disc": 2.0}}})
    assert K["p1p.met_wide"]["v"] in ("yes", "no")
    gf = res["T0"]["g_minus_f|pooled3"]
    assert K["p1p.clearance_wide"]["v"] == pytest.approx(gf["wide"][0] + 1.5)
    assert K["cost.r2_cny"]["v"] == 2.0 and K["cost.total_cny"]["v"] == 12.345
    for k, e in K.items():
        assert e["v"] is not None, k
        assert e["kind"] in KINDS, k
        if isinstance(e["v"], list):
            assert len(e["v"]) == 2 and all(x is not None for x in e["v"]), k
        if e.get("ci") is not None:
            assert len(e["ci"]) == 2
    assert K["agree.n_total"]["v"] == 25 and 0 <= K["agree.n_wide"]["v"] <= 25
    assert K["cmp.n_total"]["v"] == 25
    assert "cmp.gf_flash.t0_negative_wide" in K and "cmp.gf_pooled.t0_negative_cluster" in K
    json.dumps(K)


def test_cmp_setting_difference(res):
    e = res["cmp"]["stack|plus"]
    assert e["t0_est"] == res["T0"]["stack|plus"]["est"] and e["t07_est"] == res["T07"]["stack|plus"]["est"]
    assert e["setting_diff"]["est"] == pytest.approx(e["t07_est"] - e["t0_est"])


def test_outputs(tree, tmp_path):
    out = str(tmp_path / "o")
    T.main(["--runs", tree, "--out", out, "--B", "200"])
    for f in ("t0.json", "rv4t0_summary.json", "arms_nd.csv", "contrasts_T0.csv", "contrasts_T07.csv", "compare_T07.csv"):
        assert os.path.getsize(os.path.join(out, f)) > 100, f
    d = json.load(open(os.path.join(out, "rv4t0_summary.json")))
    assert d["_meta"]["scope"] == "posthoc-secondary" and d["_meta"]["B"] == 200 and d["_meta"]["missing_keys"] == []


def test_incomplete_t0_is_reported_not_crashing(tree, tmp_path):
    import shutil
    root = str(tmp_path / "r")
    shutil.copytree(tree, root)
    shutil.rmtree(os.path.join(root, "R4T0", "zs-27b-r2"))
    r = T.run(root, 100)
    assert r["notes"].get("incomplete") and r["T0"] == {}


# ------------------------------------------------------------------------------------------------ reduced scope (s0 against r1 only)
def _truncate(root, rel, n):
    p = os.path.join(root, rel, "preds.jsonl")
    lines = open(p).read().splitlines()[:n]
    open(p, "w").write("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def ptree(tree, tmp_path_factory):
    import shutil
    root = str(tmp_path_factory.mktemp("pruns") / "runs")
    shutil.copytree(tree, root)
    for b in C.BK:
        for arm in C.ARMS:
            shutil.rmtree(os.path.join(root, T.run_rel(b, arm, "r2")))
    for b in ("plus",):
        for arm in ("rootonly", "gated", "fixed"):
            shutil.rmtree(os.path.join(root, T.run_rel(b, arm, "r1")))
    _truncate(root, T.run_rel("plus", "zs", "r1"), 15)          # incomplete, not named in PARTIAL_OK: left out
    _truncate(root, T.run_rel("flash", "fixed", "r1"), 600)     # incomplete, named: analysed as partial
    for arm in ("rootonly", "gated"):
        shutil.rmtree(os.path.join(root, T.run_rel("flash", arm, "r1")))
    return root


def test_partial_selection_and_counts(ptree):
    r = T.run_partial(ptree, 200)
    assert "plus|zs" in r["notes"]["left_out"] and "plus|zs" not in r["arm"]
    assert "flash|fixed" in r["arm"] and r["arm"]["flash|fixed"]["complete"] is False and r["arm"]["flash|fixed"]["n"] == 600
    assert r["arm"]["flash|zs"]["complete"] and r["arm"]["27b|fixed"]["complete"]
    assert r["notes"]["left_out"]["plus|rootonly"] == "no rerun"
    e = r["arm"]["27b|zs"]
    assert e["b"] == 0 and e["c"] == 0 and e["sql_diff"] == 0.0 and e["mcnemar_p"] == 1.0   # r1 is a copy of s0
    assert r["pair"]["f_minus_zs|27b"]["complete"] and not r["pair"]["f_minus_zs|flash"]["complete"]
    assert r["pair"]["f_minus_zs|27b"]["change"]["est"] == pytest.approx(0.0)
    assert len(r["lock"]) == 25


def test_partial_flip_counts(tmp_path):
    root = str(tmp_path / "r")
    rng = np.random.RandomState(1)
    ex0 = (rng.rand(N) < 0.6).astype(int)
    ex1 = ex0.copy()
    ex1[:40] = 1 - ex1[:40]                                      # 40 flips
    sq1 = ["SELECT %d FROM t" % i if i >= 40 else "SELECT y%d" % i for i in range(N)]
    for b in C.BK:
        for arm in C.ARMS:
            write_run(root, T.run_rel(b, arm, "s0"), ex0)
    write_run(root, T.run_rel("27b", "zs", "r1"), ex1, sql=sq1)
    r = T.run_partial(root, 200)
    e = r["arm"]["27b|zs"]
    assert e["b"] + e["c"] == 40 and e["ex_changed"] == pytest.approx(100.0 * 40 / N) and e["sql_diff"] == pytest.approx(100.0 * 40 / N)
    assert e["diff"]["est"] == pytest.approx(100.0 * (ex1 - ex0).mean())
    K = T.build_keys_partial(r, {"total": {"cny_disc": 1.5, "cny_nodisc": 2.0, "calls": 10}})
    assert K["nd.27b.zs.b"]["v"] == e["b"] and K["nd.27b.zs.complete"]["v"] == "yes"
    from rv3_keys import KINDS
    for k, v in K.items():
        assert v["v"] is not None and v["kind"] in KINDS, k


def test_partial_cli(ptree, tmp_path):
    out = str(tmp_path / "o")
    T.main(["--runs", ptree, "--out", out, "--B", "200", "--cost", os.path.join(out, "none.json")])
    for f in ("t0_partial.json", "rv4t0_summary.json", "arms_partial.csv", "size_comparison.csv"):
        assert os.path.getsize(os.path.join(out, f)) > 50, f
    d = json.load(open(os.path.join(out, "rv4t0_summary.json")))
    assert d["_meta"]["mode"] == "partial" and d["_meta"]["complete"] is False


# ------------------------------------------------------------------------------------------------ round 5: all arms, every contrast s0 against r1, noise scale
def _flip(ex, idx):
    e = ex.copy()
    e[idx] = 1 - e[idx]
    return e


@pytest.fixture(scope="module")
def ftree(tmp_path_factory):
    """12 arms with s0 and r1 only. r1 = s0 except: plus-zs 30 flips (20 right->wrong), flash-gated 40 flips, flash-fixed 10 flips, 27b-rootonly 6 flips;
    the root SQL of flash-gated r1 differs from flash-fixed r1 on the first 100 questions."""
    root = str(tmp_path_factory.mktemp("fruns") / "runs")
    rng = np.random.RandomState(11)
    flips = {("plus", "zs"): 30, ("flash", "gated"): 40, ("flash", "fixed"): 10, ("27b", "rootonly"): 6}
    for b in C.BK:
        for arm in C.ARMS:
            ex0 = (rng.rand(N) < LEVEL[arm]).astype(int)
            write_run(root, T.run_rel(b, arm, "s0"), ex0)
            k = flips.get((b, arm), 0)
            if k:
                idx = np.where(ex0 == 1)[0][:20].tolist() + np.where(ex0 == 0)[0][:k - 20].tolist() if k > 20 else np.arange(k)
                ex1 = _flip(ex0, np.array(idx, int))
                sq = ["SELECT %d FROM t" % i if i not in set(idx) else "SELECT z%d" % i for i in range(N)]
            else:
                ex1, sq = ex0, None
            rs = ["SELECT %d FROM t" % i if not ((b, arm) == ("flash", "gated") and i < 100) else "SELECT other%d" % i for i in range(N)]
            write_run(root, T.run_rel(b, arm, "r1"), ex1, sql=sq, root_sql=rs)
    return root


@pytest.fixture(scope="module")
def fres(ftree):
    return T.run_partial(ftree, 300)


def test_round5_all_arms_complete(fres):
    assert not fres["notes"]["left_out"] and len(fres["arm"]) == 12 and all(e["complete"] for e in fres["arm"].values())
    assert len(fres["ctr"]) == 25            # 6 contrasts x (3 backbones + pooled) + W1
    assert set(fres["ctr"]) >= {"g_minus_f|pooled3", "loop_g|pooled3", "f_minus_zs|plus", "stack|27b", "w1|weak2"}


def test_round5_arm_noise_and_counts(fres):
    e = fres["arm"]["plus|zs"]
    assert e["b"] + e["c"] == 30
    assert e["sd_net"] == pytest.approx(100.0 * np.sqrt(30) / N) and e["sd_run"] == pytest.approx(e["sd_net"] / np.sqrt(2))
    assert fres["arm"]["flash|gated"]["b"] + fres["arm"]["flash|gated"]["c"] == 40
    assert fres["arm"]["27b|zs"]["sd_net"] == 0.0


def test_round5_contrast_change_and_noise(fres):
    # change = r1 estimate minus s0 estimate, exactly (same questions)
    for k, e in fres["ctr"].items():
        assert e["change"]["est"] == pytest.approx(e["r1"]["est"] - e["s0"]["est"]), k
        lo, hi = e["change"]["wide"]
        assert lo <= e["change"]["est"] + 1e-9 and e["change"]["est"] <= hi + 1e-9, k
    a = fres["arm"]
    # only plus-zs moves among the plus arms: the two-arm noise equals that arm's, and the direct noise too (the per-question change of the contrast is that arm's flip, sign aside)
    e = fres["ctr"]["stack|plus"]
    assert e["sd_indep"] == pytest.approx(a["plus|zs"]["sd_net"]) and e["sd_direct"] == pytest.approx(a["plus|zs"]["sd_net"])
    # two arms moving: independence formula adds the variances; flash gated minus fixed
    e = fres["ctr"]["g_minus_f|flash"]
    assert e["sd_indep"] == pytest.approx(np.sqrt(a["flash|gated"]["sd_net"] ** 2 + a["flash|fixed"]["sd_net"] ** 2))
    # pooled over three backbones: sqrt(sum of variances) / 3
    e = fres["ctr"]["g_minus_f|pooled3"]
    want = np.sqrt(sum(fres["ctr"]["g_minus_f|%s" % b]["sd_indep"] ** 2 for b in C.BK)) / 3.0
    assert e["sd_indep"] == pytest.approx(want)
    # W1: mean of two backbones
    w = fres["ctr"]["w1|weak2"]
    assert w["sd_indep"] == pytest.approx(np.sqrt(sum(fres["ctr"]["g_minus_zs|%s" % b]["sd_indep"] ** 2 for b in ("flash", "27b"))) / 2.0)
    # no movement, no noise
    assert fres["ctr"]["g_minus_f|plus"]["sd_indep"] == 0.0 and fres["ctr"]["g_minus_f|plus"]["same_wide"]
    assert fres["ctr"]["g_minus_f|plus"]["z_indep"] == 0.0


def test_round5_chain_root_check(fres):
    ch = fres["chain"]
    assert ch["plus"]["gf_root_same_s0"] == 100.0 and ch["plus"]["gf_root_same_r1"] == 100.0
    assert ch["flash"]["gf_root_same_s0"] == 100.0 and ch["flash"]["gf_root_same_r1"] == pytest.approx(100.0 * (N - 100) / N)
    assert ch["flash"]["gf_same_root_n"] == N - 100
    assert ch["27b"]["gr_root_same_r1"] == 100.0


def test_round5_keys_and_outputs(ftree, tmp_path):
    out = str(tmp_path / "o")
    cost = str(tmp_path / "cost.json")
    json.dump({"total": {"cny_disc": 50.0, "cny_nodisc": 80.0, "calls": 1000}}, open(cost, "w"))
    prev = str(tmp_path / "prev.json")
    json.dump({"total": {"cny_disc": 36.5, "cny_nodisc": 51.5, "calls": 400}}, open(prev, "w"))
    T.main(["--runs", ftree, "--out", out, "--B", "200", "--cost", cost, "--cost-prev", prev])
    for f in ("t0_partial.json", "rv4t0_summary.json", "arms_partial.csv", "size_comparison.csv", "contrasts_r1.csv"):
        assert os.path.getsize(os.path.join(out, f)) > 50, f
    d = json.load(open(os.path.join(out, "rv4t0_summary.json")))
    assert d["_meta"]["mode"] == "r1" and d["_meta"]["complete"] is True and d["_meta"]["scope"] == "posthoc-secondary"
    K = d["keys"]
    from rv3_keys import KINDS
    for k, e in K.items():
        assert e["v"] is not None and e["kind"] in KINDS, k
        if isinstance(e["v"], list):
            assert len(e["v"]) == 2 and all(x is not None for x in e["v"]), k
    assert K["cost.round5_cny"]["v"] == pytest.approx(13.5) and K["cost.round5_calls"]["v"] == 600
    assert K["ctr.n_total"]["v"] == 25 and "p1p.r1_clearance_wide" in K and "noise.pooled.sd_indep_range" in K
    assert K["nd.plus.zs.n"]["v"] == N and K["nd.flash.fixed.complete"]["v"] == "yes"
    assert K["ctr.g_minus_f.pooled3.change"]["v"] == pytest.approx(K["ctr.g_minus_f.pooled3.r1"]["v"] - K["ctr.g_minus_f.pooled3.s0"]["v"])
    assert K["chain.flash.gf_root_same_r1"]["v"] < 100.0
