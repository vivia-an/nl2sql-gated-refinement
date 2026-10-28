"""Cross-check eval/ves.py against the OFFICIAL VES scripts on gold-vs-gold of N BIRD dev items (default 100).
  * DAMO-ConvAI bird/llm/src/evaluation_ves.py  (iterated_execute_sql -> time_ratio)           vs ves.measure_ratio(timing="official")
  * bird-bench/mini_dev evaluation/evaluation_ves.py (iterated_execute_sql -> reward)          vs ves.measure_ratio(timing="minidev")
Timing is noisy by nature, so each official script is run TWICE per item (noise floor) and the order of the five
measurements per item is interleaved. Gold-vs-gold: every time_ratio should be ~1 (VES ~ 100, R-VES ~ 100 (reward 1)).
Pass criteria (reported, not hard-asserted except sanity): |VES_ours - mean(VES_off1, VES_off2)| <= max(3 * |VES_off1 - VES_off2|, 5).
Run on an otherwise idle worker machine: python tests/crosscheck_ves_official.py out.json [N] [ITERS]."""
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nl2sql import data, paths  # noqa: E402
from nl2sql.eval import ves  # noqa: E402
from nl2sql.eval.bird_official import rves_reward  # noqa: E402


def _load(name, path, extra_path=None):
    if extra_path:
        sys.path.insert(0, extra_path)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else None
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    tools = paths.DATA / "eval_tools"
    off = _load("official_ves", str(tools / "bird_official" / "evaluation_ves.py"))
    for mod in ("psycopg2", "pymysql"):          # mini_dev's evaluation_utils imports them at module level; unused for SQLite
        if mod not in sys.modules:
            try:
                __import__(mod)
            except ImportError:
                sys.modules[mod] = types.ModuleType(mod)
    mini = _load("minidev_ves", str(tools / "bird_minidev_eval" / "evaluation_ves.py"), str(tools / "bird_minidev_eval"))
    items = data.load_split("bird_dev", with_gold=True)
    gold = {i.qid: i.gold_sql for i in items}
    rnd = random.Random(20261027)
    sample = rnd.sample(items, N)
    load0 = os.getloadavg()[0]
    rows = []
    for it in sample:
        g, dbp = gold[it.qid], it.db_path
        r = {"qid": it.qid, "db_id": it.db_id}
        try:
            r["off1"] = off.iterated_execute_sql(g, g, dbp, ITERS)
            r["ours_o"] = ves.measure_ratio(dbp, g, g, ITERS, "official")["time_ratio"]
            r["off2"] = off.iterated_execute_sql(g, g, dbp, ITERS)
            r["mini1"] = mini.iterated_execute_sql(g, g, dbp, ITERS, "SQLite")            # reward
            r["ours_m"] = rves_reward(ves.measure_ratio(dbp, g, g, ITERS, "minidev")["time_ratio"])
            r["mini2"] = mini.iterated_execute_sql(g, g, dbp, ITERS, "SQLite")
        except Exception as e:
            r["error"] = "%s: %s" % (type(e).__name__, str(e)[:100])
        rows.append(r)
    ok = [r for r in rows if "error" not in r]
    V = lambda xs: statistics.mean(math.sqrt(x) * 100 for x in xs)        # same formula as both official compute_ves
    rep = {"n": len(rows), "n_ok": len(ok), "iters": ITERS, "load1_start": load0, "load1_end": os.getloadavg()[0],
           "VES": {"official_run1": V([r["off1"] for r in ok]), "official_run2": V([r["off2"] for r in ok]), "ours": V([r["ours_o"] for r in ok])},
           "RVES": {"minidev_run1": V([r["mini1"] for r in ok]), "minidev_run2": V([r["mini2"] for r in ok]), "ours": V([r["ours_m"] for r in ok])},
           "mean_time_ratio": {"official_run1": statistics.mean(r["off1"] for r in ok), "official_run2": statistics.mean(r["off2"] for r in ok),
                               "ours_official": statistics.mean(r["ours_o"] for r in ok)},
           "n_errors": len(rows) - len(ok), "errors": [r for r in rows if "error" in r][:5],
           "n_zero_ratio": {"official_run1": sum(1 for r in ok if r["off1"] == 0), "ours": sum(1 for r in ok if r["ours_o"] == 0)}}
    for key, a, b, c in (("VES", "official_run1", "official_run2", "ours"), ("RVES", "minidev_run1", "minidev_run2", "ours")):
        v = rep[key]
        noise = abs(v[a] - v[b])
        diff = abs(v[c] - (v[a] + v[b]) / 2)
        v["noise_floor_abs_diff_official_runs"] = noise
        v["abs_diff_ours_vs_official_mean"] = diff
        v["within_tolerance"] = bool(diff <= max(3 * noise, 5.0))
    # per-item log-ratio correlation between the two official runs and ours (rank agreement on sub-ms noise is expected to be low)
    lr = lambda k: [math.log(max(r[k], 1e-6)) for r in ok]
    try:
        import numpy as np
        rep["corr_log_ratio"] = {"off1_vs_off2": float(np.corrcoef(lr("off1"), lr("off2"))[0, 1]),
                                 "ours_vs_off1": float(np.corrcoef(lr("ours_o"), lr("off1"))[0, 1])}
    except Exception:
        pass
    rep["rows"] = rows
    if out:
        json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
