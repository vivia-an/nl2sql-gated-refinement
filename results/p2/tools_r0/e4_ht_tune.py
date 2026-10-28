"""R0 workaround (host code has no HT mini-stream wiring; X2 open issue 4). New file, nothing patched.
Builds HT mini-stream roles (per HT db: seed 12 / stream 40 / probe 24, seed 20261026) from bird_train_HT_full, makes
data.load_split('bird_train_HR') return those items for this process only, then calls the unmodified stream.run_unit.
usage: python tools_r0/e4_ht_tune.py --trial NAME --policy ws_full --params '{"eta":0.3,"rho":0.2}' --out DIR [--pollution 40] [--perm 1]"""
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concurrent.futures import ThreadPoolExecutor
from nl2sql import config as C, data, paths, stream
from nl2sql.stream import run_unit, _llm
from nl2sql.eval.score import Evaluator
from nl2sql.systems.ours.pipeline import ZeroShot
import copy

N_SEED, N_STREAM, N_PROBE = 12, 40, 24
_orig = data.load_split


def ht_mini(with_gold=None):
    sp = json.load(open(paths.SPLITS_JSON))
    items = _orig("bird_train_HT_full", with_gold=True if with_gold is None else with_gold)
    out = []
    for db in sorted(sp["ht_dbs"]):
        its = sorted([i for i in items if i.db_id == db], key=lambda i: i.qid)
        rnd = random.Random("20261026:" + db)
        rnd.shuffle(its)
        for k, it in enumerate(its[:N_SEED + N_STREAM + N_PROBE]):
            it = copy.copy(it)
            it.meta = dict(it.meta)
            it.meta["role"] = "seed" if k < N_SEED else ("stream" if k < N_SEED + N_STREAM else "probe")
            it.split = "bird_train_HR"
            out.append(it)
    return out


def patched(name, with_gold=None):
    if name == "bird_train_HR":
        return ht_mini(with_gold)
    return _orig(name, with_gold)


data.load_split = patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--policy", default="ws_full")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="exp/configs/e4.yaml")
    ap.add_argument("--pollution", type=int, default=40)
    ap.add_argument("--perm", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--conc", type=int, default=8)
    a = ap.parse_args()
    cfg_path = a.config if os.path.isabs(a.config) else str(paths.REPO / a.config)
    cfg = C.load_config(cfg_path)
    cfg.setdefault("m6", {})
    cfg["m6"]["policy"] = a.policy
    cfg["m6"]["params"] = dict(cfg["m6"].get("params") or {})
    cfg["m6"]["params"].update(json.loads(a.params))
    model = C.resolve_model("primary_actor")
    base = os.path.join(a.out, a.trial)
    os.makedirs(base, exist_ok=True)
    llm = _llm(base, conc=a.conc)
    nf_path = os.path.join(a.out, "natfail_ht.json")
    if os.path.exists(nf_path):
        natfail = json.load(open(nf_path))
    else:
        zs = ZeroShot(llm, model, cfg)
        seeds = [i for i in ht_mini() if i.meta["role"] == "seed"]
        ev = Evaluator("bird_train_HR", run_id="natfail_ht")

        def one(it):
            sql = zs.run(it)["final_sql"]
            return it.qid, sql, ev.score_sql(it, sql).get("ex")
        with ThreadPoolExecutor(max_workers=4) as ex:
            rows = list(ex.map(one, seeds))
        natfail = {q: s for q, s, e in rows if e == 0 and s}
        json.dump(natfail, open(nf_path, "w"), indent=1)
        print("natfail: %d/%d zs failures on HT seed items" % (len(natfail), len(seeds)), flush=True)
    dbs = sorted({i.db_id for i in ht_mini()})
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(dbs)) as ex:
        futs = [ex.submit(run_unit, llm, cfg, model, d, a.policy, a.pollution, a.perm, base, natfail, 0, a.workers) for d in dbs]
        for f in futs:
            print(f.result(), flush=True)
    # summarise
    tot = {"stream_n": 0, "stream_exall": 0, "probe_n": 0, "probe_exall": 0, "probe0_exall": 0, "calls": 0, "pt": 0, "ct": 0, "ev_clean": 0, "ev_pol": 0}
    per_db = {}
    for d in dbs:
        unit = "%s-p%02d-perm%d-%s" % (a.policy, a.pollution, a.perm, d)
        evs = [json.loads(l) for l in open(os.path.join(base, unit, "events.jsonl"))]
        st = [e for e in evs if e["type"] == "stream"]
        pr = [e for e in evs if e["type"] == "probe"]
        last, first = pr[-1], pr[0]
        sx = sum((0 if e["abstained"] else (e["ex"] or 0)) for e in st)
        px = sum(r["ex_all"] or 0 for r in last["items"])
        p0 = sum(r["ex_all"] or 0 for r in first["items"])
        tot["stream_n"] += len(st); tot["stream_exall"] += sx
        tot["probe_n"] += len(last["items"]); tot["probe_exall"] += px; tot["probe0_exall"] += p0
        tot["calls"] += sum(e["calls"] for e in st) + sum(r["calls"] for p in pr for r in p["items"])
        per_db[d] = {"stream": [len(st), sx], "probe_final": [len(last["items"]), px], "probe0": p0, "diag": last["diag"]}
    n = tot["stream_n"] + tot["probe_n"]
    metric = {"ex": round((tot["stream_exall"] + tot["probe_exall"]) / n, 4), "ex_probe_final": round(tot["probe_exall"] / tot["probe_n"], 4),
              "ex_stream": round(tot["stream_exall"] / tot["stream_n"], 4), "ex_probe_init": round(tot["probe0_exall"] / tot["probe_n"], 4),
              "calls_per_q": round(tot["calls"] / (tot["stream_n"] + 3 * tot["probe_n"]), 3), "n_scored": n}
    rec = {"system": "e4_" + a.policy, "trial": a.trial, "params": json.loads(a.params), "pollution": a.pollution, "perm": a.perm,
           "split": "bird_train_HT_ministream", "n_q": n, "metric": metric, "per_db": per_db, "wall_s": round(time.time() - t0),
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(rec, open(os.path.join(base, "trial_summary.json"), "w"), indent=1)
    with open(paths.EXP_DIR / "tuning_log.jsonl", "a") as f:
        f.write(json.dumps({k: rec[k] for k in ("system", "trial", "params", "pollution", "perm", "split", "n_q", "metric", "timestamp")}) + "\n")
    print("TRIAL", a.trial, json.dumps(metric), flush=True)
    llm.close()


if __name__ == "__main__":
    main()
