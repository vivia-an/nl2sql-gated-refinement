"""SYNTHETIC data generator (self-test and figure previews only; never a result).

Writes a fake `runs/` tree in the exact on-disk format of nl2sql.run (preds.jsonl, cands.jsonl, run_meta.json, config.yaml, usage.jsonl),
E4 stream units (events.jsonl / pollution.json / done), annotation CSVs and an E9 summary, so that primary.py / secondary.py can be run and
checked now. Every number is drawn from a documented data-generating process with planted effects (`effects` dict), which the self-test
compares against the estimates.  Nothing here may be cited in the paper."""
import csv
import hashlib
import json
import os

import numpy as np

DEV_DBS = [("california_schools", 89), ("card_games", 191), ("codebase_community", 186), ("debit_card_specializing", 64), ("european_football_2", 129),
           ("financial", 106), ("formula_1", 174), ("student_club", 158), ("superhero", 129), ("thrombosis_prediction", 163), ("toxicology", 145)]
HR_DBS = ["beer_factory", "bike_share_1", "retails", "movies_4", "music_platform_2", "simpson_episodes", "social_media", "shipping"]
MODEL_ID = {"qwen37plus": "qwen3.7-plus-2026-05-26", "qwen37flash": "qwen3.7-flash-2026-07-15", "qwen3527b": "qwen3.5-27b"}

DEFAULT_EFFECTS = {
    "gated_minus_fixed": 0.0,        # additive logit shift of the gated EX relative to fixed (non-executable roots repaired with equal rate)
    "repair_fixed": 0.65, "repair_gated": 0.65, "repair_blind": 0.45,
    "p_root_err": 0.07, "p_root_err_hot_db": 0.30, "hot_db": "formula_1", "p_root_empty": 0.05,
    "sem_revise_p": 0.10, "sem_harm": 0.22, "sem_gain": 0.08,
    "v3_auc_strength": 1.6, "cons_strength": 0.5,       # signal in V3 and in the consistency bits
    "base_logit": 0.9, "db_sd": 0.4,
    "zs_delta": -0.15, "dail_delta": -0.05, "bon_delta": 0.05, "sdebug_delta": 0.0,
}


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def _write_run(root, exp, system, cfg, model, seed, split, rows, cands=None, usage=None):
    mk = model
    rid = "%s-%s-%s-s%d-%s" % (exp, system, mk, seed, _sha(system + cfg + mk + str(seed))[:8])
    d = os.path.join(root, exp, rid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "preds.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if cands:
        with open(os.path.join(d, "cands.jsonl"), "w") as f:
            for c in cands:
                f.write(json.dumps(c) + "\n")
    json.dump({"run_id": rid, "system": system, "split": split, "model": MODEL_ID.get(mk, mk), "seed": seed, "config_sha256": _sha(system + cfg),
               "started": "2026-01-01 00:00:00"}, open(os.path.join(d, "run_meta.json"), "w"))
    with open(os.path.join(d, "config.yaml"), "w") as f:
        f.write("argv:\n  config: exp/configs/%s.yaml\n  exp: %s\nrun_id: %s\nresolved: {}\n" % (cfg, exp, rid))
    if usage:
        with open(os.path.join(d, "usage.jsonl"), "w") as f:
            for u in usage:
                f.write(json.dumps(u) + "\n")
    return d


def make_dev_world(root, seed=0, effects=None, exp="MIN", split="bird_dev", models=("qwen37plus",), n_scale=1.0, systems=None, with_route=True):
    """Create runs for the dev analysis. Returns the world dict (truth per item) for the self-test."""
    E = dict(DEFAULT_EFFECTS)
    E.update(effects or {})
    rng = np.random.RandomState(seed)
    items = []
    for db, n in DEV_DBS:
        for i in range(max(2, int(round(n * n_scale)))):
            items.append((db, "bird_dev:%s:%d" % (db, i)))
    n = len(items)
    dbs = np.array([d for d, _ in items])
    qids = [q for _, q in items]
    dbeff = {d: rng.randn() * E["db_sd"] for d, _ in DEV_DBS}
    diff = rng.randn(n)
    base = E["base_logit"] + np.array([dbeff[d] for d in dbs]) - 0.9 * diff
    world = {"qids": qids, "dbs": dbs}
    allsys = systems or ["zs", "dail", "sdebug", "bon_vote", "full_fixed", "full_gated", "full_gated_blind", "full_gated_route", "abl_m1", "abl_m5", "abl_q"]
    for mi, mk in enumerate(models):
        shift = [0.0, -0.5, -0.9][mi % 3]
        # ---- shared root generation
        pe = np.where(dbs == E["hot_db"], E["p_root_err_hot_db"], E["p_root_err"])
        u = rng.rand(n)
        status = np.where(u < pe, 0, np.where(u < pe + E["p_root_empty"], 1, 2))      # 0 err, 1 empty, 2 nonempty
        root_ok = np.where(status == 2, rng.rand(n) < sig(base + shift), np.where(status == 1, rng.rand(n) < 0.12, False))
        u_rep = rng.rand(n)
        u_sem = rng.rand(n)
        u_gain = rng.rand(n)

        def final_for(kind):
            fin = root_ok.copy()
            executes = status > 0
            calls = np.full(n, 3.0 if kind == "fixed" else 2.0)
            if kind == "root":
                return fin, executes, np.full(n, 2.0), np.zeros(n)
            prep = {"fixed": E["repair_fixed"], "gated": E["repair_gated"], "blind": E["repair_blind"]}[kind]
            # repair of non-executable roots
            flip = rng.rand(n) < 0.04
            rep = np.where(flip, rng.rand(n), u_rep) < prep
            repaired_ok = rng.rand(n) < sig(base + shift - 0.3)
            err = status == 0
            fin = np.where(err & rep, repaired_ok, fin)
            executes = executes | (err & rep)
            calls = calls + np.where(err, 2.0 if kind == "fixed" else 1.0, 0.0)
            nrev = np.where(err, 1.0, 0.0)
            if kind == "fixed":
                trig = (status > 0) & (u_sem < E["sem_revise_p"])
                harm = trig & root_ok & (rng.rand(n) < E["sem_harm"])
                gain = trig & (~root_ok) & (u_gain < E["sem_gain"])
                fin = np.where(harm, False, fin)
                fin = np.where(gain, True, fin)
                calls = calls + np.where(trig, 2.0, 0.0)
                nrev = nrev + trig
            elif kind == "gated" and E["gated_minus_fixed"] != 0.0:
                # planted accuracy shift of the gated variant relative to the fixed one (logit scale, applied by random flips on executable roots)
                p = abs(E["gated_minus_fixed"]) * 0.25
                if E["gated_minus_fixed"] > 0:
                    fin = np.where((status > 0) & (~fin) & (rng.rand(n) < p), True, fin)
                else:
                    fin = np.where((status > 0) & fin & (rng.rand(n) < p), False, fin)
            return fin, executes, calls, nrev

        def tokens(calls):
            pt = (calls * (2300 + 300 * rng.rand(n))).astype(int)
            ct = (calls * (90 + 60 * rng.rand(n))).astype(int)
            return pt, ct, (pt * 0.4).astype(int)

        results = {}
        for sysname in allsys:
            cfg = sysname
            system = sysname
            if sysname in ("full_fixed",):
                fin, ex_ok, calls, nrev = final_for("fixed")
                cfg = "full_fixed"
            elif sysname in ("full_gated", "full_gated_route", "abl_m1", "abl_m5", "abl_q", "full_gated_blind"):
                kind = "blind" if sysname == "full_gated_blind" else "gated"
                fin, ex_ok, calls, nrev = final_for(kind)
                system = "full_gated"
                cfg = {"full_gated_route": "full_gated_cons_lr", "full_gated_blind": "full_gated_blind"}.get(sysname, sysname)
                if sysname == "abl_m1":
                    fin = np.where(rng.rand(n) < 0.03, ~fin, fin)
                if sysname == "abl_m5":
                    fin, ex_ok, calls, nrev = final_for("root")
                    calls = np.full(n, 1.0)
                if sysname == "abl_q":                        # legacy rendering: more root errors in the hot db
                    extra = (dbs == E["hot_db"]) & (rng.rand(n) < 0.2)
                    fin = np.where(extra & (status > 0), False, fin)
            else:
                dl = {"zs": E["zs_delta"], "dail": E["dail_delta"], "bon_vote": E["bon_delta"], "sdebug": E["sdebug_delta"]}.get(sysname, 0.0)
                pe_ = np.where(dbs == E["hot_db"], 0.25, 0.04)
                okx = rng.rand(n) > pe_
                fin = okx & (rng.rand(n) < sig(base + shift + dl + 0.1))
                ex_ok = okx
                calls = np.full(n, {"zs": 1.0, "dail": 2.0, "sdebug": 3.2, "bon_vote": 5.0}.get(sysname, 3.0)) + (rng.rand(n) < 0.2)
                nrev = np.zeros(n)
            pt, ct, ca = tokens(calls)
            # V3 score of the final answer and consistency
            cor = fin.astype(float)
            v3 = np.clip(np.round((8.4 + E["v3_auc_strength"] * (cor - 0.5) + rng.randn(n) * 1.0) * 2) / 2.0, 1, 10)
            v3 = np.where(rng.rand(n) < 0.35, 10.0, v3)
            agree_p = np.where(fin, 0.5 + 0.12 * E["cons_strength"], 0.5 - 0.1 * E["cons_strength"])
            bits = (rng.rand(n, 4) < agree_p[:, None]) & ex_ok[:, None]
            cons = bits.mean(1)
            nonempty = (status == 2) | (ex_ok & (status == 0))
            z = -3.0 + 0.30 * v3 + 1.2 * cons + 0.4 * nonempty
            rscore = 10 * sig(z)
            rows, cands = [], []
            for i in range(n):
                row = {"qid": qids[i], "db_id": dbs[i], "split": split, "system": system, "model": MODEL_ID.get(mk, mk), "seed": 0, "final_sql": "SELECT 1",
                       "exec_ok": bool(ex_ok[i]), "ex": int(fin[i]), "abstained": bool(v3[i] < 7.5 if sysname.startswith("full") else False),
                       "critic": {"syntax": 10.0 if ex_ok[i] else 3.0, "semantic": float(v3[i]), "efficiency": 9.0, "score": float(v3[i])},
                       "iterations": int(nrev[i]), "actions": ["accept"], "calls": int(calls[i]), "prompt_tokens": int(pt[i]), "completion_tokens": int(ct[i]),
                       "cached_tokens": int(ca[i]), "embed_calls": 2, "rerank_calls": 2, "latency_s": float(calls[i] * 2.4), "error_class": None,
                       "soft_f1": float(fin[i]) * 0.9 + 0.05, "jaccard": float(fin[i]) * 0.85, "overlap": float(fin[i]) * 0.9,
                       "auto_class": None if fin[i] else ["value_mismatch", "row_subset", "row_superset", "shape_mismatch", "empty_result", "exec_error"][int(rng.randint(0, 6))],
                       "final_status": "ok_nonempty",
                       "stages": {"gen": {"calls": 1, "prompt": 1, "completion": 1, "cached": 0, "latency_s": 1.0}}}
                if not ex_ok[i]:
                    row["final_status"] = "err_syntax"
                    row["auto_class"] = "exec_error" if not fin[i] else None
                if sysname in ("full_gated_route", "full_gated") and with_route:
                    row["consistency"] = float(cons[i]); row["n_samples"] = 4; row["cons_bits"] = [int(b) for b in bits[i]]
                    row["route_score"] = float(rscore[i]); row["route_kind"] = "v3+consistency"
                    row["confidence"] = float(rscore[i])
                rows.append(row)
                if system.startswith("full"):
                    st = ["err_syntax", "ok_empty", "ok_nonempty"][int(status[i])]
                    cands.append({"qid": qids[i], "db_id": dbs[i], "cand_idx": 0, "action": "root", "status": st, "ex": int(root_ok[i]), "score": 9.0, "score_kind": "v3"})
                    if nrev[i] > 0:
                        cands.append({"qid": qids[i], "db_id": dbs[i], "cand_idx": 1, "action": "revise", "status": "ok_nonempty" if ex_ok[i] else "err_syntax", "ex": int(fin[i]),
                                      "score": 9.0, "score_kind": "v3"})
            usage = [{"ts": 0, "kind": "chat", "model": MODEL_ID.get(mk, mk), "tag": "gen", "pt": int(pt.sum()), "ct": int(ct.sum()), "cached_tokens": int(ca.sum()),
                      "latency_s": 1.0, "cached": False, "mode": "realtime"}]
            _write_run(root, exp, system, cfg, mk, 0, split, rows, cands, usage)
            results[sysname] = {"ex": fin, "calls": calls, "status": status, "root_ok": root_ok}
            # second and third repetitions for the headline systems (stability analyses)
            if mi == 0 and sysname in ("zs", "full_gated", "full_fixed"):
                for sd in (101, 202):
                    r2 = [dict(r, seed=sd, ex=int(r["ex"] if rng.rand() > 0.04 else 1 - r["ex"])) for r in rows]
                    _write_run(root, exp, system, cfg, mk, sd, split, r2, cands if sysname != "zs" else None, usage)
        world.setdefault("results", {})[mk] = results
    world.update({"status": status, "root_ok": root_ok})
    return world


# ------------------------------------------------------------------------------------------ E4
def make_e4(root, seed=0, effects=None, exp="MIN", policies=("ws_full", "append", "memrl"), pollutions=(0, 40), perms=(1, 2), dbs=None, n_probe=41, n_stream=75):
    """E4 stream units. Planted: probe EX drops with pollution for `append`, less for `ws_full` (effects: ws_resist, append_drop, memrl_drop)."""
    E = {"base": 0.70, "append_drop": 0.14, "ws_drop": 0.06, "memrl_drop": 0.10, "evict_polluted": 0.55, "evict_clean": 0.03}
    E.update(effects or {})
    rng = np.random.RandomState(seed + 7)
    dbs = dbs or HR_DBS
    qprobe = {d: ["bird_train:%s:p%d" % (d, i) for i in range(n_probe)] for d in dbs}
    qdiff = {d: rng.randn(n_probe) for d in dbs}
    drop = {"ws_full": E["ws_drop"], "ws_evict": E["ws_drop"] * 1.2, "append": E["append_drop"], "memrl": E["memrl_drop"], "static": E["append_drop"] * 0.8}
    for pol in policies:
        for p in pollutions:
            for perm in perms:
                for d in dbs:
                    unit = "%s-p%02d-perm%d-%s" % (pol, p, perm, d)
                    out = os.path.join(root, exp, "E4-%s" % pol, unit)
                    os.makedirs(out, exist_ok=True)
                    n_seed = 21
                    n_pol = int(round(p / 100.0 * n_seed))
                    json.dump({"pollution_log": {}, "n_seed": n_seed, "n_polluted": n_pol}, open(os.path.join(out, "pollution.json"), "w"))
                    ev = open(os.path.join(out, "events.jsonl"), "w")
                    ck = [0, int(round(0.4 * n_stream)), n_stream]
                    for ci, c in enumerate(ck):
                        frac = ci / 2.0
                        eff = (drop[pol] * (p / 40.0)) * (0.3 + 0.7 * frac)
                        pr = 0.70 + 0.02 * frac * (pol != "static") - eff
                        items = []
                        for i, q in enumerate(qprobe[d]):
                            pp = 1 / (1 + np.exp(-(np.log(pr / (1 - pr)) + 0.9 * qdiff[d][i])))
                            ex = int(rng.rand() < pp)
                            items.append({"qid": q, "ex": ex, "ex_all": ex, "n_ex": 4, "n_polluted_ex": 0, "calls": 3, "prompt_tokens": 5000, "completion_tokens": 300})
                        ev_pol = int(n_pol * (E["evict_polluted"] if pol in ("ws_full", "ws_evict") else 0.0) * frac)
                        ev_clean = int(n_seed * E["evict_clean"] * frac * (pol in ("ws_full", "ws_evict")) + 5 * frac * (pol in ("ws_full", "ws_evict")))
                        diag = {"size": n_seed + int(c * 0.8) - ev_pol - ev_clean, "size_exp": 21, "polluted_alive": n_pol - ev_pol, "share_polluted_exp": 0.3,
                                "evicted_total": ev_pol + ev_clean, "evicted_polluted": ev_pol, "evicted_clean": ev_clean, "mean_ws_polluted": 5.0, "mean_ws_clean_exp": 6.0}
                        ev.write(json.dumps({"type": "probe", "after": c, "n_stream": n_stream, "items": items, "diag": diag}) + "\n")
                        if ci < 2:
                            for i in range(c, ck[ci + 1]):
                                ok = int(rng.rand() < pr)
                                ev.write(json.dumps({"type": "stream", "i": i, "qid": "bird_train:%s:s%d" % (d, i), "ex": ok, "abstained": bool(rng.rand() < 0.03),
                                                     "calls": 3, "prompt_tokens": 5000, "completion_tokens": 300, "n_ex": 4, "n_polluted_ex": 0, "score": 8.0}) + "\n")
                    ev.close()
                    open(os.path.join(out, "done"), "w").write("x")


# ------------------------------------------------------------------------------------------ E10, E9
def make_annotations(out_dir, world, seed=0, n=120):
    """annotation key + two LLM annotator runs' CSVs + adjudicated final labels + a spot-check file for failures of zs and full_gated."""
    rng = np.random.RandomState(seed + 11)
    os.makedirs(out_dir, exist_ok=True)
    labs = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]
    pri = np.array([0.2, 0.15, 0.2, 0.1, 0.12, 0.08, 0.1, 0.05])
    items = []
    for sysname in ("zs", "full_gated"):
        r = world["results"][list(world["results"])[0]][sysname]
        fail = np.where(r["ex"] == 0)[0][:n // 2]
        for i in fail:
            items.append((world["qids"][i], sysname, ["exec_error", "value_mismatch", "row_subset"][int(rng.randint(0, 3))]))
    with open(os.path.join(out_dir, "annotation_key.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "qid", "system", "auto_class", "stratum", "stratum_population", "stratum_sample", "weight"])
        pop = 300
        for k, (q, s, ac) in enumerate(items):
            w.writerow(["A%04d" % k, q, s, ac, ac, pop, len(items) // 6, pop / float(max(1, len(items) // 6))])
    truth = [int(rng.choice(8, p=pri)) for _ in items]
    for tag, noise in (("A1", 0.12), ("A2", 0.18)):
        with open(os.path.join(out_dir, "annotation_%s.csv" % tag), "w") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "primary_label", "secondary_label", "rationale", "notes"])
            for k in range(len(items)):
                t = truth[k] if rng.rand() > noise else int(rng.randint(0, 8))
                w.writerow(["A%04d" % k, labs[t], "", "r", ""])
    with open(os.path.join(out_dir, "final_labels.csv"), "w") as f:           # adjudicator output (tools/annotation_kappa.py analyze --adjudicated)
        w = csv.writer(f)
        w.writerow(["item_id", "final_label", "final_rationale"])
        for k in range(len(items)):
            w.writerow(["A%04d" % k, labs[truth[k]], "r"])
    with open(os.path.join(out_dir, "spot_check.csv"), "w") as f:              # optional authors' spot check (SYNTHETIC)
        w = csv.writer(f)
        w.writerow(["item_id", "primary_label"])
        for k in range(min(50, len(items))):
            w.writerow(["A%04d" % k, labs[truth[k] if rng.rand() > 0.2 else int(rng.randint(0, 8))]])


def make_e9(out_dir, models=("qwen37plus", "qwen37flash")):
    for mk in models:
        d = os.path.join(out_dir, "E9", mk, "bird_dev")
        os.makedirs(d, exist_ok=True)
        json.dump({"split": "bird_dev", "model": MODEL_ID.get(mk, mk), "n": 200, "n_ok": 200, "q_repro_0.9": 0.02, "q_repro_0.7": 0.11, "sql_exact": 0.04, "table_overlap": 0.4},
                  open(os.path.join(d, "summary.json"), "w"))


def make_all(root, seed=1, effects=None, e4_effects=None, models=("qwen37plus", "qwen37flash", "qwen3527b"), n_scale=1.0, spider=True):
    """dev world + extra backbones (zs/dail/full_fixed/full_gated only) + E4 + annotations + E9 + a Spider variant. -> (world, paths dict)."""
    os.makedirs(root, exist_ok=True)
    open(os.path.join(root, "SYNTHETIC"), "w").write("synthetic self-test data; results derived from it must never reach the paper\n")
    world = make_dev_world(root, seed, effects, models=models[:1], n_scale=n_scale)
    for k, mk in enumerate(models[1:], 1):
        make_dev_world(root, seed + 100 * k, effects, models=(mk,), n_scale=n_scale, systems=["zs", "dail", "full_fixed", "full_gated"])
    if spider:
        for sp in ("spider_dev", "spider_syn"):
            make_dev_world(root, seed + 7, effects, exp="SP", split=sp, models=models[:1], n_scale=0.12, systems=["zs", "dail", "full_fixed", "full_gated"])
    make_e4(root, seed, e4_effects)
    ann = os.path.join(root, "_annot")
    make_annotations(ann, world, seed)
    e9 = os.path.join(root, "_probes")
    make_e9(e9)
    return world, {"e4": os.path.join(root, "MIN"), "annot": ann, "e9": e9}


if __name__ == "__main__":                       # python -m analysis_pipeline.synth <root>  -> synthetic runs + E4 + annotations + probes (marker file SYNTHETIC)
    import sys
    make_all(sys.argv[1], seed=int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    print("synthetic runs written to", sys.argv[1])
