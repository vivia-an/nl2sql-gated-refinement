"""Stateful stream drivers: E4 (corpus evolution / pollution on HR), E4c (dev stream without feedback), M6 burn-in.

  python -m nl2sql.stream e4 --policy ws_full --pollution 40 --perm 3 --config exp/configs/e4.yaml --out <runs>/E4/<unit> [--stream-split HR|HT]
  python -m nl2sql.stream natfail --config exp/configs/e4.yaml --out <file.json>      # zs failures on HR seed questions
  python -m nl2sql.stream burnin --config exp/configs/full_fixed.yaml --n 1500 --out exp/frozen/corpus_burnin.sqlite
  python -m nl2sql.stream e4c --policy ws_full --config exp/configs/e4c.yaml --perm 1 --out <dir>   (needs dev lock)

--checkpoints "0,40,100" (e4 only, X6): where the probe set is evaluated, in % of the stream. Default 0,40,100 = the registered protocol. "0,100" skips the
40 % probe (603 + 2 x 330 runs per HR job instead of 603 + 3 x 330, about -21 % calls); the primary contrast (final checkpoint) is unaffected, but the
mid-stream trajectory is then not measured. 0 and 100 are required; the list is stored as `checkpoints_pct` in pollution.json of every unit.

A unit = (policy, pollution, perm, db); it is resumable at unit granularity (a `done` marker). A unit that was interrupted starts again
from scratch and REWRITES its events.jsonl / pollution.json (no duplicated events after a restart). Gold execution results are used
for simulated feedback on train data only; on dev the evaluator scores but nothing flows back into the corpus.

--stream-split HT: the HT mini-stream for tuning (X4; replaces R0's e4_ht_tune.py hack): per HT db (bird_train_HT_full) a seeded draw
of 12 seed / 40 stream / 24 probe questions; gold from bird_train_HT_full. With --trial NAME a trial_summary.json is written and a
line is appended to exp/tuning_log.jsonl (split "bird_train_HT_ministream")."""
import argparse
import json
import os
import random
import copy
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

from . import cache_archive, config as C
from . import data, lock, paths
from .eval.score import Evaluator
from .llm import LLMClient
from .meter import Meter
from .systems.ours.corpus import POLICIES, ExperienceCorpus
from .systems.ours.pipeline import FullSystem, Resources, ZeroShot, _pool_vecs, resolved_cfg
from .systems.ours import pollution


def _llm(out, cache=None, conc=8):
    os.makedirs(out, exist_ok=True)
    return LLMClient(cache_path=cache or str(paths.CACHE / "llm"), usage_path=os.path.join(out, "usage.jsonl"),
                     key_file=paths.KEY_FILE, max_concurrency_per_model=conc)


_POOL_CACHE = {}
_POOL_LOCK = threading.Lock()


def _pool_corpus(llm, cfg, policy, params, seed, pool="bird_train_pool"):
    """New ExperienceCorpus pre-filled with the pool history. The pool items + embeddings are loaded once per process; the first
    caller embeds under the lock, concurrent units wait instead of embedding the pool themselves (was a racy mutable default arg)."""
    with _POOL_LOCK:
        if pool not in _POOL_CACHE:
            items = data.load_split(pool, with_gold=True)
            _POOL_CACHE[pool] = (items, _pool_vecs(llm, cfg["models"]["embed"], items, pool))
        items, vecs = _POOL_CACHE[pool]
    c = ExperienceCorpus(llm, cfg["models"]["embed"], policy, params, seed=seed)
    c.add_history(items, vecs=vecs)
    return c


# ----------------------------------------------------------------------------- stream splits (HR = registered; HT = tuning mini-stream)
HT_MINI = {"seed": 12, "stream": 40, "probe": 24, "rng": "20261026"}


def ht_ministream_items(with_gold=True):
    """HT mini-stream items (roles in meta["role"]): per HT db (sorted) the qids of bird_train_HT_full sorted, shuffled with
    random.Random("<rng>:<db>"), first 12 = seed, next 40 = stream, next 24 = probe. Same draw as R0's e4_ht_tune.py."""
    sp = json.load(open(paths.SPLITS_JSON))
    items = data.load_split("bird_train_HT_full", with_gold=with_gold)
    n_s, n_t, n_p = HT_MINI["seed"], HT_MINI["stream"], HT_MINI["probe"]
    out = []
    for db in sorted(sp["ht_dbs"]):
        its = sorted([i for i in items if i.db_id == db], key=lambda i: i.qid)
        random.Random("%s:%s" % (HT_MINI["rng"], db)).shuffle(its)
        for k, it in enumerate(its[: n_s + n_t + n_p]):
            it = copy.copy(it)
            it.meta = dict(it.meta)
            it.meta["role"] = "seed" if k < n_s else ("stream" if k < n_s + n_t else "probe")
            out.append(it)
    return out


def stream_split(name):
    """-> (items with roles, evaluator split, db list). name: HR (registered E4) or HT (tuning mini-stream)."""
    sp = json.load(open(paths.SPLITS_JSON))
    if name == "HR":
        return data.load_split("bird_train_HR"), "bird_train_HR", list(sp["hr_dbs"])
    if name == "HT":
        return ht_ministream_items(), "bird_train_HT_full", sorted(sp["ht_dbs"])
    raise ValueError("unknown --stream-split %r (HR|HT)" % name)


def _score(ev, item, sql):
    s = ev.score_sql(item, sql)
    return s.get("ex"), s


def probe(system, ev, corpus, items, seed, workers):
    corpus._matrix()
    res = []

    def one(it):
        with Meter() as m:
            out = system.run(it, seed=seed, corpus=corpus)
        ex, _ = _score(ev, it, out["final_sql"])
        ex_acc = ex if not out["abstained"] else 0
        pol = [e for e in out["exemplar_ids"] if corpus.entries.get(e) is not None and corpus.entries[e].polluted]
        return {"qid": it.qid, "ex": ex, "ex_all": ex_acc, "n_ex": len(out["exemplar_ids"]), "n_polluted_ex": len(pol),
                "calls": m.calls, "prompt_tokens": m.prompt_tokens, "completion_tokens": m.completion_tokens}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(one, items))
    return res


DEFAULT_CHECKPOINTS = (0, 40, 100)


def parse_checkpoints(spec):
    """--checkpoints "0,40,100": probe checkpoints as percentages of the stream length (X6). The first (0) and the last (100) are required: 0 gives the
    initial probe (ex_probe_init), 100 the final probe (the primary contrast, SAP P4). "0,100" skips the 40 % probe (-21 % calls per E4 job)."""
    if spec is None:
        return DEFAULT_CHECKPOINTS
    try:
        vals = sorted({float(x) for x in (spec.split(",") if isinstance(spec, str) else spec) if str(x).strip() != ""})
    except ValueError:
        raise ValueError("--checkpoints must be comma-separated percentages, got %r" % (spec,))
    if not vals or vals[0] != 0 or vals[-1] != 100 or any(v < 0 or v > 100 for v in vals):
        raise ValueError("--checkpoints must lie in [0, 100] and include 0 and 100, got %r" % (spec,))
    return tuple(int(v) if v == int(v) else v for v in vals)


def run_unit(llm, cfg, model, db_id, policy, p, perm, base_out, natfail, seed=0, workers=4, stream_limit=None, probe_limit=None,
             feedback=True, split_name="HR", checkpoints=DEFAULT_CHECKPOINTS):
    unit = "%s-p%02d-perm%d-%s" % (policy, p, perm, db_id)
    out = os.path.join(base_out, unit)
    if os.path.exists(os.path.join(out, "done")):
        return unit, "skipped"
    os.makedirs(out, exist_ok=True)
    cfg = resolved_cfg(cfg)
    params = dict(cfg["m6"].get("params") or {})
    if not feedback:
        params["phi"] = 0.0
    corpus = _pool_corpus(llm, cfg, policy, params, seed=1000 * perm + seed)
    all_items, eval_split, _ = stream_split(split_name)
    hr = [i for i in all_items if i.db_id == db_id]
    seed_items = [i for i in hr if i.meta["role"] == "seed"]
    probe_items = [i for i in hr if i.meta["role"] == "probe"]
    stream_items = [i for i in hr if i.meta["role"] == "stream"]
    stream_items = [stream_items[j] for j in np.random.RandomState(perm).permutation(len(stream_items))]
    if stream_limit:
        stream_items = stream_items[:stream_limit]
    if probe_limit:
        probe_items = probe_items[:probe_limit]
    res = Resources(llm, cfg)
    system = FullSystem(llm, model, cfg)
    system.pipe.res = res
    ev = Evaluator(eval_split, run_id=unit)
    st = res.db_store(seed_items[0])
    polluted, plog = pollution.pollute_entries(seed_items, p / 100.0, 20261030, st.schema, natfail, seed_items[0].db_path)
    for it in seed_items:
        corpus.add_entry("seed:" + it.qid, it.db_id, it.question, it.evidence, polluted.get(it.qid, it.gold_sql), critic_score=None,
                         ex_self=None, polluted=it.qid in polluted, origin="seed", seed_entry=True, kind="exp")
    events = open(os.path.join(out, "events.jsonl"), "w")       # "w": a restarted unit replays from scratch, so no duplicate events
    json.dump({"pollution_log": plog, "n_seed": len(seed_items), "n_polluted": len(polluted), "checkpoints_pct": list(checkpoints)}, open(os.path.join(out, "pollution.json"), "w"))
    ck = sorted({int(round(c / 100.0 * len(stream_items))) for c in checkpoints})     # default (0, 40, 100) = {0, round(0.4 n), n}
    pol = POLICIES[policy]

    def checkpoint(i):
        r = probe(system, ev, corpus, probe_items, seed, workers)
        events.write(json.dumps({"type": "probe", "after": i, "n_stream": len(stream_items), "items": r, "diag": corpus.stats()}) + "\n")
        events.flush()

    for i, it in enumerate(stream_items):
        if i in ck:
            checkpoint(i)
        with Meter() as m:
            o = system.run(it, seed=seed, corpus=corpus)
        ex, sc = _score(ev, it, o["final_sql"])
        ids = o["exemplar_ids"]
        n_pol = sum(1 for e in ids if corpus.entries.get(e) is not None and corpus.entries[e].polluted)
        events.write(json.dumps({"type": "stream", "i": i, "qid": it.qid, "ex": ex, "abstained": o["abstained"], "calls": m.calls,
                                 "prompt_tokens": m.prompt_tokens, "completion_tokens": m.completion_tokens, "n_ex": len(ids),
                                 "n_polluted_ex": n_pol, "score": o["critic"]["score"] if o["critic"] else None}) + "\n")
        if pol["write"] != "none":
            gold_ex = ex if feedback else None
            corpus.on_use(ids, it.qid, gold_ex)
            if not o["abstained"]:
                corpus.add_entry("str:" + it.qid, it.db_id, it.question, it.evidence, o["final_sql"], o["critic"]["score"] if o["critic"] else None,
                                 ex_self=gold_ex, kind="exp", origin="stream", step=i)
            corpus.evict()
        events.flush()
    checkpoint(len(stream_items))
    events.close()
    open(os.path.join(out, "done"), "w").write(time.strftime("%F %T"))
    return unit, "ok"


def summarize_trial(base, dbs, policy, pollution, perm):
    """Aggregate the events.jsonl of all units of a trial (same numbers as R0's e4_ht_tune.py)."""
    tot = {"stream_n": 0, "stream_exall": 0, "probe_n": 0, "probe_exall": 0, "probe0_exall": 0, "calls": 0, "probe_runs": 0}
    per_db = {}
    for d in dbs:
        unit = "%s-p%02d-perm%d-%s" % (policy, pollution, perm, d)
        evs = [json.loads(l) for l in open(os.path.join(base, unit, "events.jsonl"))]
        st = [e for e in evs if e["type"] == "stream"]
        pr = [e for e in evs if e["type"] == "probe"]
        last, first = pr[-1], pr[0]
        sx = sum((0 if e["abstained"] else (e["ex"] or 0)) for e in st)
        px = sum(r["ex_all"] or 0 for r in last["items"])
        p0 = sum(r["ex_all"] or 0 for r in first["items"])
        tot["stream_n"] += len(st)
        tot["stream_exall"] += sx
        tot["probe_n"] += len(last["items"])
        tot["probe_runs"] += len(last["items"]) * len(pr)
        tot["probe_exall"] += px
        tot["probe0_exall"] += p0
        tot["calls"] += sum(e["calls"] for e in st) + sum(r["calls"] for p_ in pr for r in p_["items"])
        per_db[d] = {"stream": [len(st), sx], "probe_final": [len(last["items"]), px], "probe0": p0, "diag": last["diag"]}
    n = tot["stream_n"] + tot["probe_n"]
    metric = {"ex": round((tot["stream_exall"] + tot["probe_exall"]) / n, 4), "ex_probe_final": round(tot["probe_exall"] / tot["probe_n"], 4),
              "ex_stream": round(tot["stream_exall"] / tot["stream_n"], 4), "ex_probe_init": round(tot["probe0_exall"] / tot["probe_n"], 4),
              "calls_per_q": round(tot["calls"] / (tot["stream_n"] + tot["probe_runs"]), 3), "n_scored": n}
    return metric, per_db, n


def cmd_e4(a):
    cfg = C.load_config(a.config)
    cfg.setdefault("m6", {})
    if a.policy_params:                       # tuning: override m6.params for this trial (HT mini-stream tuning, TUNING.md)
        cfg["m6"]["policy"] = a.policy
        cfg["m6"]["params"] = dict(cfg["m6"].get("params") or {})
        cfg["m6"]["params"].update(json.loads(a.policy_params))
    model = C.resolve_model(a.model)
    base = os.path.join(a.out, a.trial) if a.trial else a.out
    llm = _llm(base, conc=a.conc)
    natfail = json.load(open(a.natfail)) if a.natfail and os.path.exists(a.natfail) else {}
    _, _, all_dbs = stream_split(a.stream_split)
    dbs = a.dbs.split(",") if a.dbs else all_dbs
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(dbs)) as ex:
        futs = [ex.submit(run_unit, llm, cfg, model, d, a.policy, a.pollution, a.perm, base, natfail, a.seed, a.workers, a.stream_limit,
                          a.probe_limit, True, a.stream_split, parse_checkpoints(a.checkpoints)) for d in dbs]
        for f in futs:
            print(f.result(), flush=True)
    if a.trial:
        metric, per_db, n = summarize_trial(base, dbs, a.policy, a.pollution, a.perm)
        params = json.loads(a.policy_params or "{}")
        rec = {"system": "e4_" + a.policy, "trial": a.trial, "params": params, "pollution": a.pollution, "perm": a.perm,
               "split": "bird_train_HT_ministream" if a.stream_split == "HT" else "bird_train_HR", "n_q": n, "metric": metric,
               "per_db": per_db, "wall_s": round(time.time() - t0), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(rec, open(os.path.join(base, "trial_summary.json"), "w"), indent=1)
        with open(paths.EXP_DIR / "tuning_log.jsonl", "a") as f:
            f.write(json.dumps({k: rec[k] for k in ("system", "trial", "params", "pollution", "perm", "split", "n_q", "metric", "timestamp")}) + "\n")
        print("TRIAL", a.trial, json.dumps(metric), flush=True)
    cache_archive.finish_run(base, store=llm.cache)


def cmd_natfail(a):
    """Run zs on all seed questions of the stream split; record the wrong SQLs (natural pollution material)."""
    cfg = C.load_config(a.config)
    model = C.resolve_model(a.model)
    llm = _llm(os.path.dirname(a.out) or ".")
    zs = ZeroShot(llm, model, cfg)
    all_items, eval_split, _ = stream_split(a.stream_split)
    items = [i for i in all_items if i.meta["role"] == "seed"]
    ev = Evaluator(eval_split, run_id="natfail")

    def one(it):
        sql = zs.run(it)["final_sql"]
        ex, _ = _score(ev, it, sql)
        return it.qid, sql, ex

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(one, items))
    nf = {q: s for q, s, e in rows if e == 0 and s}
    json.dump(nf, open(a.out, "w"), indent=1)
    print("zs on %d seed questions: %d failures stored in %s" % (len(items), len(nf), a.out))
    cache_archive.finish_run(os.path.dirname(os.path.abspath(a.out)), store=llm.cache)


def cmd_burnin(a):
    """M6 burn-in on POOL57 (dev gold never available as feedback): stream of n questions stratified by db (seed 20261028),
    batch-synchronous updates (batch = --batch items run concurrently against the same snapshot), then freeze."""
    cfg = resolved_cfg(C.load_config(a.config))
    model = C.resolve_model(a.model)
    llm = _llm(os.path.dirname(a.out) or ".")
    corpus = _pool_corpus(llm, cfg, "ws_full", dict(cfg["m6"].get("params") or {}), seed=20261028)
    pool = data.load_split("bird_train_pool", with_gold=True)
    by = defaultdict(list)
    for it in pool:
        by[it.db_id].append(it)
    rnd = random.Random(20261028)
    sel = []
    dbs = sorted(by)
    per = a.n // len(dbs) + 1
    for d in dbs:
        sel += rnd.sample(by[d], min(per, len(by[d])))
    rnd.shuffle(sel)
    sel = sel[: a.n]
    res = Resources(llm, cfg)
    system = FullSystem(llm, model, cfg)
    system.pipe.res = res
    ev = Evaluator("bird_train_pool", run_id="burnin")
    log = open(a.out + ".log.jsonl", "w")      # burn-in always replays from scratch (corpus state is not checkpointed)
    t0 = time.time()
    done_ids = set()
    for s in range(0, len(sel), a.batch):
        batch = sel[s:s + a.batch]
        corpus._matrix()

        def one(it):
            o = system.run(it, seed=0, corpus=corpus)
            ex, _ = _score(ev, it, o["final_sql"])
            return it, o, ex

        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            outs = list(ex.map(one, batch))
        for it, o, ex_ in outs:
            corpus.on_use(o["exemplar_ids"], it.qid, ex_)
            if not o["abstained"]:
                corpus.add_entry("bi:" + it.qid, it.db_id, it.question, it.evidence, o["final_sql"], o["critic"]["score"], ex_self=ex_,
                                 kind="exp", origin="burnin")
            log.write(json.dumps({"qid": it.qid, "ex": ex_, "abstained": o["abstained"]}) + "\n")
        corpus.evict()
        log.flush()
        print("[burnin] %d/%d %.0fs %s" % (min(s + a.batch, len(sel)), len(sel), time.time() - t0, json.dumps(corpus.stats())), flush=True)
    corpus.save(a.out)
    from .util import sha256_file
    print("saved", a.out, sha256_file(a.out))


def cmd_e4c(a):
    """Dev stream, no gold feedback. Requires a dev-lock entry for --lock-system (config incl. m6.path frozen corpus)."""
    cfg_path = a.config if os.path.isabs(a.config) else str(paths.REPO / a.config)
    ok, msg = lock.verify(a.lock_system, a.split, a.seed, cfg_path)
    if not ok:
        raise SystemExit("REFUSED: " + msg)
    cfg = resolved_cfg(C.load_config(cfg_path))
    model = C.resolve_model(a.model)
    llm = _llm(a.out, conc=a.conc)
    items = data.load_split(a.split)
    items = [items[j] for j in np.random.RandomState(a.perm).permutation(len(items))]
    if a.stream_limit:
        items = items[: a.stream_limit]
    params = dict(cfg["m6"].get("params") or {})
    params["phi"] = 0.0
    res = Resources(llm, cfg)
    base = res.corpus_for(items[0])      # frozen burn-in store (m6.path) or raw pool
    corpus = ExperienceCorpus(llm, cfg["models"]["embed"], a.policy, params, seed=a.perm)
    if cfg["m6"].get("path"):
        corpus = ExperienceCorpus.load(str(paths.REPO / cfg["m6"]["path"]), llm, cfg["models"]["embed"], a.policy, params, a.perm)
    else:
        corpus.add_history(data.load_split("bird_train_pool", with_gold=True))
    system = FullSystem(llm, model, cfg)
    system.pipe.res = res
    ev = Evaluator(a.split, run_id="e4c-%s-perm%d" % (a.policy, a.perm))
    os.makedirs(a.out, exist_ok=True)
    f = open(os.path.join(a.out, "events.jsonl"), "w")      # a restarted stream replays from scratch: rewrite, never duplicate
    pol = POLICIES[a.policy]
    for i, it in enumerate(items):
        o = system.run(it, seed=a.seed, corpus=corpus)
        ex, _ = _score(ev, it, o["final_sql"])
        f.write(json.dumps({"i": i, "qid": it.qid, "ex": ex, "abstained": o["abstained"], "score": o["critic"]["score"] if o["critic"] else None}) + "\n")
        f.flush()
        if pol["write"] != "none" and not o["abstained"]:
            corpus.on_use(o["exemplar_ids"], it.qid, None)
            corpus.add_entry("e4c:" + it.qid, it.db_id, it.question, it.evidence, o["final_sql"], o["critic"]["score"], ex_self=None, kind="exp")
            corpus.evict()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("e4", "natfail", "burnin", "e4c"):
        s = sub.add_parser(name)
        s.add_argument("--config", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--model", default="primary_actor")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--workers", type=int, default=4)
        s.add_argument("--conc", type=int, default=8)
        if name in ("e4", "natfail"):
            s.add_argument("--stream-split", default="HR", choices=["HR", "HT"], help="HR = registered E4 stream; HT = tuning mini-stream")
        if name == "e4":
            s.add_argument("--policy", required=True, choices=list(POLICIES))
            s.add_argument("--policy-params", default=None, help='JSON merged into m6.params (tuning), e.g. \'{"eta":0.5,"rho":0}\'')
            s.add_argument("--trial", default=None, help="tuning trial name: units go to <out>/<trial>/, trial_summary.json + exp/tuning_log.jsonl line")
            s.add_argument("--pollution", type=int, required=True)
            s.add_argument("--perm", type=int, required=True)
            s.add_argument("--dbs", default=None)
            s.add_argument("--natfail", default=None)
            s.add_argument("--stream-limit", type=int, default=None)
            s.add_argument("--probe-limit", type=int, default=None)
            s.add_argument("--checkpoints", default=None, help="probe checkpoints in %% of the stream, must include 0 and 100 (default 0,40,100; "
                           "\"0,100\" drops the mid-stream probe: 603 + 2x330 instead of 603 + 3x330 runs per E4 job)")
        if name == "burnin":
            s.add_argument("--n", type=int, default=1500)
            s.add_argument("--batch", type=int, default=16)
        if name == "e4c":
            s.add_argument("--policy", required=True, choices=["ws_full", "static"])
            s.add_argument("--perm", type=int, required=True)
            s.add_argument("--split", default="bird_dev")
            s.add_argument("--lock-system", default="e4c")
            s.add_argument("--stream-limit", type=int, default=None)
    a = ap.parse_args()
    {"e4": cmd_e4, "natfail": cmd_natfail, "burnin": cmd_burnin, "e4c": cmd_e4c}[a.cmd](a)


if __name__ == "__main__":
    main()
