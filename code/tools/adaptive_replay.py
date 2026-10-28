#!/usr/bin/env python3
"""Offline replay of the adaptive path (route.adaptive, X7) on the stored X5 consistency runs; NO LLM call (worker machine venv; numpy).

  adaptive_replay.py --set HR=$R/runs/X5/2-hr-cons:$R/runs/X5/1b-hr-gated --set HT=$R/runs/X5/2-ht-cons:$R/runs/X5/1b-ht-gated
                     [--thetas 1.0,0.75,0.5] [--ks 4,2] [--empty vote,demote] [--procs 3] [--boot 2000]
                     [--samples-cache DIR] --out results/x7/adaptive_replay.json

A consistency run (full_gated_cons.yaml: k = 4 extra T = 0.7 samples of the root prompt in ONE n = 4 request) stored the root, the final
answer and, in its call-cache archive (`llm_cache.jsonl.gz`), the four sample SQL texts. This tool recovers the samples, executes root and
samples read-only on the BIRD-train databases, scores each against gold (EX; gold is used for scoring only) and then applies the SAME decision
function as the live system (`nl2sql.systems.ours.route.adaptive_decide`) for every setting:
    consistency (share of samples with the root's result set) >= theta_fast  ->  the root ("fast")
    otherwise the execution-result majority over root + samples ("vote"; ties -> root; `empty=demote`: non-empty groups outrank empty ones).
Cost per question = the calls the live adaptive path makes: M1 + generation + ONE sampling request (n <= 4 samples = 1 call) = 3.0 chat calls
(no critic, no revision); tokens are the stored M1 / generation / sampling tokens (sampling completion scaled by k / 4 when k < 4).
Reference rows: root only (no sampling), `gated` = the final answer of the same consistency run (execution-gated loop, identical roots; its
calls are M1 + generation + revisions, the V3 / sampling calls of that run are NOT counted), the independent gated run (`:GATED_RUN`, other
root draws: an API non-determinism floor), vote_always (theta > 1), and the oracle (any of root + k samples correct; an upper bound, not a policy).
Differences vs `gated` are paired on qid, db-clustered 95 % percentile bootstrap (databases resampled with replacement).
A sample SQL that is byte-identical in two runs is executed once per run here; the replay executes with `execute_robust` (30 s, cold retry) like the scorer."""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from x7_common import HERE, load_items, load_preds, load_root_cands, paired, read_archive, read_jsonl, split_of  # noqa: F401  (sys.path set there)

_W = {}


def _init(split):
    from nl2sql.eval.score import Evaluator
    _W["ev"] = Evaluator(split, run_id="x7-adaptive-replay")
    _W["items"] = load_items(split)


def _exec_one(item, sql):
    """-> {"ok", "key", "empty", "ex", "err"}; key = order/duplicate-insensitive result digest (equal <=> equal result sets), None if the execution failed."""
    from nl2sql import db
    from nl2sql.eval.score import rows_digest
    from nl2sql.systems.ours import m5 as M5
    if not sql or not sql.strip():
        return {"ok": False, "key": None, "empty": False, "ex": 0, "err": "empty sql"}
    res = db.execute_robust(item.db_path, sql, timeout_s=30, max_rows=None)
    sc = _W["ev"].score_sql(item, sql, pred_res=res)
    key = rows_digest(res["rows"][:10000]) if res["ok"] else None
    return {"ok": bool(res["ok"]), "key": key, "empty": bool(res["ok"] and M5.exec_status(res) == "ok_empty"), "ex": int(sc.get("ex") or 0),
            "err": None if res["ok"] else str(res.get("error"))[:80]}


def _collect_one(args):
    qid, root_sql, samples = args
    item = _W["items"][qid]
    return qid, {"root": _exec_one(item, root_sql), "samples": [_exec_one(item, s) for s in samples]}


def map_samples(archive, roots, items):
    """{qid: [sample sql, ...]} from the n > 1 requests of the archive. A sampling request has the same messages as the root generation call
    (temperature 0, n absent), whose answer is the root SQL stored in cands.jsonl; the question text disambiguates equal root SQLs."""
    from nl2sql.util import strip_sql
    gen = {}
    samp = []
    for r in archive:
        mj = json.dumps(r["messages"], sort_keys=True)
        n = r["params"].get("n")
        if n and int(n) > 1:
            samp.append((mj, r))
        elif float(r["params"].get("temperature") or 0.0) == 0.0 and isinstance(r["text"], str):
            gen.setdefault(mj, []).append(strip_sql(r["text"]))
    by_sql = {}
    for q, c in roots.items():
        by_sql.setdefault(c["sql"].strip(), []).append(q)
    out, ambiguous, unmatched = {}, 0, 0
    for mj, r in samp:
        user = r["messages"][-1]["content"]
        cand = []
        for t in gen.get(mj, []):
            cand.extend(by_sql.get(t.strip(), []))
        cand = [q for q in dict.fromkeys(cand) if items[q].question in user]
        if not cand:
            unmatched += 1
            continue
        if len(cand) > 1:
            ambiguous += 1
        texts = r["text"] if isinstance(r["text"], list) else [r["text"]]
        for q in cand:
            if q not in out:                              # first chunk (seed + 1) only
                out[q] = [strip_sql(t) for t in texts]
    return out, {"sampling_requests": len(samp), "unmatched": unmatched, "ambiguous": ambiguous}


def collect(cons_dir, procs, cache_dir=None):
    preds, roots = load_preds(cons_dir), load_root_cands(cons_dir)
    split = split_of(preds)
    cache_f = os.path.join(cache_dir, os.path.basename(os.path.abspath(cons_dir)) + ".samples.json") if cache_dir else None
    if cache_f and os.path.exists(cache_f):
        return json.load(open(cache_f)), preds, roots, split
    items = load_items(split)
    arch = os.path.join(cons_dir, "llm_cache.jsonl.gz")
    if not os.path.exists(arch):
        raise SystemExit("no llm_cache.jsonl.gz in %s (the sample texts live there)" % cons_dir)
    samples, info = map_samples(read_archive(arch), roots, items)
    todo = [(q, roots[q]["sql"], samples[q]) for q in preds if q in roots and q in samples]
    print("[collect] %s: %d questions, %d with samples %s" % (cons_dir, len(preds), len(todo), info), flush=True)
    todo.sort(key=lambda t: t[0])
    with Pool(procs, initializer=_init, initargs=(split,)) as p:
        rows = dict(p.imap_unordered(_collect_one, [todo[i] for i in range(len(todo))], chunksize=4))
    data = {"info": info, "split": split, "rows": rows}
    if cache_f:
        os.makedirs(cache_dir, exist_ok=True)
        json.dump(data, open(cache_f, "w"))
    return data, preds, roots, split


def stage_cost(row, stages, k=4):
    """(calls, prompt tokens, completion tokens) of the named stages of a stored preds row; sampling completion scaled to k samples."""
    st = row.get("stages") or {}
    c = p = o = 0
    for s in stages:
        v = st.get(s)
        if v:
            c += v.get("calls", 0)
            p += v.get("prompt", 0)
            o += v.get("completion", 0) * ((k / 4.0) if s == "cons" else 1)
    return c, p, o


def yuan(p, o, pin=2.0, pout=8.0):
    return (p * pin + o * pout) / 1e6


def replay_set(name, cons_dir, gated_dir, args):
    data, preds, roots, split = collect(cons_dir, args.procs, args.samples_cache)
    from nl2sql.systems.ours.route import adaptive_decide
    rows = data["rows"]
    qids = sorted(q for q in rows if q in preds and preds[q].get("ex") is not None and rows[q]["root"]["ex"] is not None)
    dbs = np.array([preds[q]["db_id"] for q in qids])
    n_miss = len(preds) - len(qids)
    # validation of the recovered data against the stored run: root EX re-scored vs cands.jsonl, sample bits vs stored cons_bits
    mism_root = sum(1 for q in qids if roots[q].get("ex") is not None and roots[q]["ex"] != rows[q]["root"]["ex"])
    bit_ok = bit_n = 0
    for q in qids:
        p = preds[q]
        if p.get("cons_bits") and p.get("final_sql", "").strip() == roots[q]["sql"].strip():
            bit_n += 1
            mine = [int(rows[q]["root"]["key"] is not None and s["key"] == rows[q]["root"]["key"]) for s in rows[q]["samples"]]
            bit_ok += int(mine == p["cons_bits"])
    root_ex = np.array([rows[q]["root"]["ex"] for q in qids], float)
    final_ex = np.array([preds[q]["ex"] for q in qids], float)
    out = {"split": split, "n": len(qids), "n_missing": n_miss, "databases": sorted(set(dbs)), "collect_info": data["info"],
           "validation": {"root_ex_mismatch_vs_cands": mism_root, "cons_bits_reproduced": "%d/%d" % (bit_ok, bit_n)}, "refs": {}, "configs": []}

    def ref(label, ex, calls, pt, ct):
        out["refs"][label] = {"ex": float(np.mean(ex)), "calls_per_q": float(np.mean(calls)), "yuan_per_q": float(np.mean([yuan(a, b) for a, b in zip(pt, ct)]))}

    c_root = [stage_cost(preds[q], ("m1", "gen")) for q in qids]
    c_gated = [stage_cost(preds[q], ("m1", "gen", "rev", "rev_t2")) for q in qids]
    ref("root_only", root_ex, [c[0] for c in c_root], [c[1] for c in c_root], [c[2] for c in c_root])
    ref("gated_same_roots", final_ex, [c[0] for c in c_gated], [c[1] for c in c_gated], [c[2] for c in c_gated])
    ex_ref_indep = None
    if gated_dir:
        gp = load_preds(gated_dir)
        both = [q for q in qids if q in gp and gp[q].get("ex") is not None]
        out["refs"]["gated_independent_run"] = {"ex": float(np.mean([gp[q]["ex"] for q in both])), "calls_per_q": float(np.mean([gp[q].get("calls", 0) for q in both])),
                                                "n": len(both)}
        ex_ref_indep = {q: gp[q]["ex"] for q in both}
    for k in args.ks:
        pool = np.array([max([rows[q]["root"]["ex"]] + [s["ex"] for s in rows[q]["samples"][:k]]) for q in qids], float)
        out["refs"]["oracle_root_plus_%d_samples" % k] = {"ex": float(pool.mean())}
        for emp in args.empty:
            for theta in args.thetas + [1.01]:            # 1.01 = vote_always
                ex_c, fast, ncons = [], [], []
                for q in qids:
                    r = rows[q]
                    cand = [r["root"]] + r["samples"][:k]
                    d = adaptive_decide([c["key"] for c in cand], [c["empty"] for c in cand], theta, "root", emp)
                    ex_c.append(cand[d["chosen"]]["ex"])
                    fast.append(d["path"] == "fast")
                    ncons.append(d["cons"])
                ex_c = np.array(ex_c, float)
                cs = [stage_cost(preds[q], ("m1", "gen", "cons"), k) for q in qids]
                cfg = {"k": k, "theta_fast": theta if theta <= 1 else "vote_always", "empty": emp, "ex": float(ex_c.mean()),
                       "frac_fast": float(np.mean(fast)), "calls_per_q": float(np.mean([c[0] for c in cs])),
                       "yuan_per_q": float(np.mean([yuan(c[1], c[2]) for c in cs])),
                       "vs_gated_same_roots": paired(dbs, final_ex, ex_c, args.boot), "vs_root_only": paired(dbs, root_ex, ex_c, args.boot)}
                if ex_ref_indep:
                    b = [q for q in qids if q in ex_ref_indep]
                    ix = [i for i, q in enumerate(qids) if q in ex_ref_indep]
                    cfg["vs_gated_independent_run"] = paired(dbs[ix], [ex_ref_indep[q] for q in b], ex_c[ix], args.boot)
                out["configs"].append(cfg)
    return out


def table(name, res):
    lines = ["%s (%s, n=%d, dbs=%d)  refs: %s" % (name, res["split"], res["n"], len(res["databases"]),
                                                  "; ".join("%s %.4f" % (k, v["ex"]) for k, v in res["refs"].items()))]
    lines.append("%-4s %-7s %-12s %7s %6s %7s %7s  %-28s %-28s" % ("k", "empty", "theta_fast", "EX", "fast", "calls", "yuan/q", "vs gated (same roots) pp [CI] W/L",
                                                                    "vs gated (indep run) pp [CI]"))
    for c in res["configs"]:
        g = c["vs_gated_same_roots"]
        i = c.get("vs_gated_independent_run")
        lines.append("%-4s %-7s %-12s %7.4f %6.2f %7.2f %7.4f  %+5.2f [%+5.2f,%+5.2f] %d/%d      %s" % (
            c["k"], c["empty"], c["theta_fast"], c["ex"], c["frac_fast"], c["calls_per_q"], c["yuan_per_q"], g["diff_pp"], g["ci95_pp"][0] if g["ci95_pp"] else float("nan"),
            g["ci95_pp"][1] if g["ci95_pp"] else float("nan"), g["b_right_a_wrong"], g["a_right_b_wrong"],
            ("%+5.2f [%+5.2f,%+5.2f]" % (i["diff_pp"], i["ci95_pp"][0], i["ci95_pp"][1])) if i and i["ci95_pp"] else ""))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", action="append", required=True, help="NAME=CONS_RUN_DIR[:GATED_RUN_DIR]")
    ap.add_argument("--thetas", default="1.0,0.75,0.5")
    ap.add_argument("--ks", default="4,2")
    ap.add_argument("--empty", default="vote,demote")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--samples-cache", default=None, help="directory for the recovered / executed samples (reused on re-runs)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    a.thetas = [float(x) for x in a.thetas.split(",")]
    a.ks = [int(x) for x in a.ks.split(",")]
    a.empty = a.empty.split(",")
    res = {}
    for s in a.set:
        name, rest = s.split("=", 1)
        cons, _, gated = rest.partition(":")
        res[name] = replay_set(name, cons, gated or None, a)
        print(table(name, res[name]), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1, default=str)
    return res


if __name__ == "__main__":
    main()
