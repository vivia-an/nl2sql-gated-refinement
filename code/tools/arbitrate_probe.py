#!/usr/bin/env python3
"""Judge probe for arbitration (X5, HEADROOM section 5 row 3). Replays the stored E6 trees (BIRD-train HR only): for every question the
root is compared with its depth-1 candidates of the chosen action(s); whenever both execute and the result sets differ, ONE judge call
(`Pipeline.arbitrate`, the same prompt as m5.arbitrate: question, evidence, schema, both SQL + result previews, A/B order flipped by qid
hash) picks a winner, which then faces the next candidate (knock-out). The stored EX labels are used only for scoring the picks.

  arbitrate_probe.py --trees $R/runs/P2_E6/trees_1101.jsonl --model qwen37plus --out $R/runs/X5/arb_probe.json
                     [--actions revise] [--limit N] [--workers 8] [--conc 8] [--quote-keywords] [--qids-file F]

Reports: n questions, n disagreeing pairs, n mixed pairs (exactly one side correct), judge accuracy on mixed pairs (success bar in
HEADROOM: >= ~70 %; always taking the revision wins only 40 % there), and the implied EX of "root + judged revision" vs the root alone and
vs revise-always, with the db-clustered 95 % CI of the paired differences. Prompt approximations (tree nodes do not store them): no M1 graph,
no exemplars / business terms / lineage hints; the schema block is the full-schema DDL (which every HR/HT question used anyway), without the
`likely relevant` flags. No dev / mini-dev / Spider data is touched (`bird_train_HR` items only)."""
import argparse
import json
import os
import sys
import types
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import numpy as np  # noqa: E402


def make_node(pipe, item, sql, sections):
    from nl2sql import db
    res = db.execute(item.db_path, sql, timeout_s=30, max_rows=10000)
    return {"sql": sql, "res": res, "rkey": db.rows_key(res["rows"]) if res["ok"] else None, "graph": None,
            "ctx": types.SimpleNamespace(sections_core=lambda s=sections: s)}


def probe_question(pipe, item, tree, actions, sections, seed=0):
    """-> record for one tree. Nodes: tree['nodes'] with id, parent, action, sql, ex."""
    nodes = tree["nodes"]
    root = nodes[0]
    cands = [n for n in nodes if n["parent"] == root["id"] and n["action"] in actions]
    rn = make_node(pipe, item, root["sql"], sections)
    champ_n, champ_t = rn, root
    calls, picks = 0, []
    for c in cands:
        cn = make_node(pipe, item, c["sql"], sections)
        if not (champ_n["res"]["ok"] and cn["res"]["ok"]) or champ_n["rkey"] == cn["rkey"]:
            continue
        chosen, info = pipe.arbitrate(item, champ_n, cn, seed)
        calls += 1
        won = "cand" if chosen is cn else "champ"
        picks.append({"cand_id": c["id"], "action": c["action"], "champ_ex": champ_t["ex"], "cand_ex": c["ex"], "pick": won,
                      "parse_ok": info.get("arb_parse_ok"), "flip": info.get("arb_flip")})
        if chosen is cn:
            champ_n, champ_t = cn, c
    first = next((n for n in cands if n["action"] == actions[0]), None)
    return {"qid": tree["qid"], "db": tree["db_id"], "root_ex": root["ex"], "judged_ex": champ_t["ex"],
            "revise_always_ex": (first["ex"] if first is not None else root["ex"]), "n_cands": len(cands), "calls": calls, "picks": picks}


def summarize(recs, boot=1000, seed=20261101):
    n = len(recs)
    picks = [dict(p, qid=r["qid"], db=r["db"]) for r in recs for p in r["picks"]]
    mixed = [p for p in picks if p["champ_ex"] != p["cand_ex"] and None not in (p["champ_ex"], p["cand_ex"])]
    right = [p for p in mixed if (p["pick"] == "cand") == (p["cand_ex"] == 1)]
    cand_right = [p for p in mixed if p["cand_ex"] == 1]
    out = {"n_questions": n, "n_pairs_disagree": len(picks), "n_mixed": len(mixed),
           "judge_acc_on_mixed": (len(right) / float(len(mixed))) if mixed else None,
           "revise_always_acc_on_mixed": (len(cand_right) / float(len(mixed))) if mixed else None,
           "judge_calls": int(sum(r["calls"] for r in recs)), "judge_calls_per_q": sum(r["calls"] for r in recs) / float(max(n, 1)),
           "unparsed": int(sum(1 for p in picks if p["parse_ok"] is False))}
    if n:
        ex = {k: np.array([r[k] if r[k] is not None else 0 for r in recs], float) for k in ("root_ex", "judged_ex", "revise_always_ex")}
        dbs = np.array([r["db"] for r in recs])
        u = np.unique(dbs)
        mem = {d: np.where(dbs == d)[0] for d in u}
        rng = np.random.RandomState(seed)

        def ci(f):
            vals = []
            for _ in range(boot):
                idx = np.concatenate([mem[d] for d in rng.choice(u, len(u), replace=True)])
                vals.append(f(idx))
            return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        for k, v in ex.items():
            out["ex_" + k] = float(v.mean())
        out["judged_minus_root_pp"] = {"diff": 100 * float((ex["judged_ex"] - ex["root_ex"]).mean()),
                                       "ci": [100 * x for x in ci(lambda i: (ex["judged_ex"][i] - ex["root_ex"][i]).mean())]}
        out["judged_minus_revise_always_pp"] = {"diff": 100 * float((ex["judged_ex"] - ex["revise_always_ex"]).mean()),
                                                "ci": [100 * x for x in ci(lambda i: (ex["judged_ex"][i] - ex["revise_always_ex"][i]).mean())]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trees", required=True)
    ap.add_argument("--split", default="bird_train_HR", choices=["bird_train_HR", "bird_train_HT_full"])
    ap.add_argument("--model", default="qwen37plus")
    ap.add_argument("--actions", default="revise")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--qids-file", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--quote-keywords", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None)
    a = ap.parse_args(argv)
    from nl2sql import config as C, data, paths
    from nl2sql.llm import LLMClient
    from nl2sql.systems.ours.pipeline import Pipeline
    model = C.resolve_model(a.model)
    items = {i.qid: i for i in data.load_split(a.split)}
    trees = [json.loads(l) for l in open(a.trees) if l.strip()]
    trees = [t for t in trees if t["qid"] in items]
    if a.qids_file:
        keep = {l.strip() for l in open(a.qids_file) if l.strip()}
        trees = [t for t in trees if t["qid"] in keep]
    if a.limit:
        trees = trees[:a.limit]
    actions = [x for x in a.actions.split(",") if x]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    llm = LLMClient(cache_path=a.cache or str(paths.CACHE / "llm"), usage_path=a.out + ".usage.jsonl", key_file=paths.KEY_FILE,
                    max_concurrency_per_model=a.conc)
    pipe = Pipeline(llm, model, {"ddl": {"quote_keywords": a.quote_keywords}})

    def one(t):
        it = items[t["qid"]]
        sc = data.get_schema(it)
        sections = "# Schema\n" + data.render_ddl(sc, quote_keywords=a.quote_keywords)
        return probe_question(pipe, it, t, actions, sections)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        recs = list(ex.map(one, trees))
    summ = summarize(recs)
    summ.update({"trees": a.trees, "model": model, "actions": actions, "quote_keywords": a.quote_keywords})
    json.dump({"summary": summ, "records": recs}, open(a.out, "w"), indent=1, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    print(json.dumps(summ, indent=1))
    llm.close()


if __name__ == "__main__":
    main()
