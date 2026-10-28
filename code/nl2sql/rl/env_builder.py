"""E6 cached branching environment builder (HR only). Per question a tree:
root = M1+M3+P-GEN+Critic (tier1, T=0). depth1: revise x2 (T=0; T=0.7 seed 7), re_retrieve x2 (same), escalate x1.
depth2: from each depth-1 node revise, re_retrieve, escalate (escalate only from tier-1 nodes) at T=0; depth3: revise from each
depth-2 node. Nodes of the same action with identical SQL under one parent are merged (weight = multiplicity). Gold EX is stored
for rewards only (never in the state). Resumable: one JSON line per question.

python -m nl2sql.rl.env_builder --config exp/configs/full_fixed.yaml --out <dir>/trees.jsonl [--limit N] [--workers K] [--folds 0,1]
                                 [--shard k/n] [--skip-done-from other_trees.jsonl,...]

Any exception while building a tree (not only LLMError) becomes a record in <out>.errors.jsonl and the run goes on; usage goes to
<dir>/usage.jsonl (cost_report.py also globs usage*.jsonl, so old usage_builder.jsonl files are counted too)."""
import argparse
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from .. import cache_archive, config as C
from .. import data, paths
from ..eval.score import Evaluator
from ..llm import LLMClient, LLMError
from ..meter import Meter
from ..systems.ours.pipeline import FullFixed, resolved_cfg
from ..util import json_default

ACTOR_TAGS = ("gen", "gen_rr", "gen_rr_t2", "rev", "rev_t2")


def node_record(nid, parent, node, ev, item, weight=1):
    st = node["usage"]["stages"]
    tok_actor = sum(v["prompt"] + v["completion"] for k, v in st.items() if k in ACTOR_TAGS)
    tok_crit = sum(v["prompt"] + v["completion"] for k, v in st.items() if k == "crit")
    tok_root_extra = sum(v["prompt"] + v["completion"] for k, v in st.items() if k == "m1")
    sc = ev.score_sql(item, node["sql"])          # fresh full execution (the node's own result is capped at 10k rows)
    c = node["critic"]
    return {"id": nid, "parent": parent, "action": node["action"], "weight": weight, "sql": node["sql"], "tier": node["tier"],
            "k": node["k"], "state": [float(x) for x in node["_state"]], "ex": sc.get("ex"), "exec_ok": sc.get("exec_ok"),
            "score": c["score"], "syntax": c["syntax"], "semantic": c["semantic"], "efficiency": c["efficiency"], "status": c["status"],
            "calls": node["usage"]["calls"], "tok_actor": tok_actor, "tok_crit": tok_crit, "tok_m1": tok_root_extra,
            "prompt_tokens": node["usage"]["prompt_tokens"], "completion_tokens": node["usage"]["completion_tokens"],
            "latency_s": node["usage"]["latency_s"]}


def build_tree(sys_, ev, item, seed=0):
    pipe = sys_.pipe
    root = pipe.root(item, seed, T=0.0)
    root["_state"] = pipe.state(root)
    recs = [node_record(0, None, root, ev, item)]
    frontier = [(0, root)]

    def expand(pid, pnode, specs):
        out = []
        seen = {}
        for action, T, sd in specs:
            if action == "escalate" and (pnode["tier"] == 1 or not pipe.tier2):
                continue
            ch = pipe.act(item, pnode, action, T=T, seed=sd)
            ch["_state"] = pipe.state(ch)
            key = (action, ch["sql"].strip())
            if key in seen:
                recs[seen[key]]["weight"] += 1
                continue
            nid = len(recs)
            seen[key] = nid
            recs.append(node_record(nid, pid, ch, ev, item))
            out.append((nid, ch))
        return out

    d1 = expand(0, root, [("revise", 0.0, seed), ("revise", 0.7, 7), ("re_retrieve", 0.0, seed), ("re_retrieve", 0.7, 7), ("escalate", 0.0, seed)])
    d2 = []
    for pid, pn in d1:
        d2 += expand(pid, pn, [("revise", 0.0, seed), ("re_retrieve", 0.0, seed), ("escalate", 0.0, seed)])
    for pid, pn in d2:
        expand(pid, pn, [("revise", 0.0, seed)])
    return {"qid": item.qid, "db_id": item.db_id, "fold": item.meta.get("fold"), "nodes": recs}


def shard_items(items, shard):
    """'k/n' -> the items with index % n == k (index in the full, unfiltered order, so a restart keeps the assignment)."""
    if not shard:
        return items
    k, n = (int(x) for x in str(shard).split("/"))
    assert 0 <= k < n, "bad --shard %s" % shard
    return [i for j, i in enumerate(items) if j % n == k]


def read_done(paths_):
    done = set()
    for p in paths_:
        if p and os.path.exists(p):
            for l in open(p):
                try:
                    r = json.loads(l)
                    if "nodes" in r:                 # only real trees count as done; error records are retried
                        done.add(r["qid"])
                except Exception:
                    pass
    return done


def build_one(sys_, ev, it, seed=0):
    """-> ("tree", dict) | ("error", dict). Any exception (not only LLMError) becomes an error record: the caller's loop must go on."""
    try:
        t = build_tree(sys_, ev, it, seed)
        json.dumps(t, default=json_default)          # surface non-serialisable values here, with the qid, not in the writer
        return "tree", t
    except Exception as e:
        return "error", {"qid": it.qid, "db_id": it.db_id, "error": "%s: %s" % (type(e).__name__, str(e)[:200]),
                         "trace": traceback.format_exc()[-500:], "ts": time.strftime("%Y-%m-%d %H:%M:%S")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="trees.jsonl (one JSON line per question; resumable). Errors -> <out>.errors.jsonl, usage -> <dir>/usage.jsonl")
    ap.add_argument("--model", default="primary_actor")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--folds", default=None)
    ap.add_argument("--split", default="bird_train_HR")
    ap.add_argument("--shard", default=None, help="k/n: build only items with index %% n == k (several processes split one fold; give each its own --out)")
    ap.add_argument("--skip-done-from", default=None, help="comma-separated trees.jsonl files whose questions are skipped (e.g. other shards' outputs)")
    ap.add_argument("--cache", default=None, help="call-cache dir (default <paths.CACHE>/llm, machine-local)")
    a = ap.parse_args(argv)
    cfg = resolved_cfg(C.load_config(a.config if os.path.isabs(a.config) else str(paths.REPO / a.config)))
    model = C.resolve_model(a.model)
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    llm = LLMClient(a.cache or str(paths.CACHE / "llm"), os.path.join(out_dir, "usage.jsonl"), paths.KEY_FILE, a.conc)
    sys_ = FullFixed(llm, model, cfg)
    ev = Evaluator(a.split, run_id="env_builder")
    items = data.load_split(a.split)
    if a.folds:
        fs = {int(x) for x in a.folds.split(",")}
        items = [i for i in items if i.meta.get("fold") in fs]
    if a.limit:
        items = items[: a.limit]
    items = shard_items(items, a.shard)
    done = read_done([a.out] + (a.skip_done_from.split(",") if a.skip_done_from else []))
    todo = [i for i in items if i.qid not in done]
    print("[env] n=%d todo=%d shard=%s out=%s" % (len(items), len(todo), a.shard, a.out), flush=True)
    t0 = time.time()
    k = n_err = 0
    err_path = a.out + ".errors.jsonl"
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(build_one, sys_, ev, it): it for it in todo}
        for fu in as_completed(futs):
            try:
                kind, t = fu.result()
            except Exception as e:                    # build_one cannot raise, but the loop must never end early
                kind, t = "error", {"qid": futs[fu].qid, "error": "%s: %s" % (type(e).__name__, str(e)[:200])}
            if kind == "error":
                n_err += 1
                print("[env] tree failed %s: %s" % (t["qid"], t["error"][:160]), flush=True)
                with open(err_path, "a") as f:
                    f.write(json.dumps(t, default=json_default) + "\n")
                continue
            with open(a.out, "a") as f:
                f.write(json.dumps(t, default=json_default) + "\n")
            k += 1
            if k % 10 == 0:
                print("[env] %d/%d %.0fs errors=%d" % (k, len(todo), time.time() - t0, n_err), flush=True)
    print("[env] done built=%d errors=%d (errors are retried on restart)" % (k, n_err), flush=True)
    if n_err:
        print("[env] WARNING: %d trees failed, see %s" % (n_err, err_path), flush=True)
    cache_archive.finish_run(out_dir, store=llm.cache)
    llm.close()


if __name__ == "__main__":
    main()
