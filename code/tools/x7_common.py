"""Shared helpers of the X7 offline tools (adaptive_replay.py, value_check.py, empty_roots.py). Worker machine venv (needs the nl2sql package, numpy).
No LLM call anywhere; BIRD-train tuning splits only (HT / HR); gold SQL is used for ANALYSIS and is never shown to a system."""
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

TUNING_SPLITS = ("bird_train_HR", "bird_train_HT")


def read_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def run_dir(p):
    """Directory of a run (accepts the directory or its preds.jsonl)."""
    return os.path.dirname(os.path.abspath(p)) if p.endswith(".jsonl") else os.path.abspath(p)


def load_preds(p):
    """{qid: row} of a run (directory or preds.jsonl)."""
    d = run_dir(p)
    return {r["qid"]: r for r in read_jsonl(os.path.join(d, "preds.jsonl"))}


def load_root_cands(p):
    """{qid: candidate row with cand_idx 0} of a run (cands.jsonl)."""
    f = os.path.join(run_dir(p), "cands.jsonl")
    out = {}
    if os.path.exists(f):
        for c in read_jsonl(f):
            if c.get("cand_idx") == 0:
                out[c["qid"]] = c
    return out


def split_of(preds):
    sp = {r.get("split") for r in preds.values()}
    sp.discard(None)
    if len(sp) != 1:
        raise SystemExit("run mixes splits: %s" % sp)
    s = sp.pop()
    if s not in TUNING_SPLITS:
        raise SystemExit("X7 tools work on tuning splits only (%s), got %r" % (", ".join(TUNING_SPLITS), s))
    return s


def load_items(split):
    """{qid: Item} with gold SQL (tuning splits only)."""
    if split not in TUNING_SPLITS:
        raise SystemExit("X7 tools work on tuning splits only, got %r" % split)
    from nl2sql import data
    return {i.qid: i for i in data.load_split(split, with_gold=True)}


def read_archive(path):
    """llm_cache.jsonl.gz -> list of dict(messages, params, text) (text: str or list of str for n > 1)."""
    out = []
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            req = json.loads(r["req"])
            resp = json.loads(r["resp"])
            out.append({"messages": req["messages"], "params": req["params"], "text": resp.get("text"), "key": r.get("key")})
    return out


def bootstrap_diff(dbs, d, B=2000, seed=20261101):
    """db-clustered percentile CI of mean(d) (databases resampled with replacement) - same convention as tools/paired_compare.py."""
    import numpy as np
    dbs, d = np.asarray(dbs), np.asarray(d, float)
    u = np.unique(dbs)
    if len(u) < 2:
        return None
    rng = np.random.RandomState(seed)
    mem = {x: np.where(dbs == x)[0] for x in u}
    vals = []
    for _ in range(B):
        idx = np.concatenate([mem[x] for x in rng.choice(u, len(u), replace=True)])
        vals.append(d[idx].mean())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def paired(dbs, a, b, B=2000):
    """Paired comparison b - a (a, b: 0/1 arrays over the same questions): diff in pp, db-clustered 95 % CI in pp, discordant counts."""
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    ci = bootstrap_diff(dbs, d, B)
    return {"n": int(len(d)), "ex_a": float(a.mean()) if len(a) else None, "ex_b": float(b.mean()) if len(b) else None,
            "diff_pp": 100 * float(d.mean()) if len(d) else None, "ci95_pp": [100 * x for x in ci] if ci else None,
            "b_right_a_wrong": int(((b == 1) & (a == 0)).sum()), "a_right_b_wrong": int(((a == 1) & (b == 0)).sum()),
            "n_databases": int(len(np.unique(dbs)))}
