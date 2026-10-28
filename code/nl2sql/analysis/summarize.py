"""Run summaries (smoke level; full statistics are S1's). results/<exp>.json flat keys per CONTRACT section 4."""
import json
import math
import os
import random
from collections import Counter, defaultdict


def _load(path):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def cluster_bootstrap_ci(vals, clusters, B=2000, seed=20261101, alpha=0.05):
    """Mean with db-clustered percentile bootstrap CI (resample clusters with replacement)."""
    by = defaultdict(list)
    for v, c in zip(vals, clusters):
        if v is not None:
            by[c].append(v)
    keys = sorted(by)
    if not keys:
        return None, None, None
    rnd = random.Random(seed)
    allv = [v for k in keys for v in by[k]]
    mean = sum(allv) / len(allv)
    if len(keys) < 2:
        return mean, None, None
    ms = []
    for _ in range(B):
        s = n = 0
        for _k in range(len(keys)):
            g = by[keys[rnd.randrange(len(keys))]]
            s += sum(g)
            n += len(g)
        ms.append(s / n)
    ms.sort()
    return mean, ms[int(alpha / 2 * B)], ms[min(B - 1, int((1 - alpha / 2) * B))]


def _pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summarize_run(preds_path):
    rows = _load(preds_path)
    n = len(rows)
    if n == 0:
        return {"n": 0}
    scored = [r for r in rows if r.get("ex") is not None]
    ex_vals = [r["ex"] for r in scored]
    acc = [r for r in scored if not r.get("abstained")]
    S = {"n": n, "n_scored": len(scored)}
    S["ex"] = sum(ex_vals) / len(ex_vals) if ex_vals else None
    S["ex_all_abstain_wrong"] = (sum(r["ex"] for r in acc) / len(scored)) if scored else None
    S["coverage"] = len(acc) / len(scored) if scored else None
    S["ex_accepted"] = (sum(r["ex"] for r in acc) / len(acc)) if acc else None
    m, lo, hi = cluster_bootstrap_ci(ex_vals, [r["db_id"] for r in scored])
    S["ex_ci"] = [lo, hi]
    for k in ("soft_f1", "jaccard", "overlap"):
        v = [r[k] for r in scored if r.get(k) is not None]
        S[k] = sum(v) / len(v) if v else None
    S["exec_ok_rate"] = sum(1 for r in rows if r.get("exec_ok")) / n
    for k, name in (("calls", "calls_per_q"), ("prompt_tokens", "prompt_tokens_per_q"), ("completion_tokens", "completion_tokens_per_q"),
                    ("cached_tokens", "cached_tokens_per_q"), ("embed_calls", "embed_calls_per_q"), ("rerank_calls", "rerank_calls_per_q"),
                    ("exec_s", "exec_s_per_q")):
        S[name] = sum(r.get(k, 0) or 0 for r in rows) / n
    lat = [r.get("latency_s", 0) for r in rows]
    S["latency_mean_s"] = sum(lat) / n
    S["latency_p50_s"] = _pct(lat, 0.5)
    S["latency_p95_s"] = _pct(lat, 0.95)
    S["api_fail"] = sum(1 for r in rows if r.get("error_class") == "api_fail")
    S["parse_fail"] = sum(1 for r in rows if r.get("error_class") == "parse_fail")
    S["sys_error"] = sum(1 for r in rows if str(r.get("error_class")).startswith("sys_error"))
    S["gold_fail"] = sum(1 for r in rows if r.get("auto_class") == "gold_fail")
    S["error_breakdown"] = dict(Counter((r.get("auto_class") or r.get("error_class")) for r in rows if r.get("ex") == 0 or r.get("ex") is None and r.get("error_class")))
    # per-stage token accounting (tag -> per-question means over ALL questions)
    st = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "cached": 0, "latency_s": 0.0})
    for r in rows:
        for tag, v in (r.get("stages") or {}).items():
            for k in st[tag]:
                st[tag][k] += v.get(k, 0)
    S["stages_per_q"] = {tag: {k: round(v / n, 3) for k, v in d.items()} for tag, d in sorted(st.items())}
    S["actions"] = dict(Counter(a for r in rows for a in (r.get("actions") or [])))
    S["iterations_mean"] = sum(r.get("iterations", 0) or 0 for r in rows) / n
    byd = defaultdict(list)
    for r in scored:
        byd[r["db_id"]].append(r["ex"])
    ts = [r["ex_ts"] for r in scored if r.get("ex_ts") is not None]      # Spider test-suite EX, only when scored
    if ts:
        S["ex_ts"] = sum(ts) / len(ts)
        S["ex_ts_n"] = len(ts)
        S["ex_ts_ci"] = list(cluster_bootstrap_ci([r["ex_ts"] for r in scored if r.get("ex_ts") is not None],
                                                  [r["db_id"] for r in scored if r.get("ex_ts") is not None])[1:])
    vp = os.path.join(os.path.dirname(os.path.abspath(preds_path)), "ves.jsonl")     # written by eval/ves_score.py
    if os.path.exists(vp):
        from ..eval.ves_score import summarize_ves
        v = summarize_ves(vp, preds_path)
        for k in ("ves", "rves", "ves_ci", "rves_ci"):
            if k in v:
                S[k] = v[k]
        S["ves_coverage"] = v.get("coverage")
    S["ex_by_db"] = {d: [round(sum(v) / len(v), 4), len(v)] for d, v in sorted(byd.items())}
    return S


def to_flat(exp, system, modelkey, summ, extra_prefix=""):
    """Flat keys <exp>.<system>.<modelkey>.<metric>[.mean|ci_lo|ci_hi|n] for results/<exp>.json."""
    p = "%s.%s.%s" % (exp, system, modelkey)
    out = {}
    for k in ("ex", "ex_ts", "ves", "rves", "ex_all_abstain_wrong", "coverage", "ex_accepted", "soft_f1", "jaccard", "overlap", "calls_per_q",
              "prompt_tokens_per_q", "completion_tokens_per_q", "cached_tokens_per_q", "latency_mean_s", "latency_p95_s"):
        if summ.get(k) is not None:
            out["%s.%s" % (p, k)] = summ[k]
    if summ.get("ex_ci") and summ["ex_ci"][0] is not None:
        out[p + ".ex.ci_lo"], out[p + ".ex.ci_hi"] = summ["ex_ci"]
    if summ.get("ex_ts_ci") and summ["ex_ts_ci"][0] is not None:
        out[p + ".ex_ts.ci_lo"], out[p + ".ex_ts.ci_hi"] = summ["ex_ts_ci"]
    out[p + ".n"] = summ["n"]
    return out


if __name__ == "__main__":
    import sys
    print(json.dumps(summarize_run(sys.argv[1]), indent=1))
