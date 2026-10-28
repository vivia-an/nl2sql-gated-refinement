"""汇总 runs/**/usage*.jsonl（含 usage_builder.jsonl / usage_batch.jsonl）的花费（元）。python3 cost_report.py <runs_dir> [--by exp|model|tag]。
我们自己的缓存命中（cached=true）不计费；隐式缓存 cached_tokens 按 cache_hit_ratio 计；batch 按 batch_ratio。"""
import argparse, collections, glob, json, os, sys

def load_prices(path):
    out = {"models": {}, "cache_hit_ratio": 0.2, "batch_ratio": 0.5}
    cur = None
    for line in open(path):
        s = line.split("#")[0].rstrip()
        if not s.strip():
            continue
        if s.startswith("cache_hit_ratio"):
            out["cache_hit_ratio"] = float(s.split(":")[1])
        elif s.startswith("batch_ratio"):
            out["batch_ratio"] = float(s.split(":")[1])
        elif "{in:" in s:
            name, rest = s.strip().split("{", 1)
            name = name.rstrip(": ")
            d = {}
            for kv in rest.rstrip("}").split(","):
                k, v = kv.split(":")
                d[k.strip()] = float(v)
            out["models"][name.strip()] = d
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--prices", default=os.path.join(os.path.dirname(__file__), "..", "..", "exp", "prices.yaml"))
    ap.add_argument("--by", default="exp", choices=["exp", "model", "tag", "run"])
    a = ap.parse_args()
    P = load_prices(a.prices)
    agg = collections.defaultdict(lambda: [0, 0.0, 0, 0, 0])  # n_calls, yuan, pt, ct, unpriced
    for f in glob.glob(os.path.join(a.runs, "**", "usage*.jsonl"), recursive=True):
        rel = os.path.relpath(f, a.runs).split(os.sep)
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("cached"):
                continue
            m = r.get("model", "?")
            pr = P["models"].get(m)
            pt = r.get("pt", 0) or 0
            ct = r.get("ct", 0) or 0
            key = {"exp": rel[0], "run": "/".join(rel[:-1]), "model": m, "tag": r.get("tag", r.get("kind", "?"))}[a.by]
            g = agg[key]
            g[0] += 1; g[2] += pt; g[3] += ct
            if pr is None:
                g[4] += 1
                continue
            cached = min(r.get("cached_tokens", 0) or 0, pt)
            y = ((pt - cached) * pr["in"] + cached * pr["in"] * P["cache_hit_ratio"] + ct * pr["out"]) / 1e6
            if r.get("mode") == "batch":
                y *= P["batch_ratio"]
            g[1] += y
    tot = 0.0
    print("%-48s %9s %10s %12s %10s" % (a.by, "calls", "yuan", "prompt_tok", "compl_tok"))
    for k, g in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print("%-48s %9d %10.2f %12d %10d%s" % (k[:48], g[0], g[1], g[2], g[3], ("  unpriced=%d" % g[4]) if g[4] else ""))
        tot += g[1]
    print("TOTAL yuan = %.2f" % tot)

if __name__ == "__main__":
    main()
