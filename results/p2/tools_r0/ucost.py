"""cost over usage*.jsonl under given dirs (cost_report.py only globs usage.jsonl; env_builder writes usage_builder.jsonl).
python ucost.py <prices.yaml> <dir> [<dir>...]  -> yuan by model/exp"""
import sys, os, glob, json, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/tools")
import cost_report as cr
P = cr.load_prices(sys.argv[1])
by = collections.defaultdict(lambda: [0, 0.0, 0, 0])
for d in sys.argv[2:]:
    for f in glob.glob(os.path.join(d, "**", "usage*.jsonl"), recursive=True):
        rel = os.path.relpath(f, d).split(os.sep)[0]
        for line in open(f):
            try: r = json.loads(line)
            except ValueError: continue
            if r.get("cached"): continue
            pr = P["models"].get(r.get("model"))
            pt = r.get("pt", 0) or 0; ct = r.get("ct", 0) or 0
            g = by[(rel, r.get("model"))]
            g[0] += 1; g[2] += pt; g[3] += ct
            if pr is None: continue
            c = min(r.get("cached_tokens", 0) or 0, pt)
            y = ((pt - c) * pr["in"] + c * pr["in"] * P["cache_hit_ratio"] + ct * pr["out"]) / 1e6
            if r.get("mode") == "batch": y *= P["batch_ratio"]
            g[1] += y
tot = 0
for k, g in sorted(by.items(), key=lambda kv: -kv[1][1]):
    print("%-40s %-28s calls=%7d yuan=%9.2f pt=%11d ct=%9d" % (k[0][:40], k[1], g[0], g[1], g[2], g[3])); tot += g[1]
print("TOTAL yuan = %.2f" % tot)
