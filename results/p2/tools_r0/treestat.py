import json, sys, collections
ts = [json.loads(l) for l in open(sys.argv[1])]
print("trees", len(ts), "nodes/q", sum(len(t["nodes"]) for t in ts) / len(ts), "calls/q", sum(n["calls"] for t in ts for n in t["nodes"]) / len(ts))
print("tier", collections.Counter(n["tier"] for t in ts for n in t["nodes"]))
print("root ex", sum(t["nodes"][0]["ex"] or 0 for t in ts) / len(ts))
