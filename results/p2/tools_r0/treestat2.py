import json, sys, collections
ts = [json.loads(l) for l in open(sys.argv[1])]
n = len(ts)
act = collections.Counter(x["action"] for t in ts for x in t["nodes"])
root_ex = sum(t["nodes"][0]["ex"] or 0 for t in ts) / n
any_ex = sum(1 for t in ts if any(x["ex"] for x in t["nodes"])) / n
d1 = sum(1 for t in ts if any(x["ex"] for x in t["nodes"] if x["parent"] == 0)) / n
res = {"n_trees": n, "nodes_per_q": sum(len(t["nodes"]) for t in ts) / n, "calls_per_q_all_nodes": sum(x["calls"] for t in ts for x in t["nodes"]) / n,
       "tier_nodes": dict(collections.Counter(x["tier"] for t in ts for x in t["nodes"])), "actions": dict(act), "root_ex": round(root_ex, 4),
       "any_node_ex1": round(any_ex, 4), "any_depth1_ex1_or_root": round(sum(1 for t in ts if t["nodes"][0]["ex"] or any(x["ex"] for x in t["nodes"] if x["parent"] == 0)) / n, 4),
       "per_fold": dict(collections.Counter(t["fold"] for t in ts)), "per_db": dict(collections.Counter(t["db_id"] for t in ts))}
print(json.dumps(res, indent=1))
json.dump(res, open(sys.argv[2], "w"), indent=1)
