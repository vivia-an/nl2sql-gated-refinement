import json, sys
ts = [json.loads(l) for l in open(sys.argv[1])]
for t in ts:
    n = t["nodes"]
    print(t["qid"], "root calls", n[0]["calls"], "prompt", n[0]["prompt_tokens"], "score", n[0]["score"], "ex", n[0]["ex"], [(x["action"], x["tier"], x["calls"], x["ex"]) for x in n[1:6]])
