import json, sys, collections
c = collections.Counter(); tag = collections.defaultdict(lambda: [0, 0])
for l in open(sys.argv[1]):
    r = json.loads(l)
    c[(r.get("kind"), bool(r.get("cached")))] += 1
    tag[r.get("tag")][int(bool(r.get("cached")))] += 1
print(dict(c)); print({k: v for k, v in tag.items()})
