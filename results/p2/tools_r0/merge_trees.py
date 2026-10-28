"""Merge tree shards into one trees.jsonl (dedupe by qid, skip error records); report coverage vs bird_train_HR."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nl2sql import data
root = sys.argv[1]
out = sys.argv[2]
want = [i.qid for i in data.load_split("bird_train_HR")]
seen = {}
for f in sorted(glob.glob(root + "/f*/trees.jsonl")):
    for l in open(f):
        try:
            t = json.loads(l)
        except ValueError:
            continue
        if "nodes" in t and t["qid"] not in seen:
            seen[t["qid"]] = l if l.endswith("\n") else l + "\n"
miss = [q for q in want if q not in seen]
with open(out, "w") as f:
    for q in want:
        if q in seen:
            f.write(seen[q])
print("trees", len(seen), "of", len(want), "missing", len(miss), miss[:5])
