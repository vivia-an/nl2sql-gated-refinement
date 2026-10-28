"""rl.train.baselines for all 12 reward cells (train.py `base` only evaluates c_h=0.5, lam=0). Baselines are parameter-free (LR stopper is fit on all trees,
so it is optimistic on CV folds; flagged). python e6_base_cells.py <trees> <outdir>"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nl2sql.rl import train as T
from nl2sql.rl.env import TreeSet
ts = TreeSet.load(sys.argv[1])
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
res = {}
for (c, l) in T.CELLS:
    d = os.path.join(out, "tmp"); os.makedirs(d, exist_ok=True)
    r = T.baselines(ts, d, c_h=c, lam=l)
    res["ch%s_lam%s" % (c, l)] = r
json.dump(res, open(os.path.join(out, "baselines_all_cells.json"), "w"), indent=1)
print("ok", len(res))
