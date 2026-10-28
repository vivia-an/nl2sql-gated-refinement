import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("e", os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_exec_results.py")); e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
from common import *
S = e.collect(); done = set()
for l in open(e.OUTF):
    try: r = json.loads(l); done.add((r["db_id"], r["sql"]))
    except Exception: pass
tot = collections.Counter(d for d, _ in S); rem = collections.Counter(d for d, s in S if (d, s) not in done)
print({d: (tot[d], rem[d]) for d in tot})
tm = collections.Counter(); n = collections.Counter()
for l in open(e.OUTF):
    r = json.loads(l); tm[r["db_id"]] += r["t"]; n[r["db_id"]] += 1
print({d: round(tm[d] / n[d], 2) for d in tm})
