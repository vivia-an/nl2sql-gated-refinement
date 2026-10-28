"""R0 wrapper around nl2sql.rl.env_builder.main (no code patched): (1) any exception in build_tree becomes an {"error"} record instead of ending
the main result loop (whose executor block would then silently drain the queue), (2) non-JSON values are coerced and reported."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nl2sql.rl import env_builder as EB
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_safe import _py

_orig = EB.build_tree
_seen = set()


def safe(sys_, ev, item, seed=0):
    try:
        t = _orig(sys_, ev, item, seed)
    except Exception as e:      # env_builder.one() only catches LLMError; any other exception would end its result loop and silently burn the queue
        import traceback
        print("[builder_safe] tree failed %s: %s %s" % (item.qid, type(e).__name__, str(e)[:200]), flush=True)
        return {"qid": item.qid, "error": "%s: %s" % (type(e).__name__, str(e)[:150])}
    try:
        json.dumps(t)
        return t
    except TypeError as e:
        if item.qid not in _seen:
            _seen.add(item.qid)
            bad = []
            for n in t["nodes"]:
                for k, v in n.items():
                    try:
                        json.dumps(v)
                    except TypeError:
                        bad.append((n["id"], k, type(v).__name__))
            print("[builder_safe] non-JSON in %s: %s | %s" % (item.qid, e, bad[:6]), flush=True)
        return json.loads(json.dumps(t, default=_py))


EB.build_tree = safe

# optional sharding (env R0_SHARD="k/n", R0_EXTRA_DONE=<trees.jsonl whose qids are skipped>): lets several processes split one fold
_shard = os.environ.get("R0_SHARD")
if _shard:
    from nl2sql import data as _data
    _k, _n = map(int, _shard.split("/"))
    _skip = set()
    for _p in filter(None, os.environ.get("R0_EXTRA_DONE", "").split(",")):
        for _l in open(_p):
            try:
                _skip.add(json.loads(_l)["qid"])
            except Exception:
                pass
    _ols = _data.load_split

    def _ls(name, with_gold=None):
        its = _ols(name, with_gold)
        its = [i for i in its if i.qid not in _skip]
        return [i for j, i in enumerate(its) if j % _n == _k]
    _data.load_split = _ls
sys.argv = [sys.argv[0]] + sys.argv[1:]
EB.main()
