"""Attach executed-result hashes (out/exec_results.jsonl from 01_exec_results.py) to tree nodes, gold and HT systems."""
from common import *
from trees import *

_ER = None


def ER():
    global _ER
    if _ER is None:
        _ER = {}
        for l in open(OUT + "/exec_results.jsonl"):
            try:
                r = json.loads(l)
            except Exception:
                continue
            _ER[(r["db_id"], r["sql"])] = r
    return _ER


_TRAIN = None


def gold_sql(qid):
    global _TRAIN
    if _TRAIN is None:
        _TRAIN = json.load(open(TRAIN))
    _, db, idx = qid.split(":")
    return _TRAIN[int(idx)]["SQL"]


def key(db, sql):
    """result key: ('H', hash) for executed ok, ('E',) for error/timeout (all errors form one 'no answer' bucket, never agree with gold)"""
    r = ER().get((db, sql))
    if r is None:
        return ("?",)
    if not r["ok"]:
        return ("E",)
    return ("H", r["h"])


def annotate(Qs):
    miss = 0
    for q in Qs:
        g = ER().get((q.db, gold_sql(q.qid)))
        q.gold = g
        q.gkey = key(q.db, gold_sql(q.qid))
        for n in q.nodes:
            n["rk"] = key(q.db, n["sql"])
            if n["rk"] == ("?",):
                miss += 1
    return miss
