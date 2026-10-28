"""R0 wrapper around nl2sql.run.main (no code patched). Reason: in run.py:pass_() an exception raised by write() (json.dumps of a row
containing a non-JSON type, e.g. numpy) leaves the `with ThreadPoolExecutor` block, whose __exit__ waits for ALL queued items while
discarding their results (observed: HR run silently burned ~240 items of API calls after 03:31). This wrapper (1) reports the offending
field and (2) coerces numpy/other scalars to python types so the row can be written.
usage: python tools_r0/run_safe.py <same args as python -m nl2sql.run>"""
import json, sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nl2sql import run as RUN

_orig = RUN.process_item
_seen = set()


def _py(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def _clean(x, qid, what):
    try:
        json.dumps(x, ensure_ascii=False)
        return x
    except TypeError as e:
        if (qid, what) not in _seen:
            _seen.add((qid, what))
            print("[run_safe] non-JSON value in %s of %s: %s" % (what, qid, e), flush=True)
        return json.loads(json.dumps(x, ensure_ascii=False, default=_py))


def safe(system, evaluator, item, seed, cfg, run_info):
    row, cands = _orig(system, evaluator, item, seed, cfg, run_info)
    return _clean(row, item.qid, "row"), [_clean(c, item.qid, "cand") for c in cands]


RUN.process_item = safe
if __name__ == '__main__':
    RUN.main(sys.argv[1:])
