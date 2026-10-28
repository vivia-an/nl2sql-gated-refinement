"""Objective auto error classes (EXPERIMENT_SPEC E10 decision list), applied to a failed prediction.
Inputs are execution results from db.execute (dicts with ok/rows/error)."""
from ..db import error_kind


def classify(pred, gold):
    """pred, gold: db.execute dicts. Returns a class string for EX=0 cases."""
    if not pred["ok"]:
        k = error_kind(pred["error"])
        if k == "timeout":
            return "timeout"
        return {"syntax": "exec_error:syntax", "identifier": "exec_error:unknown_identifier"}.get(k, "exec_error:other")
    if "ambiguous column" in (pred.get("error") or ""):
        return "exec_error:ambiguous_column"
    pr, gr = [tuple(r) for r in pred["rows"]], [tuple(r) for r in gold["rows"]]
    if not pr and gr:
        return "empty_result"
    pw = len(pr[0]) if pr else 0
    gw = len(gr[0]) if gr else 0
    if pw != gw:
        return "shape_mismatch"
    # column permutation: equal after reordering columns (try sorting columns by their multiset of values)
    try:
        pcols = sorted(tuple(sorted(map(repr, col))) for col in zip(*pr))
        gcols = sorted(tuple(sorted(map(repr, col))) for col in zip(*gr))
        if pcols == gcols and set(map(lambda r: tuple(sorted(map(repr, r))), pr)) == set(map(lambda r: tuple(sorted(map(repr, r))), gr)):
            return "column_permutation"
    except Exception:
        pass
    ps, gs = set(pr), set(gr)
    if ps > gs:
        return "row_superset"
    if ps < gs:
        return "row_subset"
    return "value_mismatch"
