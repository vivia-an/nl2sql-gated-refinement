#!/usr/bin/env python3
"""Headroom of value probing / empty-result gating (X7, part A) from stored runs; NO LLM call; BIRD-train tuning splits only (worker machine venv).

  empty_roots.py --run P0_HR=$R/runs/X5/1b-hr-quote [--run G_HR=$R/runs/X5/1b-hr-gated ...] [--probe-n 8] [--out results/x7/empty_roots.json]

For every run (preds.jsonl + cands.jsonl; cand_idx 0 = the root) it looks at the roots that EXECUTE but are empty / one all-NULL row (status
ok_empty, the population of `m5.gate_empty` + `m5.probe_values`):
  * how many, how many the gold SQL answers non-empty (those are wrong; roots whose gold is empty too can be right), root EX, final EX of the run,
    revised or not, repaired (root wrong -> final right) / harmed (root right -> final wrong);
  * for the wrong ones with a text literal in WHERE / JOIN / HAVING (`probe.literal_columns`): is the gold SQL filtering the same column with another
    literal ("same column, other value": the case probing can fix), another column, or no text literal at all; does the gold literal differ only in case /
    punctuation ("format only"); and the ACTUAL probe (`probe.probe_failed_sql`, n values, the code the live system runs) - does the list contain the gold
    literal, is the list complete (column has <= n distinct short values), does the gold literal exist in the column at all;
  * `probe_visible` = wrong empty roots (gold non-empty) for which the probe shows the gold literal = an upper bound of what the probe block can repair;
  * for the other (non-empty) roots: one-row/one-column results equal to 0 (COUNT = 0 / SUM = 0) whose gold is not 0 - the degenerate-result extension.
Gold SQL is used for this analysis only (never shown to a system)."""
import argparse
import json
import os
from collections import Counter

from x7_common import load_items, load_preds, load_root_cands, split_of  # noqa: F401  (sys.path set there)


def _norm(s):
    import re
    return re.sub(r"[^0-9a-z]", "", str(s).casefold())


def gold_rows(ev, item):
    g = ev.gold_exec(item)
    if not g or not g.get("ok"):
        return None
    if g.get("big"):
        return [("big",)] * int(g.get("n_rows", 1))
    return [tuple(r) for r in g["rows"]]


def is_empty(rows):
    return (not rows) or (len(rows) == 1 and all(c is None for c in rows[0]))


def analyse_run(label, path, probe_n, max_len=40, near=False):
    from nl2sql import data, db
    from nl2sql.eval.score import Evaluator
    from nl2sql.systems.ours import probe as PR
    preds, roots = load_preds(path), load_root_cands(path)
    split = split_of(preds)
    items = load_items(split)
    ev = Evaluator(split, run_id="x7-empty-roots")
    schemas = {}
    emp, zero = [], []
    for q, c in sorted(roots.items()):
        r = preds.get(q)
        if r is None:
            continue
        if c.get("status") == "ok_empty":
            emp.append((q, c, r))
        elif c.get("status") == "ok_nonempty" and c.get("n_rows") == 1 and c.get("n_cols") == 1:
            zero.append((q, c, r))
    out = {"label": label, "split": split, "run": os.path.abspath(path), "n_questions": len(preds), "n_roots": len(roots), "n_empty_roots": len(emp)}
    rows = []
    cause = Counter()
    for q, c, r in emp:
        it = items[q]
        if it.db_path not in schemas:
            schemas[it.db_path] = data.get_schema(it)
        sch = schemas[it.db_path]
        g = gold_rows(ev, it)
        gold_empty = None if g is None else is_empty(g)
        rec = {"qid": q, "db_id": it.db_id, "root_ex": c.get("ex"), "final_ex": r.get("ex"), "revised": r.get("iterations", 0) > 0, "gold_ok": g is not None,
               "gold_empty": gold_empty}
        rec["repaired"] = bool(rec["root_ex"] == 0 and rec["final_ex"] == 1)
        rec["harmed"] = bool(rec["root_ex"] == 1 and rec["final_ex"] == 0)
        if rec["root_ex"] == 0 and gold_empty is False:
            rc = PR.literal_columns(c["sql"], sch, max_literals=10)
            gc = PR.literal_columns(it.gold_sql, sch, max_literals=10)
            gmap = {(x["table"], x["column"]): x["literals"] for x in gc}
            rmap = {(x["table"], x["column"]): x["literals"] for x in rc}
            probes = PR.probe_failed_sql(it.db_path, c["sql"], sch, n=probe_n, max_len=max_len, max_cols=4, timeout_s=10.0, near=near)
            if not rc:
                rec["cause"] = "no_text_literal_in_root"
            else:
                same = [k for k in rmap if k in gmap]
                diff = [k for k in same if set(l.replace("%", "") for l in rmap[k]) != set(l.replace("%", "") for l in gmap[k])]
                if diff:
                    rec["cause"] = "same_column_other_value"
                    k = diff[0]
                    rl, gl = [l.replace("%", "") for l in rmap[k]], [l.replace("%", "") for l in gmap[k]]
                    rec["format_only"] = bool({_norm(x) for x in rl} & {_norm(x) for x in gl})
                    rec["wrong_column"] = "%s.%s" % k
                    rec["root_literals"], rec["gold_literals"] = rl, gl
                    pv = next((p for p in probes if (p["table"], p["column"]) == k), None)
                    rec["probe_shown"] = pv is not None
                    if pv:
                        rec["probe_complete"] = pv["complete"]
                        rec["probe_visible"] = any(x in pv["values"] for x in gl)
                    ex_sql = "SELECT 1 FROM `%s` WHERE `%s` = '%s' LIMIT 1" % (k[0], k[1], gl[0].replace("'", "''"))
                    e = db.execute(it.db_path, ex_sql, timeout_s=20, max_rows=1)
                    rec["gold_literal_exists_in_root_column"] = bool(e["ok"] and e["rows"])
                elif same:
                    rec["cause"] = "same_literals_other_reason"
                else:
                    rec["cause"] = "gold_filters_other_columns" if gmap else "gold_has_no_text_literal"
                    rec["root_cols"], rec["gold_cols"] = sorted("%s.%s" % k for k in rmap), sorted("%s.%s" % k for k in gmap)
                    rec["probe_visible"] = False
                    # root's columns are wrong: could the probe show the gold literal in a root column anyway?
                    rec["probe_shown"] = bool(probes)
            cause[rec["cause"]] += 1
        rows.append(rec)
    wrong = [x for x in rows if x["root_ex"] == 0 and x["gold_empty"] is False]
    out.update({
        "gold_nonempty": sum(1 for x in rows if x["gold_empty"] is False), "gold_empty": sum(1 for x in rows if x["gold_empty"] is True),
        "gold_failed": sum(1 for x in rows if x["gold_empty"] is None),
        "root_right": sum(1 for x in rows if x["root_ex"] == 1), "root_wrong": sum(1 for x in rows if x["root_ex"] == 0),
        "wrong_with_gold_nonempty": len(wrong),
        "revised_in_run": sum(1 for x in rows if x["revised"]),
        "repaired_in_run": sum(1 for x in rows if x["repaired"]), "harmed_in_run": sum(1 for x in rows if x["harmed"]),
        "repair_rate_of_wrong_in_run": (sum(1 for x in wrong if x["repaired"]) / float(len(wrong))) if wrong else None,
        "repair_rate_of_revised_wrong_in_run": (lambda rv: (sum(1 for x in rv if x["repaired"]) / float(len(rv))) if rv else None)([x for x in wrong if x["revised"]]),
        "cause": dict(cause),
        "same_column_other_value_format_only": sum(1 for x in wrong if x.get("cause") == "same_column_other_value" and x.get("format_only")),
        "probe_shown_for_wrong": sum(1 for x in wrong if x.get("probe_shown")),
        "probe_visible": sum(1 for x in wrong if x.get("probe_visible")),
        "probe_visible_complete_list": sum(1 for x in wrong if x.get("probe_visible") and x.get("probe_complete")),
        "gold_literal_exists_in_root_column": sum(1 for x in wrong if x.get("gold_literal_exists_in_root_column")),
        "probe_visible_pp_of_all_questions": 100.0 * sum(1 for x in wrong if x.get("probe_visible")) / max(len(preds), 1),
        "empty_wrong_pp_of_all_questions": 100.0 * len(wrong) / max(len(preds), 1),
        "probe_n": probe_n, "probe_near": near,
    })
    # degenerate "0" aggregates among non-empty roots (COUNT = 0, SUM = 0) whose gold is not 0
    z = []
    for q, c, r in zero:
        it = items[q]
        res = db.execute(it.db_path, c["sql"], timeout_s=30, max_rows=2)
        if res["ok"] and len(res["rows"]) == 1 and res["rows"][0][0] in (0, 0.0):
            g = gold_rows(ev, it)
            z.append({"qid": q, "root_ex": c.get("ex"), "final_ex": r.get("ex"), "gold_zero": None if g is None else (len(g) == 1 and g[0][0] in (0, 0.0))})
    out["zero_aggregate_roots"] = {"n": len(z), "root_wrong": sum(1 for x in z if x["root_ex"] == 0), "wrong_with_nonzero_gold": sum(1 for x in z if x["root_ex"] == 0 and x["gold_zero"] is False)}
    out["rows"] = rows
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", required=True, help="LABEL=RUN_DIR")
    ap.add_argument("--probe-n", type=int, default=8)
    ap.add_argument("--probe-near", action="store_true", help="evaluate the m5.probe_near variant (values resembling the failed literal first)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-rows", action="store_true", help="omit the per-question rows from the JSON")
    a = ap.parse_args(argv)
    res = {}
    for spec in a.run:
        label, path = spec.split("=", 1)
        res[label] = analyse_run(label, path, a.probe_n, near=a.probe_near)
        r = res[label]
        print("%-8s n=%d empty roots=%d (gold non-empty %d, gold empty %d, root right %d) wrong+gold-nonempty=%d | in-run: revised %d repaired %d harmed %d | cause %s | probe visible %d (complete list %d) | zero-aggregate roots %s"
              % (label, r["n_questions"], r["n_empty_roots"], r["gold_nonempty"], r["gold_empty"], r["root_right"], r["wrong_with_gold_nonempty"], r["revised_in_run"],
                 r["repaired_in_run"], r["harmed_in_run"], r["cause"], r["probe_visible"], r["probe_visible_complete_list"], r["zero_aggregate_roots"]), flush=True)
        if a.no_rows:
            r.pop("rows")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False, default=str)
    return res


if __name__ == "__main__":
    main()
