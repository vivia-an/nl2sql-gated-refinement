#!/usr/bin/env python3
"""Offline feasibility of VALUE GROUNDING (X7, part C): how many wrong answers of the tuning runs are wrong because a filter literal differs
from the literal the gold SQL uses, and where could the right literal have come from?  Analysis only: no retrieval is implemented, no LLM call,
BIRD-train tuning splits (HR / HT) only.  GOLD SQL IS USED FOR THIS ANALYSIS AND IS NEVER SHOWN TO A SYSTEM.

  value_check.py --run gated_HR=$R/runs/X5/1b-hr-gated --run gated_HT=$R/runs/X5/1b-ht-gated [--run p0_HR=... ] --pool gated=gated_HR,gated_HT
                 [--classes value_mismatch,row_subset,row_superset,empty_result] [--out results/x7/value_check.json]

Population: rows with ex == 0 whose auto_class (stored by the scorer) is in --classes (default value_mismatch, row_subset, row_superset,
empty_result).  For each wrong answer the literals of the WHERE / HAVING clauses of the gold SQL are compared with the literals of the predicted SQL
(any clause) and with the question + evidence text.  A gold literal is
  * PRESENT   in the prediction: equal to a predicted literal (strings after removing LIKE '%' wildcards; numbers by value), or a substring of one;
  * (ii) VERBATIM    in question / evidence: exact substring (numbers: token-bounded);
  * (iii) APPROXIMATE: not verbatim, but case-insensitive substring ("case"), equal after dropping punctuation / spaces / underscores ("format"),
                     all word stems occur among the text's word stems ("stem"), or a window of the text is >= 0.8 similar ("fuzzy");
                     numbers: equal numeric value to some number in the text ("number format");
  * (iv) ABSENT from question / evidence: needs a database lookup / world knowledge (codes, abbreviations, other wording).
Answer level, over the gold literals that are NOT present in the prediction ("missing"): none_missing (all gold literals present, or the gold
has no WHERE / HAVING literal) | verbatim_only | approx (>= 1 approximate, none absent) | absent (>= 1 absent).  `pure_value_error` is the tight
case: prediction and gold filter exactly the same columns (text or numeric, table.column resolved against the schema) with different literals.
Upper bound of value grounding (pp of ALL questions): wrong answers with a missing literal that is approximate or absent (the literal cannot be
copied from the text, so retrieval / lookup could matter); `retrievable_by_lookup` additionally requires the gold literal to exist in its column
of the database.  For absent literals the column cardinality (COUNT(DISTINCT), 20 s guard) shows how many a prompt could simply enumerate.
These are loose upper bounds: a wrong answer can have several causes, and fixing a literal need not fix the answer."""
import argparse
import difflib
import json
import os
import re
import sys
from collections import Counter, defaultdict

from x7_common import TUNING_SPLITS, load_items, load_preds, split_of  # noqa: F401  (sys.path set there)

CLASSES = ("value_mismatch", "row_subset", "row_superset", "empty_result")
_NUM = re.compile(r"^[+-]?\d+(\.\d+)?$")


# ------------------------------------------------------------------------------------------------ literals
def _lit_value(lit):
    v = lit.this
    return v if isinstance(v, str) else str(v)


def extract_literals(sql, clauses=("Where", "Having")):
    """-> list of (text, is_number) for the literals under the given clauses of `sql` (LIKE wildcards '%' removed from strings; duplicates kept
    once). Falls back to quoted strings / bare numbers by regex when sqlglot cannot parse the statement."""
    out = []
    try:
        import sqlglot
        from sqlglot import exp
        tree = sqlglot.parse_one(sql, read="sqlite")
        types = tuple(getattr(exp, c) for c in clauses)
        for lit in tree.find_all(exp.Literal, bfs=False):
            if lit.find_ancestor(*types) is None:
                continue
            t = _lit_value(lit)
            isnum = not lit.is_string
            out.append((t if isnum else t.replace("%", ""), isnum))
    except Exception:
        for m in re.finditer(r"'((?:[^']|'')*)'|(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])", sql or ""):
            if m.group(1) is not None:
                out.append((m.group(1).replace("''", "'").replace("%", ""), False))
            else:
                out.append((m.group(2), True))
    seen, res = set(), []
    for t, n in out:
        if (t, n) not in seen and t.strip() != "":
            seen.add((t, n))
            res.append((t, n))
    return res


def _norm(s):
    return re.sub(r"[^0-9a-z]", "", s.casefold())


def _words(s):
    return re.findall(r"[0-9A-Za-z]+", s.casefold())


def _stem(w):
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("es", ""), ("s", ""), ("ing", ""), ("ed", "")):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)] + rep
    return w


def _numbers(text):
    return [float(x.replace(",", "")) for x in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", text) if x.replace(",", "").replace(".", "", 1).isdigit()]


def classify_literal(lit, is_number, text):
    """-> "verbatim" | "case" | "format" | "stem" | "fuzzy" | "number" | "absent" (see module docstring)."""
    if is_number:
        if re.search(r"(?<![\d.])%s(?![\d])" % re.escape(lit), text):
            return "verbatim"
        try:
            v = float(lit)
        except ValueError:
            return "absent"
        return "number" if any(abs(v - x) < 1e-9 for x in _numbers(text)) else "absent"
    if lit in text:
        return "verbatim"
    if lit.casefold() in text.casefold():
        return "case"
    nl = _norm(lit)
    if nl and nl in _norm(text):
        return "format"
    lw = [_stem(w) for w in _words(lit)]
    if lw and set(lw) <= {_stem(w) for w in _words(text)}:
        return "stem"
    tw = _words(text)
    n = max(1, len(_words(lit)))
    lc = lit.casefold()
    for i in range(0, max(1, len(tw) - n + 1)):
        win = " ".join(tw[i:i + n])
        if len(lc) >= 4 and difflib.SequenceMatcher(None, lc, win).ratio() >= 0.8:
            return "fuzzy"
    return "absent"


def present_in(lit, is_number, pred_lits):
    """True when the gold literal occurs among the predicted literals: equal text, equal number, or (string) a substring of a predicted string."""
    for t, n in pred_lits:
        if lit == t:
            return True
        if is_number and n:
            try:
                if abs(float(lit) - float(t)) < 1e-9:
                    return True
            except ValueError:
                pass
        if not is_number and not n and lit in t:
            return True
    return False


# ------------------------------------------------------------------------------------------------ db checks
def column_info(item, table, column, timeout=20):
    """-> {"exists": gold literal exists in the column (filled by caller), "n_distinct": COUNT(DISTINCT) or None}"""
    from nl2sql import db
    r = db.execute(item.db_path, "SELECT COUNT(DISTINCT `%s`) FROM `%s`" % (column.replace("`", "``"), table.replace("`", "``")), timeout_s=timeout, max_rows=1)
    return int(r["rows"][0][0]) if r["ok"] and r["rows"] else None


def literal_exists(item, table, column, lit, is_number, timeout=20):
    from nl2sql import db
    q = "`%s`" % column.replace("`", "``")
    if is_number:
        sql = "SELECT 1 FROM `%s` WHERE %s = %s LIMIT 1" % (table.replace("`", "``"), q, lit)
    else:
        sql = "SELECT 1 FROM `%s` WHERE %s = '%s' LIMIT 1" % (table.replace("`", "``"), q, lit.replace("'", "''"))
    r = db.execute(item.db_path, sql, timeout_s=timeout, max_rows=1)
    return bool(r["ok"] and r["rows"])


# ------------------------------------------------------------------------------------------------ analysis
def analyse_row(item, pred_sql, schema, db_checks=True):
    """-> dict describing the literal situation of one wrong answer."""
    from nl2sql.systems.ours import probe as PR
    text = (item.question or "") + " \n " + (item.evidence or "")
    gold = extract_literals(item.gold_sql)
    pred_all = extract_literals(pred_sql, clauses=("Where", "Having", "Join", "Select"))
    gcond = PR.literal_columns(item.gold_sql, schema, max_literals=50, text_only=False)
    pcond = PR.literal_columns(pred_sql, schema, max_literals=50, text_only=False)
    gcols = {(c["table"], c["column"]): c for c in gcond}
    pcols = {(c["table"], c["column"]): c for c in pcond}
    lits = []
    for t, isn in gold:
        pres = present_in(t, isn, pred_all)
        cat = classify_literal(t, isn, text)
        col = next((k for k, c in gcols.items() if t in [x.replace("%", "") if not isn else x for x in c["literals"]]), None)
        lits.append({"lit": t, "number": isn, "present": pres, "text_class": cat, "gold_col": "%s.%s" % col if col else None, "_col": col})
    missing = [l for l in lits if not l["present"]]
    if not lits:
        level = "no_gold_literal"
    elif not missing:
        level = "none_missing"
    else:
        cats = {l["text_class"] for l in missing}
        level = "absent" if "absent" in cats else ("approx" if cats - {"verbatim"} else "verbatim_only")
    pure = bool(gcols) and set(gcols) == set(pcols) and any(set(gcols[k]["literals"]) != set(pcols[k]["literals"]) for k in gcols)
    # database checks for the literals that cannot be copied from the text
    for l in missing:
        l["exists_in_gold_col"] = l["n_distinct"] = None
        if db_checks and l["_col"] and l["text_class"] in ("absent", "case", "format", "stem", "fuzzy", "number"):
            t, c = l["_col"]
            l["exists_in_gold_col"] = literal_exists(item, t, c, l["lit"], l["number"])
            l["n_distinct"] = column_info(item, t, c) if l["text_class"] == "absent" else None
    for l in lits:
        l.pop("_col", None)
    return {"level": level, "pure_value_error": pure, "gold_literals": lits,
            "pred_literals": [t for t, _ in pred_all][:12], "same_columns": sorted("%s.%s" % k for k in gcols if k in pcols),
            "gold_columns": sorted("%s.%s" % k for k in gcols), "pred_columns": sorted("%s.%s" % k for k in pcols)}


def summarise(rows, n_total):
    """rows: analyse_row dicts of the wrong answers; n_total: all questions of the run set (denominator for pp)."""
    c = Counter(r["level"] for r in rows)
    n = max(len(rows), 1)
    sub = Counter()
    for r in rows:
        for l in r["gold_literals"]:
            if not l["present"]:
                sub[l["text_class"]] += 1
    nonverb = [r for r in rows if r["level"] in ("approx", "absent")]
    lookup = [r for r in nonverb if any((not l["present"]) and l["text_class"] != "verbatim" and l.get("exists_in_gold_col") for l in r["gold_literals"])]
    absent_lits = [l for r in rows for l in r["gold_literals"] if not l["present"] and l["text_class"] == "absent"]
    card = [l["n_distinct"] for l in absent_lits if l.get("n_distinct") is not None]
    out = {"n_wrong_in_classes": len(rows), "n_questions": n_total,
           "levels": dict(c), "levels_frac_of_wrong": {k: v / float(n) for k, v in c.items()},
           "missing_literal_text_classes": dict(sub),
           "pure_value_error": sum(1 for r in rows if r["pure_value_error"]),
           "upper_bound_value_grounding": {"n": len(nonverb), "pp_of_all_questions": 100.0 * len(nonverb) / max(n_total, 1),
                                           "retrievable_by_lookup_n": len(lookup), "retrievable_by_lookup_pp": 100.0 * len(lookup) / max(n_total, 1),
                                           "pure_value_error_and_nonverbatim_n": sum(1 for r in nonverb if r["pure_value_error"]),
                                           "verbatim_only_n": c.get("verbatim_only", 0)},
           "absent_literals": {"n": len(absent_lits), "with_column_cardinality": len(card),
                               "card_le_8": sum(1 for x in card if x <= 8), "card_le_20": sum(1 for x in card if x <= 20),
                               "card_le_50": sum(1 for x in card if x <= 50), "card_gt_50": sum(1 for x in card if x > 50)}}
    return out


def run_label(label, path, classes, db_checks, examples=4):
    preds = load_preds(path)
    split = split_of(preds)
    items = load_items(split)
    from nl2sql import data
    wrong = [r for r in preds.values() if r.get("ex") == 0 and r.get("auto_class") in classes]
    rows, ex = [], defaultdict(list)
    schemas = {}
    for r in sorted(wrong, key=lambda r: r["qid"]):
        it = items[r["qid"]]
        if it.db_path not in schemas:
            schemas[it.db_path] = data.get_schema(it)
        a = analyse_row(it, r.get("final_sql") or "", schemas[it.db_path], db_checks)
        a.update({"qid": r["qid"], "db_id": r["db_id"], "auto_class": r["auto_class"]})
        rows.append(a)
        if len(ex[a["level"]]) < examples:
            ex[a["level"]].append({"qid": r["qid"], "question": it.question, "evidence": it.evidence, "gold_literals": [(l["lit"], l["text_class"], l["present"]) for l in a["gold_literals"]],
                                   "pred_literals": a["pred_literals"], "gold_columns": a["gold_columns"], "pred_columns": a["pred_columns"]})
    n_all = len(preds)
    s = summarise(rows, n_all)
    s.update({"split": split, "run": os.path.abspath(path), "n_rows_total": n_all, "n_wrong_all": sum(1 for r in preds.values() if r.get("ex") == 0),
              "classes": list(classes), "examples": dict(ex)})
    return s, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", required=True, help="LABEL=RUN_DIR (preds.jsonl of an HR / HT run)")
    ap.add_argument("--pool", action="append", default=[], help="NAME=LABEL1,LABEL2: pooled summary over several labels (e.g. HR + HT of one system)")
    ap.add_argument("--classes", default=",".join(CLASSES))
    ap.add_argument("--no-db", action="store_true", help="skip the database lookups (existence / cardinality)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    classes = tuple(a.classes.split(","))
    res, allrows, ntot = {}, {}, {}
    for spec in a.run:
        label, path = spec.split("=", 1)
        res[label], allrows[label] = run_label(label, path, classes, not a.no_db)
        ntot[label] = res[label]["n_rows_total"]
    for spec in a.pool:
        name, labels = spec.split("=", 1)
        ls = labels.split(",")
        rows = [r for l in ls for r in allrows[l]]
        res[name] = summarise(rows, sum(ntot[l] for l in ls))
        res[name]["pooled_from"] = ls
        res[name]["n_wrong_all"] = sum(res[l]["n_wrong_all"] for l in ls)
    for k, v in res.items():
        print("%-12s wrong-in-classes %4d / %s all wrong; levels %s; UB value grounding %.2f pp (lookup-confirmed %.2f pp); pure value error %d; absent-literal column cardinality %s"
              % (k, v["n_wrong_in_classes"], v.get("n_wrong_all", "?"), v["levels"], v["upper_bound_value_grounding"]["pp_of_all_questions"],
                 v["upper_bound_value_grounding"]["retrievable_by_lookup_pp"], v["pure_value_error"], v["absent_literals"]))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False, default=str)
    return res


if __name__ == "__main__":
    main()
