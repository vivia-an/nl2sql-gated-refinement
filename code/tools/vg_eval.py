#!/usr/bin/env python3
"""Offline evaluation of VALUE GROUNDING (X9; `m2.value_grounding`, nl2sql/systems/ours/values.py) on the TUNING data only - no LLM call.

  vg_eval.py --run HR=$R/runs/X5/1b-hr-gated --run HT=$R/runs/X5/1b-ht-gated
             [--variant NAME=JSON ...] [--rows-cache DIR] [--no-db] [--out results/x9/vg_eval.json]

Inputs: the stored X5 gated runs (preds.jsonl: final SQL, ex, auto_class, ctx_tables = the tables the model saw) and the BIRD-train gold SQL of the
tuning splits (bird_train_HR / bird_train_HT; X7's tools/value_check.py lineage).  GOLD SQL IS USED FOR THIS ANALYSIS ONLY, never as input of the hint
generator (`ValueIndex.hints` gets question + evidence + shown tables).  For every variant (parameter overrides of `values.VG_DEFAULTS`; `base` = defaults):
  * recall: the 'lookup-reachable' answers of X7 (wrong answers in the value / row classes with a missing gold literal that is NOT verbatim in
    question / evidence but stored in its column = 49 pooled): is the gold literal among the hints?  literal level (any hint with that exact string),
    column level (same string AND the gold column), answer level 'any' / 'all' target literals.  Diagnostics: `in_index` (the literal made it into the
    value index; the rest is lost to the build limits) and recall by text class (case / format / stem / fuzzy / number / absent).  Also the 27
    verbatim-only answers (mis-copies of text the model already sees) and all 80 non-verbatim answers.
  * precision / noise over ALL questions of the runs: share with >= 1 hint, hints and prompt tokens per question, hint kinds, `gold_hit` (the hint
    string is a WHERE / HAVING literal of the gold SQL), `distractor` (not), and the harm indicators on currently CORRECT answers: questions with
    hints, distractors per question, `conflict` (the correct SQL filters a hinted column with another literal than the hint).
  * cost: questions with >= 1 hint = the generation prompts that change (new chat calls on a cached rerun), hint time per question.
The index limits actually hit (status per column, values, build seconds) are reported per database."""
import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict

from x7_common import load_items, load_preds, split_of  # noqa: F401  (sys.path set there)

import value_check as VC  # noqa: E402

CLASSES = VC.CLASSES


def _labels_rows(label, path, items_cache, db_checks, cache_dir):
    """-> (preds, items, wrong-answer analysis rows of value_check) with an optional JSON cache of the (slow) database checks."""
    preds = load_preds(path)
    split = split_of(preds)
    items = load_items(split)
    cf = os.path.join(cache_dir, "vc_rows_%s.json" % label) if cache_dir else None
    if cf and os.path.exists(cf):
        rows = json.load(open(cf))["rows"]
        return preds, items, rows
    from nl2sql import data
    wrong = [r for r in preds.values() if r.get("ex") == 0 and r.get("auto_class") in CLASSES]
    rows, schemas = [], {}
    for r in sorted(wrong, key=lambda r: r["qid"]):
        it = items[r["qid"]]
        if it.db_path not in schemas:
            schemas[it.db_path] = data.get_schema(it)
        a = VC.analyse_row(it, r.get("final_sql") or "", schemas[it.db_path], db_checks)
        a.update({"qid": r["qid"], "db_id": r["db_id"], "auto_class": r["auto_class"]})
        rows.append(a)
    if cf:
        os.makedirs(cache_dir, exist_ok=True)
        json.dump({"rows": rows}, open(cf, "w"), default=str)
    return preds, items, rows


def targets_of(row):
    """The gold literals a value hint could supply: missing in the prediction, not verbatim in question / evidence, stored in the gold column."""
    return [l for l in row["gold_literals"] if not l["present"] and l["text_class"] != "verbatim" and l.get("exists_in_gold_col") and l.get("gold_col")]


def population(rows):
    nonverb = [r for r in rows if r["level"] in ("approx", "absent")]
    reach = [r for r in nonverb if targets_of(r)]
    verb = [r for r in rows if r["level"] == "verbatim_only"]
    return {"reachable": reach, "nonverbatim": nonverb, "verbatim_only": verb}


def hint_set(hs):
    return {(h["table"] + "." + h["column"], h["literal"]) for h in hs}


def answer_recall(rows, hints_of, index_vals):
    """Literal / column / answer level recall over the answers `rows`, targets = missing non-verbatim literals stored in their column."""
    lit_n = lit_hit = col_hit = in_idx = 0
    any_lit = all_lit = any_col = 0
    by_class = defaultdict(lambda: [0, 0])
    detail = []
    for r in rows:
        tg = [l for l in r["gold_literals"] if not l["present"] and l["text_class"] != "verbatim" and l.get("exists_in_gold_col") and l.get("gold_col")]
        if not tg:
            continue
        hs = hints_of[r["qid"]]
        strs = {h["literal"] for h in hs}
        hc = hint_set(hs)
        a_lit, a_col = [], []
        for l in tg:
            lit_n += 1
            ok_l = l["lit"] in strs
            ok_c = (l["gold_col"], l["lit"]) in hc
            ii = l["lit"] in index_vals.get((r["db_id"], l["gold_col"]), ())
            lit_hit += ok_l
            col_hit += ok_c
            in_idx += ii
            a_lit.append(ok_l)
            a_col.append(ok_c)
            by_class[l["text_class"]][0] += 1
            by_class[l["text_class"]][1] += ok_c
        any_lit += any(a_lit)
        all_lit += all(a_lit)
        any_col += any(a_col)
        detail.append({"qid": r["qid"], "targets": [(l["lit"], l["gold_col"], l["text_class"]) for l in tg], "hit_col": a_col, "n_hints": len(hs)})
    n_ans = len(detail)
    return {"answers": n_ans, "literals": lit_n, "lit_recall": lit_hit / float(max(lit_n, 1)), "col_recall": col_hit / float(max(lit_n, 1)),
            "in_index": in_idx / float(max(lit_n, 1)), "answer_any_lit": any_lit, "answer_all_lit": all_lit, "answer_any_col": any_col,
            "answer_any_col_frac": any_col / float(max(n_ans, 1)),
            "by_text_class": {k: {"n": v[0], "col_recall": v[1] / float(max(v[0], 1))} for k, v in sorted(by_class.items())}}, detail


def verbatim_recall(rows, hints_of):
    """The mis-copy answers (every missing gold literal is verbatim in question / evidence): is the stored spelling of the gold column among the hints?
    (A hint exists only when the literal is in the column, so this also tells how often the verbatim text is a stored value.)"""
    n = hit = ans_any = 0
    for r in rows:
        tg = [l for l in r["gold_literals"] if not l["present"] and l["text_class"] == "verbatim" and l.get("gold_col")]
        if not tg:
            continue
        hc = hint_set(hints_of[r["qid"]])
        h = [(l["gold_col"], l["lit"]) in hc for l in tg]
        n += len(tg)
        hit += sum(h)
        ans_any += any(h)
    return {"answers": len(rows), "literals": n, "col_hit": hit, "answers_any": ans_any}


def noise(preds, items, hints_of, blocks, schemas):
    from nl2sql.systems.ours import probe as PR
    nq = len(hints_of)
    ns = [len(h) for h in hints_of.values()]
    with_h = [n for n in ns if n]
    toks = [blocks[q]["tokens"] for q in hints_of]
    kinds = Counter(h["kind"] for hs in hints_of.values() for h in hs)
    ident = sum(1 for hs in hints_of.values() for h in hs if h["identical"])
    tot_h = sum(ns)
    gold_hit = distractor = 0
    correct_q = correct_with = correct_distr = correct_conf = 0
    wrong_with = 0
    new_useful = 0           # gold-literal hints that the gated final SQL did NOT contain
    for q, hs in hints_of.items():
        it, pr = items[q], preds[q]
        gl = {t for t, _n in VC.extract_literals(it.gold_sql)}
        fs = pr.get("final_sql") or ""
        distr_here = 0
        for h in hs:
            if h["literal"] in gl:
                gold_hit += 1
                if h["literal"] not in fs:
                    new_useful += 1
            else:
                distractor += 1
                distr_here += 1
        if pr.get("ex") == 1:
            correct_q += 1
            if hs:
                correct_with += 1
                correct_distr += distr_here
                try:
                    used = {(c["table"], c["column"]): set(c["literals"]) for c in PR.literal_columns(fs, schemas[it.db_path], max_literals=50)}
                except Exception:
                    used = {}
                if any((h["table"], h["column"]) in used and h["literal"] not in used[(h["table"], h["column"])] for h in hs):
                    correct_conf += 1
        elif hs:
            wrong_with += 1
    return {"questions": nq, "with_hint": len(with_h), "with_hint_frac": len(with_h) / float(max(nq, 1)),
            "hints_total": tot_h, "hints_per_q": tot_h / float(max(nq, 1)), "hints_per_q_with_hint": statistics.mean(with_h) if with_h else 0.0,
            "hints_p90": sorted(ns)[int(0.9 * (nq - 1))] if nq else 0,
            "tokens_per_q": statistics.mean(toks) if toks else 0.0, "tokens_per_q_with_hint": statistics.mean([t for t in toks if t]) if any(toks) else 0.0,
            "tokens_p90": sorted(toks)[int(0.9 * (nq - 1))] if nq else 0, "tokens_total": sum(toks),
            "kinds": dict(kinds), "identical_frac": ident / float(max(tot_h, 1)),
            "gold_hit": gold_hit, "distractor": distractor, "gold_hit_precision": gold_hit / float(max(tot_h, 1)),
            "new_useful_hints": new_useful,
            "correct_answers": correct_q, "correct_with_hint": correct_with, "correct_distractors_per_q_with_hint": correct_distr / float(max(correct_with, 1)),
            "correct_with_conflicting_hint": correct_conf, "correct_conflict_frac": correct_conf / float(max(correct_q, 1)),
            "wrong_with_hint": wrong_with}


def run(args):
    from nl2sql import data
    from nl2sql.systems.ours import values as VG
    from nl2sql.util import est_tokens
    base = VG.params()
    if args.build:
        base.update(json.loads(args.build))
    variants = {"base": {}}
    for v in args.variant:
        name, js = v.split("=", 1)
        variants[name] = json.loads(js)
    for name, ov in variants.items():
        for k in ("max_len", "max_distinct", "max_total", "scan_rows", "stmt_timeout_s", "budget_s"):
            if k in ov and ov[k] != base[k]:
                raise SystemExit("variant %s changes build parameter %s: build parameters are shared by all variants (pass them as base via --build JSON)" % (name, k))
    out = {"variants": {}, "index": {}, "params_base": base}
    runs = {}
    schemas, idx_of, index_vals = {}, {}, {}
    for spec in args.run:
        label, path = spec.split("=", 1)
        preds, items, rows = _labels_rows(label, path, None, not args.no_db, args.rows_cache)
        runs[label] = (preds, items, rows)
        for q, it in items.items():
            if q in preds and it.db_path not in schemas:
                schemas[it.db_path] = data.get_schema(it)
    # build / load the indexes (this is where the limits of the build matter); timings are of THIS call (a cache hit is fast)
    for label, (preds, items, rows) in runs.items():
        for q in preds:
            it = items[q]
            if it.db_path in idx_of:
                continue
            t0 = time.time()
            idx = VG.get_index(it.db_path, schemas[it.db_path], base)
            idx_of[it.db_path] = idx
            m = idx.meta
            trunc = sorted("%s.%s" % (c["table"], c["column"]) for c in idx.cols if c["status"] == "truncated")
            bad = sorted("%s.%s:%s" % (c["table"], c["column"], c["status"]) for c in idx.cols if c["status"] in ("timeout", "error", "budget", "total_cap"))
            out["index"][it.db_id] = {"db_bytes": os.path.getsize(it.db_path), "values": m.get("n_values"), "text_columns": m.get("n_cols"), "status": m.get("status"),
                                      "build_s": m.get("elapsed_s"), "load_s": round(time.time() - t0, 2), "truncated_columns": trunc[:40], "n_truncated": len(trunc),
                                      "skipped_columns": bad[:40], "n_skipped": len(bad)}
            for c in idx.cols:
                index_vals[(it.db_id, c["table"] + "." + c["column"])] = set(c["values"])
    for name, ov in variants.items():
        p = dict(base)
        p.update(ov)
        VG.params({"m2": {"vg": {k: p[k] for k in VG.VG_DEFAULTS if p[k] != VG.VG_DEFAULTS[k]}}})      # validates
        res = {"params": ov, "splits": {}}
        pooled_h, pooled_blocks, pooled_p, pooled_i, pooled_rows = {}, {}, {}, {}, []
        for label, (preds, items, rows) in runs.items():
            hints_of, blocks, ms = {}, {}, []
            for q, pr in preds.items():
                it = items[q]
                t0 = time.time()
                shown = pr.get("ctx_tables")
                hs = idx_of[it.db_path].hints("%s\n%s" % (it.question or "", it.evidence or ""), shown, p)
                ms.append(1000 * (time.time() - t0))
                hints_of[q] = hs
                b = VG.render_block(hs, True)
                blocks[q] = {"tokens": est_tokens(b) if b else 0}
            pop = population(rows)
            rec = {}
            for pname, prow in pop.items():
                if pname == "verbatim_only":
                    continue
                r, det = answer_recall(prow, hints_of, index_vals)
                rec[pname] = r
                if pname == "reachable":
                    rec["reachable_detail"] = det
            rec["verbatim_only"] = verbatim_recall(pop["verbatim_only"], hints_of)
            nz = noise(preds, items, hints_of, blocks, schemas)
            nz["hint_ms_mean"] = statistics.mean(ms) if ms else 0.0
            nz["hint_ms_max"] = max(ms) if ms else 0.0
            res["splits"][label] = {"recall": rec, "noise": nz, "n_wrong_in_classes": len(rows)}
            pooled_h.update(hints_of)
            pooled_blocks.update(blocks)
            pooled_p.update(preds)
            pooled_i.update(items)
            pooled_rows.extend(rows)
        pop = population(pooled_rows)
        rp, _ = answer_recall(pop["reachable"], pooled_h, index_vals)
        rv = verbatim_recall(pop["verbatim_only"], pooled_h)
        rn, _ = answer_recall(pop["nonverbatim"], pooled_h, index_vals)
        res["pooled"] = {"recall_reachable": rp, "recall_nonverbatim_with_stored_gold": rn, "recall_verbatim_only": rv, "noise": noise(pooled_p, pooled_i, pooled_h, pooled_blocks, schemas),
                         "reachable_answers": len(pop["reachable"]), "nonverbatim_answers": len(pop["nonverbatim"])}
        out["variants"][name] = res
        z, rr = res["pooled"]["noise"], res["pooled"]["recall_reachable"]
        print("%-14s reachable %d: col recall %.3f (lit %.3f, in index %.3f), answers any/all %d/%d of %d | with hint %.1f%% hints/q %.2f tokens/q %.1f (with hint %.1f) | gold-hit prec %.3f"
              " | correct w/ hint %d (conflict %d)" % (name, rr["answers"], rr["col_recall"], rr["lit_recall"], rr["in_index"], rr["answer_any_col"], rr["answer_all_lit"],
                                                   rr["answers"], 100 * z["with_hint_frac"], z["hints_per_q"], z["tokens_per_q"], z["tokens_per_q_with_hint"], z["gold_hit_precision"],
                                                   z["correct_with_hint"], z["correct_with_conflicting_hint"]), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False, default=str)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", required=True, help="LABEL=RUN_DIR (preds.jsonl of a gated HR / HT run)")
    ap.add_argument("--variant", action="append", default=[], help="NAME=JSON overrides of values.VG_DEFAULTS (query-time parameters)")
    ap.add_argument("--build", default=None, help="JSON overrides of the BUILD parameters (max_len, max_distinct, max_total, scan_rows, stmt_timeout_s, budget_s)")
    ap.add_argument("--rows-cache", default=None, help="directory for the cached value_check rows (database checks are slow)")
    ap.add_argument("--no-db", action="store_true", help="skip value_check's existence / cardinality lookups (then no literal is 'reachable')")
    ap.add_argument("--out", default=None)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
