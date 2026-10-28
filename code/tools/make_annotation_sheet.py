#!/usr/bin/env python3
"""E10 annotation sheet generator (runs on a worker machine: needs the sqlite files; Python >= 3.8).

Annotators (SAP Amendment 2, 2026-10-26): two independent Claude Sonnet 5.5 runs (A1, A2) = "LLM annotation", NOT human or expert
annotation. annotation_A1.csv / annotation_A2.csv hold the same items in independent random orders, blind to system and auto class. Give each
run ONLY its own csv + schemas/ (never annotation_key.csv / sheet_meta.json). Instructions: docs/ANNOTATION_PROMPT.md. Agreement, adjudication
file and prevalence tables: code/tools/annotation_kappa.py.

  python tools/make_annotation_sheet.py --split bird_dev --preds zs=<run>/preds.jsonl full=<run>/preds.jsonl --out <dir>
         [--n 300] [--per-stratum 25] [--seed 20261027] [--pilot 30] [--rows 10]

Frame = EX=0 items of the listed systems (SPEC E10: zs and full, r0). Strata = collapsed auto class x system:
  exec_error+timeout | empty_result | shape_mismatch(+permutation) | row_superset/subset | value_mismatch | other
(the 'other' bucket also takes gold_fail and any class not listed). Allocation: `per-stratum` items per stratum, strata smaller than
that are taken whole and the shortfall is redistributed proportionally to the remaining capacity of the other strata, until
`n` items are drawn (or the frame is exhausted). Draws use random.Random(seed) on the sorted item list => deterministic.
Outputs in <out>/ :
  annotation_A1.csv, annotation_A2.csv   same items, independent random order, blinded (no system, no auto class); blank label columns
  annotation_key.csv                     item_id, qid, system, auto_class, stratum, stratum_population, stratum_sample, weight (analysis only)
  pilot_A1.csv / pilot_A2.csv            optional (--pilot N): items disjoint from the main sample, same format, to calibrate the guideline once
  schemas/<db_id>.txt                    schema (DDL with sample values) referenced by `schema_ref`
  sheet_meta.json                        parameters, frame sizes per stratum, sha256 of the files
  adjudication:  python tools/make_annotation_sheet.py --adjudicate A1_filled.csv A2_filled.csv --out <dir>  ->  adjudication.csv with
                 the disagreeing items only and the two labels in random (blind) positions label_X / label_Y, plus Cohen kappa preview.
Material per item (SPEC): question, evidence, schema, gold SQL, predicted SQL, first 10 rows of both results. Item order is random."""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

STRATA = ["exec_error+timeout", "empty_result", "shape_mismatch(+permutation)", "row_superset/subset", "value_mismatch", "other"]
LABELS = {"L1": "schema linking (wrong table/column)", "L2": "join path", "L3": "filter/value", "L4": "aggregation/grouping/ordering",
          "L5": "wrong criterion (exploratory)", "L6": "gold suspect", "L7": "ambiguous question", "L8": "other"}
ANNOT_COLS = ["primary_label", "secondary_label", "rationale", "notes"]


def stratum_of(auto_class):
    c = auto_class or "other"
    if c.startswith("exec_error") or c == "timeout":
        return "exec_error+timeout"
    if c == "empty_result":
        return "empty_result"
    if c in ("shape_mismatch", "column_permutation"):
        return "shape_mismatch(+permutation)"
    if c in ("row_superset", "row_subset"):
        return "row_superset/subset"
    if c == "value_mismatch":
        return "value_mismatch"
    return "other"


def allocate(sizes, n, per):
    """sizes: {stratum: population}. per-stratum quota with whole-stratum take for small ones and proportional redistribution."""
    take = {s: min(per, sizes[s]) for s in sizes}
    while sum(take.values()) > n:                         # per * strata > n (only with non-default settings): trim the largest takes
        take[max(sorted(take), key=lambda x: take[x])] -= 1
    short = n - sum(take.values())
    while short > 0:
        cap = {s: sizes[s] - take[s] for s in sizes if sizes[s] - take[s] > 0}
        if not cap:
            break
        tot = sum(cap.values())
        add = {s: min(cap[s], int(short * cap[s] / tot)) for s in cap}
        if sum(add.values()) == 0:                       # remainder: one by one in order of remaining capacity
            for s in sorted(cap, key=lambda x: -cap[x])[:short]:
                add[s] = 1
        for s, a in add.items():
            take[s] += a
        short = n - sum(take.values())
    return take


def fmt_rows(res, k):
    if not res["ok"]:
        return "ERROR: %s" % str(res["error"])[:300]
    rows = res["rows"]
    lines = ["columns: %s | rows: %d%s" % (", ".join(res.get("cols") or []), len(rows), "+" if res.get("truncated") else "")]
    for r in rows[:k]:
        lines.append(" | ".join(("NULL" if c is None else str(c))[:60] for c in r))
    if not rows:
        lines.append("(no rows)")
    return "\n".join(lines)


def read_preds(spec):
    name, path = spec.split("=", 1)
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return name, {r["qid"]: r for r in rows}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def adjudicate(args):
    a = {r["item_id"]: r for r in csv.DictReader(open(args.adjudicate[0], encoding="utf-8"))}
    b = {r["item_id"]: r for r in csv.DictReader(open(args.adjudicate[1], encoding="utf-8"))}
    ids = [i for i in a if i in b]
    la = [a[i]["primary_label"].strip() for i in ids]
    lb = [b[i]["primary_label"].strip() for i in ids]
    dis = [i for i in ids if a[i]["primary_label"].strip() != b[i]["primary_label"].strip()]
    rnd = random.Random(args.seed + 1)
    out = []
    for i in dis:
        flip = rnd.random() < 0.5
        x, y = (b[i], a[i]) if flip else (a[i], b[i])
        r = {k: a[i][k] for k in a[i] if k not in ANNOT_COLS}
        r.update({"label_X": x["primary_label"], "rationale_X": x["rationale"], "label_Y": y["primary_label"], "rationale_Y": y["rationale"],
                  "final_label": "", "final_rationale": ""})
        out.append(r)
    cols = [k for k in out[0]] if out else ["item_id"]
    write_csv(os.path.join(args.out, "adjudication.csv"), out, cols)
    print(json.dumps({"n_items": len(ids), "n_disagree": len(dis), "pct_agree": round(1 - len(dis) / max(1, len(ids)), 4),
                      "cohen_kappa_primary": cohen_kappa(la, lb)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="bird_dev")
    ap.add_argument("--preds", nargs="*", default=[], help="name=path/to/preds.jsonl (one per system)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--per-stratum", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--adjudicate", nargs=2, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.adjudicate:
        return adjudicate(args)
    from nl2sql import data, db
    items = {i.qid: i for i in data.load_split(args.split)}
    gold = data.load_gold(args.split, run_id="annotation_sheet")
    frame = []
    for spec in args.preds:
        name, preds = read_preds(spec)
        for qid, r in preds.items():
            if r.get("ex") == 0 and qid in items:
                frame.append({"qid": qid, "system": name, "auto_class": r.get("auto_class") or r.get("error_class") or "other",
                              "final_sql": r.get("final_sql") or ""})
    frame.sort(key=lambda r: (r["system"], r["qid"]))
    cells = defaultdict(list)
    for r in frame:
        r["stratum"] = stratum_of(r["auto_class"])
        cells[(r["system"], r["stratum"])].append(r)
    systems = sorted({r["system"] for r in frame})
    keys = [(s, st) for s in systems for st in STRATA]
    sizes = {k: len(cells.get(k, [])) for k in keys}
    take = allocate({"%s|%s" % k: v for k, v in sizes.items()}, args.n, args.per_stratum)
    rnd = random.Random(args.seed)
    chosen, rest = [], []
    for k in keys:
        pool = list(cells.get(k, []))
        rnd.shuffle(pool)
        t = take["%s|%s" % k]
        chosen += pool[:t]
        rest += pool[t:]
    # pilot: disjoint items (also disjoint by qid from the main sample), drawn at random from the remainder
    pilot = []
    if args.pilot:
        used = {r["qid"] for r in chosen}
        cand = [r for r in rest if r["qid"] not in used]
        rnd.shuffle(cand)
        pilot = cand[: args.pilot]

    def materialise(sel, prefix):
        out = []
        for r in sel:
            it = items[r["qid"]]
            g = db.execute(it.db_path, gold[r["qid"]], 30, args.rows)
            p = db.execute(it.db_path, r["final_sql"], 30, args.rows) if r["final_sql"] else {"ok": False, "error": "empty sql", "rows": [], "cols": []}
            out.append(dict(r, db_id=it.db_id, question=it.question, evidence=it.evidence or "", gold_sql=gold[r["qid"]],
                            gold_result=fmt_rows(g, args.rows), pred_result=fmt_rows(p, args.rows)))
        return out

    main_items = materialise(chosen, "A")
    pilot_items = materialise(pilot, "P")
    os.makedirs(os.path.join(args.out, "schemas"), exist_ok=True)
    for r in main_items + pilot_items:
        fn = os.path.join(args.out, "schemas", r["db_id"] + ".txt")
        if not os.path.exists(fn):
            it = items[r["qid"]]
            open(fn, "w", encoding="utf-8").write(data.get_schema(it)["ddl"])

    sheet_cols = ["item_id", "db_id", "question", "evidence", "schema_ref", "gold_sql", "pred_sql", "gold_result", "pred_result"] + ANNOT_COLS

    def emit(rows, prefix, tag):
        ids = ["%s%03d" % (prefix, i + 1) for i in range(len(rows))]
        order = list(range(len(rows)))
        rnd.shuffle(order)
        base = []
        for i in order:
            r = rows[i]
            base.append(dict(item_id=ids[i], db_id=r["db_id"], question=r["question"], evidence=r["evidence"], schema_ref="schemas/%s.txt" % r["db_id"],
                             gold_sql=r["gold_sql"], pred_sql=r["final_sql"], gold_result=r["gold_result"], pred_result=r["pred_result"],
                             primary_label="", secondary_label="", rationale="", notes=""))
        for a in ("A1", "A2"):                              # item ids are the same; the order is independent per annotator
            o = list(base)
            random.Random("%s-%s-%d" % (tag, a, args.seed)).shuffle(o)
            write_csv(os.path.join(args.out, "%s_%s.csv" % (tag, a)), o, sheet_cols)
        return dict(zip(range(len(rows)), ids))

    ids_main = emit(main_items, "A", "annotation")
    ids_pilot = emit(pilot_items, "P", "pilot") if pilot_items else {}
    pop = Counter()
    for r in frame:
        pop[(r["system"], r["stratum"])] += 1
    smp = Counter((r["system"], r["stratum"]) for r in main_items)
    key_rows = []
    for i, r in enumerate(main_items):
        k = (r["system"], r["stratum"])
        key_rows.append({"item_id": ids_main[i], "qid": r["qid"], "system": r["system"], "auto_class": r["auto_class"], "stratum": r["stratum"],
                         "stratum_population": pop[k], "stratum_sample": smp[k], "weight": round(pop[k] / smp[k], 6)})
    for i, r in enumerate(pilot_items):
        key_rows.append({"item_id": ids_pilot[i], "qid": r["qid"], "system": r["system"], "auto_class": r["auto_class"], "stratum": r["stratum"],
                         "stratum_population": pop[(r["system"], r["stratum"])], "stratum_sample": 0, "weight": ""})
    write_csv(os.path.join(args.out, "annotation_key.csv"), key_rows,
              ["item_id", "qid", "system", "auto_class", "stratum", "stratum_population", "stratum_sample", "weight"])
    meta = {"split": args.split, "systems": systems, "seed": args.seed, "n_requested": args.n, "per_stratum": args.per_stratum,
            "n_main": len(main_items), "n_pilot": len(pilot_items), "frame_size": len(frame),
            "population": {"%s|%s" % k: pop[k] for k in keys}, "sample": {"%s|%s" % k: smp[k] for k in keys},
            "files": {fn: sha(os.path.join(args.out, fn)) for fn in sorted(os.listdir(args.out)) if fn.endswith(".csv")}}
    json.dump(meta, open(os.path.join(args.out, "sheet_meta.json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("n_main", "n_pilot", "frame_size")}), "\n" + "\n".join(
        "  %-10s %-30s pop=%-4d sample=%d" % (s, st, pop[(s, st)], smp[(s, st)]) for s in systems for st in STRATA))


if __name__ == "__main__":
    main()
