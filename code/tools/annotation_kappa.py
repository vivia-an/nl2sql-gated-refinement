#!/usr/bin/env python3
"""E10 agreement + prevalence tool (host side, stdlib only, Python >= 3.6). SAP Amendment 2 (section 14): the two annotators are two
independent Claude Sonnet 5.5 runs ("LLM annotation"); kappa is agreement between two runs of ONE model, never human/expert agreement.

  annotation_kappa.py split   --sheet annotation_A1.csv --chunk 50 --out DIR            -> DIR/chunk_01.csv ... (header repeated)
  annotation_kappa.py merge   --labels DIR/labels_chunk_*.csv --out DIR/labels_A1.csv   -> one labels file (validated)
  annotation_kappa.py analyze --a1 labels_A1.csv --a2 labels_A2.csv [--sheet annotation_A1.csv] [--key annotation_key.csv]
                              [--adjudicated final_labels.csv] [--boot 2000] [--seed 20261101] --out DIR

A labels file is a CSV with at least item_id, primary_label (L1..L8) and optionally secondary_label, rationale, notes; a filled annotation sheet
(annotation_A*.csv with the label columns filled) is accepted too.

analyze writes into DIR:
  agreement.json      n, percent agreement, Cohen kappa (unweighted), Gwet AC1 for (a) the primary label (8 categories) and (b) L5-vs-rest, each
                      with an item-bootstrap percentile 95 % CI (--boot reps, seed --seed = SAP bootstrap seed 20261101); per-label counts of both
                      runs; per-system kappa if --key is given
  confusion.csv       rows = run A1, columns = run A2, counts, with marginals
  adjudication.csv    the DISAGREEING items only: material of --sheet (question, evidence, schema_ref, gold/pred SQL and results), label_X/rationale_X
                      and label_Y/rationale_Y with A1/A2 in random positions (random.Random(seed+1) per item), empty final_label/final_rationale.
                      The columns do not say which run is X.  (A mapping X/Y -> A1/A2 is stored in xy_key.json, NOT to be shown to the adjudicator.)
  prevalence_overall.csv / prevalence_<system>.csv
                      stratum-weighted prevalence of L1..L8 (weights = stratum population / stratum sample from --key), for run A1, run A2 and, when
                      --adjudicated is given, the final label (= common label where the runs agree, adjudicated label where they disagree),
                      with stratified-bootstrap 95 % CI (items resampled with replacement inside each stratum)
Everything is exploratory (SAP section 14); nothing here tests a hypothesis."""
import argparse
import csv
import glob
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

LABELS = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]
NEED_RATIONALE = {"L5", "L6", "L7"}


# ----------------------------------------------------------------------------- agreement statistics
def percent_agreement(a, b):
    return sum(1 for x, y in zip(a, b) if x == y) / float(len(a)) if a else None


def cohen_kappa(a, b):
    """Unweighted Cohen kappa. pe == 1 (both raters use one and the same single category): 1.0 if they agree everywhere, else None."""
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / float(n)
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / float(n * n)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else None
    return (po - pe) / (1.0 - pe)


def gwet_ac1(a, b, q):
    """Gwet's AC1 for two raters and q categories: pe = 1/(q-1) * sum_k pi_k (1 - pi_k), pi_k = mean of the two raters' marginal shares."""
    n = len(a)
    if n == 0 or q < 2:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / float(n)
    ca, cb = Counter(a), Counter(b)
    cats = set(ca) | set(cb)
    pe = sum(((ca.get(k, 0) + cb.get(k, 0)) / (2.0 * n)) * (1 - (ca.get(k, 0) + cb.get(k, 0)) / (2.0 * n)) for k in cats) / (q - 1.0)
    return (po - pe) / (1.0 - pe) if pe < 1.0 else None


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p
    f, c = int(math.floor(k)), int(math.ceil(k))
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def boot_ci(stat, a, b, reps, rnd):
    """Item bootstrap percentile 95 % CI of stat(a, b); replicates where the statistic is undefined are dropped (their number is returned)."""
    n = len(a)
    vals, dropped = [], 0
    for _ in range(reps):
        idx = [rnd.randrange(n) for _ in range(n)]
        v = stat([a[i] for i in idx], [b[i] for i in idx])
        if v is None or v != v:
            dropped += 1
        else:
            vals.append(v)
    vals.sort()
    return [pct(vals, 0.025), pct(vals, 0.975)], dropped


def r4(x):
    return None if x is None else round(x, 4)


def agreement_block(a, b, q, reps, rnd):
    out = {"n": len(a), "pct_agree": r4(percent_agreement(a, b))}
    k = cohen_kappa(a, b)
    ac = gwet_ac1(a, b, q)
    kci, kdrop = boot_ci(cohen_kappa, a, b, reps, rnd) if reps else ([None, None], 0)
    aci, adrop = boot_ci(lambda x, y: gwet_ac1(x, y, q), a, b, reps, rnd) if reps else ([None, None], 0)
    out.update({"kappa": r4(k), "kappa_ci95": [r4(kci[0]), r4(kci[1])], "kappa_boot_dropped": kdrop,
                "ac1": r4(ac), "ac1_ci95": [r4(aci[0]), r4(aci[1])], "ac1_boot_dropped": adrop})
    return out


# ----------------------------------------------------------------------------- csv helpers
def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_labels(path, strict=True):
    """-> {item_id: row}. Validates labels; raises SystemExit with the list of problems."""
    rows = read_csv(path)
    out, bad = {}, []
    for r in rows:
        iid = (r.get("item_id") or "").strip()
        lab = (r.get("primary_label") or "").strip().upper()
        if not iid:
            bad.append("row without item_id")
            continue
        if iid in out:
            bad.append("duplicate item_id %s" % iid)
        if lab not in LABELS:
            bad.append("%s: invalid primary_label %r" % (iid, r.get("primary_label")))
        sec = (r.get("secondary_label") or "").strip().upper()
        if sec and sec not in LABELS:
            bad.append("%s: invalid secondary_label %r" % (iid, r.get("secondary_label")))
        if lab in NEED_RATIONALE and not (r.get("rationale") or "").strip():
            bad.append("%s: %s without rationale" % (iid, lab))
        out[iid] = {"primary_label": lab, "secondary_label": sec, "rationale": (r.get("rationale") or "").strip(),
                    "notes": (r.get("notes") or "").strip()}
    if bad and strict:
        sys.exit("labels file %s is not acceptable:\n  %s" % (path, "\n  ".join(bad[:30]) + ("\n  ... %d more" % (len(bad) - 30) if len(bad) > 30 else "")))
    return out


# ----------------------------------------------------------------------------- split / merge
def cmd_split(a):
    rows = read_csv(a.sheet)
    cols = list(rows[0].keys())
    for c in ("primary_label", "secondary_label", "rationale", "notes"):
        if c not in cols:
            sys.exit("%s has no column %s: is it an annotation sheet?" % (a.sheet, c))
    if any((r.get("primary_label") or "").strip() for r in rows):
        sys.exit("refusing to split a sheet that already contains labels")
    os.makedirs(a.out, exist_ok=True)
    n_chunks = (len(rows) + a.chunk - 1) // a.chunk
    for k in range(n_chunks):
        write_csv(os.path.join(a.out, "chunk_%02d.csv" % (k + 1)), rows[k * a.chunk:(k + 1) * a.chunk], cols)
    print("%d items -> %d chunks of <= %d in %s (file order kept: the order is the annotator's own random order)" % (len(rows), n_chunks, a.chunk, a.out))


def cmd_merge(a):
    files = sorted(sum([glob.glob(p) for p in a.labels], []))
    if not files:
        sys.exit("no label files matched")
    merged = {}
    for f in files:
        part = load_labels(f)
        for iid, r in part.items():
            if iid in merged:
                sys.exit("item_id %s appears in more than one chunk (%s)" % (iid, f))
            merged[iid] = r
    if a.expect:
        want = {(r.get("item_id") or "").strip() for r in read_csv(a.expect)}
        miss, extra = sorted(want - set(merged)), sorted(set(merged) - want)
        if miss or extra:
            sys.exit("merged labels do not match %s: missing %d (%s...), unexpected %d (%s...)" % (a.expect, len(miss), miss[:5], len(extra), extra[:5]))
    cols = ["item_id", "primary_label", "secondary_label", "rationale", "notes"]
    write_csv(a.out, [dict(item_id=k, **v) for k, v in sorted(merged.items())], cols)
    print("merged %d files, %d items -> %s; counts %s" % (len(files), len(merged), a.out, dict(sorted(Counter(v["primary_label"] for v in merged.values()).items()))))


# ----------------------------------------------------------------------------- prevalence
def weighted_prev(items, labels_of, weight_of):
    tot = sum(weight_of(i) for i in items)
    if tot <= 0:
        return {k: None for k in LABELS}
    c = defaultdict(float)
    for i in items:
        c[labels_of(i)] += weight_of(i)
    return {k: c.get(k, 0.0) / tot for k in LABELS}


def strat_boot_prev(items, strata_of, labels_of, weight_of, reps, rnd):
    """Stratified bootstrap: resample items with replacement within each stratum (stratum size and weight fixed)."""
    by = defaultdict(list)
    for i in items:
        by[strata_of(i)].append(i)
    draws = {k: [] for k in LABELS}
    for _ in range(reps):
        sample = []
        for s, its in by.items():
            sample += [its[rnd.randrange(len(its))] for _ in range(len(its))]
        p = weighted_prev(sample, labels_of, weight_of)
        for k in LABELS:
            if p[k] is not None:
                draws[k].append(p[k])
    out = {}
    for k in LABELS:
        d = sorted(draws[k])
        out[k] = [pct(d, 0.025), pct(d, 0.975)]
    return out


def prevalence_table(items, sources, reps, rnd):
    """sources: {name: function(item) -> label}. -> list of rows (label x source) with estimate and CI."""
    rows = []
    for name, lab in sources.items():
        est = weighted_prev(items, lab, lambda i: i["weight"])
        ci = strat_boot_prev(items, lambda i: i["stratum"], lab, lambda i: i["weight"], reps, rnd) if reps else {}
        unw = Counter(lab(i) for i in items)
        for k in LABELS:
            rows.append({"source": name, "label": k, "n_sample": unw.get(k, 0), "weighted_prevalence": r4(est[k]),
                         "ci95_lo": r4(ci.get(k, [None, None])[0]), "ci95_hi": r4(ci.get(k, [None, None])[1])})
    return rows


# ----------------------------------------------------------------------------- analyze
def cmd_analyze(a):
    os.makedirs(a.out, exist_ok=True)
    A1, A2 = load_labels(a.a1), load_labels(a.a2)
    ids = sorted(set(A1) & set(A2))
    only1, only2 = sorted(set(A1) - set(A2)), sorted(set(A2) - set(A1))
    if only1 or only2:
        print("WARNING: %d items only in A1, %d only in A2 (ignored): %s" % (len(only1), len(only2), (only1 + only2)[:6]), file=sys.stderr)
    if not ids:
        sys.exit("no common items")
    l1 = [A1[i]["primary_label"] for i in ids]
    l2 = [A2[i]["primary_label"] for i in ids]
    rnd = random.Random(a.seed)
    key = {}
    if a.key:
        for r in read_csv(a.key):
            key[r["item_id"]] = r
    res = {"note": "LLM annotation: agreement between two independent Claude Sonnet 5.5 runs (SAP section 14); consistency of one model, not validity; "
                   "reported before adjudication; exploratory",
           "n_items": len(ids), "boot_reps": a.boot, "boot_seed": a.seed,
           "primary_label": agreement_block(l1, l2, len(LABELS), a.boot, rnd),
           "L5_vs_rest": agreement_block(["L5" if x == "L5" else "rest" for x in l1], ["L5" if x == "L5" else "rest" for x in l2], 2, a.boot, rnd),
           "counts_A1": dict(sorted(Counter(l1).items())), "counts_A2": dict(sorted(Counter(l2).items()))}
    if key:
        per = {}
        for s in sorted({key[i]["system"] for i in ids if i in key}):
            sel = [i for i in ids if i in key and key[i]["system"] == s]
            x, y = [A1[i]["primary_label"] for i in sel], [A2[i]["primary_label"] for i in sel]
            per[s] = {"n": len(sel), "pct_agree": r4(percent_agreement(x, y)), "kappa": r4(cohen_kappa(x, y)), "ac1": r4(gwet_ac1(x, y, len(LABELS)))}
        res["per_system"] = per
    # confusion table
    cm = defaultdict(int)
    for x, y in zip(l1, l2):
        cm[(x, y)] += 1
    rows = []
    for x in LABELS:
        r = {"A1\\A2": x}
        for y in LABELS:
            r[y] = cm.get((x, y), 0)
        r["total_A1"] = sum(cm.get((x, y), 0) for y in LABELS)
        rows.append(r)
    rows.append(dict({"A1\\A2": "total_A2"}, **dict((y, sum(cm.get((x, y), 0) for x in LABELS)) for y in LABELS), total_A1=len(ids)))
    write_csv(os.path.join(a.out, "confusion.csv"), rows, ["A1\\A2"] + LABELS + ["total_A1"])
    # disagreement list (blinded positions)
    sheet = {}
    if a.sheet:
        for r in read_csv(a.sheet):
            sheet[r["item_id"]] = r
    mat_cols = ["question", "evidence", "schema_ref", "gold_sql", "pred_sql", "gold_result", "pred_result"]
    dis, xy = [], {}
    for i in ids:
        if A1[i]["primary_label"] == A2[i]["primary_label"]:
            continue
        flip = random.Random("%d|%s" % (a.seed + 1, i)).random() < 0.5       # deterministic per item
        x, y = (A2[i], A1[i]) if flip else (A1[i], A2[i])
        xy[i] = {"X": "A2" if flip else "A1", "Y": "A1" if flip else "A2"}
        row = {"item_id": i}
        row.update({c: sheet.get(i, {}).get(c, "") for c in mat_cols})
        row.update({"label_X": x["primary_label"], "rationale_X": x["rationale"], "label_Y": y["primary_label"], "rationale_Y": y["rationale"],
                    "final_label": "", "final_rationale": ""})
        dis.append(row)
    write_csv(os.path.join(a.out, "adjudication.csv"), dis, ["item_id"] + mat_cols + ["label_X", "rationale_X", "label_Y", "rationale_Y", "final_label", "final_rationale"])
    json.dump(xy, open(os.path.join(a.out, "xy_key.json"), "w"), indent=1, sort_keys=True)
    res["n_disagree"] = len(dis)
    if a.sheet:
        miss = [r["item_id"] for r in dis if not r["question"]]
        if miss:
            print("WARNING: %d disagreeing items are not in --sheet; their material columns are empty" % len(miss), file=sys.stderr)
    # prevalence
    final = None
    if a.adjudicated:
        fin = {}
        for r in read_csv(a.adjudicated):
            lab = (r.get("final_label") or "").strip().upper()
            if lab not in LABELS:
                sys.exit("%s: invalid final_label %r for %s" % (a.adjudicated, r.get("final_label"), r.get("item_id")))
            fin[r["item_id"].strip()] = lab
        missing = [r["item_id"] for r in dis if r["item_id"] not in fin]
        if missing:
            sys.exit("adjudicated file lacks %d disagreeing items: %s" % (len(missing), missing[:5]))
        final = {i: (A1[i]["primary_label"] if A1[i]["primary_label"] == A2[i]["primary_label"] else fin[i]) for i in ids}
        diff_both = sum(1 for i in ids if A1[i]["primary_label"] != A2[i]["primary_label"] and fin[i] not in (A1[i]["primary_label"], A2[i]["primary_label"]))
        res["n_adjudicated_label_differs_from_both_runs"] = diff_both
        res["n_adjudicated_equals_A1"] = sum(1 for i in ids if A1[i]["primary_label"] != A2[i]["primary_label"] and fin[i] == A1[i]["primary_label"])
        res["n_adjudicated_equals_A2"] = sum(1 for i in ids if A1[i]["primary_label"] != A2[i]["primary_label"] and fin[i] == A2[i]["primary_label"])
    if key:
        items = []
        for i in ids:
            k = key.get(i)
            if not k or not k.get("weight"):
                continue
            items.append({"id": i, "system": k["system"], "stratum": k["system"] + "|" + k["stratum"], "weight": float(k["weight"])})
        srcs = {"A1": lambda it: A1[it["id"]]["primary_label"], "A2": lambda it: A2[it["id"]]["primary_label"]}
        if final:
            srcs["final"] = lambda it: final[it["id"]]
        prev_rows = prevalence_table(items, srcs, a.boot, rnd)
        cols = ["source", "label", "n_sample", "weighted_prevalence", "ci95_lo", "ci95_hi"]
        write_csv(os.path.join(a.out, "prevalence_overall.csv"), prev_rows, cols)
        for s in sorted({it["system"] for it in items}):
            sub = [it for it in items if it["system"] == s]
            # within a system the weights are stratum population / sample; prevalence is the system's own failure mix
            write_csv(os.path.join(a.out, "prevalence_%s.csv" % s), prevalence_table(sub, srcs, a.boot, rnd), cols)
        res["prevalence_note"] = ("weights = stratum population / stratum sample (annotation_key.csv); overall = weighted mix of the failures of the "
                                  "listed systems (a failure of each system counts once), per-system files = that system's failures only; CI = stratified bootstrap")
    json.dump(res, open(os.path.join(a.out, "agreement.json"), "w"), indent=1)
    p, l5 = res["primary_label"], res["L5_vs_rest"]
    print("n=%d agree=%.3f | primary: kappa=%s %s AC1=%s %s | L5-vs-rest: kappa=%s %s AC1=%s | disagree=%d -> %s/adjudication.csv" % (
        len(ids), p["pct_agree"], p["kappa"], p["kappa_ci95"], p["ac1"], p["ac1_ci95"], l5["kappa"], l5["kappa_ci95"], l5["ac1"], len(dis), a.out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("split")
    s.add_argument("--sheet", required=True)
    s.add_argument("--chunk", type=int, default=50)
    s.add_argument("--out", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--labels", nargs="+", required=True, help="label files or globs")
    m.add_argument("--out", required=True)
    m.add_argument("--expect", default=None, help="the sheet the labels must cover exactly (item_id column)")
    z = sub.add_parser("analyze")
    z.add_argument("--a1", required=True)
    z.add_argument("--a2", required=True)
    z.add_argument("--sheet", default=None, help="annotation sheet (material for adjudication.csv)")
    z.add_argument("--key", default=None, help="annotation_key.csv (strata, weights; analysis only)")
    z.add_argument("--adjudicated", default=None, help="final_labels.csv / adjudication.csv with final_label filled")
    z.add_argument("--boot", type=int, default=2000)
    z.add_argument("--seed", type=int, default=20261101)
    z.add_argument("--out", required=True)
    a = ap.parse_args()
    {"split": cmd_split, "merge": cmd_merge, "analyze": cmd_analyze}.get(a.cmd, lambda _: ap.print_help())(a)


if __name__ == "__main__":
    main()
