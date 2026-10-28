#!/usr/bin/env python
"""RESCORE: re-score the STORED dev predictions against corrected BIRD gold (SAP Amendment 7, section 20; scope `robustness-posthoc`).

No model or API call. Stages (each writes only below --out, default results/robust/):

  sets     data-level overlap of the corrected sets with the dev ids (no prediction, no execution)       -> overlap.json
  exec     executes original + corrected gold (dev DB and Mini-Dev DB) and every pinned run's final_sql
           (and the roots of full_fixed) once, scores them with the primary analysis' Evaluator       -> items_ex.csv.gz, gold_info.json
  analyze  contrasts, intervals, reading rule, E10, flips                                              -> rescore.json, e10.json, flips.csv
  all      sets + exec + analyze

The EX of the primary analysis is `nl2sql.eval.score.Evaluator.score_sql` (prediction: execute_robust 30 s; gold: 120 s, cached;
set equality of row tuples; a failing gold gives EX 0 and stays in the denominator). Here the same class is used with the gold dictionary
replaced, so "original" and "corrected" differ only by the gold. Statistics reuse the committed `analysis_pipeline.stats/routing`
(db-clustered paired percentile bootstrap, B = 10 000, seed 20261101). Corrected gold SQL text is never written to the output.

Run on a worker machine (numpy + the data tree):
  python code/tools/rescore_corrected.py all --data $R/data --runs $R/runs --work $R/rescore_work --out results/robust   (R = the shared tree, CONTRACT section 1)
"""
from __future__ import print_function

import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(CODE_DIR)

SEED = 20261101
B_DEFAULT = 10000
MARGIN_PP = 1.5            # SAP 13.3 negligible margin (type N rule)
STABLE_PP = 1.0            # SAP 20.7 'stable'
STABLE_AUC = 0.02
GOLDS = ["orig", "G1", "G2", "G3", "G4"]
VARIANTS = collections.OrderedDict([
    ("G1", ("repos/text_to_sql_benchmarks/data/arcwise_plat_sql_only_with_diff.json", "Arcwise-Plat-SQL (Jin repo), primary")),
    ("G2", ("repos/text_to_sql_benchmarks/data/arcwise_plat_full_with_diff.json", "Arcwise-Plat full (Jin repo)")),
    ("G3", ("repos/ReViSQL/data/arcwise_plat_sql.json", "Arcwise-Plat-SQL (ReViSQL copy)")),
    ("G4", ("repos/ReViSQL/data/arcwise_plat_full.json", "Arcwise-Plat full (ReViSQL copy)")),
])
JIN_DEV100 = ("bird_dev_corrected_jin/dev/dev_corrected/dev.json", "bird_dev_corrected_jin/dev/dev_20240627/dev.json")
DEV_JSON = "bird/dev/dev_20240627/dev.json"
MINI_DB_DIR = "bird/minidev/minidev/MINIDEV/dev_databases"
DEV_DB_DIR = "bird/dev/dev_20240627/dev_databases"

# label -> run dir relative to runs/ (SAP 20.4)
RUNS = collections.OrderedDict([
    ("plus.zs", "MIN/M1/zs-s0"), ("plus.fixed", "MIN/M1/fixed-s0"), ("plus.gated", "MIN/M1/gated-s0"),
    ("plus.route", "MIN/M1/gated-route-s0"), ("plus.rootonly", "MIN/M3/rootonly-s0"),
    ("plus.acsql", "MIN/M2/acsql-s0"), ("plus.macsql", "MIN/M2/macsql-s0"), ("plus.sdebug", "MIN/M2/sdebug-s0"),
    ("plus.bonvote2", "MIN/M2/bonvote-s0"), ("plus.bonvote3", "MIN/M2/bonvote3-s0"),
    ("flash.zs", "MIN/M5/zs-flash-s0"), ("flash.fixed", "MIN/M5/fixed-flash-s0"), ("flash.gated", "MIN/M5/gated-flash-s0"),
    ("flash.rootonly", "MIN/M5/rootonly-flash-s0"),
    ("q27b.zs", "MIN/M5/zs-27b-s0"), ("q27b.fixed", "MIN/M5/fixed-27b-s0"), ("q27b.gated", "MIN/M5/gated-27b-s0"),
    ("q27b.rootonly", "MIN/M5/rootonly-27b-s0"),
])
ROOT_RUN = "plus.fixed"            # roots of this run are scored (E10)
EXEC_STATUS = ("ok_nonempty", "ok_empty")
PLUS_SYSTEMS = ["plus.zs", "plus.fixed", "plus.gated", "plus.rootonly", "plus.acsql", "plus.macsql", "plus.sdebug", "plus.bonvote2", "plus.bonvote3"]
KEYWORDS = ["DISTINCT", "LEFT JOIN", "JOIN", "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "CAST", "COUNT", "SUM", "AVG", "MAX", "MIN",
            "CASE", "IN", "EXISTS", "UNION"]


# ------------------------------------------------------------------------------------------------------------ pure helpers
def sstrip(x):
    return (x or "").strip()


def wsnorm(x):
    return re.sub(r"\s+", " ", x or "").strip()


def qid_of(row):
    return "bird_dev:%s:%s" % (row["db_id"], row["question_id"])


def sql_sha(sql):
    return hashlib.sha1((sql or "").strip().encode("utf-8", "replace")).hexdigest()[:12]


def kw_set(sql):
    u = " " + re.sub(r"\s+", " ", (sql or "").upper()) + " "
    out = set()
    for k in KEYWORDS:
        if re.search(r"(?<![A-Z_])" + k.replace(" ", r"\s+") + r"(?![A-Z_])", u):
            out.add(k)
    return out


def kw_diff(a, b):
    """keywords present in exactly one of the two SQL texts, sorted, joined by '|'"""
    return "|".join(sorted(kw_set(a) ^ kw_set(b)))


def build_sets(dev_by_id, variant_rows, discordant=()):
    """dev_by_id: {question_id:int -> dev row}; variant_rows: {question_id:int -> row of the corrected release}.
    -> dict(all=[qid], same=[qid], sql_text_changed={qid:bool}, q_changed, e_changed, db_discordant=[qid]).  qid = bird_dev:<db>:<id>.
    S_same = question AND evidence AND db_id unchanged (str.strip equality); gold SQL changed = whitespace-collapsed inequality."""
    out = {"all": [], "same": [], "sql_text_changed": {}, "q_changed": {}, "e_changed": {}, "db_discordant": []}
    for qi in sorted(variant_rows):
        if qi not in dev_by_id:
            continue
        r, o = variant_rows[qi], dev_by_id[qi]
        q = qid_of(o)
        if q in discordant:
            out["db_discordant"].append(q)
            continue
        qc = sstrip(r.get("question")) != sstrip(o.get("question"))
        ec = sstrip(r.get("evidence")) != sstrip(o.get("evidence"))
        dc = r.get("db_id") != o.get("db_id")
        out["all"].append(q)
        out["q_changed"][q], out["e_changed"][q] = qc, ec
        out["sql_text_changed"][q] = wsnorm(r.get("SQL")) != wsnorm(o.get("SQL"))
        if not (qc or ec or dc):
            out["same"].append(q)
    return out


def same_result(g1, g2):
    """compare two gold-info dicts (ok, digest): True iff identical outcome (both fail, or both ok with equal set digests)."""
    if g1 is None or g2 is None:
        return None
    if not g1["ok"] and not g2["ok"]:
        return True
    if g1["ok"] != g2["ok"]:
        return False
    return g1["digest"] == g2["digest"]


def overlap_ci(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def sign(x):
    return (x > 0) - (x < 0)


def classify(kind, d0, i0, d1, i1, s, i_s, unit="pp"):
    """SAP 20.7. kind 'S' (positive-effect claim) or 'N' (negligible-difference claim). unit 'pp' or 'auc'."""
    ov = overlap_ci(i0, i1)
    if kind == "S":
        robust = (d0 != 0) and sign(d1) == sign(d0) and ov
    else:
        margin = MARGIN_PP if unit == "pp" else None
        robust = ov and (abs(d1) <= margin if margin is not None else True)
    small = STABLE_PP if unit == "pp" else STABLE_AUC
    stable = bool(robust and ((i_s[0] <= 0 <= i_s[1]) or abs(s) < small))
    return {"robust": bool(robust), "stable": stable, "overlap": bool(ov), "sign_same": sign(d1) == sign(d0)}


def newcombe_diff(k1, n1, k2, n2, wilson):
    """Newcombe hybrid-score interval for p1 - p2 (square-and-add of Wilson limits)."""
    if not n1 or not n2:
        return None
    p1, p2 = k1 / float(n1), k2 / float(n2)
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return (d - ((p1 - l1) ** 2 + (u2 - p2) ** 2) ** 0.5, d + ((u1 - p1) ** 2 + (p2 - l2) ** 2) ** 0.5)


def revision_class(root_ex, final_ex, root_exec, changed):
    """E10: 'helped'|'hurt'|'neutral' for an executable-root revision, 'repair_*' for a non-executable root; None if unchanged."""
    if not changed:
        return None
    pre = "" if root_exec else "repair_"
    if root_ex == 0 and final_ex == 1:
        return pre + "helped"
    if root_ex == 1 and final_ex == 0:
        return pre + "hurt"
    return pre + "neutral"


def jl(path):
    out = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def jload(path):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def dump(obj, path):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True, ensure_ascii=False)


def load_variants(data_dir):
    dev = jload(os.path.join(data_dir, DEV_JSON))
    dev_by_id = {int(r["question_id"]): r for r in dev}
    var = {}
    for g, (p, _) in VARIANTS.items():
        rows = jload(os.path.join(data_dir, p))
        var[g] = {int(r["question_id"]): r for r in rows}
    return dev_by_id, var


# ------------------------------------------------------------------------------------------------------------ stage: sets
def stage_sets(a):
    dev_by_id, var = load_variants(a.data)
    out = {"_meta": {"scope": "robustness-posthoc", "sap": "section 20 (Amendment 7)", "stage": "sets", "n_dev": len(dev_by_id)}, "variants": {}}
    for g, rows in var.items():
        st = build_sets(dev_by_id, rows)
        n_same_chg = sum(1 for q in st["same"] if st["sql_text_changed"][q])
        out["variants"][g] = {
            "label": VARIANTS[g][1], "file": VARIANTS[g][0], "n_items": len(rows), "n_in_dev": len([i for i in rows if i in dev_by_id]),
            "n_S_all": len(st["all"]), "n_S_same": len(st["same"]),
            "n_q_changed": sum(st["q_changed"].values()), "n_e_changed": sum(st["e_changed"].values()),
            "n_sql_text_changed_S_all": sum(st["sql_text_changed"].values()), "n_sql_text_changed_S_same": n_same_chg,
            "S_same_by_db": dict(collections.Counter(q.split(":")[1] for q in st["same"])),
        }
    try:
        jc = {int(r["question_id"]): r for r in jload(os.path.join(a.data, JIN_DEV100[0]))}
        jo = {int(r["question_id"]): r for r in jload(os.path.join(a.data, JIN_DEV100[1]))}
        g1 = set(var["G1"])
        out["jin_dev100"] = {"n": len(jc), "n_in_dev": len(set(jc) & set(dev_by_id)), "n_in_plat": len(set(jc) & g1),
                             "n_union_with_plat": len(set(jc) | g1), "n_q_changed": sum(sstrip(jc[i]["question"]) != sstrip(jo[i]["question"]) for i in jc),
                             "n_e_changed": sum(sstrip(jc[i].get("evidence")) != sstrip(jo[i].get("evidence")) for i in jc),
                             "n_sql_changed": sum(wsnorm(jc[i]["SQL"]) != wsnorm(jo[i]["SQL"]) for i in jc),
                             "note": "not re-scored (corrected database files, n < 150); overlap only"}
    except Exception as e:     # data tree without the dev-100
        out["jin_dev100"] = {"error": str(e)}
    out["question_set_of_runs"] = {"split": "bird_dev", "n": len(dev_by_id), "source": DEV_JSON}
    dump(out, os.path.join(a.out, "overlap.json"))
    print(json.dumps({g: {k: v for k, v in d.items() if k.startswith("n_")} for g, d in out["variants"].items()}, indent=1))
    return out


# ------------------------------------------------------------------------------------------------------------ stage: exec
_G = {}   # worker globals


def _item(qid, db_id, db_path):
    from nl2sql.data import Item
    return Item(qid=qid, db_id=db_id, question="", evidence="", gold_sql="", db_path=db_path, split="bird_dev")


def make_evaluator(gold):
    """Evaluator with the gold dictionary replaced (no data.load_gold, hence no dev-access log entry)."""
    from nl2sql.eval.score import Evaluator, PRED_TIMEOUT_S, GOLD_TIMEOUT_S
    ev = Evaluator.__new__(Evaluator)
    ev.split, ev.timeout_s, ev.gold_timeout_s = "bird_dev", PRED_TIMEOUT_S, max(GOLD_TIMEOUT_S, PRED_TIMEOUT_S)
    ev.gold, ev.gold2, ev.items, ev.use_ts = dict(gold), {}, {}, False
    return ev


def _w_init(cache_dir, gold_path, data_dir):
    os.environ["NL2SQL_CACHE"] = cache_dir
    if CODE_DIR not in sys.path:
        sys.path.insert(0, CODE_DIR)
    golds = jload(gold_path)
    _G["ev"] = {g: make_evaluator(golds[g]) for g in golds}
    _G["data"] = data_dir


def _w_gold(task):
    g, qid, db_id, which = task
    from nl2sql.eval.score import rows_digest
    base = os.path.join(_G["data"], MINI_DB_DIR if which == "mini" else DEV_DB_DIR)
    it = _item(qid, db_id, os.path.join(base, db_id, db_id + ".sqlite"))
    ev = _G["ev"][g]
    sql = ev.gold.get(qid, "")
    t0 = time.time()
    try:
        r = ev.gold_exec(it)
    except Exception as e:          # never let one gold kill the pool
        r = {"ok": False, "error": "%s: %s" % (type(e).__name__, e), "rows": [], "cols": []}
    info = {"ok": bool(r and r.get("ok")), "error": (str(r.get("error") or "")[:100] if r else "no gold")}
    if info["ok"]:
        if r.get("big"):
            info.update({"n_rows": r["n_rows"], "digest": r["rows_digest"], "big": True})
        else:
            info.update({"n_rows": len(r["rows"]), "digest": rows_digest(r["rows"]), "big": False})
        info["n_cols"] = len(r.get("cols") or [])
    else:
        info.update({"n_rows": None, "digest": None, "n_cols": None, "big": False})
    info["elapsed_s"] = round(time.time() - t0, 3)
    info["sha"] = sql_sha(sql)
    return (g, qid, which), info


def _w_pred(task):
    qid, db_id, sql = task
    from nl2sql.db import execute_robust
    it = _item(qid, db_id, os.path.join(_G["data"], DEV_DB_DIR, db_id, db_id + ".sqlite"))
    t0 = time.time()
    try:
        pr = execute_robust(it.db_path, sql or "", timeout_s=30.0, max_rows=None)
    except Exception as e:
        pr = {"ok": False, "rows": [], "error": "%s: %s" % (type(e).__name__, e)}
    out = {"exec_ok": bool(pr["ok"]), "err": str(pr.get("error") or "")[:80]}
    for g, ev in _G["ev"].items():
        try:
            sc = ev.score_sql(it, sql or "", pred_res=pr)
            out[g] = sc["ex"]
            out["gf_" + g] = 1 if sc.get("gold_fail") else 0
        except Exception as e:
            out[g] = None
            out["gf_" + g] = None
            out["err_" + g] = "%s: %s" % (type(e).__name__, e)
    out["elapsed_s"] = round(time.time() - t0, 3)
    return (qid, sql), out


def route_features(rows, coef, k):
    """verbatim logic of analysis_pipeline.primary.route_scores at cccbafd (v3, route_ca per row; NaN as None)."""
    import math
    v3s, cas = [], []
    for r in rows:
        c = r.get("critic") or {}
        v3 = float(c["score"]) if (c.get("semantic") is not None and c.get("score") is not None) else None
        v3s.append(v3)
        bits = r.get("cons_bits")
        cons = (float(sum(bits[:k])) / len(bits[:k]) if (k and bits) else r.get("consistency"))
        st = r.get("final_status")
        nonempty = (1.0 if st == "ok_nonempty" else 0.0) if st is not None else (1.0 if r.get("exec_ok") else None)
        feats = {"v3": v3, "exec": c.get("syntax"), "consistency": cons, "nonempty": nonempty}
        vals = [feats.get(n) for n in coef["features"]]
        if any(v is None for v in vals):
            cas.append(None)
        else:
            z = coef["intercept"] + sum(w * v for w, v in zip(coef["coef"], vals))
            cas.append(10.0 / (1.0 + math.exp(-z)))
    return v3s, cas


def stage_exec(a):
    from multiprocessing import Pool
    work = a.work
    os.makedirs(work, exist_ok=True)
    cache = os.path.join(work, "cache")
    os.makedirs(cache, exist_ok=True)
    os.environ["NL2SQL_CACHE"] = cache
    sys.path.insert(0, CODE_DIR)
    dev_by_id, var = load_variants(a.data)
    ids = sorted(set().union(*[set(v) for v in var.values()]) & set(dev_by_id))
    qid_db = {qid_of(dev_by_id[i]): dev_by_id[i]["db_id"] for i in ids}
    golds = {"orig": {qid_of(dev_by_id[i]): dev_by_id[i]["SQL"] for i in ids}}
    for g, rows in var.items():
        golds[g] = {qid_of(dev_by_id[i]): rows[i]["SQL"] for i in ids if i in rows}
    gpath = os.path.join(work, "golds.json")
    dump(golds, gpath)           # corrected gold text stays in the worker machine work dir
    t0 = time.time()
    pool = Pool(a.procs, initializer=_w_init, initargs=(cache, gpath, a.data))
    try:
        # gold stage: every gold on the dev DB; corrected golds also on the Mini-Dev DB
        tasks = []
        for g in GOLDS:
            for q in golds[g]:
                tasks.append((g, q, qid_db[q], "dev"))
                if g != "orig":
                    tasks.append((g, q, qid_db[q], "mini"))
        ginfo = {}
        for k, info in pool.imap_unordered(_w_gold, tasks, chunksize=4):
            ginfo[k] = info
        print("gold stage: %d executions, %.0f s" % (len(tasks), time.time() - t0), flush=True)
        # kw diffs and per-item gold table
        gold_info = {}
        for g in GOLDS:
            for q in golds[g]:
                rec = {"dev": ginfo[(g, q, "dev")]}
                if g != "orig":
                    rec["mini"] = ginfo[(g, q, "mini")]
                    rec["kw_diff"] = kw_diff(golds["orig"][q], golds[g][q])
                    rec["sql_text_changed"] = wsnorm(golds["orig"][q]) != wsnorm(golds[g][q])
                gold_info["%s|%s" % (g, q)] = rec
        # prediction stage
        preds, tasks_set, rows_meta = {}, {}, []
        coef = jload(a.route_coef)
        qids = sorted(golds["orig"])
        qset = set(qids)
        for label, rel in RUNS.items():
            rd = os.path.join(a.runs, rel)
            rows = {}
            for r in jl(os.path.join(rd, "preds.jsonl")):
                rows[r["qid"]] = r
            sel = [rows[q] for q in qids if q in rows]
            v3 = ca = None
            if label == "plus.route":
                v3s, cas = route_features(sel, coef, 4)
                v3 = dict(zip([r["qid"] for r in sel], v3s))
                ca = dict(zip([r["qid"] for r in sel], cas))
            for r in sel:
                stored = r.get("ex")
                ec = r.get("error_class")
                if stored is None and ec in ("api_fail", "parse_fail"):
                    usable, forced = 1, 0
                elif stored is None:
                    usable, forced = 0, None
                else:
                    usable, forced = 1, None
                sql = r.get("final_sql") or ""
                rows_meta.append({"kind": "final", "label": label, "qid": r["qid"], "db": r["db_id"], "usable": usable, "forced0": forced,
                                  "ex_stored": stored, "err_class": ec or "", "sql": sql, "sha": sql_sha(sql), "root_status": "",
                                  "v3": (v3 or {}).get(r["qid"]), "route_ca": (ca or {}).get(r["qid"])})
                if usable and forced is None:
                    tasks_set[(r["qid"], sql)] = r["db_id"]
            if label == ROOT_RUN:
                for c in jl(os.path.join(rd, "cands.jsonl")):
                    if c.get("action") == "root" and c["qid"] in qset:
                        sql = c.get("sql") or ""
                        rows_meta.append({"kind": "root", "label": label, "qid": c["qid"], "db": qid_db[c["qid"]], "usable": 1, "forced0": None,
                                          "ex_stored": c.get("ex"), "err_class": "", "sql": sql, "sha": sql_sha(sql),
                                          "root_status": c.get("status") or "", "v3": None, "route_ca": None})
                        tasks_set[(c["qid"], sql)] = qid_db[c["qid"]]
        ptasks = sorted(((q, db, s) for (q, s), db in tasks_set.items()), key=lambda t: (t[1], t[0]))
        print("prediction stage: %d distinct (item, SQL) executions" % len(ptasks), flush=True)
        res = {}
        n = 0
        for k, out in pool.imap_unordered(_w_pred, ptasks, chunksize=8):
            res[k] = out
            n += 1
            if n % 2000 == 0:
                print("  %d / %d  %.0f s" % (n, len(ptasks), time.time() - t0), flush=True)
    finally:
        pool.close()
        pool.join()
    os.makedirs(a.out, exist_ok=True)
    cols = ["kind", "label", "qid", "db", "usable", "ex_stored", "err_class", "exec_ok", "sha", "root_status", "v3", "route_ca"] + GOLDS
    slow = []
    with gzip.open(os.path.join(a.out, "items_ex.csv.gz"), "wt", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for m in rows_meta:
            r = res.get((m["qid"], m["sql"]))
            if m["usable"] == 0:
                exs, eo = [""] * len(GOLDS), ""
            elif m["forced0"] is not None:
                exs, eo = [0] * len(GOLDS), 0
            else:
                exs = [("" if r[g] is None else r[g]) for g in GOLDS]
                eo = 1 if r["exec_ok"] else 0
                if r["elapsed_s"] > 20:
                    slow.append([m["label"], m["qid"], r["elapsed_s"]])
            w.writerow([m["kind"], m["label"], m["qid"], m["db"], m["usable"], "" if m["ex_stored"] is None else m["ex_stored"], m["err_class"],
                        eo, m["sha"], m["root_status"], "" if m["v3"] is None else m["v3"], "" if m["route_ca"] is None else m["route_ca"]] + exs)
    dump(gold_info, os.path.join(a.out, "gold_info.json"))
    meta = {"scope": "robustness-posthoc", "stage": "exec", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "n_gold_exec": len(tasks),
            "n_pred_exec": len(ptasks), "runs": RUNS, "slow_over_20s": slow[:50], "elapsed_s": round(time.time() - t0, 1),
            "variant_files": {g: VARIANTS[g][0] for g in VARIANTS}, "gold_timeout_s": 120, "pred_timeout_s": 30,
            "variant_sha256": {g: sha256_file(os.path.join(a.data, VARIANTS[g][0])) for g in VARIANTS}}
    dump(meta, os.path.join(a.out, "exec_meta.json"))
    print("exec done in %.0f s" % (time.time() - t0))


# ------------------------------------------------------------------------------------------------------------ stage: analyze
def load_items(path):
    """-> {(kind,label): {qid: dict}}"""
    out = collections.defaultdict(dict)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["kind"], r["label"])][r["qid"]] = r
    return out


def _num(x):
    return None if x in ("", None) else float(x)


class Ctx(object):
    """per (variant, set) analysis context: item list, db labels, cached bootstraps"""

    def __init__(self, S, items, qids, B):
        self.S, self.items, self.qids, self.B = S, items, list(qids), B
        self._boots = {}

    def usable(self, labels, kind="final"):
        out = []
        for q in self.qids:
            ok = True
            for l in labels:
                r = self.items[(kind, l)].get(q)
                if r is None or r["usable"] != "1":
                    ok = False
                    break
            if ok:
                out.append(q)
        return out

    def exv(self, label, g, qids, kind="final"):
        import numpy as np
        return np.array([_num(self.items[(kind, label)][q][g]) for q in qids], float)

    def dbs(self, qids):
        import numpy as np
        return np.array([q.split(":")[1] for q in qids])

    def boot(self, qids):
        key = hashlib.md5("|".join(qids).encode()).hexdigest()
        if key not in self._boots:
            self._boots[key] = self.S.ClusterBoot(self.dbs(qids), self.B, SEED)
        return self._boots[key]


def _ci(S, arr):
    lo, hi = S.pctl_ci(arr)
    return (lo, hi)


def paired_contrast(ctx, pairs, gk, qids=None, drop_db=None, with_item=True):
    """pairs: [(a_label, b_label)] averaged (pooled estimator = mean of the differences). Returns dict in percentage points."""
    import numpy as np
    S = ctx.S
    labels = sorted(set([x for p in pairs for x in p]))
    q = qids if qids is not None else ctx.usable(labels)
    if drop_db:
        q = [x for x in q if x.split(":")[1] != drop_db]
    if len(q) < 20:
        return {"n": len(q), "status": "too few items"}
    vo = np.mean([ctx.exv(a, "orig", q) - ctx.exv(b, "orig", q) for a, b in pairs], axis=0)
    vc = np.mean([ctx.exv(a, gk, q) - ctx.exv(b, gk, q) for a, b in pairs], axis=0)
    bt = ctx.boot(q)
    bo, bc, bs = bt.means(vo) * 100, bt.means(vc) * 100, bt.means(vc - vo) * 100
    out = {"n": len(q), "n_dbs": bt.k, "d0": float(vo.mean() * 100), "d1": float(vc.mean() * 100), "shift": float((vc - vo).mean() * 100),
           "ci0": _ci(S, bo), "ci1": _ci(S, bc), "ci_shift": _ci(S, bs)}
    if with_item:
        out["item_ci0"] = _ci(S, S.item_boot_means(vo, ctx.B, SEED) * 100)
        out["item_ci1"] = _ci(S, S.item_boot_means(vc, ctx.B, SEED) * 100)
        out["item_ci_shift"] = _ci(S, S.item_boot_means(vc - vo, ctx.B, SEED) * 100)
    if len(pairs) == 1:
        a, b = pairs[0]
        ao, bo_, ac, bc_ = ctx.exv(a, "orig", q), ctx.exv(b, "orig", q), ctx.exv(a, gk, q), ctx.exv(b, gk, q)
        out["disc_orig"] = {"first_only": int(((ao == 1) & (bo_ == 0)).sum()), "second_only": int(((ao == 0) & (bo_ == 1)).sum())}
        out["disc_corr"] = {"first_only": int(((ac == 1) & (bc_ == 0)).sum()), "second_only": int(((ac == 0) & (bc_ == 1)).sum())}
    return out


def level_row(ctx, label, gk, qids):
    import numpy as np
    S = ctx.S
    vo, vc = ctx.exv(label, "orig", qids), ctx.exv(label, gk, qids)
    bt = ctx.boot(qids)
    return {"n": len(qids), "ex0": float(vo.mean() * 100), "ex1": float(vc.mean() * 100), "shift": float((vc - vo).mean() * 100),
            "ci0": _ci(S, bt.means(vo) * 100), "ci1": _ci(S, bt.means(vc) * 100), "ci_shift": _ci(S, bt.means(vc - vo) * 100)}


def h4_route(ctx, gk, drop_db=None):
    """P3 readouts on the routing run, labels rescored. Differences route_ca - v3 (original, corrected, shift)."""
    import numpy as np
    from analysis_pipeline import routing
    S = ctx.S
    lab = "plus.route"
    q = [x for x in ctx.usable([lab]) if ctx.items[("final", lab)][x]["v3"] != "" and ctx.items[("final", lab)][x]["route_ca"] != ""]
    if drop_db:
        q = [x for x in q if x.split(":")[1] != drop_db]
    v3 = np.array([float(ctx.items[("final", lab)][x]["v3"]) for x in q])
    ca = np.array([float(ctx.items[("final", lab)][x]["route_ca"]) for x in q])
    yo, yc = ctx.exv(lab, "orig", q).astype(int), ctx.exv(lab, gk, q).astype(int)
    bt = ctx.boot(q)
    sc = {"route_ca": ca, "v3": v3}
    bso = routing.boot_metrics(sc, yo, bt)
    bsc = routing.boot_metrics(sc, yc, bt)
    pto = routing.point_metrics(sc, yo)
    ptc = routing.point_metrics(sc, yc)
    out = {"n": len(q), "ex0": float(yo.mean() * 100), "ex1": float(yc.mean() * 100)}
    for j, nm in ((0, "dauroc"), (1, "daurc"), (3, "dacc60")):
        d0 = float(pto["route_ca"][j] - pto["v3"][j])
        d1 = float(ptc["route_ca"][j] - ptc["v3"][j])
        bo = bso["route_ca"][:, j] - bso["v3"][:, j]
        bc = bsc["route_ca"][:, j] - bsc["v3"][:, j]
        out[nm] = {"d0": d0, "d1": d1, "shift": d1 - d0, "ci0": _ci(S, bo), "ci1": _ci(S, bc), "ci_shift": _ci(S, bc - bo)}
    out["auroc_route_ca"] = {"o": float(pto["route_ca"][0]), "c": float(ptc["route_ca"][0])}
    out["auroc_v3"] = {"o": float(pto["v3"][0]), "c": float(ptc["v3"][0])}
    return out


def e10_set(ctx, gk, drop_db=None):
    """revision effect of full_fixed (plus) on the set, original vs corrected gold (SAP 20.5 item 4a/4b)."""
    import numpy as np
    S = ctx.S
    q = []
    for x in ctx.qids:
        f, r = ctx.items[("final", ROOT_RUN)].get(x), ctx.items[("root", ROOT_RUN)].get(x)
        if f is None or r is None or f["usable"] != "1" or r["usable"] != "1":
            continue
        if f[gk] == "" or r[gk] == "" or f["orig"] == "" or r["orig"] == "":
            continue
        q.append(x)
    if drop_db:
        q = [x for x in q if x.split(":")[1] != drop_db]
    out = {"n": len(q)}
    vec = {}
    for g in ("orig", gk):
        cnt = collections.Counter()
        v_exec, v_rep = [], []
        for x in q:
            f, r = ctx.items[("final", ROOT_RUN)][x], ctx.items[("root", ROOT_RUN)][x]
            changed = f["sha"] != r["sha"]
            rexec = r["root_status"] in EXEC_STATUS
            rc = revision_class(int(float(r[g])), int(float(f[g])), rexec, changed)
            if rc:
                cnt[rc] += 1
            delta = int(float(f[g])) - int(float(r[g]))
            v_exec.append(delta if (changed and rexec) else 0)
            v_rep.append(delta if (changed and not rexec) else 0)
        vec[g] = (np.array(v_exec, float), np.array(v_rep, float))
        out["counts_" + ("orig" if g == "orig" else "corr")] = dict(cnt)
        h, u = cnt.get("helped", 0), cnt.get("hurt", 0)
        out["sign_test_p_" + ("orig" if g == "orig" else "corr")] = float(S.mcnemar_exact(h, u)["p_two"]) if (h + u) else None
    bt = ctx.boot(q) if len(q) >= 20 else None
    for nm, idx in (("net_exec", 0), ("net_repair", 1)):
        vo, vc = vec["orig"][idx], vec[gk][idx]
        d = {"d0": float(vo.mean() * 100), "d1": float(vc.mean() * 100), "shift": float((vc - vo).mean() * 100)}
        if bt is not None:
            bo, bc, bs = bt.means(vo) * 100, bt.means(vc) * 100, bt.means(vc - vo) * 100
            d.update({"ci0": _ci(S, bo), "ci1": _ci(S, bc), "ci_shift": _ci(S, bs)})
        out[nm] = d
    return out


def pair_audit(ctx, a, b, gk, eff):
    """SAP 20.10 (descriptive): for the pair (a, b) every item whose paired outcome (a right?, b right?) differs between original and
    corrected gold, counted by transition, by effective gold change and by whether the two final SQL texts differ (a revision)."""
    q = ctx.usable([a, b])
    cnt = collections.Counter()
    tot = collections.Counter()
    for x in q:
        ra, rb = ctx.items[("final", a)][x], ctx.items[("final", b)][x]
        so = (int(float(ra["orig"])), int(float(rb["orig"])))
        sc = (int(float(ra[gk])), int(float(rb[gk])))
        revised = ra["sha"] != rb["sha"]
        tot["revised" if revised else "same_sql"] += 1
        if so != sc:
            cnt["%d%d->%d%d|eff=%d|%s" % (so[0], so[1], sc[0], sc[1], int(bool(eff.get(x))), "differing_sql" if revised else "same_sql")] += 1
    only_b_orig = sum(1 for x in q if int(float(ctx.items[("final", a)][x]["orig"])) == 0 and int(float(ctx.items[("final", b)][x]["orig"])) == 1)
    only_b_corr = sum(1 for x in q if int(float(ctx.items[("final", a)][x][gk])) == 0 and int(float(ctx.items[("final", b)][x][gk])) == 1)
    return {"n": len(q), "pairs_with_differing_sql": tot["revised"], "transitions": dict(cnt),
            "second_only_right_orig": only_b_orig, "second_only_right_corr": only_b_corr}


def case_list(flips_rows, per_cell=5):
    """SAP 20.5 item 5: fixed-rule cases. flips_rows: lists as written to flips.csv (variant first). For G1 and the runs
    plus.gated, plus.fixed, flash.zs, flash.gated: per direction the first `per_cell` flips by ascending numeric question id, S_same items first."""
    out = {}
    for run in ("plus.gated", "plus.fixed", "flash.zs", "flash.gated"):
        for dirn in ("gain", "loss"):
            rows = [r for r in flips_rows if r[0] == "G1" and r[1] == run and r[4] == dirn]
            rows.sort(key=lambda r: (0 if r[5] else 1, int(r[2].split(":")[2])))
            out["%s|%s" % (run, dirn)] = [{"qid": r[2], "db": r[3], "in_S_same": bool(r[5]), "gold_sql_text_changed": bool(r[6]), "effective_gold_change": bool(r[7]),
                                           "gold_rows": "%s->%s" % (r[8], r[9]), "gold_cols": "%s->%s" % (r[10], r[11]), "keywords_in_exactly_one_gold": r[12]}
                                          for r in rows[:per_cell]]
            out["%s|%s" % (run, dirn) + "|n_total"] = len(rows)
    return out


def stage_analyze(a):
    import numpy as np
    sys.path.insert(0, CODE_DIR)
    from analysis_pipeline import stats as S
    t0 = time.time()
    items = load_items(os.path.join(a.out, "items_ex.csv.gz"))
    gold_info = jload(os.path.join(a.out, "gold_info.json"))
    dev_by_id, var = load_variants(a.data)
    B = a.B
    result = {"_meta": {"scope": "robustness-posthoc", "sap": "section 20 (Amendment 7)", "B": B, "seed": SEED, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "primary": "G1 / S_same", "stats_sha256": sha256_file(os.path.join(CODE_DIR, "analysis_pipeline", "stats.py")),
                        "routing_sha256": sha256_file(os.path.join(CODE_DIR, "analysis_pipeline", "routing.py")),
                        "git_rescore_sha": sha256_file(os.path.abspath(__file__))[:16]},
              "variants": {}}
    flips_rows = []
    headline_by_set = {}
    for gk in ["G1", "G2", "G3", "G4"]:
        discord = set()
        eff, textchg = {}, {}
        for key, rec in gold_info.items():
            g, q = key.split("|", 1)
            if g != gk:
                continue
            d, m = rec["dev"], rec["mini"]
            if same_result(d, m) is False:
                discord.add(q)
            o = gold_info.get("orig|" + q, {}).get("dev")
            eff[q] = (same_result(o, d) is False)
            textchg[q] = rec["sql_text_changed"]
        st = build_sets(dev_by_id, var[gk], discord)
        vres = {"n_db_discordant": len(discord), "db_discordant": sorted(discord), "sets": {}}
        # gold change counts
        for sn, qs in (("S_all", st["all"]), ("S_same", st["same"])):
            vres["sets"][sn] = {"n": len(qs), "sql_text_changed": sum(1 for q in qs if textchg[q]), "effective_gold_change": sum(1 for q in qs if eff[q]),
                                "text_changed_but_same_result": sum(1 for q in qs if textchg[q] and not eff[q]),
                                "result_changed_without_text_change": sum(1 for q in qs if eff[q] and not textchg[q]),
                                "gold_fail_orig": sum(1 for q in qs if not gold_info["orig|" + q]["dev"]["ok"]),
                                "gold_fail_corr": sum(1 for q in qs if not gold_info["%s|%s" % (gk, q)]["dev"]["ok"])}
        # validation gate (stored vs re-executed original EX)
        gate = {}
        for label in RUNS:
            agree = tot = 0
            bad = []
            for q in st["all"]:
                r = items[("final", label)].get(q)
                if r is None or r["usable"] != "1" or r["ex_stored"] == "":
                    continue
                tot += 1
                if r["orig"] != "" and int(float(r["orig"])) == int(float(r["ex_stored"])):
                    agree += 1
                else:
                    bad.append(q)
            gate[label] = {"n": tot, "agree": agree, "rate": (agree / float(tot)) if tot else None, "passes_99": bool(tot and agree / float(tot) >= 0.99), "disagree_qids": bad[:30]}
        r_root = {"n": 0, "agree": 0}
        for q in st["all"]:
            r = items[("root", ROOT_RUN)].get(q)
            if r and r["ex_stored"] != "" and r["orig"] != "":
                r_root["n"] += 1
                r_root["agree"] += int(int(float(r["orig"])) == int(float(r["ex_stored"])))
        gate["plus.fixed.root"] = r_root
        vres["validation_gate"] = gate
        for sn, qs in (("S_same", st["same"]), ("S_all", st["all"])):
            ctx = Ctx(S, items, qs, B)
            cs = {}
            allq = ctx.usable(list(RUNS))
            cs["common_n"] = len(allq)
            # levels
            cs["levels"] = {lab: level_row(ctx, lab, gk, ctx.usable([lab])) for lab in RUNS}
            exo = [cs["levels"][l]["ex0"] for l in PLUS_SYSTEMS]
            exc = [cs["levels"][l]["ex1"] for l in PLUS_SYSTEMS]
            rev = 0
            for i in range(len(exo)):
                for j in range(i + 1, len(exo)):
                    if sign(exo[i] - exo[j]) * sign(exc[i] - exc[j]) < 0:
                        rev += 1
            cs["rank_plus"] = {"spearman": S.spearman(exo, exc), "reversed_pairs": rev, "n_pairs": len(exo) * (len(exo) - 1) // 2, "systems": PLUS_SYSTEMS}
            # contrasts
            defs = collections.OrderedDict()

            def add(cid, concl, kind, pairs, desc):
                defs[cid] = (concl, kind, pairs, desc)
            add("H1.plus", "C1", "N", [("plus.gated", "plus.fixed")], "gated - fixed, qwen3.7-plus")
            add("H1.flash", "C1", "N", [("flash.gated", "flash.fixed")], "gated - fixed, qwen3.7-flash")
            add("H1.q27b", "C1", "N", [("q27b.gated", "q27b.fixed")], "gated - fixed, qwen3.5-27b")
            add("H1.pooled3", "C1", "N", [("plus.gated", "plus.fixed"), ("flash.gated", "flash.fixed"), ("q27b.gated", "q27b.fixed")], "P1' pooled over three backbones")
            add("H2.flash", "C2", "S", [("flash.gated", "flash.zs")], "gated - zs, qwen3.7-flash")
            add("H2.q27b", "C2", "S", [("q27b.gated", "q27b.zs")], "gated - zs, qwen3.5-27b")
            add("H2.pooled_weak", "C2", "S", [("flash.gated", "flash.zs"), ("q27b.gated", "q27b.zs")], "W1 pooled over the two weak backbones")
            add("H2.plus_control", "C2", "N", [("plus.gated", "plus.zs")], "gated - zs, qwen3.7-plus (strong-backbone control)")
            for bb in ("plus", "flash", "q27b"):
                add("H2b.%s.gated_minus_rootonly" % bb, None, "S", [("%s.gated" % bb, "%s.rootonly" % bb)], "loop over root, %s" % bb)
                add("H2b.%s.rootonly_minus_zs" % bb, None, "S", [("%s.rootonly" % bb, "%s.zs" % bb)], "root-only pipeline - zs, %s" % bb)
            add("H3.fixed_minus_macsql", None, "S", [("plus.fixed", "plus.macsql")], "full_fixed - macsql (P2)")
            for bsl in ("macsql", "acsql", "sdebug", "bonvote2", "bonvote3", "zs"):
                add("H3.gated_minus_%s" % bsl, None, "S", [("plus.gated", "plus.%s" % bsl)], "full_gated - %s" % bsl)
            cres = collections.OrderedDict()
            for cid, (concl, kind, pairs, desc) in defs.items():
                r = paired_contrast(ctx, pairs, gk, with_item=(concl is not None))
                if "d0" in r:
                    r.update(classify(kind, r["d0"], r["ci0"], r["d1"], r["ci1"], r["shift"], r["ci_shift"], "pp"))
                r.update({"conclusion": concl, "type": kind, "desc": desc})
                if concl is not None:
                    rn = paired_contrast(ctx, pairs, gk, drop_db="thrombosis_prediction", with_item=False)
                    r["sens_no_thr"] = {k: rn.get(k) for k in ("n", "d0", "d1", "shift", "ci0", "ci1")}
                cres[cid] = r
            # per-db table for the two pooled headline contrasts
            for cid in ("H1.pooled3", "H2.pooled_weak"):
                pairs = defs[cid][2]
                labs = sorted(set(x for p in pairs for x in p))
                qq = ctx.usable(labs)
                rows = {}
                for db in sorted(set(x.split(":")[1] for x in qq)):
                    sub = [x for x in qq if x.split(":")[1] == db]
                    vo = np.mean([ctx.exv(x, "orig", sub) - ctx.exv(y, "orig", sub) for x, y in pairs], axis=0)
                    vc = np.mean([ctx.exv(x, gk, sub) - ctx.exv(y, gk, sub) for x, y in pairs], axis=0)
                    rows[db] = {"n": len(sub), "d0": float(vo.mean() * 100), "d1": float(vc.mean() * 100)}
                cres[cid]["by_db"] = rows
            cs["contrasts"] = cres
            cs["pair_audit"] = {bb: pair_audit(ctx, "%s.gated" % bb, "%s.fixed" % bb, gk, eff) for bb in ("plus", "flash", "q27b")}
            # H4 and H5
            try:
                h4 = h4_route(ctx, gk)
                for nm in ("dauroc", "daurc", "dacc60"):
                    h4[nm].update(classify("S", h4[nm]["d0"], h4[nm]["ci0"], h4[nm]["d1"], h4[nm]["ci1"], h4[nm]["shift"], h4[nm]["ci_shift"], "auc"))
                h4["dauroc"]["conclusion"], h4["dauroc"]["type"] = "C3", "S"
                h4["sens_no_thr"] = {k: v for k, v in h4_route(ctx, gk, "thrombosis_prediction").items() if k in ("n", "dauroc")}
                cs["H4.route"] = h4
            except Exception as e:
                cs["H4.route"] = {"error": "%s: %s" % (type(e).__name__, e)}
            try:
                e10 = e10_set(ctx, gk)
                ne = e10["net_exec"]
                if "ci0" in ne:
                    ne.update(classify("N", ne["d0"], ne["ci0"], ne["d1"], ne["ci1"], ne["shift"], ne["ci_shift"], "pp"))
                    ne["conclusion"], ne["type"] = "C4", "N"
                cs["H5.revision"] = e10
            except Exception as e:
                cs["H5.revision"] = {"error": "%s: %s" % (type(e).__name__, e)}
            vres["sets"][sn]["analysis"] = cs
            print("variant %s %s done %.0f s" % (gk, sn, time.time() - t0), flush=True)
        # flips (all variants; the CSV is written for every variant, keyed by variant)
        for label in RUNS:
            for q in st["all"]:
                r = items[("final", label)].get(q)
                if r is None or r["usable"] != "1" or r["orig"] == "" or r[gk] == "":
                    continue
                if int(float(r["orig"])) != int(float(r[gk])):
                    gi = gold_info["%s|%s" % (gk, q)]
                    go = gold_info["orig|" + q]["dev"]
                    flips_rows.append([gk, label, q, r["db"], "gain" if int(float(r[gk])) == 1 else "loss", int(q in set(st["same"])),
                                       int(gi["sql_text_changed"]), int(eff[q]), go["n_rows"], gi["dev"]["n_rows"], go["n_cols"], gi["dev"]["n_cols"], gi["kw_diff"]])
        result["variants"][gk] = vres
        headline_by_set[gk] = None
    # E10 L6 share (G1 and the others)
    result["cases_fixed_rule"] = case_list(flips_rows)
    result["e10_l6"] = l6_share(a, items, gold_info, dev_by_id, var, S)
    result["headline"] = headline(result)
    dump(result, os.path.join(a.out, "rescore.json"))
    with io.open(os.path.join(a.out, "flips.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "run", "qid", "db", "direction", "in_S_same", "gold_sql_text_changed", "effective_gold_change", "gold_rows_orig",
                    "gold_rows_corr", "gold_cols_orig", "gold_cols_corr", "keywords_in_exactly_one_gold"])
        for r in sorted(flips_rows, key=lambda r: (r[0], r[1], int(r[2].split(":")[2]))):
            w.writerow(r)
    print("analyze done %.0f s" % (time.time() - t0))


def l6_share(a, items, gold_info, dev_by_id, var, S):
    """SAP 20.5 item 4c: share of E10 items labelled L6 whose gold is changed."""
    import numpy as np
    key = {r["item_id"]: r for r in csv.DictReader(io.open(os.path.join(a.e10_dir, "annotation_key.csv"), encoding="utf-8"))}
    lab = {r["item_id"]: r["final_label"] for r in csv.DictReader(io.open(os.path.join(a.e10_dir, "labels_final_all.csv"), encoding="utf-8"))}
    sheet = [(i, key[i], lab[i]) for i in sorted(lab) if i in key and i.startswith("A")]
    out = {"n_sheet_main": len(sheet), "n_L6": sum(1 for _, _, l in sheet if l == "L6")}
    for gk in ["G1", "G2", "G3", "G4"]:
        discord = set()
        for k, rec in gold_info.items():
            g, q = k.split("|", 1)
            if g == gk and same_result(rec["dev"], rec["mini"]) is False:
                discord.add(q)
        st = build_sets(dev_by_id, var[gk], discord)
        sets = {"S_all": set(st["all"]), "S_same": set(st["same"])}
        vg = {}
        for sn, qs in sets.items():
            rows = []
            for iid, k, l in sheet:
                q = k["qid"]
                if q not in qs:
                    continue
                gi = gold_info["%s|%s" % (gk, q)]
                o = gold_info["orig|" + q]["dev"]
                rows.append({"item": iid, "qid": q, "label": l, "system": k["system"], "w": float(k["weight"]), "strat": k["system"] + "|" + k["stratum"],
                             "text": bool(gi["sql_text_changed"]), "eff": same_result(o, gi["dev"]) is False})
            d = {"n_in_set": len(rows)}
            for flag in ("text", "eff"):
                grp = {}
                for name, f in (("L6", lambda r: r["label"] == "L6"), ("non_L6", lambda r: r["label"] != "L6")):
                    sub = [r for r in rows if f(r)]
                    k_ = sum(1 for r in sub if r[flag])
                    n_ = len(sub)
                    grp[name] = {"n": n_, "k": k_, "share": (k_ / float(n_)) if n_ else None, "wilson": S.wilson(k_, n_) if n_ else None}
                nc = newcombe_diff(grp["L6"]["k"], grp["L6"]["n"], grp["non_L6"]["k"], grp["non_L6"]["n"], S.wilson)
                grp["diff_L6_minus_nonL6"] = {"diff": (grp["L6"]["share"] - grp["non_L6"]["share"]) if (grp["L6"]["n"] and grp["non_L6"]["n"]) else None, "newcombe": nc,
                                             "corroborated": bool(nc and nc[0] > 0)}
                # stratum-weighted share among L6 (domain estimate) with stratified bootstrap
                sub = [r for r in rows if r["label"] == "L6"]
                if sub:
                    wsum = sum(r["w"] for r in sub)
                    wk = sum(r["w"] for r in sub if r[flag])
                    rng = np.random.RandomState(SEED)
                    byst = collections.defaultdict(list)
                    for r in rows:
                        byst[r["strat"]].append(r)
                    bs = []
                    for _ in range(a.B):
                        num = den = 0.0
                        for st_, lst in byst.items():
                            idx = rng.randint(0, len(lst), size=len(lst))
                            for j in idx:
                                r = lst[j]
                                if r["label"] == "L6":
                                    den += r["w"]
                                    num += r["w"] * (1 if r[flag] else 0)
                        if den > 0:
                            bs.append(num / den)
                    grp["L6_weighted"] = {"share": wk / wsum, "ci": S.pctl_ci(bs)}
                d[flag] = grp
            d["by_system"] = {s: {"n": sum(1 for r in rows if r["system"] == s), "n_L6": sum(1 for r in rows if r["system"] == s and r["label"] == "L6"),
                                  "n_L6_eff": sum(1 for r in rows if r["system"] == s and r["label"] == "L6" and r["eff"])} for s in ("full", "zs")}
            d["by_label_eff"] = {l: {"n": sum(1 for r in rows if r["label"] == l), "eff": sum(1 for r in rows if r["label"] == l and r["eff"])}
                                 for l in sorted(set(r["label"] for r in rows))}
            d["items_L6_effective"] = [r["item"] for r in rows if r["label"] == "L6" and r["eff"]]
            vg[sn] = d
        out[gk] = vg
    return out


def headline(result):
    """SAP 20.7: conclusion-level verdicts for G1, S_same (primary) and S_all (wording sensitivity)."""
    out = {}
    g1 = result["variants"]["G1"]["sets"]
    for sn in ("S_same", "S_all"):
        cs = g1[sn]["analysis"]
        items = collections.OrderedDict()
        for cid, r in cs["contrasts"].items():
            if r.get("conclusion"):
                items[cid] = r
        h4 = cs.get("H4.route", {}).get("dauroc")
        if h4 and "robust" in h4:
            items["H4.dauroc"] = h4
        h5 = cs.get("H5.revision", {}).get("net_exec")
        if h5 and "robust" in h5:
            items["H5.net_exec"] = h5
        byc = collections.OrderedDict()
        for cid, r in items.items():
            c = r["conclusion"]
            byc.setdefault(c, []).append((cid, r))
        verdict = {}
        for c, lst in byc.items():
            verdict[c] = {"contrasts": {cid: {"robust": r["robust"], "stable": r["stable"]} for cid, r in lst},
                          "unchanged": all(r["robust"] for _, r in lst), "unchanged_and_stable": all(r["stable"] for _, r in lst),
                          "not_robust": [cid for cid, r in lst if not r["robust"]]}
        out[sn] = verdict
    for c in out["S_same"]:
        a_, b_ = out["S_same"][c], out["S_all"].get(c)
        a_["wording_sensitive"] = bool(b_ and any(out["S_same"][c]["contrasts"][k]["robust"] != b_["contrasts"].get(k, {}).get("robust") for k in a_["contrasts"]))
    return out


# ------------------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["sets", "exec", "analyze", "all"])
    root = os.environ.get("NL2SQL_ROOT", "")            # the shared data/runs tree (see docs/CONTRACT.md); or pass the three paths
    ap.add_argument("--data", default=os.environ.get("NL2SQL_DATA", os.path.join(root, "data")))
    ap.add_argument("--runs", default=os.path.join(root, "runs"))
    ap.add_argument("--work", default=os.path.join(root, "rescore_work"))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "robust"))
    ap.add_argument("--e10-dir", dest="e10_dir", default=os.path.join(REPO, "results", "e10"))
    ap.add_argument("--route-coef", default=os.path.join(REPO, "exp", "frozen", "route_coef_hr.json"))
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    a = ap.parse_args(argv)
    if a.stage in ("sets", "all"):
        stage_sets(a)
    if a.stage in ("exec", "all"):
        stage_exec(a)
    if a.stage in ("analyze", "all"):
        stage_analyze(a)


if __name__ == "__main__":
    main()
