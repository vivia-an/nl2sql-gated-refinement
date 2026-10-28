"""Figure data layer: one schema D (plain dict of JSON-safe blocks), two producers.

    reference(runs_root, repo)  -> D   built from the TUNING-SET run directories (BIRD train HT-300 / HR-1101 / E4 mini-stream) with the SAME analysis
                                      functions the confirmatory pipeline uses (secondary.summarize, e4.curves / eviction_diag, primary.route_scores,
                                      stats.ClusterBoot).  Written to figs/data/figdata_reference.json together with the source of every block.
    final(vals)                 -> D   built from the flat keys of results/*.json (scope confirmatory; see make_all.load).  Blocks whose keys are absent
                                      are None (the figure is then skipped with a message, never drawn from placeholders).

Blocks that exist only as tuning-set evidence in BOTH modes (quote-fix before/after, PPO / arbitration / M3-force / value-grounding bounds)
are carried by `tuning_only` blocks; they are read from figs/data/figdata_reference.json in final mode and stay stamped "tuning set".

Block schemas (all numbers in the units of the axis label: EX in %, deltas in pp, cost in CNY per question):
  cost      {"split", "cells":[{"model","system","ex","lo","hi","calls","cost","n"}], "deltas":[{"model","contrast","m","lo","hi"}], "notes":[...], "src":[...]}
  repair    {"calls":[{"group","system","m1","gen","crit","rev"}], "rootfinal":[{"group","system","root","final","delta","dlo","dhi","n"}], "src":[...]}
  repair_tuning (tuning_only) {"errrate":[{"group","arm","k","n","rate","lo","hi"}], "src":[...]}
  rc        {"panels":[{"name","n","curves":{"v3":{"cov","risk","bcov","lo","hi"},"route_ca":{...}},"auroc":{"v3":[m,lo,hi],"route_ca":[...]},"dauroc":[m,lo,hi],
                        "acc60":{"v3":x,"route_ca":x}}], "src":[...]}
  pollution {"p":40,"policies":[{"policy","ex","lo","hi","n","ex_p0"?}], "evict":{"policy","recall":[m,lo,hi],"precision":[...],"fpr":[...]}, "src":[...]}
  negative  (tuning_only) {"ppo":{...},"arb":{...},"forest":[...],"value":{...},"src":[...]}
  perdb     {"cmp","split","rows":[{"db","n","delta","lo","hi","loo"}], "overall":[m,lo,hi]|None, "src":[...]}
  case      (tuning_only) {"qid","db","question","evidence","root_sql","error","final_sql","final_result","gold_sql","calls","rule","src"}   worked example of Figure 1
  errors    {"split","auto":[{"system","n_fail","n","classes":{class: count}}], "llm":None | {"systems":[{"system","n","labels":{Lk:[est,lo,hi,count]}}]}, "src":[...]}
"""
import glob
import json
import os

import numpy as np

from .. import e4 as E4
from .. import primary
from .. import runs as R
from .. import secondary as SEC
from .. import stats as S

MODELS = [("qwen37flash", "Qwen3.7-Flash"), ("qwen3527b", "Qwen3.5-27B"), ("qwen37plus", "Qwen3.7-Plus")]
MODEL_LABEL = dict(MODELS)
B_BOOT = 10000


class RunDir(object):
    """Minimal stand-in for runs.Run (what secondary.summarize / run_cost need): .rows, .path, .usage_files()."""

    def __init__(self, path):
        if not os.path.exists(os.path.join(path, "preds.jsonl")):
            sub = sorted(d for d in glob.glob(os.path.join(path, "*")) if os.path.exists(os.path.join(d, "preds.jsonl")))
            if not sub:
                raise IOError("no preds.jsonl below %s" % path)
            path = sub[0]
        self.path = path
        self.rows = {r["qid"]: r for r in R.jl(os.path.join(path, "preds.jsonl"))}
        self.run_id = os.path.basename(path)

    def usage_files(self):
        return sorted(glob.glob(os.path.join(self.path, "usage*.jsonl")))

    def roots(self):
        p = os.path.join(self.path, "cands.jsonl")
        return {c["qid"]: c for c in R.jl(p) if c.get("action") == "root"} if os.path.exists(p) else {}


def _ci3(t):
    return [float(t[0]), float(t[1]), float(t[2])]


def _paired(a, b, B=B_BOOT):
    """a, b: RunDir; -> (delta_pp, lo, hi, n) of EX(a) - EX(b) on common items, db-clustered percentile bootstrap (the pair_block rule)."""
    A = R.align({"a": a, "b": b})
    ea, eb, dbs = A["ex"]["a"], A["ex"]["b"], A["dbs"]
    boot = S.ClusterBoot(dbs, B)
    bt = 100.0 * (boot.means(ea) - boot.means(eb))
    lo, hi = S.pctl_ci(bt)
    return float(100.0 * (ea - eb).mean()), float(lo), float(hi), int(len(ea))


# ================================================================================================================ reference producer
def reference(runs, repo, prices_path=None):
    """runs = $R/runs on the worker machine (or a copy with the same layout).  -> D"""
    prices = SEC.load_prices(prices_path or os.path.join(repo, "exp", "prices.yaml"))
    D = {"_meta": {"mode": "reference", "scope": "tuning-set", "runs_root": runs, "note": "BIRD train HT-300 (4 dbs) / HR-1101 (8 dbs) / E4 mini-stream; design evidence, not results"}}

    def rd(rel):
        return RunDir(os.path.join(runs, rel))

    def summ(run, B=B_BOOT):
        return SEC.summarize(run, B, prices)

    # ---------------------------------------------------------------- HT-300: three backbones x {zs, fixed, gated}
    ht = {"qwen37flash": {"zs": "X8/qwen37flash-zs", "full_fixed": "X8/qwen37flash-fixed", "full_gated": "X8/qwen37flash-gated"},
          "qwen3527b": {"zs": "X8/qwen3527b-zs", "full_fixed": "X8/qwen3527b-fixed", "full_gated": "X8/qwen3527b-gated"},
          "qwen37plus": {"zs": "P2_base/P2_base-zs_ours-qwen37plus-s0-80485cc1", "full_fixed": "X5/1b-ht-quote", "full_gated": "X5/1b-ht-gated"}}
    HT = {m: {s: rd(p) for s, p in d.items()} for m, d in ht.items()}
    SU = {m: {s: summ(r) for s, r in d.items()} for m, d in HT.items()}
    cells, deltas, src = [], [], []
    for m, _ in MODELS:
        for s in ("zs", "full_fixed", "full_gated"):
            u = SU[m][s]
            cells.append({"model": m, "system": s, "ex": 100 * u["ex"][0], "lo": 100 * u["ex"][1], "hi": 100 * u["ex"][2], "calls": u["calls"],
                          "cost": u.get("cost_per_q"), "n": u["n"], "calls_stage": {k: u["calls_" + k] for k in SEC.STAGES}})
            src.append("%s -> %s/summary.json (recomputed by secondary.summarize from preds.jsonl, usage.jsonl)" % ((m, s), ht[m][s]))
        for a, b, nm in (("full_gated", "zs", "gated_vs_zs"), ("full_gated", "full_fixed", "gated_vs_fixed")):
            d, lo, hi, n = _paired(HT[m][a], HT[m][b])
            deltas.append({"model": m, "contrast": nm, "m": d, "lo": lo, "hi": hi, "n": n})
    D["cost"] = {"split": "HT-300 (BIRD train, 4 databases)", "cells": cells, "deltas": deltas, "src": src,
                 "notes": ["Qwen3.7-Plus zero-shot = P2_base zs_ours (legacy schema renderer, run before the identifier-quoting fix); Flash and 27B zero-shot ran with the fixed renderer.",
                           "cost = chat tokens x list price (exp/prices.yaml), embedding and rerank excluded."]}
    # ---------------------------------------------------------------- F3: calls by stage, root -> final, exec-error rate
    HR = {"full_fixed_legacy": rd("P2_HR"), "full_fixed": rd("X5/1b-hr-quote"), "full_gated": rd("X5/1b-hr-gated")}
    HTP = {"full_fixed_legacy": rd("P2_base/P2_base-full_fixed-qwen37plus-s0-d37d2097"), "full_fixed": HT["qwen37plus"]["full_fixed"], "full_gated": HT["qwen37plus"]["full_gated"]}
    SHR = {k: summ(v, 2000) for k, v in HR.items()}
    calls = []
    for grp, key, getter in (("HR-1101 Plus", "qwen37plus", lambda s: SHR[s]), ("HT-300 Flash", "qwen37flash", lambda s: SU["qwen37flash"][s]),
                             ("HT-300 27B", "qwen3527b", lambda s: SU["qwen3527b"][s]), ("HT-300 Plus", "qwen37plus", lambda s: SU["qwen37plus"][s])):
        for s in ("full_fixed", "full_gated"):
            u = getter(s)
            calls.append({"group": grp, "system": s, "m1": u["calls_m1"], "gen": u["calls_gen"], "crit": u["calls_crit"], "rev": u["calls_rev"], "calls": u["calls"], "n": u["n"]})
    rootfinal = []
    for grp, rr in (("HR-1101", HR), ("HT-300", HTP)):
        for s in ("full_fixed_legacy", "full_fixed", "full_gated"):
            run = rr[s]
            roots = run.roots()
            qs = sorted(q for q in run.rows if q in roots and R.ex_value(run.rows[q]) is not None and roots[q].get("ex") is not None)
            fin = np.array([R.ex_value(run.rows[q]) for q in qs], float)
            rt = np.array([int(roots[q]["ex"]) for q in qs], float)
            dbs = np.array([run.rows[q].get("db_id") for q in qs])
            boot = S.ClusterBoot(dbs, B_BOOT)
            lo, hi = S.pctl_ci(100.0 * (boot.means(fin) - boot.means(rt)))
            rootfinal.append({"group": grp, "system": s, "root": 100 * float(rt.mean()), "final": 100 * float(fin.mean()), "delta": 100 * float((fin - rt).mean()),
                              "dlo": float(lo), "dhi": float(hi), "n": len(qs), "gain": int(((fin == 1) & (rt == 0)).sum()), "harm": int(((fin == 0) & (rt == 1)).sum())})
    D["repair"] = {"calls": calls, "rootfinal": rootfinal,
                   "src": ["calls: secondary.summarize stages (m1, gen, crit, rev) of X8/*, X5/1b-{hr,ht}-{quote,gated}",
                           "root EX: cands.jsonl action=root ex; final EX: preds.jsonl ex; CI of final-root: db-clustered percentile bootstrap B=10000 seed 20261101",
                           "legacy arm: P2_HR, P2_base full_fixed (schema renderer before the identifier-quoting fix)"]}
    # quote fix: root execution-failure rate, legacy vs quoted (beer_factory subset of HR, HR, HT)
    err = []
    beer_q = RunDir(os.path.join(runs, "X5/1a-beer-quote"))
    for grp, leg, quo, subset in (("beer_factory (HR subset)", HR["full_fixed_legacy"], beer_q, set(beer_q.rows)), ("HR-1101", HR["full_fixed_legacy"], HR["full_fixed"], None),
                                  ("HT-300", HTP["full_fixed_legacy"], HTP["full_fixed"], None)):
        for arm, run in (("legacy", leg), ("quoted", quo)):
            roots = run.roots()
            qs = [q for q in roots if (subset is None or q in subset)]
            bad = [q for q in qs if str(roots[q].get("status", "")).startswith("err") or roots[q].get("status") == "timeout"]
            lo, hi = S.wilson(len(bad), len(qs))
            err.append({"group": grp, "arm": arm, "k": len(bad), "n": len(qs), "rate": 100.0 * len(bad) / len(qs), "lo": 100 * lo, "hi": 100 * hi})
    D["repair_tuning"] = {"errrate": err, "tuning_only": True, "src": ["cands.jsonl root status in {err_syntax, err_identifier, timeout}; Wilson 95 % (item level)",
                                                                       "X5/1a-beer-quote vs P2_HR on the same 131 beer_factory questions"]}
    # ---------------------------------------------------------------- F4: risk-coverage (X5 2-hr-cons LODO, 2-ht-cons held-out with the frozen HR coefficients)
    D["rc"] = _rc_reference(runs, repo)
    # ---------------------------------------------------------------- F5: pollution
    D["pollution"] = _pollution_reference(runs)
    # ---------------------------------------------------------------- F7: per-database delta gated - fixed on HR
    a, b = HR["full_gated"], HR["full_fixed"]
    A = R.align({"a": a, "b": b})
    d = (A["ex"]["a"] - A["ex"]["b"]).astype(float)
    boot = S.ClusterBoot(A["dbs"], B_BOOT)
    rows = []
    loo = dict(boot.loo(lambda idx: 100.0 * d[idx].mean()))
    for j, db in enumerate(boot.u):
        idx = boot.members[j]
        lo, hi = S.pctl_ci(100.0 * S.item_boot_means(d[idx], 2000))
        rows.append({"db": db, "n": int(len(idx)), "delta": float(100.0 * d[idx].mean()), "lo": float(lo), "hi": float(hi), "loo": loo.get(db)})
    olo, ohi = S.pctl_ci(100.0 * (boot.means(A["ex"]["a"]) - boot.means(A["ex"]["b"])))
    D["perdb"] = {"cmp": "gated minus fixed", "split": "HR-1101 (BIRD train, 8 databases)", "rows": rows, "overall": [float(100.0 * d.mean()), float(olo), float(ohi)],
                  "src": ["X5/1b-hr-gated minus X5/1b-hr-quote (same questions); per-db CI: item-level paired bootstrap 2000 draws within db (secondary.run_e1 rule); pooled CI: db-clustered bootstrap B=10000",
                          "leave-one-db-out value: pooled delta without that database"]}
    D["negative"] = negative_tuning(repo)
    D["case"] = _case_reference(runs)
    D["errors"] = {"split": "HT-300 (BIRD train, 4 databases), Qwen3.7-Plus",
                   "auto": [_auto_row(lab, HT["qwen37plus"][lab]) for lab in ("zs", "full_fixed", "full_gated")],
                   "llm": None,
                   "src": ["preds.jsonl auto_class of every item with ex = 0 (same rule as e10.auto_class_tables); zs = P2_base zs_ours (legacy renderer), fixed = X5/1b-ht-quote, gated = X5/1b-ht-gated",
                           "llm: E10 LLM labels (L1-L8) do not exist yet; panel stays empty until results/E10.json carries E10.llm.*"]}
    return D


def _auto_row(label, run):
    fails = [r for r in run.rows.values() if R.ex_value(r) == 0]
    cnt = {}
    for r in fails:
        c = r.get("auto_class") or r.get("error_class") or "other"
        cnt[c] = cnt.get(c, 0) + 1
    return {"system": label, "n_fail": len(fails), "n": len(run.rows), "classes": cnt}


CASE_RUN, CASE_QID, CASE_MODEL = "X8/qwen37flash-gated", "bird_train:law_episode:1294", "Qwen3.7-Flash"


def _case_reference(runs, run_rel=CASE_RUN, qid=CASE_QID, model=CASE_MODEL):
    """Worked example of Figure 1: the same question as the introduction table (law_episode item 1294, HT-300, gated loop, Qwen3.7-Flash), so that one running example
    carries through the paper.  It is checked against the rule that makes it a repair case: the root failed with an identifier error, exactly one revision followed, and
    the final answer has EX = 1.  The SQLite message is obtained by re-executing the root query read-only on the BIRD train database (the pipeline passes the same text
    to the Actor)."""
    import sqlite3
    data_root = os.path.join(os.path.dirname(runs.rstrip("/")), "data", "bird", "train", "train")
    run = RunDir(os.path.join(runs, run_rel))
    cs = [c for c in R.jl(os.path.join(run.path, "cands.jsonl")) if c["qid"] == qid]
    root = [c for c in cs if c.get("action") == "root"][0]
    row = run.rows[qid]
    assert root.get("status") == "err_identifier" and R.ex_value(row) == 1 and len(cs) == 2 and cs[1].get("action") == "revise", "case no longer meets the repair rule"
    rev = cs[1]
    train = json.load(open(os.path.join(data_root, "train.json")))
    it = train[int(qid.split(":")[-1])]
    db = qid.split(":")[1]
    assert it["db_id"] == db, (it["db_id"], db)
    con = sqlite3.connect("file:%s?mode=ro" % os.path.join(data_root, "train_databases", db, db + ".sqlite"), uri=True)
    try:
        con.execute(root["sql"]).fetchmany(1)
        err = None
    except Exception as e:                                                      # the SQLite message is data here
        err = str(e)
    res = con.execute(rev["sql"]).fetchmany(3)
    gold = con.execute(it["SQL"]).fetchmany(3)
    assert res == gold and row["final_sql"].strip() == rev["sql"].strip()
    return {"tuning_only": True, "qid": qid, "db": db, "model": model, "question": it["question"], "evidence": it["evidence"], "root_sql": root["sql"], "error": err,
            "final_sql": rev["sql"], "final_result": [list(r) for r in res], "gold_sql": it["SQL"], "calls": row["calls"],
            "rule": "root status err_identifier, one revision, final EX = 1 (checked); same item as the introduction table",
            "src": ["runs/%s cands.jsonl + preds.jsonl; question, evidence and gold SQL from BIRD train.json; error text by read-only re-execution of the root SQL" % run_rel]}


def _rc_reference(runs, repo):
    sys_path_tools = os.path.join(repo, "code", "tools")
    import sys
    if sys_path_tools not in sys.path:
        sys.path.insert(0, sys_path_tools)
    import fit_route_score as F
    coef_path = os.path.join(runs, "X5", "route", "route_coef_hr.json")
    if not os.path.exists(coef_path):                                           # the fit json lives next to the run dirs on the host copy only
        coef_path = os.path.join(repo, "results", "raw-ignored", "X5", "x5-2-fit", "route_coef_hr.json")
    coef = json.load(open(coef_path))
    panels = []
    for name, rel, mode in (("HR-1101, cross-validated", "X5/2-hr-cons", "lodo"), ("HT-300 (held out)", "X5/2-ht-cons", "frozen")):
        run = RunDir(os.path.join(runs, rel))
        rows = [run.rows[q] for q in sorted(run.rows)]
        v3, _ = primary.route_scores(rows)
        y = np.array([R.ex_value(r) if R.ex_value(r) is not None else -1 for r in rows])
        dbs = np.array([r.get("db_id") for r in rows])
        if mode == "frozen":
            _, ca = primary.route_scores(rows, coef, None)
        else:
            X, yy, dd, qids, dropped = F.load_rows([os.path.join(run.path, "preds.jsonl")], coef["features"], None)
            pl = F.lodo_predict(X, yy, dd, 1.0)
            m = {q: 10.0 * p for q, p in zip(qids, pl)}
            ca = np.array([m.get(r["qid"], np.nan) for r in rows])
        ok = np.isfinite(v3) & np.isfinite(ca) & (y >= 0)
        v3, ca, y, dbs = v3[ok], ca[ok], y[ok], dbs[ok]
        sc = {"v3": v3, "route_ca": ca}
        boot = S.ClusterBoot(dbs, 1000)
        pan = {"name": name, "n": int(len(y)), "curves": {}, "auroc": {}, "acc60": {}, "aurc": {}}
        cov = np.arange(1, len(y) + 1) / float(len(y))
        for k, s in sc.items():
            cum = S._expected_cum_acc(s, y)
            step = max(1, len(y) // 200)
            sel = list(range(0, len(y), step)) + [len(y) - 1]
            pan["curves"][k] = {"cov": [float(cov[i]) for i in sel], "risk": [float(1 - cum[i]) for i in sel]}
            ab = boot.stat(lambda idx, s=s: S.auroc(s[idx], y[idx]), B=1000)
            lo, hi = S.pctl_ci(ab)
            pan["auroc"][k] = [float(S.auroc(s, y)), float(lo), float(hi)]
            pan["acc60"][k] = float(cum[max(1, int(round(0.6 * len(y)))) - 1])
            pan["aurc"][k] = float((1 - cum).mean())
        pts = 20
        bcov = np.linspace(1.0 / pts, 1.0, pts)
        B = 1000
        out = {k: np.empty((B, pts)) for k in sc}
        for b in range(B):
            idx = np.concatenate([boot.members[j] for j in boot.draws[b]])
            for k, s in sc.items():
                cum = S._expected_cum_acc(s[idx], y[idx])
                out[k][b] = [1.0 - cum[max(1, int(round(c * len(cum)))) - 1] for c in bcov]
        for k in sc:
            pan["curves"][k]["bcov"] = [float(c) for c in bcov]
            pan["curves"][k]["lo"] = [float(v) for v in np.percentile(out[k], 2.5, axis=0)]
            pan["curves"][k]["hi"] = [float(v) for v in np.percentile(out[k], 97.5, axis=0)]
        dd_ = boot.stat(lambda idx: S.auroc(ca[idx], y[idx]) - S.auroc(v3[idx], y[idx]), B=1000)
        lo, hi = S.pctl_ci(dd_)
        pan["dauroc"] = [float(S.auroc(ca, y) - S.auroc(v3, y)), float(lo), float(hi)]
        pan["src"] = "%s preds.jsonl; v3 = critic score of the final answer; route_ca = %s" % (rel, "LODO logistic on (v3, consistency, nonempty) via tools/fit_route_score.lodo_predict" if mode == "lodo" else "frozen coefficients results/raw-ignored/X5/x5-2-fit/route_coef_hr.json (primary.route_scores)")
        panels.append(pan)
    return {"panels": panels, "src": ["X5/2-hr-cons, X5/2-ht-cons (full_gated + 4 consistency samples), coefficients X5/2-fit/route_coef_hr.json (k = 4 samples)"]}


def _pollution_reference(runs):
    root = os.path.join(runs, "MIN", "E4")
    units, notes = E4.load_all([root])
    pols = [p for p in ("append", "memrl", "ws_full") if any(u["policy"] == p and u["p"] == 40 for u in units)]
    out = []
    for p in pols:
        cv = E4.curves(units, p, 40, B=2000)
        rows = E4.probe_rows(units, p, 40)
        e = {"policy": p, "ex": 100 * cv["y"][-1], "lo": 100 * cv["ci_lo"][-1], "hi": 100 * cv["ci_hi"][-1], "n": int(len(rows["y"])), "n_units": cv["n_units"],
             "ex_start": 100 * cv["y"][0]}
        cv0 = E4.curves(units, p, 0, B=2000)
        if cv0:
            e["ex_p0"] = 100 * cv0["y"][-1]
            e["n_p0"] = int(len(E4.probe_rows(units, p, 0)["y"]))
            e["n_units_p0"] = cv0["n_units"]
        out.append(e)
    ev = E4.eviction_diag(units, "ws_full", 40)
    # counts for Wilson intervals
    npol = ev["n_polluted_seed"]
    evn = ev["evicted_polluted"] + ev["evicted_clean"]
    fpr_den = 0
    for u in units:
        if u["policy"] == "ws_full" and u["p"] == 40 and u["probes"]:
            fpr_den += (u.get("n_seed") or 0) - (u.get("n_polluted") or 0) + sum(1 for e in u["stream"] if not e.get("abstained"))
    w = lambda k, n: [100.0 * k / n] + [100 * v for v in S.wilson(k, n)]
    evict = {"policy": "ws_full", "recall": w(ev["evicted_polluted"], npol), "precision": w(ev["evicted_polluted"], evn), "fpr": w(ev["evicted_clean"], fpr_den),
             "counts": {"evicted_polluted": ev["evicted_polluted"], "evicted_clean": ev["evicted_clean"], "n_polluted_seed": npol, "clean_entries_ever": fpr_den}}
    others = {}
    for p in ("memrl", "append"):
        e2 = E4.eviction_diag(units, p, 40)
        others[p] = (e2["evicted_polluted"] + e2["evicted_clean"]) if e2 else None
    evict["evicted_total_other_policies"] = others
    return {"p": 40, "policies": out, "evict": evict, "notes": notes,
            "src": ["runs/MIN/E4 units (events.jsonl + done): e4.curves (final probe, db-clustered bootstrap over 8 dbs, 2 stream permutations pooled), e4.eviction_diag (Wilson 95 % on counts)",
                    "p = 0 reference: probe at the end of the stream of the p00 units, where complete"]}


def negative_tuning(repo):
    """Tuning-only evidence (also used in final mode).  Every number read from a committed results file."""
    def J(rel):
        return json.load(open(os.path.join(repo, "results", rel)))
    e6 = J("p2/e6_summary.json")
    base = J("p2/e6_baselines.json")
    x5 = J("x5/x5_tuning_summary.json")
    vc = J("x7/value_check.json")
    er = J("x7/empty_roots.json")
    ad = J("x7/adaptive_replay.json")
    ppo = [{"c_h": c["c_h"], "lam": c["lam"], "ex": 100 * c["ex_all"], "sd": 100 * c["ex_all_sd_folds"], "calls": c["calls"], "algo": "PPO"} for c in e6["cv"]]
    a2c = [{"c_h": c["c_h"], "lam": c["lam"], "ex": 100 * c["ex_all"], "sd": 100 * c["ex_all_sd_folds"], "calls": c["calls"], "algo": "A2C"} for c in e6["cv_a2c"]]
    p0 = [{"name": k.replace("P0_theta", "θ="), "ex": 100 * v["ex_all"], "calls": v["calls"]} for k, v in base.items() if k.startswith("P0_theta")]
    fk = [{"name": k, "ex": 100 * v["ex_all"], "calls": v["calls"]} for k, v in base.items() if k.startswith("fixed_k")]

    def acc(a, n):
        k = int(round(a * n))
        lo, hi = S.wilson(k, n)
        return [100.0 * k / n, 100 * lo, 100 * hi, k, n]
    arb = {"pool": {"judge": acc(x5["TS.arb.pool.judge_acc"], x5["TS.arb.pool.n_mixed"]), "always": acc(x5["TS.arb.pool.revise_always_acc"], x5["TS.arb.pool.n_mixed"]),
                    "delta": [x5["TS.arb.pool.delta_ex.mean"], x5["TS.arb.pool.delta_ex.ci_lo"], x5["TS.arb.pool.delta_ex.ci_hi"]]},
           "revise": {"judge": acc(x5["TS.arb.revise.judge_acc"], x5["TS.arb.revise.n_mixed"]), "always": acc(x5["TS.arb.revise.revise_always_acc"], x5["TS.arb.revise.n_mixed"]),
                      "delta": [x5["TS.arb.revise.delta_ex.mean"], x5["TS.arb.revise.delta_ex.ci_lo"], x5["TS.arb.revise.delta_ex.ci_hi"]]}}

    def adapt(split):
        c = [c for c in ad[split]["configs"] if c["k"] == 4 and c["theta_fast"] == 1.0 and c["empty"] == "vote"][0]["vs_gated_same_roots"]
        return [c["diff_pp"], c["ci95_pp"][0], c["ci95_pp"][1], c["n"]]
    forest = [{"short": "Forced M3\nrouting (HT)", "label": "Forced M3 routing (HT-300)", "m": x5["TS.m3.forced.delta_ex.mean"], "lo": x5["TS.m3.forced.delta_ex.ci_lo"], "hi": x5["TS.m3.forced.delta_ex.ci_hi"], "n": x5["TS.m3.n"]},
              {"short": "Arbitration\n(HR trees)", "label": "Arbitration judge (HR trees)", "m": arb["pool"]["delta"][0], "lo": arb["pool"]["delta"][1], "hi": arb["pool"]["delta"][2], "n": x5["TS.arb.pool.n_mixed"]},
              {"short": "Routed compute\n(HR)", "label": "Confidence-routed compute (HR-1101)", "m": adapt("HR")[0], "lo": adapt("HR")[1], "hi": adapt("HR")[2], "n": adapt("HR")[3]},
              {"short": "Routed compute\n(HT)", "label": "Confidence-routed compute (HT-300)", "m": adapt("HT")[0], "lo": adapt("HT")[1], "hi": adapt("HT")[2], "n": adapt("HT")[3]}]
    g = vc["gated"]
    ub = g["upper_bound_value_grounding"]
    n = g["n_questions"]
    value = {"n_questions": n, "bars": [
        {"short": "Literal in the\nquestion text", "label": "Filter literal in question text (mis-copied, no lookup needed)", "pp": 100.0 * g["levels"]["verbatim_only"] / n, "k": g["levels"]["verbatim_only"]},
        {"short": "Literal needs a\nvalue lookup", "label": "Literal not in text, needs a value lookup (upper bound)", "pp": ub["pp_of_all_questions"], "k": ub["n"]},
        {"short": "...stored in\nits column", "label": "of which gold literal stored in its column", "pp": ub["retrievable_by_lookup_pp"], "k": ub["retrievable_by_lookup_n"]},
        {"short": "Empty result,\nprobe (HR)", "label": "Wrong empty result with literal visible in DISTINCT probe (HR)", "pp": er["GATED_HR"]["probe_visible_pp_of_all_questions"], "k": er["GATED_HR"]["probe_visible"]},
        {"short": "Empty result,\nprobe (HT)", "label": "same, HT-300", "pp": er["GATED_HT"]["probe_visible_pp_of_all_questions"], "k": er["GATED_HT"]["probe_visible"]}]}
    best = lambda L: max(L, key=lambda c: c["ex"])
    return {"tuning_only": True, "ppo": {"ppo": ppo, "a2c": a2c, "p0": p0, "fixed_k": fk, "p0_ref": {"ex": 100 * base["P0_theta8.5"]["ex_all"], "calls": base["P0_theta8.5"]["calls"]},
                                        "best_ppo": best(ppo), "best_a2c": best(a2c), "n": 1101},
            "arb": arb, "forest": forest, "value": value,
            "src": ["results/p2/e6_summary.json (cv, cv_a2c), results/p2/e6_baselines.json", "results/x5/x5_tuning_summary.json (TS.arb.*, TS.m3.*)",
                    "results/x7/value_check.json (gated, pooled HR+HT), results/x7/empty_roots.json (GATED_HR, GATED_HT), results/x7/adaptive_replay.json (k=4, theta_fast=1, vote)"]}


# ================================================================================================================ final producer
def final(vals, tuning_blocks=None, post=None):
    """vals = flat dict of results/*.json (scope-checked by make_all.load).  Absent keys -> block None."""
    V = vals.get
    D = {"_meta": {"mode": "final"}}
    # ---- F2 cost-accuracy plane: E2.<model>.<system>.* (E2 is written for every model with complete zs / fixed / gated r0 runs)
    cells, deltas = [], []
    for m, _ in MODELS:
        for s in ("zs", "full_fixed", "full_gated"):
            p = "E2.%s.%s" % (m, s)
            if V(p + ".ex.mean") is None:
                continue
            cells.append({"model": m, "system": s, "ex": 100 * V(p + ".ex.mean"), "lo": 100 * V(p + ".ex.ci_lo", V(p + ".ex.mean")), "hi": 100 * V(p + ".ex.ci_hi", V(p + ".ex.mean")),
                          "calls": V(p + ".calls.mean"), "cost": V(p + ".cost_per_q.mean"), "n": V(p + ".n"),
                          "calls_stage": {k: V(p + ".calls_%s.mean" % k) for k in SEC.STAGES}})
        for nm, key in (("gated_vs_zs", "E2.%s.gain_gated_vs_zs.delta_ex" % m), ("gated_vs_fixed", "P1.%s.delta_ex" % m)):
            if V(key + ".mean") is None and nm == "gated_vs_fixed" and m == "qwen37plus":
                key = "P1.delta_ex"                     # primary.py reports the primary contrast for the primary model only
            if V(key + ".mean") is not None:
                deltas.append({"model": m, "contrast": nm, "m": V(key + ".mean"), "lo": V(key + ".ci_lo"), "hi": V(key + ".ci_hi")})
    x1 = {"m": V("X1.flash_vs_plus_zs.delta_ex.mean"), "lo": V("X1.flash_vs_plus_zs.delta_ex.ci_lo"), "hi": V("X1.flash_vs_plus_zs.delta_ex.ci_hi"), "cost_ratio": V("X1.flash_vs_plus_zs.cost_ratio.mean", V("X1.flash_vs_plus_zs.cost_ratio"))} \
        if V("X1.flash_vs_plus_zs.delta_ex.mean") is not None else None
    D["cost"] = {"split": "BIRD dev (11 databases)", "cells": cells, "deltas": deltas, "x1": x1, "notes": [], "src": ["results/E2.json, results/X1.json"]} if len(cells) >= 2 else None
    # ---- F3
    calls, rootfinal = [], []
    for m, lab in MODELS:
        for s in ("full_fixed", "full_gated"):
            p = "E2.%s.%s" % (m, s)
            if V(p + ".calls.mean") is not None and V(p + ".calls_m1.mean") is not None:
                calls.append({"group": lab.replace("Qwen", "").replace("3.7-", "").replace("3.5-", ""), "system": s, **{k: V(p + ".calls_%s.mean" % k) for k in SEC.STAGES}, "calls": V(p + ".calls.mean")})
    for s in ("full_fixed", "full_gated"):
        tot = sum(V("E1.rootstatus.%s.%s.n" % (s, st), 0) for st in ("err", "empty", "nonempty"))
        if tot:
            root = sum(V("E1.rootstatus.%s.%s.n" % (s, st), 0) * V("E1.rootstatus.%s.%s.ex_root" % (s, st), 0) for st in ("err", "empty", "nonempty")) / tot
            fin = sum(V("E1.rootstatus.%s.%s.n" % (s, st), 0) * V("E1.rootstatus.%s.%s.ex_final" % (s, st), 0) for st in ("err", "empty", "nonempty")) / tot
            rootfinal.append({"group": "dev", "system": s, "root": 100 * root, "final": 100 * fin, "delta": 100 * (fin - root), "dlo": None, "dhi": None, "n": tot})
    # database-clustered 95% interval of "final minus root" (the loop effect inside the run): results/posthoc2 (RV2, post hoc), Qwen3.7-Plus; used only if its estimate equals the one computed here
    for r in rootfinal:
        k = "loop.plus.eff_in.%s" % ("fixed" if r["system"] == "full_fixed" else "gated")
        e = (post or {}).get(k)
        if e and e.get("ci") and abs(e["v"] - r["delta"]) < 0.01:
            r["dlo"], r["dhi"] = e["ci"]
        elif e:
            print("[figures] WARNING: %s (%.3f) differs from final minus root of the figure (%.3f); interval not drawn" % (k, e["v"], r["delta"]))
    D["repair"] = {"calls": calls, "rootfinal": rootfinal, "src": ["results/E2.json, results/E1.json" + (", results/posthoc2/rv2_summary.json (loop.plus.eff_in.*, interval)" if post else "")]} if (calls or rootfinal) else None
    D["repair_tuning"] = (tuning_blocks or {}).get("repair_tuning")
    # ---- F4: E7 curves on the routing run
    panels = []
    if V("E7.curve.v3.coverage") is not None and V("E7.curve.route_ca.coverage") is not None:
        pan = {"name": "Development set", "n": V("E7.n", V("setup.bird_dev.n_questions")), "curves": {}, "auroc": {}, "acc60": {}, "aurc": {}}
        for s in ("v3", "route_ca"):
            c = {"cov": V("E7.curve.%s.coverage" % s), "risk": V("E7.curve.%s.risk" % s)}
            if V("E7.band.%s.risk_lo" % s) is not None:
                c.update(bcov=V("E7.band.%s.coverage" % s), lo=V("E7.band.%s.risk_lo" % s), hi=V("E7.band.%s.risk_hi" % s))
            pan["curves"][s] = c
            a = V("E7.%s.auroc.mean" % s)
            if a is not None:
                pan["auroc"][s] = [a, V("E7.%s.auroc.ci_lo" % s, a), V("E7.%s.auroc.ci_hi" % s, a)]
            if V("E7.%s.acc_ex_cov60.mean" % s) is not None:
                pan["acc60"][s] = V("E7.%s.acc_ex_cov60.mean" % s)
        if V("P3.dauroc.mean") is not None:
            pan["dauroc"] = [V("P3.dauroc.mean"), V("P3.dauroc.ci_lo"), V("P3.dauroc.ci_hi")]
        panels.append(pan)
    D["rc"] = {"panels": panels, "src": ["results/E7.json, results/primary.json"]} if panels else None
    # ---- F5
    pols = []
    for p in ("append", "memrl", "ws_full", "ws_evict", "static"):
        if V("E4.%s.p40.final_ex.mean" % p) is not None:
            e = {"policy": p, "ex": 100 * V("E4.%s.p40.final_ex.mean" % p), "lo": 100 * V("E4.%s.p40.final_ex.ci_lo" % p), "hi": 100 * V("E4.%s.p40.final_ex.ci_hi" % p), "n": None,
                 "n_units": V("E4.%s.p40.n_units" % p), "ex_start": 100 * V("E4.%s.p40.init_ex.mean" % p, 0)}
            if V("E4.%s.p0.final_ex.mean" % p) is not None:
                e["ex_p0"] = 100 * V("E4.%s.p0.final_ex.mean" % p)
            pols.append(e)
    ev = None
    if V("E4.evict.ws_full.p40.recall.mean") is not None:
        ev = {"policy": "ws_full"}
        for k in ("recall", "precision", "fpr"):
            m = V("E4.evict.ws_full.p40.%s.mean" % k)
            ev[k] = [100 * m, None, None] if m is not None else None
    D["pollution"] = {"p": 40, "policies": pols, "evict": ev, "notes": [], "src": ["results/E4.json"]} if pols else None
    # ---- F7
    pre = None
    for k in sorted(vals):
        if k.startswith("E1.perdb_delta.full_gated_vs_full_fixed.") and k.endswith(".overall_pp"):
            pre = k[:-len(".overall_pp")]
    if pre:
        dbs = sorted({k[len(pre) + 1:].split(".")[0] for k in vals if k.startswith(pre + ".") and k.endswith(".delta_pp")})
        rows = [{"db": d, "n": V("%s.%s.n" % (pre, d), 0), "delta": V("%s.%s.delta_pp" % (pre, d)), "lo": V("%s.%s.ci_lo" % (pre, d)), "hi": V("%s.%s.ci_hi" % (pre, d)),
                 "loo": V("%s.%s.loo_delta_pp" % (pre, d))} for d in dbs]
        ov = V("P1.delta_ex.mean")
        D["perdb"] = {"cmp": "gated minus fixed", "split": "development set (%d databases)" % len(dbs), "rows": rows,
                      "overall": [ov, V("P1.delta_ex.ci_lo"), V("P1.delta_ex.ci_hi")] if ov is not None else [V(pre + ".overall_pp"), None, None], "src": ["results/E1.json, results/primary.json"]}
    else:
        D["perdb"] = None
    D["negative"] = (tuning_blocks or {}).get("negative")
    D["case"] = (tuning_blocks or {}).get("case")
    D["errors"] = _errors_final(vals)
    return D


LLM_LABELS = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]


def _errors_final(vals):
    """F8 from results/E10.json: automatic classes of failed answers (E10.auto.<system>.<class>.n) and, when the LLM labels exist, L1-L8 prevalence
    (E10.llm.<system>.<Lk>.prevalence.{mean,ci_lo,ci_hi}).  A block with no automatic classes at all returns None (the figure is skipped)."""
    V = vals.get
    auto = []
    for sysn in ("zs", "full_fixed", "full_gated"):
        pre = "E10.auto.%s." % sysn
        nf = V(pre + "n_fail")
        if nf is None:
            continue
        cls = {}
        for k in vals:
            if k.startswith(pre) and k.endswith(".n") and k != pre + "n_fail":
                cls[k[len(pre):-2]] = vals[k]
        auto.append({"system": sysn, "n_fail": nf, "n": None, "classes": cls})
    llm = None
    systems = []
    for sysn in ("zs", "full", "full_fixed", "full_gated"):                  # the annotation sheet calls the fixed loop `full` (failures of full_fixed, see internal note E10)
        if V("E10.llm.%s.n" % sysn) is None:
            continue
        labs = {}
        for l in LLM_LABELS:
            m = V("E10.llm.%s.%s.prevalence.mean" % (sysn, l))
            if m is not None:
                labs[l] = [100 * m, 100 * V("E10.llm.%s.%s.prevalence.ci_lo" % (sysn, l), m), 100 * V("E10.llm.%s.%s.prevalence.ci_hi" % (sysn, l), m), V("E10.llm.%s.%s.count" % (sysn, l))]
        if labs:
            systems.append({"system": "full_fixed" if sysn == "full" else sysn, "n": V("E10.llm.%s.n" % sysn), "labels": labs})
    if systems:
        llm = {"systems": systems, "kappa": [V("E10.kappa.mean"), V("E10.kappa.ci_lo"), V("E10.kappa.ci_hi")] if V("E10.kappa.mean") is not None else None,
               "n_annotated": V("E10.n_annotated")}
    if not auto and not llm:
        return None
    return {"split": "BIRD dev (11 databases)", "auto": auto, "llm": llm, "src": ["results/E10.json"]}
