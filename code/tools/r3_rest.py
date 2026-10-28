#!/usr/bin/env python3
"""R3 / E-R: retrieval activity on the development set, routing-score controls and the P5' re-run without timeout roots. Post hoc, descriptive; no model call
(only stored runs, stored completions and read-only SQLite executions on the dev databases). SAP section 26.

    worker machine:  cd code && NL2SQL_ROOT=$R PYTHONPATH=. python tools/r3_rest.py --runs $R/runs --data $R/data --out results/r3 --part all [--B 10000]

part retrieval  schema routing / truncation / flags per backbone from the stored preds (`ctx_full_schema`, `ctx_tables`, `ctx_tokens`), tables per database,
                and the stack contrast (root-only minus single call) by schema-size stratum (item-level intervals; the strata are made of whole databases).
part routing    on the Plus routing run (MIN/M1/gated-route-s0, 1 534 questions): AUROC and accepted-EX of v3, the frozen route score, the agreement share alone,
                the non-empty flag alone, a logistic refit WITHOUT the agreement feature (fitted on HR like the frozen score, and leave-one-database-out on dev),
                and an equal-or-lower-cost five-sample vote rebuilt from the stored n = 4 completions of that run (the answer = majority execution result of the
                final answer and the four samples, ties to the final answer; confidence = share of the majority group).
part p5         P5' (gated versus blind retry on the failing roots) as published, and without the roots whose failure is an execution time-out."""
import argparse
import collections
import json
import os
import re
import sqlite3
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3_common as C  # noqa: E402
from r3_common import S  # noqa: E402

EXEC_FAIL = ("err_syntax", "err_identifier", "timeout")
COVS = (0.4, 0.6, 0.8)


def conv(o):
    if isinstance(o, dict):
        return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [conv(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ===================================================================================================== retrieval
def n_tables(data_dir, db_id):
    p = os.path.join(data_dir, "bird/dev/dev_20240627/dev_databases", db_id, db_id + ".sqlite")
    c = sqlite3.connect("file:%s?mode=ro" % p, uri=True)
    try:
        return c.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchone()[0]
    finally:
        c.close()


def part_retrieval(runs_root, data_dir, B):
    out = {"by_backbone": {}, "by_db": {}, "k_t": 6, "fallback_tokens": 6000}
    dbs_seen = {}
    for b in C.BK:
        out["by_backbone"][b] = {}
        for arm in ("rootonly", "gated", "fixed"):
            rows = C.load_preds(runs_root, C.run_path(b, arm, 0))
            full = [r for r in rows.values() if r.get("ctx_full_schema") is True]
            routed = [r for r in rows.values() if r.get("ctx_full_schema") is False]
            na = [r for r in rows.values() if r.get("ctx_full_schema") is None]
            out["by_backbone"][b][arm] = {"n": len(rows), "full_schema": len(full), "routed": len(routed), "no_field": len(na),
                                          "max_ctx_tokens": max(int(r.get("ctx_tokens") or 0) for r in rows.values()),
                                          "n_examples_min": min(len(r.get("exemplar_ids") or []) for r in rows.values()),
                                          "n_examples_max": max(len(r.get("exemplar_ids") or []) for r in rows.values())}
            for r in rows.values():
                dbs_seen.setdefault(r["db_id"], {"tok": [], "n": 0})
                dbs_seen[r["db_id"]]["tok"].append(int(r.get("ctx_tokens") or 0))
                dbs_seen[r["db_id"]]["n"] += 1
    for d, v in sorted(dbs_seen.items()):
        nt = n_tables(data_dir, d)
        out["by_db"][d] = {"n_tables": nt, "ctx_tokens": int(np.median(v["tok"])), "n_questions": v["n"] // (3 * 3)}
    big = [d for d, v in out["by_db"].items() if v["n_tables"] > out["k_t"]]
    out["dbs_more_than_kt_tables"] = len(big)
    out["share_questions_dbs_more_than_kt_tables"] = 100.0 * sum(out["by_db"][d]["n_questions"] for d in big) / float(sum(v["n_questions"] for v in out["by_db"].values()))
    out["max_tokens_any_db"] = max(v["ctx_tokens"] for v in out["by_db"].values())
    # stack contrast by schema-size stratum (whole databases; item-level paired bootstrap)
    thr = 3000
    out["strata_threshold_tokens"] = thr
    out["stack_by_stratum"] = {}
    for b in C.BK:
        z = C.load_preds(runs_root, C.run_path(b, "zs", 0))
        ro = C.load_preds(runs_root, C.run_path(b, "rootonly", 0))
        qs = sorted(set(z) & set(ro))
        d = np.array([C.ex_of(ro[q]) - C.ex_of(z[q]) for q in qs], float)
        big_q = np.array([out["by_db"][z[q]["db_id"]]["ctx_tokens"] >= thr for q in qs])
        dbs = np.array([z[q]["db_id"] for q in qs])
        e = {}
        for nm, m in (("large", big_q), ("small", ~big_q)):
            est = 100.0 * d[m].mean()
            ic = tuple(100.0 * x for x in S.pctl_ci(S.item_boot_means(d[m], B=B)))
            e[nm] = {"est": est, "item_ci": ic, "n": int(m.sum()), "n_dbs": int(len(set(dbs[m])))}
        out["stack_by_stratum"][b] = e
    return out


# ===================================================================================================== routing
def hr_rows(runs_root):
    rel = "X5/2-hr-cons"
    return os.path.join(runs_root, rel, "preds.jsonl")


def part_routing(runs_root, B):
    import fit_route_score as F
    from analysis_pipeline import primary
    run = C.load_preds(runs_root, "MIN/M1/gated-route-s0")
    qs = sorted(run)
    rows = [run[q] for q in qs]
    coef = json.load(open(os.path.join(os.path.dirname(os.path.abspath(runs_root.rstrip("/"))), "exp/frozen/route_coef_hr.json")))
    v3, ca = primary.route_scores(rows, coef, 4)
    feats = [F.row_features(r, 4) for r in rows]
    cons = np.array([f["consistency"] if f["consistency"] is not None else np.nan for f in feats], float)
    nonempty = np.array([f["nonempty"] if f["nonempty"] is not None else np.nan for f in feats], float)
    y = np.array([C.ex_of(r) if C.ex_of(r) is not None else -1 for r in rows])
    dbs = np.array([r["db_id"] for r in rows])
    ok = np.isfinite(v3) & np.isfinite(ca) & np.isfinite(cons) & np.isfinite(nonempty) & (y >= 0)
    v3, ca, cons, nonempty, y, dbs = v3[ok], ca[ok], cons[ok], nonempty[ok], y[ok], dbs[ok]
    n = int(ok.sum())
    # HR-fitted models (same data, regulariser and standardisation as the frozen score)
    Xhr, yhr, _, _, _ = F.load_rows([hr_rows(runs_root)], ("v3", "consistency", "nonempty"), k=4)
    m3 = F.fit_logistic(Xhr, yhr, 1.0)
    m2 = F.fit_logistic(Xhr[:, [0, 2]], yhr, 1.0)
    m1 = F.fit_logistic(Xhr[:, [0]], yhr, 1.0)
    Xd = np.column_stack([v3, cons, nonempty])
    scores = collections.OrderedDict([
        ("v3", v3), ("route_ca_frozen", ca), ("agreement_only", cons), ("nonempty_only", nonempty),
        ("hr_refit_3feat", F.predict(m3, Xd)), ("hr_refit_no_agreement", F.predict(m2, Xd[:, [0, 2]])), ("hr_refit_v3_only", F.predict(m1, Xd[:, [0]])),
        ("dev_lodo_3feat", F.lodo_predict(Xd, y, dbs)), ("dev_lodo_no_agreement", F.lodo_predict(Xd[:, [0, 2]], y, dbs)),
    ])
    out = {"n": n, "ex_rate": 100.0 * float(y.mean()), "scores": {}, "diff": {}, "hr_n": int(len(yhr)),
           "coef_no_agreement": {"features": ["v3", "nonempty"], "coef": [float(x) for x in m2["coef"]], "intercept": m2["intercept"]},
           "coef_3feat_refit": {"coef": [float(x) for x in m3["coef"]], "intercept": m3["intercept"]}}
    boot = S.ClusterBoot(dbs, B=B)
    # AUROC + accepted EX at coverage, cluster bootstrap
    def metrics(s, idx):
        return S.auroc(s[idx], y[idx])
    aucs = {}
    for nm, s in scores.items():
        est = S.auroc(s, y)
        bt = boot.stat(lambda idx, s=s: S.auroc(s[idx], y[idx]), B=min(B, 4000))
        aucs[nm] = bt
        rc = {}
        for c in COVS:
            rng = np.random.RandomState(1)
            reps = []
            for _ in range(20):
                o = np.lexsort((rng.rand(len(y)), -s))
                k = max(1, int(round(c * len(y))))
                reps.append(float(y[o][:k].mean()))
            rc[str(c)] = 100.0 * float(np.mean(reps))
        out["scores"][nm] = {"auroc": est, "auroc_ci": S.pctl_ci(bt), "acc_at_cov": rc}
    for a_, b_ in (("route_ca_frozen", "v3"), ("hr_refit_no_agreement", "v3"), ("route_ca_frozen", "hr_refit_no_agreement"), ("dev_lodo_no_agreement", "v3"),
                   ("dev_lodo_3feat", "dev_lodo_no_agreement"), ("agreement_only", "v3"), ("hr_refit_3feat", "v3")):
        bt = boot.stat(lambda idx, a_=a_, b_=b_: S.auroc(scores[a_][idx], y[idx]) - S.auroc(scores[b_][idx], y[idx]), B=min(B, 4000))
        out["diff"]["%s_minus_%s" % (a_, b_)] = {"est": S.auroc(scores[a_], y) - S.auroc(scores[b_], y), "ci": S.pctl_ci(bt)}
    # the stored five-sample vote
    out["vote"] = vote_baseline(runs_root, qs, ok, run, scores, y, dbs, B)
    return out


def _samples_by_qid(runs_root, rel, items):
    """qid -> [4 sample SQL strings] rebuilt from the chat-call archive of the routing run (requests with params.n == 4; the question is the one after
    '# Request' of the generation prompt)."""
    from nl2sql import cache_archive as CA
    from nl2sql.util import strip_sql
    recs = CA.read_archive(os.path.join(runs_root, rel, CA.ARCHIVE))
    heads = collections.defaultdict(list)
    for q, it in items.items():
        heads[it.question.strip()].append(it)
    out, ambiguous, n_req = {}, set(), 0
    for rec in recs:
        req = json.loads(rec["req"])
        if (req["params"].get("n") or 1) < 2:
            continue
        n_req += 1
        user = req["messages"][-1]["content"]
        if "# Request\nQuestion: " not in user:
            continue
        tail = user.rsplit("# Request\nQuestion: ", 1)[1]
        cand = [it for it in heads.get(tail.split("\nEvidence: ", 1)[0].strip(), [])
                if tail.split("\nEvidence: ", 1)[1].startswith(it.evidence or "(none)")] if "\nEvidence: " in tail else []
        if len(cand) != 1:
            ambiguous.add(len(cand))
            continue
        t = json.loads(rec["resp"])["text"]
        t = t if isinstance(t, list) else [t]
        out.setdefault(cand[0].qid, []).extend(strip_sql(x) for x in t)
    return out, n_req


def _vote(item, sqls, ev, db, ex0, cache):
    """sqls[0] = the candidate the vote is anchored on (EX ex0 known), sqls[1:] = samples -> (EX of the majority answer, vote share)."""
    keys = []
    for sql in sqls:
        if sql not in cache:
            r = db.execute(item.db_path, sql, timeout_s=30, max_rows=10000) if sql and sql.strip() else {"ok": False, "rows": []}
            cache[sql] = db.rows_key(r["rows"]) if r["ok"] else None
        keys.append(cache[sql])
    groups = collections.OrderedDict()
    for i, k in enumerate(keys):
        if k is not None:
            groups.setdefault(k, []).append(i)
    if not groups:
        return ex0, 1.0 / len(sqls)
    best = max(groups.values(), key=lambda g: (len(g), 1 if 0 in g else 0, -g[0]))
    share = len(best) / float(len(sqls))
    if 0 in best:
        return ex0, share
    sc = ev.score_sql(item, sqls[best[0]])
    return (0 if sc.get("ex") is None else int(sc["ex"])), share


def vote_baseline(runs_root, qs, ok_mask, run, scores, y, dbs, B):
    """Equal-cost vote from the stored n = 4 completions of the routing run: (A) the final answer plus the four samples, (B) the root plus the four samples
    (no critic, no loop). Majority group of execution results; ties go to the group of the anchor; confidence = share of the majority group."""
    from nl2sql import data, db
    from nl2sql.eval.score import Evaluator
    rel = "MIN/M1/gated-route-s0"
    items = {i.qid: i for i in data.load_split("bird_dev")}
    samples, n_req = _samples_by_qid(runs_root, rel, items)
    roots = {r["qid"]: r for r in C.jl(os.path.join(runs_root, rel, "cands.jsonl")) if r.get("cand_idx") == 0}
    ev = Evaluator("bird_dev", run_id="r3-vote")
    qlist = [q for q, k in zip(qs, ok_mask) if k]
    res = {"A": ([], []), "B": ([], [])}
    final_ex, n_missing = [], 0
    for q in qlist:
        it = items[q]
        sm = samples.get(q) or []
        if len(sm) != 4:
            n_missing += 1
        final_ex.append(C.ex_of(run[q]))
        cache = {}
        for nm, sql0, ex0 in (("A", run[q]["final_sql"], C.ex_of(run[q])), ("B", roots[q]["sql"], roots[q].get("ex"))):
            e, sh = _vote(it, [sql0] + sm, ev, db, int(ex0 or 0), cache)
            res[nm][0].append(e)
            res[nm][1].append(sh)
    final_ex = np.array(final_ex, float)
    assert len(final_ex) == len(y), (len(final_ex), len(y))
    out = {"n": len(final_ex), "n_requests_n4": n_req, "n_questions_with_4_samples": sum(1 for q in qlist if len(samples.get(q) or []) == 4),
           "n_questions_without_4_samples": n_missing, "ex_final": 100.0 * float(final_ex.mean()), "variants": {}}
    boot = S.ClusterBoot(dbs, B=B)
    Bb = min(B, 4000)
    sc_route = scores["route_ca_frozen"]
    for nm, label in (("A", "final answer + 4 samples"), ("B", "root + 4 samples")):
        ve, vs = np.array(res[nm][0], float), np.array(res[nm][1], float)
        e = {"label": label, "ex_vote": 100.0 * float(ve.mean()), "auroc_vote_share": S.auroc(vs, ve.astype(int)) if len(set(ve.tolist())) > 1 else None, "cov": {}}
        e["auroc_vote_share_ci"] = S.pctl_ci(boot.stat(lambda idx: S.auroc(vs[idx], ve[idx].astype(int)), B=Bb))
        e["auroc_route_minus_vote"] = {"est": S.auroc(sc_route, final_ex.astype(int)) - e["auroc_vote_share"],
                                       "ci": S.pctl_ci(boot.stat(lambda idx: S.auroc(sc_route[idx], final_ex[idx].astype(int)) - S.auroc(vs[idx], ve[idx].astype(int)), B=Bb))}
        for c in COVS:
            r_ = {}
            for k2, sc, ee in (("route", sc_route, final_ex), ("vote", vs, ve)):
                rng = np.random.RandomState(1)
                reps = []
                for _ in range(20):
                    o = np.lexsort((rng.rand(len(ee)), -sc))
                    reps.append(float(ee[o][:max(1, int(round(c * len(ee))))].mean()))
                r_[k2] = 100.0 * float(np.mean(reps))
            e["cov"][str(c)] = {"acc_route": r_["route"], "acc_vote": r_["vote"], "correct_per_100_asked_route": c * r_["route"], "correct_per_100_asked_vote": c * r_["vote"]}
        out["variants"][nm] = e
    out["auroc_route_on_final_answers"] = S.auroc(sc_route, final_ex.astype(int))
    return out


# ===================================================================================================== P5'
def p5_population(runs_root):
    """failing gated roots (status in EXEC_FAIL) of the three backbones with the repair outcome of the gated loop and of the blind retry."""
    def pop(b):
        g = C.load_preds(runs_root, C.run_path(b, "gated", 0))
        bl = C.load_preds(runs_root, "MIN/M3/blind-s0" if b == "plus" else "MIN/M5/blind-%s-s0" % b)
        f = C.load_preds(runs_root, C.run_path(b, "fixed", 0))
        gc = {r["qid"]: r for r in C.jl(os.path.join(runs_root, C.run_path(b, "gated", 0), "cands.jsonl")) if r.get("cand_idx") == 0}
        fc = {r["qid"]: r for r in C.jl(os.path.join(runs_root, C.run_path(b, "fixed", 0), "cands.jsonl")) if r.get("cand_idx") == 0}
        out = []
        for q, c in gc.items():
            if c.get("status") in EXEC_FAIL and q in g:
                a, bb = C.ex_of(g[q]), (C.ex_of(bl[q]) if q in bl else None)
                if bb is None:
                    continue
                st_f = fc.get(q, {}).get("status")
                same = (fc.get(q, {}).get("sql") or "").strip() == (c.get("sql") or "").strip()
                out.append({"b": b, "qid": q, "db": g[q]["db_id"], "gated": 1 if a == 1 else 0, "blind": int(bb), "status": c.get("status"),
                            "timeout_only": bool(c.get("status") == "timeout" and same and st_f in ("ok_nonempty", "ok_empty"))})
        return out
    allr = []
    for b in C.BK:
        allr += pop(b)
    return allr


def part_p5(runs_root, B):
    allr = p5_population(runs_root)

    def stat(rows, label):
        a = np.array([r["gated"] for r in rows]); bb = np.array([r["blind"] for r in rows])
        dbs = np.array([r["db"] for r in rows])
        b_ = int(((a == 1) & (bb == 0)).sum()); c_ = int(((a == 0) & (bb == 1)).sum())
        mc = S.mcnemar_exact(b_, c_)
        dd = (a - bb).astype(float)
        boot = S.ClusterBoot(dbs, B=B)
        est = 100.0 * dd.mean()
        ci = tuple(100.0 * x for x in S.pctl_ci(boot.means(dd)))
        ic = tuple(100.0 * x for x in S.pctl_ci(S.item_boot_means(dd, B=B)))
        return {"label": label, "n": len(rows), "repaired_gated": int(a.sum()), "repaired_blind": int(bb.sum()), "rate_gated": 100.0 * float(a.mean()),
                "rate_blind": 100.0 * float(bb.mean()), "diff_pp": est, "ci": ci, "item_ci": ic, "b": b_, "c": c_,
                "p_one_greater": mc["p_one_greater"], "p_two": mc["p_two"], "n_dbs": int(len(set(dbs.tolist())))}
    out = {"timeout_roots": {b: sum(1 for r in allr if r["b"] == b and r["status"] == "timeout") for b in C.BK},
           "timeout_only_roots": {b: sum(1 for r in allr if r["b"] == b and r["timeout_only"]) for b in C.BK},
           "all": stat(allr, "all failing roots"),
           "minus_timeout_only": stat([r for r in allr if not r["timeout_only"]], "without timeout-only roots"),
           "minus_any_timeout": stat([r for r in allr if r["status"] != "timeout"], "without any timeout root")}
    out["by_backbone"] = {b: stat([r for r in allr if r["b"] == b], b) for b in C.BK}
    return out


def write_csvs(part, res, out):
    W = lambda name, h, rows: C.write_csv(os.path.join(out, name), h, rows)
    if part == "retrieval":
        W("retrieval_dbs.csv", ["db_id", "n_tables", "median_ctx_tokens", "n_questions"], [[d, v["n_tables"], v["ctx_tokens"], v["n_questions"]] for d, v in sorted(res["by_db"].items())])
        W("retrieval_runs.csv", ["backbone", "arm", "n", "full_schema", "routed", "max_ctx_tokens"], [[b, a, e["n"], e["full_schema"], e["routed"], e["max_ctx_tokens"]] for b, d in res["by_backbone"].items() for a, e in d.items()])
        W("retrieval_stack_strata.csv", ["backbone", "stratum", "n", "n_dbs", "stack_pp", "item_lo", "item_hi"], [[b, s_, e["n"], e["n_dbs"], e["est"], e["item_ci"][0], e["item_ci"][1]] for b, d in res["stack_by_stratum"].items() for s_, e in d.items()])
    elif part == "routing":
        W("routing_scores.csv", ["score", "auroc", "lo", "hi", "acc40", "acc60", "acc80"], [[k, e["auroc"], e["auroc_ci"][0], e["auroc_ci"][1], e["acc_at_cov"]["0.4"], e["acc_at_cov"]["0.6"], e["acc_at_cov"]["0.8"]] for k, e in res["scores"].items()])
        W("routing_diffs.csv", ["contrast", "diff", "lo", "hi"], [[k, e["est"], e["ci"][0], e["ci"][1]] for k, e in res["diff"].items()])
        W("routing_vote.csv", ["variant", "ex_vote", "auroc_vote", "auroc_route_minus_vote", "lo", "hi", "coverage", "acc_route", "acc_vote", "correct_per_100_route", "correct_per_100_vote"],
          [[v, e["ex_vote"], e["auroc_vote_share"], e["auroc_route_minus_vote"]["est"], e["auroc_route_minus_vote"]["ci"][0], e["auroc_route_minus_vote"]["ci"][1], c, x["acc_route"], x["acc_vote"], x["correct_per_100_asked_route"], x["correct_per_100_asked_vote"]]
           for v, e in res["vote"]["variants"].items() for c, x in e["cov"].items()])
    elif part == "p5":
        rows = [[k, e["n"], e["repaired_gated"], e["repaired_blind"], e["rate_gated"], e["rate_blind"], e["diff_pp"], e["ci"][0], e["ci"][1], e["item_ci"][0], e["item_ci"][1], e["b"], e["c"], e["p_one_greater"], e["p_two"]]
                for k, e in [("all", res["all"]), ("minus_timeout_only", res["minus_timeout_only"]), ("minus_any_timeout", res["minus_any_timeout"])] + sorted(res["by_backbone"].items())]
        W("p5.csv", ["population", "n", "gated", "blind", "rate_gated", "rate_blind", "diff_pp", "cl_lo", "cl_hi", "item_lo", "item_hi", "b", "c", "p_one", "p_two"], rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="all", choices=("all", "retrieval", "routing", "p5"))
    ap.add_argument("--B", type=int, default=C.B_DEFAULT)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for part, fn in (("retrieval", lambda: part_retrieval(a.runs, a.data, a.B)), ("p5", lambda: part_p5(a.runs, a.B)), ("routing", lambda: part_routing(a.runs, a.B))):
        if a.part in ("all", part):
            res = fn()
            with open(os.path.join(a.out, "%s.json" % part), "w") as f:
                json.dump(conv(res), f, indent=1)
            write_csvs(part, conv(res), a.out)
            print("[r3_rest] %s written" % part)


if __name__ == "__main__":
    main()
