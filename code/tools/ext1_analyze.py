#!/usr/bin/env python3
"""EXT1 analysis: external validity on Spider-Realistic (SAP Amendment 8, docs/SAP.md section 21). New file; edits nothing of analysis_pipeline/,
make_macros.py, make_tables.py or analyze_all.sh (it imports only `analysis_pipeline.stats`, read-only).

  python code/tools/ext1_analyze.py analyze --runs $R/runs/EXT1 --out results/ext1 [--B 10000] [--data-file .../spider-realistic.json]
  python code/tools/ext1_analyze.py cost    --runs $R/runs/EXT1 [--prior 0.58]          # stop rule of SAP 21.8: exit 2 above CNY 100, exit 3 above 150
  (runs on a worker machine: needs numpy; python >= 3.6)

Run directories: <runs>/<arm>-<bk>-s0 with arm in zs, fixed, gated, rootonly and bk in plus, flash, 27b (exp/queues/gen_ext1.py).
Everything is scope `external-validity`: the output file never enters a `\\res{}` macro or the Holm family P*; family X = {XB1, XB3} (m = 2).
Definitions (SAP 21.5): XB1 = mean over {flash, 27b} of EX_ts(gated) - EX_ts(zs), H0 <= 0; XB2 = plus gated - zs, two-sided CI with a reading rule;
XB3 = pooled over {plus, flash, 27b}: (a) mean EX_ts(gated) - EX_ts(fixed) vs -1.5 pp, (b) mean calls ratio gated/fixed vs 0.80, p = max(p_a, p_b)."""
import argparse
import csv
import glob
import gzip
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(CODE, ".."))
sys.path.insert(0, CODE)
from analysis_pipeline import stats as S  # noqa: E402

SCOPE = "external-validity"
BACKBONES = ("plus", "flash", "27b")
WK = ("flash", "27b")
ARMS = ("zs", "fixed", "gated", "rootonly")
N_EXPECTED = 508
MARGIN_A = 1.5                  # XB3(a): H0a dEX <= -1.5 pp
RATIO_NULL = 0.80               # XB3(b): H0b R >= 0.80
NI_XB2 = 3.0                    # XB2 magnitude statements
SENS_DROP_DB = "world_1"        # SAP 21.6
CASES_PER_DIRECTION = 3
STOP_SOFT, CAP_HARD = 100.0, 150.0
BIRD_REF = {"XB1": 2.93, "XB1.ci": [1.56, 4.12], "XB2": 0.65, "XB2.ci": [-0.97, 2.51], "XB3.ratio": 0.580, "XB3.ratio.ci": [0.566, 0.597],
            "XB3.dex": -0.26, "XB3.dex.ci": [-0.93, 0.30], "source": "internal note KEYS sections 1-2, results/W1.json (BIRD dev, r0)"}


# ---------------------------------------------------------------------------------------------------------------- loading
def jl(path):
    out = []
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def ex_ts_value(r):
    """test-suite EX of a preds row: 0/1, or None when the row is not a measurement. api_fail -> 0 (SAP 10); sys_error -> None; ex_ts null -> None."""
    ec = r.get("error_class")
    if ec == "api_fail":
        return 0
    if ec and str(ec).startswith("sys_error"):
        return None
    v = r.get("ex_ts")
    return None if v is None else int(v)


class Arm(object):
    def __init__(self, path, rows):
        self.path = path
        self.rows = rows                      # qid -> row (last row wins)
        self.n_sys_error = sum(1 for r in rows.values() if str(r.get("error_class") or "").startswith("sys_error"))
        self.n_api_fail = sum(1 for r in rows.values() if r.get("error_class") == "api_fail")
        self.n_null = sum(1 for r in rows.values() if ex_ts_value(r) is None)
        self.complete = len(rows) >= N_EXPECTED and self.n_null == 0

    def status(self):
        return "complete" if self.complete else "incomplete: n=%d, sys_error=%d, ex_ts null=%d" % (len(self.rows), self.n_sys_error, self.n_null)


def load_arm(runs, arm, bk):
    d = os.path.join(runs, "%s-%s-s0" % (arm, bk))
    p = os.path.join(d, "preds.jsonl")
    if not os.path.exists(p):
        return None
    rows = {}
    for r in jl(p):
        rows[r["qid"]] = r
    return Arm(d, rows)


def load_all(runs):
    return {(bk, arm): load_arm(runs, arm, bk) for bk in BACKBONES for arm in ARMS}


def align(arms):
    """arms: dict name -> Arm (all complete). -> dict(qids, dbs, ex (dict name -> int array), plain, calls, tok, rows)."""
    qs = None
    for a in arms.values():
        qs = set(a.rows) if qs is None else qs & set(a.rows)
    qids = sorted(qs)
    dbs = np.array([next(iter(arms.values())).rows[q]["db_id"] for q in qids])
    out = {"qids": qids, "dbs": dbs, "ex": {}, "plain": {}, "calls": {}, "rows": {}}
    for k, a in arms.items():
        rs = [a.rows[q] for q in qids]
        out["rows"][k] = rs
        out["ex"][k] = np.array([ex_ts_value(r) for r in rs], int)
        out["plain"][k] = np.array([int(r.get("ex") or 0) for r in rs], int)
        out["calls"][k] = np.array([float(r.get("calls", 0) or 0) for r in rs])
    return out


# ---------------------------------------------------------------------------------------------------------------- statistics
def same_sign_share(dbs, d, delta):
    """share of databases (weight: equal) whose mean item difference has the sign of the pooled effect; zero-difference databases are not counted."""
    if delta == 0:
        return None
    u = sorted(set(dbs.tolist()))
    vals = [float(np.mean(d[dbs == x])) for x in u]
    vals = [v for v in vals if v != 0]
    return (sum(1 for v in vals if (v > 0) == (delta > 0)) / float(len(vals))) if vals else None


def contrast(ex_a, ex_b, dbs, B, boot=None):
    """paired EX difference A - B in pp: db-clustered percentile CI, item-level CI, leave-one-db-out range, same-sign share, exact McNemar. One-sided p's: H0 <= 0."""
    boot = boot or S.ClusterBoot(dbs, B)
    d = (np.asarray(ex_a) - np.asarray(ex_b)).astype(float)
    delta = 100.0 * float(d.mean())
    bt = 100.0 * (boot.means(ex_a) - boot.means(ex_b))
    ib = 100.0 * S.item_boot_means(d, B)
    lo, hi = S.pctl_ci(bt)
    ilo, ihi = S.pctl_ci(ib)
    loo = boot.loo(lambda idx: 100.0 * d[idx].mean())
    b = int(np.sum((np.asarray(ex_a) == 1) & (np.asarray(ex_b) == 0)))
    c = int(np.sum((np.asarray(ex_a) == 0) & (np.asarray(ex_b) == 1)))
    mc = S.mcnemar_exact(b, c)
    return {"delta_pp": delta, "ci_lo": lo, "ci_hi": hi, "item_ci_lo": ilo, "item_ci_hi": ihi, "boot": bt, "item_boot": ib, "n": int(len(d)), "n_dbs": boot.k,
            "loo_min": min(v for _, v in loo) if loo else None, "loo_max": max(v for _, v in loo) if loo else None,
            "same_sign_share": same_sign_share(dbs, d, delta), "b": b, "c": c, "mcnemar_p_two": mc["p_two"],
            "p_1s": S.p_le(bt, 0.0), "p_item_1s": S.p_le(ib, 0.0)}


def pooled(ex_a, ex_b, dbs, B, boot=None):
    """mean over backbones of the paired differences (pp), common database resample; ex_a, ex_b: dict backbone -> int array."""
    boot = boot or S.ClusterBoot(dbs, B)
    ks = list(ex_a)
    bt = np.mean([100.0 * (boot.means(ex_a[k]) - boot.means(ex_b[k])) for k in ks], axis=0)
    d = np.mean([(ex_a[k] - ex_b[k]).astype(float) for k in ks], axis=0)
    delta = 100.0 * float(d.mean())
    ib = np.mean([100.0 * S.item_boot_means((ex_a[k] - ex_b[k]).astype(float), B) for k in ks], axis=0)    # same seed -> same item resample for every backbone
    lo, hi = S.pctl_ci(bt)
    ilo, ihi = S.pctl_ci(ib)
    loo = boot.loo(lambda idx: 100.0 * d[idx].mean())
    return {"delta_pp": delta, "ci_lo": lo, "ci_hi": hi, "item_ci_lo": ilo, "item_ci_hi": ihi, "boot": bt, "item_boot": ib, "n": int(len(d)), "n_dbs": boot.k,
            "loo_min": min(v for _, v in loo) if loo else None, "loo_max": max(v for _, v in loo) if loo else None,
            "same_sign_share": same_sign_share(dbs, d, delta), "per_backbone": {k: 100.0 * float((ex_a[k] - ex_b[k]).mean()) for k in ks},
            "p_1s": S.p_le(bt, 0.0), "p_item_1s": S.p_le(ib, 0.0)}


def xb3_block(ex_g, ex_f, calls_g, calls_f, dbs, B, boot=None):
    """XB3 on the backbones present in the dicts: (a) pooled dEX vs -MARGIN_A, (b) pooled calls ratio vs RATIO_NULL, intersection-union."""
    boot = boot or S.ClusterBoot(dbs, B)
    o = pooled(ex_g, ex_f, dbs, B, boot)
    ks = list(ex_g)
    rb = np.mean([boot.ratio(calls_g[k], calls_f[k]) for k in ks], axis=0)
    rib = np.mean([S.item_boot_ratio(calls_g[k], calls_f[k], B) for k in ks], axis=0)
    r_per = {k: float(np.sum(calls_g[k]) / max(np.sum(calls_f[k]), 1e-12)) for k in ks}
    r = float(np.mean(list(r_per.values())))
    rlo, rhi = S.pctl_ci(rb)
    rilo, rihi = S.pctl_ci(rib)
    o["p_a_1s"] = S.p_le(o["boot"], -MARGIN_A)
    o["p_a_item_1s"] = S.p_le(o["item_boot"], -MARGIN_A)
    o["ratio"] = {"mean": r, "ci_lo": rlo, "ci_hi": rhi, "item_ci_lo": rilo, "item_ci_hi": rihi, "per_backbone": r_per,
                  "p_b_1s": S.p_ge(rb, RATIO_NULL), "p_b_item_1s": S.p_ge(rib, RATIO_NULL)}
    o["p_1s"] = max(o["p_a_1s"], o["ratio"]["p_b_1s"])
    o["p_item_1s"] = max(o["p_a_item_1s"], o["ratio"]["p_b_item_1s"])
    o["cl_ok"] = bool(o["ci_lo"] is not None and o["ci_lo"] > -MARGIN_A and rhi is not None and rhi < RATIO_NULL)
    o["item_ok"] = bool(o["item_ci_lo"] > -MARGIN_A and rihi < RATIO_NULL)
    return o


# ---------------------------------------------------------------------------------------------------------------- reading rules (SAP 21.5)
def read_xb1(p_adj, p_adj_item, ci_lo, ci_hi, delta, per_backbone, bird_ref=BIRD_REF["XB1"]):
    """SAP 21.5 XB1, first line that applies. per_backbone: dict backbone -> point estimate (pp)."""
    all_pos = all(v > 0 for v in per_backbone.values())
    if p_adj is not None and p_adj < 0.05 and ci_lo is not None and ci_lo > 0:
        word = "confirmed" if all_pos else "supported on average, not on each backbone"
    elif p_adj_item is not None and p_adj_item < 0.05:
        word = "supported at item level, not at database level"
    elif delta > 0:
        word = "direction consistent, not confirmed"
    else:
        word = "not confirmed"
    contradicted = bool(ci_hi is not None and ci_hi < 0)
    return {"verdict": word, "replicates_b": word == "confirmed", "contradicted": contradicted,
            "size_consistent_with_bird": bool(ci_lo is not None and ci_lo <= bird_ref <= ci_hi),
            "text": ("gated is worse than zero-shot on the weak backbones: finding (b) is contradicted on this benchmark" if contradicted else word)}


def read_xb2(ci_lo, ci_hi, delta):
    """SAP 21.5 XB2 (two-sided db-clustered 95 % CI of plus gated - zs)."""
    if ci_lo > 0:
        line, text = "gain", "a gain on the strong backbone; finding (c) is not replicated"
    elif ci_hi < 0:
        line, text = "loss", "gated is worse than zero-shot on the strong backbone (a loss); 'no gain' holds only trivially"
    else:
        line, text = "no_detectable_gain", "no detectable gain on the strong backbone (finding (c) replicated in the sense of 'not distinguishable from zero')"
    ni = bool(ci_lo > -NI_XB2)
    eq = bool(ci_lo > -NI_XB2 and ci_hi < NI_XB2)
    if line == "no_detectable_gain":
        text += "; " + ("equivalent within +/-%.1f pp" % NI_XB2 if eq else ("non-inferior (lower bound > -%.1f pp) but equivalence not shown" % NI_XB2 if ni
                                                                          else "the interval does not exclude differences larger than %.1f pp" % NI_XB2))
    return {"line": line, "text": text, "non_inferior_3pp": ni, "equivalent_3pp": eq}


def read_xb3(p_adj, p_adj_item, ci_lo, ci_hi_ratio, item_ok_flag, delta, ratio):
    """SAP 21.5 XB3. ci_lo: lower bound of pooled dEX (pp); ci_hi_ratio: upper bound of the pooled calls ratio."""
    cl_ok = bool(ci_lo is not None and ci_lo > -MARGIN_A and ci_hi_ratio is not None and ci_hi_ratio < RATIO_NULL)
    fav = bool(delta > -MARGIN_A and ratio < RATIO_NULL)
    if p_adj is not None and p_adj < 0.05 and cl_ok:
        word = "confirmed"
    elif p_adj_item is not None and p_adj_item < 0.05 and item_ok_flag:
        word = "supported at item level, not at database level"
    elif fav:
        word = "direction consistent, not confirmed"
    else:
        word = "not confirmed"
    if delta < -MARGIN_A:
        text = "the gated loop loses more accuracy than the margin on this benchmark; finding (a) is contradicted"
    elif ratio >= RATIO_NULL:
        text = "no call saving on this benchmark"
    elif word == "confirmed":
        text = "finding (a) replicates: fewer calls and accuracy kept within %.1f pp" % MARGIN_A
    elif ratio < RATIO_NULL:
        text = "fewer calls, accuracy parity not shown"
    else:
        text = word
    return {"verdict": word, "replicates_a": word == "confirmed", "text": text}


# ---------------------------------------------------------------------------------------------------------------- cost
def load_prices(path):
    out = {"models": {}, "cache_hit_ratio": 0.2, "batch_ratio": 0.5}
    for line in open(path):
        s = line.split("#")[0].rstrip()
        if not s.strip():
            continue
        if s.startswith("cache_hit_ratio"):
            out["cache_hit_ratio"] = float(s.split(":")[1])
        elif s.startswith("batch_ratio"):
            out["batch_ratio"] = float(s.split(":")[1])
        elif "{in:" in s:
            name, rest = s.strip().split("{", 1)
            d = {}
            for kv in rest.rstrip("}").split(","):
                k, v = kv.split(":")
                d[k.strip()] = float(v)
            out["models"][name.strip().rstrip(":").strip()] = d
    return out


def run_cost(run_dir, prices, cache_free=False):
    """CNY of a run from usage*.jsonl. cache_free=False: every chat row at its recorded tokens, whether or not it was served from our cache (the E11
    convention: cost of the system, independent of cache state). cache_free=True: rows with cached=true are free (money actually spent; the stop rule).
    Embedding rows are priced separately (prices UNVERIFIED) and returned as `embed`."""
    ratio = prices.get("cache_hit_ratio", 0.2)
    chat = emb = 0.0
    unpriced = 0
    ncalls = 0
    for f in sorted(glob.glob(os.path.join(run_dir, "usage*.jsonl"))):
        for r in jl(f):
            if cache_free and r.get("cached"):
                continue
            pr = prices["models"].get(r.get("model"))
            if pr is None:
                unpriced += 1
                continue
            pt, ct = r.get("pt", 0) or 0, r.get("ct", 0) or 0
            if r.get("kind") in ("embed", "rerank"):
                emb += pt * pr["in"] / 1e6
                continue
            cached = min(r.get("cached_tokens", 0) or 0, pt)
            y = ((pt - cached) * pr["in"] + cached * pr["in"] * ratio + ct * pr["out"]) / 1e6
            chat += y * (prices["batch_ratio"] if r.get("mode") == "batch" else 1.0)
            ncalls += 1
    return {"chat": chat, "embed": emb, "unpriced": unpriced, "calls": ncalls}


def stop_rule(total, next_estimate=0.0):
    """SAP 21.8: 0 = ok, 2 = above the soft stop (no further launches), 3 = launching the next arm would pass the hard cap."""
    if total + next_estimate > CAP_HARD:
        return 3
    if total > STOP_SOFT:
        return 2
    return 0


# ---------------------------------------------------------------------------------------------------------------- output helper
class Res(object):
    def __init__(self):
        self.d = {}
        self.notes = []

    def set(self, k, v):
        if isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.bool_,)):
            v = bool(v)
        self.d[k] = v

    def ci(self, k, est, lo, hi):
        self.set(k, est)
        self.set(k + ".ci_lo", lo)
        self.set(k + ".ci_hi", hi)


def put_contrast(res, pre, o):
    res.ci(pre + ".delta_ex", o["delta_pp"], o["ci_lo"], o["ci_hi"])
    for k in ("item_ci_lo", "item_ci_hi", "n", "n_dbs", "loo_min", "loo_max", "same_sign_share", "p_1s", "p_item_1s"):
        res.set("%s.%s" % (pre, k), o.get(k))
    for k in ("b", "c", "mcnemar_p_two"):
        if k in o:
            res.set("%s.%s" % (pre, k), o[k])
    if "per_backbone" in o:
        for k, v in o["per_backbone"].items():
            res.set("%s.per_backbone.%s" % (pre, k), v)


# ---------------------------------------------------------------------------------------------------------------- analysis
def arm_table(arms, prices, runs):
    rows = []
    for (bk, arm), a in sorted(arms.items(), key=lambda kv: (BACKBONES.index(kv[0][0]), ARMS.index(kv[0][1]))):
        if a is None:
            rows.append({"backbone": bk, "arm": arm, "status": "not run"})
            continue
        r = {"backbone": bk, "arm": arm, "status": a.status(), "n": len(a.rows), "api_fail": a.n_api_fail, "sys_error": a.n_sys_error}
        if a.complete:
            rs = [a.rows[q] for q in sorted(a.rows)]
            ex = np.array([ex_ts_value(x) for x in rs], int)
            dbs = np.array([x["db_id"] for x in rs])
            boot = S.ClusterBoot(dbs, 10000)
            lo, hi = S.pctl_ci(boot.means(ex))
            plain = np.array([int(x.get("ex") or 0) for x in rs])
            n = float(len(rs))
            cost = run_cost(a.path, prices)
            nc = max(int(ex.sum()), 1)
            r.update({"ex_ts": float(ex.mean()), "ex_ts_ci_lo": lo, "ex_ts_ci_hi": hi, "ex_plain": float(plain.mean()),
                      "exec_ok": float(np.mean([bool(x.get("exec_ok")) for x in rs])),
                      "calls_per_q": float(np.mean([x.get("calls", 0) for x in rs])), "prompt_tok_per_q": float(np.mean([x.get("prompt_tokens", 0) for x in rs])),
                      "completion_tok_per_q": float(np.mean([x.get("completion_tokens", 0) for x in rs])),
                      "latency_mean_s": float(np.mean([x.get("latency_s", 0.0) for x in rs])),
                      "revised_share": float(np.mean([("revise" in (x.get("actions") or [])) for x in rs])),
                      "cost_cny_per_q": cost["chat"] / n, "cost_cny_per_correct": cost["chat"] / nc, "cost_cny_total": cost["chat"],
                      "embed_cny_total": cost["embed"], "unpriced_rows": cost["unpriced"]})
        rows.append(r)
    return rows


def gate_descriptives(arms, res):
    """no hypothesis: items whose gated run revised (the gate fired = root failed to execute / timed out), their outcome, with Wilson intervals."""
    out = {}
    for bk in BACKBONES:
        g, ro = arms.get((bk, "gated")), arms.get((bk, "rootonly"))
        if g is None or not g.complete:
            continue
        rev = [r for r in g.rows.values() if "revise" in (r.get("actions") or [])]
        n, k = len(g.rows), len(rev)
        fin_exec = sum(1 for r in rev if r.get("exec_ok"))
        fin_ok = sum(1 for r in rev if ex_ts_value(r) == 1)
        d = {"n": n, "gate_fired": k, "gate_fired_share": k / float(n), "final_executes": fin_exec, "final_correct": fin_ok}
        if k:
            d["final_executes_wilson"] = list(S.wilson(fin_exec, k))
            d["final_correct_wilson"] = list(S.wilson(fin_ok, k))
        if ro is not None and ro.complete:
            d["rootonly_exec_fail"] = sum(1 for r in ro.rows.values() if not r.get("exec_ok"))
        out[bk] = d
        for kk, v in d.items():
            res.set("EXT1.gate.%s.%s" % (bk, kk), v)
    return out


def case_rows(al, a, b, qid_to_item, per_dir=CASES_PER_DIRECTION):
    """fixed rule (SAP 21.11): discordant items sorted by qid (ascending), first `per_dir` per direction."""
    ea, eb = al["ex"][a], al["ex"][b]
    idx = {q: i for i, q in enumerate(al["qids"])}
    out = {}
    for name, cond in (("%s_right_%s_wrong" % (a, b), (ea == 1) & (eb == 0)), ("%s_right_%s_wrong" % (b, a), (ea == 0) & (eb == 1))):
        qs = sorted(q for q in al["qids"] if cond[idx[q]])
        lst = []
        for q in qs[:per_dir]:
            i = idx[q]
            it = qid_to_item.get(q, {})
            lst.append({"qid": q, "db_id": al["dbs"][i], "question": it.get("question"), "gold": it.get("query"),
                        "sql_" + a: al["rows"][a][i].get("final_sql"), "sql_" + b: al["rows"][b][i].get("final_sql"),
                        "ex_ts_" + a: int(ea[i]), "ex_ts_" + b: int(eb[i])})
        out[name] = {"n_discordant": int(len(qs)), "cases": lst}
    return out


def gate_cases(arms, bk, per_dir=CASES_PER_DIRECTION):
    g = arms.get((bk, "gated"))
    if g is None or not g.complete:
        return []
    qs = sorted(q for q, r in g.rows.items() if "revise" in (r.get("actions") or []))
    return [{"qid": q, "actions": g.rows[q].get("actions"), "final_sql": g.rows[q].get("final_sql"), "ex_ts": ex_ts_value(g.rows[q]),
             "exec_ok": g.rows[q].get("exec_ok")} for q in qs[:per_dir]]


def load_items(data_file):
    if not data_file or not os.path.exists(data_file):
        return {}
    rows = json.load(open(data_file, encoding="utf-8"))
    return {"spider_realistic:%s:%d" % (r["db_id"], i): r for i, r in enumerate(rows)}


def run_family(arms, B, drop_db=None):
    """the three hypotheses on the (possibly db-reduced) data. Returns dict with blocks and the family-X Holm; missing arms -> status strings."""
    out = {"status": {}}
    need = {"XB1": [(b, a) for b in WK for a in ("gated", "zs")], "XB2": [("plus", "gated"), ("plus", "zs")],
            "XB3": [(b, a) for b in BACKBONES for a in ("gated", "fixed")]}
    ok = {}
    for h, ks in need.items():
        miss = [("%s/%s" % (b, a)) for (b, a) in ks if arms.get((b, a)) is None or not arms[(b, a)].complete]
        ok[h] = not miss
        out["status"][h] = "ok" if not miss else "not run: missing or incomplete " + ", ".join(miss)
    used = {k: v for k, v in arms.items() if v is not None and v.complete}
    al = {}
    for bk in BACKBONES:
        sub = {a: used[(bk, a)] for a in ARMS if (bk, a) in used}
        if sub:
            al[bk] = align(sub)
    if not al:
        return out
    # common qid set across backbones (all arms of all backbones are the same 508 items; guard anyway)
    ref = None
    for bk, x in al.items():
        ref = set(x["qids"]) if ref is None else ref & set(x["qids"])
    qids = sorted(ref)
    dbs_full = None
    sel = {}
    for bk, x in al.items():
        pos = {q: i for i, q in enumerate(x["qids"])}
        idx = np.array([pos[q] for q in qids])
        if drop_db:
            idx = np.array([i for i in idx if x["dbs"][i] != drop_db])
        sel[bk] = idx
        if dbs_full is None:
            dbs_full = x["dbs"][idx]
    boot = S.ClusterBoot(dbs_full, B)
    E = lambda bk, a: al[bk]["ex"][a][sel[bk]]
    C = lambda bk, a: al[bk]["calls"][a][sel[bk]]
    out["n"] = int(len(dbs_full))
    out["dbs"] = dbs_full
    out["al"], out["sel"] = al, sel
    if ok["XB1"]:
        out["XB1"] = pooled({b: E(b, "gated") for b in WK}, {b: E(b, "zs") for b in WK}, dbs_full, B, boot)
        out["XB1_members"] = {b: contrast(E(b, "gated"), E(b, "zs"), dbs_full, B, boot) for b in WK}
    if ok["XB2"]:
        out["XB2"] = contrast(E("plus", "gated"), E("plus", "zs"), dbs_full, B, boot)
    if ok["XB3"]:
        out["XB3"] = xb3_block({b: E(b, "gated") for b in BACKBONES}, {b: E(b, "fixed") for b in BACKBONES},
                               {b: C(b, "gated") for b in BACKBONES}, {b: C(b, "fixed") for b in BACKBONES}, dbs_full, B, boot)
        out["XB3_members"] = {b: xb3_block({b: E(b, "gated")}, {b: E(b, "fixed")}, {b: C(b, "gated")}, {b: C(b, "fixed")}, dbs_full, B, boot) for b in BACKBONES}
    # Holm over family X (m = 2); a test that cannot be run counts with p = 1
    p2 = {"XB1": (min(1.0, 2.0 * out["XB1"]["p_1s"]) if "XB1" in out else None), "XB3": (min(1.0, 2.0 * out["XB3"]["p_1s"]) if "XB3" in out else None)}
    p2i = {"XB1": (min(1.0, 2.0 * out["XB1"]["p_item_1s"]) if "XB1" in out else None), "XB3": (min(1.0, 2.0 * out["XB3"]["p_item_1s"]) if "XB3" in out else None)}
    out["holm"] = S.holm(p2, m=2)
    out["holm_item"] = S.holm(p2i, m=2)
    out["p2s"] = p2
    # descriptive decompositions (db-clustered, no Holm)
    desc = {}
    for bk in BACKBONES:
        if bk not in al:
            continue
        for (a, b) in (("gated", "zs"), ("gated", "fixed"), ("gated", "rootonly"), ("rootonly", "zs"), ("fixed", "zs")):
            if a in al[bk]["ex"] and b in al[bk]["ex"]:
                desc["%s.%s_minus_%s" % (bk, a, b)] = contrast(E(bk, a), E(bk, b), dbs_full, B, boot)
    out["desc"] = desc
    wk_ro = all((b in al and "rootonly" in al[b]["ex"] and "zs" in al[b]["ex"] and "gated" in al[b]["ex"]) for b in WK)
    if wk_ro:
        out["XB1_loop"] = pooled({b: E(b, "gated") for b in WK}, {b: E(b, "rootonly") for b in WK}, dbs_full, B, boot)
        out["XB1_stack"] = pooled({b: E(b, "rootonly") for b in WK}, {b: E(b, "zs") for b in WK}, dbs_full, B, boot)
    return out


def verdicts(fam):
    v = {}
    if "XB1" in fam:
        o = fam["XB1"]
        v["XB1"] = read_xb1(fam["holm"]["XB1"], fam["holm_item"]["XB1"], o["ci_lo"], o["ci_hi"], o["delta_pp"], o["per_backbone"])
    if "XB2" in fam:
        o = fam["XB2"]
        v["XB2"] = read_xb2(o["ci_lo"], o["ci_hi"], o["delta_pp"])
    if "XB3" in fam:
        o = fam["XB3"]
        v["XB3"] = read_xb3(fam["holm"]["XB3"], fam["holm_item"]["XB3"], o["ci_lo"], o["ratio"]["ci_hi"], o["item_ok"], o["delta_pp"], o["ratio"]["mean"])
    return v


def put_family(res, pre, fam):
    for h, s in fam["status"].items():
        res.set("%s.%s.status" % (pre, h), s)
    if "n" in fam:
        res.set(pre + ".n", fam["n"])
    for h in ("XB1", "XB2"):
        if h in fam:
            put_contrast(res, "%s.%s" % (pre, h), fam[h])
    if "XB1_members" in fam:
        for b, o in fam["XB1_members"].items():
            put_contrast(res, "%s.XB1.member.%s" % (pre, b), o)
    if "XB3" in fam:
        o = fam["XB3"]
        put_contrast(res, pre + ".XB3", o)
        res.ci(pre + ".XB3.calls_ratio", o["ratio"]["mean"], o["ratio"]["ci_lo"], o["ratio"]["ci_hi"])
        for k in ("item_ci_lo", "item_ci_hi", "p_b_1s", "p_b_item_1s"):
            res.set("%s.XB3.calls_ratio.%s" % (pre, k), o["ratio"][k])
        for b, r in o["ratio"]["per_backbone"].items():
            res.set("%s.XB3.calls_ratio.per_backbone.%s" % (pre, b), r)
        res.set(pre + ".XB3.p_a_1s", o["p_a_1s"]); res.set(pre + ".XB3.p_a_item_1s", o["p_a_item_1s"])
        res.set(pre + ".XB3.cl_ok", o["cl_ok"]); res.set(pre + ".XB3.item_ok", o["item_ok"])
        for b, m in fam["XB3_members"].items():
            put_contrast(res, "%s.XB3.member.%s" % (pre, b), m)
            res.ci("%s.XB3.member.%s.calls_ratio" % (pre, b), m["ratio"]["mean"], m["ratio"]["ci_lo"], m["ratio"]["ci_hi"])
            res.set("%s.XB3.member.%s.cl_ok" % (pre, b), m["cl_ok"])
    if "holm" in fam:
        for h, p in fam["holm"].items():
            res.set("%s.%s.p_adj" % (pre, h), p)
            res.set("%s.%s.p_2s" % (pre, h), fam["p2s"][h])
            res.set("%s.%s.p_adj_item" % (pre, h), fam["holm_item"][h])
    for nm in ("XB1_loop", "XB1_stack"):
        if nm in fam:
            put_contrast(res, "%s.%s" % (pre, nm), fam[nm])
    for k, o in fam.get("desc", {}).items():
        put_contrast(res, "%s.desc.%s" % (pre, k), o)


def analyze(args):
    prices = load_prices(args.prices)
    arms = load_all(args.runs)
    res = Res()
    res.set("EXT1.B", args.B)
    res.set("EXT1.seed_boot", S.SEED_BOOT)
    table = arm_table(arms, prices, args.runs)
    for r in table:
        for k, v in r.items():
            if k not in ("backbone", "arm"):
                res.set("EXT1.arm.%s.%s.%s" % (r["backbone"], r["arm"], k), v)
    gate_descriptives(arms, res)
    fam = run_family(arms, args.B)
    put_family(res, "EXT1", fam)
    v = verdicts(fam)
    for h, x in v.items():
        for k, val in x.items():
            res.set("EXT1.%s.%s" % (h, k if k != "verdict" else "verdict"), val)
    # sensitivity without the dominant database
    fam_s = run_family(arms, args.B, drop_db=SENS_DROP_DB)
    put_family(res, "EXT1.sens_no_%s" % SENS_DROP_DB, fam_s)
    vs = verdicts(fam_s)
    for h, x in vs.items():
        res.set("EXT1.sens_no_%s.%s.verdict" % (SENS_DROP_DB, h), x.get("verdict", x.get("line")))
        res.set("EXT1.%s.sens_same_verdict" % h, (x.get("verdict", x.get("line")) == v[h].get("verdict", v[h].get("line"))) if h in v else None)
    # cases
    items = load_items(args.data_file)
    cases = {"_meta": {"rule": "SAP 21.11: discordant items sorted by qid (ascending string order), first %d per direction; no hand picking" % CASES_PER_DIRECTION,
                       "scope": SCOPE}, "contrasts": {}, "gate_fired": {}}
    for h, bk, a, b in (("XB1_flash", "flash", "gated", "zs"), ("XB1_27b", "27b", "gated", "zs"), ("XB2_plus", "plus", "gated", "zs"), ("XB3_plus", "plus", "gated", "fixed")):
        x = fam.get("al", {}).get(bk)
        if x and a in x["ex"] and b in x["ex"]:
            cases["contrasts"][h] = case_rows(x, a, b, items)
    for bk in BACKBONES:
        cases["gate_fired"][bk] = gate_cases(arms, bk)
    # references
    res.set("EXT1.bird_reference", BIRD_REF)
    res.set("EXT1.family_X.note", "Holm over {XB1, XB3}, m = 2; XB2 and all descriptive contrasts outside the Holm adjustment (SAP 21.6)")
    complete = sorted("%s/%s" % (b, a) for (b, a), x in arms.items() if x is not None and x.complete)
    res.set("EXT1.arms_complete", complete)
    res.set("EXT1.arms_incomplete_or_missing", sorted("%s/%s" % (b, a) for (b, a), x in arms.items() if x is None or not x.complete))
    meta = {"scope": SCOPE, "label": "secondary external validity, SAP Amendment 8 (section 21); never enters family P*, Holm of earlier amendments or results_macros.tex",
            "split": "spider_realistic", "metric": "Spider test-suite EX (ex_ts), primary; plain EX reported", "runs": args.runs, "B": args.B,
            "tool": "code/tools/ext1_analyze.py", "n_expected": N_EXPECTED}
    os.makedirs(args.out, exist_ok=True)
    clean = {k: _jsonable(v) for k, v in sorted(res.d.items())}
    clean["_meta"] = meta
    json.dump(clean, open(os.path.join(args.out, "ext1_summary.json"), "w"), indent=1, sort_keys=True)
    # csv tables
    cols = ["backbone", "arm", "status", "n", "ex_ts", "ex_ts_ci_lo", "ex_ts_ci_hi", "ex_plain", "exec_ok", "calls_per_q", "prompt_tok_per_q", "completion_tok_per_q",
            "latency_mean_s", "revised_share", "cost_cny_per_q", "cost_cny_per_correct", "cost_cny_total", "embed_cny_total", "api_fail", "sys_error"]
    with open(os.path.join(args.out, "ext1_arms.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in table:
            w.writerow({k: (("%.6g" % v) if isinstance(v, float) else v) for k, v in r.items()})
    ccols = ["contrast", "n", "delta_pp", "ci_lo", "ci_hi", "item_ci_lo", "item_ci_hi", "loo_min", "loo_max", "same_sign_share", "b", "c", "mcnemar_p_two", "p_1s"]
    with open(os.path.join(args.out, "ext1_contrasts.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=ccols, extrasaction="ignore")
        w.writeheader()
        rows = []
        for h in ("XB1", "XB2"):
            if h in fam:
                rows.append(dict(fam[h], contrast=h))
        for b, o in fam.get("XB1_members", {}).items():
            rows.append(dict(o, contrast="XB1.member.%s (gated-zs)" % b))
        if "XB3" in fam:
            rows.append(dict(fam["XB3"], contrast="XB3 pooled (gated-fixed)"))
            for b, o in fam["XB3_members"].items():
                rows.append(dict(o, contrast="XB3.member.%s (gated-fixed)" % b))
        for nm in ("XB1_loop", "XB1_stack"):
            if nm in fam:
                rows.append(dict(fam[nm], contrast=nm))
        for k, o in fam.get("desc", {}).items():
            rows.append(dict(o, contrast="desc." + k))
        for r in rows:
            w.writerow({k: (("%.6g" % v) if isinstance(v, float) else v) for k, v in r.items()})
    json.dump(_jsonable(cases), open(os.path.join(args.out, "ext1_cases.json"), "w"), indent=1, ensure_ascii=False)
    print("wrote", args.out, "| complete arms:", len(complete), "| verdicts:", json.dumps({h: x.get("verdict", x.get("line")) for h, x in v.items()}))
    return 0


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def cost_cmd(args):
    prices = load_prices(args.prices)
    tot = args.prior
    print("%-18s %10s %10s %10s" % ("run", "spent_cny", "list_cny", "embed_cny"))
    for d in sorted(glob.glob(os.path.join(args.runs, "*-s0"))):
        a = run_cost(d, prices, cache_free=True)
        b = run_cost(d, prices, cache_free=False)
        spent = a["chat"] + a["embed"]
        tot += spent
        print("%-18s %10.3f %10.3f %10.3f" % (os.path.basename(d), spent, b["chat"], b["embed"]))
    rc = stop_rule(tot, args.next_estimate)
    print("TOTAL spent CNY %.2f (prior %.2f included) | soft stop %.0f, hard cap %.0f | stop-rule code %d" % (tot, args.prior, STOP_SOFT, CAP_HARD, rc))
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["analyze", "cost"])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "ext1"))
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--prices", default=os.path.join(REPO, "exp", "prices.yaml"))
    ap.add_argument("--data-file", default=os.path.join(os.environ.get("NL2SQL_ROOT", os.path.expanduser("~/nl2sql_work")), "data", "spider_realistic", "spider-realistic.json"))
    ap.add_argument("--prior", type=float, default=0.0, help="spend already booked (the smoke) for the cost command")
    ap.add_argument("--next-estimate", type=float, default=0.0)
    a = ap.parse_args(argv)
    return analyze(a) if a.cmd == "analyze" else cost_cmd(a)


if __name__ == "__main__":
    sys.exit(main())
