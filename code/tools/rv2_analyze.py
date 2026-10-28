#!/usr/bin/env python3
"""RV2: post hoc analyses requested by the review round (SAP Amendment 10, section 23). Scope `posthoc-secondary`. No model or API call.

    worker machine:  PYTHONPATH=<code> python code/tools/rv2_analyze.py --runs $R/runs --data $R/data --results <host results copy> --exp <exp dir> --out <out dir> [--B 10000]

Reads only stored runs (`preds.jsonl`, `cands.jsonl`, `usage*.jsonl`), `results/robust/{items_ex.csv.gz,rescore.json}`, `results/e10/*`, `exp/{prices.yaml,frozen/route_coef_hr.json}`.
Writes `rv2_summary.json` (flat entries {"v", "ci", "item_ci", "n", "meaning"} keyed by the grammar of `rv2_keys.py`) and the csv files of SAP 23.9.
Imports `analysis_pipeline.{stats,runs,routing}` and `rescore_corrected` read-only (same bootstrap machinery as the primary analysis)."""
import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import rv2_keys as K  # noqa: E402
from analysis_pipeline import routing, runs as R, stats as S  # noqa: E402

SEED = S.SEED_BOOT
EXEC_FAIL = ("err_syntax", "err_identifier", "timeout")
EXEC_OK = ("ok_nonempty", "ok_empty")
MARGIN = 1.5
RATIO_NULL = 0.80
BK = K.BK
BIRD_DEV_N = 1534

RUNS = collections.OrderedDict()
RUNS[("plus", "zs")] = "MIN/M1/zs-s0"
RUNS[("plus", "rootonly")] = "MIN/M3/rootonly-s0"
RUNS[("plus", "gated")] = "MIN/M1/gated-s0"
RUNS[("plus", "blind")] = "MIN/M3/blind-s0"
RUNS[("plus", "fixed")] = "MIN/M1/fixed-s0"
RUNS[("plus", "route")] = "MIN/M1/gated-route-s0"
RUNS[("plus", "acsql")] = "MIN/M2/acsql-s0"
RUNS[("plus", "bonvote2")] = "MIN/M2/bonvote-s0"
RUNS[("plus", "bonvote3")] = "MIN/M2/bonvote3-s0"
RUNS[("plus", "macsql")] = "MIN/M2/macsql-s0"
RUNS[("plus", "sdebug")] = "MIN/M2/sdebug-s0"
for _b, _t in (("flash", "flash"), ("27b", "27b")):
    for _a in ("zs", "rootonly", "gated", "blind", "fixed"):
        RUNS[(_b, _a)] = "MIN/M5/%s-%s-s0" % (_a, _t)
SEED_RUNS = {("plus", "gated", 1): "MIN/M1/gated-s101", ("plus", "gated", 2): "MIN/M1/gated-s202",
             ("plus", "zs", 1): "MIN/M1/zs-s101", ("plus", "zs", 2): "MIN/M1/zs-s202"}
PF_REFS = collections.OrderedDict([("zs", "MIN/M1/zs-s0"), ("full_fixed", "MIN/M1/fixed-s0"), ("full_gated", "MIN/M1/gated-s0"), ("route_ca", "MIN/M1/gated-route-s0")])
PF_ARMS = collections.OrderedDict([("pf_n0", "PF/pf_n0-s0"), ("pf_n250_q100", "PF/pf_n250_q100-s0"), ("pf_n1000_q100", "PF/pf_n1000_q100-s0"), ("pf_n1000_q085", "PF/pf_n1000_q085-s0")])

OUT = collections.OrderedDict()                 # key -> entry
KINDS = dict(K.expected_keys())
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print("[rv2] " + s, flush=True)


def _f(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [_f(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def put(key, v, ci=None, item_ci=None, n=None, meaning=""):
    if key not in KINDS:
        raise KeyError("key not in the grammar: " + key)
    if key in OUT:
        raise KeyError("duplicate key " + key)
    OUT[key] = {"v": _f(v), "kind": KINDS[key], "ci": _f(ci), "item_ci": _f(item_ci), "n": _f(n), "meaning": meaning}


# ------------------------------------------------------------------------------------------------ loading
class Run(object):
    def __init__(self, root, rel):
        self.rel = rel
        self.path = os.path.join(root, rel)
        self.rows = {}
        for r in R.jl(os.path.join(self.path, "preds.jsonl")):
            self.rows[r["qid"]] = r
        self.cands = collections.defaultdict(list)
        cp = os.path.join(self.path, "cands.jsonl")
        if os.path.exists(cp):
            for c in R.jl(cp):
                self.cands[c["qid"]].append(c)
            for q in self.cands:
                self.cands[q].sort(key=lambda c: c.get("cand_idx", 0))
        h = hashlib.sha256()
        with open(os.path.join(self.path, "preds.jsonl"), "rb") as f:
            for ch in iter(lambda: f.read(1 << 20), b""):
                h.update(ch)
        self.sha = h.hexdigest()
        self._usage = None

    @property
    def usage(self):
        if self._usage is None:
            u = []
            for fn in sorted(os.listdir(self.path)):
                if fn.startswith("usage") and fn.endswith(".jsonl"):
                    u.extend(R.jl(os.path.join(self.path, fn)))
            self._usage = u
        return self._usage

    def root(self, q):
        cs = self.cands.get(q)
        return cs[0] if cs else None


_CB = {}


def CB(dbs, B):
    key = (len(dbs), hashlib.md5(",".join(map(str, dbs)).encode()).hexdigest(), B)
    if key not in _CB:
        _CB[key] = S.ClusterBoot(dbs, B)
    return _CB[key]


_IDX = {}


def item_idx(n, B):
    if (n, B) not in _IDX:
        if len(_IDX) > 6:
            _IDX.clear()
        _IDX[(n, B)] = np.random.RandomState(SEED).randint(0, n, size=(B, n))        # identical draw to stats.item_boot_means
    return _IDX[(n, B)]


def item_means(v, B):
    v = np.asarray(v, float)
    return v[item_idx(len(v), B)].mean(1)


def item_ratio(num, den, B):
    num, den = np.asarray(num, float), np.asarray(den, float)
    ix = item_idx(len(num), B)
    return num[ix].sum(1) / np.maximum(den[ix].sum(1), 1e-12)


def tcrit(df):
    from scipy import stats as sps
    return float(sps.t.ppf(0.975, df))


def jack_ci(est, loo, g):
    """delete-one-database jackknife SE with t(g - 1) quantile (descriptive, SAP 23.1 f / 23.11)."""
    if g < 3 or not loo:
        return [None, None]
    v = np.asarray(loo, float)
    se = math.sqrt((g - 1.0) / g * float(((v - v.mean()) ** 2).sum()))
    t = tcrit(g - 1)
    return [est - t * se, est + t * se]


def pack(est, bt, ib, loo, g, n):
    ci = list(S.pctl_ci(bt))
    ici = list(S.pctl_ci(ib))
    vb, vi = float(np.var(bt, ddof=1)), float(np.var(ib, ddof=1))
    return {"est": float(est), "ci": ci, "item_ci": ici, "t_ci": jack_ci(est, loo, g), "deff": (vb / vi) if vi > 0 else None,
            "loo_min": min(loo) if loo else None, "loo_max": max(loo) if loo else None, "g": int(g), "n": int(n), "bt": bt, "ib": ib}


def est_mean(d, dbs, B, scale=100.0):
    d = np.asarray(d, float)
    boot = CB(dbs, B)
    loo = [scale * v for _, v in boot.loo(lambda idx: d[idx].mean())]
    return pack(scale * d.mean(), scale * boot.means(d), scale * item_means(d, B), loo, boot.k, len(d))


def est_ratio(num, den, dbs, B):
    num, den = np.asarray(num, float), np.asarray(den, float)
    boot = CB(dbs, B)
    loo = [v for _, v in boot.loo(lambda idx: num[idx].sum() / max(den[idx].sum(), 1e-12))]
    return pack(num.sum() / max(den.sum(), 1e-12), boot.ratio(num, den), item_ratio(num, den, B), loo, boot.k, len(num))


def est_ratio_pooled(nums, dens, dbs, B):
    """mean over backbones of the ratios of sums (P1' (b) estimator), common resample."""
    boot = CB(dbs, B)
    ks = list(nums)
    bt = np.mean([boot.ratio(nums[k], dens[k]) for k in ks], axis=0)
    ib = np.mean([item_ratio(nums[k], dens[k], B) for k in ks], axis=0)
    est = float(np.mean([nums[k].sum() / max(dens[k].sum(), 1e-12) for k in ks]))
    loo = [v for _, v in boot.loo(lambda idx: np.mean([nums[k][idx].sum() / max(dens[k][idx].sum(), 1e-12) for k in ks]))]
    return pack(est, bt, ib, loo, boot.k, len(dbs))


def est_stat(fn, dbs, B, bt, ib, scale=1.0):
    """generic statistic fn(idx) with given bootstrap arrays (already scaled)."""
    boot = CB(dbs, B)
    loo = [scale * v for _, v in boot.loo(fn)]
    return pack(scale * fn(np.arange(len(dbs))), bt, ib, loo, boot.k, len(dbs))


def mc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    x = int(((a == 1) & (b == 0)).sum())
    y = int(((a == 0) & (b == 1)).sum())
    return x, y, float(S.mcnemar_exact(x, y)["p_two"])


# ------------------------------------------------------------------------------------------------ reading classes (SAP 23.3)
def cls_ni(lo, hi, est=None):
    return "non-inferior" if lo > -MARGIN else "not shown"


def cls_ratio(lo, hi, est=None):
    return "saving shown" if hi < RATIO_NULL else "not shown"


def cls_ratio1(lo, hi, est=None):
    return "below 1" if hi < 1.0 else ("above 1" if lo > 1.0 else "contains 1")


def cls_sup(lo, hi, est=None):
    return "positive" if lo > 0 else ("negative" if hi < 0 else "contains 0")


def cls_band(lo, hi, est):
    if lo > 0:
        return "improves" if est >= MARGIN else "small improvement"
    if hi < 0:
        return "harms" if est <= -MARGIN else "small decline"
    if lo > -MARGIN and hi < MARGIN:
        return "no difference larger than 1.5 pp"
    return "inconclusive"


def rule_of(i):
    if i.startswith("P1p_ratio") or i == "XB3_ratio":
        return "RATIO(0.80)", cls_ratio
    if i.startswith("P1p.") or i in ("XB3", "XB3_plus", "XB3_flash", "XB3_27b"):
        return "NI(1.5)", cls_ni
    if i in ("PF3b", "PF3b_q100"):
        return "RATIO(1)", cls_ratio1
    if i.startswith("PF"):
        return "BAND(1.5)", cls_band
    if i.startswith("RES."):
        _, s, cell, gold = i.split(".")
        if gold == "shift":
            return "SUP(0)", cls_sup
        if cell in ("C1_plus", "C1_flash", "C1_27b", "C1_pooled", "C2_plus", "C4"):
            return "BAND(1.5)", cls_band
        return "SUP(0)", cls_sup
    return "SUP(0)", cls_sup


def put_dual(i, e, mcn=None):
    """e: pack() dict; mcn: (b, c, p) or None."""
    rname, fn = rule_of(i)
    cc, ci_, ct = (fn(e["ci"][0], e["ci"][1], e["est"]), fn(e["item_ci"][0], e["item_ci"][1], e["est"]),
                   (fn(e["t_ci"][0], e["t_ci"][1], e["est"]) if e["t_ci"][0] is not None else "n/a"))
    pre = "dual.%s." % i
    put(pre + "est", e["est"], n=e["n"])
    put(pre + "n", e["n"]); put(pre + "g", e["g"])
    put(pre + "ci", e["ci"]); put(pre + "item_ci", e["item_ci"]); put(pre + "t_ci", e["t_ci"] if e["t_ci"][0] is not None else "n/a")
    if mcn is None:
        put(pre + "p_mcnemar", "n/a"); put(pre + "b", "n/a"); put(pre + "c", "n/a")
    else:
        put(pre + "b", mcn[0]); put(pre + "c", mcn[1]); put(pre + "p_mcnemar", mcn[2])
    put(pre + "deff", e["deff"] if e["deff"] is not None else "n/a")
    put(pre + "loo_min", e["loo_min"] if e["loo_min"] is not None else "n/a"); put(pre + "loo_max", e["loo_max"] if e["loo_max"] is not None else "n/a")
    put(pre + "rule", rname)
    put(pre + "class_cluster", cc); put(pre + "class_item", ci_); put(pre + "class_t", ct)
    put(pre + "reading_changes", "yes" if cc != ci_ else "no")
    put(pre + "reading_changes_t", "yes" if (ct != "n/a" and ct != cc) else "no")
    DUAL_ROWS.append({"id": i, "rule": rname, "n": e["n"], "g": e["g"], "est": e["est"], "ci_lo": e["ci"][0], "ci_hi": e["ci"][1], "item_lo": e["item_ci"][0],
                      "item_hi": e["item_ci"][1], "t_lo": e["t_ci"][0], "t_hi": e["t_ci"][1], "p_mcnemar": (mcn[2] if mcn else ""), "b": (mcn[0] if mcn else ""),
                      "c": (mcn[1] if mcn else ""), "deff": e["deff"], "loo_min": e["loo_min"], "loo_max": e["loo_max"], "class_cluster": cc, "class_item": ci_,
                      "class_t": ct, "reading_changes": "yes" if cc != ci_ else "no"})


DUAL_ROWS = []
CSV_ROWS = collections.defaultdict(list)


# ------------------------------------------------------------------------------------------------ main frame
class Frame(object):
    def __init__(self, runs_root, B):
        self.B = B
        self.root = runs_root
        self.runs = {}
        for k, rel in RUNS.items():
            if os.path.exists(os.path.join(runs_root, rel, "preds.jsonl")):
                self.runs[k] = Run(runs_root, rel)
            else:
                log("MISSING run", k, rel)
        g = self.runs[("plus", "gated")]
        self.qids = sorted(g.rows)
        self.dbs = np.array([g.rows[q].get("db_id") or q.split(":")[1] for q in self.qids])
        self.n = len(self.qids)
        self.ex = {}
        self.null = {}
        for k, r in self.runs.items():
            v = np.full(self.n, -1, int)
            for i, q in enumerate(self.qids):
                row = r.rows.get(q)
                if row is not None:
                    x = R.ex_value(row)
                    if x is not None:
                        v[i] = x
            self.ex[k] = v
            self.null[k] = int((v < 0).sum())
        for k, nn in self.null.items():
            if nn:
                log("ex null/missing in", k, nn)

    def pair_mask(self, *keys):
        m = np.ones(self.n, bool)
        for k in keys:
            m &= self.ex[k] >= 0
        return m

    def col(self, k, field):
        r = self.runs[k]
        return np.array([float(r.rows[q].get(field) or 0.0) if q in r.rows else 0.0 for q in self.qids])


def diff_est(F, ka, kb, B):
    m = F.pair_mask(ka, kb)
    a, b = F.ex[ka][m], F.ex[kb][m]
    e = est_mean(a - b, F.dbs[m], B)
    e["mc"] = mc(a, b)
    return e, a, b, m


def pooled_est(F, pairs, B):
    """pairs: list of (ka, kb); mean over pairs of the differences, items valid in all."""
    m = F.pair_mask(*[x for p in pairs for x in p])
    d = np.mean([F.ex[a][m] - F.ex[b][m] for a, b in pairs], axis=0)
    return est_mean(d, F.dbs[m], B), d, m


# ------------------------------------------------------------------------------------------------ analysis 1: loop effect
def root_ex(F, key):
    """EX of the root candidate (cand_idx 0) per item; -1 where unavailable."""
    r = F.runs[key]
    v = np.full(F.n, -1, int)
    for i, q in enumerate(F.qids):
        c = r.root(q)
        if c is not None and c.get("ex") is not None:
            v[i] = int(c["ex"])
    return v


def analysis_loop(F, B):
    log("analysis 1: loop effect")
    per = {}                                           # b -> dict of vectors
    for b in BK:
        arms = {a: F.ex[(b, a)] for a in K.ARMS}
        rg, rf = root_ex(F, (b, "gated")), root_ex(F, (b, "fixed"))
        per[b] = {"arms": arms, "rootcand": {"gated": rg, "fixed": rf}}
    keys_needed = [(b, a) for b in BK for a in K.ARMS]
    m_all = F.pair_mask(*keys_needed)
    for b in BK:
        m_all &= (per[b]["rootcand"]["gated"] >= 0) & (per[b]["rootcand"]["fixed"] >= 0)
    log("loop: items valid in all arms and root candidates of all backbones:", int(m_all.sum()), "of", F.n)
    # per-item effect vectors (on the common mask m_all so that pooled = mean of the three and every backbone uses the same items)
    eff = {}
    for b in BK:
        a = per[b]["arms"]
        for arm in ("gated", "fixed"):
            eff[(b, "in", arm)] = (a[arm] - per[b]["rootcand"][arm])[m_all]
            eff[(b, "btw", arm)] = (a[arm] - a["rootonly"])[m_all]
        eff[(b, "stack")] = (a["rootonly"] - a["zs"])[m_all]
    dbs = F.dbs[m_all]
    boot = CB(dbs, B)
    res = {}
    for b in PB_ALL():
        for d in ("in", "btw"):
            for arm in ("gated", "fixed"):
                v = np.mean([eff[(x, d, arm)] for x in (BK if b == "pooled" else (b,))], axis=0)
                e = est_mean(v, dbs, B)
                res[(b, d, arm)] = e
                e["v"] = v
                k = "loop.%s.eff_%s.%s" % (b, d, arm)
                put(k, e["est"], ci=e["ci"], item_ci=e["item_ci"], n=e["n"])
                if b != "pooled":
                    if d == "in":
                        hb = int(((per[b]["arms"][arm][m_all] == 1) & (per[b]["rootcand"][arm][m_all] == 0)).sum())
                        hc = int(((per[b]["arms"][arm][m_all] == 0) & (per[b]["rootcand"][arm][m_all] == 1)).sum())
                    else:
                        hb = int(((per[b]["arms"][arm][m_all] == 1) & (per[b]["arms"]["rootonly"][m_all] == 0)).sum())
                        hc = int(((per[b]["arms"][arm][m_all] == 0) & (per[b]["arms"]["rootonly"][m_all] == 1)).sum())
                    put(k + ".b", hb); put(k + ".c", hc); put(k + ".p", float(S.mcnemar_exact(hb, hc)["p_two"]))
                CSV_ROWS["loop"].append({"backbone": b, "definition": d, "arm": arm, "n": e["n"], "est_pp": e["est"], "ci_lo": e["ci"][0], "ci_hi": e["ci"][1],
                                         "item_lo": e["item_ci"][0], "item_hi": e["item_ci"][1], "t_lo": e["t_ci"][0], "t_hi": e["t_ci"][1], "deff": e["deff"]})
        # levels
        for arm in K.ARMS:
            if b == "pooled":
                v = np.mean([per[x]["arms"][arm][m_all] for x in BK], axis=0)
            else:
                v = per[b]["arms"][arm][m_all].astype(float)
            ee = est_mean(v, dbs, B, scale=1.0)
            put("loop.%s.ex.%s" % (b, arm), ee["est"], ci=ee["ci"], item_ci=ee["item_ci"], n=ee["n"])
        for arm in ("gated", "fixed"):
            if b == "pooled":
                v = np.mean([per[x]["rootcand"][arm][m_all] for x in BK], axis=0)
            else:
                v = per[b]["rootcand"][arm][m_all].astype(float)
            ee = est_mean(v, dbs, B, scale=1.0)
            put("loop.%s.rootcand.%s" % (b, arm), ee["est"], ci=ee["ci"], item_ci=ee["item_ci"], n=ee["n"])
        v = np.mean([eff[(x, "stack")] for x in (BK if b == "pooled" else (b,))], axis=0)
        e = est_mean(v, dbs, B)
        res[(b, "stack")] = e
        put("loop.%s.stack" % b, e["est"], ci=e["ci"], item_ci=e["item_ci"], n=e["n"])
        for arm in ("gated", "fixed"):
            put("loop.%s.loop_btw_minus_in.%s" % (b, arm), res[(b, "btw", arm)]["est"] - res[(b, "in", arm)]["est"])
        for d in ("in", "btw"):
            eg, ef = res[(b, d, "gated")], res[(b, d, "fixed")]
            defined = bool(ef["ci"][0] > 0 or ef["ci"][1] < 0)
            put("loop.%s.share_%s.defined" % (b, d), "yes" if defined else "no")
            share = eg["est"] / ef["est"] if ef["est"] != 0 else None
            if share is None:
                put("loop.%s.share_%s" % (b, d), "n/a")
            elif defined:
                sb = eg["bt"] / np.where(ef["bt"] == 0, np.nan, ef["bt"])
                si = eg["ib"] / np.where(ef["ib"] == 0, np.nan, ef["ib"])
                put("loop.%s.share_%s" % (b, d), share, ci=list(S.pctl_ci(sb[np.isfinite(sb)])), item_ci=list(S.pctl_ci(si[np.isfinite(si)])))
            else:
                put("loop.%s.share_%s" % (b, d), share)                       # point value only, no interval (SAP 23.2 item 3)
            put("loop.%s.margin_over_eff_%s" % (b, d), (MARGIN / ef["est"]) if ef["est"] > 0 else "n/a")
            for arm in ("gated", "fixed"):
                e = res[(b, d, arm)]
                ok = e["ci"][0] > 0 and e["item_ci"][0] > 0
                put("loop.%s.claim_%s.%s" % (b, d, arm), "the loop improves accuracy" if ok else "no loop effect detected")
    # seeds
    for arm in ("gated", "zs"):
        vals = []
        for rep in (0, 1, 2):
            rel = RUNS[("plus", arm)] if rep == 0 else SEED_RUNS[("plus", arm, rep)]
            p = os.path.join(F.root, rel, "preds.jsonl")
            if os.path.exists(p):
                rows = {r["qid"]: r for r in R.jl(p)}
                ex = [R.ex_value(r) for r in rows.values()]
                ex = [x for x in ex if x is not None]
                vals.append(100.0 * float(np.mean(ex)))
        put("loop.seedsd.plus." + arm, float(np.std(vals, ddof=1)) if len(vals) >= 2 else "n/a")
    return res, per, m_all


def PB_ALL():
    return K.PB


# failing roots -------------------------------------------------------------------------------------------------------------------------------
def analysis_roots(F, B):
    log("analysis 1.6: failing roots of fixed vs gated")
    cause_all = []
    out_rows = []
    for b in BK:
        g, f = F.runs[(b, "gated")], F.runs[(b, "fixed")]
        qs = [q for q in F.qids if g.root(q) is not None and f.root(q) is not None]
        fg = {q for q in qs if g.root(q).get("status") in EXEC_FAIL}
        ff = {q for q in qs if f.root(q).get("status") in EXEC_FAIL}
        both, og, of = fg & ff, fg - ff, ff - fg
        diff = sorted(og | of)
        same_sql = sum(1 for q in diff if (g.root(q).get("sql") or "").strip() == (f.root(q).get("sql") or "").strip())
        to = sum(1 for q in diff if "timeout" in (g.root(q).get("status"), f.root(q).get("status")))
        exd = sum(1 for q in diff if g.root(q).get("exemplar_ids") != f.root(q).get("exemplar_ids"))
        fexd = sum(1 for q in diff if g.rows[q].get("ex") != f.rows[q].get("ex"))
        sql_diff_all = sum(1 for q in qs if (g.root(q).get("sql") or "").strip() != (f.root(q).get("sql") or "").strip())
        st_diff_all = collections.Counter((g.root(q).get("status"), f.root(q).get("status")) for q in qs if g.root(q).get("status") != f.root(q).get("status"))
        norm = lambda t: " ".join((t or "").split())
        ws_only = sum(1 for q in qs if (g.root(q).get("sql") or "").strip() != (f.root(q).get("sql") or "").strip() and norm(g.root(q).get("sql")) == norm(f.root(q).get("sql")))
        ex_all = sum(1 for q in qs if g.root(q).get("exemplar_ids") != f.root(q).get("exemplar_ids"))
        ctx_all = sum(1 for q in qs if g.rows[q].get("ctx_tokens") != f.rows[q].get("ctx_tokens"))
        gen_g = [x for x in g.usage if x.get("kind") == "chat" and x.get("tag") == "gen"]
        fkeys = {x.get("key") for x in f.usage if x.get("kind") == "chat" and x.get("tag") == "gen"}
        unc = [x for x in gen_g if not x.get("cached")]
        unc_in_f = sum(1 for x in unc if x.get("key") in fkeys)
        prov = ("gated run: %d of %d generation requests were live (not cache replays), %d of those have a request key that also occurs in the fixed run (a true cache miss); "
                "root exemplar lists differ in %d items, context token counts in %d items, root SQL text in %d items (%d of them by whitespace only)"
                % (len(unc), len(gen_g), unc_in_f, ex_all, ctx_all, sql_diff_all, ws_only))
        put("roots.%s.n_items" % b, len(qs)); put("roots.%s.n_fail.gated" % b, len(fg)); put("roots.%s.n_fail.fixed" % b, len(ff))
        put("roots.%s.both" % b, len(both)); put("roots.%s.only_gated" % b, len(og)); put("roots.%s.only_fixed" % b, len(of))
        put("roots.%s.diff.n" % b, len(diff)); put("roots.%s.diff.same_sql" % b, same_sql); put("roots.%s.diff.diff_sql" % b, len(diff) - same_sql)
        put("roots.%s.diff.timeout_involved" % b, to); put("roots.%s.diff.exemplars_differ" % b, exd); put("roots.%s.diff.final_ex_differs" % b, fexd)
        ex_s = [g.rows[q].get("exec_s") for q in diff if g.root(q).get("status") == "timeout"]
        if not diff:
            cause = "no difference: the failing roots of the two runs are the same %d items; %s" % (len(both), prov)
        else:
            cause = ("%d items fail in only one run (gated only %d, fixed only %d); of these %d have an identical root SQL text and %d a different one; "
                     "%d involve a root with status timeout; status pairs that differ: %s; %s"
                     % (len(diff), len(og), len(of), same_sql, len(diff) - same_sql, to,
                        "; ".join("%s vs %s: %d" % (a, c, n) for (a, c), n in sorted(st_diff_all.items(), key=lambda t: str(t))), prov))
        put("roots.%s.cause" % b, cause)
        cause_all.append("%s: %s" % (b, cause))
        out_rows.append({"backbone": b, "n_fail_gated": len(fg), "n_fail_fixed": len(ff), "both": len(both), "only_gated": len(og), "only_fixed": len(of),
                         "diff_same_sql": same_sql, "diff_timeout": to, "qids_differ": ";".join(diff)})
    plus_only = [r for r in out_rows if r["backbone"] == "plus"][0]
    put("roots.cause", ("Two mechanisms, stated as found (no re-run). (1) Plus: the root SQL text is the same cache replay in both runs; a statement that finished in the fixed run's "
                        "evaluation exceeded the execution time limit in the gated run's evaluation (status timeout); the gated loop treats a timeout as a failing root and revises, the fixed loop "
                        "saw a result. Plus: gated %d vs fixed %d failing roots, %d items differ, %d of them with identical SQL; execution time depends on machine load, so this part is not "
                        "reproducible to the item. (2) Flash and 27b: the gated run issued live generation requests whose request keys are absent from the fixed run (the prompt of the root "
                        "request was not identical to the fixed run's; why it differed is not determined by the stored fields, a candidate is non-deterministic rerank output), so the root SQL was regenerated and differs in some items. Per backbone: " % (
                            plus_only["n_fail_gated"], plus_only["n_fail_fixed"], plus_only["only_gated"] + plus_only["only_fixed"], plus_only["diff_same_sql"]) + " | ".join(cause_all)))
    CSV_ROWS["roots"] = out_rows


# ------------------------------------------------------------------------------------------------ analysis 3: dual intervals
def analysis_dual_dev(F, B):
    log("analysis 3a: P1', W1")
    # P1' EX and calls ratio
    pairs = [((b, "gated"), (b, "fixed")) for b in BK]
    e, d, m = pooled_est(F, pairs, B)
    put_dual("P1p.pooled", e)
    nums = {b: F.col((b, "gated"), "calls")[m] for b in BK}
    dens = {b: F.col((b, "fixed"), "calls")[m] for b in BK}
    put_dual("P1p_ratio.pooled", est_ratio_pooled(nums, dens, F.dbs[m], B))
    for b in BK:
        e, a, bb, mm = diff_est(F, (b, "gated"), (b, "fixed"), B)
        put_dual("P1p." + b, e, e["mc"])
        put_dual("P1p_ratio." + b, est_ratio(F.col((b, "gated"), "calls")[mm], F.col((b, "fixed"), "calls")[mm], F.dbs[mm], B))
    e, d, m = pooled_est(F, [((b, "gated"), (b, "zs")) for b in ("flash", "27b")], B)
    put_dual("W1.pooled", e)
    for b in ("flash", "27b", "plus"):
        e, a, bb, mm = diff_est(F, (b, "gated"), (b, "zs"), B)
        put_dual("W1." + b, e, e["mc"])


def repair_pop(g, blind):
    """(qid, repaired by gated, repaired by blind) for the failing roots of g (EXEC_FAIL)."""
    out = []
    for q, cs in g.cands.items():
        if cs[0].get("status") in EXEC_FAIL and q in g.rows:
            a = R.ex_value(g.rows[q])
            bl = R.ex_value(blind.rows[q]) if q in blind.rows else None
            out.append((q, g.rows[q].get("db_id") or q.split(":")[1], 1 if a == 1 else 0, -1 if bl is None else bl))
    return sorted(out)


def analysis_p5(F, B):
    log("analysis 3b: P5'")
    pool = []
    for b in BK:
        pool.extend([(b,) + x for x in repair_pop(F.runs[(b, "gated")], F.runs[(b, "blind")])])
    ok = [x for x in pool if x[4] >= 0]
    rep = np.array([x[3] for x in ok]); brep = np.array([x[4] for x in ok]); dbs_r = np.array([x[2] for x in ok])
    dd = (rep - brep).astype(float)
    n_R = len(ok)
    log("P5': n_R =", n_R, "(pooled population", len(pool), ")")
    bdev = CB(F.dbs, B)                                          # the 11 dev databases are the clusters (as weak.run_p5prime)
    num = np.array([dd[dbs_r == u].sum() for u in bdev.u]); den = np.array([float((dbs_r == u).sum()) for u in bdev.u])
    bt = 100.0 * num[bdev.draws].sum(1) / np.maximum(den[bdev.draws].sum(1), 1.0)
    loo = []
    for j in range(len(bdev.u)):
        keep = np.arange(len(bdev.u)) != j
        loo.append(100.0 * num[keep].sum() / max(den[keep].sum(), 1.0))
    est = 100.0 * dd.mean()
    ib = 100.0 * item_means(dd, B)
    e = pack(est, bt, ib, loo, bdev.k, n_R)
    put_dual("P5p", e, mc(rep, brep))
    return n_R


def route_vectors(F, run, coef, k=4):
    from analysis_pipeline import primary
    rows = [run.rows[q] for q in F.qids]
    v3, ca = primary.route_scores(rows, coef, k)
    y = np.array([R.ex_value(r) if R.ex_value(r) is not None else -1 for r in rows])
    ok = np.isfinite(v3) & np.isfinite(ca) & (y >= 0)
    return v3, ca, y, ok


def item_boot_metrics(scores, y, B, chunk=500):
    """item-level paired bootstrap of routing.metric_vector for every score; same index draws for all scores."""
    n = len(y)
    out = {k: np.empty((B, 5)) for k in scores}
    ix = item_idx(n, B)
    for b in range(B):
        idx = ix[b]
        yy = y[idx]
        for k, s in scores.items():
            out[k][b] = routing.metric_vector(s[idx], yy)
    return out


def metric_diff_blocks(ca, v3, y, dbs, B):
    """-> dict name -> pack() for AUROC / AURC / acc60 (route_ca - v3); acc60 in pp."""
    boot = CB(dbs, B)
    sc = {"route_ca": ca, "v3": v3}
    bc = routing.boot_metrics(sc, y, boot)
    bi = item_boot_metrics(sc, y, B)
    out = {}
    for nm, j, scale in (("auroc", 0, 1.0), ("aurc", 1, 1.0), ("acc60", 3, 100.0)):
        def fn(idx, j=j):
            return routing.metric_vector(ca[idx], y[idx])[j] - routing.metric_vector(v3[idx], y[idx])[j]
        bt = scale * (bc["route_ca"][:, j] - bc["v3"][:, j])
        ib = scale * (bi["route_ca"][:, j] - bi["v3"][:, j])
        out[nm] = est_stat(fn, dbs, B, bt, ib, scale=scale)
    return out, bc, bi


def analysis_p3(F, B, coef):
    log("analysis 3c: P3")
    run = F.runs[("plus", "route")]
    v3, ca, y, ok = route_vectors(F, run, coef)
    log("P3: n =", int(ok.sum()))
    blocks, bc, bi = metric_diff_blocks(ca[ok], v3[ok], y[ok], F.dbs[ok], B)
    put_dual("P3", blocks["auroc"]); put_dual("P3_aurc", blocks["aurc"]); put_dual("P3_acc60", blocks["acc60"])
    return v3, ca, y, ok


# ------------------------------------------------------------------------------------------------ analysis 3d: RESCORE cells
def load_items(path):
    out = collections.defaultdict(dict)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["kind"], r["label"])][r["qid"]] = r
    return out


def numv(x):
    return None if x in ("", None) else float(x)


def analysis_rescore(F, B, args, coef):
    import rescore_corrected as RC
    log("analysis 3d: RESCORE cells")
    items = load_items(os.path.join(args.results, "robust", "items_ex.csv.gz"))
    rs = json.load(open(os.path.join(args.results, "robust", "rescore.json")))
    disc = set(rs["variants"]["G1"]["db_discordant"])
    dev_by_id, var = RC.load_variants(args.data)
    st = RC.build_sets(dev_by_id, var["G1"], disc)
    sets = {"S_all": st["all"], "S_same": st["same"]}
    log("sets: S_all", len(sets["S_all"]), "S_same", len(sets["S_same"]))
    lab = {"plus": "plus", "flash": "flash", "27b": "q27b"}

    def L(b, a):
        return "%s.%s" % (lab[b], a)

    cells = {}
    for sname, qs in sets.items():
        def usable(labels, kind="final"):
            return [q for q in qs if all((items[(kind, l)].get(q) is not None and items[(kind, l)][q]["usable"] == "1") for l in labels)]

        def vec(label, g, q, kind="final"):
            return np.array([numv(items[(kind, label)][x][g]) for x in q], float)

        def binary_cell(cid, pairs):
            labels = sorted(set(x for p in pairs for x in p))
            q = usable(labels)
            dbs = np.array([x.split(":")[1] for x in q])
            vo = np.mean([vec(a, "orig", q) - vec(b, "orig", q) for a, b in pairs], axis=0)
            vc = np.mean([vec(a, "G1", q) - vec(b, "G1", q) for a, b in pairs], axis=0)
            eo, ec, es = est_mean(vo, dbs, B), est_mean(vc, dbs, B), est_mean(vc - vo, dbs, B)
            mco = mcc = None
            if len(pairs) == 1:
                a, b = pairs[0]
                mco = mc(vec(a, "orig", q), vec(b, "orig", q)); mcc = mc(vec(a, "G1", q), vec(b, "G1", q))
            put_dual("RES.%s.%s.orig" % (sname, cid), eo, mco); put_dual("RES.%s.%s.corr" % (sname, cid), ec, mcc); put_dual("RES.%s.%s.shift" % (sname, cid), es)

        binary_cell("C1_plus", [(L("plus", "gated"), L("plus", "fixed"))])
        binary_cell("C1_flash", [(L("flash", "gated"), L("flash", "fixed"))])
        binary_cell("C1_27b", [(L("27b", "gated"), L("27b", "fixed"))])
        binary_cell("C1_pooled", [(L(b, "gated"), L(b, "fixed")) for b in BK])
        binary_cell("C2_flash", [(L("flash", "gated"), L("flash", "zs"))])
        binary_cell("C2_27b", [(L("27b", "gated"), L("27b", "zs"))])
        binary_cell("C2_pooled", [(L(b, "gated"), L(b, "zs")) for b in ("flash", "27b")])
        binary_cell("C2_plus", [(L("plus", "gated"), L("plus", "zs"))])
        # C3 routing
        rl = "plus.route"
        q = [x for x in usable([rl]) if items[("final", rl)][x]["v3"] != "" and items[("final", rl)][x]["route_ca"] != ""]
        dbs = np.array([x.split(":")[1] for x in q])
        v3 = np.array([float(items[("final", rl)][x]["v3"]) for x in q]); ca = np.array([float(items[("final", rl)][x]["route_ca"]) for x in q])
        yo, yc = vec(rl, "orig", q).astype(int), vec(rl, "G1", q).astype(int)
        bo, bco, bio = metric_diff_blocks(ca, v3, yo, dbs, B)
        bc_, bcc, bic = metric_diff_blocks(ca, v3, yc, dbs, B)
        boot = CB(dbs, B)
        for nm, j, scale in (("auroc", 0, 1.0), ("aurc", 1, 1.0), ("acc60", 3, 100.0)):
            def fshift(idx, j=j):
                return (routing.metric_vector(ca[idx], yc[idx])[j] - routing.metric_vector(v3[idx], yc[idx])[j]) - (routing.metric_vector(ca[idx], yo[idx])[j] - routing.metric_vector(v3[idx], yo[idx])[j])
            btS = scale * ((bcc["route_ca"][:, j] - bcc["v3"][:, j]) - (bco["route_ca"][:, j] - bco["v3"][:, j]))
            ibS = scale * ((bic["route_ca"][:, j] - bic["v3"][:, j]) - (bio["route_ca"][:, j] - bio["v3"][:, j]))
            es = est_stat(fshift, dbs, B, btS, ibS, scale=scale)
            put_dual("RES.%s.C3_%s.orig" % (sname, nm), bo[nm]); put_dual("RES.%s.C3_%s.corr" % (sname, nm), bc_[nm]); put_dual("RES.%s.C3_%s.shift" % (sname, nm), es)
        # C4: net effect of executable-root revisions of plus.fixed
        qr = []
        for x in qs:
            f_, r_ = items[("final", "plus.fixed")].get(x), items[("root", "plus.fixed")].get(x)
            if f_ is None or r_ is None or f_["usable"] != "1" or r_["usable"] != "1":
                continue
            if f_["G1"] == "" or r_["G1"] == "" or f_["orig"] == "" or r_["orig"] == "":
                continue
            qr.append(x)
        dbs = np.array([x.split(":")[1] for x in qr])
        vv = {}
        cnt = {}
        for g in ("orig", "G1"):
            ve = []
            h = u = 0
            for x in qr:
                f_, r_ = items[("final", "plus.fixed")][x], items[("root", "plus.fixed")][x]
                changed = f_["sha"] != r_["sha"]
                rexec = r_["root_status"] in RC.EXEC_STATUS
                dlt = int(float(f_[g])) - int(float(r_[g]))
                ve.append(dlt if (changed and rexec) else 0)
                if changed and rexec:
                    h += int(dlt > 0); u += int(dlt < 0)
            vv[g] = np.array(ve, float); cnt[g] = (h, u)
        eo, ec, es = est_mean(vv["orig"], dbs, B), est_mean(vv["G1"], dbs, B), est_mean(vv["G1"] - vv["orig"], dbs, B)
        put_dual("RES.%s.C4.orig" % sname, eo, (cnt["orig"][0], cnt["orig"][1], float(S.mcnemar_exact(*cnt["orig"])["p_two"])))
        put_dual("RES.%s.C4.corr" % sname, ec, (cnt["G1"][0], cnt["G1"][1], float(S.mcnemar_exact(*cnt["G1"])["p_two"])))
        put_dual("RES.%s.C4.shift" % sname, es)
    return items, sets, RC


# ------------------------------------------------------------------------------------------------ analysis 3e/f: EXT1 and PF
def analysis_ext1(args, B):
    import ext1_analyze as X
    log("analysis 3e: EXT1")
    arms = X.load_all(os.path.join(args.runs, "EXT1"))
    bad = [k for k, v in arms.items() if v is None or not v.complete]
    if bad:
        log("EXT1 arms incomplete", bad)
    al = {}
    for bk in X.BACKBONES:
        al[bk] = X.align({a: arms[(bk, a)] for a in X.ARMS})
    qids = sorted(set.intersection(*[set(al[b]["qids"]) for b in al]))
    sel = {}
    for bk in al:
        pos = {q: i for i, q in enumerate(al[bk]["qids"])}
        sel[bk] = np.array([pos[q] for q in qids])
    dbs = al["plus"]["dbs"][sel["plus"]]
    E = lambda b, a: al[b]["ex"][a][sel[b]]
    C = lambda b, a: al[b]["calls"][a][sel[b]]
    WK = ("flash", "27b")
    d = np.mean([E(b, "gated") - E(b, "zs") for b in WK], axis=0)
    put_dual("XB1", est_mean(d, dbs, B))
    for b in WK:
        put_dual("XB1_" + b, est_mean(E(b, "gated") - E(b, "zs"), dbs, B), mc(E(b, "gated"), E(b, "zs")))
    put_dual("XB1_loop", est_mean(np.mean([E(b, "gated") - E(b, "rootonly") for b in WK], axis=0), dbs, B))
    put_dual("XB1_stack", est_mean(np.mean([E(b, "rootonly") - E(b, "zs") for b in WK], axis=0), dbs, B))
    put_dual("XB2", est_mean(E("plus", "gated") - E("plus", "zs"), dbs, B), mc(E("plus", "gated"), E("plus", "zs")))
    put_dual("XB3", est_mean(np.mean([E(b, "gated") - E(b, "fixed") for b in BK], axis=0), dbs, B))
    put_dual("XB3_ratio", est_ratio_pooled({b: C(b, "gated") for b in BK}, {b: C(b, "fixed") for b in BK}, dbs, B))
    for b in BK:
        put_dual("XB3_" + b, est_mean(E(b, "gated") - E(b, "fixed"), dbs, B), mc(E(b, "gated"), E(b, "fixed")))
    log("EXT1: n =", len(qids), "clusters", len(set(dbs.tolist())))


def analysis_pf(args, B):
    import pf_analyze as P
    log("analysis 3f: PF")
    runs = {}
    for k, rel in list(PF_REFS.items()) + list(PF_ARMS.items()):
        runs[k] = P.load_run(os.path.join(args.runs, rel))
    qids = None
    for k, r in runs.items():
        s = set(q for q, row in r["rows"].items() if R.ex_value(row) is not None)
        qids = s if qids is None else qids & s
    qids = sorted(qids)
    dbs = np.array([runs["zs"]["rows"][q].get("db_id") or q.split(":")[1] for q in qids])
    ex = {k: np.array([R.ex_value(runs[k]["rows"][q]) for q in qids], int) for k in runs}
    calls = {k: np.array([runs[k]["rows"][q].get("calls", 0) or 0 for q in qids], float) for k in runs}
    log("PF: n =", len(qids))

    def c(i, a, b):
        put_dual(i, est_mean(ex[a] - ex[b], dbs, B), mc(ex[a], ex[b]))
    c("PF1", "pf_n1000_q100", "pf_n0"); c("PF2", "pf_n1000_q085", "full_fixed"); c("PF2_n0", "pf_n0", "full_fixed"); c("PF2_q100", "pf_n1000_q100", "full_fixed")
    c("PF3a", "pf_n1000_q085", "full_gated"); c("PF3a_q100", "pf_n1000_q100", "full_gated")
    c("PF4_250_vs_0", "pf_n250_q100", "pf_n0"); c("PF4_1000_vs_250", "pf_n1000_q100", "pf_n250_q100"); c("PF4_noise_cost", "pf_n1000_q100", "pf_n1000_q085")
    put_dual("PF3b", est_ratio(calls["pf_n1000_q085"], calls["full_gated"], dbs, B))
    put_dual("PF3b_q100", est_ratio(calls["pf_n1000_q100"], calls["full_gated"], dbs, B))


# ------------------------------------------------------------------------------------------------ analysis 2: cost, latency, Pareto
def load_prices(path):
    import yaml
    return yaml.safe_load(open(path))


def price_of(prices, model):
    pm = prices.get("models") or {}
    if model in pm:
        return pm[model]
    for k in pm:
        if model and (k.startswith(model) or model.startswith(k)):
            return pm[k]
    return None


def run_costs(run, prices):
    ratio = float(prices.get("cache_hit_ratio", 0.2))
    bratio = float(prices.get("batch_ratio", 0.5))
    tot = collections.Counter()
    unpriced = 0
    emb_ph = 0.0
    emb_tok_missing = False
    for r in run.usage:
        kind = r.get("kind")
        pr = price_of(prices, r.get("model"))
        pt, ct = r.get("pt", 0) or 0, r.get("ct", 0) or 0
        if kind in ("embed", "rerank"):
            tot[kind + "_req"] += 1
            if r.get("pt") is None:
                emb_tok_missing = True
            ph = (prices.get("models") or {}).get("qwen3.7-text-embedding" if kind == "embed" else "qwen3.7-text-rerank")
            if ph:
                emb_ph += pt * ph["in"] / 1e6
            continue
        if pr is None:
            unpriced += 1
            continue
        cached = min(r.get("cached_tokens", 0) or 0, pt)
        mult = bratio if r.get("mode") == "batch" else 1.0
        tot["calls"] += 1
        tot["pt"] += pt; tot["ct"] += ct; tot["cached"] += cached
        tot["cny_disc"] += mult * ((pt - cached) * pr["in"] + cached * pr["in"] * ratio + ct * pr["out"]) / 1e6
        tot["cny_nodisc"] += mult * (pt * pr["in"] + ct * pr["out"]) / 1e6
        if r.get("mode") == "batch":
            tot["batch_rows"] += 1
    tot["unpriced"] = unpriced
    tot["emb_placeholder"] = emb_ph
    tot["emb_tok_missing"] = int(emb_tok_missing)
    return tot


def item_cost(run, qids, prices):
    """per-item list-price cost with discount from preds token counts (usage rows carry no qid)."""
    ratio = float(prices.get("cache_hit_ratio", 0.2))
    out = []
    for q in qids:
        r = run.rows.get(q)
        pr = price_of(prices, r.get("model")) if r else None
        if r is None or pr is None:
            out.append(0.0)
            continue
        pt, ct = r.get("prompt_tokens", 0) or 0, r.get("completion_tokens", 0) or 0
        cached = min(r.get("cached_tokens", 0) or 0, pt)
        out.append(((pt - cached) * pr["in"] + cached * pr["in"] * ratio + ct * pr["out"]) / 1e6)
    return np.array(out)


def analysis_cost(F, B, prices):
    log("analysis 2: cost, latency, Pareto")
    pts = {}
    cost_info = {}
    for b in BK:
        for cfg in K.COST_CFG[b]:
            k = (b, cfg)
            if k not in F.runs:
                raise RuntimeError("missing run for cost: %s" % (k,))
            r = F.runs[k]
            n = len(r.rows)
            t = run_costs(r, prices)
            ex = F.ex[k]
            m = ex >= 0
            lat = np.array([x.get("latency_s") or 0.0 for x in r.rows.values()])
            pc = float(sum(x.get("prompt_tokens", 0) or 0 for x in r.rows.values()))
            check = t["pt"] / pc if pc else None
            cost_info[k] = {"n": n, "usage_vs_preds_prompt_tokens": check, "unpriced": t["unpriced"], "batch_rows": t["batch_rows"], "emb_tok_missing": t["emb_tok_missing"]}
            pre = "cost.%s.%s." % (b, cfg)
            e = est_mean(ex[m].astype(float), F.dbs[m], B, scale=1.0)
            put(pre + "calls", t["calls"] / n); put(pre + "tokens", (t["pt"] + t["ct"]) / n); put(pre + "prompt_tok", t["pt"] / n); put(pre + "compl_tok", t["ct"] / n)
            put(pre + "cached_tok", t["cached"] / n)
            put(pre + "cny_q", t["cny_disc"] / n); put(pre + "cny_q_nodisc", t["cny_nodisc"] / n)
            put(pre + "cny_1000", 1000.0 * t["cny_disc"] / n); put(pre + "cny_1000_nodisc", 1000.0 * t["cny_nodisc"] / n)
            put(pre + "embed_req", float(np.mean([x.get("embed_calls", 0) or 0 for x in r.rows.values()])))
            put(pre + "rerank_req", float(np.mean([x.get("rerank_calls", 0) or 0 for x in r.rows.values()])))
            put(pre + "cny_emb_placeholder", "n/a")          # tokens of embedding / rerank requests are not recorded per item; usage rows exist only for live requests
            put(pre + "lat_mean", float(lat.mean())); put(pre + "lat_p50", float(np.percentile(lat, 50))); put(pre + "lat_p95", float(np.percentile(lat, 95)))
            put(pre + "ex", e["est"], ci=e["ci"], item_ci=e["item_ci"], n=e["n"]); put(pre + "n", n)
            pts[k] = (t["cny_disc"] / n, e["est"])
            CSV_ROWS["cost"].append({"backbone": b, "cfg": cfg, "n": n, "calls_q": t["calls"] / n, "tokens_q": (t["pt"] + t["ct"]) / n, "cny_q": t["cny_disc"] / n, "cny_q_nodisc": t["cny_nodisc"] / n,
                                     "cny_1000": 1000.0 * t["cny_disc"] / n, "embed_req_q": float(np.mean([x.get("embed_calls", 0) or 0 for x in r.rows.values()])), "rerank_req_q": float(np.mean([x.get("rerank_calls", 0) or 0 for x in r.rows.values()])), "lat_mean_s": float(lat.mean()),
                                     "lat_p50_s": float(np.percentile(lat, 50)), "lat_p95_s": float(np.percentile(lat, 95)), "ex": e["est"], "ex_lo": e["ci"][0], "ex_hi": e["ci"][1]})
    for k, v in cost_info.items():
        if v["unpriced"] or (v["usage_vs_preds_prompt_tokens"] is not None and abs(v["usage_vs_preds_prompt_tokens"] - 1) > 0.02):
            log("cost check", k, v)
    # route extra
    g, rr = F.runs[("plus", "gated")], F.runs[("plus", "route")]
    tg, tr = run_costs(g, prices), run_costs(rr, prices)
    n = len(rr.rows)
    ex_comp = collections.Counter()
    for x in rr.usage:
        if x.get("kind") == "chat" and x.get("tag") in ("crit", "cons"):
            ex_comp[x["tag"]] += 1
    n_samples = [r.get("n_samples") for r in rr.rows.values() if r.get("n_samples") is not None]
    comp_per_q = (1.0 + float(np.mean(n_samples))) if n_samples else None
    g_cc = sum(1 for x in g.usage if x.get("kind") == "chat" and x.get("tag") in ("crit", "cons"))
    r_cc = sum(1 for x in rr.usage if x.get("kind") == "chat" and x.get("tag") in ("crit", "cons"))
    put("cost.plus.route_extra.calls", (tr["calls"] - tg["calls"]) / n)
    put("cost.plus.route_extra.requests", (r_cc - g_cc) / n)
    put("cost.plus.route_extra.completions", comp_per_q if comp_per_q is not None else "n/a")
    put("cost.plus.route_extra.cny_q", (tr["cny_disc"] - tg["cny_disc"]) / n); put("cost.plus.route_extra.cny_q_nodisc", (tr["cny_nodisc"] - tg["cny_nodisc"]) / n)
    put("route.check.completions", comp_per_q if comp_per_q is not None else "n/a")
    put("route.check.requests", (r_cc - g_cc) / n)
    log("route requests: crit+cons chat rows route", r_cc, "gated", g_cc, "n", n, "n_samples mean", (float(np.mean(n_samples)) if n_samples else None))
    put("cost.price_date", "2026-10-26"); put("cost.price_source", "exp/prices.yaml (DashScope Beijing list price, CNY per million tokens, read 2026-10-26 from the vendor price page)")
    put("cost.embed_rerank_included", "No. Embedding and rerank requests per question are the counts recorded in preds.jsonl (embed_req, rerank_req); they count live HTTP requests only (a vector served from the local embedding cache is not counted), so the embedding count differs between arms for reasons unrelated to the method and is a lower bound of what a cold run issues; they are not priced in cny_q because (i) their per-token prices in exp/prices.yaml are unverified placeholders and (ii) the tokens of these requests are not recorded per item (usage rows exist only for live requests). cny_emb_placeholder is therefore not computable (n/a).")
    # Pareto
    names = list(pts)
    nondom = []
    for k in names:
        c, e = pts[k]
        dom = [o for o in names if o != k and pts[o][0] <= c + 1e-12 and pts[o][1] >= e - 1e-12 and (pts[o][0] < c - 1e-12 or pts[o][1] > e + 1e-12)]
        put("pareto.%s.%s.dominated_by" % k, ", ".join("%s %s" % x for x in dom) if dom else "none")
        if not dom:
            nondom.append(k)
    put("pareto.nondominated", "; ".join("%s %s" % k for k in sorted(nondom, key=lambda x: pts[x][0])))
    put("pareto.nondominated_k", len(nondom))
    for rowk in names:
        CSV_ROWS["pareto"].append({"backbone": rowk[0], "cfg": rowk[1], "cny_q": pts[rowk][0], "ex": pts[rowk][1], "nondominated": int(rowk in nondom)})
    # claims
    A = ("plus", "zs")
    for cl in K.PARETO_CLAIMS:
        _, rest = cl.split("_zs_vs_")
        bb, arm = rest.split("_")
        B_ = (bb, arm)
        e, a, b_, m = diff_est(F, A, B_, B)
        put("pareto.claim.%s.dex" % cl, e["est"], ci=e["ci"], item_ci=e["item_ci"], n=e["n"])
        qs = [q for q, mm in zip(F.qids, m) if mm]
        ca, cb = item_cost(F.runs[A], qs, prices), item_cost(F.runs[B_], qs, prices)
        er = est_ratio(cb, ca, F.dbs[m], B)
        put("pareto.claim.%s.cost_ratio" % cl, er["est"], ci=er["ci"], item_ci=er["item_ci"], n=er["n"])
        cA, eA = pts[A]; cB, eB = pts[B_]
        point = (cA <= cB + 1e-12) and (eA >= eB - 1e-12) and (cA < cB - 1e-12 or eA > eB + 1e-12)
        dec_lo = min(e["ci"][0], e["item_ci"][0])
        shown = point and dec_lo > 0
        wrong = max(e["ci"][1], e["item_ci"][1]) < 0
        if wrong:
            txt = "no: the single call is worse than the pipeline (interval below 0)"
        elif point and shown:
            txt = "yes, shown (EX interval above 0 on both intervals, cost lower)"
        elif point:
            txt = "yes, point-estimate dominance only (EX interval contains 0, cost lower)"
        else:
            txt = "no (point estimates: cost %.4f vs %.4f, EX %.3f vs %.3f)" % (cA, cB, eA, eB)
        txt += "; exact McNemar, plus zero-shot only right / other only right: %d / %d, p = %.3f" % (e["mc"][0], e["mc"][1], e["mc"][2])
        put("pareto.claim.%s.dominates" % cl, txt)
    return pts


# ------------------------------------------------------------------------------------------------ analysis 4: routing
def analysis_routing(F, B, prices, coef, rvec):
    log("analysis 4: routing")
    v3, ca, y, ok = rvec
    r = F.runs[("plus", "route")]
    n_all = len(r.rows)
    t = run_costs(r, prices)
    cny_q = t["cny_disc"] / n_all
    sc = {"v3": v3[ok], "route_ca": ca[ok]}
    yy = y[ok]
    dbs = F.dbs[ok]
    boot = CB(dbs, B)
    n = len(yy)
    ix = item_idx(n, B)
    for s, score in sc.items():
        cum = S._expected_cum_acc(score, yy)
        ks = [max(1, int(round(cv / 100.0 * n))) for cv in K.ROUTE_COV]
        bc = np.empty((B, len(ks)))                                 # accepted-set EX per cluster resample (ranking recomputed inside each resample)
        bi = np.empty((B, len(ks)))
        for b in range(B):
            idx = np.concatenate([boot.members[j] for j in boot.draws[b]])
            cu = S._expected_cum_acc(score[idx], yy[idx])
            bc[b] = [cu[min(k, len(cu)) - 1] for k in ks]
            cu = S._expected_cum_acc(score[ix[b]], yy[ix[b]])
            bi[b] = [cu[k - 1] for k in ks]
        for j, cv in enumerate(K.ROUTE_COV):
            k = ks[j]
            acc = float(cum[k - 1])
            ci, ici = list(S.pctl_ci(bc[:, j])), list(S.pctl_ci(bi[:, j]))
            pre = "route.%s.cov%d." % (s, cv)
            put(pre + "n_acc", k)
            put(pre + "err", 1.0 - acc, ci=[1.0 - ci[1], 1.0 - ci[0]], item_ci=[1.0 - ici[1], 1.0 - ici[0]], n=k)
            put(pre + "ex_acc", acc, ci=ci, item_ci=ici, n=k)
            put(pre + "ex_per_asked", acc * k / n, ci=[ci[0] * k / n, ci[1] * k / n], item_ci=[ici[0] * k / n, ici[1] * k / n], n=n)
            put(pre + "cny_q", cny_q); put(pre + "cny_per_accepted", cny_q * n / k)
            CSV_ROWS["route"].append({"score": s, "coverage": cv, "n_acc": k, "n": n, "err": 1.0 - acc, "ex_acc": acc, "ex_ci_lo": ci[0], "ex_ci_hi": ci[1], "ex_item_lo": ici[0], "ex_item_hi": ici[1], "ex_per_asked": acc * k / n, "cny_q": cny_q, "cny_per_accepted": cny_q * n / k})
    for lab, key in (("gated", ("plus", "gated")), ("bonvote2", ("plus", "bonvote2")), ("bonvote3", ("plus", "bonvote3"))):
        m = F.ex[key] >= 0
        put("route.ref.%s.ex" % lab, float(F.ex[key][m].mean()))
        tt = run_costs(F.runs[key], prices)
        put("route.ref.%s.cny_q" % lab, tt["cny_disc"] / len(F.runs[key].rows))
    # check against paper's 19.5 % wrong at 60 % coverage (formal pass uses the same score vector)
    put("route.check.err60_route_ca", OUT["route.route_ca.cov60.err"]["v"])


# ------------------------------------------------------------------------------------------------ analysis 5: heterogeneity
def mahal(theta, V):
    try:
        return float(theta.dot(np.linalg.solve(V, theta)))
    except np.linalg.LinAlgError:
        return None


def analysis_het(F, per, m_all, B):
    from scipy import stats as sps
    log("analysis 5: heterogeneity")
    dbs = F.dbs[m_all]
    boot = CB(dbs, B)
    G = boot.k
    out_read = []
    for fam, arm_b in (("gain", "zs"), ("loop", "rootonly")):
        D = np.stack([(per[b]["arms"]["gated"] - per[b]["arms"][arm_b])[m_all].astype(float) for b in BK], axis=1)          # n x 3
        ests = D.mean(0) * 100.0
        for j, b in enumerate(BK):
            e = est_mean(D[:, j], dbs, B)
            put("het.%s.%s" % (fam, b), e["est"], ci=e["ci"], item_ci=e["item_ci"], n=e["n"])
        C = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
        theta = C.dot(ests)
        # cluster bootstrap covariance (T1 chi-square reference) and delete-one jackknife covariance (Hotelling)
        BT = np.stack([100.0 * boot.means(D[:, j]) for j in range(3)], axis=1)            # B x 3
        Vb = np.cov(C.dot(BT.T))
        Wb = mahal(theta, Vb)
        loo = np.array([[100.0 * D[np.setdiff1d(np.arange(len(dbs)), boot.members[g]), j].mean() for j in range(3)] for g in range(G)])
        Cl = C.dot(loo.T)                                                                   # 2 x G
        Vj = (G - 1.0) / G * (Cl - Cl.mean(1, keepdims=True)).dot((Cl - Cl.mean(1, keepdims=True)).T)
        T2j = mahal(theta, Vj)
        p_chi2 = float(sps.chi2.sf(Wb, 2)) if Wb is not None else None
        if T2j is not None:
            Fst = T2j * (G - 2.0) / (2.0 * (G - 1.0))
            pF = float(sps.f.sf(Fst, 2, G - 2))
        else:
            pF = None
        put("het.%s.T1.stat" % fam, T2j if T2j is not None else "n/a"); put("het.%s.T1.df1" % fam, 2); put("het.%s.T1.df2" % fam, G - 2)
        put("het.%s.T1.p_F" % fam, pF if pF is not None else "n/a"); put("het.%s.T1.p_chi2" % fam, p_chi2 if p_chi2 is not None else "n/a")
        n = len(dbs)
        Cd = D.dot(C.T)                                                                      # n x 2 (in fractions)
        Vi = np.cov(Cd.T) / n
        thf = Cd.mean(0)
        Wi = mahal(thf, Vi)
        pi = float(sps.chi2.sf(Wi, 2)) if Wi is not None else None
        put("het.%s.T2.stat" % fam, Wi if Wi is not None else "n/a"); put("het.%s.T2.df" % fam, 2); put("het.%s.T2.p" % fam, pi if pi is not None else "n/a")
        for pr, (i, j) in (("flash_plus", (1, 0)), ("27b_plus", (2, 0)), ("27b_flash", (2, 1))):
            e = est_mean(D[:, i] - D[:, j], dbs, B)
            put("het.%s.pair.%s" % (fam, pr), e["est"], ci=e["ci"], item_ci=e["item_ci"], n=e["n"])
        diffs = (pF is not None and pF < 0.05 and pi is not None and pi < 0.05)
        txt = ("The %s differs between backbones (cluster Hotelling p = %.3f and item-level Wald p = %.3f, both < 0.05)." % (fam, pF, pi)) if diffs else (
            "Heterogeneity of the %s across the three backbones is inconclusive (cluster Hotelling F(2, %d) p = %s; item-level Wald chi-square(2) p = %s)." % (
                fam, G - 2, ("%.3f" % pF) if pF is not None else "n/a", ("%.3f" % pi) if pi is not None else "n/a"))
        put("het.%s.reading" % fam, txt)
        CSV_ROWS["het"].append({"family": fam, "gain_plus": ests[0], "gain_flash": ests[1], "gain_27b": ests[2], "T1_stat": T2j, "T1_p_F": pF, "T1_p_chi2": p_chi2, "T2_stat": Wi, "T2_p": pi})
    put("het.spearman_dropped", "Drop the sentence that the gain correlates with single-call EX (Spearman -0.5 over three backbones): with three points |rho| = 0.5 has an exact two-sided p of 5/6. Use the heterogeneity statement of het.gain.reading instead.")


# ------------------------------------------------------------------------------------------------ analysis 6: revisions
def analysis_rev(F, B, args, items, sets, RC):
    log("analysis 6: helped / hurt")
    f = F.runs[("plus", "fixed")]
    cnt = collections.Counter()
    ids = collections.defaultdict(list)
    for q in F.qids:
        root = f.root(q)
        row = f.rows[q]
        if root is None:
            continue
        changed = (row.get("final_sql") or "").strip() != (root.get("sql") or "").strip()
        if not changed:
            continue
        r0, r1 = root.get("ex"), R.ex_value(row)
        if r0 is None or r1 is None:
            cnt["unscored"] += 1
            continue
        kind = "helped" if (r0 == 0 and r1 == 1) else ("hurt" if (r0 == 1 and r1 == 0) else "neutral")
        if root.get("status") in EXEC_OK:
            cnt[kind] += 1; ids[kind].append(q); cnt["n"] += 1
        else:
            cnt["repair_" + kind] += 1
    put("rev.def", ("Population: fixed loop (plus), 1,534 dev items; an item enters if the root candidate executes (status ok_nonempty or ok_empty) and the final SQL text differs from the "
                    "root SQL text (strip-equal). helped = root EX 0 and final EX 1; hurt = root EX 1 and final EX 0; neutral = same EX. EX is execution accuracy against the original BIRD "
                    "dev reference query. Repair of a non-executing root is counted separately. The annotation labels do not enter the counts."))
    for k in ("n", "helped", "hurt", "neutral"):
        put("rev.e10.%s" % k, cnt[k])
    put("rev.all.n", cnt["n"] + cnt["repair_helped"] + cnt["repair_hurt"] + cnt["repair_neutral"])
    put("rev.all.helped", cnt["helped"] + cnt["repair_helped"]); put("rev.all.hurt", cnt["hurt"] + cnt["repair_hurt"]); put("rev.all.neutral", cnt["neutral"] + cnt["repair_neutral"])
    for k in ("repair_helped", "repair_hurt", "repair_neutral"):
        put("rev.all.%s" % k, cnt[k])
    # corrected references
    for sname, qs in sets.items():
        for g, gk in (("orig", "orig"), ("corr", "G1")):
            c = collections.Counter()
            n_items = 0
            for x in qs:
                fi, ri = items[("final", "plus.fixed")].get(x), items[("root", "plus.fixed")].get(x)
                if fi is None or ri is None or fi["usable"] != "1" or ri["usable"] != "1" or fi["G1"] == "" or ri["G1"] == "" or fi["orig"] == "" or ri["orig"] == "":
                    continue
                n_items += 1
                changed = fi["sha"] != ri["sha"]
                rexec = ri["root_status"] in RC.EXEC_STATUS
                if changed and rexec:
                    d = int(float(fi[gk])) - int(float(ri[gk]))
                    c["n"] += 1
                    c["helped" if d > 0 else ("hurt" if d < 0 else "neutral")] += 1
            put("rev.%s.%s.n_items" % (sname, g), n_items); put("rev.%s.%s.n_rev" % (sname, g), c["n"])
            for k in ("helped", "hurt", "neutral"):
                put("rev.%s.%s.%s" % (sname, g, k), c[k])
    # annotation labels
    labels = {r["item_id"]: r["final_label"] for r in csv.DictReader(open(os.path.join(args.results, "e10", "labels_final_all.csv")))}
    key = list(csv.DictReader(open(os.path.join(args.results, "e10", "annotation_key.csv"))))
    by_q = collections.defaultdict(list)
    for r in key:
        if r["system"] == "full":
            by_q[r["qid"]].append(r["item_id"])
    lab_cnt = collections.Counter()
    for kind in ("helped", "hurt", "neutral"):
        for q in ids[kind]:
            lab = "nolabel"
            if q in by_q:
                lab = labels.get(by_q[q][0], "nolabel")
            lab_cnt[(lab, kind)] += 1
    for lab in K.LABELS_E10:
        for kind in ("helped", "hurt", "neutral"):
            put("rev.label.%s.%s" % (lab, kind), lab_cnt[(lab, kind)])
    extra = {lab for (lab, _) in lab_cnt} - set(K.LABELS_E10)
    if extra:
        log("labels outside L1-L6/nolabel in revision items:", sorted(extra), {k: v for k, v in lab_cnt.items() if k[0] in extra})
    put("rev.sign_p", float(S.mcnemar_exact(cnt["helped"], cnt["hurt"])["p_two"]))
    m = F.ex[("plus", "fixed")] >= 0
    d = np.zeros(F.n)
    for i, q in enumerate(F.qids):
        root = f.root(q)
        if root is None or root.get("ex") is None or F.ex[("plus", "fixed")][i] < 0:
            continue
        if root.get("status") in EXEC_OK and (f.rows[q].get("final_sql") or "").strip() != (root.get("sql") or "").strip():
            d[i] = F.ex[("plus", "fixed")][i] - int(root["ex"])
    e = est_mean(d[m], F.dbs[m], B)
    put("rev.net", e["est"], ci=e["ci"], item_ci=e["item_ci"], n=e["n"])
    put("rev.annotation_in_counts", "no: the counts above are execution-based (EX against the original reference); annotation labels (rev.label.*) are attached afterwards for the items that happen to be in the annotated failure sample")
    CSV_ROWS["revisions"] = [{"set": "dev", "kind": k, "n": cnt[k]} for k in ("n", "helped", "hurt", "neutral", "repair_helped", "repair_hurt", "repair_neutral")]


# ------------------------------------------------------------------------------------------------ driver
def write_csv(path, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", required=True, help="dataset root ($R/data)")
    ap.add_argument("--results", required=True, help="copy of the host results/ directory (robust/, e10/)")
    ap.add_argument("--exp", required=True, help="copy of the host exp/ directory (prices.yaml, frozen/route_coef_hr.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    B = a.B
    prices = load_prices(os.path.join(a.exp, "prices.yaml"))
    coef = json.load(open(os.path.join(a.exp, "frozen", "route_coef_hr.json")))
    F = Frame(a.runs, B)
    log("frame: %d dev items, %d clusters" % (F.n, len(set(F.dbs.tolist()))))
    res, per, m_all = analysis_loop(F, B)
    analysis_roots(F, B)
    analysis_dual_dev(F, B)
    analysis_p5(F, B)
    rvec = analysis_p3(F, B, coef)
    items, sets, RC = analysis_rescore(F, B, a, coef)
    analysis_ext1(a, B)
    analysis_pf(a, B)
    analysis_cost(F, B, prices)
    analysis_routing(F, B, prices, coef, rvec)
    analysis_het(F, per, m_all, B)
    analysis_rev(F, B, a, items, sets, RC)
    missing = [k for k, _ in K.expected_keys() if k not in OUT]
    extra = [k for k in OUT if k not in KINDS]
    log("keys: %d written, %d expected, %d missing, %d extra" % (len(OUT), len(KINDS), len(missing), len(extra)))
    meta = {"scope": "posthoc-secondary", "sap": "docs/SAP.md section 23 (Amendment 10)", "B": B, "seed": SEED, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runs": {"%s.%s" % k: {"dir": r.rel, "n": len(r.rows), "preds_sha256": r.sha} for k, r in F.runs.items()},
            "n_dev": F.n, "n_clusters_dev": len(set(F.dbs.tolist())), "price_date": "2026-10-26", "api_calls": 0, "missing_keys": missing, "log": LOG,
            "interval_notes": "cluster = ClusterBoot percentile (B=10000, seed 20261101); item = paired i.i.d. bootstrap; t_ci = delete-one-database jackknife SE with t(G-1) (descriptive)"}
    json.dump({"_meta": meta, "keys": OUT}, io.open(os.path.join(a.out, "rv2_summary.json"), "w", encoding="utf-8"), indent=0, sort_keys=False, ensure_ascii=False)
    write_csv(os.path.join(a.out, "dual.csv"), DUAL_ROWS)
    for nm, fn in (("loop", "loop.csv"), ("roots", "roots.csv"), ("cost", "cost.csv"), ("pareto", "pareto.csv"), ("route", "route.csv"), ("revisions", "revisions.csv")):
        write_csv(os.path.join(a.out, fn), CSV_ROWS.get(nm, []))
    json.dump(CSV_ROWS.get("het", []), io.open(os.path.join(a.out, "het.json"), "w", encoding="utf-8"), indent=1)
    log("done in %.0f s" % (time.time() - t0))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
