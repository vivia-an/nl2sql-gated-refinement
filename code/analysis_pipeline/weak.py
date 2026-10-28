"""Backbone-family analyses of SAP Amendment 3 (docs/SAP.md section 15): W1, P1 per backbone (S-J) and pooled P1', Holm placement in family P*, X1 (S-I).

    stages `W1` and `X1` of analysis_pipeline.run_all  ->  results/W1.json, results/X1.json   (flat keys, same conventions as primary.py)

What is implemented, and where the SAP says so
* W1 (15.3): systems `full_gated` vs `zs` (= `zs_ours`) on every b in Wk = {qwen37flash, qwen3527b}, run r0. dEX_Wk = mean over b of [EX(gated,b) - EX(zs,b)],
  H0: dEX_Wk <= 0, one-sided, doubled into Holm. Database-clustered paired percentile bootstrap, B as given, ONE database resample shared by all backbones
  (15.3 "common database resample across backbones"). Sign-consistency rule: `confirmed` additionally needs a positive point estimate on every member of Wk;
  otherwise the verdict is `supported on average, not on each backbone`. Always reported: per-backbone dEX with CI and exact McNemar p (S-H), the same
  contrast on qwen37plus (control).  The item-level check of the pooled mean is the item-level paired bootstrap of the per-item mean difference (there is no
  exact test for a mean over backbones; documented in internal note KEYS).
* P1 per backbone (15.3 "Per-backbone claim ... family S-J"): `P1.<backbone>.*` is primary.p1_core on that backbone (identical code, identical bootstrap seed:
  `P1.qwen37plus.*` reproduces `P1.*` of primary.json). S-J = Holm over the per-backbone IUT p-values (m = 3), keys `P1.<b>.p_adj_sj`, `P1.sj.all_confirmed`.
* P1' pooled (15.3): dEX_Bk = mean over b of [EX(gated,b) - EX(fixed,b)] against -1.5 pp, R_Bk = mean over b of [sum calls(gated,b) / sum calls(fixed,b)] against 0.80,
  p = max(p_a, p_b); keys `P1prime.*`. NOT the same object as `P1.*` of primary.json, which is the single-backbone test of SAP 13.3 (qwen37plus).
* P5' (15.3): pooled over Bk, R = items whose root fails to execute on full_gated r0, comparator full_gated_blind r0, exact one-sided McNemar (the Holm p) and the
  db-clustered CI of the same difference (cluster criterion; Holm p = exact McNemar as P5 of primary.py); keys `P5prime.*`.
* Family P* (15.3): Holm over {P1prime, W1, P3, P5prime} (m = 4; 3 if P3 is dropped or P5' fails the pooled n_R >= 40 rule). P3 enters with its p from primary.json.
  Values: `W1.p_adj`, `P1prime.p_adj`, `P5prime.p_adj`, `Pstar.P3.p_adj` (+ `.verdict`). The paper's P1/P3/P5 keys are redirected to these by make_macros.OVERRIDES.
* Secondary rows (15.2/15.5): P2 = S-A row, P4 = S-E headline row, Holm inside their own families (`P2.p_adj_sa`, `P4.p_adj_se`; stage `Psec`).
No threshold, margin or rule of primary.py is changed; primary.json is read, never written."""
import numpy as np

from . import primary, runs as R, stats as S
from .secondary import pair_block, run_cost

BK = ("qwen37plus", "qwen37flash", "qwen3527b")         # SAP 15.3: Bk
WK = ("qwen37flash", "qwen3527b")                       # SAP 15.3: Wk
M1_MARGIN, RATIO_NULL, N_R_MIN, THR = primary.M1_MARGIN, primary.RATIO_NULL, primary.N_R_MIN, primary.THR
SJ_WORDS = "supported on average, not on each backbone"


def _get(reg, label, mk, res):
    try:
        return reg.get(label, model=mk)
    except R.RunError as e:
        res.note("W1: %s" % e)
        return None


def _pooled_diff(ex_a, ex_b, dbs, B, boot=None):
    """mean over backbones of the paired EX differences (pp); common database resample for every backbone. ex_a, ex_b: dict backbone -> int array."""
    boot = boot or S.ClusterBoot(dbs, B)
    mks = list(ex_a)
    bt = np.mean([100.0 * (boot.means(ex_a[m]) - boot.means(ex_b[m])) for m in mks], axis=0)
    d_item = np.mean([ex_a[m] - ex_b[m] for m in mks], axis=0).astype(float)
    delta = 100.0 * float(d_item.mean())
    ib = 100.0 * S.item_boot_means(d_item, B)
    lo, hi = S.pctl_ci(bt)
    ilo, ihi = S.pctl_ci(ib)
    loo = boot.loo(lambda idx: 100.0 * d_item[idx].mean())
    out = {"delta_pp": delta, "ci_lo": lo, "ci_hi": hi, "item_ci_lo": ilo, "item_ci_hi": ihi, "boot": bt, "item_boot": ib, "n": int(len(d_item)), "n_dbs": boot.k,
           "loo_min": min(v for _, v in loo) if loo else None, "loo_max": max(v for _, v in loo) if loo else None,
           "same_sign_share": primary._same_sign_share(np.asarray(dbs), d_item, delta),
           "per_backbone": {m: 100.0 * float((ex_a[m] - ex_b[m]).mean()) for m in mks}}
    return out


def _put_pooled(res, pre, o):
    res.ci(pre + ".delta_ex", o["delta_pp"], o["ci_lo"], o["ci_hi"])
    res.set(pre + ".delta_ex.item_ci_lo", o["item_ci_lo"]); res.set(pre + ".delta_ex.item_ci_hi", o["item_ci_hi"])
    for k in ("n", "n_dbs", "loo_min", "loo_max", "same_sign_share"):
        res.set("%s.%s" % (pre, k), o[k])


def _sens_no_thr(fn, dbs):
    """SAP 8: the same estimate without thrombosis_prediction (None if the database is absent)."""
    keep = np.asarray(dbs) != THR
    return fn(keep) if keep.any() and (~keep).any() else None


# ------------------------------------------------------------------------------------------ P1 per backbone (S-J) and pooled P1'
def run_p1_backbones(reg, res, B):
    got, done = {}, []
    for mk in BK:
        g, f = _get(reg, "full_gated", mk, res), _get(reg, "full_fixed", mk, res)
        if g is None or f is None:
            res.note("P1.%s: gated or fixed r0 missing or incomplete (no per-backbone key written)" % mk)
            continue
        A = R.align({"g": g, "f": f})
        cg, cf = R.col(A["rows"]["g"], "calls"), R.col(A["rows"]["f"], "calls")
        o = primary.p1_core(A["ex"]["g"], A["ex"]["f"], cg, cf, A["dbs"], B)
        pre = "P1.%s" % mk
        res.ci(pre + ".delta_ex", o["delta_pp"], o["ci_lo"], o["ci_hi"])
        res.set(pre + ".delta_ex.item_ci_lo", o["item_ci_lo"]); res.set(pre + ".delta_ex.item_ci_hi", o["item_ci_hi"])
        rr = o["ratio"]
        res.ci(pre + ".calls_ratio", rr["mean"], rr["ci_lo"], rr["ci_hi"])
        res.set(pre + ".calls_ratio.ci_hi_1s95", rr["ci_hi_1s95"])
        for k in ("calls_gated", "calls_fixed", "p_1s", "p_item_1s"):
            res.set("%s.calls_ratio.%s" % (pre, k), rr[k])
        res.set(pre + ".p_1s", o["p_1s_iut"]); res.set(pre + ".p_item_1s", o["p_item_1s_iut"])
        for k in ("n", "n10", "n01", "n_dbs", "same_sign_share"):
            res.set("%s.%s" % (pre, k), o[k])
        got[mk] = (A, o)
        done.append(mk)
    res.set("P1.n_backbones", len(done))
    res.set("P1.backbones", ", ".join(done) if done else "none")
    if done:
        res.set("P1.backbones_scope", "per-backbone replication of the P1 test (SAP 15.3 S-J, secondary); `P1.delta_ex` of primary.json is the qwen37plus test only")
    # S-J: Holm over the per-backbone IUT p-values (m = 3)
    if done:
        two = {mk: min(1.0, 2.0 * got[mk][1]["p_1s_iut"]) for mk in done}
        two_i = {mk: min(1.0, 2.0 * got[mk][1]["p_item_1s_iut"]) for mk in done}
        adj, adj_i = S.holm(two, len(BK)), S.holm(two_i, len(BK))
        conf = []
        for mk in done:
            o = got[mk][1]
            res.set("P1.%s.p_adj_sj" % mk, adj[mk]); res.set("P1.%s.p_adj_sj_item" % mk, adj_i[mk])
            v = S.verdict(adj[mk], o["cl_ok"], o["item_ok"], o["favourable"], adj_i[mk])
            res.set("P1.%s.verdict_sj" % mk, v)
            conf.append(v == "confirmed")
        res.set("P1.sj.m", len(BK))
        res.set("P1.sj.all_confirmed", bool(len(done) == len(BK) and all(conf)))
        res.set("P1.sj.reading", "'on each backbone' is allowed only if all three S-J tests are confirmed (SAP 15.3); otherwise list the backbones that were and were not")
    # pooled P1' (needs every member of Bk)
    if len(done) < len(BK):
        res.set("P1prime.status", "not run: gated/fixed r0 missing on " + ", ".join(m for m in BK if m not in done))
        return None
    al = R.align(dict([("g_" + m, reg.get("full_gated", model=m)) for m in BK] + [("f_" + m, reg.get("full_fixed", model=m)) for m in BK]))
    dbs = al["dbs"]
    exg = {m: al["ex"]["g_" + m] for m in BK}
    exf = {m: al["ex"]["f_" + m] for m in BK}
    cg = {m: R.col(al["rows"]["g_" + m], "calls") for m in BK}
    cf = {m: R.col(al["rows"]["f_" + m], "calls") for m in BK}

    def core(mask):
        d = dbs[mask]
        boot = S.ClusterBoot(d, B)
        o = _pooled_diff({m: exg[m][mask] for m in BK}, {m: exf[m][mask] for m in BK}, d, B, boot)
        rb = np.mean([boot.ratio(cg[m][mask], cf[m][mask]) for m in BK], axis=0)
        rib = np.mean([S.item_boot_ratio(cg[m][mask], cf[m][mask], B) for m in BK], axis=0)
        r = float(np.mean([cg[m][mask].sum() / max(cf[m][mask].sum(), 1e-12) for m in BK]))
        rlo, rhi = S.pctl_ci(rb)
        ilo, ihi = S.pctl_ci(rib)
        o["ratio"] = {"mean": r, "ci_lo": rlo, "ci_hi": rhi, "item_ci_lo": ilo, "item_ci_hi": ihi, "p_1s": S.p_ge(rb, RATIO_NULL), "p_item_1s": S.p_ge(rib, RATIO_NULL)}
        o["p_a"], o["p_item_a"] = S.p_le(o["boot"], -M1_MARGIN), S.p_le(o["item_boot"], -M1_MARGIN)
        o["p_1s"] = max(o["p_a"], o["ratio"]["p_1s"]); o["p_item_1s"] = max(o["p_item_a"], o["ratio"]["p_item_1s"])
        o["cl_ok"] = bool(o["ci_lo"] > -M1_MARGIN and rhi < RATIO_NULL)
        o["item_ok"] = bool(o["item_ci_lo"] > -M1_MARGIN and ihi < RATIO_NULL)
        o["favourable"] = bool(o["delta_pp"] > -M1_MARGIN and r < RATIO_NULL)
        return o
    o = core(np.ones(len(dbs), bool))
    res.set("P1prime.status", "ok")
    _put_pooled(res, "P1prime", o)
    res.ci("P1prime.calls_ratio", o["ratio"]["mean"], o["ratio"]["ci_lo"], o["ratio"]["ci_hi"])
    res.ci("P1prime.calls_saved", 1.0 - o["ratio"]["mean"], 1.0 - o["ratio"]["ci_hi"], 1.0 - o["ratio"]["ci_lo"])
    res.set("P1prime.p_a", o["p_a"]); res.set("P1prime.p_b", o["ratio"]["p_1s"]); res.set("P1prime.p_1s", o["p_1s"]); res.set("P1prime.p_item_1s", o["p_item_1s"])
    res.set("P1prime.n_backbones", len(BK)); res.set("P1prime.margin_pp", M1_MARGIN); res.set("P1prime.ratio_null", RATIO_NULL)
    for m in BK:
        res.set("P1prime.delta_ex.%s" % m, o["per_backbone"][m])
    s = _sens_no_thr(core, dbs)
    if s is not None:
        res.ci("P1prime.sens_no_thr.delta_ex", s["delta_pp"], s["ci_lo"], s["ci_hi"]); res.set("P1prime.sens_no_thr.p_1s", s["p_1s"])
        res.set("P1prime.sens_no_thr.same_decision", bool(s["cl_ok"] == o["cl_ok"] and (s["delta_pp"] > 0) == (o["delta_pp"] > 0)))
    return o


# ------------------------------------------------------------------------------------------ W1
def run_w1(reg, res, B):
    runs = {}
    for mk in WK:
        g, z = _get(reg, "full_gated", mk, res), _get(reg, "zs", mk, res)
        if g is None or z is None:
            res.set("W1.status", "not run: full_gated or zs r0 missing or incomplete on " + mk)
            return None
        runs["g_" + mk], runs["z_" + mk] = g, z
    al = R.align(runs)
    dbs = al["dbs"]
    exg = {m: al["ex"]["g_" + m] for m in WK}
    exz = {m: al["ex"]["z_" + m] for m in WK}
    boot = S.ClusterBoot(dbs, B)
    o = _pooled_diff(exg, exz, dbs, B, boot)
    o["p_1s"], o["p_item_1s"] = S.p_le(o["boot"], 0.0), S.p_le(o["item_boot"], 0.0)
    o["cl_ok"], o["item_ok"], o["favourable"] = bool(o["ci_lo"] > 0), bool(o["item_ci_lo"] > 0), bool(o["delta_pp"] > 0)
    sign_ok = bool(all(v > 0 for v in o["per_backbone"].values()))
    res.set("W1.status", "ok")
    res.set("W1.n_backbones", len(WK)); res.set("W1.backbones", ", ".join(WK))
    _put_pooled(res, "W1", o)
    res.set("W1.p_1s", o["p_1s"]); res.set("W1.p_item_1s", o["p_item_1s"]); res.set("W1.sign_consistent", sign_ok)
    res.set("W1.dropped_ex_null", al["dropped"]["ex_null"])
    res.set("W1.reading_rule", "W1 compares the pipeline with a single call, not the loop alone; the loop is separated by E3.<backbone>.loop_over_root (SAP 15.3 reading rule)")
    for m in WK:                                                # S-H rows: per-backbone dEX with CI and exact McNemar p
        pair_block(res, "W1.%s" % m, exg[m], exz[m], dbs, B, boot)
        res.set("W1.%s.positive" % m, bool(o["per_backbone"][m] > 0))
    # control: the strong backbone (same contrast, expected <= 1 pp)
    gp, zp = _get(reg, "full_gated", "qwen37plus", res), _get(reg, "zs", "qwen37plus", res)
    if gp is not None and zp is not None:
        A = R.align({"a": gp, "b": zp})
        pair_block(res, "W1.control_plus", A["ex"]["a"], A["ex"]["b"], A["dbs"], B)
    s = _sens_no_thr(lambda mask: _pooled_diff({m: exg[m][mask] for m in WK}, {m: exz[m][mask] for m in WK}, dbs[mask], B), dbs)
    if s is not None:
        res.ci("W1.sens_no_thr.delta_ex", s["delta_pp"], s["ci_lo"], s["ci_hi"]); res.set("W1.sens_no_thr.p_1s", S.p_le(s["boot"], 0.0))
        res.set("W1.sens_no_thr.same_decision", bool(((s["ci_lo"] > 0) == o["cl_ok"]) and (s["delta_pp"] > 0) == (o["delta_pp"] > 0)))
    return o


# ------------------------------------------------------------------------------------------ P5' (pooled over Bk)
def run_p5prime(reg, res, B):
    """SAP 15.3 P5': R = dev items of full_gated r0 (pooled over Bk) whose root candidate fails to execute; comparator full_gated_blind r0 on the same backbone;
    paired on R, one-sided exact McNemar (gated right / blind wrong vs the reverse) = the pre-registered test and the Holm p, exactly as P5 in primary.py (13.3).
    The CI of the paired repair-rate difference (and `p_1s_cluster`) is the db-clustered bootstrap with ONE database resample shared with W1 and P1' (the dev
    databases are the clusters, databases without R items contribute zero); it enters the verdict as the cluster criterion. Inclusion rule: pooled n_R >= 40 and a blind run on every backbone of Bk.
    Always reported: repair rate per backbone and per error class with Wilson intervals, executable rate, blind rate."""
    pooled = []                                           # (backbone, row, class, blind outcome)
    blind_ok = True
    dev_dbs = None
    for mk in BK:
        g, bl = _get(reg, "full_gated", mk, res), _get(reg, "full_gated_blind", mk, res)
        if g is None:
            res.set("P5prime.status", "not run: full_gated r0 missing or incomplete on " + mk)
            return None
        pop = primary.repair_population(g)
        if pop is None:
            res.set("P5prime.status", "not run: %s has no cands.jsonl (root candidates)" % g.run_id)
            return None
        if dev_dbs is None:
            dev_dbs = np.array([g.rows[q].get("db_id") for q in sorted(g.rows)])
        if bl is None:
            blind_ok = False
        for r, c in pop:
            v = None
            if bl is not None and r["qid"] in bl.rows:
                v = R.ex_value(bl.rows[r["qid"]])
            pooled.append((mk, r, c, -1 if v is None else int(v)))
    n_R = len(pooled)
    res.set("P5prime.n_R", n_R)
    for mk in BK:
        res.set("P5prime.n_R.%s" % mk, sum(1 for x in pooled if x[0] == mk))
    in_family = bool(n_R >= N_R_MIN and blind_ok)
    res.set("P5prime.in_family", in_family)
    rep = np.array([1 if R.ex_value(r) == 1 else 0 for _, r, _, _ in pooled])
    exe = np.array([1 if r.get("exec_ok") else 0 for _, r, _, _ in pooled])
    dbs = np.array([r.get("db_id") for _, r, _, _ in pooled])
    bk = np.array([m for m, _, _, _ in pooled])
    cls = np.array([c for _, _, c, _ in pooled])
    brep = np.array([x[3] for x in pooled])
    res.set("P5prime.n_dbs", int(len(set(dbs.tolist()))))
    if n_R:
        lo, hi = S.clopper_pearson(int(rep.sum()), n_R)
        res.ci("P5prime.repair_rate", float(rep.mean()), lo, hi)
        elo, ehi = S.clopper_pearson(int(exe.sum()), n_R)
        res.ci("P5prime.executable_rate", float(exe.mean()), elo, ehi)
        for mk in BK:
            sel = bk == mk
            if sel.sum():
                w = S.wilson(int(rep[sel].sum()), int(sel.sum()))
                res.ci("P5prime.by_backbone.%s.repair_rate" % mk, float(rep[sel].mean()), w[0], w[1])
        for c in ("syntax", "unknown identifier", "ambiguous column", "timeout", "other"):
            sel = cls == c
            key = "P5prime.by_class.%s" % c.replace(" ", "_")
            res.set(key + ".n", int(sel.sum()))
            if sel.sum():
                w = S.wilson(int(rep[sel].sum()), int(sel.sum()))
                res.ci(key + ".repair_rate", float(rep[sel].mean()), w[0], w[1])
    if not blind_ok:
        res.set("P5prime.status", "descriptive: full_gated_blind r0 missing on a backbone of Bk (n_R = %d)" % n_R)
        return None
    okb = brep >= 0
    blo, bhi = S.clopper_pearson(int(brep[okb].sum()), int(okb.sum()))
    res.ci("P5prime.blind_rate", float(brep[okb].mean()), blo, bhi)
    b = int(((rep == 1) & (brep == 0) & okb).sum())
    c = int(((rep == 0) & (brep == 1) & okb).sum())
    mc = S.mcnemar_exact(b, c)
    res.set("P5prime.b", b); res.set("P5prime.c", c)
    res.set("P5prime.mcnemar.p_1s", mc["p_one_greater"]); res.set("P5prime.mcnemar.p_2s", mc["p_two"])
    # common database resample: dev databases are the clusters
    boot = S.ClusterBoot(dev_dbs, B)
    dd = (rep - brep).astype(float)
    num = np.array([dd[okb & (dbs == u)].sum() for u in boot.u])
    den = np.array([float((okb & (dbs == u)).sum()) for u in boot.u])
    bt = num[boot.draws].sum(1) / np.maximum(den[boot.draws].sum(1), 1.0)
    lo, hi = S.pctl_ci(bt)
    diff = float(dd[okb].mean())
    res.ci("P5prime.diff_rate_pp", 100.0 * diff, 100.0 * lo, 100.0 * hi)
    res.set("P5prime.p_1s_cluster", S.p_le(bt, 0.0))
    res.set("P5prime.p_1s", mc["p_one_greater"]); res.set("P5prime.p_item_1s", mc["p_one_greater"])   # as P5 of primary.py: the pre-registered test is the exact one-sided McNemar on R
    res.set("P5prime.status", "ok" if in_family else "descriptive: n_R = %d < %d (family m reduced)" % (n_R, N_R_MIN))
    return {"p_1s": mc["p_one_greater"], "p_item_1s": mc["p_one_greater"], "cl_ok": bool(lo is not None and lo > 0), "item_ok": bool(mc["p_one_greater"] < 0.05),
            "favourable": bool(b > c), "in_family": in_family}


# ------------------------------------------------------------------------------------------ family P* (SAP 15.3, m = 4) and verdicts
def run_pstar(reg, res, o_p1, o_w1, o_p5, primary_res):
    """Holm over {P1prime, W1, P3, P5prime} exactly as 15.3 (one-sided p doubled, m fixed by the inclusion rules: 4, or 3 when P3 is dropped or P5' fails
    its inclusion rule), verdict words of 13.3 (+ the W1 sign rule). P3 enters with its db-clustered p from primary.json (same data, same seed, same
    database resample). Writes P1prime/W1/P5prime .p_2s .p_adj .p_adj_item .verdict and Pstar.P3.* and Pstar.family.*."""
    if primary_res is None:
        res.note("Pstar: primary result not given (run the stage `primary` in the same call): no P* Holm values written")
        return
    pd = primary_res.d
    members, two, two_i, info = [], {}, {}, {}
    if o_p1 is not None:
        members.append("P1prime"); two["P1prime"] = min(1.0, 2.0 * o_p1["p_1s"]); two_i["P1prime"] = min(1.0, 2.0 * o_p1["p_item_1s"]); info["P1prime"] = o_p1
    if o_w1 is not None:
        members.append("W1"); two["W1"] = min(1.0, 2.0 * o_w1["p_1s"]); two_i["W1"] = min(1.0, 2.0 * o_w1["p_item_1s"]); info["W1"] = o_w1
    dropped = []
    if pd.get("P3.status") == "ok" or (pd.get("P3.status") or "").startswith("not run"):
        members.append("P3"); two["P3"] = pd.get("P3.p_2s"); two_i["P3"] = pd.get("P3.p_item_2s")
        info["P3"] = {"cl_ok": bool(pd.get("P3.dauroc.ci_lo", -1) > 0), "item_ok": bool(pd.get("P3.dauroc.item_ci_lo", -1) > 0), "favourable": bool(pd.get("P3.dauroc.mean", -1) > 0)}
    else:
        dropped.append("P3 (route_ca not adopted)")
    if o_p5 is not None and o_p5["in_family"]:
        members.append("P5prime"); two["P5prime"] = min(1.0, 2.0 * o_p5["p_1s"]); two_i["P5prime"] = min(1.0, 2.0 * o_p5["p_item_1s"]); info["P5prime"] = o_p5
    elif res.d.get("P5prime.n_R") is not None:
        dropped.append("P5prime (pooled n_R = %d < %d or blind run missing)" % (res.d["P5prime.n_R"], N_R_MIN))
    m = len(members)
    adj, adj_i = S.holm(two, m), S.holm(two_i, m)
    res.set("Pstar.family.m", m); res.set("Pstar.family.members", ", ".join(members)); res.set("Pstar.family.dropped", ", ".join(dropped) if dropped else "none")
    res.set("Pstar.family.label", "family P* of SAP Amendment 3 (15.3): Holm, family-wise 0.05, one-sided p doubled; supersedes the SAP 13.3 family in primary.json")
    for name in members:
        o = info[name]
        pre = "Pstar.P3" if name == "P3" else name
        if two[name] is not None:
            res.set(pre + ".p_2s", two[name]); res.set(pre + ".p_adj", adj[name])
        if two_i[name] is not None:
            res.set(pre + ".p_adj_item", adj_i[name])
        if two[name] is None:
            continue
        v = S.verdict(adj[name], o["cl_ok"], o["item_ok"], o["favourable"], adj_i[name])
        if name == "W1" and v == "confirmed" and not res.d.get("W1.sign_consistent", False):
            v = SJ_WORDS                                          # SAP 15.3 sign-consistency rule written in advance
        res.set(pre + ".verdict", v)


# ------------------------------------------------------------------------------------------ secondary rows: P2 in S-A, P4 in S-E (SAP 15.2 / 15.5)
def run_secondary_rows(res, primary_res, e1_res, e4_res):
    """P2 is the S-A row 'full vs B*' (15.2): its adjusted p is the Holm value of E1.full_vs.<B*> inside S-A (two-sided exact McNemar, family = systems actually
    compared, 13.5). P4 is the S-E headline row `ws_full` - `append` at p = 40 % (15.2): Holm inside S-E = the seven rows of 13.5 plus this one (m = 8); absent
    comparators count as p = 1. Neither is a confirmatory claim; the verdict words are those of 13.3 applied to the secondary adjusted p."""
    pd = primary_res.d
    b = pd.get("P2.bstar")
    if b and e1_res is not None and ("E1.full_vs.%s.p_adj" % b) in e1_res.d:
        p = e1_res.d["E1.full_vs.%s.p_adj" % b]
        res.set("P2.p_adj_sa", p); res.set("P2.sa_family_m", e1_res.d.get("E1.full_vs.family_m"))
        lo, ilo, d = pd.get("P2.delta_ex.ci_lo"), pd.get("P2.delta_ex.item_ci_lo"), pd.get("P2.delta_ex.mean")
        if lo is not None:
            res.set("P2.verdict_sa", S.verdict(p, bool(lo > -primary.M2_MARGIN), bool(ilo > -primary.M2_MARGIN), bool(d > -primary.M2_MARGIN), None))
        res.set("P2.sa_note", "S-A secondary row (SAP 15.2): Holm over the E1.full_vs comparisons; not part of the confirmatory family")
    if pd.get("P4.p_lrt") is not None and e4_res is not None:
        ps = {"append@p40": pd["P4.p_lrt"]}
        for comp, pp in (("static", 40), ("ws_evict", 40), ("memrl", 40), ("append", 20), ("static", 20), ("ws_evict", 20), ("memrl", 20)):
            ps["%s@p%d" % (comp, pp)] = e4_res.d.get("E4.contrast.ws_full_vs_%s.p%d.p" % (comp, pp))
        adj = S.holm(ps, len(ps))
        res.set("P4.p_adj_se", adj["append@p40"]); res.set("P4.se_family_m", len(ps))
        res.set("P4.verdict_se", S.verdict(adj["append@p40"], bool(pd.get("P4.logodds.ci_lo", -1) > 0), bool(pd.get("P4.p_mcnemar", 1) < 0.05 and pd.get("P4.diff_pp.mean", -1) > 0),
                                           bool(pd.get("P4.logodds.mean", -1) > 0), None))
        res.set("P4.se_note", "S-E headline row (SAP 15.2): Holm over the seven S-E rows of 13.5 plus ws_full - append at 40 % (m = 8)")


# ------------------------------------------------------------------------------------------ X1 (S-I)
def _item_costs(run, prices):
    ratio = float(prices.get("cache_hit_ratio", 0.2))
    out = []
    for q in sorted(run.rows):
        r = run.rows[q]
        pr = (prices.get("models") or {}).get(r.get("model"))
        if pr is None:
            out.append(np.nan); continue
        pt, ct = r.get("prompt_tokens", 0) or 0, r.get("completion_tokens", 0) or 0
        cached = min(r.get("cached_tokens", 0) or 0, pt)
        out.append(((pt - cached) * pr["in"] + cached * pr["in"] * ratio + ct * pr["out"]) / 1e6)
    return dict(zip(sorted(run.rows), out))


def run_x1(reg, res, B, prices):
    """SAP 15.5 S-I: full_gated on a weak backbone against zs on qwen37plus: dEX (db-clustered CI) and the list-price cost ratio per question (exp/prices.yaml,
    the E11 cost model: chat tokens from usage*.jsonl x list price, cache-hit ratio and batch ratio as in prices.yaml). Descriptive/directional: margin 3.0 pp and
    cost-ratio bound 0.70 are reported as estimates with intervals; no equivalence claim unless both bounds hold. qwen3527b is the exploratory analogue."""
    zp = _get(reg, "zs", "qwen37plus", res)
    if zp is None or not prices:
        res.set("X1.status", "not run: zs r0 on qwen37plus or exp/prices.yaml missing")
        return
    n_ok = 0
    for mk, tag in (("qwen37flash", "flash_vs_plus_zs"), ("qwen3527b", "q3527b_vs_plus_zs")):
        g = _get(reg, "full_gated", mk, res)
        if g is None:
            continue
        A = R.align({"a": g, "b": zp})
        pre = "X1." + tag
        o = pair_block(res, pre, A["ex"]["a"], A["ex"]["b"], A["dbs"], B)
        ca, cb = run_cost(g, prices), run_cost(zp, prices)
        if not ca or not cb or not cb["per_q"]:
            continue
        res.set(pre + ".cost_ratio", ca["per_q"] / cb["per_q"])
        res.set(pre + ".cost_per_q.gated", ca["per_q"]); res.set(pre + ".cost_per_q.zs_plus", cb["per_q"])
        # interval of the cost ratio: db-clustered bootstrap of the ratio of per-question list-price costs from the preds token counts (usage rows carry no qid)
        ia, ib = _item_costs(g, prices), _item_costs(zp, prices)
        qs = [q for q in A["qids"] if np.isfinite(ia.get(q, np.nan)) and np.isfinite(ib.get(q, np.nan))]
        if qs:
            idx = {q: i for i, q in enumerate(A["qids"])}
            sel = np.array([idx[q] for q in qs])
            va, vb = np.array([ia[q] for q in qs]), np.array([ib[q] for q in qs])
            bt = S.ClusterBoot(A["dbs"][sel], B).ratio(va, vb)
            lo, hi = S.pctl_ci(bt)
            res.set(pre + ".cost_ratio.ci_lo", lo); res.set(pre + ".cost_ratio.ci_hi", hi)
            res.set(pre + ".cost_ratio.preds_based", float(va.sum() / max(vb.sum(), 1e-12)))
            res.set(pre + ".ni_bound_holds", bool(o["ci"][0] > -3.0))
            res.set(pre + ".cost_bound_holds", bool(hi < 0.70))
            res.set(pre + ".equivalence_claim_allowed", bool(o["ci"][0] > -3.0 and hi < 0.70))
        res.set(pre + ".margin_pp", 3.0); res.set(pre + ".cost_ratio_bound", 0.70)
        res.set(pre + ".scope", "descriptive" if mk == "qwen37flash" else "descriptive (exploratory analogue)")
        n_ok += 1
    res.set("X1.status", "ok" if n_ok else "not run: no complete full_gated r0 on a weak backbone")
    res.set("X1.scope", "descriptive: SAP 15.5 S-I is directional (estimates with intervals, no equivalence claim unless both bounds hold)")


# ------------------------------------------------------------------------------------------ driver used by run_all
def run_w1_stage(reg, res, B, primary_res=None):
    o_p1 = run_p1_backbones(reg, res, B)
    o_w1 = run_w1(reg, res, B)
    o_p5 = run_p5prime(reg, res, B)
    run_pstar(reg, res, o_p1, o_w1, o_p5, primary_res)
