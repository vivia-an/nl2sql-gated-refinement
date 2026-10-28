"""RV2 key grammar (SAP Amendment 10, section 23): the complete list of base keys that results/posthoc2/rv2_summary.json must hold.

Pure Python 3.6 (no numpy): imported by make_xmacros.py (host) and by rv2_analyze.py (worker machine). A base key `K` may carry `K.ci` (cluster 95 % interval), `K.item_ci`,
`K.t_ci`, `K.n`. The summary file stores, per base key, {"v": value, "kind": kind, "ci": [lo, hi] | null, "item_ci": [...] | null, "n": int | null, "meaning": str};
`na_ok` lists keys that may be a string "n/a" by design (undefined statistic)."""
BK = ("plus", "flash", "27b")
PB = BK + ("pooled",)
ARMS = ("zs", "rootonly", "gated", "fixed")
COST_CFG = {"plus": ("zs", "rootonly", "gated", "blind", "fixed", "route", "acsql", "bonvote2", "bonvote3", "macsql", "sdebug"),
            "flash": ("zs", "rootonly", "gated", "blind", "fixed"), "27b": ("zs", "rootonly", "gated", "blind", "fixed")}
COST_FIELDS = ("calls", "tokens", "prompt_tok", "compl_tok", "cached_tok", "cny_q", "cny_q_nodisc", "cny_1000", "cny_1000_nodisc", "embed_req", "rerank_req",
               "cny_emb_placeholder", "lat_mean", "lat_p50", "lat_p95", "ex", "n")
DUAL_FIELDS = ("est", "n", "g", "ci", "item_ci", "t_ci", "p_mcnemar", "b", "c", "deff", "loo_min", "loo_max", "rule", "class_cluster", "class_item", "class_t",
               "reading_changes", "reading_changes_t")
NO_MCNEMAR = ("P1p_ratio", "P3", "P3_aurc", "P3_acc60")                # not binary paired contrasts: p_mcnemar / b / c are the string "n/a"
RES_CELLS = tuple(["C1_" + b for b in PB] + ["C2_flash", "C2_27b", "C2_pooled", "C2_plus", "C3_auroc", "C3_aurc", "C3_acc60", "C4"])
RES_SETS = ("S_same", "S_all")
RES_GOLD = ("orig", "corr", "shift")
PF_IDS = ("PF1", "PF2", "PF2_n0", "PF2_q100", "PF3a", "PF3a_q100", "PF3b", "PF3b_q100", "PF4_250_vs_0", "PF4_1000_vs_250", "PF4_noise_cost")
XB_IDS = ("XB1", "XB1_flash", "XB1_27b", "XB1_loop", "XB1_stack", "XB2", "XB3", "XB3_ratio", "XB3_plus", "XB3_flash", "XB3_27b")
DUAL_IDS = (["P1p." + b for b in PB] + ["P1p_ratio." + b for b in PB] + ["W1.pooled", "W1.flash", "W1.27b", "W1.plus", "P5p", "P3", "P3_aurc", "P3_acc60"]
            + ["RES.%s.%s.%s" % (s, c, g) for s in RES_SETS for c in RES_CELLS for g in RES_GOLD] + list(XB_IDS) + list(PF_IDS))
ROUTE_COV = (40, 60, 80)
ROUTE_FIELDS = ("n_acc", "err", "ex_acc", "ex_per_asked", "cny_q", "cny_per_accepted")
PARETO_CLAIMS = ("plus_zs_vs_flash_gated", "plus_zs_vs_27b_gated", "plus_zs_vs_flash_fixed", "plus_zs_vs_27b_fixed", "plus_zs_vs_plus_gated", "plus_zs_vs_plus_fixed")
LABELS_E10 = ("L1", "L2", "L3", "L4", "L5", "L6", "nolabel")
NA_OK_SUFFIX = (".cause", ".dominates", ".reading", ".claim")


def dual_kind(i):
    """unit kind of the numeric fields (est, ci, item_ci, t_ci, loo_*) of dual id i."""
    if i.startswith("P1p_ratio") or i in ("XB3_ratio", "PF3b", "PF3b_q100"):
        return "x2"
    if i in ("P3", "P3_aurc") or i.endswith("C3_auroc") or i.endswith("C3_aurc") or i.split(".")[-2:-1] in (["C3_auroc"], ["C3_aurc"]):
        return "x3s"
    return "pp"


def expected_keys():
    """-> list of (key, kind). kind in: pct pp fpp x1 x2 x3 x3s int cny2 cny4 tok p str."""
    out = []
    A = out.append
    # --- loop
    for b in PB:
        for arm in ARMS:
            A(("loop.%s.ex.%s" % (b, arm), "pct"))
        for arm in ("gated", "fixed"):
            A(("loop.%s.rootcand.%s" % (b, arm), "pct"))
        for d in ("in", "btw"):
            for arm in ("gated", "fixed"):
                k = "loop.%s.eff_%s.%s" % (b, d, arm)
                A((k, "pp"))
                if b != "pooled":
                    A((k + ".p", "p")); A((k + ".b", "int")); A((k + ".c", "int"))
            A(("loop.%s.share_%s" % (b, d), "x2")); A(("loop.%s.share_%s.defined" % (b, d), "str"))
            A(("loop.%s.margin_over_eff_%s" % (b, d), "x2"))
            A(("loop.%s.claim_%s.gated" % (b, d), "str")); A(("loop.%s.claim_%s.fixed" % (b, d), "str"))
        A(("loop.%s.stack" % b, "pp")); A(("loop.%s.loop_btw_minus_in.gated" % b, "pp")); A(("loop.%s.loop_btw_minus_in.fixed" % b, "pp"))
    for arm in ("gated", "zs"):
        A(("loop.seedsd.plus." + arm, "pctv2"))
    # --- failing roots
    for b in BK:
        for k in ("n_items", "n_fail.gated", "n_fail.fixed", "both", "only_gated", "only_fixed", "diff.n", "diff.same_sql", "diff.diff_sql", "diff.timeout_involved",
                  "diff.exemplars_differ", "diff.final_ex_differs"):
            A(("roots.%s.%s" % (b, k), "int"))
        A(("roots.%s.cause" % b, "str"))
    A(("roots.cause", "str"))
    # --- cost
    for b in BK:
        for cfg in COST_CFG[b]:
            for f in COST_FIELDS:
                k = "cost.%s.%s.%s" % (b, cfg, f)
                A((k, {"calls": "x2", "tokens": "tok", "prompt_tok": "tok", "compl_tok": "tok", "cached_tok": "tok", "cny_q": "cny4", "cny_q_nodisc": "cny4", "cny_1000": "cny2",
                       "cny_1000_nodisc": "cny2", "embed_req": "x2", "rerank_req": "x2", "cny_emb_placeholder": "cny4", "lat_mean": "x1", "lat_p50": "x1", "lat_p95": "x1",
                       "ex": "pct", "n": "int"}[f]))
    for f, kd in (("calls", "x2"), ("requests", "x2"), ("completions", "x2"), ("cny_q", "cny4"), ("cny_q_nodisc", "cny4")):
        A(("cost.plus.route_extra." + f, kd))
    A(("cost.price_date", "str")); A(("cost.price_source", "str")); A(("cost.embed_rerank_included", "str"))
    A(("pareto.nondominated", "str")); A(("pareto.nondominated_k", "int"))
    for b in BK:
        for cfg in COST_CFG[b]:
            A(("pareto.%s.%s.dominated_by" % (b, cfg), "str"))
    for c in PARETO_CLAIMS:
        A(("pareto.claim.%s.dex" % c, "pp")); A(("pareto.claim.%s.cost_ratio" % c, "x2")); A(("pareto.claim.%s.dominates" % c, "str"))
    # --- dual
    for i in DUAL_IDS:
        for f in DUAL_FIELDS:
            uk = dual_kind(i)
            kd = {"est": uk, "n": "int", "g": "int", "ci": uk, "item_ci": uk, "t_ci": uk, "p_mcnemar": "p", "b": "int", "c": "int", "deff": "x2", "loo_min": uk,
                  "loo_max": uk, "rule": "str", "class_cluster": "str", "class_item": "str", "class_t": "str", "reading_changes": "str", "reading_changes_t": "str"}[f]
            A(("dual.%s.%s" % (i, f), kd))
    # --- routing
    for s in ("v3", "route_ca"):
        for cv in ROUTE_COV:
            for f in ROUTE_FIELDS:
                A(("route.%s.cov%d.%s" % (s, cv, f), {"n_acc": "int", "err": "pct", "ex_acc": "pct", "ex_per_asked": "pct", "cny_q": "cny4", "cny_per_accepted": "cny4"}[f]))
    for r in ("gated", "bonvote2", "bonvote3"):
        A(("route.ref.%s.ex" % r, "pct")); A(("route.ref.%s.cny_q" % r, "cny4"))
    A(("route.check.err60_route_ca", "pct")); A(("route.check.completions", "x2")); A(("route.check.requests", "x2"))
    # --- heterogeneity
    for fam in ("gain", "loop"):
        for b in BK:
            A(("het.%s.%s" % (fam, b), "pp"))
        for t, fields in (("T1", ("stat", "df1", "df2", "p_F", "p_chi2")), ("T2", ("stat", "df", "p"))):
            for f in fields:
                A(("het.%s.%s.%s" % (fam, t, f), {"stat": "x2", "df1": "int", "df2": "int", "df": "int", "p_F": "p", "p_chi2": "p", "p": "p"}[f]))
        for pr in ("flash_plus", "27b_plus", "27b_flash"):
            A(("het.%s.pair.%s" % (fam, pr), "pp"))
        A(("het.%s.reading" % fam, "str"))
    A(("het.spearman_dropped", "str"))
    # --- revisions
    A(("rev.def", "str"))
    for k in ("n", "helped", "hurt", "neutral"):
        A(("rev.e10.%s" % k, "int"))
    for k in ("n", "helped", "hurt", "neutral", "repair_helped", "repair_hurt", "repair_neutral"):
        A(("rev.all.%s" % k, "int"))
    for s in RES_SETS:
        for g in ("orig", "corr"):
            for k in ("n_items", "n_rev", "helped", "hurt", "neutral"):
                A(("rev.%s.%s.%s" % (s, g, k), "int"))
    for lab in LABELS_E10:
        for k in ("helped", "hurt", "neutral"):
            A(("rev.label.%s.%s" % (lab, k), "int"))
    A(("rev.sign_p", "p")); A(("rev.net", "pp")); A(("rev.annotation_in_counts", "str"))
    return out


if __name__ == "__main__":
    ks = expected_keys()
    print(len(ks), "base keys")
