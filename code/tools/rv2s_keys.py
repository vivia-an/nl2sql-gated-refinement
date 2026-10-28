"""RV2S key grammar: post hoc statistics add-on S2 (internal note S2, SAP section 25). Pure Python (no numpy), imported by make_xmacros.py (host) and by
rv2_small.py (worker machine). The summary file results/posthoc2/rv2s_summary.json must hold exactly these base keys (macro prefix `RV2.`):

  small.<id>.*          small-cluster readings of every dual-interval contrast of RV2 and of the loop effects (S1)
  small.pstar.*         the confirmatory family P* re-tested with exact sign-flip and t (S1)
  deff.*                design-effect audit (S2)
  margin.*              ratio of the 1.5 pp margin to the loop effect, with intervals (S3)
  cost.nodisc.*         cost ratios with and without the cache discount, Pareto front, route per correct answer, cached-token audit (S4)
  gate.*                gate-fire share and helped / hurt with sign test (S5)

Kinds are those of make_xmacros.fnum plus `p4` (four decimals)."""
import rv2_keys as K

BK, PB = K.BK, K.PB
DUAL_IDS = tuple(K.DUAL_IDS)
LOOP_IDS = tuple(["loop.%s.eff_%s.%s" % (b, d, a) for b in PB for d in ("in", "btw") for a in ("gated", "fixed")] + ["loop.%s.stack" % b for b in PB])
SMALL_IDS = DUAL_IDS + LOOP_IDS
CI_TYPES = ("cl", "item", "t", "tdb", "sf", "sfu", "wide", "wideu", "old")        # wide = hull(cl, item, t, sf) (same estimand), wideu = hull of the first six, old = hull(cl, item)
P_TYPES = ("t", "tdb", "sf", "sfu")
CLASS_TYPES = CI_TYPES
PSTAR_IDS = ("P1p_ex", "P1p_calls", "P1p", "W1", "P3")
PSTAR_FAMILY = ("P1p", "W1", "P3", "P5p")
DS = ("bird", "spider")
GATE_ARMS = ("gated", "fixed")
MARGIN_SENS = ("half_eff", "abs05")
COST_REFS = ("plus_zs", "plus_gated", "self_zs")
MODES = ("disc", "nodisc")
DEFF_SETS = ("all", "ex", "ratio", "g11", "g19", "cr")
SETUP4 = ("p1_plus", "w_plus", "w_flash", "w_27b")


def small_kind(i):
    if i.startswith("loop."):
        return "pp"
    return K.dual_kind(i)


def expected_keys():
    out = []
    A = out.append
    # ------------------------------------------------------------------ S1
    for i in SMALL_IDS:
        uk = small_kind(i)
        A(("small.%s.est" % i, uk)); A(("small.%s.n" % i, "int")); A(("small.%s.g" % i, "int"))
        A(("small.%s.method" % i, "str")); A(("small.%s.rule" % i, "str")); A(("small.%s.null" % i, uk))
        for t in CI_TYPES:
            A(("small.%s.%s_ci" % (i, t), uk))
        for t in P_TYPES:
            A(("small.%s.p_%s" % (i, t), "p4"))
        for t in CLASS_TYPES:
            A(("small.%s.class_%s" % (i, t), "str"))
        A(("small.%s.reading_changes" % i, "str")); A(("small.%s.reading_changes_u" % i, "str"))
    A(("small.def", "str"))
    for pid in PSTAR_IDS:
        for f in ("p_boot", "p_sf1", "p_sfu1", "p_t1") + (("p_item",) if pid in ("P1p", "W1", "P3") else ()):
            A(("small.pstar.%s.%s" % (pid, f), "p4"))
        A(("small.pstar.%s.ok_wide" % pid, "str"))
    for fam in PSTAR_FAMILY:
        A(("small.pstar.%s.p2_boot" % fam, "p4")); A(("small.pstar.%s.padj_boot" % fam, "p4"))
        A(("small.pstar.%s.p2_sf" % fam, "p4")); A(("small.pstar.%s.padj_sf" % fam, "p4"))
        A(("small.pstar.%s.p2_sfu" % fam, "p4")); A(("small.pstar.%s.padj_sfu" % fam, "p4"))
        A(("small.pstar.%s.p2_t" % fam, "p4")); A(("small.pstar.%s.padj_t" % fam, "p4"))
        A(("small.pstar.%s.holds_sf" % fam, "str")); A(("small.pstar.%s.holds_sfu" % fam, "str")); A(("small.pstar.%s.holds_t" % fam, "str"))
    A(("small.pstar.sf_min_p1", "p4")); A(("small.pstar.def", "str"))
    # ------------------------------------------------------------------ S2
    for s in DEFF_SETS:
        A(("deff.%s.n" % s, "int")); A(("deff.%s.min" % s, "x2")); A(("deff.%s.max" % s, "x2")); A(("deff.%s.median" % s, "x2"))
        A(("deff.%s.n_lt1" % s, "int")); A(("deff.%s.n_gt1" % s, "int")); A(("deff.%s.n_gt2" % s, "int"))
    A(("deff.all.min_id", "str")); A(("deff.all.max_id", "str")); A(("deff.all.ids_gt2", "str")); A(("deff.all.ids_lt0.5", "str")); A(("deff.all.n_lt0.5", "int"))
    A(("deff.all.share_lt1", "pct"))
    for tag in SETUP4:
        A(("deff.setup4.%s.b2000" % tag, "x2")); A(("deff.setup4.%s.b10000" % tag, "x2")); A(("deff.setup4.%s.stored" % tag, "x2"))
    A(("deff.setup4.min", "x2")); A(("deff.setup4.max", "x2")); A(("deff.setup4.source", "str"))
    A(("deff.source_of_larger_values", "str"))
    # ------------------------------------------------------------------ S3
    for b in PB:
        for d in ("in", "btw"):
            for a in ("fixed", "gated"):
                p = "margin.%s.%s.%s." % (b, d, a)
                A((p + "eff", "pp")); A((p + "ratio", "x2"))
                for t in ("cl", "item", "t", "sf", "wide"):
                    A((p + t + "_text", "str")); A((p + "lo_" + t, "x2")); A((p + "hi_" + t, "x2"))
                A((p + "p_nonpos_boot", "pct")); A((p + "wide_contains_1", "str"))
    for tag in MARGIN_SENS:
        p = "margin.sens.%s." % tag
        A((p + "margin", "pp"))
        for t in ("cl", "item", "t", "sf"):
            A((p + "lo_" + t, "pp")); A((p + "met_" + t, "str"))
        A((p + "met_wide", "str")); A((p + "lo_wide", "pp"))
    A(("margin.def", "str"))
    # ------------------------------------------------------------------ S4
    for b in BK:
        for cfg in K.COST_CFG[b]:
            for ref in COST_REFS:
                if ref == "plus_zs" and (b, cfg) == ("plus", "zs"):
                    continue
                if ref == "plus_gated" and (b, cfg) == ("plus", "gated"):
                    continue
                if ref == "self_zs" and cfg == "zs":
                    continue
                for m in MODES:
                    A(("cost.nodisc.ratio.%s.%s.vs_%s.%s" % (b, cfg, ref, m), "x2"))
            A(("cost.nodisc.cached.%s.%s.share" % (b, cfg), "pct"))
            A(("cost.nodisc.cached.%s.%s.rows_cached" % (b, cfg), "int"))
            A(("cost.nodisc.cached.%s.%s.rows" % (b, cfg), "int"))
            A(("cost.nodisc.%s.%s.dominated_by" % (b, cfg), "str"))
    for m in MODES:
        A(("cost.nodisc.front.%s.set" % m, "str")); A(("cost.nodisc.front.%s.k" % m, "int"))
    A(("cost.nodisc.front.changed", "str"))
    for c in K.PARETO_CLAIMS:
        A(("cost.nodisc.claim.%s.dominates" % c, "str"))
        A(("cost.nodisc.%s.cost_ratio" % c, "x2")); A(("cost.nodisc.%s.cost_ratio_disc" % c, "x2"))
    for b in BK:
        for arm in ("gated", "fixed"):             # the same six ratios under the name "<pipeline>_vs_plus_zs" (pipeline cost divided by the single-call cost)
            A(("cost.nodisc.%s_%s_vs_plus_zs.cost_ratio" % (b, arm), "x2")); A(("cost.nodisc.%s_%s_vs_plus_zs.cost_ratio_disc" % (b, arm), "x2"))
    for x in ("gated", "bonvote2", "bonvote3", "route"):
        for m in MODES:
            A(("cost.nodisc.route.cny_q.%s.%s" % (x, m), "cny4"))
    for x in ("route_ca60", "gated", "bonvote2", "bonvote3"):
        for m in MODES:
            A(("cost.nodisc.route.per_correct.%s.%s" % (x, m), "cny4"))
    for m in MODES:
        A(("cost.nodisc.route.per_correct.route_vs_gated.%s" % m, "x2"))
    A(("cost.nodisc.cached.note", "str")); A(("cost.nodisc.def", "str"))
    # ------------------------------------------------------------------ S5
    A(("gate.def", "str"))
    for ds in DS:
        for b in BK:
            for a in GATE_ARMS:
                p = "gate.%s.%s.%s." % (ds, b, a)
                A((p + "n", "int")); A((p + "fired", "int")); A((p + "share", "pct"))
    for ds, defs in (("bird", ("in", "btw")), ("spider", ("btw",))):
        for b in BK:
            for a in GATE_ARMS:
                for d in defs:
                    p = "gate.hh.%s.%s.%s.%s." % (ds, b, a, d)
                    A((p + "b", "int")); A((p + "c", "int")); A((p + "p_two", "p")); A((p + "p_one", "p")); A((p + "net", "pp"))
    A(("gate.hh.def", "str"))
    return out


if __name__ == "__main__":
    ks = expected_keys()
    assert len({k for k, _ in ks}) == len(ks), "duplicate keys"
    print(len(ks), "base keys")
