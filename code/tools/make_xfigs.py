#!/usr/bin/env python3
"""make_xfigs.py -- new figures and supplement tables for the final manuscript (task FIGS).

Everything plotted or tabulated is READ from results files; no number is typed here (the only constants are plan constants:
the P1' margin, figure widths, panel layout).  Each figure writes its data block to figs/data/<stem>.json for provenance.

  python code/tools/make_xfigs.py data     # figs/data/fig_corrected_gold.json, fig_critic_diag.json, fig_frontier.json (pure python, runs on the host)
  python code/tools/make_xfigs.py tables   # paper/supp_tables/*.tex (pure python, runs on the host)
  python code/tools/make_xfigs.py figs     # figs/fig_corrected_gold, fig_critic_diag, fig_frontier .{pdf,png}  (needs matplotlib: worker machine)
  python code/tools/make_xfigs.py all      # all of the above

Sources:  results/robust/rescore.json (RESCORE, scope robustness-posthoc), results/diag/{critic_diag_fig,critic_diag_summary,critic_v2_summary,
critic_ve_summary}.json (exploratory-posthoc, simulated), results/posthoc2/{cost,pareto}.csv (RV2, post hoc), results/ext1/* (external-validity), results/e10/* (LLM annotation), docs/SAP.md (amendment log).
Figure size: MDPI full width 17.5 cm (paper/preamble.tex \\figincl prints wide figures at 17.5 cm), drawn at that size so that 8 pt text prints at 8 pt.
Style: code/analysis_pipeline/figures/style.py (imported, not edited): DejaVu Sans 8 pt, Okabe-Ito, Type 42 fonts.
"""
import argparse
import csv
import gzip  # noqa: F401  (kept for parity with sibling tools)
import hashlib
import importlib.util
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RES = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figs")
SUPP = os.path.join(ROOT, "paper", "supp_tables")
CM = 1.0 / 2.54
FULL_W_CM = 17.5          # MDPI full width (paper/preamble.tex prints wide figures at 17.5 cm)
P1_MARGIN_PP = -1.5       # plan constant: non-inferiority margin of P1' (SAP 13.3 / 15.3)

# the backbone order: strong first (control), then the weak ones, pooled last
H1_ROWS = [("plus", "H1.plus"), ("q27b", "H1.q27b"), ("flash", "H1.flash"), ("pooled", "H1.pooled3")]
H2_ROWS = [("plus", "H2.plus_control"), ("q27b", "H2.q27b"), ("flash", "H2.flash"), ("pooled", "H2.pooled_weak")]
ROW_LABEL = {("H1", "plus"): "Plus (strong)", ("H1", "q27b"): "27B", ("H1", "flash"): "Flash", ("H1", "pooled"): "Pooled, all 3",
             ("H2", "plus"): "Plus (control)", ("H2", "q27b"): "27B", ("H2", "flash"): "Flash", ("H2", "pooled"): "Pooled, 27B + Flash"}
VARIANTS = ["V0", "VA", "VB", "VC", "VD"]
VARIANT_LABEL = {"V0": "V0 (pipeline)", "VA": "VA (grounded)", "VB": "VB (+ memory)", "VC": "VC (+ verifier)", "VD": "VD (+ appeal)"}
VE_ARMS = ["none", "obey", "A", "A0", "B", "C"]


# ----------------------------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------------------------
def jload(*parts):
    p = os.path.join(*parts)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def need(d, *keys):
    """d[k1][k2]...; a missing key raises with the path (no silent defaults: a figure must not invent a number)."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError("missing key %s in results (at %r)" % (".".join(map(str, keys)), k))
        cur = cur[k]
    return cur


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def pct(x):
    return 100.0 * x


# ----------------------------------------------------------------------------------------------------------------------------------
# data block 1: corrected gold forest
# ----------------------------------------------------------------------------------------------------------------------------------
def corrected_gold_data(res=RES):
    src = os.path.join(res, "robust", "rescore.json")
    r = jload(src)
    meta = need(r, "_meta")
    g1 = need(r, "variants", "G1", "sets")
    out = {"_meta": {"scope": need(meta, "scope"), "primary": need(meta, "primary"), "B": need(meta, "B"), "seed": need(meta, "seed"),
                     "source": "results/robust/rescore.json", "source_sha256": sha256(src), "script": "code/tools/make_xfigs.py",
                     "p1_margin_pp": P1_MARGIN_PP, "unit": "percentage points of EX",
                     "interval": "95% database-clustered paired percentile bootstrap, one resample shared by original and corrected gold",
                     "reading": "robustness-posthoc; rule of SAP 20.7: robust = sign unchanged and intervals overlap; stable = shift interval contains 0 or |shift| < 1 pp"},
           "sets": {}}
    for set_name in ("S_same", "S_all"):
        ana = need(g1, set_name, "analysis", "contrasts")
        head = need(r, "headline", set_name)
        panels = {}
        for pname, rows in (("H1", H1_ROWS), ("H2", H2_ROWS)):
            lst = []
            for short, key in rows:
                c = need(ana, key)
                conc = "C1" if pname == "H1" else "C2"
                flags = need(head, conc, "contrasts", key)
                lst.append({"key": key, "row": short, "desc": need(c, "desc"), "n": need(c, "n"), "n_dbs": need(c, "n_dbs"),
                            "orig": need(c, "d0"), "orig_ci": list(need(c, "ci0")), "corr": need(c, "d1"), "corr_ci": list(need(c, "ci1")),
                            "shift": need(c, "shift"), "shift_ci": list(need(c, "ci_shift")),
                            "robust": need(flags, "robust"), "stable": need(flags, "stable")})
            panels[pname] = lst
        out["sets"][set_name] = panels
    return out


# ----------------------------------------------------------------------------------------------------------------------------------
# data block 2: critic diagnostic
# ----------------------------------------------------------------------------------------------------------------------------------
def critic_diag_data(res=RES):
    d = os.path.join(res, "diag")
    v2 = jload(d, "critic_v2_summary.json")
    ve = jload(d, "critic_ve_summary.json")
    meta2, meta_ve = need(v2, "_meta"), need(ve, "_meta")
    a = {"n_injected": need(meta2, "n_injected"), "n_gold_controls": need(meta2, "n_gold_controls"), "rows": []}
    for v in VARIANTS:
        p, c = need(v2, "variants", v, "point"), need(v2, "variants", v, "ci")
        a["rows"].append({"variant": v, "label": VARIANT_LABEL[v],
                          "detection": pct(need(p, "detection")), "detection_ci": [pct(need(c, "detection", "lo")), pct(need(c, "detection", "hi"))],
                          "fpr": pct(need(p, "fpr")), "fpr_ci": [pct(need(c, "fpr", "lo")), pct(need(c, "fpr", "hi"))]})
    vc, vd = {k: a["rows"][3][k] for k in ("detection", "fpr")}, {k: a["rows"][4][k] for k in ("detection", "fpr")}
    a["vd_equals_vc"] = vc == vd
    qs = need(meta_ve, "q")
    units = need(ve, "units_count")
    n_right = need(units, "F-inj-right")
    n_wrong = need(units, "F-inj-wrong") + need(units, "F-gold-wrong")      # wrong claims: injected items + gold controls (WCR denominator)
    b = {"q": qs, "n_correct_claims": n_right, "n_wrong_claims": n_wrong, "arms": {}}
    maxdiff = 0.0
    for arm in VE_ARMS:
        rows = []
        for q in qs:
            qk = "%.2f" % q
            pt, ci = need(ve, "by_q", qk, "point", arm), need(ve, "by_q", qk, "ci", arm)
            rows.append({"q": q, "ccr": pct(need(pt, "CCR")), "ccr_ci": [pct(need(ci, "CCR", "lo")), pct(need(ci, "CCR", "hi"))],
                         "wcr": pct(need(pt, "WCR")), "wcr_ci": [pct(need(ci, "WCR", "lo")), pct(need(ci, "WCR", "hi"))]})
        b["arms"][arm] = rows
    for i in range(len(qs)):          # arm C (old VD prompt) is omitted from the plot because it overlaps 'obey'; the overlap is measured here
        for m in ("ccr", "wcr"):
            maxdiff = max(maxdiff, abs(b["arms"]["C"][i][m] - b["arms"]["obey"][i][m]))
    b["arm_C_max_abs_diff_to_obey_pp"] = maxdiff
    return {"_meta": {"scope": need(meta2, "reading"), "ve_label": need(meta_ve, "label"), "script": "code/tools/make_xfigs.py",
                      "source": ["results/diag/critic_v2_summary.json", "results/diag/critic_ve_summary.json"],
                      "source_sha256": {n: sha256(os.path.join(d, n)) for n in ("critic_v2_summary.json", "critic_ve_summary.json")},
                      "interval": need(meta_ve, "ci"), "q_rule": need(meta_ve, "q_rule"),
                      "unit": "percent", "arms": need(meta_ve, "arms")},
            "panel_a": a, "panel_bc": b}


# ----------------------------------------------------------------------------------------------------------------------------------
# data block 3: cost-accuracy frontier (RV2, results/posthoc2)
# ----------------------------------------------------------------------------------------------------------------------------------
FRONTIER_BACKBONES = [("plus", "Qwen3.7-Plus"), ("flash", "Qwen3.7-Flash"), ("27b", "Qwen3.5-27B")]
FRONTIER_CFG = [("zs", "single call"), ("rootonly", "root-only"), ("gated", "gated loop"), ("blind", "blind retry"), ("fixed", "fixed loop"),
                ("route", "gated loop + routing score"), ("acsql", "AC-SQL"), ("macsql", "MAC-SQL"), ("sdebug", "Self-Debug"),
                ("bonvote2", "best-of-2 vote"), ("bonvote3", "best-of-3 vote")]


def _dominated(points, cost_key):
    """Point-estimate dominance (RV2 / SAP 23): b dominates a if cost_b <= cost_a and ex_b >= ex_a with at least one strict.  Returns the set of (backbone, cfg) not dominated."""
    keep = set()
    for a in points:
        dom = False
        for b in points:
            if b is a:
                continue
            if b[cost_key] <= a[cost_key] and b["ex"] >= a["ex"] and (b[cost_key] < a[cost_key] or b["ex"] > a["ex"]):
                dom = True
                break
        if not dom:
            keep.add((a["backbone"], a["cfg"]))
    return keep


def frontier_data(res=RES):
    d = os.path.join(res, "posthoc2")
    cp, pp, sp = os.path.join(d, "cost.csv"), os.path.join(d, "pareto.csv"), os.path.join(d, "rv2_summary.json")
    with open(cp, encoding="utf-8") as f:
        cost = list(csv.DictReader(f))
    with open(pp, encoding="utf-8") as f:
        par = {(r["backbone"], r["cfg"]): r for r in csv.DictReader(f)}
    summ = jload(sp)
    meta = need(summ, "_meta")
    pts = []
    for r in cost:
        key = (r["backbone"], r["cfg"])
        if key not in par:
            raise KeyError("pareto.csv has no row for %s" % (key,))
        pts.append({"backbone": r["backbone"], "cfg": r["cfg"], "n": int(r["n"]), "ex": pct(float(r["ex"])), "ex_ci": [pct(float(r["ex_lo"])), pct(float(r["ex_hi"]))],
                    "cny_1000": float(r["cny_1000"]), "cny_1000_nodisc": 1000.0 * float(r["cny_q_nodisc"]), "nondominated_rv2": par[key]["nondominated"] == "1"})
    for p in pts:                                                                          # internal consistency: the published front is reproduced from the same numbers
        assert abs(p["cny_1000"] - 1000.0 * float(par[(p["backbone"], p["cfg"])]["cny_q"])) < 1e-9
    front = _dominated(pts, "cny_1000")
    pub = set(k for k, v in par.items() if v["nondominated"] == "1")
    if front != pub:
        raise AssertionError("recomputed front %s differs from results/posthoc2/pareto.csv %s" % (sorted(front), sorted(pub)))
    front_nd = _dominated(pts, "cny_1000_nodisc")
    return {"_meta": {"scope": need(meta, "scope"), "sap": need(meta, "sap"), "B": need(meta, "B"), "seed": need(meta, "seed"), "price_date": need(meta, "price_date"),
                      "n_dev": need(meta, "n_dev"), "n_clusters_dev": need(meta, "n_clusters_dev"),
                      "sources": {"results/posthoc2/cost.csv": sha256(cp), "results/posthoc2/pareto.csv": sha256(pp), "results/posthoc2/rv2_summary.json": sha256(sp)},
                      "script": "code/tools/make_xfigs.py", "unit_x": "CNY per 1000 questions, list price (chat models only; embedding and rerank not priced)",
                      "unit_y": "EX, % of 1534 BIRD dev questions", "interval": "95% database-clustered percentile bootstrap (11 clusters)",
                      "layout": "round 4: (a) with the cache discount and (b) full price show the 12 study configurations (3 backbones x single call, root-only, gated, fixed); ring = not dominated among all 21 on that price basis; (c) Plus configurations and comparators as a dot plot, cost printed, bold = not dominated; root-only and gated markers are drawn 4 pt apart horizontally (display only)",
                      "x_discount": "panel a: cost with the cache discount (cny_q); panel b: every prompt token at the full input price (cny_q_nodisc); 27b records no cached tokens (its points coincide in the two panels); flash records cached tokens only in the single call and the fixed loop (small shift except the single call)",
                      "front": "point-estimate Pareto over the 21 configurations; panel a front = cost with the discount (RV2 definition; recomputed here, equal to pareto.csv); panel b front = same rule on the cost without the discount (not in RV2; computed here)",
                      "front_without_discount_same_set": front_nd == front, "front_without_discount": sorted("%s %s" % k for k in front_nd)},
            "points": pts, "front": sorted("%s %s" % k for k in front), "front_nodisc": sorted("%s %s" % k for k in front_nd)}


# ----------------------------------------------------------------------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------------------------------------------------------------------
def _style():
    path = os.path.join(ROOT, "code", "analysis_pipeline", "figures", "style.py")
    spec = importlib.util.spec_from_file_location("fig_style", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _save(fig, stem, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, "%s.%s" % (stem, ext))
        fig.savefig(p)                      # fixed figure size, no bbox tightening: printed width = FULL_W_CM exactly
        paths.append(p)
    return paths


def _signed(v, d=1):
    s = ("%+." + str(d) + "f") % v
    return s.replace("-", "−")


def draw_corrected_gold(data, out_dir=FIGS, height_cm=6.8):
    st = _style()
    plt = st.plt()
    from matplotlib.lines import Line2D
    from matplotlib.transforms import blended_transform_factory
    ss = data["sets"]["S_same"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W_CM * CM, height_cm * CM), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.10)
    titles = {"H1": "Gated minus fixed loop", "H2": "Gated minus single call"}
    for ax, pname, letter in zip(axes, ("H1", "H2"), ("a", "b")):
        rows = ss[pname]
        ys = [3.6, 2.6, 1.6, 0.35]
        for r, y in zip(rows, ys):
            pooled = r["row"] == "pooled"
            for key, dy, mk, col, fc in (("orig", +0.17, "o", st.DGREY, "white"), ("corr", -0.17, "D", st.VERM, st.VERM)):
                est, ci = r[key], r[key + "_ci"]
                ax.plot(ci, [y + dy, y + dy], color=col, lw=1.2 if not pooled else 1.8, solid_capstyle="butt", zorder=2)
                ax.plot([ci[0], ci[0]], [y + dy - 0.06, y + dy + 0.06], color=col, lw=0.9, zorder=2)
                ax.plot([ci[1], ci[1]], [y + dy - 0.06, y + dy + 0.06], color=col, lw=0.9, zorder=2)
                ax.plot([est], [y + dy], marker=mk, ms=4.3 if mk == "o" else 4.0, mfc=fc, mec=col, mew=1.0, ls="none", zorder=3)
            tr = blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(1.02, y - 0.17, _signed(r["corr"]), transform=tr, ha="left", va="center", fontsize=8, color=st.VERM)
        ax.axvline(0, color=st.INK, lw=0.8, zorder=1)
        if pname == "H1":
            ax.axvline(data["_meta"]["p1_margin_pp"], color=st.GREY, lw=0.8, ls=":", zorder=1)
            ax.text(data["_meta"]["p1_margin_pp"] - 0.2, 4.25, "margin %s" % _signed(data["_meta"]["p1_margin_pp"]), ha="right", va="bottom", fontsize=8, color=st.DGREY)
        ax.set_yticks(ys)
        ax.set_yticklabels([ROW_LABEL[(pname, r["row"])] for r in rows])
        for lab, r in zip(ax.get_yticklabels(), rows):
            if r["row"] == "pooled":
                lab.set_fontweight("bold")
        ax.set_ylim(-0.35, 4.75)
        ax.set_xlim(-7.5, 7.5)
        ax.set_xlabel("EX difference (pp), n = %d" % rows[0]["n"])
        ax.axhline(1.0, color=st.LGREY, lw=0.6, zorder=0)
        st.grid_x(ax)
        st.direction_note(ax, "worse", "better", y=-0.30)
        st.panel_title(ax, letter, titles[pname])
    handles = [Line2D([0], [0], color=st.DGREY, marker="o", mfc="white", mec=st.DGREY, ms=4.3, lw=1.2, label="original reference queries"),
               Line2D([0], [0], color=st.VERM, marker="D", mfc=st.VERM, mec=st.VERM, ms=4.0, lw=1.2, label="corrected reference queries"),
               Line2D([0], [0], color=st.GREY, ls=":", lw=0.8, label="plan margin of P1′")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, fontsize=8, handlelength=2.2, columnspacing=1.4)
    return _save(fig, "fig_corrected_gold", out_dir)


def draw_critic_diag(data, out_dir=FIGS, height_cm=7.2):
    st = _style()
    plt = st.plt()
    from matplotlib.lines import Line2D
    A, BC = data["panel_a"], data["panel_bc"]
    fig = plt.figure(figsize=(FULL_W_CM * CM, height_cm * CM), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.07)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.92, 1.04, 1.04])
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) detection and false-positive rate of V0 and VA-VD
    ys = list(range(len(A["rows"])))[::-1]
    for r, y in zip(A["rows"], ys):
        for key, dy, mk, col, fc in (("detection", 0.15, "o", st.INK, st.INK), ("fpr", -0.15, "D", st.PURPLE, st.PURPLE)):
            ci = r[key + "_ci"]
            axa.plot(ci, [y + dy, y + dy], color=col, lw=1.2, solid_capstyle="butt", zorder=2)
            axa.plot([r[key]], [y + dy], marker=mk, ms=4.2, mfc=fc, mec=col, ls="none", zorder=3)
    axa.set_yticks(ys)
    axa.set_yticklabels([r["label"] for r in A["rows"]])
    axa.set_xlim(0, 100)
    axa.set_ylim(-0.6, len(ys) - 0.4)
    axa.set_xlabel("Rate (%)")
    st.grid_x(axa)
    st.panel_title(axa, "a", "Injected errors")
    axa.text(0.62, 0.04, "exploratory", transform=axa.transAxes, ha="center", va="bottom", fontsize=8, color=st.REFCOL, style="italic")

    # (b), (c) VE arbitration vs simulated user accuracy q
    spec = {"A": (st.BLUE, "o", -0.012, "-"), "A0": (st.SKY, "v", 0.0, "-"), "B": (st.GREEN, "s", 0.012, "-")}
    qs = BC["q"]
    for ax, key, ttl, letter in ((axb, "ccr", "Correct claims kept", "b"), (axc, "wcr", "Wrong claims dropped", "c")):
        ax.plot(qs, [r[key] for r in BC["arms"]["obey"]], color=st.INK, ls=":", lw=1.2, zorder=1)
        ax.plot(qs, [r[key] for r in BC["arms"]["none"]], color=st.GREY, ls="--", lw=1.2, zorder=1)
        for arm, (col, mk, dx, ls) in spec.items():
            rows = BC["arms"][arm]
            xs = [q + dx for q in qs]
            ax.plot(xs, [r[key] for r in rows], color=col, lw=1.2, ls=ls, zorder=2)
            for x, r in zip(xs, rows):
                ax.plot([x, x], r[key + "_ci"], color=col, lw=0.9, zorder=2)
            ax.plot(xs, [r[key] for r in rows], ls="none", marker=mk, ms=4.2, mfc=col, mec=col if arm != "A0" else st.BLUE, zorder=3)
        ax.set_xlim(0.66, 1.04)
        ax.set_xticks(qs)
        ax.set_xticklabels(["%.2f" % q for q in qs])
        ax.set_xlabel("User accuracy $q$")
        st.grid_y(ax)
        st.panel_title(ax, letter, ttl)
    axb.set_ylim(64, 102)
    axb.set_ylabel("Kept (%% of %d claims)" % BC["n_correct_claims"])
    axb.text(0.97, 0.04, "simulated user", transform=axb.transAxes, ha="right", va="bottom", fontsize=8, color=st.REFCOL, style="italic")
    axc.set_ylim(-4, 104)
    axc.set_ylabel("Dropped (%% of %d claims)" % BC["n_wrong_claims"])
    axc.text(0.03, 0.96, "simulated user", transform=axc.transAxes, ha="left", va="top", fontsize=8, color=st.REFCOL, style="italic")

    h1 = [Line2D([0], [0], color=st.INK, marker="o", ms=4.2, lw=1.2, label="detection, injected (n = %d)" % A["n_injected"]),
          Line2D([0], [0], color=st.PURPLE, marker="D", ms=4.2, lw=1.2, label="false positives, reference controls (n = %d)" % A["n_gold_controls"])]
    h2 = [Line2D([0], [0], color=st.BLUE, marker="o", ms=4.2, lw=1.2, label="A (same model)"),
          Line2D([0], [0], color=st.SKY, marker="v", ms=4.2, lw=1.2, label="A, no user statement"),
          Line2D([0], [0], color=st.GREEN, marker="s", ms=4.2, lw=1.2, label="B (other model)"),
          Line2D([0], [0], color=st.INK, ls=":", lw=1.2, label="obey the user"),
          Line2D([0], [0], color=st.GREY, ls="--", lw=1.2, label="no appeal")]
    l1 = fig.legend(handles=h1, loc="outside upper left", ncol=1, frameon=False, fontsize=8, handlelength=2.0, bbox_to_anchor=(0.0, 1.0))
    fig.legend(handles=h2, loc="outside upper right", ncol=3, frameon=False, fontsize=8, handlelength=2.0, columnspacing=1.0)
    del l1
    return _save(fig, "fig_critic_diag", out_dir)


# Round 4 (R3 N-3, R2 N7): Figure 2 is split in two.  Top row: the twelve configurations of the study (three backbones x single call, root-only, gated loop, fixed loop), one panel
# per price basis ("with cache discount" / "full price"); colour = backbone, shape = configuration, a black ring = not dominated by any of the 21 configurations on that price basis;
# text labels only for the backbones and the two configurations the text compares.  Bottom: the Plus comparators as a dot plot (EX with its interval, cost printed on the right, bold =
# on the front for that price basis).  The two configurations whose costs differ by a few per cent (root-only, gated loop) are drawn FRONTIER_DX points apart horizontally (display only).
FRONTIER_MARK = {"zs": "o", "rootonly": "s", "gated": "^", "fixed": "D"}
FRONTIER_CFG_LAB = {"zs": "single call", "rootonly": "root-only", "gated": "gated loop", "fixed": "fixed loop"}
FRONTIER_OTHER = ("blind", "route", "acsql", "macsql", "sdebug", "bonvote2", "bonvote3")
FRONTIER_OTHER_LAB = {"blind": "blind retry", "route": "gated loop + routing score", "acsql": "AC-SQL", "macsql": "MAC-SQL", "sdebug": "Self-Debug",
                      "bonvote2": "best-of-2 vote", "bonvote3": "best-of-3 vote"}
FRONTIER_SIZE = {"o": 7.0, "s": 6.2, "^": 7.2, "D": 5.8, "x": 6.4}
FRONTIER_DX = {"rootonly": -4.0, "gated": 4.0}      # points; display offset of the two nearly coincident configurations
FRONTIER_BB_LAB = {"plus": "Plus", "flash": "Flash", "27b": "27B"}
# text in the top panels: (text, anchor key or None for the backbone name, x, y in data coordinates, colour key); leader lines run from the text to the anchor point
FRONTIER_HA = {"a": {"flash": "center", "plus": "right", "27b": "center"}, "b": {"flash": "center", "27b": "right", "plus": "left"}}   # keeps 27B and Plus names apart
FRONTIER_TEXT = [("Flash", ("flash", "zs"), "flash"), ("27B", ("27b", "gated"), "27b"), ("Plus", ("plus", "zs"), "plus")]   # backbone names, centred above the top of one interval bar


def draw_frontier(data, out_dir=FIGS, height_cm=11.3):
    st = _style()
    plt = st.plt()
    from matplotlib.lines import Line2D
    from matplotlib.transforms import ScaledTranslation
    col = {"plus": st.BLUE, "flash": st.ORANGE, "27b": st.GREEN}
    pts = data["points"]
    byk = dict(((p["backbone"], p["cfg"]), p) for p in pts)
    fig = plt.figure(figsize=(FULL_W_CM * CM, height_cm * CM), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.06, wspace=0.05, hspace=0.02)
    gs = fig.add_gridspec(2, 12, height_ratios=[1.28, 1.0])
    ax_a, ax_b = fig.add_subplot(gs[0, 0:6]), fig.add_subplot(gs[0, 6:12], sharey=None)
    ax_l, ax_c, ax_t = fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:9], sharey=None), fig.add_subplot(gs[1, 9:12])
    ylo = min(p["ex_ci"][0] for p in pts) - 1.2
    yhi = max(p["ex_ci"][1] for p in pts) + 1.2
    ytop = yhi + 1.6

    def shifted(ax, cfg):
        return ax.transData + ScaledTranslation(FRONTIER_DX.get(cfg, 0.0) / 72.0, 0.0, fig.dpi_scale_trans)

    for ax, basis, letter, ttl in ((ax_a, "disc", "a", "With the cache discount"), (ax_b, "nodisc", "b", "Full price")):
        ckey = "cny_1000" if basis == "disc" else "cny_1000_nodisc"
        front = set(data["front"] if basis == "disc" else data["front_nodisc"])
        study = [p for p in pts if p["cfg"] in FRONTIER_MARK]
        for p in study:                                                                            # interval bars first (95%, database-clustered)
            tr = shifted(ax, p["cfg"])
            ax.plot([p[ckey]] * 2, p["ex_ci"], color=col[p["backbone"]], lw=1.0, alpha=0.75, solid_capstyle="butt", transform=tr, zorder=1.5, clip_on=False)
            ax.plot([p[ckey]] * 2, p["ex_ci"], ls="none", marker="_", ms=3.6, mew=1.0, color=col[p["backbone"]], alpha=0.75, transform=tr, zorder=1.5, clip_on=False)
        for p in sorted(study, key=lambda p: ("%s %s" % (p["backbone"], p["cfg"]) in front)):
            m = FRONTIER_MARK[p["cfg"]]
            on = "%s %s" % (p["backbone"], p["cfg"]) in front
            ax.plot([p[ckey]], [p["ex"]], marker=m, ms=FRONTIER_SIZE[m], mfc=col[p["backbone"]], mec=st.INK if on else "white", mew=1.5 if on else 0.6, ls="none",
                    transform=shifted(ax, p["cfg"]), zorder=4 if on else 3, clip_on=False)
        for txt, key, ck in FRONTIER_TEXT:
            p = byk[key]
            ax.text(p[ckey], p["ex_ci"][1] + 0.5, txt, ha=FRONTIER_HA[letter][ck], va="bottom", fontsize=8, color=col[ck], fontweight="bold", zorder=6)
        ax.set_xscale("log")
        xt = [0.2, 0.5, 1, 2, 5, 10, 20]
        ax.set_xticks(xt)
        ax.set_xticklabels(["%g" % t for t in xt])
        ax.minorticks_off()
        ax.set_xlim(0.15, 30)
        ax.set_ylim(ylo, ytop)
        ax.set_xlabel("Cost, CNY per 1000 questions (log scale)")
        st.grid_y(ax)
        st.panel_title(ax, letter, ttl)
    ax_a.set_ylabel("Execution accuracy on BIRD dev (%)")
    # ---- panel c: Plus, the study configurations as a reference and the comparators; one row each, ordered by the cost with the discount
    rows = [("plus", c) for c in ("zs", "gated", "fixed")] + sorted([("plus", c) for c in FRONTIER_OTHER if ("plus", c) in byk], key=lambda k: byk[k]["cny_1000"])
    n = len(rows)
    ys = {k: n - 1 - i for i, k in enumerate(rows)}
    fa, fb = set(data["front"]), set(data["front_nodisc"])
    for k in rows:
        p, y = byk[k], ys[k]
        ax_c.plot(p["ex_ci"], [y, y], color=st.BLUE, lw=1.3, alpha=0.8, solid_capstyle="butt", zorder=2)
        ax_c.plot(p["ex_ci"], [y, y], ls="none", marker="|", ms=4.5, mew=1.1, color=st.BLUE, alpha=0.8, zorder=2)
        if k[1] in FRONTIER_MARK:
            m = FRONTIER_MARK[k[1]]
            ax_c.plot([p["ex"]], [y], marker=m, ms=FRONTIER_SIZE[m], mfc=st.BLUE, mec="white", mew=0.6, ls="none", zorder=4)
        else:
            ax_c.plot([p["ex"]], [y], marker="x", ms=FRONTIER_SIZE["x"], mec=st.BLUE, mew=1.6, ls="none", zorder=4)
        lab = ("Plus " + FRONTIER_CFG_LAB[k[1]]) if k[1] in FRONTIER_MARK else FRONTIER_OTHER_LAB[k[1]]
        ax_l.text(1.0, y, lab, ha="right", va="center", fontsize=8, color=st.INK)
        key = "%s %s" % k
        ax_t.text(0.55, y, "%.2f" % p["cny_1000"], ha="right", va="center", fontsize=8, fontweight="bold" if key in fa else "normal", color=st.INK)
        ax_t.text(1.02, y, "%.2f" % p["cny_1000_nodisc"], ha="right", va="center", fontsize=8, fontweight="bold" if key in fb else "normal", color=st.INK)
    ax_c.axhline(n - 3.5, color=st.LGREY, lw=0.7)
    ax_c.set_ylim(-0.7, n - 0.3)
    ax_t.set_ylim(-0.7, n - 0.3)
    ax_c.set_yticks([])
    ax_l.set_ylim(-0.7, n - 0.3)
    ax_l.axis("off")
    ax_c.set_xlim(ylo, yhi)
    ax_c.set_xticks([55, 60, 65, 70, 75])
    ax_c.spines["left"].set_visible(False)
    ax_c.set_xlabel("Execution accuracy on BIRD dev (%), 95% interval")
    st.grid_x(ax_c)
    ax_t.axis("off")
    ax_t.text(0.55, n - 0.3 + 0.1, "discount", ha="right", va="bottom", fontsize=8, color=st.DGREY)
    ax_t.text(1.02, n - 0.3 + 0.1, "full price", ha="right", va="bottom", fontsize=8, color=st.DGREY)
    st.panel_title(ax_c, "c", "Plus: configurations and comparators", pad=4)
    ax_t.text(0.0, n - 0.3 + 1.45, "Cost, CNY per 1000", ha="left", va="bottom", fontsize=8, color=st.INK)      # round 6: clear of the two column headers below it
    # ---- legend: colours and shapes in one block
    gray = dict(mfc="#bbbbbb", mec=st.DGREY, mew=0.7, ls="none")
    hb = [Line2D([], [], marker="o", ls="none", ms=6.5, mfc=col[k], mec="white", label=lab) for k, lab in FRONTIER_BACKBONES]
    hs = [Line2D([], [], marker="o", ms=6.0, label="single call", **gray), Line2D([], [], marker="s", ms=5.5, label="root-only", **gray),
          Line2D([], [], marker="^", ms=6.5, label="gated loop", **gray), Line2D([], [], marker="D", ms=5.0, label="fixed loop", **gray)]
    hx = [Line2D([], [], marker="o", ms=6.0, mfc="#bbbbbb", mec=st.INK, mew=1.5, ls="none", label="not dominated"),
          Line2D([], [], marker="x", ms=6.0, mec=st.DGREY, mew=1.5, ls="none", label="other comparator"),
          Line2D([], [], color=st.DGREY, lw=1.0, label="95% interval")]
    handles = [hb[0], hs[0], hb[1], hs[1], hb[2], hs[2], hx[0], hs[3], hx[2], hx[1]]
    fig.legend(handles=handles, loc="outside upper center", ncol=5, frameon=False, fontsize=8, handlelength=1.4, columnspacing=1.2, handletextpad=0.4, labelspacing=0.3)
    return _save(fig, "fig_frontier", out_dir)


# ----------------------------------------------------------------------------------------------------------------------------------
# supplement tables (pure python; LaTeX bodies for \input, written to paper/supp_tables/)
# ----------------------------------------------------------------------------------------------------------------------------------
MINUS = "\\ensuremath{-}"


def tex_escape(s):
    """Escape plain text (and convert the few markdown / unicode leftovers of docs/SAP.md) for LaTeX."""
    s = s.replace("\\*", "*").replace("\\_", "_").replace("`", "")
    uni = {"→": "$\\rightarrow$", "′": "'", "≥": "$\\geq$", "≤": "$\\leq$", "§": "\\S{}", "κ": "$\\kappa$",
           "×": "$\\times$", "−": "-", "–": "--", "—": "---", "±": "$\\pm$", "→ ": "$\\rightarrow$ "}
    out = []
    for ch in s:
        if ch in uni:
            out.append(uni[ch])
        elif ch == "_":
            out.append("\\_\\allowbreak{}")
        elif ch in "&%#{}":
            out.append("\\" + ch)
        elif ch == "$":
            out.append("\\$")
        elif ch == "~":
            out.append("\\textasciitilde{}")
        elif ch == "^":
            out.append("\\textasciicircum{}")
        elif ch == "\\":
            out.append("\\textbackslash{}")
        elif ord(ch) > 127:
            raise ValueError("unmapped non-ASCII character %r in %r" % (ch, s[:60]))
        else:
            out.append(ch)
    return "".join(out)


def num(x, d=1, sign=False):
    s = ("%+." if sign else "%.") + str(d) + "f"
    return (s % x).replace("-", MINUS)


def ci(lo, hi, d=1, sign=True):
    return "[%s, %s]" % (num(lo, d, sign), num(hi, d, sign))


def est_ci(x, lo, hi, d=1, sign=True):
    return "%s %s" % (num(x, d, sign), ci(lo, hi, d, sign))


def rate_ci(rate, lo, hi, d=1):
    return "%s %s" % (num(pct(rate), d), ci(pct(lo), pct(hi), d, False))


def table_rows(rows):
    return "\n".join(" & ".join(r) + " \\\\" for r in rows)


def write_tex(name, body, out_dir=None):
    out_dir = out_dir or SUPP
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name + ".tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write("%% GENERATED by code/tools/make_xfigs.py (tables) -- do not edit. Rows of a tabularx incl. closing \\bottomrule; wrapper in paper/supplement.tex.\n" + body.rstrip() + "\n\\bottomrule\n")
    return p


# ---- S4 (diag per category) -------------------------------------------------------------------------------------------------------
def tab_diag(res=RES):
    s = jload(res, "diag", "critic_diag_summary.json")
    fg = jload(res, "diag", "critic_diag_fig.json")
    order = need(fg, "categories")
    rows = []

    def one(name, blk, auroc):
        det, fp = need(blk, "detect_triggered"), need(blk, "fp_triggered_gold")
        rep = need(blk, "repair_round1_all")
        rows.append([name, "%d / %d" % (need(blk, "n_pert"), need(blk, "n_gold")), rate_ci(det["rate"], det["lo"], det["hi"]),
                     rate_ci(fp["rate"], fp["lo"], fp["hi"]), "%s %s" % (num(auroc["auroc"], 3), ci(auroc["lo"], auroc["hi"], 3, False)),
                     rate_ci(rep["rate"], rep["lo"], rep["hi"])])
    for c in order:
        blk = need(s, "per_category", c)
        one("\\texttt{%s}" % c.replace("_", "\\_\\allowbreak{}"), blk, need(blk, "auroc_paired_controls"))
    p = need(fg, "auroc_pooled")
    one("All categories (pooled)", need(s, "pooled", "all"), {"auroc": p["auroc"], "lo": p["lo"], "hi": p["hi"]})
    body = table_rows(rows[:-1]) + "\n\\midrule\n" + table_rows(rows[-1:])
    return write_tex("tab_s_diag", body)


# ---- VA-VD ------------------------------------------------------------------------------------------------------------------------
def tab_va_vd(res=RES):
    v2 = jload(res, "diag", "critic_v2_summary.json")
    rows = []
    base = need(v2, "variants", "V0", "point")
    for v in VARIANTS:
        p, c = need(v2, "variants", v, "point"), need(v2, "variants", v, "ci")
        if v == "V0":
            dfp = "--"
            rule = "reference"
        else:
            d = need(v2, "decision", v)
            dfp = est_ci(pct(d["delta_fpr"]), pct(d["delta_fpr_ci"]["lo"]), pct(d["delta_fpr_ci"]["hi"]))
            rule = "met" if d["improves_rule_18_7"] else "not met"
        rows.append([VARIANT_LABEL[v].replace("(", "(").replace("+", "+"),
                     rate_ci(p["detection"], c["detection"]["lo"], c["detection"]["hi"]), rate_ci(p["fpr"], c["fpr"]["lo"], c["fpr"]["hi"]), dfp,
                     rate_ci(p["harm"], c["harm"]["lo"], c["harm"]["hi"]), num(p["auroc"], 3),
                     num(need(v2, "calls", v, "calls_per_query"), 2), rule])
    del base
    return write_tex("tab_s_va_vd", table_rows(rows))


# ---- VE ---------------------------------------------------------------------------------------------------------------------------
ARM_NAME = {"none": "No appeal (VA as is)", "obey": "Obey the user", "A": "Arbiter A, neutral prompt", "A0": "A0, no user statement",
            "B": "B, other model family", "C": "C, old VD prompt (flags only)"}


def tab_ve(res=RES):
    ve = jload(res, "diag", "critic_ve_summary.json")
    meta = need(ve, "_meta")
    rows = []
    for qi, q in enumerate(need(meta, "q")):
        qk = "%.2f" % q
        if qi:
            rows.append(["\\midrule"])
        for arm in ("none", "obey", "A", "A0", "B", "C"):
            p, c = need(ve, "by_q", qk, "point", arm), need(ve, "by_q", qk, "ci", arm)

            def cell(m):
                lo, hi = c[m]["lo"], c[m]["hi"]
                if abs(hi - lo) < 1e-12:
                    return num(pct(p[m]))
                return rate_ci(p[m], lo, hi)
            cpq = need(ve, "by_q", qk, "calls_per_query")
            calls = num(cpq[arm], 2) if arm in cpq else "0"
            rows.append([("$q=%s$: " % qk if arm == "none" else "") + ARM_NAME[arm], cell("J"), cell("FPR"), cell("CCR"), cell("WCR"), cell("MER"), calls])
    body = "\n".join((" & ".join(r) + " \\\\") if len(r) > 1 else r[0] for r in rows)
    return write_tex("tab_s_ve", body)


def tab_ve_quality(res=RES):
    ve = jload(res, "diag", "critic_ve_summary.json")
    cells = [("firmness_F_claim_correct", "Hold a correct claim"), ("firmness_P_gold", "Hold a right pass"),
             ("responsiveness_F_claim_wrong_injected", "Drop a wrong claim, injected item"), ("responsiveness_F_gold_flag", "Drop a flag on a reference-query control"),
             ("responsiveness_P_injected", "Flag a passed wrong query")]
    rows = []
    for key, name in cells:
        n = need(ve, "arbitration_quality", "A", "point", key, "n")
        row = [name, str(n)]
        for arm in ("A", "A0", "B", "C"):
            pt = need(ve, "arbitration_quality", arm, "point", key)
            if pt["n"] == 0:
                row.append("n/a")
                continue
            cc = need(ve, "arbitration_quality", arm, "ci", key) if key in need(ve, "arbitration_quality", arm, "ci") else None
            row.append(rate_ci(pt["rate"], cc["lo"], cc["hi"]) if cc and (cc["hi"] - cc["lo"]) > 1e-12 else num(pct(pt["rate"])))
        rows.append(row)
    return write_tex("tab_s_ve_quality", table_rows(rows))


# ---- RESCORE ----------------------------------------------------------------------------------------------------------------------
def _model_names(t):
    """display spelling of the model names in labels that come from result files (round 5, A12): code font only for the identifiers themselves"""
    for o, n in (("qwen3.7-plus", "Qwen3.7-Plus"), ("qwen3.7-flash", "Qwen3.7-Flash"), ("qwen3.5-27b", "Qwen3.5-27B")):
        t = t.replace(o, n)
    return t


def _rescore_rows(contr, keys):
    rows = []
    for k in keys:
        c = need(contr, k)
        verdict = ("robust" if c.get("robust") else "not robust") + (", stable" if c.get("stable") else "")
        rows.append([_model_names(tex_escape(need(c, "desc"))), str(need(c, "n")), est_ci(c["d0"], *c["ci0"]), est_ci(c["d1"], *c["ci1"]), est_ci(c["shift"], *c["ci_shift"]), verdict])
    return rows


def tab_rescore(res=RES):
    r = jload(res, "robust", "rescore.json")
    out = {}
    for set_name, nm in (("S_same", "tab_s_rescore_same"), ("S_all", "tab_s_rescore_all")):
        contr = need(r, "variants", "G1", "sets", set_name, "analysis", "contrasts")
        keys = [k for k in contr if k.startswith(("H1.", "H2.", "H2b.", "H3."))]
        rows = _rescore_rows(contr, keys)
        ana = need(r, "variants", "G1", "sets", set_name, "analysis")
        rt = need(ana, "H4.route")
        for key, label, d in (("dauroc", "AUROC, route\\_ca $-$ V3", 3), ("daurc", "AURC, route\\_ca $-$ V3 (lower is better)", 3)):
            c = need(rt, key)
            rows.append([label, str(need(rt, "n")), est_ci(c["d0"], *c["ci0"], d=d), est_ci(c["d1"], *c["ci1"], d=d), est_ci(c["shift"], *c["ci_shift"], d=d),
                         ("robust" if c["robust"] else "not robust") + (", stable" if c["stable"] else "")])
        net = need(ana, "H5.revision", "net_exec")
        rows.append(["Net effect of executable-root revisions, fixed loop, Plus (pp of set)", str(need(ana, "H5.revision", "n")), est_ci(net["d0"], *net["ci0"]),
                     est_ci(net["d1"], *net["ci1"]), est_ci(net["shift"], *net["ci_shift"]), ("robust" if net["robust"] else "not robust") + (", stable" if net["stable"] else "")])
        out[nm] = write_tex(nm, table_rows(rows))
    return out


def tab_gold_variants(res=RES):
    r = jload(res, "robust", "rescore.json")
    rows = []
    cols = [("H1.flash", "Flash"), ("H1.q27b", "27B"), ("H1.pooled3", "pooled"), ("H2.flash", "Flash"), ("H2.q27b", "27B"), ("H2.pooled_weak", "pooled")]
    for g in ("G1", "G2", "G3", "G4"):
        s = need(r, "variants", g, "sets", "S_same")
        contr = need(s, "analysis", "contrasts")
        row = [g, str(need(r, "variants", g, "sets", "S_same", "n")), str(need(r, "variants", g, "sets", "S_all", "n"))]
        for k, _ in cols:
            c = need(contr, k)
            row.append("%s%s%s" % (num(c["d0"], 1, True), "$\\rightarrow$", num(c["d1"], 1, True)) + ("$^{\\ast}$" if not c["robust"] else ""))
        rows.append(row)
    body = table_rows(rows)
    # L6 corroboration (changed gold result among E10 L6-labelled items against the other labels)
    e = []
    for g in ("G1", "G2", "G3", "G4"):
        for sn in ("S_all", "S_same"):
            blk = need(r, "e10_l6", g, sn, "eff")
            l6, nl6, df = need(blk, "L6"), need(blk, "non_L6"), need(blk, "diff_L6_minus_nonL6")
            e.append([g, sn.replace("_", "\\_"), "%d / %d" % (l6["k"], l6["n"]), "%d / %d" % (nl6["k"], nl6["n"]), est_ci(df["diff"], *df["newcombe"], d=2)])
    write_tex("tab_s_l6", table_rows(e))
    return write_tex("tab_s_gold_variants", body)


# ---- EXT1 -------------------------------------------------------------------------------------------------------------------------
def tab_ext1(res=RES, raw=None):
    arms = list(csv.DictReader(open(os.path.join(res, "ext1", "ext1_arms.csv"), encoding="utf-8")))
    bk = {"plus": "Qwen3.7-Plus", "flash": "Qwen3.7-Flash", "27b": "Qwen3.5-27B"}
    an = {"zs": "single call", "fixed": "fixed loop", "gated": "gated loop", "rootonly": "root only"}
    rows = []
    for a in arms:
        rows.append([bk[a["backbone"]], an[a["arm"]], est_ci(pct(float(a["ex_ts"])), pct(float(a["ex_ts_ci_lo"])), pct(float(a["ex_ts_ci_hi"])), sign=False),
                     num(pct(float(a["ex_plain"]))), num(float(a["calls_per_q"]), 2), num(1000 * float(a["cost_cny_per_q"]), 2)])
    write_tex("tab_s_ext1_arms", table_rows(rows))
    con = list(csv.DictReader(open(os.path.join(res, "ext1", "ext1_contrasts.csv"), encoding="utf-8")))
    summ = jload(res, "ext1", "ext1_summary.json")
    crow = []
    for c in con:
        loo = "%s to %s" % (num(float(c["loo_min"]), 2, True), num(float(c["loo_max"]), 2, True)) if c["loo_min"] else "--"
        item = ci(float(c["item_ci_lo"]), float(c["item_ci_hi"]), 2) if c["item_ci_lo"] else "--"
        crow.append([tex_escape(c["contrast"]), c["n"], est_ci(float(c["delta_pp"]), float(c["ci_lo"]), float(c["ci_hi"]), d=2), item, loo])
    write_tex("tab_s_ext1_contrasts", table_rows(crow))
    srow = []

    def flat(prefix):
        lo, hi = "%s.delta_ex.ci_lo" % prefix, "%s.delta_ex.ci_hi" % prefix
        if lo in summ and hi in summ:
            return est_ci(need(summ, prefix + ".delta_ex"), summ[lo], summ[hi], d=2)
        return num(need(summ, prefix + ".delta_ex"), 2, True) + " [--]"
    for xb in ("XB1", "XB2", "XB3"):
        full, sens = "EXT1.%s" % xb, "EXT1.sens_no_world_1.%s" % xb
        srow.append([xb, str(need(summ, full + ".n")), flat(full), str(need(summ, sens + ".n")), flat(sens)])
    write_tex("tab_s_ext1_world1", table_rows(srow))
    # per-database net difference gated minus single call (items, from the stored predictions; needs results/raw-ignored/EXT1)
    raw = raw or os.path.join(res, "raw-ignored", "EXT1")
    if os.path.isdir(raw):
        per = {}
        for b in ("plus", "flash", "27b"):
            def load(arm):
                d = {}
                with open(os.path.join(raw, "ext1-%s-%s" % (b, arm), "preds.jsonl"), encoding="utf-8") as f:
                    for ln in f:
                        r = json.loads(ln)
                        d[r["qid"]] = (r["db_id"], r["ex_ts"])
                return d
            g_, z_ = load("gated"), load("zs")
            assert set(g_) == set(z_)
            for q, (db, e1) in g_.items():
                e0 = z_[q][1]
                assert e1 is not None and e0 is not None, "ex_ts null for %s (not a measurement)" % q
                cell = per.setdefault(db, {"n": 0}).setdefault(b, 0)
                per[db][b] = cell + int(e1) - int(e0)
                per[db]["n"] = per[db]["n"] if b != "plus" else per[db]["n"] + 1
        prow = []
        for db in sorted(per, key=lambda d: (-per[d]["n"], d)):
            v = per[db]
            prow.append(["\\texttt{%s}" % db.replace("_", "\\_\\allowbreak{}"), str(v["n"])] + ["%s (%s)" % (num(100.0 * v[b] / v["n"], 1, True), num(v[b], 0, True)) for b in ("plus", "flash", "27b")])
        write_tex("tab_s_ext1_perdb", table_rows(prow))
    return True


# ---- E10 --------------------------------------------------------------------------------------------------------------------------
LABELS = [("L1", "schema linking", "wrong table or column; the intended metric and filters are otherwise as in the reference query"),
          ("L2", "join path", "right tables and columns, different join condition, type, chain or set of joined tables"),
          ("L3", "filter or value", "a predicate, literal, operator, boundary, format or NULL handling is wrong, missing or extra"),
          ("L4", "aggregation, grouping, ordering", "aggregate, DISTINCT, GROUP BY unit, HAVING, ORDER BY, LIMIT or tie handling differs without changing the defined quantity"),
          ("L5", "wrong criterion (exploratory)", "linking and joins right, same output shape, but a different definition of the quantity than the evidence defines"),
          ("L6", "reference query suspect", "the reference query contradicts the question or the evidence text, or its result is degenerate"),
          ("L7", "ambiguous question", "two readings that a careful reader accepts; the prediction implements one and the reference the other"),
          ("L8", "other", "none of the above fits (answer formatting, rounding, output columns)")]


def tab_e10(res=RES):
    e = os.path.join(res, "e10")
    ag = jload(e, "agreement.json")
    conf = list(csv.reader(open(os.path.join(e, "confusion.csv"), encoding="utf-8")))
    hdr = conf[0][1:-1]
    diag = {}
    for row in conf[1:]:
        if row[0] in hdr:
            diag[row[0]] = int(row[1 + hdr.index(row[0])])

    def prev(name):
        d = {}
        for r in csv.DictReader(open(os.path.join(e, name), encoding="utf-8")):
            if r["source"] == "final":
                d[r["label"]] = (float(r["weighted_prevalence"]), float(r["ci95_lo"]), float(r["ci95_hi"]))
        return d
    pz, pf = prev("prevalence_zs.csv"), prev("prevalence_full.csv")
    c1, c2 = need(ag, "counts_A1"), need(ag, "counts_A2")
    rows = []
    for lab, name, defn in LABELS:
        rows.append(["%s %s" % (lab, tex_escape(name)), tex_escape(defn), "%d / %d / %d" % (c1[lab], c2[lab], diag[lab]),
                     rate_ci(*pf[lab]), rate_ci(*pz[lab])])
    write_tex("tab_s_e10_labels", table_rows(rows))
    pl = need(ag, "primary_label")
    l5 = need(ag, "L5_vs_rest")
    arow = [["Items labelled by both runs", str(need(ag, "n_items")), "--"],
            ["Percent agreement (8 labels)", num(pct(pl["pct_agree"])), "--"],
            ["Cohen kappa (8 labels)", num(pl["kappa"], 3), ci(pl["kappa_ci95"][0], pl["kappa_ci95"][1], 3, False)],
            ["Gwet AC1 (8 labels)", num(pl["ac1"], 3), ci(pl["ac1_ci95"][0], pl["ac1_ci95"][1], 3, False)],
            ["Cohen kappa, L5 against the rest", num(l5["kappa"], 3), ci(l5["kappa_ci95"][0], l5["kappa_ci95"][1], 3, False)],
            ["Items disagreeing between the two runs (adjudicated)", str(need(ag, "n_disagree")), "--"]]
    write_tex("tab_s_e10_agree", table_rows(arow))
    return True


# ---- amendment log (parsed from docs/SAP.md) -------------------------------------------------------------------------------------
def tab_log(sap=os.path.join(ROOT, "docs", "SAP.md")):
    """Amendment log rows of docs/SAP.md (one row per amendment, first occurrence wins), sorted by amendment number;
    notes that are not amendments (addenda, incident, post hoc notes of Sections 24 to 27) follow in the order of the SAP."""
    seen, rows = set(), []
    pat = re.compile(r"^\| (20\d\d-\d\d-\d\d) \| ")
    for ln in open(sap, encoding="utf-8"):
        if not pat.match(ln):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        if len(cells) < 5:                                   # reading copy of the SAP: the Author column is not part of it
            cells.insert(2, "")
        m = re.match(r"Amendment (\d+)([a-z]?)", cells[1])
        key = (int(m.group(1)), m.group(2)) if m else (999, re.sub(r"\W+", "", cells[1])[:60])
        if key in seen:
            continue
        seen.add(key)
        if not m:
            key = (999, "%04d" % len(rows))
        entry = re.sub(r"\s*\(\u00a7\d+\)", "", cells[1])
        entry = re.sub(r"Plat gold\b", "Plat reference queries", entry)               # prose says 'reference query', not 'gold'
        entry = re.sub(r"\bgold SQL results\b", "reference-query results", entry)
        before = cells[3].replace("no `code/prompts/**`, no `exp/models.yaml` touched", "no prompt or model-configuration file touched")  # no repository paths in the table
        entry = entry.replace("scope `robustness-posthoc`", "scope: robustness check, post hoc")               # V4 L-6: the rule label in prose, the code token only in code font
        entry = entry.replace("`exp/dev_lock.json`", "the lock file").replace("exp/dev_lock.json", "the lock file")      # no repository paths in the table
        effect = re.sub(r"`?\\(?:ts|x)?res`?(?:, `?\\(?:ts|x)?res`?)*", "the result macros", cells[4])      # round 5 (R2 N8): no LaTeX macro names in the supplement text
        # W3: no working-file names, review-round labels or process words in the supplement text (the SAP row keeps its date, scope and numbers)
        clean = lambda s: re.sub(r"\bsmoke\b", "pilot", re.sub(r"`?results_macros\.tex`?", "the result macros", s.replace("round-4 statistics", "further statistics")))
        entry, before, effect = clean(entry), clean(before), clean(effect)
        rows.append((key, [cells[0], tex_escape(entry), tex_escape(before), tex_escape(effect)]))
    rows.sort(key=lambda kv: (kv[0][0], str(kv[0][1])))
    return write_tex("tab_s_log", table_rows([r for _, r in rows]))


# ----------------------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["data", "tables", "figs", "all"])
    ap.add_argument("--results", default=RES)
    ap.add_argument("--figs", default=FIGS)
    ap.add_argument("--supp", default=SUPP)
    a = ap.parse_args(argv)
    globals()["SUPP"] = a.supp
    res = a.results
    if a.what in ("data", "figs", "all"):
        cg, cd, fr = corrected_gold_data(res), critic_diag_data(res), frontier_data(res)
        write_json(os.path.join(a.figs, "data", "fig_corrected_gold.json"), cg)
        write_json(os.path.join(a.figs, "data", "fig_critic_diag.json"), cd)
        write_json(os.path.join(a.figs, "data", "fig_frontier.json"), fr)
        print("data ->", os.path.join(a.figs, "data"))
    if a.what in ("figs", "all"):
        for p in draw_corrected_gold(cg, a.figs) + draw_critic_diag(cd, a.figs) + draw_frontier(fr, a.figs):
            print("fig ->", p)
    if a.what in ("tables", "all"):
        for fn in (tab_diag, tab_va_vd, tab_ve, tab_ve_quality, tab_rescore, tab_gold_variants, tab_ext1, tab_e10):
            fn(res)
        tab_log()
        print("tables ->", a.supp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
