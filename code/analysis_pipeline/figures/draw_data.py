"""F2-F7: data figures.  Each function takes its block of the figure-data dict D (figdata.py) and returns a matplotlib Figure, or a string (reason) if the block
is absent.  All text >= 8 pt; sizes are the final print sizes (17 cm full width unless noted)."""
import math

from . import style as st

CM = st.CM
STAGE_LAB = {"m1": "M1 intent model", "gen": "generation", "crit": "Critic call", "rev": "revision"}
STAGE_COL = {"m1": "#dddddd", "gen": st.SKY, "crit": st.VERM, "rev": st.ORANGE}
STAGE_HATCH = {"m1": "", "gen": "", "crit": "////", "rev": "...."}


def _err(lo, m, hi):
    return [[max(m - lo, 0.0)], [max(hi - m, 0.0)]]


# ====================================================================================================================== F2
def _frontier(pts):
    """Non-dominated points (lower cost, higher EX), by point estimates.  pts: list of (cost, ex, key) -> list sorted by cost."""
    out, best = [], -1e9
    for c, y, k in sorted(pts):
        if y > best:
            out.append((c, y, k))
            best = y
    return out


def fig_cost(B, mode, synthetic=False):
    if not B or len(B["cells"]) < 2:
        return "cost block absent (needs EX, CI and cost for >= 2 cells)"
    P = st.plt()
    from matplotlib.lines import Line2D
    fig, (a, b) = P.subplots(1, 2, figsize=(st.W2, 8.3 * CM), gridspec_kw={"width_ratios": [1.28, 1]}, constrained_layout=True)
    models = [m for m in ("qwen37flash", "qwen3527b", "qwen37plus") if any(c["model"] == m for c in B["cells"])]
    have_cost = all(c.get("cost") is not None for c in B["cells"])
    if not have_cost:
        a.text(0.5, 0.5, "cost per question missing in results", transform=a.transAxes, ha="center")
    pts, where = [], {}
    for m in models:
        cs = {c["system"]: c for c in B["cells"] if c["model"] == m}
        if have_cost:
            for s_ in ("zs", "full_fixed", "full_gated"):
                if s_ not in cs:
                    continue
                c = cs[s_]
                x, y = 1000.0 * c["cost"], c["ex"]
                a.errorbar([x], [y], yerr=_err(c["lo"], c["ex"], c["hi"]), fmt="none", ecolor=st.SYS_COL[s_], elinewidth=0.7, alpha=0.40, capsize=0, zorder=2)
                pts.append((x, y, (m, s_)))
                where[(m, s_)] = (x, y)
            if ("full_fixed" in cs) and ("full_gated" in cs):                                  # what gating does on this backbone: same questions, fewer calls
                (x0, y0), (x1, y1) = where[(m, "full_fixed")], where[(m, "full_gated")]
                a.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=st.GREY, shrinkA=4, shrinkB=4, mutation_scale=7), zorder=2)
    fr = _frontier(pts) if have_cost else []
    if len(fr) >= 2:
        a.plot([p[0] for p in fr], [p[1] for p in fr], color=st.INK, lw=1.0, ls=(0, (4, 2)), zorder=3)
    for m in models:
        for s_ in ("zs", "full_fixed", "full_gated"):
            if (m, s_) not in where:
                continue
            x, y = where[(m, s_)]
            on = any(p[2] == (m, s_) for p in fr)
            a.plot([x], [y], marker=st.MODEL_MARK[m], ms=7.0, mfc=st.SYS_COL[s_], mec=st.INK if on else "white", mew=1.1 if on else 0.5, ls="none", zorder=4)
    # headline contrast, point estimates: cheapest backbone with the gated loop against the strongest backbone zero-shot
    if ("qwen37flash", "full_gated") in where and ("qwen37plus", "zs") in where:
        (x1, y1), (x2, y2) = where[("qwen37flash", "full_gated")], where[("qwen37plus", "zs")]
        x1d = B.get("x1")
        txt = ("Flash gated vs Plus zero-shot (X1):\n%.0f%% of the cost, %+.1f pp [%+.1f, %+.1f]" % (100.0 * x1 / x2, x1d["m"], x1d["lo"], x1d["hi"])) if x1d and x1d.get("lo") is not None \
            else ("Flash gated vs Plus zero-shot:\n%.0f%% of the cost, %+.1f pp" % (100.0 * x1 / x2, y1 - y2))
        a.annotate(txt, xy=(x1, y1), xytext=(0.17, 0.22), textcoords="axes fraction", ha="left", va="center", fontsize=8,
                   arrowprops=dict(arrowstyle="-", lw=0.6, color=st.INK, shrinkB=5), bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.92), zorder=5)
    a.set_xscale("log")
    xt = [0.2, 0.5, 1, 2, 5, 10]
    a.set_xticks(xt)
    a.set_xticklabels([("%g" % t) for t in xt])
    a.minorticks_off()
    a.set_xlim(0.14, 14)
    a.set_xlabel("Chat-model cost (CNY per 1000 questions, list price, log scale)")
    a.set_ylabel("Execution accuracy (%)")
    lo = [c["lo"] for c in B["cells"]]
    hi = [c["hi"] for c in B["cells"]]
    a.set_ylim(min(lo) - 2, max(hi) + 2)
    st.grid_y(a)
    st.panel_title(a, "a", "Accuracy against cost, %s" % B["split"])
    # ---- paired differences
    ds = B.get("deltas") or []
    rows = []
    for m in models:
        for k in ("gated_vs_zs", "gated_vs_fixed"):
            d = [x for x in ds if x["model"] == m and x["contrast"] == k]
            if d:
                rows.append((m, k, d[0]))
    ycur, yt, yl = 0.0, [], []
    col = {"gated_vs_zs": st.GREEN, "gated_vs_fixed": st.PURPLE}
    for m in models:
        mine = [r for r in rows if r[0] == m]
        for j, (mm, k, d) in enumerate(mine):
            y = -(ycur + 0.45 * j)
            b.errorbar([d["m"]], [y], xerr=_err(d["lo"], d["m"], d["hi"]), fmt="none", ecolor=col[k], elinewidth=1.1, capsize=2, capthick=0.8, zorder=2)
            b.plot([d["m"]], [y], marker=st.MODEL_MARK[m], ms=6.5, mfc=col[k], mec="white", mew=0.5, ls="none", zorder=4)
            b.text(max(d["hi"], d["m"]) + 0.5, y, ("%+.1f" % d["m"]).replace("-", "\u2212"), va="center", ha="left", fontsize=8)
        if mine:
            yt.append(-(ycur + 0.45 * (len(mine) - 1) / 2.0))
            yl.append(st.MODEL_LAB[m].replace("Qwen", ""))
            ycur += 1.4
    b.axvline(0, color=st.INK, lw=0.7, zorder=1)
    b.set_yticks(yt)
    b.set_yticklabels(yl)
    b.set_ylim(-ycur + 0.55, 1.5)
    lo_ = [r[2]["lo"] for r in rows]
    hi_ = [r[2]["hi"] for r in rows]
    if rows:
        b.set_xlim(min(lo_ + [0]) - 1.0, max(hi_ + [0]) + 2.6)
    b.set_xlabel("Difference in EX (percentage points)")
    st.grid_x(b)
    st.direction_note(b, "gated worse", "gated better", y=-0.21)
    st.panel_title(b, "b", "Paired differences, 95% CI, original reference queries")
    # legend above panel (a) / (b): colour = system, marker = backbone, arrow = fixed -> gated, dashed = frontier; panel (b) keys sit in the same row
    hs = [Line2D([], [], marker="o", ls="none", ms=6.5, mfc=st.SYS_COL[s_], mec="white", label=st.SYS_LAB[s_]) for s_ in ("zs", "full_fixed", "full_gated")]
    hm = [Line2D([], [], marker=st.MODEL_MARK[m], ls="none", ms=6.5, mfc="#bbbbbb", mec="#555555", mew=0.5, label=st.MODEL_LAB[m]) for m in models]
    hf = [Line2D([], [], color=st.INK, lw=1.0, ls=(0, (4, 2)), marker="o", ms=5, mfc="#bbbbbb", mec=st.INK, label="Pareto frontier"),
          Line2D([], [], color=st.GREY, lw=0.9, marker=">", ms=4, label="fixed to gated loop")]
    fig.legend(handles=hs + hm + hf, loc="outside upper right", ncol=4, fontsize=8, columnspacing=1.0, handletextpad=0.3, labelspacing=0.25)
    hd = [Line2D([], [], marker="o", ls="-", lw=1.1, ms=6, color=col[k], mec="white", label=l) for k, l in (("gated_vs_zs", "gated minus zero-shot"), ("gated_vs_fixed", "gated minus fixed loop")) if any(r[1] == k for r in rows)]
    b.legend(handles=hd, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=8, labelspacing=0.25, borderaxespad=0.1, frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)
    st.ref_tag(fig, mode, synthetic)
    return fig


# ====================================================================================================================== F3
def fig_repair(B, T, mode, synthetic=False):
    """Figure 3 (main text), round 3: (a) calls per question by stage, (b) EX of the root and of the final answer.  The former panel (b), the tuning-set quoting bug, is now its own
    supplement figure (fig_quoting); T is therefore not read here (kept in the signature so that the figure data and the callers stay unchanged)."""
    if not B or not (B.get("calls") or B.get("rootfinal")):
        return "repair block absent"
    P = st.plt()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig = P.figure(figsize=(st.W2, 6.9 * CM), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0])
    # ---------------------------------------------------------------- (a) calls by stage
    a = fig.add_subplot(gs[0, 0])
    groups = []
    for r in B["calls"]:
        if r["group"] not in groups:
            groups.append(r["group"])
    xs, tick, tl, gcent = 0.0, [], [], []
    for g in groups:
        for j, s in enumerate(("full_fixed", "full_gated")):
            r = [x for x in B["calls"] if x["group"] == g and x["system"] == s]
            if not r:
                continue
            r = r[0]
            bottom = 0.0
            for k in ("m1", "gen", "crit", "rev"):
                v = r.get(k) or 0.0
                a.bar(xs + 0.42 * j, v, 0.38, bottom=bottom, color=STAGE_COL[k], edgecolor=st.INK, lw=0.4, hatch=STAGE_HATCH[k], zorder=2)
                bottom += v
            a.text(xs + 0.42 * j, bottom + 0.08, "%.2f" % bottom, ha="center", va="bottom", fontsize=8, rotation=90)
            tick.append(xs + 0.42 * j)
            tl.append("fixed" if s == "full_fixed" else "gated")
        gcent.append((xs + 0.21, g))
        xs += 1.25
    a.set_xticks(tick)
    a.set_xticklabels(tl)
    for x, g in gcent:
        a.annotate(g.replace(" ", "\n", 1), (x, 0), xycoords=("data", "axes fraction"), xytext=(0, -17), textcoords="offset points", ha="center", va="top", fontsize=8)
    a.set_ylabel("Answer-path LLM calls per question")
    a.set_ylim(0, 6.0)
    a.set_yticks([0, 1, 2, 3, 4])
    st.grid_y(a)
    a.legend(handles=[Patch(fc=STAGE_COL[k], ec=st.INK, lw=0.4, hatch=STAGE_HATCH[k], label=STAGE_LAB[k]) for k in ("m1", "gen", "crit", "rev")], loc="upper center", fontsize=8,
             labelspacing=0.2, ncol=2, handlelength=1.2, columnspacing=1.0)
    st.panel_title(a, "a", "Calls per question by stage" + (", development set" if mode == "final" else ""))
    # ---------------------------------------------------------------- (b) root -> final EX
    b = fig.add_subplot(gs[0, 1])
    rf = B["rootfinal"]
    rowsb = []
    for g in ("HR-1101", "HT-300", "dev"):
        for s in ("full_fixed_legacy", "full_fixed", "full_gated"):
            r = [x for x in rf if x["group"] == g and x["system"] == s]
            if r:
                rowsb.append(r[0])
    groups_b = sorted(set(r["group"] for r in rowsb))
    single = groups_b == ["dev"]
    y = 0.0
    yt, yl = [], []
    last = None
    xmin = min(r["root"] for r in rowsb) - 0.4
    xmax = max(r["final"] for r in rowsb) + 1.7                           # room for the gain label right of the filled marker
    for r in rowsb:
        if last is not None and r["group"] != last:
            y -= 0.6
        last = r["group"]
        c_ = st.SYS_COL[r["system"]]
        b.plot([r["root"], r["final"]], [y, y], color=c_, lw=2.2, zorder=2, solid_capstyle="butt")
        b.plot([r["root"]], [y], marker="o", ms=6.5, mfc="white", mec=c_, mew=1.4, zorder=3, ls="none")
        b.plot([r["final"]], [y], marker="o", ms=6.5, mfc=c_, mec=c_, zorder=3, ls="none")
        b.text(r["final"] + 0.22, y, st.fmt_ci(r["delta"], r.get("dlo"), r.get("dhi"), 1) + ("" if single else "  (n = %d)" % r["n"]), va="center", ha="left", fontsize=8)
        yt.append(y)
        nm = {"full_fixed_legacy": "fixed loop, legacy renderer", "full_fixed": "fixed loop", "full_gated": "gated loop"}[r["system"]]
        yl.append(nm if single else nm + ",  " + {"dev": "development set"}.get(r["group"], r["group"].replace("-", " ")))
        y -= 1.0
    b.set_yticks(yt)
    b.set_yticklabels(yl)
    b.set_ylim(y + 0.45, 1.25)
    b.set_xlim(xmin, xmax)
    b.set_xlabel("Execution accuracy of Qwen3.7-Plus (%)")
    st.grid_x(b)
    b.legend(handles=[Line2D([], [], marker="o", ls="none", mfc="white", mec=st.INK, mew=1.2, ms=6.5, label="root"),
                      Line2D([], [], marker="o", ls="none", mfc=st.INK, mec=st.INK, ms=6.5, label="final answer")], loc="upper left", fontsize=8, labelspacing=0.2, ncol=2,
             handletextpad=0.2, columnspacing=0.8, borderaxespad=0.1)
    ttl = "Root against final answer" + (", n = %d" % rowsb[0]["n"] if single and rowsb else "")
    st.panel_title(b, "b", ttl)
    st.ref_tag(fig, mode, synthetic)
    return fig


def fig_quoting(T, mode, synthetic=False):
    """Supplement figure (round 3): the tuning-set quoting defect, moved out of Figure 3.  Root execution-failure rate with the legacy schema renderer and with quoted SQLite
    keywords, fixed loop; tuning set only, in every mode."""
    if not T or not T.get("errrate"):
        return "quoting block absent"
    P = st.plt()
    from matplotlib.patches import Patch
    fig = P.figure(figsize=(QUOTING_W_CM * CM, 6.6 * CM), constrained_layout=True)
    c = fig.add_subplot(1, 1, 1)
    er = T["errrate"]
    grs = []
    for r in er:
        if r["group"] not in grs:
            grs.append(r["group"])
    for i, g in enumerate(grs):
        for j, arm in enumerate(("legacy", "quoted")):
            r = [x for x in er if x["group"] == g and x["arm"] == arm][0]
            col = st.ORANGE if arm == "legacy" else st.BLUE
            x = i + 0.21 * (2 * j - 1)
            c.bar(x, r["rate"], 0.36, color=col, edgecolor=st.INK, lw=0.4, hatch="" if arm == "quoted" else "\\\\\\\\", zorder=2)
            c.errorbar([x], [r["rate"]], yerr=[[r["rate"] - r["lo"]], [r["hi"] - r["rate"]]], fmt="none", ecolor=st.INK, elinewidth=0.7, capsize=1.5, zorder=3)
            c.text(x + 0.03, r["hi"] + 1.0, "%.1f" % r["rate"], ha="center", va="bottom", fontsize=8, rotation=90)
    c.set_xticks(range(len(grs)))
    c.set_xticklabels([g.replace(" (", "\n(").replace("HR-1101", "HR-1101\n(8 databases)").replace("HT-300", "HT-300\n(4 databases)") for g in grs])
    c.set_ylim(0, 68)
    c.set_ylabel("Root fails to execute (%)")
    st.grid_y(c)
    c.legend(handles=[Patch(fc=st.ORANGE, ec=st.INK, lw=0.4, hatch="\\\\\\\\", label="legacy schema renderer"), Patch(fc=st.BLUE, ec=st.INK, lw=0.4, label="SQLite keywords quoted")],
             loc="upper right", fontsize=8, labelspacing=0.2, handlelength=1.2)
    st.panel_title(c, "", "Identifier quoting, fixed loop")
    st.tuning_stamp(c, "tuning set", x=1.0, y=1.0)
    st.ref_tag(fig, mode, synthetic)
    return fig


QUOTING_W_CM = 11.0


# ====================================================================================================================== F4
def fig_rc(B, mode, synthetic=False):
    if not B or not B.get("panels"):
        return "risk-coverage block absent"
    P = st.plt()
    pans = B["panels"]
    n = len(pans)
    fig = P.figure(figsize=(st.W2, 7.4 * CM), constrained_layout=True)
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1.0] * n + [0.8])
    col = {"v3": st.DGREY, "route_ca": st.BLUE}
    lab = {"v3": "V3 Critic score", "route_ca": r"routing score $\rho$"}
    ls = {"v3": (0, (4, 1.5)), "route_ca": "-"}
    tops = [max(c.get("hi") or c["risk"]) for p in pans for c in p["curves"].values() if c.get("risk")]
    ymax = max(tops) * 100 * 1.38
    first = None
    for i, p in enumerate(pans):
        ax = fig.add_subplot(gs[0, i], sharey=first)
        first = first or ax
        aurc = p.get("aurc") or {}
        for k in ("v3", "route_ca"):
            c = p["curves"].get(k)
            if not c:
                continue
            if c.get("lo") and c.get("bcov"):
                ax.fill_between([100 * v for v in c["bcov"]], [100 * v for v in c["lo"]], [100 * v for v in c["hi"]], color=col[k], alpha=0.14, lw=0, zorder=1)
            ax.plot([100 * v for v in c["cov"]], [100 * v for v in c["risk"]], color=col[k], ls=ls[k], lw=1.5, zorder=3, label=lab[k])
        c0 = p["curves"].get("v3") or p["curves"].get("route_ca")
        if c0:                                                     # oracle ranking: all correct answers first -> risk 0 up to coverage = EX, then (c - EX) / c
            full = 100 * c0["risk"][-1]
            acc = 1.0 - full / 100.0
            xs = [20.0, 100 * acc, 100.0]
            ax.plot(xs, [0.0, 0.0, full], color=st.GREY, lw=0.8, ls=(0, (1, 1.5)), zorder=2, label="oracle ranking")
            ax.axhline(full, color=st.LGREY, lw=0.7, zorder=0)
        ax.axvline(60, color=st.GREY, lw=0.6, ls=":", zorder=1)
        a60 = p.get("acc60") or {}
        note = []
        if aurc.get("v3") is not None and aurc.get("route_ca") is not None:
            note.append("AURC %.3f (V3), %.3f (routing)" % (aurc["v3"], aurc["route_ca"]))
        if "v3" in a60 and "route_ca" in a60:
            note.append("EX at 60%% coverage:\n%.1f%% (V3), %.1f%% (routing)" % (100 * a60["v3"], 100 * a60["route_ca"]))
        if note:
            ax.text(21, ymax * 0.985, "\n".join(note), ha="left", va="top", fontsize=8, bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.85), zorder=6)
        ax.set_xlim(20, 100)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("Coverage (% answered)")
        if i == 0:
            ax.set_ylabel("Risk (% wrong among answered)")
        else:
            ax.tick_params(labelleft=False)
        st.grid_y(ax)
        st.panel_title(ax, "abcd"[i], p["name"] + (", n = %d" % p["n"] if p.get("n") else ""))
        if i == 0:
            hh, ll = ax.get_legend_handles_labels()
    fig.legend(hh, ll, loc="outside upper right", ncol=3, fontsize=8, columnspacing=1.2, handlelength=1.8, handletextpad=0.4)
    ax = fig.add_subplot(gs[0, n])
    yy, yt, yl = 0.0, [], []
    xs_all = []
    for p in pans:
        for j, k in enumerate(("v3", "route_ca")):
            a = (p.get("auroc") or {}).get(k)
            if not a:
                continue
            y = -(yy + 0.45 * j)
            ax.errorbar([a[0]], [y], xerr=[[a[0] - a[1]], [a[2] - a[0]]], fmt="none", ecolor=col[k], elinewidth=1.1, capsize=2, zorder=2)
            ax.plot([a[0]], [y], marker="o", ms=6, mfc=col[k], mec="white", mew=0.5, ls="none", zorder=3)
            xs_all += [a[1], a[2]]
        d = p.get("dauroc")
        if d:
            ax.text(0.5, -(yy + 0.45 + 0.78), "difference\n%s\n%s" % ("%+.3f" % d[0], "[%+.3f, %+.3f]" % (d[1], d[2])), ha="center", va="center", fontsize=8, transform=ax.get_yaxis_transform())
        yt.append(-(yy + 0.225))
        yl.append(p["name"].split(",")[0])
        yy += 2.0
    ax.set_yticks(yt)
    ax.set_yticklabels(yl)
    ax.set_ylim(-yy + 0.25, 0.45)
    if xs_all:
        ax.set_xlim(min(xs_all) - 0.03, max(xs_all) + 0.03)
    ax.set_xlabel("AUROC")
    st.grid_x(ax)
    st.panel_title(ax, "abcd"[n], "AUROC, 95% CI")
    st.ref_tag(fig, mode, synthetic)
    return fig


# ====================================================================================================================== F5
def fig_pollution(B, mode, synthetic=False):
    if not B or not B.get("policies"):
        return "pollution block absent"
    P = st.plt()
    from matplotlib.lines import Line2D
    ev = B.get("evict")
    fig = P.figure(figsize=(st.W2, 6.6 * CM), constrained_layout=True)
    gs = fig.add_gridspec(1, 2 if ev else 1, width_ratios=[1.8, 1.0] if ev else [1])
    a = fig.add_subplot(gs[0, 0])
    pols = B["policies"]
    for i, p in enumerate(pols):
        col = st.POLICY_COL[p["policy"]]
        a.errorbar([i], [p["ex"]], yerr=_err(p["lo"], p["ex"], p["hi"]), fmt="none", ecolor=col, elinewidth=1.1, capsize=3, capthick=0.9, zorder=2)
        a.plot([i], [p["ex"]], marker={"append": "s", "memrl": "^", "ws_full": "o"}.get(p["policy"], "o"), ms=7.5, mfc=col, mec="white", mew=0.6, ls="none", zorder=4)
        a.text(i + 0.12, p["ex"], "%.1f" % p["ex"], ha="left", va="center", fontsize=8, weight="bold", color=st.INK,
               bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.9), zorder=5)
        if p.get("ex_p0") is not None:
            a.plot([i - 0.12], [p["ex_p0"]], marker="D", ms=6, mfc="white", mec=st.INK, mew=1.0, ls="none", zorder=4)
    a.set_xticks(range(len(pols)))
    a.set_xticklabels([st.POLICY_LAB[p["policy"]] for p in pols])
    a.set_xlim(-0.6, len(pols) - 0.4)
    lo = min(p["lo"] for p in pols)
    hi = max(p["hi"] for p in pols)
    a.set_ylim(max(0, math.floor(lo / 5.0) * 5 - 2), math.ceil(hi / 5.0) * 5 + 2)
    a.set_ylabel("Probe EX after the stream (%)")
    st.grid_y(a)
    if any(p.get("ex_p0") is not None for p in pols):
        a.legend(handles=[Line2D([], [], marker="o", ls="none", mfc="#999999", mec="white", ms=7, label="40% wrong seeds (95% CI)"),
                          Line2D([], [], marker="D", ls="none", mfc="white", mec=st.INK, ms=6, label="no pollution (0%)")], loc="lower right", fontsize=8, labelspacing=0.2)
    n = pols[0].get("n")
    st.panel_title(a, "a", "Final probe, 40% wrong seed entries" + (" (n = %d probes)" % n if n else ""))
    st.tuning_stamp(a, "tuning set", x=1.0, y=1.02)
    if ev:
        b = fig.add_subplot(gs[0, 1])
        cnt = ev.get("counts") or {}
        ep, ec, npol, nclean = cnt.get("evicted_polluted"), cnt.get("evicted_clean"), cnt.get("n_polluted_seed"), cnt.get("clean_entries_ever")
        c_rec = "\n(%d of %d wrong seeds)" % (ep, npol) if cnt else ""
        c_pre = "\n(%d of %d evicted)" % (ep, ep + ec) if cnt else ""
        c_fpr = "\n(%d of %d clean entries)" % (ec, nclean) if cnt else ""
        items = [("recall", "Wrong seeds evicted" + c_rec), ("precision", "Evicted entries\nthat were wrong" + c_pre), ("fpr", "Clean entries evicted" + c_fpr)]
        for i, (k, lbl) in enumerate(items):
            v = ev.get(k)
            if not v:
                continue
            y = -i
            b.barh(y, v[0], 0.55, color=st.BLUE, edgecolor=st.INK, lw=0.4, zorder=2)
            if v[1] is not None:
                b.errorbar([v[0]], [y], xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="none", ecolor=st.INK, elinewidth=0.8, capsize=2.5, zorder=3)
            b.text((v[2] if v[2] is not None else v[0]) + 1.5, y, "%.1f%%" % v[0], va="center", ha="left", fontsize=8)
        b.set_yticks([-i for i in range(len(items))])
        b.set_yticklabels([x[1] for x in items])
        b.set_xlim(0, 100)
        b.set_ylim(-len(items) + 0.4, 0.5)
        b.set_xlabel("Share of entries (%)")
        st.grid_x(b)
        oth = ev.get("evicted_total_other_policies") or {}
        none_ = bool(oth) and all(v == 0 for v in oth.values() if v is not None)
        st.panel_title(b, "b", "Evictions by the value-weighted store" + (" (others: none)" if none_ else ""))
        st.tuning_stamp(b, "tuning set", x=0.99, y=0.03)
    st.ref_tag(fig, mode, synthetic)
    return fig


# ====================================================================================================================== F6
def fig_negative(N, mode, synthetic=False):
    if not N:
        return "negative-results block absent"
    P = st.plt()
    from matplotlib.lines import Line2D
    fig, ax = P.subplots(2, 2, figsize=(st.W2, 13.4 * CM), constrained_layout=True)
    # ---------------------------------------------------------------- (a) PPO / A2C cells vs P0
    a = ax[0][0]
    lamcol = {0.0: st.BLUE, 0.01: st.VERM, 0.03: st.GREEN}
    pp = N["ppo"]
    for algo, mk in (("ppo", "o"), ("a2c", "s")):
        for c in pp[algo]:
            col = lamcol[c["lam"]]
            a.plot([c["calls"]], [c["ex"]], marker=mk, ms=5, mfc=col, mec="white", mew=0.4, ls="none", zorder=3)
    p0 = sorted(pp["p0"], key=lambda r: r["calls"])
    a.plot([r["calls"] for r in p0], [r["ex"] for r in p0], color=st.INK, lw=1.3, zorder=2)
    a.plot([r["calls"] for r in p0], [r["ex"] for r in p0], marker="D", ms=4.5, mfc=st.INK, mec="white", mew=0.4, ls="none", zorder=4)
    fk = sorted(pp["fixed_k"], key=lambda r: r["calls"])
    a.plot([r["calls"] for r in fk], [r["ex"] for r in fk], color=st.INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
    ref = pp["p0_ref"]
    a.annotate("fixed loop (P0),\nscore threshold 7 to 9.5", (p0[0]["calls"], p0[0]["ex"]), xytext=(3.9, 69.5), fontsize=8, ha="left", va="center",
               arrowprops=dict(arrowstyle="-", lw=0.6, color=st.INK))
    a.annotate("fixed number of\nrevisions k = 0 to 3", (fk[-1]["calls"], fk[-1]["ex"]), xytext=(6.4, 60.0), fontsize=8, ha="left", va="center",
               arrowprops=dict(arrowstyle="-", lw=0.6, color=st.INK))
    a.set_xlabel("LLM calls per question")
    a.set_ylabel("EX incl. hand-offs as wrong (%)")
    a.set_xlim(2.8, 9.4)
    a.set_ylim(44.5, 71.5)
    st.grid_y(a)
    hs = [Line2D([], [], marker="o", ls="none", ms=5, mfc="#999999", mec="white", label="PPO cell"), Line2D([], [], marker="s", ls="none", ms=5, mfc="#999999", mec="white", label="A2C cell")]
    hl = [Line2D([], [], marker="o", ls="none", ms=5, mfc=lamcol[l], mec="white", label="penalty %g" % l) for l in (0.0, 0.01, 0.03)]
    a.legend(handles=hs + hl, loc="lower right", fontsize=8, labelspacing=0.2, ncol=2, columnspacing=0.8, title="token penalty (colour)", title_fontsize=8, alignment="left")
    st.panel_title(a, "a", "Learned controller")
    st.tuning_stamp(a, "tuning set", x=1.0, y=1.02)
    # ---------------------------------------------------------------- (b) arbitration accuracy
    b = ax[0][1]
    arb = N["arb"]
    rows = [("revise", "root vs\nrevision"), ("pool", "root vs all\ncandidates")]
    yt, yl = [], []
    for i, (k, lbl) in enumerate(rows):
        base = -1.25 * i
        for j, (kk, col, name) in enumerate((("judge", st.BLUE, "judge call"), ("always", st.GREY, "always take the revision"))):
            v = arb[k][kk]
            y = base - 0.4 * j
            b.errorbar([v[0]], [y], xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="none", ecolor=col, elinewidth=1.1, capsize=2.5, zorder=2)
            b.plot([v[0]], [y], marker="o", ms=6, mfc=col, mec="white", mew=0.5, ls="none", zorder=3)
            b.text(v[2] + 1.5, y, "%.0f%%" % v[0], va="center", ha="left", fontsize=8)
        yt.append(base - 0.2)
        yl.append("%s\n(n = %d)" % (lbl, arb[k]["judge"][4]))
    b.axvline(50, color=st.GREY, lw=0.6, ls=":")
    b.set_yticks(yt)
    b.set_yticklabels(yl)
    b.set_ylim(-1.25 * (len(rows) - 1) - 1.9, 0.4)
    b.set_xlim(20, 100)
    b.set_xlabel("Accuracy where root and revision differ (%)")
    st.grid_x(b)
    b.legend(handles=[Line2D([], [], marker="o", ls="none", ms=6, mfc=st.BLUE, mec="white", label="judge call"),
                      Line2D([], [], marker="o", ls="none", ms=6, mfc=st.GREY, mec="white", label="always the revision")], loc="lower center", fontsize=8, labelspacing=0.2)
    st.panel_title(b, "b", "Arbitration judge")
    st.tuning_stamp(b, "tuning set", x=1.0, y=1.02)
    # ---------------------------------------------------------------- (c) forest of EX differences
    c = ax[1][0]
    fo = N["forest"]
    for i, r in enumerate(fo):
        y = -i
        c.errorbar([r["m"]], [y], xerr=[[r["m"] - r["lo"]], [r["hi"] - r["m"]]], fmt="none", ecolor=st.VERM, elinewidth=1.1, capsize=2.5, zorder=2)
        c.plot([r["m"]], [y], marker="o", ms=6, mfc=st.VERM, mec="white", mew=0.5, ls="none", zorder=3)
        c.text(max(r["hi"], 0) + 0.5, y + 0.0, ("%+.1f" % r["m"]).replace("-", "\u2212"), va="center", ha="left", fontsize=8)
    c.axvline(0, color=st.INK, lw=0.7)
    c.set_yticks([-i for i in range(len(fo))])
    c.set_yticklabels([r["short"] for r in fo])
    c.set_ylim(-len(fo) + 0.5, 0.5)
    c.set_xlim(min(r["lo"] for r in fo) - 1.0, max(max(r["hi"] for r in fo), 0) + 2.8)
    c.set_xlabel("Change in EX (percentage points)")
    st.grid_x(c)
    st.panel_title(c, "c", "Switches left off, 95% CI")
    st.tuning_stamp(c, "tuning set", x=1.0, y=1.02)
    # ---------------------------------------------------------------- (d) value-grounding upper bounds
    d = ax[1][1]
    V = N["value"]
    bars = V["bars"]
    colr = [st.SKY, st.BLUE, st.BLUE, st.ORANGE, st.ORANGE]
    for i, r in enumerate(bars):
        d.barh(-i, r["pp"], 0.6, color=colr[i % len(colr)], edgecolor=st.INK, lw=0.4, hatch=["", "", "////", "", "////"][i % 5], zorder=2)
        d.text(r["pp"] + 0.12, -i, "%.1f (%d)" % (r["pp"], r["k"]), va="center", ha="left", fontsize=8)
    d.set_yticks([-i for i in range(len(bars))])
    d.set_yticklabels([r["short"] for r in bars])
    d.set_ylim(-len(bars) + 0.4, 0.5)
    d.set_xlim(0, 8.2)
    d.set_xlabel("Upper bound on EX gain (pp)")
    st.grid_x(d)
    st.panel_title(d, "d", "Value look-ups")
    st.tuning_stamp(d, "tuning set", x=1.0, y=1.02)
    st.ref_tag(fig, mode, synthetic)
    return fig


# ====================================================================================================================== F7
def fig_perdb(B, mode, synthetic=False):
    if not B or not B.get("rows"):
        return "per-database block absent"
    P = st.plt()
    from matplotlib.lines import Line2D
    rows = sorted(B["rows"], key=lambda r: r["delta"])
    fig, ax = P.subplots(figsize=(12.0 * CM, (0.62 * len(rows) + 4.6) * CM), constrained_layout=True)
    ys = list(range(len(rows)))
    for y, r in zip(ys, rows):
        ax.errorbar([r["delta"]], [y], xerr=[[r["delta"] - r["lo"]], [r["hi"] - r["delta"]]], fmt="none", ecolor=st.BLUE, elinewidth=1.1, capsize=2.5, zorder=2)
        ax.plot([r["delta"]], [y], marker="o", ms=6, mfc=st.BLUE, mec="white", mew=0.5, ls="none", zorder=3)
        if r.get("loo") is not None:
            ax.plot([r["loo"]], [y], marker="|", ms=7, mew=1.2, color=st.GREY, ls="none", zorder=2)
    ov = B.get("overall")
    yo = -1.4
    if ov and ov[0] is not None:
        ax.axvline(ov[0], color=st.VERM, lw=0.9, ls=(0, (4, 2)), zorder=1)
        if ov[1] is not None:
            ax.errorbar([ov[0]], [yo], xerr=[[ov[0] - ov[1]], [ov[2] - ov[0]]], fmt="none", ecolor=st.VERM, elinewidth=1.4, capsize=3, zorder=2)
        ax.plot([ov[0]], [yo], marker="D", ms=7, mfc=st.VERM, mec="white", mew=0.5, ls="none", zorder=3)
    ax.axvline(0, color=st.INK, lw=0.7, zorder=1)
    lbls = ["%s (n = %d)" % (r["db"], r["n"]) for r in rows] + (["pooled (databases as clusters)"] if ov and ov[0] is not None else [])
    ax.set_yticks(ys + ([yo] if ov and ov[0] is not None else []))
    ax.set_yticklabels(lbls)
    ax.set_ylim(yo - 3.4, len(rows) - 0.4)
    allx = [r["lo"] for r in rows] + [r["hi"] for r in rows] + [0]
    if ov and ov[0] is not None:
        allx += [ov[0]] + ([ov[1], ov[2]] if ov[1] is not None else [])
    ax.set_xlim(min(allx) - 1.0, max(allx) + 1.0)
    ax.set_xlabel("Gated minus fixed loop, EX (percentage points)")
    st.direction_note(ax, "gated loop worse", "gated loop better", y=-0.13)
    st.grid_x(ax)
    h = [Line2D([], [], marker="o", ls="-", color=st.BLUE, ms=6, mec="white", label="database, item-level 95% CI"),
         Line2D([], [], marker="|", ls="none", color=st.GREY, ms=7, mew=1.2, label="pooled without this database"),
         Line2D([], [], marker="D", ls="-", color=st.VERM, ms=6, mec="white", label="pooled, database-clustered 95% CI")]
    ax.legend(handles=h, loc="lower right", fontsize=8, labelspacing=0.2, frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)
    ax.set_title(B["split"], loc="left", fontsize=8, pad=4)
    st.ref_tag(fig, mode, synthetic)
    return fig


# ====================================================================================================================== F8
AUTO_ORDER = ["exec_error", "timeout", "empty_result", "shape_mismatch", "column_permutation", "row_subset", "row_superset", "value_mismatch"]
AUTO_LAB = {"exec_error": "execution error", "timeout": "timeout", "empty_result": "empty result", "shape_mismatch": "wrong shape (rows or columns)",
            "column_permutation": "columns permuted", "row_subset": "missing rows (subset)", "row_superset": "extra rows (superset)",
            "value_mismatch": "right shape, wrong values", "other": "other", "gold_fail": "reference query fails"}
LLM_LAB = {"L1": "L1 schema linking", "L2": "L2 join path", "L3": "L3 filter or value", "L4": "L4 aggregation, grouping, order", "L5": "L5 wrong criterion",
           "L6": "L6 reference query suspect", "L7": "L7 ambiguous question", "L8": "L8 other"}
SYS_ERR_MARK = {"zs": "o", "full_fixed": "s", "full_gated": "D"}


def _auto_key(c):
    return "exec_error" if c.startswith("exec_error") else c


def fig_errors(E, mode, synthetic=False):
    if not E:
        return "error block absent (needs E10.auto.* or E10.llm.*)"
    P = st.plt()
    from matplotlib.lines import Line2D
    only_a = (mode == "final") and not (E.get("llm") and E["llm"].get("systems"))        # a confirmatory figure never carries the "not available yet" placeholder
    if only_a:
        print("[figures] fig_errors: panel (b) omitted (no E10.llm.* in the results); the figure has panel (a) only")
        fig, a = P.subplots(figsize=(0.62 * st.W2, 7.6 * CM), constrained_layout=True)
        b = None
    else:
        fig, (a, b) = P.subplots(1, 2, figsize=(st.W2, 7.6 * CM), gridspec_kw={"width_ratios": [1.12, 1.0]}, constrained_layout=True)
    # ---------------------------------------------------------------- (a) automatic classes of failed answers (Cleveland dot plot, one row per class)
    auto = E.get("auto") or []
    classes = []
    for r in auto:
        for k in r["classes"]:
            k2 = _auto_key(k)
            if k2 not in classes:
                classes.append(k2)
    share = {}
    for r in auto:
        tot = max(r["n_fail"], 1)
        for k, v in r["classes"].items():
            share[(r["system"], _auto_key(k))] = share.get((r["system"], _auto_key(k)), 0.0) + 100.0 * v / tot
    ref = "full_gated" if any(r["system"] == "full_gated" for r in auto) else (auto[0]["system"] if auto else None)
    classes.sort(key=lambda k: -share.get((ref, k), 0.0))
    for y in range(len(classes)):
        a.axhline(-y, color="#efefef", lw=0.6, zorder=0)
    for jr, r in enumerate(auto):
        xs = [share.get((r["system"], k), 0.0) for k in classes]
        off = 0.24 * (jr - (len(auto) - 1) / 2.0)
        a.plot(xs, [-y - off for y in range(len(classes))], marker=SYS_ERR_MARK.get(r["system"], "o"), ms=5.5, mfc=st.SYS_COL[r["system"]], mec="white", mew=0.5, ls="none", zorder=3,
               label="%s (%d failed)" % (st.SYS_LAB[r["system"]], r["n_fail"]))
    a.set_yticks([-y for y in range(len(classes))])
    a.set_yticklabels([AUTO_LAB.get(k, k) for k in classes])
    a.set_ylim(-len(classes) + 0.4, 0.6)
    a.set_xlim(0, max(share.values() or [10]) * 1.12)
    a.set_xlabel("Share of failed answers (%%), %s" % E["split"].split(" (")[0].replace("BIRD dev", "development set"))
    st.grid_x(a)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), fontsize=8, labelspacing=0.2, handletextpad=0.3, frameon=False, ncol=1)        # below the rows: never on a data point
    st.panel_title(a, "a" if b is not None else "", "Automatic class of failed answers")
    if b is None:
        st.ref_tag(fig, mode, synthetic)
        return fig
    # ---------------------------------------------------------------- (b) LLM labels L1-L8: empty until E10 labels exist
    llm = E.get("llm")
    st.panel_title(b, "b", "Model-based failure annotation")
    labs = list(LLM_LAB)
    if llm and llm.get("systems"):
        ss = llm["systems"]
        ref2 = [s_ for s_ in ss if s_["system"] == "full_fixed"] or [s_ for s_ in ss if s_["system"] == "full_gated"] or ss      # the fixed loop is the E10 headline system
        order = sorted(labs, key=lambda l: -(ref2[0]["labels"].get(l, [0])[0]))
        for y, l in enumerate(order):
            for j, s_ in enumerate(ss):
                v = s_["labels"].get(l)
                if not v:
                    continue
                yy = -y - 0.22 * (j - (len(ss) - 1) / 2.0)
                b.errorbar([v[0]], [yy], xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="none", ecolor=st.SYS_COL[s_["system"]], elinewidth=0.9, capsize=1.8, zorder=2)
                b.plot([v[0]], [yy], marker=SYS_ERR_MARK.get(s_["system"], "o"), ms=5.5, mfc=st.SYS_COL[s_["system"]], mec="white", mew=0.5, ls="none", zorder=3)
        b.set_yticks([-y for y in range(len(order))])
        b.set_yticklabels([LLM_LAB[l] for l in order])
        b.set_ylim(-len(order) - 1.5, 0.6)
        b.set_xlim(0, None)
        b.set_xlabel("Share of annotated failures (%), 95% CI")
        st.grid_x(b)
        b.legend(handles=[Line2D([], [], marker=SYS_ERR_MARK.get(s_["system"], "o"), ls="none", ms=5.5, mfc=st.SYS_COL[s_["system"]], mec="white", label="%s (n = %s)" % (st.SYS_LAB[s_["system"]], s_["n"])) for s_ in ss],
                 loc="lower right", fontsize=8, labelspacing=0.2)
    else:
        b.set_xlim(0, 1)
        b.set_ylim(0, 1)
        b.set_xticks([])
        b.set_yticks([])
        for sp in b.spines.values():
            sp.set_visible(True)
            sp.set_linestyle((0, (4, 3)))
            sp.set_color(st.GREY)
        b.text(0.5, 0.60, "not available yet", ha="center", va="center", fontsize=8, weight="bold", color=st.DGREY)
        b.text(0.5, 0.46, "fills from the E10 LLM labels\n(L1-L8, two runs + adjudication)\nwhen results/E10.json\ncarries E10.llm.*", ha="center", va="center", fontsize=8,
               color=st.DGREY, linespacing=1.3)
    st.ref_tag(fig, mode, synthetic)
    return fig
