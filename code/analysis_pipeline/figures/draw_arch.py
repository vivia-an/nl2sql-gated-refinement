"""F1: teaser figure.  (a) system overview with the execution gate (vector drawing, no data; source paper/sections/03_method.tex, docs/PATENT_METHOD.md), (b) a worked
example: a real question of the tuning set whose first query failed to execute and was repaired by the gate (block 'case' of the figure data; shown in every mode with the
stamp 'tuning set').  Round 3 (2026-10-28): the former panel (b), a bar pair of calls and cost per question squeezed into the diagram, is dropped (Section 4.1 and F2 give the
numbers; the data stay in the figure data block 'cost'); the former panel (c) is now (b).  Canvas units are millimetres (figure width 170 mm), text 8 pt, SQL in DejaVu Sans
Mono 8 pt; the PDF is saved at <= 175 mm so that the 17.5 cm print width never shrinks it."""
import difflib
import re

from . import style as st

TOP_H = 107.0                       # height (mm) of the pipeline canvas; the worked-example canvas height follows from the SQL length (_ex_height)
CW = 1.70                           # mm per character of DejaVu Sans Mono 8 pt
LH = 3.5                            # mm per SQL line
RED_BG, GRN_BG = "#f2b9a6", "#a9dcc3"


def _fmt_sql(sql, width):
    """Whitespace-only reformatting: keep the pipeline's own line breaks, move ON / AND to an indented continuation line when a line is too long, hard-wrap the rest."""
    out = []
    for ln in [l.rstrip() for l in sql.strip().split("\n")]:
        ind = len(ln) - len(ln.lstrip())
        ln = ln.strip()
        parts = [ln]
        if len(ln) + ind > width:
            parts = re.split(r"\s+(?=ON\b|AND\b|OR\b)", ln)
        for j, p in enumerate(parts):
            pre = " " * (ind + (2 if j else 0))
            while len(pre) + len(p) > width:
                cut = p.rfind(" ", 0, width - len(pre))
                cut = cut if cut > 0 else width - len(pre)
                out.append(pre + p[:cut])
                p = p[cut:].strip()
                pre = " " * (ind + 4)
            out.append(pre + p)
    return out


def _tokens(lines):
    toks = []
    for i, ln in enumerate(lines):
        for m in re.finditer(r"\w+|[^\w\s]", ln):
            toks.append((m.group(0), i, m.start(), m.end()))
    return toks


def _diff_spans(a_lines, b_lines):
    """Token-level diff of two formatted SQL texts -> (spans_a, spans_b) of (line, c0, c1) to highlight (deleted or replaced / inserted or replaced)."""
    ta, tb = _tokens(a_lines), _tokens(b_lines)
    sm = difflib.SequenceMatcher(None, [t[0] for t in ta], [t[0] for t in tb], autojunk=False)
    sa, sb = [], []
    for tag, i0, i1, j0, j1 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            sa += [(t[1], t[2], t[3]) for t in ta[i0:i1]]
        if tag in ("replace", "insert"):
            sb += [(t[1], t[2], t[3]) for t in tb[j0:j1]]
    return sa, sb


def _merge(spans):
    spans = sorted(spans)
    out = []
    for l, c0, c1 in spans:
        if out and out[-1][0] == l and c0 - out[-1][2] <= 1:
            out[-1] = (l, out[-1][1], max(c1, out[-1][2]))
        else:
            out.append((l, c0, c1))
    return out


def _ex_height(case):
    wc = int((83.0 - 7.0) / CW)
    nl = max(len(_fmt_sql(case["root_sql"], wc)), len(_fmt_sql(case["final_sql"], wc)))
    nq = _wrap("Question: " + case["question"], 112).count("\n") + 1
    return 0.8 + 4.6 + 4.2 * nq + 1.6 + 6.0 + nl * LH + 8.5 + 1.5


def draw(mode="final", synthetic=False, case=None, repair=None, cost=None):
    P = st.plt()
    EX_H = _ex_height(case) if case else 0.0
    from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch, Rectangle
    W, H = 170.0, 100.0
    fig = P.figure(figsize=(W / 25.4, (TOP_H + EX_H) / 25.4))
    HT = TOP_H + EX_H
    ax = fig.add_axes([0, (HT - TOP_H) / HT, 1, TOP_H / HT])
    ax.set_xlim(0, W)
    ax.set_ylim(-1.0, 106.0)
    ax.axis("off")

    FILL = {"ctx": "#dcecf7", "mem": "#f3dfeb", "actor": "#fbe8c9", "exec": "#d5eadf", "route": "#e4e4f4", "out": "#eeeeee", "opt": "#ffffff"}

    def box(x, y, w, h, text, kind="ctx", dashed=False, bold_first=True, fs=8, lw=0.8):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.8", fc=FILL[kind], ec=st.INK if not dashed else st.DGREY, lw=lw,
                           ls=(0, (3, 2)) if dashed else "-", zorder=2)
        ax.add_patch(p)
        lines = text.split("\n")
        # first line bold (module name), others regular, stacked from the top
        lh = 3.35
        n = len(lines)
        y0 = y + h / 2.0 + (n - 1) * lh / 2.0
        for i, ln in enumerate(lines):
            ax.text(x + w / 2.0, y0 - i * lh, ln, ha="center", va="center", fontsize=fs, weight="bold" if (i == 0 and bold_first) else "normal", color=st.INK, zorder=4)

    def arrow(p0, p1, dashed=False, rad=0.0, color=None, lw=1.0, ms=7, style="-|>"):
        a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, lw=lw, color=color or (st.DGREY if dashed else st.INK), ls=(0, (3, 2)) if dashed else "-",
                            connectionstyle="arc3,rad=%g" % rad, shrinkA=0, shrinkB=0, zorder=3)
        ax.add_patch(a)

    def line(pts, dashed=False, color=None, lw=1.0, ms=7):
        c = color or (st.DGREY if dashed else st.INK)
        ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]], color=c, lw=lw, ls=(0, (3, 2)) if dashed else "-", solid_capstyle="butt", zorder=3)
        arrow(pts[-2], pts[-1], dashed=dashed, color=color, lw=lw, ms=ms)

    def lab(x, y, s, ha="center", va="center", fs=8, style="normal", color=None):
        ax.text(x, y, s, ha=ha, va=va, fontsize=fs, style=style, color=color or st.INK, zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9))

    # ------------------------------------------------------------------ row 1: question, M1, M3 and its two knowledge sources
    box(0, 82, 21, 16, "Question q\n+ evidence", "out")
    box(27, 82, 30, 16, "M1 intent model\ntask graph g", "ctx")
    box(63, 70, 42, 28, "M3 hybrid retrieval\ndomain routing,\nBM25 + dense,\nrerank: context C", "ctx")
    box(112, 85, 36, 13, "M2 knowledge fusion\nschema, history,\nlineage, terms", "ctx")
    box(112, 70, 36, 13, "M6 experience store\nvalue-weighted\nrecall, top 4", "mem")
    arrow((21, 90), (27, 90))
    arrow((57, 90), (63, 90))
    arrow((112, 91.5), (105, 91.5))
    arrow((112, 76.5), (105, 76.5))

    # ------------------------------------------------------------------ row 2: actor-critic loop container
    cont = FancyBboxPatch((50, 27), 113, 39, boxstyle="round,pad=0,rounding_size=2.5", fc="#fafafa", ec=st.GREY, lw=0.8, zorder=1)
    ax.add_patch(cont)
    ax.text(52.5, 28.2, "Actor-critic loop (M4, M5)", ha="left", va="bottom", fontsize=8, style="italic", color=st.DGREY, zorder=4)
    box(54, 42, 28, 16, "M4 Actor\ngenerate root y0,\nrevise y(t+1)", "actor")
    box(88, 43, 19, 15, "Execute\nread-only,\n30 s limit", "exec")
    gx, gy, gw, gh = 124, 48.5, 12.0, 12.5
    ax.add_patch(Polygon([(gx - gw, gy), (gx, gy + gh), (gx + gw, gy), (gx, gy - gh)], closed=True, fc="#cfe3f3", ec=st.INK, lw=0.8, zorder=2))
    ax.text(gx, gy + 3.0, "Execution", ha="center", va="center", fontsize=8, weight="bold", zorder=4)
    ax.text(gx, gy - 0.4, "gate", ha="center", va="center", fontsize=8, weight="bold", zorder=4)
    ax.text(gx, gy - 4.0, "fails?", ha="center", va="center", fontsize=8, zorder=4)
    box(142, 41, 18, 22, "M7\nlearned\npolicy\n(off)", "opt", dashed=True, bold_first=True)
    arrow((82, 50), (88, 50))
    arrow((107, gy), (gx - gw, gy))
    arrow((142, 52), (gx + gw, gy + 1.5), dashed=True, style="<|-|>", ms=6)
    lab(151, 37.5, "replaces\nthe gate", fs=8, style="italic", color=st.DGREY)
    # context into the actor
    arrow((66, 70), (66, 58))
    lab(64.4, 68.0, "C, g", ha="right", fs=8)
    # revise loop (error or timeout): gate top -> back to the actor
    line([(gx, gy + gh), (gx, 60.3), (76, 60.3), (76, 58)], ms=7)
    lab(100, 63.4, "yes: revise with the error text, up to 3 times", fs=8)
    # fixed-policy comparator: critic on every candidate
    box(76, 32.5, 40, 8.6, "Critic V3, fixed loop only\nrevise if score < 8.5", "opt", dashed=True, bold_first=False)
    arrow((97.5, 43), (97.5, 41.1), dashed=True)
    line([(76, 36.8), (68, 36.8), (68, 42)], dashed=True, ms=6)

    # ------------------------------------------------------------------ row 3: answer, routing, decision (right to left), memory feedback
    box(118, 4, 34, 17, "Answer\nbest candidate by\nexecution status", "exec")
    box(62, 4, 50, 17, "Routing score rho\nV3 of the answer + agreement\nof k = 4 sampled queries", "route")
    box(0, 4, 56, 17, "Decision\nrho above threshold: return SQL\nelse: hand off to a person", "out")
    arrow((gx, gy - gh), (gx, 21))
    lab(139, 24.5, "no: accept", fs=8)
    arrow((118, 12.5), (112, 12.5))
    arrow((62, 12.5), (56, 12.5))
    # outcome feedback to the experience store (training stream only)
    line([(152, 12.5), (168, 12.5), (168, 76.5), (148, 76.5)], dashed=True, ms=6)
    ax.text(165.6, 44, "outcome feedback (training stream only)", rotation=90, ha="center", va="center", fontsize=8, color=st.DGREY, zorder=5)
    # ------------------------------------------------------------------ legend: in the free area left of the loop container (the former panel (b) sat here)
    ax.plot([0, 8], [62.0, 62.0], color=st.INK, lw=1.0, clip_on=False)
    ax.text(10, 62.0, "pipeline (gated loop)", fontsize=8, va="center", clip_on=False)
    ax.plot([0, 8], [53.0, 53.0], color=st.DGREY, lw=1.0, ls=(0, (3, 2)), clip_on=False)
    ax.text(10, 53.0, "comparator, optional\nor training-only path", fontsize=8, va="center", linespacing=1.25, clip_on=False)
    if case:
        ax.text(0.0, 41.5, "Numbered badges mark the\nsteps of the example in (b)", fontsize=8, va="center", linespacing=1.25, clip_on=False)
    # ------------------------------------------------------------------ step badges that tie (a) to the worked example (c)
    if case:
        def badge(x, y, t):
            ax.add_patch(FancyBboxPatch((x - 3.4, y - 1.9), 6.8, 3.8, boxstyle="round,pad=0,rounding_size=1.9", fc=st.INK, ec="none", zorder=6))
            ax.text(x, y - 0.1, t, ha="center", va="center", fontsize=8, color="white", weight="bold", zorder=7)
        badge(54, 58, "1,4")
        badge(88, 58, "2,5")
        badge(gx - 9.5, gy + 9.0, "3,6")
    ax.text(0.0, 105.5, r"$\mathbf{(a)}$", ha="left", va="top", fontsize=8)
    ax.text(7.0, 105.5, "Pipeline with the execution gate", ha="left", va="top", fontsize=8)
    if case:
        _example(fig, HT, EX_H, case, P)
    st.ref_tag(fig, mode, synthetic)
    return fig


def _example(fig, HT, EX_H, case, P):
    """(b) worked example: root query, SQLite error, revision; token-level diff highlights; line prefixes - / + so that it also reads in grayscale."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    W = 170.0
    ax = fig.add_axes([0, 0, 1, EX_H / HT])
    ax.set_xlim(0, W)
    ax.set_ylim(0, EX_H)
    ax.axis("off")
    r = fig.canvas.get_renderer()
    t = ax.text(0, 0, "M" * 50, family="monospace", fontsize=8)
    bb = t.get_window_extent(r)
    t.remove()
    cw = bb.width / 50.0 / (fig.get_figwidth() * fig.dpi) * W                    # mm per character
    BW = 83.0
    width_chars = int((BW - 7.0) / cw)
    ra, rb = _fmt_sql(case["root_sql"], width_chars), _fmt_sql(case["final_sql"], width_chars)
    sa, sb = _diff_spans(ra, rb)
    sa, sb = _merge(sa), _merge(sb)
    top = EX_H - 0.8
    ax.text(5.0, top, "Worked example: database %s, %s, gated loop" % (case["db"], case.get("model", "Qwen3.7-Plus")), ha="left", va="top", fontsize=8)
    wq = _wrap("Question: " + case["question"], 112)
    ax.text(5.0, top - 4.6, wq, ha="left", va="top", fontsize=8, linespacing=1.25)
    nq = wq.count("\n") + 1
    ytop = top - 4.6 - 4.2 * nq - 1.6
    nl = max(len(ra), len(rb))
    bh = 6.0 + nl * LH + 8.5
    for i, (x0, title, lines, spans, bg, sign, foot, fcol) in enumerate([
            (0.5, "1  root query y0, 2  execute, 3  gate: fails", ra, sa, RED_BG, "-", "SQLite: " + str(case["error"]) + "  \u2192  revise with this text", st.VERM),
            (86.0, "4  revision y1, 5  execute, 6  gate: passes", rb, sb, GRN_BG, "+", "1 row %s = reference result  \u2192  accept; EX = 1, %d calls" % (_res(case["final_result"]), case["calls"]), st.GREEN)]):
        ax.add_patch(FancyBboxPatch((x0, ytop - bh), BW, bh, boxstyle="round,pad=0,rounding_size=1.5", fc="#fbfbfb", ec=st.GREY, lw=0.8, zorder=1))
        ax.text(x0 + 2.0, ytop - 1.2, title, ha="left", va="top", fontsize=8, weight="bold", color=st.INK, zorder=3)
        y0 = ytop - 6.2
        for (l, c0, c1) in spans:
            ax.add_patch(Rectangle((x0 + 5.5 + c0 * cw, y0 - (l + 1) * LH + 0.35), (c1 - c0) * cw, LH - 0.1, fc=bg, ec="none", zorder=2))
        marked = set(l for (l, _, _) in spans)
        for k, ln in enumerate(lines):
            ax.text(x0 + 2.0, y0 - k * LH - 0.1, sign if k in marked else " ", ha="left", va="top", family="monospace", fontsize=8, color=fcol, weight="bold", zorder=3)
            ax.text(x0 + 5.5, y0 - k * LH, ln, ha="left", va="top", family="monospace", fontsize=8, color=st.INK, zorder=3)
        ax.text(x0 + 2.0, ytop - bh + 1.2, _wrap(foot, 44), ha="left", va="bottom", fontsize=8, color=fcol, weight="bold", zorder=3, linespacing=1.2)
    ax.text(W / 2.0, ytop - bh / 2.0, "\u25B6", ha="center", va="center", fontsize=8, color=st.DGREY)
    st.panel_letter(fig, 0.0, EX_H / HT, "b")
    ax.text(W - 0.5, EX_H - 0.8, "tuning set", ha="right", va="top", fontsize=8, color=st.REFCOL, style="italic")


def _res(rows):
    return "(" + ", ".join(repr(x) for x in rows[0]) + ")" if rows else "()"


def _wrap(s, n):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return "\n".join(lines)
