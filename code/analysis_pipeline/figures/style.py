"""Common plotting style: MDPI sizes (single column 8.5 cm, full width 17 cm), fonts >= 8 pt, Okabe-Ito palette (colour-blind safe), grayscale-safe via
marker shapes / hatches / direct labels, no chart junk.  Needs matplotlib (worker machine only; host Python 3.6 has none)."""
import os

CM = 1.0 / 2.54
W1, W2 = 8.5 * CM, 17.0 * CM          # figure widths in inches

# Okabe-Ito
BLUE, VERM, GREEN, PURPLE, ORANGE, SKY, YELLOW, BLACK = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"
GREY, LGREY, DGREY = "#8c8c8c", "#d9d9d9", "#4d4d4d"
INK = "#1a1a1a"
REFCOL = "#B00020"

SYS_COL = {"zs": DGREY, "full_fixed": ORANGE, "full_gated": BLUE, "full_fixed_legacy": VERM}     # fixed (light) vs gated (dark) differ in luminance: grayscale-safe
SYS_LAB = {"zs": "single call", "full_fixed": "fixed loop", "full_gated": "gated loop", "full_fixed_legacy": "fixed loop, legacy renderer"}
MODEL_MARK = {"qwen37flash": "o", "qwen3527b": "^", "qwen37plus": "s"}
MODEL_LAB = {"qwen37flash": "Qwen3.7-Flash", "qwen3527b": "Qwen3.5-27B", "qwen37plus": "Qwen3.7-Plus"}
POLICY_LAB = {"append": "append-only", "memrl": "utility-ranked\n(MemRL-style)", "ws_full": "value-weighted\n(ws_full)", "ws_evict": "ws_evict", "static": "static"}
POLICY_COL = {"append": DGREY, "memrl": ORANGE, "ws_full": BLUE, "ws_evict": GREEN, "static": GREY}


def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt_
    plt_.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 4.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK, "axes.grid": False,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.handletextpad": 0.5, "legend.borderaxespad": 0.2, "hatch.linewidth": 0.5,
        "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "mathtext.default": "regular", "font.monospace": ["DejaVu Sans Mono"]})
    return plt_


def panel_title(ax, letter, text, pad=4):
    """Bold panel letter (as in the caption: '(a) ...'), regular title text."""
    ax.set_title((r"$\mathbf{(%s)}$ %s" % (letter, text)) if letter else text, loc="left", fontsize=8, pad=pad)


def panel_letter(fig, x, y, letter, ha="left", va="top"):
    """Bold panel letter at figure-fraction coordinates (for hand-drawn canvases)."""
    fig.text(x, y, r"$\mathbf{(%s)}$" % letter, ha=ha, va=va, fontsize=8)


def direction_note(ax, left, right, y=-0.30):
    """Small grey 'which side is better' note under a signed axis (axes-fraction coordinates)."""
    ax.text(0.0, y, "\u2190 " + left, transform=ax.transAxes, ha="left", va="top", fontsize=8, color=DGREY)
    ax.text(1.0, y, right + " \u2192", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=DGREY)


def grid_x(ax):
    ax.grid(axis="x", color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)


def grid_y(ax):
    ax.grid(axis="y", color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)


def ref_tag(fig, mode, synthetic=False):
    """Corner label.  reference mode: 'REFERENCE (tuning set)' on every figure; final mode: none; synthetic previews: a loud watermark."""
    if mode == "reference":
        fig.text(0.0, 1.0, "REFERENCE (tuning set)", ha="left", va="bottom", fontsize=8, color=REFCOL, weight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=REFCOL, lw=0.8))
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC - NOT RESULTS", fontsize=26, color="#CC0000", alpha=0.22, ha="center", va="center", rotation=28, weight="bold")


# printed width (cm) of each paper figure: paper/preamble.tex \\figincl prints 'wide' at 17.5 cm; a PDF wider than that is shrunk and its 8 pt text with it (round 3, T9)
PRINT_W_CM = {"fig1_architecture": 17.5, "fig_repair": 17.5, "fig_risk_coverage": 17.5, "fig_pollution": 17.5, "fig_negative": 17.5, "fig_errors": 17.5, "fig_quoting": 11.0}


def fit_width(fig, max_cm, pad_inches=0.04):
    """Shrink the figure width until its tight bounding box (plus the save padding) is no wider than max_cm: the PDF is then printed at scale >= 1, so 8 pt text stays >= 8 pt."""
    for _ in range(6):
        fig.canvas.draw()
        tb = fig.get_tightbbox(fig.canvas.get_renderer())
        w_cm = (tb.width + 2 * pad_inches) / CM
        if w_cm <= max_cm + 0.005:
            return w_cm
        fig.set_size_inches(fig.get_figwidth() - (w_cm - max_cm + 0.02) * CM, fig.get_figheight())
    return w_cm


def save(fig, out_dir, stem, max_w_cm=None):
    os.makedirs(out_dir, exist_ok=True)
    mw = max_w_cm or PRINT_W_CM.get(stem)
    if mw:
        fit_width(fig, mw)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, "%s.%s" % (stem, ext))
        fig.savefig(p, bbox_inches="tight", pad_inches=0.04)
        paths.append(p)
    return paths


def tuning_stamp(ax, text="tuning set", x=1.0, y=1.02, ha="right"):
    """Permanent stamp on panels whose evidence is tuning-set only, in every mode."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va="bottom", fontsize=8, color=REFCOL, style="italic")


def fmt_ci(m, lo, hi, d=1, sign=True):
    f = "%+." + str(d) + "f" if sign else "%." + str(d) + "f"
    if lo is None or hi is None:
        out = (f % m)
    else:
        out = (f + " [" + f + ", " + f + "]") % (m, lo, hi)
    return out.replace("-", "\u2212")                         # true minus sign, as in the other figures
