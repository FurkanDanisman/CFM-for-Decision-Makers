"""One look for every figure of the paper (the style of the interval-score figures, is_hbars.py):
STIX text and math (Times, as the paper), TrueType embedding, single-outcome models blue and 2D models orange,
horizontal bars with capped error bars, dotted top/right frame, three ticks per panel.
Figures are drawn at the CLeaR text width, so the font sizes are the printed sizes.

    import paper_style as PS; PS.apply()
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

TEXTWIDTH = 433.62 / 72.27                                       # CLeaR \textwidth, inches
COL_1D, COL_2D = "#3f7fbf", "#e6820e"
COL_GREY = "0.93"                                                # background of the average panels
ERR = dict(elinewidth=0.9, ecolor="0.2", capsize=1.6, capthick=0.9)
TITLE, LABEL, TICK, YTICK = 10.5, 10.5, 9.5, 10


def apply():
    plt.rcParams.update({"font.size": LABEL, "axes.titlesize": TITLE, "axes.labelsize": LABEL,
                         "xtick.labelsize": TICK, "ytick.labelsize": YTICK, "legend.fontsize": LABEL,
                         "font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.linewidth": 0.5})


def frame(ax):
    """dotted top/right frame (as Do-PFN Fig. 3)"""
    for s in ("top", "right"):
        ax.spines[s].set_linestyle((0, (1, 1.5)))
    ax.tick_params(axis="x", length=2, width=0.5, pad=1.5)
    ax.tick_params(axis="y", length=0, pad=1.5)


def three(top, steps=(1, 1.5, 2, 2.5, 3, 4, 5, 6, 8)):
    """exactly three ticks 0, s, 2s, with s the largest round step that keeps 2s inside [0, top]"""
    s = max(m * 10 ** e for e in range(-4, 6) for m in steps if 2 * m * 10 ** e <= top)
    return [0, s, 2 * s]


def fmt(v, _=None):
    if v >= 1000:
        return f"{v / 1000:g}k"
    return f"{v:g}"


def short_ticks(top, steps):
    """three ticks 0, s, 2s; when s has a long label (e.g. 0.05), extend the axis (at most 35%) to a step with a
    short label (0.1) so narrow panels never have touching tick labels. Returns (ticks, axis top)."""
    t = three(top, steps)
    if max(len(fmt(x)) for x in t) <= 3:
        return t, top
    for s in sorted(m * 10 ** e for e in range(-4, 6) for m in (1, 2, 2.5, 5)):
        if s > t[1] and 2 * s <= 1.35 * top and max(len(fmt(s)), len(fmt(2 * s))) <= 3:
            return [0, s, 2 * s], max(top, 2 * s * 1.04)
    return [t[0], t[2]], top                                     # long labels in a narrow panel: drop the middle tick


def hbars(ax, vals, errs, colors, height=0.8, steps=(1, 1.5, 2, 2.5, 3, 4, 5, 6, 8), compact=False):
    """horizontal bars top-down, own linear axis from 0 with three ticks"""
    vals, errs = np.asarray(vals, float), np.asarray(errs, float)
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, height=height, xerr=errs, color=colors, edgecolor="black", linewidth=0.7, error_kw=ERR)
    top = np.nanmax(vals + np.nan_to_num(errs)) * 1.08
    ticks = three(top, steps)
    if compact:
        ticks, top = short_ticks(top, steps)
    ax.set_xlim(0, top)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
    ax.set_ylim(-0.6, len(vals) - 0.4)
    frame(ax)
    return y


def rel_panel(ax, ratios, y, xmax, title=False):
    """last column: each baseline's error relative to the 2D model (2D = 1, orange dashed line); lower is better,
    a bar past the line = the baseline's error is that many times the 2D model's"""
    ratios = np.asarray(ratios, float)
    ax.barh(y, ratios, height=0.8, color=COL_1D, edgecolor="black", linewidth=0.7, zorder=2)
    ax.axvline(1, color=COL_2D, lw=1.8, ls=(0, (3, 1.5)), zorder=3)
    xmax = max(xmax, 1.9)
    for yy, r in zip(y, ratios):                                 # value right of the bar end or of the 2D line
        ax.text(max(r, 1) + 0.05 * xmax, yy, (f"\u00d7{r:.2f}" if abs(r - 1) < 0.095 else f"\u00d7{r:.1f}") if r < 9.95 else f"\u00d7{r:.0f}", va="center",
                ha="left", fontsize=9)
    ax.set_xlim(0, xmax)
    ax.set_yticks([])
    ax.set_xticks([1])
    ax.set_xticklabels(["2D"], color=COL_2D, fontweight="bold")
    ax.set_facecolor(COL_GREY)
    frame(ax)
    if title:
        ax.set_title("Relative\nto 2D", pad=3, fontweight="bold", linespacing=1.0)


def pct_limits(P):
    """common x-range of the 'Compared to 2D (%)' panels of one figure, with room for the printed values"""
    P = np.asarray(P, float)
    neg, pos = max(0, -P.min()), max(0, P.max())
    pad = 0.5 * (neg + pos) + 25
    return -max(60, neg + pad), max(60, pos + pad)


def pct_panel(ax, pct, y, lims, title=False):
    """last column: the 2D model (orange) compared to the 1D baseline (the 0 line, labelled 1D); pct is
    100 (1 - error of 2D / error of the baseline), drawn as its negative, the change of the 2D model:
    - (left) = the 2D model's error is lower, + (right) = higher"""
    pct = -np.asarray(pct, float)
    lims = (-lims[1], -lims[0])
    ax.axvline(0, color="black", lw=0.9, zorder=1)
    ax.barh(y, pct, height=0.8, color=COL_2D, edgecolor="black", linewidth=0.7, zorder=2)
    for yy, x in zip(y, pct):
        ax.text(x + 4 if x >= 0 else x - 4, yy, ("+" if x >= 0 else "-") + str(abs(round(x))), va="center",
                ha="left" if x >= 0 else "right", fontsize=9)
    ax.set_xlim(*lims)
    ax.set_xticks([0])
    ax.set_xticklabels(["1D"])
    ax.set_yticks([])
    ax.set_facecolor(COL_GREY)
    frame(ax)
    if title:
        ax.set_title(r"$\Delta$ (%)", pad=3, fontweight="bold")


def avg_panel(ax, vals, errs, colors, title=False):
    """last column: the average over datasets (+- SE), its value printed at the bar end, no axis ticks"""
    vals, errs = np.asarray(vals, float), np.asarray(errs, float)
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, height=0.8, xerr=errs, color=colors, edgecolor="black", linewidth=0.7, error_kw=ERR)
    top = np.nanmax(vals + np.nan_to_num(errs))
    for yy, v, e in zip(y, vals, np.nan_to_num(errs)):
        ax.text(v + e + 0.05 * top, yy, (f"{v:.2f}" if v < 10 else f"{v:.1f}") if v < 100 else f"{v:.0f}", va="center", ha="left", fontsize=9)
    ax.set_xlim(0, top * 1.75)
    ax.set_ylim(-0.6, len(vals) - 0.4)
    ax.set_xticks([])                                            # no ticks: the values are printed
    ax.set_yticks([])
    ax.set_facecolor(COL_GREY)
    frame(ax)
    if title:
        ax.set_title("Average", pad=3, fontweight="bold")


def gmean_se(V, E, sig=None):
    """geometric mean over datasets (axis 1) of V / sig, with its delta-method SE (datasets independent)"""
    V, E = np.asarray(V, float), np.asarray(E, float)
    s = np.ones(V.shape[1]) if sig is None else np.asarray(sig, float)
    X, S = V / s, E / s
    g = np.exp(np.mean(np.log(X), axis=1))
    return g, g * np.sqrt(np.sum((S / X) ** 2, axis=1)) / V.shape[1]
