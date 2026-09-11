"""Grouped bar chart: 1D vs 2D head, one cell of the ComplexMech benchmark.

Four groups, each a 1D/2D pair:

    Do-PFN        / Do-PFN 2D        dopfn_native  / dopfn_bb
    UWYK No-Anc   / UWYK No-Anc 2D   uwyk1d-noanc  / graph2d-noanc
    UWYK Anc      / UWYK Anc 2D      uwyk1d-v3a    / graph2d-v3a
    CausalPFN-C   / CausalPFN-C 2D   cpfn1d        / cpfn2d

sqrt(PEHE) and ATE L1 go in SEPARATE panels, never a shared axis — they are
different measures on different scales.

The dashed line is the null: the score of predicting tau=0 for every query. A
bar reaching it has no skill, so it is the reference the eye needs, not zero.
Bars are labelled with their value, which also supplies the non-colour encoding
the palette check requires.

Palette #D97706 / #B91C1C validated with the dataviz six-checks (light surface):
CVD separation dE 16.2 deutan / 14.6 tritan, normal-vision 18.8, both >= 3:1
contrast — all PASS.

Usage
-----
    python UWYK_Fig3_4/plot_1d_vs_2d.py --root $SCRATCH/cmech_v3 \
        --context 1000 --nodes 50
    python UWYK_Fig3_4/plot_1d_vs_2d.py --root ... --subset total --cpfn2d log
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.join(os.path.dirname(_HERE), "benchmarks")
for _p in (_BENCH, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

C_1D, C_2D = "#D97706", "#B91C1C"
INK, INK_MUTED, GRID = "#1c1917", "#57534e", "#e7e5e4"
SURFACE = "#fcfcfb"


def build(args):
    from aggregate_cmech_methods import (
        combine_total, load_cell, null_row, stats,
    )

    cp2 = f"cpfn2d_{args.cpfn2d}"
    groups = [
        ("Do-PFN",       ("dopfn_native", None), ("dopfn_bb", None)),
        ("UWYK No-Anc",  ("uwyk1d", "noanc"),    ("graph2d", "noanc")),
        ("UWYK Anc",     ("uwyk1d", "v3a"),      ("graph2d", "v3a")),
        ("CausalPFN-C",  ("cpfn1d", None),       (cp2, None)),
    ]
    root = os.path.join(args.root, f"N{args.context}")
    ds_all = f"CMECH_n{args.nodes}_{args.subset}"

    def series(subdir, tag):
        if args.subset == "total":
            gz = load_cell(root, subdir, tag, f"CMECH_n{args.nodes}_zero")
            gn = load_cell(root, subdir, tag, f"CMECH_n{args.nodes}_nonzero")
            if not gz or not gn:
                return None
            got = combine_total(gn, gz, args.nodes, args.data_root)
        else:
            got = load_cell(root, subdir, tag, ds_all)
        if not got:
            return None
        return (stats([v[0] for v in got.values()]),
                stats([abs(v[1] - v[2]) for v in got.values()]))

    rows, missing = [], []
    for label, one, two in groups:
        a, b = series(*one), series(*two)
        if a is None or b is None:
            missing.append(label)
            continue
        rows.append((label, a, b))

    npe, nl1 = null_row(args.nodes, args.subset, args.data_root)
    nulls = (float(np.mean(npe)) if npe else np.nan,
             float(np.mean(nl1)) if nl1 else np.nan)
    return rows, nulls, missing


def draw(rows, nulls, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("sqrt(PEHE)", 0, nulls[0]), ("ATE $L_1$", 1, nulls[1])]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor=SURFACE)
    x = np.arange(len(rows))
    # 2px-equivalent gap between the paired bars, per the mark spec.
    w, gap = 0.34, 0.02

    for ax, (name, mi, null) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        v1 = [r[1][mi]["mean"] for r in rows]
        e1 = [r[1][mi]["sem"] for r in rows]
        v2 = [r[2][mi]["mean"] for r in rows]
        e2 = [r[2][mi]["sem"] for r in rows]

        b1 = ax.bar(x - w/2 - gap/2, v1, w, yerr=e1, capsize=3,
                    color=C_1D, label="1D head",
                    error_kw=dict(ecolor=INK_MUTED, lw=1.1))
        b2 = ax.bar(x + w/2 + gap/2, v2, w, yerr=e2, capsize=3,
                    color=C_2D, label="2D head",
                    error_kw=dict(ecolor=INK_MUTED, lw=1.1))

        if np.isfinite(null):
            # Labelled via the legend, not inline: an inline annotation collides
            # with the bar value labels whenever a bar sits near the null, which
            # is most of them at high node counts.
            ax.axhline(null, ls=(0, (5, 3)), lw=1.3, color=INK_MUTED, zorder=1)

        top = max(max(np.add(v1, e1)), max(np.add(v2, e2)),
                  null if np.isfinite(null) else 0)
        # Extra headroom on the left panel: it carries the legend, which
        # otherwise grazes the tallest bar's value label.
        ax.set_ylim(0, top * (1.40 if mi == 0 else 1.22))
        for bars, vals in ((b1, v1), (b2, v2)):
            for rect, v in zip(bars, vals):
                ax.text(rect.get_x() + rect.get_width()/2,
                        rect.get_height() + top * 0.025, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7.5, color=INK)

        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows], fontsize=9, color=INK)
        ax.set_ylabel(name, fontsize=10, color=INK)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.xaxis.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=0)

    from matplotlib.lines import Line2D
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_1D, label="1D head"),
        plt.Rectangle((0, 0), 1, 1, color=C_2D, label="2D head"),
        Line2D([0], [0], ls=(0, (5, 3)), lw=1.3, color=INK_MUTED,
               label="null — predict $\\tau$=0 (no skill)"),
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=8.5,
                   loc="upper left", labelcolor=INK)
    sub = {"nonzero": "queries with a non-zero effect",
           "zero": "queries with an exactly-zero effect",
           "total": "all queries"}[args.subset]
    fig.suptitle(f"ComplexMech — {args.nodes} nodes, context N={args.context} "
                 f"({sub}); lower is better",
                 fontsize=11.5, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"[plot] {args.out}")

    print(f"\n{'group':<16}{'1D':>18}{'2D':>18}   (metric: sqrt(PEHE))")
    for label, a, b in rows:
        print(f"{label:<16}{a[0]['mean']:>10.4f} ± {a[0]['sem']:<5.4f}"
              f"{b[0]['mean']:>10.4f} ± {b[0]['sem']:<5.4f}")
    print(f"{'null':<16}{nulls[0]:>10.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="OUT_ROOT holding N<ctx>/ dirs")
    ap.add_argument("--context", type=int, default=1000)
    ap.add_argument("--nodes", type=int, default=50)
    ap.add_argument("--subset", default="nonzero",
                    choices=["nonzero", "zero", "total"])
    ap.add_argument("--cpfn2d", default="pooled", choices=["pooled", "log"])
    ap.add_argument("--data-root", default=os.environ.get(
        "UWYK_FIG34_DATA", os.path.join(_HERE, "data")))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(args.root,
                                f"bar_1d_vs_2d_n{args.nodes}_N{args.context}"
                                f"_{args.subset}.png")

    rows, nulls, missing = build(args)
    if missing:
        print(f"[warn] no data for: {', '.join(missing)} — omitted from the plot")
    if not rows:
        raise SystemExit(f"no data under {args.root}/N{args.context}")
    draw(rows, nulls, args)


if __name__ == "__main__":
    main()
