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
Bars carry no value labels by default (--bar-labels re-enables them); the y axis
carries the magnitudes. The palette clears the >= 3:1 contrast check on its own,
so no label-based relief is required for it.

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


def draw_one(rows, mi, args, out_path, show_legend):
    """One metric, one figure. No suptitle, no null line — axis labels carry it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ylab = ("sqrt(PEHE)", "ATE $L_1$")[mi]
    fig, ax = plt.subplots(figsize=(6.6, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    x = np.arange(len(rows))
    w, gap = 0.34, 0.02
    v1 = [r[1][mi]["mean"] for r in rows]
    e1 = [r[1][mi]["sem"] for r in rows]
    v2 = [r[2][mi]["mean"] for r in rows]
    e2 = [r[2][mi]["sem"] for r in rows]

    ax.bar(x - w/2 - gap/2, v1, w, yerr=e1, capsize=3, color=C_1D,
           label="1D head", error_kw=dict(ecolor=INK_MUTED, lw=1.1))
    ax.bar(x + w/2 + gap/2, v2, w, yerr=e2, capsize=3, color=C_2D,
           label="2D head", error_kw=dict(ecolor=INK_MUTED, lw=1.1))

    top = max(max(np.add(v1, e1)), max(np.add(v2, e2)))
    if args.bar_labels:
        ax.set_ylim(0, top * (1.30 if show_legend else 1.16))
        for xs, vals in ((x - w/2 - gap/2, v1), (x + w/2 + gap/2, v2)):
            for xi, v in zip(xs, vals):
                ax.text(xi, v + top * 0.025, f"{v:.3f}", ha="center",
                        va="bottom", fontsize=7.5, color=INK)
    else:
        # No value labels: the y axis carries the magnitudes. Less headroom is
        # needed, which lets the bars fill more of the panel.
        ax.set_ylim(0, top * (1.22 if show_legend else 1.06))

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=9, color=INK)
    ax.set_ylabel(ylab, fontsize=10.5, color=INK)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.xaxis.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=0)
    if show_legend:
        ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def render_cell(args):
    """Build one (nodes, context, subset) cell and write its two figures."""
    rows, _nulls, missing = build(args)
    if not rows:
        return [], missing
    # Legend rides on exactly one panel of the set; elsewhere it is redundant
    # and eats headroom. Default: the N=50 / d=5 cell, per the figure plan.
    if args.legend == "always":
        show = True
    elif args.legend == "never":
        show = False
    else:
        show = (args.context == args.legend_context
                and args.nodes == args.legend_nodes)
    made = []
    for mi, key in ((0, "pehe"), (1, "ate")):
        out = os.path.join(
            args.outdir,
            f"bar_{key}_d{args.nodes}_N{args.context}_{args.subset}.png")
        made.append(draw_one(rows, mi, args, out, show))
    return made, missing


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
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--all", action="store_true",
                    help="render every (node count, context) combination")
    ap.add_argument("--all-nodes", type=int, nargs="+",
                    default=[5, 10, 20, 30, 40, 50])
    ap.add_argument("--all-contexts", type=int, nargs="+",
                    default=[50, 100, 250, 500, 1000])
    ap.add_argument("--legend", default="auto", choices=["auto", "always", "never"],
                    help="auto = legend only on the --legend-context/--legend-nodes "
                         "cell (default N=50, d=5)")
    ap.add_argument("--legend-context", type=int, default=50)
    ap.add_argument("--legend-nodes", type=int, default=5)
    ap.add_argument("--bar-labels", action="store_true",
                    help="print each bar's value above it (off by default; the "
                         "y axis already carries the magnitude)")
    args = ap.parse_args()
    if args.outdir is None:
        args.outdir = os.path.join(args.root, "figures")
    os.makedirs(args.outdir, exist_ok=True)

    combos = ([(n, c) for c in args.all_contexts for n in args.all_nodes]
              if args.all else [(args.nodes, args.context)])

    made, skipped = [], []
    for n, c in combos:
        args.nodes, args.context = n, c
        try:
            files, missing = render_cell(args)
        except Exception as exc:  # noqa: BLE001
            skipped.append((n, c, str(exc)[:70]))
            continue
        if not files:
            skipped.append((n, c, "no data"))
            continue
        if missing:
            print(f"  [warn] d={n} N={c}: omitted {', '.join(missing)}")
        made += files

    for f in made:
        print(f"[plot] {f}")
    print(f"\n{len(made)} figures -> {args.outdir}")
    if skipped:
        print(f"{len(skipped)} cells skipped:")
        for n, c, why in skipped:
            print(f"   d={n:<3} N={c:<5} {why}")


if __name__ == "__main__":
    main()
