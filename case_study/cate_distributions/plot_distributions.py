"""Plot CATE and ATE distributions per case study for a set of context sizes.

For each case study we pool the generated realizations (see generation.py):
  * CATE distribution — every per-row tau_i = mu_1(x_i) - mu_0(x_i), pooled
    across all realizations of that (case, N) cell.
  * ATE  distribution — one value per realization: mean_i tau_i.

Layout: one row per case study, one column per context size N. CATE is drawn as
a filled orange histogram; ATE as an outlined dark-green step histogram (the
shape difference is a secondary encoding so the two are distinguishable beyond
colour). Both are densities so the 100-point ATE and the many-point CATE are
comparable on one axis.

    python case_study/plot_distributions.py \
        --data-dir case_study/data --context-sizes 200 500 1000 \
        --out case_study/cate_ate_distributions.png
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generation import CASE_STUDIES

CATE_COLOR = "#E8820C"      # orange
ATE_COLOR = "#1B5E20"       # dark green
INK = "#222222"             # text / axis ink (never the series colour)
MUTED = "#8A8A8A"           # recessive grid / zero line


def load_cell(data_dir: str, case: str, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cate_pooled, ate_per_realization) for one (case, N) cell."""
    paths = sorted(glob.glob(os.path.join(data_dir, case, f"N{N}", f"{case}_*.npz")))
    cate_pool: List[np.ndarray] = []
    ate: List[float] = []
    for p in paths:
        c = np.load(p, allow_pickle=True)["cate"].reshape(-1)
        cate_pool.append(c)
        ate.append(float(c.mean()))
    if not paths:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(cate_pool), np.asarray(ate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="case_study/data")
    ap.add_argument("--context-sizes", nargs="*", type=int, default=[200, 500, 1000])
    ap.add_argument("--cases", nargs="*", default=list(CASE_STUDIES))
    ap.add_argument("--out", default="case_study/cate_ate_distributions.png")
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--title-note", default="",
                    help="Extra text appended to the figure title (e.g. the "
                         "covariate count for a d-sweep). Default: none.")
    ap.add_argument("--shift", type=float, default=None,
                    help="Applied cate-shift beta (for the title). If omitted, "
                         "read from <data-dir>/manifest.json.")
    a = ap.parse_args()

    shift = a.shift
    if shift is None:
        mpath = os.path.join(a.data_dir, "manifest.json")
        if os.path.exists(mpath):
            import json
            shift = json.load(open(mpath)).get("cate_shift", 0.0)

    cases, Ns = a.cases, a.context_sizes
    nrow, ncol = len(cases), len(Ns)

    # Pre-load everything; compute a robust per-row x-range shared across N so
    # the columns are directly comparable for a given case study.
    data: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}
    for case in cases:
        for N in Ns:
            data[(case, N)] = load_cell(a.data_dir, case, N)

    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.1 * nrow),
                             squeeze=False)

    for i, case in enumerate(cases):
        pooled = np.concatenate([data[(case, N)][0] for N in Ns
                                 if data[(case, N)][0].size] or [np.zeros(1)])
        lo, hi = np.percentile(pooled, [0.5, 99.5])
        if lo == hi:                       # (near-)degenerate: pad a little
            lo, hi = lo - 0.5, hi + 0.5
        pad = 0.05 * (hi - lo)
        xlim = (lo - pad, hi + pad)
        edges = np.linspace(*xlim, a.bins + 1)

        for j, N in enumerate(Ns):
            ax = axes[i][j]
            cate, ate = data[(case, N)]

            if cate.size:
                ax.hist(np.clip(cate, *xlim), bins=edges, density=True,
                        color=CATE_COLOR, alpha=0.55, edgecolor="none",
                        label="CATE (per-unit)")
            if ate.size:
                ax.hist(np.clip(ate, *xlim), bins=edges, density=True,
                        histtype="step", color=ATE_COLOR, linewidth=2.0,
                        label="ATE (per-dataset)")

            ax.axvline(0.0, color=MUTED, lw=1.0, ls="--", zorder=0)
            ax.set_xlim(xlim)
            ax.set_yticks([])
            ax.tick_params(colors=INK, labelsize=8)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(MUTED)

            if i == 0:
                ax.set_title(f"N = {N}", color=INK, fontsize=11, pad=8)
            if j == 0:
                ax.set_ylabel(case.replace("_", " "), color=INK, fontsize=9,
                              rotation=0, ha="right", va="center", labelpad=12)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 1.005),
               labelcolor=INK)
    shift_txt = ("" if not shift else
                 f"   (direct-effect cases shifted by β = {shift:+g}; "
                 "mediated cases stay at 0)")
    note = f"  —  {a.title_note}" if a.title_note else ""
    fig.suptitle("CATE vs ATE distributions across case studies" + note + shift_txt,
                 y=1.03, fontsize=13, color=INK)
    fig.text(0.5, -0.01, "treatment effect", ha="center", color=INK, fontsize=10)

    fig.tight_layout(rect=[0.02, 0.0, 1, 0.99])
    fig.savefig(a.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"[plot] wrote {a.out}")

    # Console + JSON summary (sidecar next to the PNG).
    import json
    summary = {"shift": shift, "rows": []}
    for case in cases:
        row = {"case": case, "by_N": {}}
        for N in Ns:
            cate, ate = data[(case, N)]
            if not cate.size:
                continue
            row["by_N"][str(N)] = {
                "cate_mean": float(cate.mean()), "cate_std": float(cate.std()),
                "ate_mean": float(ate.mean()), "ate_std": float(ate.std())}
            print(f"{case:34s} N={N:<5d} "
                  f"CATE[mean={cate.mean():+.3f} std={cate.std():.3f}] "
                  f"ATE[mean={ate.mean():+.3f} std={ate.std():.3f}]")
        summary["rows"].append(row)
    json.dump(summary, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=2)


if __name__ == "__main__":
    main()
