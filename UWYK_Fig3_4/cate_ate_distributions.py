"""Distribution of true CATE and ATE per dataset, for the Fig-3 / Fig-4 priors.

For every cell (prior x n_nodes x regime) this reports:

* **CATE**, pooled over all units of all realizations. tau_i = Y_do1 - Y_do0 with
  the exogenous noise held fixed across arms, so this is a per-unit ITE.
* **ATE**, one value per realization (dataset): ATE_r = mean_i tau_i. Its spread
  across realizations is the spread of effect sizes the prior produces.
* **Within-realization CATE sd**, averaged over realizations. This is the oracle
  sqrt(PEHE) floor: no model can score below it, and it is 0 exactly when the
  prior generates no treatment-effect heterogeneity.

The last one is the number to read first. If it is ~0, PEHE on that cell is an
ATE benchmark wearing a PEHE label, because the best possible tau_hat(x) is a
constant.

`hide_fraction` is not an axis here. The ComplexMechIDK YAMLs are byte-identical
across `hide_0.0 ... hide_1.0` except for `hide_fraction_matrix`, which only masks
the ancestor matrix handed to the model -- it never touches the SCM, so the CATE
and ATE distributions are the same for every hide fraction.

Regimes `path_YT` and `path_independent_TY` have no directed T -> Y path, so tau
is identically 0 by construction; they are reported for completeness and to
confirm the generator is behaving.

Usage
-----
    export UWYK_SRC=$DEPLOY_ROOT/external/uwyk/src
    export UWYK_ROOT=$DEPLOY_ROOT/external/uwyk

    python UWYK_Fig3_4/cate_ate_distributions.py \
        --prior lingaus complexmech --nodes 2 5 20 35 50 --n-realizations 100

Reads saved realizations from --data-dir when they exist, otherwise samples them
in memory (nothing is written to the data dir). Emits, into --out-dir:
    cate_ate_stats.json    full per-cell statistics
    cate_ate_summary.md    markdown table
    cate_<prior>.png       pooled CATE histograms   (grid: n_nodes x regime)
    ate_<prior>.png        per-dataset ATE histograms
"""
from __future__ import annotations

# macOS: torch and XGBoost each ship their own OpenMP runtime, and having both
# live in one process segfaults inside XGBoost-backed mechanisms (a hard heap
# corruption -- faulthandler cannot even finish printing). Pinning OpenMP to one
# thread before torch is imported avoids it. Reproduced on the ComplexMech prior
# at 20+ nodes; harmless for the LinGaus prior, which uses no XGBoost.
# Override by exporting OMP_NUM_THREADS yourself.
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pehe_metrics import REGIMES, load_cell  # noqa: E402

_QUANTILES = (1, 5, 25, 50, 75, 95, 99)


def _describe(x: np.ndarray, prefix: str = "") -> dict:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    out = {
        f"{prefix}n": int(x.size),
        f"{prefix}mean": float(x.mean()),
        f"{prefix}sd": float(x.std()),
        f"{prefix}min": float(x.min()),
        f"{prefix}max": float(x.max()),
        f"{prefix}mean_abs": float(np.abs(x).mean()),
    }
    for q in _QUANTILES:
        out[f"{prefix}q{q:02d}"] = float(np.percentile(x, q))
    return out


def cell_stats(records: list[dict]) -> dict:
    """CATE / ATE distribution summary for one cell."""
    cates = [np.asarray(r["true_cate"], dtype=np.float64).reshape(-1) for r in records]
    pooled = np.concatenate(cates)
    ates = np.array([c.mean() for c in cates])
    within_sd = np.array([c.std() for c in cates])

    stats = {
        "n_realizations": len(records),
        "n_units_total": int(pooled.size),
        # exact zero matters: the null-effect regimes must be bitwise 0
        "frac_cate_exactly_zero": float(np.mean(pooled == 0.0)),
        "n_descendant_free": sum(1 for r in records
                                 if int(r["n_descendant_features"]) == 0),
        "mean_descendant_features": float(np.mean(
            [int(r["n_descendant_features"]) for r in records])),
        "mean_real_features": float(np.mean(
            [int(r["n_real_features"]) for r in records])),
    }
    stats.update(_describe(pooled, "cate_"))
    stats.update(_describe(ates, "ate_"))
    # Oracle sqrt(PEHE) floor and the homogeneity verdict.
    stats["oracle_pehe_floor_mean"] = float(within_sd.mean())
    stats["oracle_pehe_floor_max"] = float(within_sd.max())
    denom = float(np.abs(ates).mean())
    stats["heterogeneity_ratio"] = (
        float(within_sd.mean() / denom) if denom > 0 else 0.0
    )
    stats["_cate_pooled"] = pooled
    stats["_ates"] = ates
    return stats


def collect(prior: str, n_nodes: int, regime: str, args) -> list[dict] | None:
    """Saved realizations if present, else sample them in memory."""
    hide = 0.0  # hide never affects the SCM; see module docstring
    try:
        return load_cell(args.data_dir, prior, n_nodes, regime, hide)
    except FileNotFoundError:
        pass

    # Imported lazily: sampling needs torch + UWYK, reading saved npz does not.
    try:
        from generate_pehe_benchmark import (
            _import_uwyk, config_path, generate_realization, load_config,
        )
    except ImportError as exc:
        print(f"[skip] {prior} n={n_nodes} {regime}: no saved realizations under "
              f"{args.data_dir}, and cannot sample them here ({exc}). Either run "
              f"generate_pehe_benchmark.py first, or run this on a machine with "
              f"torch + UWYK available.", flush=True)
        return None

    cfg_file = config_path(prior, n_nodes, regime, hide)
    if not os.path.exists(cfg_file):
        print(f"[skip] no config: {cfg_file}", flush=True)
        return None
    cfg = load_config(cfg_file)
    uwyk = _import_uwyk()
    sampler = uwyk[0](cfg["scm_config"], seed=args.seed_base * 31 + 17)

    recs = []
    for r in range(args.n_realizations):
        try:
            recs.append(generate_realization(
                cfg, sampler, uwyk, regime,
                seed=args.seed_base + 1_000_000 * r,
                hide_fraction=hide,
                test_feature_mask_fraction=args.test_feature_mask_fraction,
            ))
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {prior} {regime} n={n_nodes} r={r}: {exc}", flush=True)
    return recs or None


# ── plots ─────────────────────────────────────────────────────────────────────

def _plot_grid(results: dict, prior: str, key: str, title: str, out_path: str,
               nodes: list[int], regimes: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nr, nc = len(nodes), len(regimes)
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 2.7 * nr), squeeze=False)
    for i, n in enumerate(nodes):
        for j, reg in enumerate(regimes):
            ax = axes[i][j]
            st = results.get((prior, n, reg))
            if st is None:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="0.5")
                continue
            v = st[key]
            if np.allclose(v, 0.0):
                ax.axvline(0.0, color="crimson", lw=2)
                ax.set_xlim(-1, 1)
                ax.text(0.5, 0.75, "point mass at 0", ha="center",
                        transform=ax.transAxes, fontsize=8, color="crimson")
            else:
                ax.hist(v, bins=60, color="#3b6ea5", edgecolor="none")
                ax.axvline(float(np.mean(v)), color="crimson", lw=1,
                           label=f"mean {np.mean(v):.3g}")
                ax.legend(fontsize=7, frameon=False)
            if i == 0:
                ax.set_title(reg, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{n} nodes", fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot] {out_path}", flush=True)


# ── report ────────────────────────────────────────────────────────────────────

_COLS = [
    ("prior", "{}"), ("n_nodes", "{}"), ("regime", "{}"),
    ("n_realizations", "{}"),
    ("cate_mean", "{:.4g}"), ("cate_sd", "{:.4g}"),
    ("cate_q05", "{:.4g}"), ("cate_q50", "{:.4g}"), ("cate_q95", "{:.4g}"),
    ("ate_mean", "{:.4g}"), ("ate_sd", "{:.4g}"),
    ("ate_q05", "{:.4g}"), ("ate_q95", "{:.4g}"),
    ("oracle_pehe_floor_mean", "{:.4g}"),
    ("heterogeneity_ratio", "{:.3g}"),
]


def write_markdown(rows: list[dict], path: str) -> None:
    head = "| " + " | ".join(c for c, _ in _COLS) + " |"
    rule = "|" + "|".join("---" for _ in _COLS) + "|"
    lines = [
        "# CATE / ATE distributions — UWYK Fig-3 / Fig-4 priors",
        "",
        "`oracle_pehe_floor_mean` is the mean WITHIN-dataset sd of the true ITE.",
        "That is the sqrt(PEHE) of a constant predictor -- NOT the best any model",
        "can reach; the true floor is sd(tau - E[tau|X]), which is <= it. The gap",
        "between the two is the heterogeneity actually learnable from X, measured",
        "separately by learnable_heterogeneity.py.",
        "",
        "`heterogeneity_ratio` is that constant-predictor sd over mean |ATE|. Near",
        "0 means the prior generates an effectively constant treatment effect, so",
        "PEHE there measures only ATE accuracy. Watch for cate_sd == ate_sd: that",
        "is the signature of zero within-dataset heterogeneity.",
        "",
        head, rule,
    ]
    for r in rows:
        lines.append("| " + " | ".join(
            fmt.format(r[c]) if isinstance(r[c], float) else str(r[c])
            for c, fmt in _COLS) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[table] {path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prior", nargs="+", default=["lingaus", "complexmech"],
                   choices=("lingaus", "complexmech"))
    p.add_argument("--nodes", type=int, nargs="+", default=[2, 5, 20, 35, 50])
    p.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=REGIMES)
    p.add_argument("--n-realizations", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--test-feature-mask-fraction", type=float, default=0.0)
    p.add_argument("--data-dir", default=os.path.join(_HERE, "data"))
    p.add_argument("--out-dir", default=os.path.join(_HERE, "distributions"))
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results: dict = {}
    rows: list[dict] = []

    for prior in args.prior:
        for n in args.nodes:
            for reg in args.regimes:
                recs = collect(prior, n, reg, args)
                if not recs:
                    continue
                st = cell_stats(recs)
                results[(prior, n, reg)] = st
                row = {"prior": prior, "n_nodes": n, "regime": reg}
                row.update({k: v for k, v in st.items() if not k.startswith("_")})
                rows.append(row)
                print(f"[cell] {prior} n={n} {reg}: "
                      f"ATE mean={st['ate_mean']:.4g} sd={st['ate_sd']:.4g}  "
                      f"CATE sd={st['cate_sd']:.4g}  "
                      f"oracle floor={st['oracle_pehe_floor_mean']:.4g}  "
                      f"het ratio={st['heterogeneity_ratio']:.3g}", flush=True)

    if not rows:
        raise SystemExit("no cells produced data — check UWYK_SRC / UWYK_ROOT")

    json_path = os.path.join(args.out_dir, "cate_ate_stats.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[json] {json_path}", flush=True)
    write_markdown(rows, os.path.join(args.out_dir, "cate_ate_summary.md"))

    if not args.no_plots:
        for prior in args.prior:
            present = [n for n in args.nodes
                       if any((prior, n, r) in results for r in args.regimes)]
            if not present:
                continue
            _plot_grid(results, prior, "_cate_pooled",
                       f"True CATE (pooled over units) — {prior}",
                       os.path.join(args.out_dir, f"cate_{prior}.png"),
                       present, args.regimes)
            _plot_grid(results, prior, "_ates",
                       f"ATE per dataset — {prior}",
                       os.path.join(args.out_dir, f"ate_{prior}.png"),
                       present, args.regimes)


if __name__ == "__main__":
    main()
