#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DATASETS = ("IHDP", "ACIC", "CPS", "PSID", "PSID_bal")
DEFAULT_MODES = ("anc", "noanc")

# Which direction counts as better, per metric family in the npz files. Both
# metrics we can tabulate are errors, so smaller wins; `ate` is a point estimate
# with no direction and is deliberately absent.
BETTER = {"pehe": "lower", "err": "lower"}


def discover_modes(results_dir: Path, estimator: str) -> tuple[str, ...]:
    """Adjacency tags present in the npz files, in file order. Lets one command
    summarize an ANC_MODE=v6a_only run (v6a/noanc) or a default run (anc/noanc)
    without being told which."""
    prefix = f"pehe_{estimator}_"
    for dataset in DATASETS:
        for path in sorted((results_dir / dataset).glob(f"{dataset}_r*.npz")):
            with np.load(path) as result:
                tags = [k[len(prefix):] for k in result.files if k.startswith(prefix)]
            if tags:
                return tuple(tags)
    return DEFAULT_MODES


def latest_run(results_root: Path) -> Path:
    runs = [path for path in results_root.iterdir() if path.is_dir() and path.name.isdigit()]
    return max(runs, key=lambda path: int(path.name)) if runs else results_root


def summarize(results_dir: Path, dataset: str, key: str) -> tuple[float, float, int]:
    values = []
    for path in sorted((results_dir / dataset).glob(f"{dataset}_r*.npz")):
        with np.load(path) as result:
            if key in result:
                value = float(result[key])
                if np.isfinite(value):
                    values.append(value)

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.nan, np.nan, 0
    standard_error = array.std(ddof=1) / np.sqrt(array.size) if array.size > 1 else np.nan
    return float(array.mean()), float(standard_error), int(array.size)


def best_per_column(means: dict, modes: tuple[str, ...], better: str) -> dict[str, set]:
    """Modes that win each dataset column, so the table says which row is best
    without the reader eyeballing three decimals. Columns with fewer than two
    finite entries have nothing to compare, so they get no marker; exact ties
    keep every winner."""
    winners: dict[str, set] = {}
    for dataset in DATASETS:
        finite = {mode: means[mode, dataset] for mode in modes
                  if np.isfinite(means[mode, dataset])}
        if len(finite) < 2:
            continue
        best = (min if better == "lower" else max)(finite.values())
        winners[dataset] = {mode for mode, value in finite.items() if value == best}
    return winners


def format_cell(mean: float, standard_error: float) -> str:
    if not np.isfinite(mean):
        return "—"
    if not np.isfinite(standard_error):
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {standard_error:.3f}"


def main() -> None:
    repo = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=repo / "results_graph2d_realcause",
    )
    parser.add_argument("--estimator", choices=("raw", "em"), default="raw")
    parser.add_argument(
        "--modes",
        help="Comma-separated adjacency tags to report (e.g. 'v6a,noanc'). "
        "Default: whatever tags the npz files actually contain.",
    )
    parser.add_argument(
        "--row-prefix",
        default="PEHE",
        help="Row-label prefix, e.g. '--row-prefix \"UWYK PEHE\"' when stacking "
        "the uwyk1d control table next to the joint one.",
    )
    args = parser.parse_args()
    results_dir = latest_run(args.results)
    modes = (tuple(m.strip() for m in args.modes.split(",") if m.strip())
             if args.modes else discover_modes(results_dir, args.estimator))

    summaries = {}
    means = {}
    counts = {}
    for mode in modes:
        key = f"pehe_{args.estimator}_{mode}"
        for dataset in DATASETS:
            mean, standard_error, count = summarize(results_dir, dataset, key)
            summaries[mode, dataset] = format_cell(mean, standard_error)
            means[mode, dataset] = mean
            counts[dataset] = count

    winners = best_per_column(means, modes, BETTER["pehe"])

    print(f"Run: {results_dir.name}\n")
    print("| Metric | " + " | ".join(DATASETS) + " |")
    print("|---|" + "|".join("---:" for _ in DATASETS) + "|")
    for mode in modes:
        cells = [
            (f"**{summaries[mode, dataset]}**"
             if mode in winners.get(dataset, ())
             else summaries[mode, dataset])
            for dataset in DATASETS
        ]
        print(f"| {args.row_prefix} {mode} | " + " | ".join(cells) + " |")

    print("\nRealizations: " + ", ".join(f"{dataset}={counts[dataset]}" for dataset in DATASETS))


if __name__ == "__main__":
    main()

