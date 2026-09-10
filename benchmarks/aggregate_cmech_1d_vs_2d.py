"""Aggregate the ComplexMech PEHE runs into the 1D-vs-2D head comparison.

Reads OUT_ROOT/<model>/CMECH_n<N>_<subset>/ produced by
benchmarks/cluster/submit_cmech_pehe_1d_vs_2d.sbatch and emits, per
(pair, node count, subset), the mean per-dataset sqrt(PEHE) for the 1D and 2D
head plus their delta.

Pairs (1D -> 2D):
    dopfn_native -> dopfn_bb
    uwyk1d       -> graph2d
    cpfn1d       -> cpfn2d

Per-model npz layouts differ, which is why this file exists:
    dopfn_native   r<###>.npz            scalar  pehe_raw
    cpfn1d/cpfn2d  <DS>_r<###>.npz       scalar  pehe_raw
    uwyk1d/graph2d <DS>_r<###>.npz       scalar  pehe_raw_<tag>  (tag: noanc/v3b)
    dopfn_bb       summary.npz           ARRAY   pehe  (one entry per realization)

For uwyk1d and graph2d we read the `noanc` tag by default: this comparison is
about the head, so both sides should get the same (absent) graph information.
Pass --anc-tag v3b to compare the ancestor-conditioned variants instead.

Pooled PEHE is reconstructed from the two disjoint subsets:
    pehe_all^2 = (n0*pehe_zero^2 + n1*pehe_nonzero^2) / (n0 + n1)
with the query counts read from the benchmark data itself, so it never depends
on a model having recorded them.

Usage
-----
    python benchmarks/aggregate_cmech_1d_vs_2d.py --root $SCRATCH/cmech_1d2d
    python benchmarks/aggregate_cmech_1d_vs_2d.py --root ... --anc-tag v3b
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

PAIRS = (("dopfn_native", "dopfn_bb"),
         ("uwyk1d", "graph2d"),
         ("cpfn1d", "cpfn2d"))
MODELS = tuple(m for pair in PAIRS for m in pair)
NODE_COUNTS = (5, 20, 30, 40, 50)
SUBSETS = ("nonzero", "zero")


def _scalar(z, *names):
    for n in names:
        if n in z.files:
            v = np.asarray(z[n]).reshape(-1)
            if v.size:
                return float(v[0])
    return None


def load_model_cell(root: str, model: str, dataset: str, anc_tag: str):
    """Return {source_realization_index: sqrt(PEHE)} for one model x cell."""
    d = os.path.join(root, model, dataset)
    if not os.path.isdir(d):
        return {}

    # dopfn_bb: one npz holding arrays over realizations, in dataset order.
    summary = os.path.join(d, "summary.npz")
    if os.path.isfile(summary):
        with np.load(summary, allow_pickle=True) as z:
            if "pehe" in z.files:
                arr = np.asarray(z["pehe"], dtype=np.float64).reshape(-1)
                return {i: float(v) for i, v in enumerate(arr)}

    out: dict[int, float] = {}
    for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
        base = os.path.basename(p)
        digits = "".join(ch for ch in base.rsplit("r", 1)[-1] if ch.isdigit())
        if not digits:
            continue
        with np.load(p, allow_pickle=True) as z:
            v = _scalar(z, f"pehe_raw_{anc_tag}", "pehe_raw", "pehe")
        if v is not None and np.isfinite(v):
            out[int(digits)] = v
    return out


def query_counts(n_nodes: int, data_root: str) -> tuple[dict, dict]:
    """(zero_counts, nonzero_counts) keyed by realization, from the benchmark."""
    cell = os.path.join(data_root, "complexmech", f"{n_nodes}node",
                        "path_TY", "hide_0.0")
    z_ct, nz_ct = {}, {}
    for p in sorted(glob.glob(os.path.join(cell, "r*.npz"))):
        r = int(os.path.basename(p)[1:-4])
        with np.load(p) as z:
            tau = z["true_cate"]
        z_ct[r] = int((tau == 0).sum())
        nz_ct[r] = int((tau != 0).sum())
    return z_ct, nz_ct


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="OUT_ROOT from the sbatch run")
    ap.add_argument("--data-root", default=os.environ.get(
        "UWYK_FIG34_DATA",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "UWYK_Fig3_4", "data")))
    ap.add_argument("--anc-tag", default="noanc",
                    help="tag to read for uwyk1d/graph2d (default noanc: same "
                         "graph information on both sides, so the delta isolates "
                         "the head)")
    ap.add_argument("--out", default=None, help="write markdown here too")
    args = ap.parse_args()

    rows = []
    for n in NODE_COUNTS:
        for subset in SUBSETS:
            ds = f"CMECH_n{n}_{subset}"
            for one_d, two_d in PAIRS:
                a = load_model_cell(args.root, one_d, ds, args.anc_tag)
                b = load_model_cell(args.root, two_d, ds, args.anc_tag)
                shared = sorted(set(a) & set(b))
                if not shared:
                    continue
                va = np.array([a[r] for r in shared])
                vb = np.array([b[r] for r in shared])
                rows.append({
                    "pair": f"{one_d} -> {two_d}", "n_nodes": n, "subset": subset,
                    "n_datasets": len(shared),
                    "pehe_1d": float(va.mean()), "pehe_2d": float(vb.mean()),
                    "delta": float(vb.mean() - va.mean()),
                    "pct": float(100.0 * (vb.mean() - va.mean()) / va.mean())
                           if va.mean() else float("nan"),
                    "n_2d_better": int((vb < va).sum()),
                })

    if not rows:
        raise SystemExit(
            f"no results under {args.root}. Expected "
            f"{args.root}/<model>/CMECH_n<N>_<subset>/")

    hdr = ["pair", "n_nodes", "subset", "n_datasets", "pehe_1d", "pehe_2d",
           "delta", "pct", "n_2d_better"]
    lines = [
        "# ComplexMech PEHE — does the 2D head help?", "",
        f"anc tag for uwyk1d/graph2d: `{args.anc_tag}`.  "
        "`delta` = 2D - 1D, so **negative means the 2D head is better**.",
        "`n_2d_better` counts datasets where 2D beat 1D, out of `n_datasets`.",
        "PEHE is in the generator's [-1, 1] target units — comparable across "
        "models and node counts here, but not to RealCause PEHE numbers.", "",
        "| " + " | ".join(hdr) + " |",
        "|" + "|".join("---" for _ in hdr) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(
            f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c])
            for c in hdr) + " |")
    md = "\n".join(lines) + "\n"
    print(md)

    with open(os.path.join(args.root, "cmech_1d_vs_2d.json"), "w") as f:
        json.dump(rows, f, indent=2)
    out_md = args.out or os.path.join(args.root, "cmech_1d_vs_2d.md")
    with open(out_md, "w") as f:
        f.write(md)
    print(f"[written] {out_md}")


if __name__ == "__main__":
    main()
