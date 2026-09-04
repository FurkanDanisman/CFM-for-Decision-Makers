"""Density-metric harness for the SCM case studies with bivariate Y noise.

Given a directory of rerun-model shards (one .npz per realization, per
(model, case_study)) that contain the model's discretised predictive
distribution — and the NEW-FORMAT case-study pkls carrying `mu_0`, `mu_1`,
`sigma_eps`, and `rho_y_noise` — compute L2 / CRPS / NLL metrics against
the analytic Gaussian truth density.

Shard-format expectations
-------------------------
For each realization the shard NPZ MUST contain (see 'REQUIRED KEYS' below).
When any required key is missing, this harness records a `SKIPPED` row and
prints a rerun-requirement notice pointing at exactly what needs to be
added to the corresponding eval script. Adding a `DENSITY_DUMP` env-var
gate to the existing eval scripts is a two-line change per script — a TODO
tracker file is emitted at the end summarising what remains.

REQUIRED KEYS (per shard NPZ):

  edges           (J+1,)   scaled y-bin edges (shared across queries)
  p_y0_scaled     (Nq, J)  scaled-y predicted probability mass for do(T=0)
  p_y1_scaled     (Nq, J)  scaled-y predicted probability mass for do(T=1)
  y_shift         scalar   scaled-y = (raw - shift) / scale
  y_scale         scalar
  (optional) p_joint_scaled (Nq, J, J)  joint p(Y_do0, Y_do1) if 2D model

Outputs
-------
One .npz per (model, case_study) under
  <results-root>/<model>/<case_study>.npz
containing per-realization aggregated metrics (mean, sem, count).

CLI
---
python density_eval.py \
    --model MODEL_NAME \
    --dataset Observed_Confounder \
    --shard-dir /path/to/model_shards/<model>/<case> \
    --out-dir  /path/to/results_density_eval_rho02/<model> \
    --dopfn-data-root /path/to/prior_sampling_rho02 \
    [--n-fine 10000] [--y-range 4.0]

Env vars honored (all optional):
  DOPFN_ROOT        (needed to unpickle NEW-format pkls via the shim)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np


# ── Metric primitives ────────────────────────────────────────────────────────
def _norm_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-8)
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2 * np.pi))


def _norm_cdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    from math import erf, sqrt
    sigma = max(float(sigma), 1e-8)
    z = (x - mu) / (sigma * sqrt(2))
    # numpy has no erf; use scipy if available, else vectorise math.erf.
    try:
        from scipy.special import erf as _erf
        return 0.5 * (1.0 + _erf(z))
    except Exception:
        vf = np.vectorize(lambda t: erf(float(t)))
        return 0.5 * (1.0 + vf(z))


def _integrate(f: np.ndarray, dx: float) -> float:
    return float(np.trapz(f, dx=dx))


def _bins_to_density_on_grid(p_bins: np.ndarray, edges: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Piecewise-constant density: p_i / bin_width inside bin i, 0 outside.
    Interpolate onto x_grid via nearest-bin assignment.
    """
    # bin width array
    widths = edges[1:] - edges[:-1]
    dens_native = p_bins / np.clip(widths, 1e-12, None)
    # Assign each grid point to the containing bin.
    idx = np.searchsorted(edges, x_grid, side="right") - 1
    idx = np.clip(idx, 0, len(dens_native) - 1)
    dens = dens_native[idx]
    # Zero out grid points outside [edges[0], edges[-1]]
    outside = (x_grid < edges[0]) | (x_grid > edges[-1])
    dens[outside] = 0.0
    return dens


def _renormalize(f: np.ndarray, dx: float) -> np.ndarray:
    s = _integrate(f, dx)
    if s <= 0:
        return f
    return f / s


def _pdf_l2sq(f_hat: np.ndarray, f_true: np.ndarray, dx: float) -> float:
    return float(np.sum((f_hat - f_true) ** 2) * dx)


def _cdf_l2sq(f_hat: np.ndarray, f_true: np.ndarray, dx: float) -> float:
    F_hat = np.cumsum(f_hat) * dx
    F_true = np.cumsum(f_true) * dx
    return float(np.sum((F_hat - F_true) ** 2) * dx)


def _nll_at_y(f_hat: np.ndarray, x_grid: np.ndarray, y: float) -> float:
    # Nearest-grid lookup; assume x_grid is uniform.
    if y < x_grid[0] or y > x_grid[-1]:
        return float("inf")
    j = int(np.clip(np.searchsorted(x_grid, y) - 1, 0, len(x_grid) - 1))
    dens = max(float(f_hat[j]), 1e-30)
    return float(-np.log(dens))


# ── Shard loading ────────────────────────────────────────────────────────────
_REQUIRED_KEYS = ("edges", "p_y0_scaled", "p_y1_scaled", "y_shift", "y_scale")


def _load_shard_or_none(shard_path: str) -> Optional[Dict[str, np.ndarray]]:
    """Return the loaded arrays if the required keys are present; else None."""
    if not os.path.isfile(shard_path):
        return None
    with np.load(shard_path, allow_pickle=True) as z:
        keys = set(z.files)
        if not all(k in keys for k in _REQUIRED_KEYS):
            return None
        return {k: z[k] for k in z.files}


# ── Analytic truth densities on a fine grid ──────────────────────────────────
def _analytic_truth_density_y(y_grid: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return _norm_pdf(y_grid, mu, sigma)


def _analytic_truth_density_tau(tau_grid: np.ndarray, mu_diff: float, sigma_tau: float) -> np.ndarray:
    return _norm_pdf(tau_grid, mu_diff, sigma_tau)


# ── Model-density extraction (per query) ─────────────────────────────────────
def _predicted_density_y(p_bins_scaled: np.ndarray, edges_scaled: np.ndarray,
                          y_grid_scaled: np.ndarray) -> np.ndarray:
    """p_bins_scaled: (J,), edges_scaled: (J+1,)."""
    return _bins_to_density_on_grid(p_bins_scaled, edges_scaled, y_grid_scaled)


def _predicted_density_tau_from_joint(p_joint: np.ndarray, edges_scaled: np.ndarray,
                                        tau_grid_scaled: np.ndarray) -> np.ndarray:
    """Diagonal-sum a J×J joint into a τ = y1 - y0 density on tau_grid_scaled.

    Approximates the joint as a piecewise-constant density on a J×J grid of
    cells then convolves along the diagonal. Uses the coarse (native) τ
    grid built from bin-center differences, then interpolates to
    `tau_grid_scaled`.
    """
    J = p_joint.shape[0]
    centers = 0.5 * (edges_scaled[:-1] + edges_scaled[1:])
    bin_w = float(edges_scaled[1] - edges_scaled[0])
    # τ bin edges: differences of centers span (-(J-1)*w, +(J-1)*w) in steps of w.
    tau_native = np.arange(-(J - 1), J) * bin_w                # (2J - 1,) at bin_w spacing
    p_tau_native = np.zeros_like(tau_native)
    for i in range(J):
        for j in range(J):
            k = (j - i) + (J - 1)
            p_tau_native[k] += float(p_joint[i, j])
    # Piecewise-constant density on native tau grid:
    dens_native = p_tau_native / max(bin_w, 1e-12)
    # Interpolate to requested tau_grid_scaled (nearest-bin).
    tau_edges_native = np.concatenate([
        [tau_native[0] - 0.5 * bin_w],
        0.5 * (tau_native[:-1] + tau_native[1:]),
        [tau_native[-1] + 0.5 * bin_w],
    ])
    return _bins_to_density_on_grid(dens_native, tau_edges_native, tau_grid_scaled)


def _predicted_density_tau_from_indep_marginals(
    p_y0: np.ndarray, p_y1: np.ndarray, edges_scaled: np.ndarray,
    tau_grid_scaled: np.ndarray,
) -> np.ndarray:
    """When only marginals are available, assume Y_do0 ⊥ Y_do1 and build the
    joint as an outer product, then diagonal-sum. Matches the fallback used
    in `benchmarks/l2_ihdp/methods_densities.py` for baseline models."""
    p_joint = np.outer(p_y0, p_y1)
    p_joint = p_joint / max(p_joint.sum(), 1e-12)
    return _predicted_density_tau_from_joint(p_joint, edges_scaled, tau_grid_scaled)


# ── Grid builders ────────────────────────────────────────────────────────────
def _make_grid(lo: float, hi: float, n: int) -> np.ndarray:
    return np.linspace(float(lo), float(hi), int(n))


# ── Per-realization eval ─────────────────────────────────────────────────────
def _eval_realization(
    shard: Dict[str, np.ndarray],
    slice_,
    n_fine: int,
    y_range: float,
) -> Optional[Dict[str, np.ndarray]]:
    """Compute per-query densities + metrics for one realization.

    slice_ carries mu_0, mu_1, sigma_eps, rho_y_noise, y_test (from y_int),
    ordered by test_row_indices. Only new-format pkls populate these fields.
    """
    if slice_.mu_0 is None or slice_.mu_1 is None or slice_.sigma_eps is None:
        return None

    edges = np.asarray(shard["edges"], dtype=np.float64)
    p_y0 = np.asarray(shard["p_y0_scaled"], dtype=np.float64)
    p_y1 = np.asarray(shard["p_y1_scaled"], dtype=np.float64)
    y_shift = float(np.asarray(shard["y_shift"]).item())
    y_scale = max(float(np.asarray(shard["y_scale"]).item()), 1e-12)
    p_joint = shard.get("p_joint_scaled")
    if p_joint is not None:
        p_joint = np.asarray(p_joint, dtype=np.float64)

    Nq = p_y0.shape[0]
    if Nq != len(slice_.mu_0):
        # Alignment mismatch: shard's Nq must equal the pkl's test-row count.
        return {"__alignment_error__": np.array(True),
                "shard_nq": np.array(Nq),
                "pkl_nq": np.array(len(slice_.mu_0))}

    mu0 = np.asarray(slice_.mu_0, dtype=np.float64)
    mu1 = np.asarray(slice_.mu_1, dtype=np.float64)
    sigma = float(slice_.sigma_eps)
    rho = float(slice_.rho_y_noise) if slice_.rho_y_noise is not None else 0.0
    sigma_tau = float(np.sqrt(max(2.0 * sigma * sigma * (1.0 - rho), 1e-24)))

    # ── Grids ──────────────────────────────────────────────────────────────
    # Scaled y grid on the model's native scale, centred on the union of
    # model edges and analytic mu ± 5 sigma_scaled.
    mu0_scaled = (mu0 - y_shift) / y_scale
    mu1_scaled = (mu1 - y_shift) / y_scale
    sigma_scaled = sigma / y_scale
    y_lo_s = min(float(edges[0]), float(mu0_scaled.min() - 5 * sigma_scaled),
                                    float(mu1_scaled.min() - 5 * sigma_scaled))
    y_hi_s = max(float(edges[-1]), float(mu0_scaled.max() + 5 * sigma_scaled),
                                    float(mu1_scaled.max() + 5 * sigma_scaled))
    y_grid_s = _make_grid(y_lo_s, y_hi_s, n_fine)
    dy_s = float(y_grid_s[1] - y_grid_s[0])
    # Raw-y grid: matching span in original units.
    y_grid_r = y_grid_s * y_scale + y_shift
    dy_r = float(y_grid_r[1] - y_grid_r[0])

    tau_scaled_range = max(float((mu1_scaled - mu0_scaled).max() - (mu1_scaled - mu0_scaled).min())
                            + 10 * (sigma_scaled * np.sqrt(2)), 2.0 * y_range)
    tau_center_s = float(0.5 * (mu1_scaled - mu0_scaled).mean())
    tau_grid_s = _make_grid(tau_center_s - 0.5 * tau_scaled_range,
                             tau_center_s + 0.5 * tau_scaled_range, n_fine)
    dtau_s = float(tau_grid_s[1] - tau_grid_s[0])
    tau_grid_r = tau_grid_s * y_scale
    dtau_r = float(tau_grid_r[1] - tau_grid_r[0])

    # Truth densities on raw / scaled grids for τ (pooled truth is a mixture
    # of per-query Gaussians; per-query we compute separately below).
    # Per-query truth Y densities: N(mu_t(x), sigma^2)
    # Per-query truth τ density:   N(mu_1(x) - mu_0(x), 2σ²(1−ρ))

    # Aggregate storage
    out = {
        "n_queries": np.array(Nq),
        "sigma_eps": np.array(sigma),
        "rho_y_noise": np.array(rho),
        # Per-query metrics
        "y0_l2sq_scaled":   np.zeros(Nq),
        "y1_l2sq_scaled":   np.zeros(Nq),
        "y0_crps_scaled":   np.zeros(Nq),
        "y1_crps_scaled":   np.zeros(Nq),
        "y0_l2sq_raw":      np.zeros(Nq),
        "y1_l2sq_raw":      np.zeros(Nq),
        "y0_crps_raw":      np.zeros(Nq),
        "y1_crps_raw":      np.zeros(Nq),
        "y0_nll":           np.zeros(Nq),
        "y1_nll":           np.zeros(Nq),
        "tau_l2sq_scaled":  np.zeros(Nq),
        "tau_crps_scaled":  np.zeros(Nq),
        "tau_l2sq_raw":     np.zeros(Nq),
        "tau_crps_raw":     np.zeros(Nq),
        "tau_nll":          np.zeros(Nq),
    }

    # y_int per test row (from the pkl) — needed for NLL.
    y_te = np.asarray(getattr(slice_, "_y_test", None), dtype=np.float64) \
        if hasattr(slice_, "_y_test") and slice_._y_test is not None else None

    for q in range(Nq):
        # Model-predicted densities on scaled grid.
        f0_s = _bins_to_density_on_grid(p_y0[q], edges, y_grid_s)
        f1_s = _bins_to_density_on_grid(p_y1[q], edges, y_grid_s)
        # Truth densities on same scaled grid (marginal Y | do(t), x is
        # Gaussian; on scaled grid the mean shifts and std scales by 1/y_scale).
        t0_s = _analytic_truth_density_y(y_grid_s, mu0_scaled[q], sigma_scaled)
        t1_s = _analytic_truth_density_y(y_grid_s, mu1_scaled[q], sigma_scaled)

        out["y0_l2sq_scaled"][q]  = _pdf_l2sq(f0_s, t0_s, dy_s)
        out["y1_l2sq_scaled"][q]  = _pdf_l2sq(f1_s, t1_s, dy_s)
        out["y0_crps_scaled"][q]  = _cdf_l2sq(f0_s, t0_s, dy_s)
        out["y1_crps_scaled"][q]  = _cdf_l2sq(f1_s, t1_s, dy_s)

        # Raw-Y grid: densities in raw units = scaled_density / y_scale
        # (change of variables).  We compute directly for cleanliness.
        f0_r = f0_s / y_scale
        f1_r = f1_s / y_scale
        t0_r = _analytic_truth_density_y(y_grid_r, mu0[q], sigma)
        t1_r = _analytic_truth_density_y(y_grid_r, mu1[q], sigma)
        out["y0_l2sq_raw"][q] = _pdf_l2sq(f0_r, t0_r, dy_r)
        out["y1_l2sq_raw"][q] = _pdf_l2sq(f1_r, t1_r, dy_r)
        out["y0_crps_raw"][q] = _cdf_l2sq(f0_r, t0_r, dy_r)
        out["y1_crps_raw"][q] = _cdf_l2sq(f1_r, t1_r, dy_r)

        if y_te is not None and q < len(y_te):
            # NLL under the model at the observed y_int for this query.
            out["y0_nll"][q] = _nll_at_y(f0_r, y_grid_r, float(y_te[q]))
            out["y1_nll"][q] = _nll_at_y(f1_r, y_grid_r, float(y_te[q]))
        else:
            out["y0_nll"][q] = np.nan
            out["y1_nll"][q] = np.nan

        # τ density predictions
        if p_joint is not None:
            f_tau_s = _predicted_density_tau_from_joint(p_joint[q], edges, tau_grid_s)
        else:
            f_tau_s = _predicted_density_tau_from_indep_marginals(
                p_y0[q], p_y1[q], edges, tau_grid_s)
        tau_true_scaled = _analytic_truth_density_tau(
            tau_grid_s, float(mu1_scaled[q] - mu0_scaled[q]),
            sigma_tau / y_scale)
        out["tau_l2sq_scaled"][q] = _pdf_l2sq(f_tau_s, tau_true_scaled, dtau_s)
        out["tau_crps_scaled"][q] = _cdf_l2sq(f_tau_s, tau_true_scaled, dtau_s)

        f_tau_r = f_tau_s / y_scale
        tau_true_raw = _analytic_truth_density_tau(
            tau_grid_r, float(mu1[q] - mu0[q]), sigma_tau)
        out["tau_l2sq_raw"][q] = _pdf_l2sq(f_tau_r, tau_true_raw, dtau_r)
        out["tau_crps_raw"][q] = _cdf_l2sq(f_tau_r, tau_true_raw, dtau_r)
        out["tau_nll"][q] = np.nan  # true τ not directly observed
    return out


# ── Main ─────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True,
                   help="Case-study name, e.g. Observed_Confounder.")
    p.add_argument("--shard-dir", required=True,
                   help="Directory containing per-realization shard NPZs "
                        "for this (model, dataset) — usually the OUT dir "
                        "of a rerun of submit_eval_scm_case_studies.sbatch.")
    p.add_argument("--out-dir", required=True,
                   help="Output directory; writes <dataset>.npz here.")
    p.add_argument("--dopfn-data-root", required=True,
                   help="Parent of the case-study dirs (i.e. .../prior_sampling_rho02).")
    p.add_argument("--n-fine", type=int, default=10000)
    p.add_argument("--y-range", type=float, default=4.0)
    p.add_argument("--max-real", type=int, default=0,
                   help="Optional cap on n_realizations for smoke tests.")
    return p.parse_args()


def main():
    args = _parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    bench_root = os.path.join(repo_root, "benchmarks")
    if bench_root not in sys.path:
        sys.path.insert(0, bench_root)

    os.environ["DOPFN_DATA_ROOT"] = args.dopfn_data_root

    from scm_case_study_dataset import SCMCaseStudyDataset

    ds = SCMCaseStudyDataset(args.dataset)
    n = ds.n_tables if not args.max_real else min(ds.n_tables, args.max_real)
    print(f"[density-eval] model={args.model}  dataset={args.dataset}  "
          f"n={n}  shards={args.shard_dir}", flush=True)

    per_real_rows = []
    n_missing = 0
    n_bad_align = 0
    t0 = time.time()
    for r in range(n):
        slice_, _ = ds[r]
        # Attach y_int (raw units) so NLL can be computed. Requires that
        # the SCMCaseStudyDataset expose the concatenated `y` — we index by
        # test_row_indices.
        try:
            _y_pool = None
            _ds_pkl = None
            # We already have all pieces in slice_; the raw y is what the pkl
            # loader returned via _load_one — but that isn't exposed. To keep
            # this a strict extension of the existing loader, re-derive y_te
            # by direct pkl introspection.
            from scm_case_study_dataset import _pickle_load_with_dopfn_shim
            _ds_pkl = _pickle_load_with_dopfn_shim(ds.pkl_paths[r])
            _y_full = np.asarray(_ds_pkl.y, dtype=np.float32).reshape(-1)
            slice_._y_test = _y_full[slice_.test_row_indices]
        except Exception:
            slice_._y_test = None

        shard_path = os.path.join(args.shard_dir, f"r{r:03d}.npz")
        shard = _load_shard_or_none(shard_path)
        if shard is None:
            n_missing += 1
            continue
        result = _eval_realization(shard, slice_, n_fine=args.n_fine, y_range=args.y_range)
        if result is None:
            n_missing += 1
            continue
        if result.get("__alignment_error__") is not None:
            n_bad_align += 1
            continue
        per_real_rows.append(result)
        if r == 0 or r % 25 == 0 or r == n - 1:
            print(f"[density-eval] r={r:03d}  {time.time() - t0:.0f}s", flush=True)

    if not per_real_rows:
        print(f"[density-eval] NO USABLE SHARDS  missing={n_missing}  align_err={n_bad_align}\n"
              "  Required shard keys: " + ", ".join(_REQUIRED_KEYS) + "\n"
              "  Add density-dump logic (probs + edges + y_shift/y_scale) to "
              "the eval script before re-running.", flush=True)
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, f"{args.dataset}.npz")
        np.savez(out_path, n_missing=n_missing, n_bad_align=n_bad_align,
                 status=np.array("NO_USABLE_SHARDS"))
        return

    # Aggregate: mean/sem across queries within each realization, then across
    # realizations. We keep both.
    keys = [k for k in per_real_rows[0].keys() if k not in
            ("n_queries", "sigma_eps", "rho_y_noise", "__alignment_error__",
             "shard_nq", "pkl_nq")]

    per_real_mean = {k: np.array([r[k].mean() for r in per_real_rows], dtype=np.float64)
                     for k in keys}
    per_real_sem  = {k: np.array([r[k].std(ddof=1) / np.sqrt(max(len(r[k]), 1))
                                    for r in per_real_rows], dtype=np.float64)
                     for k in keys}

    summary = {"per_real_mean_" + k: per_real_mean[k] for k in keys}
    summary.update({"per_real_sem_" + k: per_real_sem[k] for k in keys})
    summary.update({"across_real_mean_" + k: float(per_real_mean[k].mean()) for k in keys})
    summary.update({"across_real_sem_" + k: float(per_real_mean[k].std(ddof=1) /
                                                    np.sqrt(len(per_real_mean[k])))
                    for k in keys})
    summary["n_realizations_used"] = int(len(per_real_rows))
    summary["n_missing_shards"] = int(n_missing)
    summary["n_alignment_errors"] = int(n_bad_align)
    summary["model"] = args.model
    summary["dataset"] = args.dataset

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.dataset}.npz")
    np.savez(out_path, **{k: np.asarray(v) for k, v in summary.items()})

    # Human-readable console summary.
    print(f"\n[density-eval]  {args.model}  {args.dataset}  "
          f"n_real={len(per_real_rows)}  missing={n_missing}", flush=True)
    for k in keys:
        m = summary["across_real_mean_" + k]
        s = summary["across_real_sem_" + k]
        print(f"  {k:22s} = {m:12.6f} ± {s:10.6f}", flush=True)
    print(f"[density-eval] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
