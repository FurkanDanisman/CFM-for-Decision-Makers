"""Bucket borders for the 1-D head, grid edges for the 2-D head, and the
discretisation floor that says what the 2-D grid costs.

Both heads are fitted from the prior before step one (the released config
carries ``bar_dist_init_batches = 100``), in the **standardised** target space
the parity harness settled on.

The two heads cannot allocate resolution the same way, and this is forced rather
than chosen:

* The 1-D head is a ``FullSupportBarDistribution``, whose borders may be
  arbitrary. Do-PFN's released borders are plainly quantile-allocated -- widths
  run from 0.049 to 245.9 -- so fine where the mass is and very coarse in the
  tails. We reproduce that.
* ``neg_log_prob_2d`` carries a single scalar ``bin_width``, so the 2-D grid
  **must** be uniform. ``fit_edges_2d`` spans min-to-max, which on this prior
  means roughly [-17, 13] and a J=10 bin width near 3.0 -- wide enough that
  almost all mass lands in two bins. So we span a robust quantile range instead
  and let the 9-region tail structure absorb the rest.

``oracle_floor`` measures what remains: the error in tau = y1 - y0 from binning
alone, for an oracle that knows the true outcomes but may only express them on
the grid. If the trained head lands near its floor, resolution is the binding
constraint and the head is fine.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from training_dopfn_repro.batch import (  # noqa: E402
    BatchConfig,
    make_joint_batch,
    sample_single_eval_pos,
)
from training_dopfn_repro.prior import PriorConfig, sample_batch  # noqa: E402

# Fraction trimmed from each tail when spanning the 2-D grid. At J=10 this gives
# a bin width near 0.72 with ~98% of the mass inside, against ~3.0 for min/max.
DEFAULT_GRID_QUANTILES = (0.01, 0.99)

# Outer quantiles for the 1-D borders. Not 0%/100%: those are the sample min and
# max of a heavy-tailed target and swing by orders of magnitude between fits.
# See fit_bar_borders_1d for the measurements.
DEFAULT_BORDER_TRIM = (0.001, 0.999)


def collect_targets(
    n_batches: int = 100,
    prior_cfg: PriorConfig | None = None,
    batch_cfg: BatchConfig | None = None,
    seed_base: int = 900_000,
) -> dict[str, torch.Tensor]:
    """Draw prior batches and return the target values in the training y space."""
    prior_cfg = prior_cfg or PriorConfig()
    batch_cfg = batch_cfg or BatchConfig()

    ctx, arms = [], []
    for i in range(n_batches):
        rec = sample_batch(seed_base + i, prior_cfg)
        if rec["meta"]["nonfinite"]:
            continue
        sep = sample_single_eval_pos(seed_base + i, prior_cfg.seq_len, batch_cfg)
        b = make_joint_batch(rec, sep, batch_cfg)
        ctx.append(b["train_y"].reshape(-1))
        arms.append(torch.cat([b["y_do0"].reshape(-1), b["y_do1"].reshape(-1)]))

    if not ctx:
        raise RuntimeError("collect_targets: every sampled batch was non-finite")

    context = torch.cat(ctx)
    query = torch.cat(arms)
    return {
        "context": context[torch.isfinite(context)],
        "query": query[torch.isfinite(query)],
    }


def fit_bar_borders_1d(
    targets: torch.Tensor,
    num_buckets: int = 100,
    eps: float = 1e-6,
    trim: tuple[float, float] = DEFAULT_BORDER_TRIM,
) -> torch.Tensor:
    """Quantile-allocated borders, reproducing how the released head is spaced.

    ``trim`` is a deviation, and a necessary one. Spanning the full 0%-100%
    quantile range means the two outer borders *are* the sample min and max of a
    heavy-tailed target, which is violently unstable -- across five independent
    fits the span came out as [-9.8, 53.2], [-13.6, 12.4], [-12616.6, 52.3],
    [-39.1, 30.7], [-29.6, 96.5]. A single outlier can hand one bucket a width
    of 12,614, burning a large share of the head's capacity on a region holding
    one observation, and making two training runs incomparable for no reason.

    Trimming is safe precisely because this is a *FullSupportBarDistribution*:
    the outermost buckets are half-normal tails, so values beyond the outer
    borders keep positive density and the borders need only mark where the tails
    begin, not cover the range. The released model's own
    ``bar_dist_init_batches = 100`` with an untrimmed fit would have been subject
    to the same lottery, so [-56.13, 256.08] should be read as one draw from it
    rather than as a target to hit.

    Returns ``num_buckets + 1`` strictly increasing borders; quantile ties on a
    spiky target are nudged apart, since the distribution requires positive
    width in the outer buckets.
    """
    y = targets[torch.isfinite(targets)].double()
    lo, hi = trim
    qs = torch.linspace(lo, hi, num_buckets + 1, dtype=torch.float64)
    borders = torch.quantile(y, qs)

    # Enforce strict monotonicity.
    for i in range(1, borders.numel()):
        if borders[i] <= borders[i - 1]:
            borders[i] = borders[i - 1] + eps
    return borders.float()


def fit_grid_edges_2d(
    targets: torch.Tensor,
    j_2d: int = 10,
    quantiles: tuple[float, float] = DEFAULT_GRID_QUANTILES,
) -> torch.Tensor:
    """Uniform edges over a robust range -- required, since the 2-D loss assumes
    one scalar bin width. Values outside fall to the 9-region tail structure."""
    y = targets[torch.isfinite(targets)].double().numpy()
    lo, hi = np.quantile(y, quantiles)
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        lo, hi = float(np.min(y)), float(np.max(y))
    return torch.linspace(float(lo), float(hi), j_2d + 1, dtype=torch.float32)


# ---------------------------------------------------------------------------
# what the grid costs
# ---------------------------------------------------------------------------


def _bin_centres(edges: torch.Tensor) -> torch.Tensor:
    return 0.5 * (edges[:-1] + edges[1:])


def _quantise(y: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Snap each value to the centre of the bin it falls in (clamped at the ends)."""
    centres = _bin_centres(edges)
    idx = torch.bucketize(y, edges[1:-1].contiguous())
    return centres[idx.clamp(0, centres.numel() - 1)]


@dataclass
class FloorResult:
    j_2d: int
    tau_sd: float
    grid_bin_width: float
    bar_min_width: float
    mass_outside_grid: float
    n_tau_knots: int
    mean_floor: float
    mean_outside_hull: float
    density_l2_rel: float

    def __str__(self) -> str:
        return (
            f"what the J={self.j_2d} grid costs  (standardised units)\n"
            f"  tau spread (sd)              {self.tau_sd:8.4f}\n"
            f"\n"
            f"  MEANS (CATE) -- essentially free\n"
            f"    E[y] = sum p_i c_i interpolates continuously between bin\n"
            f"    centres, so a grid constrains the mean only by its hull.\n"
            f"    residual mean floor        {self.mean_floor:8.4f}"
            f"   ({self.mean_floor / self.tau_sd:5.2%} of sd)\n"
            f"    conditional means outside hull {self.mean_outside_hull:7.2%}\n"
            f"\n"
            f"  DENSITY p(tau|x) -- this is the real constraint\n"
            f"    representable tau knots    {self.n_tau_knots:8d}\n"
            f"    grid bin width             {self.grid_bin_width:8.4f}"
            f"   ({self.grid_bin_width / self.tau_sd:5.2f} sd per bin)\n"
            f"    best-case rel. L2 error    {self.density_l2_rel:8.2%}\n"
            f"    mass outside grid range    {self.mass_outside_grid:8.2%}\n"
            f"\n"
            f"  1-D head min bucket width    {self.bar_min_width:8.4f}"
            f"   (quantile-allocated, for scale)"
        )


def _density_l2_floor(tau: np.ndarray, bin_width: float, j_2d: int) -> float:
    """Best relative L2 error of any density the grid can express.

    A uniform J x J grid induces a tau density that is piecewise linear with
    knots spaced ``bin_width`` apart (the convolution of two uniform-bin
    histograms), which is what the eval pipeline's trapezoid rule assumes. The
    floor is therefore the residual of projecting the true tau density onto that
    piecewise-linear basis -- no trained model can do better.
    """
    lo, hi = -(j_2d - 1) * bin_width, (j_2d - 1) * bin_width
    grid = np.linspace(lo, hi, 4001)
    hist, edges = np.histogram(tau, bins=400, range=(lo, hi), density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    truth = np.interp(grid, centres, hist)

    knots = np.arange(-(j_2d - 1), j_2d) * bin_width
    basis = np.stack(
        [np.clip(1.0 - np.abs(grid - k) / bin_width, 0.0, None) for k in knots],
        axis=1,
    )
    coef, *_ = np.linalg.lstsq(basis, truth, rcond=None)
    resid = truth - basis @ coef
    denom = np.sqrt(np.trapezoid(truth**2, grid))
    return float(np.sqrt(np.trapezoid(resid**2, grid)) / max(denom, 1e-12))


def oracle_floor(
    n_batches: int = 60,
    j_2d: int = 10,
    num_buckets: int = 100,
    prior_cfg: PriorConfig | None = None,
    batch_cfg: BatchConfig | None = None,
    seed_base: int = 910_000,
) -> FloorResult:
    """What the grid costs, separated into means and densities.

    An oracle that sees the true outcomes and pays only for having to express
    them on a finite grid. No trained model beats it. Reported separately for
    CATE (a mean, barely constrained) and for p(tau|x) (a density, genuinely
    constrained) -- because conflating the two is the easiest way to mistake a
    resolution limit for a model failure, or vice versa.
    """
    prior_cfg = prior_cfg or PriorConfig()
    batch_cfg = batch_cfg or BatchConfig()

    fit = collect_targets(
        n_batches=n_batches, prior_cfg=prior_cfg, batch_cfg=batch_cfg,
        seed_base=seed_base,
    )
    edges = fit_grid_edges_2d(fit["query"], j_2d)
    borders = fit_bar_borders_1d(fit["query"], num_buckets)

    y0s, y1s, batch_means = [], [], []
    for i in range(n_batches):
        rec = sample_batch(seed_base + 500_000 + i, prior_cfg)
        if rec["meta"]["nonfinite"]:
            continue
        sep = sample_single_eval_pos(seed_base + i, prior_cfg.seq_len, batch_cfg)
        b = make_joint_batch(rec, sep, batch_cfg)
        y0s.append(b["y_do0"].reshape(-1))
        y1s.append(b["y_do1"].reshape(-1))
        batch_means.append(float((b["y_do1"] - b["y_do0"]).mean()))

    y0 = torch.cat(y0s)
    y1 = torch.cat(y1s)
    keep = torch.isfinite(y0) & torch.isfinite(y1)
    y0, y1 = y0[keep], y1[keep]
    tau = y1 - y0

    centres = _bin_centres(edges)
    c_lo, c_hi = float(centres[0]), float(centres[-1])

    # A grid model can hit any mean inside the hull of its bin centres, so the
    # only irreducible mean error is clipping.
    means = np.asarray([m for m in batch_means if np.isfinite(m)])
    clipped = np.clip(means, c_lo - c_hi, c_hi - c_lo)
    mean_floor = float(np.sqrt(np.mean((clipped - means) ** 2))) if means.size else 0.0
    outside_hull = (
        float(np.mean((means < c_lo - c_hi) | (means > c_hi - c_lo)))
        if means.size
        else 0.0
    )

    bw = float(edges[1] - edges[0])
    outside = (y0 < edges[0]) | (y0 > edges[-1]) | (y1 < edges[0]) | (y1 > edges[-1])

    return FloorResult(
        j_2d=j_2d,
        tau_sd=float(tau.std()),
        grid_bin_width=bw,
        bar_min_width=float((borders[1:] - borders[:-1]).min()),
        mass_outside_grid=float(outside.float().mean()),
        n_tau_knots=2 * j_2d - 1,
        mean_floor=mean_floor,
        mean_outside_hull=outside_hull,
        density_l2_rel=_density_l2_floor(tau.numpy(), bw, j_2d),
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Report the discretisation floor.")
    ap.add_argument("--n-batches", type=int, default=60)
    ap.add_argument("--j-2d", type=int, default=10)
    ap.add_argument("--seq-len", type=int, default=1000)
    args = ap.parse_args()

    print(
        oracle_floor(
            n_batches=args.n_batches,
            j_2d=args.j_2d,
            prior_cfg=PriorConfig(seq_len=args.seq_len),
        )
    )
