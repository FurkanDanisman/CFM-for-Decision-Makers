"""Variant-T smoothing: fit MALC in tau space, after projection/convolution.

The calibration pipeline has two places MALC can sit.

  Variant J  smooth the JOINT p(y0, y1), then diagonal-integrate to tau.
             2D methods use their native joint; 1D methods use the
             independence joint p_y0 (x) p_y1. Implemented by
             realcause_eval/compute_malc_ci_cell{,_1d}.py.

  Variant T  form tau FIRST -- anti-diagonal projection for a 2D method,
             independence convolution for a 1D one -- then smooth that single
             density. This module.

T is not a cheaper approximation to J; the two smooth different objects. J
regularises the dependence structure as well as the shape, because a
log-concave fit on the plane constrains how the arms co-vary. T leaves the
tau pmf's construction untouched and only regularises its shape. Where they
disagree, the disagreement is attributable to the joint, which is the point of
running both.

WHY A SEPARATE 1D FITTER. MALC_2D cannot degenerate to this case -- it fits a
density over a triangulated plane. MALC/malc_1d.py is the 1D analogue, same
four-step recipe (EM mean correction, Beta jitter, sample B, log-concave MLE),
with the MLE's integral computed in closed form rather than by quadrature.

UNIFORM SUPPORT. tau supports coming out of cate_density_metrics are uniform
by construction -- tau_atoms builds a uniform lattice, _per_arm_common_grid
and _rebin_atoms_shared both re-express onto one. MALC_1D checks and raises
rather than mis-calibrating the Beta jitter against a varying bin width, so a
non-uniform support surfaces as an error instead of a plausible wrong number.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_MALC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MALC")
if _MALC_DIR not in sys.path:
    sys.path.insert(0, _MALC_DIR)

__all__ = ["edges_from_atoms", "smooth_tau_pmf", "SmootherConfig"]


class SmootherConfig:
    """Knobs for variant T. Defaults match the agreed smoke-test settings."""

    __slots__ = ("enabled", "B", "K", "seed", "n_tau", "pad")

    def __init__(self, enabled=False, B=100, K=1, seed=20180621, n_tau=4001,
                 pad=0.0):
        self.enabled = bool(enabled)
        self.B = int(B)
        self.K = int(K)
        self.seed = int(seed)
        self.n_tau = int(n_tau)
        # Fraction of the support width to extend on each side. The MLE's
        # support is the convex hull of the B sampled points, which is strictly
        # inside the atom range, so padding buys nothing by default. Kept as a
        # knob for the case where a downstream grid must be matched exactly.
        self.pad = float(pad)

    def __repr__(self):
        return (f"SmootherConfig(enabled={self.enabled}, B={self.B}, K={self.K}, "
                f"seed={self.seed}, n_tau={self.n_tau})")


def edges_from_atoms(atoms: np.ndarray) -> np.ndarray:
    """Bin edges for a uniform lattice of atom CENTRES.

    MALC_1D is a binned-data method: it wants the n+1 edges, not the n centres.
    The atoms carry the centres, so the edges sit half a width outside on each
    end. Getting this wrong shifts every fitted density by half a bin, which is
    exactly the class of error that scores finitely and looks like a biased
    model.
    """
    atoms = np.asarray(atoms, dtype=np.float64).reshape(-1)
    if atoms.size < 2:
        raise ValueError("need at least two atoms")
    w = float(atoms[1] - atoms[0])
    return np.concatenate([atoms - 0.5 * w, [atoms[-1] + 0.5 * w]])


def smooth_tau_pmf(atoms, pmf, cfg: SmootherConfig, query_seed: int | None = None):
    """(grid, pmf) after MALC-1D smoothing. Falls back to the input on failure.

    Returns a pmf on a uniform grid of cfg.n_tau points spanning the atom
    support, normalised to sum to 1 so the downstream scorers (which treat the
    second argument as a pmf, not a density) need no change.

    `query_seed` is mixed into the fit seed so that two queries in the same
    realization do not share one Beta-jitter draw. Reproducible: derived from
    cfg.seed, not from entropy.
    """
    from malc_1d import MALC_1D, dmalc_1d

    atoms = np.asarray(atoms, dtype=np.float64).reshape(-1)
    pmf = np.asarray(pmf, dtype=np.float64).reshape(-1)
    if pmf.size != atoms.size:
        raise ValueError(f"pmf/atoms length mismatch: {pmf.size} vs {atoms.size}")

    tot = pmf.sum()
    if not np.isfinite(tot) or tot <= 0 or not np.isfinite(pmf).all():
        return atoms, pmf

    edges = edges_from_atoms(atoms)
    seed = cfg.seed if query_seed is None else (cfg.seed + 1_000_003 * int(query_seed))

    try:
        fit = MALC_1D(pmf / tot, edges, K=cfg.K, B=cfg.B, seed=seed)
    except Exception:
        # A component fit can fail legitimately -- mass concentrated in a single
        # bin leaves the Beta parameters out of range. Returning the raw pmf
        # keeps that query in the table instead of dropping it, and the raw
        # column is the honest answer for a density MALC cannot improve on.
        return atoms, pmf / tot

    lo, hi = float(edges[0]), float(edges[-1])
    if cfg.pad > 0:
        span = hi - lo
        lo, hi = lo - cfg.pad * span, hi + cfg.pad * span
    grid = np.linspace(lo, hi, cfg.n_tau)

    dens = np.nan_to_num(dmalc_1d(fit, grid), nan=0.0, posinf=0.0, neginf=0.0)
    dens = np.maximum(dens, 0.0)
    s = dens.sum()
    if s <= 0:
        return atoms, pmf / tot
    return grid, dens / s
