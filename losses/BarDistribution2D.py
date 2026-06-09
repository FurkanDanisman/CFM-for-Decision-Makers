"""
2D BarDistribution for joint (Y_do0, Y_do1) paired potential outcomes.

Mirrors UWYK's 1D BarDistribution extended to 2D.

The 2D space is divided into 9 regions based on where (Y_do0, Y_do1) falls
relative to the inner region [-1,1]×[-1,1]:

    Region 0 — inner-inner  : both in [-1,1]
    Region 1 — L0-inner     : Y_do0 < -1,    Y_do1 in [-1,1]
    Region 2 — R0-inner     : Y_do0 > +1,    Y_do1 in [-1,1]
    Region 3 — inner-L1     : Y_do0 in [-1,1], Y_do1 < -1
    Region 4 — inner-R1     : Y_do0 in [-1,1], Y_do1 > +1
    Region 5 — L0-L1        : Y_do0 < -1,    Y_do1 < -1
    Region 6 — L0-R1        : Y_do0 < -1,    Y_do1 > +1
    Region 7 — R0-L1        : Y_do0 > +1,    Y_do1 < -1
    Region 8 — R0-R1        : Y_do0 > +1,    Y_do1 > +1

Model outputs per query point:
    J² logits   → softmax → p_mat (J×J, conditional distribution within inner region)
    9 logits    → softmax → 9 region mixture weights
    4 raw params → tail scales σ_L0, σ_R0, σ_L1, σ_R1

Loss: -average_log_prob (density, not discrete probability).

Mixed-case density (one axis inside, one outside) uses the boundary row/column
of p_mat as an approximation to the conditional density at the boundary point.

Both-outside density uses a bivariate Gaussian with ρ derived from p_mat,
normalized via Sheppard's theorem for the appropriate quadrant.

At inference, call `fit_malc_inner` to smooth p_mat into a MALC 2D density.
"""

import math
import sys
import os

import torch
import torch.nn.functional as F
from torch import Tensor

# Region index constants
R_INNER = 0   # Y_do0 in [-1,1], Y_do1 in [-1,1]
R_L0    = 1   # Y_do0 < -1,      Y_do1 in [-1,1]
R_R0    = 2   # Y_do0 > +1,      Y_do1 in [-1,1]
R_L1    = 3   # Y_do0 in [-1,1], Y_do1 < -1
R_R1    = 4   # Y_do0 in [-1,1], Y_do1 > +1
R_L0L1  = 5  # Y_do0 < -1,      Y_do1 < -1
R_L0R1  = 6  # Y_do0 < -1,      Y_do1 > +1
R_R0L1  = 7  # Y_do0 > +1,      Y_do1 < -1
R_R0R1  = 8  # Y_do0 > +1,      Y_do1 > +1

N_REGIONS     = 9
N_TAIL_PARAMS = 4
SOFTPLUS_FLOOR = 1e-3


# ── Public API ─────────────────────────────────────────────────────────────────

def total_params(J: int) -> int:
    """Total output parameters per query point: J² inner logits + 9 region weights + 4 tail scales."""
    return J * J + N_REGIONS + N_TAIL_PARAMS


def make_edges(J: int, y_min: float = -1.0, y_max: float = 1.0) -> Tensor:
    """J+1 equidistant bin edges over [y_min, y_max]. Returns (J+1,)."""
    return torch.linspace(y_min, y_max, steps=J + 1)


def neg_log_prob_2d(
    pred:  Tensor,
    y0:    Tensor,
    y1:    Tensor,
    J:     int,
    edges: Tensor,
) -> Tensor:
    """
    2D log-density loss (negative mean).

    pred:  (B, M, J²+9+4) — raw model output
    y0:    (B, M) or (B, M, 1) — Y_do0 targets
    y1:    (B, M) or (B, M, 1) — Y_do1 targets
    edges: (J+1,) — bin edges on the same device as pred
    Returns: scalar
    """
    if y0.dim() == 3:
        y0 = y0.squeeze(-1)
    if y1.dim() == 3:
        y1 = y1.squeeze(-1)

    B, M, _ = pred.shape
    edges = edges.to(pred.device)
    bin_width = float((edges[-1] - edges[0]).item() / J)
    log_bw    = math.log(bin_width)

    p_mat, w_reg, sL0, sR0, sL1, sR1 = _unpack(pred, J, bin_width)
    # p_mat:  (B, M, J, J)
    # w_reg:  (B, M, 9)  — region mixture weights, already softmaxed
    # sL0, sR0, sL1, sR1: (B, M)

    bin_centers = ((edges[:-1] + edges[1:]) / 2)  # (J,)
    rho = _compute_rho(p_mat, bin_centers)          # (B, M) ∈ (-1+ε, 1-ε)

    lo = edges[0].item()
    hi = edges[-1].item()

    # Bin index for y0 and y1 (clamped to [0, J-1]; tail points get boundary bin)
    interior = edges[1:-1].contiguous()
    j0_idx = torch.bucketize(y0.contiguous(), interior, right=False).clamp(0, J - 1)
    j1_idx = torch.bucketize(y1.contiguous(), interior, right=False).clamp(0, J - 1)

    # Region masks
    in0   = (y0 >= lo) & (y0 <= hi)
    in1   = (y1 >= lo) & (y1 <= hi)
    out_L0 = y0 < lo
    out_R0 = y0 > hi
    out_L1 = y1 < lo
    out_R1 = y1 > hi

    # Region index for each (b, m) pair — used for gather
    region_idx = torch.zeros(B, M, dtype=torch.long, device=pred.device)
    region_idx[out_L0 & in1]   = R_L0
    region_idx[out_R0 & in1]   = R_R0
    region_idx[in0 & out_L1]   = R_L1
    region_idx[in0 & out_R1]   = R_R1
    region_idx[out_L0 & out_L1] = R_L0L1
    region_idx[out_L0 & out_R1] = R_L0R1
    region_idx[out_R0 & out_L1] = R_R0L1
    region_idx[out_R0 & out_R1] = R_R0R1
    # R_INNER = 0 is the default (already set)

    # ── Compute log_prob contributions per region ──────────────────────────────

    log_w = w_reg.clamp(min=1e-45).log()  # (B, M, 9)

    # Flat index into J² grid for gathering p_mat
    flat_idx = (j0_idx * J + j1_idx).unsqueeze(-1)          # (B, M, 1)
    p_flat   = p_mat.reshape(B, M, J * J)                    # (B, M, J²)
    p_at_ij  = p_flat.gather(-1, flat_idx).squeeze(-1)       # (B, M)

    # ── Region 0: inner-inner ────────────────────────────────────────────────
    lp_inner = log_w[..., R_INNER] + p_at_ij.clamp(1e-45).log() - 2 * log_bw

    # ── Regions 1,2: mixed — one Y_do0 axis outside ──────────────────────────
    # Boundary conditional: f(y1 | y0=boundary) ≈ p_mat[boundary_row, j1] / marg / bw

    row_L0   = p_mat[..., 0,   :]                          # (B, M, J) — j0=0 boundary
    row_R0   = p_mat[..., J-1, :]                          # (B, M, J) — j0=J-1 boundary
    marg_L0  = row_L0.sum(-1).clamp(1e-45)                 # (B, M)
    marg_R0  = row_R0.sum(-1).clamp(1e-45)                 # (B, M)

    p_cond_L0 = row_L0.gather(-1, j1_idx.unsqueeze(-1)).squeeze(-1)  # (B, M)
    p_cond_R0 = row_R0.gather(-1, j1_idx.unsqueeze(-1)).squeeze(-1)

    log_cond_L0 = p_cond_L0.clamp(1e-45).log() - marg_L0.log() - log_bw
    log_cond_R0 = p_cond_R0.clamp(1e-45).log() - marg_R0.log() - log_bw

    lp_L0 = log_w[..., R_L0] + _log_half_gauss(y0, lo, sL0) + log_cond_L0
    lp_R0 = log_w[..., R_R0] + _log_half_gauss(y0, hi, sR0) + log_cond_R0

    # ── Regions 3,4: mixed — one Y_do1 axis outside ──────────────────────────
    col_L1   = p_mat[..., :, 0  ]                          # (B, M, J) — j1=0 boundary
    col_R1   = p_mat[..., :, J-1]                          # (B, M, J) — j1=J-1 boundary
    marg_L1  = col_L1.sum(-1).clamp(1e-45)
    marg_R1  = col_R1.sum(-1).clamp(1e-45)

    p_cond_L1 = col_L1.gather(-1, j0_idx.unsqueeze(-1)).squeeze(-1)
    p_cond_R1 = col_R1.gather(-1, j0_idx.unsqueeze(-1)).squeeze(-1)

    log_cond_L1 = p_cond_L1.clamp(1e-45).log() - marg_L1.log() - log_bw
    log_cond_R1 = p_cond_R1.clamp(1e-45).log() - marg_R1.log() - log_bw

    lp_L1 = log_w[..., R_L1] + log_cond_L1 + _log_half_gauss(y1, lo, sL1)
    lp_R1 = log_w[..., R_R1] + log_cond_R1 + _log_half_gauss(y1, hi, sR1)

    # ── Regions 5-8: both outside — bivariate half-Gaussian ─────────────────
    # Normalization by Sheppard's theorem:
    #   Same-direction corners (L0L1, R0R1): P(Z0<0, Z1<0) = 1/4 + arcsin(ρ)/(2π)
    #   Opposite-direction corners (L0R1, R0L1): P(Z0<0, Z1>0) = 1/4 - arcsin(ρ)/(2π)
    log_norm_same = torch.log(0.25 + torch.asin(rho) / (2 * math.pi))    # (B, M)
    log_norm_opp  = torch.log((0.25 - torch.asin(rho) / (2 * math.pi)).clamp(1e-45))

    lp_L0L1 = log_w[..., R_L0L1] + _log_biv_gauss(y0, y1, lo, lo, sL0, sL1, rho) - log_norm_same
    lp_L0R1 = log_w[..., R_L0R1] + _log_biv_gauss(y0, y1, lo, hi, sL0, sR1, rho) - log_norm_opp
    lp_R0L1 = log_w[..., R_R0L1] + _log_biv_gauss(y0, y1, hi, lo, sR0, sL1, rho) - log_norm_opp
    lp_R0R1 = log_w[..., R_R0R1] + _log_biv_gauss(y0, y1, hi, hi, sR0, sR1, rho) - log_norm_same

    # ── Gather the relevant region's log_prob ─────────────────────────────────
    all_lp = torch.stack(
        [lp_inner, lp_L0, lp_R0, lp_L1, lp_R1,
         lp_L0L1, lp_L0R1, lp_R0L1, lp_R0R1],
        dim=-1
    )  # (B, M, 9)

    log_prob = all_lp.gather(-1, region_idx.unsqueeze(-1)).squeeze(-1)  # (B, M)

    return -log_prob.mean()


# ── Inference helpers ─────────────────────────────────────────────────────────

def unpack_pred(pred: Tensor, J: int, bin_width: float):
    """
    Unpack raw model output.
    Returns p_mat (J×J), region_weights (9,), and tail scales (4,).
    Handles single-query predictions (no batch/sequence dims needed).
    """
    return _unpack(pred, J, bin_width)


def fit_malc_inner(p_mat_np, grid_x, grid_y, **malc_kwargs):
    """
    Fit a MALC_2D density to the inner bin probability matrix.

    p_mat_np: numpy (J, J) array — conditional distribution within [-1,1]²
    grid_x, grid_y: numpy (J+1,) bin edges
    Returns: MALC2DFit object (call dmalc_2d on it for density evaluation)
    """
    malc_dir = os.path.join(os.path.dirname(__file__), '..', 'MALC')
    if malc_dir not in sys.path:
        sys.path.insert(0, malc_dir)
    from malc_2d import MALC_2D
    return MALC_2D(p_mat_np, grid_x, grid_y, **malc_kwargs)


def eval_density_2d(fit_obj, y0_pts, y1_pts, edges, sL0, sR0, sL1, sR1, rho, w_region):
    """
    Evaluate the full 2D density at arbitrary points using a fitted MALC object.

    fit_obj:       MALC2DFit from fit_malc_inner
    y0_pts, y1_pts: numpy (N,) arrays of evaluation points
    edges:          numpy (J+1,) bin edges
    sL0, sR0, sL1, sR1: float tail scales
    rho:            float correlation (derived from p_mat)
    w_region:       numpy (9,) region mixture weights (sum to 1)
    Returns: numpy (N,) density values
    """
    import numpy as np

    malc_dir = os.path.join(os.path.dirname(__file__), '..', 'MALC')
    if malc_dir not in sys.path:
        sys.path.insert(0, malc_dir)
    from malc_2d import dmalc_2d

    lo, hi = edges[0], edges[-1]
    pts = np.column_stack([y0_pts, y1_pts])
    N = len(y0_pts)
    density = np.zeros(N)

    in0 = (y0_pts >= lo) & (y0_pts <= hi)
    in1 = (y1_pts >= lo) & (y1_pts <= hi)

    # Region 0: inner-inner — MALC density
    mask_inner = in0 & in1
    if mask_inner.any():
        density[mask_inner] = (
            w_region[R_INNER] * dmalc_2d(fit_obj, pts[mask_inner])
        )

    # Regions 1,2: mixed — half-Gauss × MALC boundary conditional
    mask_L0 = (~in0) & (y0_pts < lo) & in1
    mask_R0 = (~in0) & (y0_pts > hi) & in1
    if mask_L0.any():
        density[mask_L0] = (
            w_region[R_L0]
            * _half_gauss_np(y0_pts[mask_L0], lo, sL0)
            * _boundary_cond_malc(fit_obj, y1_pts[mask_L0], boundary_axis=0, at_min=True)
        )
    if mask_R0.any():
        density[mask_R0] = (
            w_region[R_R0]
            * _half_gauss_np(y0_pts[mask_R0], hi, sR0)
            * _boundary_cond_malc(fit_obj, y1_pts[mask_R0], boundary_axis=0, at_min=False)
        )

    mask_L1 = in0 & (~in1) & (y1_pts < lo)
    mask_R1 = in0 & (~in1) & (y1_pts > hi)
    if mask_L1.any():
        density[mask_L1] = (
            w_region[R_L1]
            * _boundary_cond_malc(fit_obj, y0_pts[mask_L1], boundary_axis=1, at_min=True)
            * _half_gauss_np(y1_pts[mask_L1], lo, sL1)
        )
    if mask_R1.any():
        density[mask_R1] = (
            w_region[R_R1]
            * _boundary_cond_malc(fit_obj, y0_pts[mask_R1], boundary_axis=1, at_min=False)
            * _half_gauss_np(y1_pts[mask_R1], hi, sR1)
        )

    # Regions 5-8: both-outside — bivariate half-Gaussian
    norm_same = 0.25 + math.asin(rho) / (2 * math.pi)
    norm_opp  = max(0.25 - math.asin(rho) / (2 * math.pi), 1e-45)

    def _biv(y0s, y1s, mu0, mu1, s0, s1):
        import scipy.stats
        cov = np.array([[s0**2, rho*s0*s1], [rho*s0*s1, s1**2]])
        return scipy.stats.multivariate_normal.pdf(
            np.column_stack([y0s, y1s]),
            mean=[mu0, mu1], cov=cov
        )

    m_L0L1 = (~in0) & (y0_pts < lo) & (~in1) & (y1_pts < lo)
    m_L0R1 = (~in0) & (y0_pts < lo) & (~in1) & (y1_pts > hi)
    m_R0L1 = (~in0) & (y0_pts > hi) & (~in1) & (y1_pts < lo)
    m_R0R1 = (~in0) & (y0_pts > hi) & (~in1) & (y1_pts > hi)

    if m_L0L1.any():
        density[m_L0L1] = w_region[R_L0L1] * _biv(y0_pts[m_L0L1], y1_pts[m_L0L1], lo, lo, sL0, sL1) / norm_same
    if m_L0R1.any():
        density[m_L0R1] = w_region[R_L0R1] * _biv(y0_pts[m_L0R1], y1_pts[m_L0R1], lo, hi, sL0, sR1) / norm_opp
    if m_R0L1.any():
        density[m_R0L1] = w_region[R_R0L1] * _biv(y0_pts[m_R0L1], y1_pts[m_R0L1], hi, lo, sR0, sL1) / norm_opp
    if m_R0R1.any():
        density[m_R0R1] = w_region[R_R0R1] * _biv(y0_pts[m_R0R1], y1_pts[m_R0R1], hi, hi, sR0, sR1) / norm_same

    return density


# ── Private helpers ────────────────────────────────────────────────────────────

def _unpack(pred: Tensor, J: int, bin_width: float):
    JJ = J * J
    p_mat    = F.softmax(pred[..., :JJ].float(), dim=-1).reshape(*pred.shape[:-1], J, J)
    w_region = F.softmax(pred[..., JJ:JJ + N_REGIONS].float(), dim=-1)
    tail_raw = pred[..., JJ + N_REGIONS:]

    sL0 = _safe_scale(tail_raw[..., 0], bin_width)
    sR0 = _safe_scale(tail_raw[..., 1], bin_width)
    sL1 = _safe_scale(tail_raw[..., 2], bin_width)
    sR1 = _safe_scale(tail_raw[..., 3], bin_width)

    return p_mat, w_region, sL0, sR0, sL1, sR1


def _safe_scale(raw: Tensor, base: float) -> Tensor:
    return base * (F.softplus(raw.float()) + SOFTPLUS_FLOOR)


def _compute_rho(p_mat: Tensor, bin_centers: Tensor) -> Tensor:
    """Pearson ρ derived from the joint p_mat distribution. Differentiable."""
    c = bin_centers  # (J,)

    marg0 = p_mat.sum(dim=-1)   # (..., J) marginal over Y_do0
    marg1 = p_mat.sum(dim=-2)   # (..., J) marginal over Y_do1

    E0  = (c * marg0).sum(dim=-1)
    E1  = (c * marg1).sum(dim=-1)
    E02 = (c**2 * marg0).sum(dim=-1)
    E12 = (c**2 * marg1).sum(dim=-1)
    E01 = (p_mat * c.unsqueeze(-2) * c.unsqueeze(-1)).sum(dim=(-2, -1))

    cov  = E01 - E0 * E1
    var0 = (E02 - E0**2).clamp(min=1e-8)
    var1 = (E12 - E1**2).clamp(min=1e-8)
    rho  = cov / (var0.sqrt() * var1.sqrt())
    return rho.clamp(-1 + 1e-6, 1 - 1e-6)


def _log_half_gauss(y: Tensor, boundary: float, scale: Tensor) -> Tensor:
    """
    Log density of a half-Gaussian at y (the half nearest `boundary`).
    Integrates to 1 over the appropriate half-line due to the factor 2.
    """
    return (
        math.log(2.0)
        - 0.5 * ((y - boundary) / scale) ** 2
        - scale.log()
        - 0.5 * math.log(2 * math.pi)
    )


def _log_biv_gauss(
    y0: Tensor, y1: Tensor,
    mu0: float, mu1: float,
    s0: Tensor, s1: Tensor,
    rho: Tensor,
) -> Tensor:
    """Log density of the bivariate Gaussian N((mu0,mu1), Σ) at (y0, y1)."""
    z0   = (y0 - mu0) / s0
    z1   = (y1 - mu1) / s1
    rho2 = (1.0 - rho**2).clamp(min=1e-8)
    return (
        -(0.5 / rho2) * (z0**2 - 2 * rho * z0 * z1 + z1**2)
        - s0.log() - s1.log()
        - 0.5 * rho2.log()
        - math.log(2 * math.pi)
    )


def _half_gauss_np(y, boundary, scale):
    import numpy as np
    return 2.0 * np.exp(-0.5 * ((y - boundary) / scale) ** 2) / (scale * math.sqrt(2 * math.pi))


def _boundary_cond_malc(fit_obj, y_in, boundary_axis, at_min):
    """
    Approximate conditional density at the boundary using MALC.
    boundary_axis=0 → y0 is at boundary; evaluate density of y_in as y1.
    at_min=True → boundary is at grid min (e.g., y0=-1).
    """
    import numpy as np
    from malc_2d import dmalc_2d

    grid = fit_obj.grid_x if boundary_axis == 0 else fit_obj.grid_y
    boundary_val = float(grid[0] if at_min else grid[-1])
    N = len(y_in)

    if boundary_axis == 0:
        pts_bnd = np.column_stack([np.full(N, boundary_val), y_in])
    else:
        pts_bnd = np.column_stack([y_in, np.full(N, boundary_val)])

    dens_bnd = dmalc_2d(fit_obj, pts_bnd)

    # Marginal density at boundary (integral over the other axis)
    n_eval = 200
    if boundary_axis == 0:
        y_range = np.linspace(fit_obj.grid_y[0], fit_obj.grid_y[-1], n_eval)
        pts_marg = np.column_stack([np.full(n_eval, boundary_val), y_range])
    else:
        y_range = np.linspace(fit_obj.grid_x[0], fit_obj.grid_x[-1], n_eval)
        pts_marg = np.column_stack([y_range, np.full(n_eval, boundary_val)])

    marg_dens = dmalc_2d(fit_obj, pts_marg)
    dy = (y_range[-1] - y_range[0]) / (n_eval - 1)
    marginal = float(np.trapz(marg_dens, dx=dy))

    return dens_bnd / max(marginal, 1e-45)
