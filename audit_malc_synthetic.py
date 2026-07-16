"""
Test MALC on synthetic p_mat inputs where we KNOW the answer.

Four scenarios of increasing difficulty:

  (A) Single sharp Gaussian at (-0.5, -0.5), σ=0.05.
      Expected: MALC K=1 fits a tight log-concave; K=2/3 place near-degenerate components.

  (B) Single wide Gaussian at (0, 0), σ=0.3.
      Expected: MALC K=1 fits a smooth Gaussian-like log-concave; K=2/3 similar.

  (C) Bimodal: 0.5 · N((-0.5,-0.5), σ=0.1) + 0.5 · N((+0.5,+0.5), σ=0.1)
      Expected: MALC K=2 recovers the two components clearly.

  (D) Bimodal off-diagonal: 0.5 · N((-0.5,+0.5), σ=0.1) + 0.5 · N((+0.5,-0.5), σ=0.1)
      Expected: MALC K=2 recovers the two off-diagonal components.

If MALC (C) or (D) collapses to K=1-like output, MALC IS the bug.
If MALC (A)/(B) is dramatically wider than the true Gaussian at the same
peak position, MALC is over-smoothing.

Plots the RAW input, MALC(K=1), MALC(K=2), MALC(K=3), and MALC(K=2, alpha=0.1)
for each scenario.
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'MALC'))

from malc_2d import MALC_2D, dmalc_2d

OUT_DIR = os.path.join(_REPO, 'experiments')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Grid identical to what BarDistribution2D uses (J=100 bins over [-1, 1]) ──
J = 100
edges = np.linspace(-1.0, 1.0, J + 1)
bin_width = float(edges[1] - edges[0])
centers = 0.5 * (edges[:-1] + edges[1:])

# Fine eval grid (200 x 200) for MALC density evaluation
N_EVAL = 200
xs = np.linspace(edges[0], edges[-1], N_EVAL)
ys = np.linspace(edges[0], edges[-1], N_EVAL)
XX, YY = np.meshgrid(xs, ys, indexing='xy')
eval_pts = np.column_stack([XX.ravel(), YY.ravel()])


def make_p_mat_from_pdf(pdf):
    """Integrate pdf over each bin to build a probability matrix.
    p_mat[i, j] = P(Y_do0 in bin_i, Y_do1 in bin_j)."""
    p_mat = np.zeros((J, J))
    for i in range(J):
        for j in range(J):
            # Midpoint approximation with subgrid
            sub = 5
            x_sub = np.linspace(edges[j], edges[j+1], sub)
            y_sub = np.linspace(edges[i], edges[i+1], sub)
            XX_s, YY_s = np.meshgrid(x_sub, y_sub, indexing='xy')
            pts = np.column_stack([XX_s.ravel(), YY_s.ravel()])
            # Here index i = Y_do0 bin, j = Y_do1 bin
            # But we're using pts as (x, y) = (Y_do0, Y_do1)? Let's be explicit:
            # p_mat[i, j] means Y_do0 in bin_i (rows), Y_do1 in bin_j (cols)
            # So pts should have x = Y_do0 = center of bin_i (row), y = Y_do1 = center of bin_j (col)
            # But my meshgrid used x=xs=cols, y=ys=rows.
            # Just build in one loop with correct semantics
            pass
    # Simpler: p_mat[i, j] = pdf at (Y_do0=centers[i], Y_do1=centers[j]) * bin_area
    p_mat = np.zeros((J, J))
    for i in range(J):        # Y_do0 index
        for j in range(J):    # Y_do1 index
            p_mat[i, j] = pdf(centers[i], centers[j]) * (bin_width ** 2)
    p_mat = p_mat / p_mat.sum()  # renormalize
    return p_mat


def gauss_pdf(mu_y0, mu_y1, sigma):
    def f(y0, y1):
        return np.exp(-0.5 * ((y0 - mu_y0)**2 + (y1 - mu_y1)**2) / sigma**2) \
               / (2 * np.pi * sigma**2)
    return f


def mixture_pdf(components, weights):
    """List of (mu_y0, mu_y1, sigma) + list of weights."""
    def f(y0, y1):
        total = 0.0
        for (mu0, mu1, sig), w in zip(components, weights):
            total += w * np.exp(-0.5 * ((y0 - mu0)**2 + (y1 - mu1)**2) / sig**2) \
                      / (2 * np.pi * sig**2)
        return total
    return f


scenarios = {
    'A_sharp_single':    ('Single sharp Gaussian at (-0.5,-0.5), σ=0.05',
                          gauss_pdf(-0.5, -0.5, 0.05)),
    'B_wide_single':     ('Single wide Gaussian at (0,0), σ=0.3',
                          gauss_pdf(0.0, 0.0, 0.3)),
    'C_bimodal_diag':    ('Bimodal on-diagonal 50/50 at (±0.5,±0.5), σ=0.1',
                          mixture_pdf([(-0.5, -0.5, 0.1), (0.5, 0.5, 0.1)], [0.5, 0.5])),
    'D_bimodal_offdiag': ('Bimodal off-diagonal 50/50 at (-0.5,+0.5) and (+0.5,-0.5), σ=0.1',
                          mixture_pdf([(-0.5, 0.5, 0.1), (0.5, -0.5, 0.1)], [0.5, 0.5])),
}


def fit_malc_variant(p_mat, K, alpha=2.0, B=1000):
    fit = MALC_2D(
        p_mat, edges, edges,
        K=K, B_fit=B, B_select=B, alpha=alpha,
        parallel=False, seed=42,
    )
    density = dmalc_2d(fit, eval_pts).reshape(N_EVAL, N_EVAL)
    return density


for name, (title, pdf) in scenarios.items():
    print(f"\n=== {name}: {title} ===", flush=True)
    p_mat = make_p_mat_from_pdf(pdf)
    # True density on eval grid for comparison
    true_density = np.array([[pdf(xs[j], ys[i]) for j in range(N_EVAL)] for i in range(N_EVAL)])
    # Sanity: p_mat integrates to 1, true_density integrates to 1
    print(f"  p_mat sum = {p_mat.sum():.4f}, true density integral ≈ "
          f"{true_density.sum() * (xs[1]-xs[0]) * (ys[1]-ys[0]):.4f}", flush=True)

    # 5 panels: true density, raw p_mat as density, MALC K=1, MALC K=2, MALC K=3
    print(f"  fitting MALC K=1 alpha=2.0…", flush=True)
    malc_K1 = fit_malc_variant(p_mat, K=1, alpha=2.0)
    print(f"  fitting MALC K=2 alpha=2.0…", flush=True)
    malc_K2 = fit_malc_variant(p_mat, K=2, alpha=2.0)
    print(f"  fitting MALC K=3 alpha=2.0…", flush=True)
    malc_K3 = fit_malc_variant(p_mat, K=3, alpha=2.0)
    print(f"  fitting MALC K=2 alpha=0.1…", flush=True)
    malc_K2_small = fit_malc_variant(p_mat, K=2, alpha=0.1)

    # Raw p_mat rendered as a density on eval grid: assign each eval point the
    # value of the bin it falls into, divided by bin area.
    raw_2d = np.zeros((N_EVAL, N_EVAL))
    for i in range(N_EVAL):
        for j in range(N_EVAL):
            b0 = int(np.clip((xs[j] - edges[0]) / bin_width, 0, J - 1))  # Y_do0 bin
            b1 = int(np.clip((ys[i] - edges[0]) / bin_width, 0, J - 1))  # Y_do1 bin
            raw_2d[i, j] = p_mat[b0, b1] / (bin_width ** 2)

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    vmax = max(true_density.max(), raw_2d.max(),
               malc_K1.max(), malc_K2.max(), malc_K3.max())
    for ax, arr, ttl in zip(
        axes,
        [true_density, raw_2d, malc_K1, malc_K2, malc_K3],
        [f'TRUE density', f'RAW p_mat', f'MALC K=1 α=2.0',
         f'MALC K=2 α=2.0', f'MALC K=3 α=2.0'],
    ):
        im = ax.imshow(arr, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin='lower',
                       cmap='viridis', vmin=0, vmax=vmax, aspect='equal')
        ax.plot([xs[0], xs[-1]], [ys[0], ys[-1]], 'w--', lw=0.8, alpha=0.4)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel('Y_do0'); ax.set_ylabel('Y_do1')
        plt.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f'{name}: {title}', y=1.02, fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'AUDIT_synth_{name}.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}", flush=True)

    # Also compare MALC K=2 at different alpha values
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, arr, ttl in zip(
        axes,
        [raw_2d, malc_K2, malc_K2_small],
        [f'RAW p_mat', f'MALC K=2 α=2.0 (default)', f'MALC K=2 α=0.1 (less smoothing)'],
    ):
        im = ax.imshow(arr, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin='lower',
                       cmap='viridis', vmin=0, vmax=vmax, aspect='equal')
        ax.plot([xs[0], xs[-1]], [ys[0], ys[-1]], 'w--', lw=0.8, alpha=0.4)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel('Y_do0'); ax.set_ylabel('Y_do1')
        plt.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f'{name}: α comparison  {title}', y=1.02, fontsize=12)
    fig.tight_layout()
    out2 = os.path.join(OUT_DIR, f'AUDIT_synth_{name}_alpha.png')
    fig.savefig(out2, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out2}", flush=True)

print("\nDone.", flush=True)
