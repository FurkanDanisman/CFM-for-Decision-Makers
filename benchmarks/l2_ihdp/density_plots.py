"""Per-realization density diagnostic plots.

Given a `method_out` dict from an L2 evaluator (IHDP / ACIC / syn), a
`truth` object, and the analytic truth densities on Y_CENTERS and
TAU_CENTERS, save a 4-panel PNG showing method curves vs the true density:

    (0, 0): p(Y_do(0) | x_{q=0})
    (0, 1): p(Y_do(1) | x_{q=0})
    (1, 0): p(τ | x_{q=0})
    (1, 1): p(τ_ATE)

One PNG per realization. Truth is drawn as a dashed black reference line.

Import as:
    from density_plots import save_density_diag_png
    save_density_diag_png(out_dir, r, method_out, p_y0_true, p_y1_true,
                           p_tau_true, p_ate_true, wb_fn)
"""
from __future__ import annotations
import os
import numpy as np


METHOD_COLOR = {
    'ours_fn50':     '#0F8A3C',
    'ours_fn10':     '#4FBF6F',
    'ours_dopfn_bb': '#2E4A6F',
    'dopfn':         '#8A4FBE',
    'uwyk_noanc':    '#B84A2A',
    'uwyk_anc':      '#DC7A5A',
}
METHOD_LABEL = {
    'ours_fn50':     'Ours (fn=50)',
    'ours_fn10':     'Ours (fn=10)',
    'ours_dopfn_bb': 'Ours-DoPFN-bb (200K)',
    'dopfn':         'Do-PFN',
    'uwyk_noanc':    'UWYK-NoAnc',
    'uwyk_anc':      'UWYK-FullAnc',
}


def _wass_bary(p_matrix, grid, wb_fn):
    p_ate = wb_fn(p_matrix, grid)
    dx = grid[1] - grid[0]
    s = float(p_ate.sum() * dx)
    return p_ate / s if s > 0 else p_ate


def save_density_diag_png(out_path: str, r: int,
                           method_out: dict,
                           p_y0_true: np.ndarray, p_y1_true: np.ndarray,
                           p_tau_true: np.ndarray, p_ate_true: np.ndarray,
                           Y_CENTERS: np.ndarray, TAU_CENTERS: np.ndarray,
                           wb_fn,
                           q_show: int = 0) -> str:
    """Save the per-realization 4-panel PNG. Returns the file path.
       q_show: which query index to show for the per-x panels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    panels = [
        (axes[0, 0], Y_CENTERS,   'p_y0',  r'$p(Y_{do(0)} \mid x_{q}=0)$'),
        (axes[0, 1], Y_CENTERS,   'p_y1',  r'$p(Y_{do(1)} \mid x_{q}=0)$'),
        (axes[1, 0], TAU_CENTERS, 'p_tau', r'$p(\tau \mid x_{q}=0)$'),
        (axes[1, 1], TAU_CENTERS, 'p_ate', r'$p(\tau_{ATE})$'),
    ]
    truths = {
        'p_y0':  p_y0_true[q_show],
        'p_y1':  p_y1_true[q_show],
        'p_tau': p_tau_true[q_show],
        'p_ate': p_ate_true,
    }
    for ax, grid, key, title in panels:
        ax.plot(grid, truths[key], 'k--', lw=2, alpha=0.85, label='truth')
        for name, d in method_out.items():
            if key == 'p_ate':
                if 'p_tau' not in d: continue
                p = _wass_bary(np.asarray(d['p_tau']), TAU_CENTERS, wb_fn)
            else:
                if key not in d: continue
                p = np.asarray(d[key])[q_show]
            color = METHOD_COLOR.get(name, '#666666')
            label = METHOD_LABEL.get(name, name)
            ax.plot(grid, p, color=color, lw=1.4, alpha=0.9, label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('value'); ax.set_ylabel('density')
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc='upper right', fontsize=8)
    fig.suptitle(f'Density diagnostics — realization r={r}  (q={q_show})',
                  fontsize=11, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out_path
