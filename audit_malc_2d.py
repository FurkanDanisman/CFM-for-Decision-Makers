"""
Diagnose MALC output vs raw p_mat in 2D directly.

Plot side-by-side heatmaps of the raw p_mat and MALC-smoothed density (K=1,
K=2, K=3) for a few problematic queries. Also plot the tau=Y_do1-Y_do0 marginal
derived from each 2D density.

If the 2D MALC density matches the 2D raw p_mat shape, my τ derivation is
wrong. If the 2D MALC density DOESN'T match the 2D raw p_mat, MALC is
mis-fitting or my transpose/axis mapping is wrong.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch
import matplotlib.pyplot as plt

UWYK_SRC   = os.environ.get('UWYK_SRC', '/tmp/g4cfm_uwyk/src')
CHECKPOINT = 'checkpoints/step_50000_final.pt'
OUT_DIR    = '/Users/furkandanisman/R-PFN/experiments'
MALC_B     = 1000
N_EVAL     = 200
DEVICE     = torch.device('cpu')

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, UWYK_SRC); sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, 'MALC'))

from models.InterventionalPFN import InterventionalPFN
from losses.BarDistribution2D import unpack_pred, fit_malc_inner
from malc_2d import dmalc_2d
from eval_scm_gen import generate_paired_sample_with_raw


ckpt   = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
config = ckpt['config']
J      = config['J']
edges  = ckpt['edges'].to(DEVICE)
edges_np = edges.detach().cpu().numpy()
bin_width = float(edges[1] - edges[0])
centers = 0.5 * (edges_np[:-1] + edges_np[1:])

model = InterventionalPFN(
    num_features=config['num_features'],
    d_model=config['d_model'], depth=config['depth'],
    heads_feat=config['heads'], heads_samp=config['heads'],
    dropout=0.0, output_dim=J*J + 9 + 4,
    hidden_mult=config['hidden_mult'],
    normalize_features=True, normalize_treatment=False,
    use_treatment_in_query=False, use_checkpoint=False,
).to(DEVICE).eval()
model.load_state_dict(ckpt['model_state_dict'])
print("Model loaded", flush=True)

xs = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
ys = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
XX, YY = np.meshgrid(xs, ys, indexing='xy')
eval_pts = np.column_stack([XX.ravel(), YY.ravel()])


# Pick 2 problematic queries from previous run
# seed=4 q=463 (τ_true=+0.66, MALC shifted left of raw)
# seed=4 q=405 (τ_true=+0.47, similar)
targets = [(4, 463), (4, 405)]
scm_data = {}
for scm_seed in {s for s, _ in targets}:
    s = generate_paired_sample_with_raw(scm_seed=scm_seed, idx=0, n_train=1000, n_test=500)
    X_obs  = s['X_obs'].unsqueeze(0).to(DEVICE)
    T_obs  = s['T_obs'].unsqueeze(0).to(DEVICE)
    Y_obs  = s['Y_obs'].unsqueeze(0).to(DEVICE)
    X_intv = s['X_intv'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]
    true_tau = (s['Y_do1'] - s['Y_do0']).reshape(-1).numpy()
    scm_data[scm_seed] = (pred, true_tau)

# Two plotting variants: with .T and without .T so we can see which is correct
def fit_malc(p_mat_np, K, seed_hint, transpose):
    inp = p_mat_np.T if transpose else p_mat_np
    fit = fit_malc_inner(
        inp, edges_np, edges_np,
        K=K, B_fit=MALC_B, B_select=MALC_B,
        seed=seed_hint, parallel=False,
    )
    density = dmalc_2d(fit, eval_pts).reshape(N_EVAL, N_EVAL)
    return density


for scm_seed, q in targets:
    pred, true_tau = scm_data[scm_seed]
    p_mat, *_ = unpack_pred(pred[q], J, bin_width)
    p_mat_np = p_mat.detach().cpu().numpy()
    print(f"seed={scm_seed} q={q}  true_τ={true_tau[q]:+.3f}", flush=True)

    # 2D density from raw p_mat (interpolated onto same grid as MALC eval)
    # p_mat[i, j] = p(Y_do0=centers[i], Y_do1=centers[j])
    # Density_raw(y0, y1) = p_mat[bin(y0), bin(y1)] / bin_width²
    def raw_density_2d():
        d = np.zeros((N_EVAL, N_EVAL))  # d[i, j] = density at (Y_do0=xs[j], Y_do1=ys[i])
        for i in range(N_EVAL):
            for j in range(N_EVAL):
                b0 = int(np.clip((xs[j] - edges_np[0]) / bin_width, 0, J - 1))
                b1 = int(np.clip((ys[i] - edges_np[0]) / bin_width, 0, J - 1))
                # Under my convention: d[i, j] should = p(Y_do0=xs[j], Y_do1=ys[i])
                # = p_mat[b0=Y_do0_bin, b1=Y_do1_bin] / bw²
                d[i, j] = p_mat_np[b0, b1] / (bin_width ** 2)
        return d

    raw_2d = raw_density_2d()

    # MALC fits
    print("  fitting MALC K=2 (transpose)…", flush=True)
    malc_K2_T  = fit_malc(p_mat_np, 2, scm_seed + q, transpose=True)
    print("  fitting MALC K=2 (NO transpose)…", flush=True)
    malc_K2_NT = fit_malc(p_mat_np, 2, scm_seed + q, transpose=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    vmax = max(raw_2d.max(), malc_K2_T.max(), malc_K2_NT.max())
    for ax, arr, title in zip(
        axes,
        [raw_2d, malc_K2_T, malc_K2_NT],
        ['Raw p_mat  (Y_do0=x, Y_do1=y)',
         'MALC K=2, input=p_mat.T',
         'MALC K=2, input=p_mat (no transpose)'],
    ):
        im = ax.imshow(arr, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin='lower',
                        cmap='viridis', vmin=0, vmax=vmax, aspect='equal')
        ax.plot([xs[0], xs[-1]], [ys[0], ys[-1]], 'w--', lw=1, alpha=0.5, label='y=x')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Y_do0')
        ax.set_ylabel('Y_do1')
        plt.colorbar(im, ax=ax, shrink=0.7)
        ax.legend(loc='upper left', fontsize=8)
    fig.suptitle(f"2D density audit  seed={scm_seed} q={q}  τ_true={true_tau[q]:+.3f}", y=1.02)
    fig.tight_layout()
    out = f'{OUT_DIR}/AUDIT_2d_seed{scm_seed}_q{q}.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}", flush=True)

    # Compute τ marginal from each 2D density and plot
    def tau_marginal(density_2d):
        # density_2d[i, j] = density at (Y_do0=xs[j], Y_do1=ys[i])
        tau_grid = np.linspace(ys[0] - xs[-1], ys[-1] - xs[0], 401)
        out = np.zeros_like(tau_grid)
        dy0 = xs[1] - xs[0]; dy1 = ys[1] - ys[0]
        for k, t in enumerate(tau_grid):
            y1_target = xs + t
            valid = (y1_target >= ys[0]) & (y1_target <= ys[-1])
            if not np.any(valid): continue
            col_idx = np.arange(len(xs))[valid]
            row_f  = (y1_target[valid] - ys[0]) / dy1
            row_lo = np.clip(np.floor(row_f).astype(int), 0, len(ys) - 2)
            row_hi = row_lo + 1
            w_hi   = row_f - row_lo; w_lo = 1.0 - w_hi
            f_diag = w_lo * density_2d[row_lo, col_idx] + w_hi * density_2d[row_hi, col_idx]
            out[k] = f_diag.sum() * dy0
        return tau_grid, out

    tau_grid, tau_raw    = tau_marginal(raw_2d)
    _,        tau_malc_T  = tau_marginal(malc_K2_T)
    _,        tau_malc_NT = tau_marginal(malc_K2_NT)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tau_grid, tau_raw,     color='black',    lw=1.7, label='τ marginal from RAW density')
    ax.plot(tau_grid, tau_malc_T,  color='steelblue',lw=1.7, label='τ marginal from MALC (input=p_mat.T)')
    ax.plot(tau_grid, tau_malc_NT, color='crimson',  lw=1.7, ls='--',
            label='τ marginal from MALC (input=p_mat NO transpose)')
    ax.axvline(true_tau[q], color='red', ls='--', lw=1, alpha=0.6)
    ax.set_xlabel('τ (scaled)')
    ax.set_ylabel('p(τ)')
    ax.set_title(f"τ marginal from each 2D density  seed={scm_seed} q={q}  τ_true={true_tau[q]:+.3f}")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    out_t = f'{OUT_DIR}/AUDIT_tau_seed{scm_seed}_q{q}.png'
    fig.savefig(out_t, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_t}", flush=True)

print("Done.", flush=True)
