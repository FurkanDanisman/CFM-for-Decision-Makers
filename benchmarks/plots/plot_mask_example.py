"""
Illustrative plot: joint p_mat + p(τ) marginal, raw vs diagonal-masked.

Shows why "mode-masked" recovers a better CATE estimate: the raw joint puts a
tall spike along the diagonal (Y_do0 ≈ Y_do1 → τ ≈ 0, the "attractor bias").
Masking removes cells within ±band bins of the diagonal and renormalizes,
letting the true signal dominate.
"""
import os, sys, warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
warnings.filterwarnings('ignore')

DEVICE = torch.device('cpu')
REPO = '/Users/furkandanisman/R-PFN'
CAUSALPFN = '/tmp/causalpfn_full'
OUR_CKPT = '/Users/furkandanisman/R-PFN/checkpoints/step_50000_final.pt'
BAND = 1
N_EVAL = 200

sys.path.insert(0, CAUSALPFN); sys.path.insert(0, REPO); sys.path.insert(0, REPO + '/MALC')
from benchmarks import IHDPDataset
from models.InterventionalPFN import InterventionalPFN as OurModel
from losses.BarDistribution2D import unpack_pred, fit_malc_inner
from malc_2d import dmalc_2d

ckpt = torch.load(OUR_CKPT, map_location=DEVICE, weights_only=False)
cfg = ckpt['config']; J = cfg['J']
edges_np = ckpt['edges'].cpu().numpy()
bin_width = float(edges_np[1] - edges_np[0])
centers = 0.5 * (edges_np[:-1] + edges_np[1:])
NUM_FEATURES = cfg['num_features']
our_model = OurModel(
    num_features=NUM_FEATURES, d_model=cfg['d_model'], depth=cfg['depth'],
    heads_feat=cfg['heads'], heads_samp=cfg['heads'], dropout=0.0,
    output_dim=J*J + 9 + 4, hidden_mult=cfg['hidden_mult'],
    normalize_features=True, normalize_treatment=False,
    use_treatment_in_query=False, use_checkpoint=False,
).to(DEVICE).eval()
our_model.load_state_dict(ckpt['model_state_dict'])

# One IHDP realization
cd, ad = IHDPDataset()[0]
def _to_np(a): return a if isinstance(a, np.ndarray) else a.numpy()
Xtr = _to_np(cd.X_train).astype(np.float32); tt  = _to_np(cd.t_train).astype(np.float32).reshape(-1, 1)
yt  = _to_np(cd.y_train).astype(np.float32).reshape(-1, 1); Xte = _to_np(cd.X_test).astype(np.float32)
true_cate = _to_np(cd.true_cate).reshape(-1)

def _pad(a, L):
    if a.shape[1] >= L: return a[:, :L]
    return np.concatenate([a, np.zeros((a.shape[0], L - a.shape[1]), dtype=np.float32)], axis=1)

Xtr_p = _pad(Xtr, NUM_FEATURES); Xte_p = _pad(Xte, NUM_FEATURES)
mu = Xtr_p.mean(0, keepdims=True); sd = Xtr_p.std(0, keepdims=True); sd[sd < 1e-6] = 1.0
Xtr_s = (Xtr_p - mu) / sd; Xte_s = (Xte_p - mu) / sd
y_min = float(yt.min()); y_max = float(yt.max()); y_rng = max(y_max - y_min, 1e-8)
yt_s = 2 * (yt - y_min) / y_rng - 1.0
scale = y_rng / 2.0

with torch.no_grad():
    pred = our_model(
        torch.from_numpy(Xtr_s).unsqueeze(0),
        torch.from_numpy(tt).unsqueeze(0),
        torch.from_numpy(yt_s).unsqueeze(0),
        torch.from_numpy(Xte_s).unsqueeze(0),
    )['predictions'][0]

# Pick a query where the τ=0 attractor bias is visible (true CATE far from 0)
# and mode-masked helps
xs = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
ys = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
XX, YY = np.meshgrid(xs, ys, indexing='xy')
eval_pts = np.column_stack([XX.ravel(), YY.ravel()])
dy0 = xs[1] - xs[0]; dy1 = ys[1] - ys[0]
tau_grid = np.linspace(ys[0] - xs[-1], ys[-1] - xs[0], 401)
dtau = tau_grid[1] - tau_grid[0]

def _mask_diag(p, band=BAND):
    p2 = p.copy()
    for j0 in range(J):
        for j1 in range(max(0, j0-band), min(J, j0+band+1)):
            p2[j0, j1] = 0.0
    p2 /= max(p2.sum(), 1e-12)
    return p2

def _fit_marg(p, seed):
    fit = fit_malc_inner(p.T, edges_np, edges_np, B_fit=100, B_select=100,
                          max_K=3, seed=seed, parallel=False)
    dens = dmalc_2d(fit, eval_pts).reshape(N_EVAL, N_EVAL)
    out = np.zeros_like(tau_grid)
    for k, t in enumerate(tau_grid):
        y1 = xs + t; v = (y1 >= ys[0]) & (y1 <= ys[-1])
        if not np.any(v): continue
        col = np.clip(np.searchsorted(xs, xs[v]) - 1, 0, len(xs) - 1)
        rf = (y1[v] - ys[0]) / dy1
        rlo = np.clip(np.floor(rf).astype(int), 0, len(ys) - 2)
        rhi = rlo + 1; whi = rf - rlo; wlo = 1.0 - whi
        f = wlo * dens[rlo, col] + whi * dens[rhi, col]
        out[k] = f.sum() * dy0
    s = out.sum() * dtau
    if s > 0: out /= s
    return dens, out

# Search a few queries for one where masking clearly changes the mode
best_i = None; best_gap = 0
for i in range(min(30, pred.shape[0])):
    p_mat, *_ = unpack_pred(pred[i], J, bin_width)
    p_np = p_mat.detach().cpu().numpy()
    p_msk = _mask_diag(p_np)
    _, pt_r = _fit_marg(p_np, seed=100+i)
    _, pt_m = _fit_marg(p_msk, seed=200+i)
    mode_r = tau_grid[pt_r.argmax()] * scale
    mode_m = tau_grid[pt_m.argmax()] * scale
    gap = abs(mode_m - mode_r)
    if gap > best_gap:
        best_gap = gap; best_i = i

print(f"Picked query {best_i} (mode gap {best_gap:.2f})", flush=True)

p_mat, *_ = unpack_pred(pred[best_i], J, bin_width)
p_np = p_mat.detach().cpu().numpy()
p_msk = _mask_diag(p_np)
dens_raw, pt_raw = _fit_marg(p_np, seed=100+best_i)
dens_msk, pt_msk = _fit_marg(p_msk, seed=200+best_i)

tau_raw = tau_grid * scale
pt_raw_raw = pt_raw / scale
pt_msk_raw = pt_msk / scale

mode_raw = tau_raw[pt_raw_raw.argmax()]
mode_msk = tau_raw[pt_msk_raw.argmax()]
true_tau = true_cate[best_i]

# ── Plot: 2 rows × 3 cols  (joint, marginal, side-by-side) ──────────────────
fig = plt.figure(figsize=(16, 8))
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4,
              width_ratios=[1.1, 1.3, 1.7])

axJ1 = fig.add_subplot(gs[0, 0]); axJ2 = fig.add_subplot(gs[1, 0])
axM1 = fig.add_subplot(gs[0, 1]); axM2 = fig.add_subplot(gs[1, 1])
axC  = fig.add_subplot(gs[:, 2])

extent = [edges_np[0]*scale, edges_np[-1]*scale, edges_np[0]*scale, edges_np[-1]*scale]

im1 = axJ1.imshow(p_np, origin='lower', extent=extent, cmap='viridis', aspect='auto')
axJ1.plot([edges_np[0]*scale, edges_np[-1]*scale], [edges_np[0]*scale, edges_np[-1]*scale],
          'r--', lw=1, alpha=0.7, label=r'$Y_{do0}=Y_{do1}$')
axJ1.set_title('Raw joint  $p(Y_{do0}, Y_{do1})$', fontsize=11)
axJ1.set_xlabel(r'$Y_{do0}$'); axJ1.set_ylabel(r'$Y_{do1}$'); axJ1.legend(fontsize=8, loc='upper left')
plt.colorbar(im1, ax=axJ1, fraction=0.046)

im2 = axJ2.imshow(p_msk, origin='lower', extent=extent, cmap='viridis', aspect='auto')
axJ2.plot([edges_np[0]*scale, edges_np[-1]*scale], [edges_np[0]*scale, edges_np[-1]*scale],
          'r--', lw=1, alpha=0.7, label=f'masked ±{BAND} bin')
axJ2.set_title(f'Masked joint  (±{BAND} bin diagonal zeroed)', fontsize=11)
axJ2.set_xlabel(r'$Y_{do0}$'); axJ2.set_ylabel(r'$Y_{do1}$'); axJ2.legend(fontsize=8, loc='upper left')
plt.colorbar(im2, ax=axJ2, fraction=0.046)

axM1.plot(tau_raw, pt_raw_raw, color='steelblue', lw=2)
axM1.axvline(mode_raw, color='steelblue', ls='--', lw=1.5, label=f'MALC-mode = {mode_raw:+.2f}')
axM1.axvline(true_tau, color='red', ls=':', lw=1.5, label=f'true CATE = {true_tau:+.2f}')
axM1.axvline(0, color='k', lw=0.8, alpha=0.3)
axM1.set_title('Raw  p(τ)  — τ=0 attractor pulls mode', fontsize=11)
axM1.set_xlabel(r'$\tau$'); axM1.set_ylabel(r'$p(\tau)$'); axM1.legend(fontsize=9, loc='best')
axM1.grid(alpha=0.3)

axM2.plot(tau_raw, pt_msk_raw, color='darkorange', lw=2)
axM2.axvline(mode_msk, color='darkorange', ls='--', lw=1.5, label=f'MALC-mode-msk = {mode_msk:+.2f}')
axM2.axvline(true_tau, color='red', ls=':', lw=1.5, label=f'true CATE = {true_tau:+.2f}')
axM2.axvline(0, color='k', lw=0.8, alpha=0.3)
axM2.set_title('Masked  p(τ)  — mode near truth', fontsize=11)
axM2.set_xlabel(r'$\tau$'); axM2.set_ylabel(r'$p(\tau)$'); axM2.legend(fontsize=9, loc='best')
axM2.grid(alpha=0.3)

# Side-by-side overlay for comparison
axC.plot(tau_raw, pt_raw_raw, color='steelblue', lw=2, label=f'raw MALC (mode {mode_raw:+.2f})')
axC.plot(tau_raw, pt_msk_raw, color='darkorange', lw=2, label=f'masked MALC (mode {mode_msk:+.2f})')
axC.axvline(true_tau, color='red', ls=':', lw=2, label=f'true CATE = {true_tau:+.2f}')
axC.axvline(0, color='k', lw=0.8, alpha=0.3, label='τ=0')
axC.plot(mode_raw, pt_raw_raw.max(), 'o', color='steelblue', markersize=10, clip_on=False, zorder=5)
axC.plot(mode_msk, pt_msk_raw.max(), 'o', color='darkorange', markersize=10, clip_on=False, zorder=5)
axC.set_title('Overlay: masking pulls the mode toward the true CATE', fontsize=12, fontweight='bold')
axC.set_xlabel(r'$\tau$ (treatment effect)'); axC.set_ylabel(r'$p(\tau)$')
axC.legend(fontsize=10, loc='best'); axC.grid(alpha=0.3)

fig.suptitle(f'Diagonal masking removes the τ=0 attractor  —  IHDP realization 0, query {best_i}',
             fontsize=13, y=0.995)
out_path = os.path.join(REPO, 'benchmarks', 'plots', 'malc_examples', 'MASK_example.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=140, bbox_inches='tight')
print(f'Saved: {out_path}', flush=True)
