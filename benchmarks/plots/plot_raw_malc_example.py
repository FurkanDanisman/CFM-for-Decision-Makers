"""Raw-MALC-only version of MASK_example: joint p(Y_do0, Y_do1) + marginal p(τ).

Same query (IHDP realization 0, query 26) as MASK_example.png, but WITHOUT the
masked row and overlay. Just the raw joint and the raw MALC-smoothed marginal.
"""
import os, sys, warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

DEVICE = torch.device('cpu')
REPO = '/Users/furkandanisman/R-PFN'
CAUSALPFN = '/tmp/causalpfn_full'
OUR_CKPT = '/Users/furkandanisman/R-PFN/checkpoints/step_50000_final.pt'
N_EVAL = 200
QUERY = 26

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

cd, _ = IHDPDataset()[0]

def _to_np(a): return a if isinstance(a, np.ndarray) else a.numpy()
Xtr = _to_np(cd.X_train).astype(np.float32); tt = _to_np(cd.t_train).astype(np.float32).reshape(-1, 1)
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

xs = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
ys = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
XX, YY = np.meshgrid(xs, ys, indexing='xy')
eval_pts = np.column_stack([XX.ravel(), YY.ravel()])
dy0 = xs[1] - xs[0]; dy1 = ys[1] - ys[0]
tau_grid = np.linspace(ys[0] - xs[-1], ys[-1] - xs[0], 401)
dtau = tau_grid[1] - tau_grid[0]

p_mat, *_ = unpack_pred(pred[QUERY], J, bin_width)
p_np = p_mat.detach().cpu().numpy()

fit = fit_malc_inner(p_np.T, edges_np, edges_np, B_fit=100, B_select=100,
                      max_K=3, seed=100 + QUERY, parallel=False)
dens = dmalc_2d(fit, eval_pts).reshape(N_EVAL, N_EVAL)
p_tau = np.zeros_like(tau_grid)
for k, t in enumerate(tau_grid):
    y1 = xs + t; v = (y1 >= ys[0]) & (y1 <= ys[-1])
    if not np.any(v): continue
    col = np.clip(np.searchsorted(xs, xs[v]) - 1, 0, len(xs) - 1)
    rf = (y1[v] - ys[0]) / dy1
    rlo = np.clip(np.floor(rf).astype(int), 0, len(ys) - 2)
    rhi = rlo + 1; whi = rf - rlo; wlo = 1.0 - whi
    f = wlo * dens[rlo, col] + whi * dens[rhi, col]
    p_tau[k] = f.sum() * dy0
s = p_tau.sum() * dtau
if s > 0: p_tau /= s

tau_raw = tau_grid * scale
p_tau_raw = p_tau / scale
mode_raw = tau_raw[p_tau_raw.argmax()]
mean_raw = float((tau_raw * p_tau_raw).sum() * (tau_raw[1] - tau_raw[0]))
true_tau = true_cate[QUERY]

fig, (axJ, axM) = plt.subplots(1, 2, figsize=(13, 5.2),
                                gridspec_kw={'width_ratios': [1.05, 1.4]})

extent = [edges_np[0] * scale, edges_np[-1] * scale,
          edges_np[0] * scale, edges_np[-1] * scale]
im = axJ.imshow(p_np, origin='lower', extent=extent, cmap='viridis', aspect='auto')
axJ.plot([edges_np[0] * scale, edges_np[-1] * scale],
          [edges_np[0] * scale, edges_np[-1] * scale],
          'r--', lw=1, alpha=0.6, label=r'$Y_{do0}=Y_{do1}$')
axJ.set_title(r'Raw joint  $p(Y_{do0}, Y_{do1})$', fontsize=12)
axJ.set_xlabel(r'$Y_{do0}$'); axJ.set_ylabel(r'$Y_{do1}$')
axJ.legend(fontsize=9, loc='upper left')
plt.colorbar(im, ax=axJ, fraction=0.046, pad=0.02)

axM.plot(tau_raw, p_tau_raw, color='steelblue', lw=2, label='MALC-smoothed  $p(\\tau)$')
axM.axvline(mode_raw, color='steelblue', ls='--', lw=1.5,
            label=f'MALC-mode = {mode_raw:+.2f}')
axM.axvline(mean_raw, color='steelblue', ls=':', lw=1.5,
            label=f'MALC-mean = {mean_raw:+.2f}')
axM.axvline(true_tau, color='red', ls='--', lw=1.5,
            label=f'true CATE = {true_tau:+.2f}')
axM.axvline(0, color='k', lw=0.7, alpha=0.4)
axM.set_title(r'Raw MALC marginal  $p(\tau)$', fontsize=12)
axM.set_xlabel(r'$\tau$ (treatment effect)'); axM.set_ylabel(r'$p(\tau)$')
axM.grid(alpha=0.3); axM.legend(fontsize=10, loc='upper right')

fig.suptitle(f'Raw MALC estimate — IHDP realization 0, query {QUERY}',
              fontsize=13, y=1.0)
fig.tight_layout()
out_path = os.path.join(REPO, 'benchmarks', 'plots', 'RAW_MALC_example.png')
fig.savefig(out_path, dpi=140, bbox_inches='tight')
print(f'Saved: {out_path}', flush=True)
