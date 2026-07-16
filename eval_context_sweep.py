"""
Does the trained model actually use more context to sharpen its predictions?

For a fixed SCM, generate 1 sample (n_train=1000, n_test=500). Then run inference
on all 500 queries with N_context = 100, 250, 500, 1000 (subsampled from the
1000 available context rows).

For each context size, measure:
  - mean and std of E[τ] across queries
  - corr(E[τ], true τ)
  - Var(E[τ]) / Var(true τ)          (how much predicted variance the model captures)
  - Mean L1-asymmetry of p_mat        (does the joint become more arm-distinct?)
  - Mean absolute shift of p_mat argmax from marginal-mean cell (does the mode move?)

If E[τ] correlation with true τ grows with N_context, the model IS extracting
signal from the observational context. If nothing moves with context size, the
model outputs a context-independent joint (which would be a big problem).
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch

UWYK_SRC   = os.environ.get('UWYK_SRC', '/tmp/g4cfm_uwyk/src')
CHECKPOINT = os.environ.get('CHECKPOINT', 'checkpoints/step_50000_final.pt')
SCM_SEED   = int(os.environ.get('SCM_SEED', 2))
CONTEXT_SIZES = [int(x) for x in os.environ.get(
    'CONTEXT_SIZES', '100,250,500,1000,2000').split(',')]
N_QUERY_MAX = int(os.environ.get('N_QUERY_MAX', 500))
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, UWYK_SRC); sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, 'MALC'))

from models.InterventionalPFN import InterventionalPFN
from losses.BarDistribution2D import unpack_pred
from eval_scm_gen import generate_paired_sample_with_raw


# Load model
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
print(f"Model loaded. J={J} bin_width={bin_width:.4f}", flush=True)


def summarize(pred_qs, true_tau, marginal_cell):
    """pred_qs: (Q, J*J+9+4)  →  return dict of summary metrics."""
    Q = pred_qs.shape[0]
    E_tau = np.zeros(Q)
    asym_L1 = np.zeros(Q)
    argmax_shift = np.zeros(Q)   # distance from marginal-mean cell in bins
    for i in range(Q):
        p_mat, *_ = unpack_pred(pred_qs[i], J, bin_width)
        p = p_mat.detach().cpu().numpy()
        E_y0 = (centers[:, None] * p).sum()
        E_y1 = (centers[None, :] * p).sum()
        E_tau[i] = E_y1 - E_y0
        asym_L1[i] = float(np.abs(p - p.T).sum())
        j0, j1 = np.unravel_index(p.argmax(), p.shape)
        argmax_shift[i] = float(np.hypot(j0 - marginal_cell[0], j1 - marginal_cell[1]))

    corr = float(np.corrcoef(E_tau, true_tau)[0, 1]) if E_tau.std() > 1e-9 else float('nan')
    var_ratio = float(E_tau.var() / (true_tau.var() + 1e-12))
    return {
        'E_tau_mean': float(E_tau.mean()),
        'E_tau_std':  float(E_tau.std()),
        'corr':       corr,
        'var_ratio':  var_ratio,
        'asym_L1':    float(asym_L1.mean()),
        'argmax_shift': float(argmax_shift.mean()),
        'E_tau':      E_tau,
    }


# ── Get one SCM with enough context that we can sub-sample ────────────────────
# For sizes > 1000 we need to draw more; for now use n_train = max(CONTEXT_SIZES).
n_ctx_max = max(CONTEXT_SIZES)
print(f"\nGenerating SCM seed={SCM_SEED} with n_train={n_ctx_max}, n_test=500…", flush=True)
t0 = time.time()
sample = generate_paired_sample_with_raw(
    scm_seed=SCM_SEED, idx=0, n_train=n_ctx_max, n_test=500,
)
true_tau_all = (sample['Y_do1'] - sample['Y_do0']).reshape(-1).numpy()
print(f"  affine ymin={sample['ymin']:+.3f} ymax={sample['ymax']:+.3f}", flush=True)
print(f"  true τ (scaled): mean={true_tau_all.mean():+.3f}  std={true_tau_all.std():.3f}  "
      f"min={true_tau_all.min():+.3f}  max={true_tau_all.max():+.3f}", flush=True)

# Marginal-mean cell (for measuring how far the mode moves)
i_marg = int(np.argmin(np.abs(centers - true_tau_all.mean() / 2 - 0)))
marginal_cell = (i_marg, i_marg)

# Restrict queries for speed
n_q = min(N_QUERY_MAX, sample['X_intv'].shape[0])
qidx = np.arange(n_q)
true_tau_q = true_tau_all[qidx]

X_obs_full  = sample['X_obs']    # (n_ctx_max, 50)
T_obs_full  = sample['T_obs']
Y_obs_full  = sample['Y_obs']
X_intv_full = sample['X_intv'][qidx]

# ── Sweep context sizes ───────────────────────────────────────────────────────
print(f"\n{'N_ctx':>6}  {'E_tau mean':>10}  {'std':>7}  {'corr':>6}  {'Var-ratio':>9}  "
      f"{'L1-asym':>7}  {'mode-shift':>10}", flush=True)
print("-" * 72, flush=True)

results = {}
for N in CONTEXT_SIZES:
    if N > X_obs_full.shape[0]:
        print(f"  N={N} skipped (only {X_obs_full.shape[0]} context rows available)",
              flush=True)
        continue
    # Deterministic subsample: first N rows
    X_obs  = X_obs_full[:N].unsqueeze(0).to(DEVICE)
    T_obs  = T_obs_full[:N].unsqueeze(0).to(DEVICE)
    Y_obs  = Y_obs_full[:N].unsqueeze(0).to(DEVICE)
    X_intv = X_intv_full.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]

    r = summarize(pred, true_tau_q, marginal_cell)
    results[N] = r
    print(f"  {N:>4d}   {r['E_tau_mean']:+.4f}    {r['E_tau_std']:.4f}   "
          f"{r['corr']:+.3f}   {r['var_ratio']:.4f}    "
          f"{r['asym_L1']:.4f}   {r['argmax_shift']:.2f}", flush=True)

# Save arrays for inspection
os.makedirs('eval_context_sweep', exist_ok=True)
np.save('eval_context_sweep/true_tau.npy', true_tau_q)
for N, r in results.items():
    np.save(f'eval_context_sweep/E_tau_N{N}.npy', r['E_tau'])
print(f"\nSaved per-context E[τ] arrays to eval_context_sweep/", flush=True)

# ── Interpretation summary ────────────────────────────────────────────────────
sizes = sorted(results.keys())
if len(sizes) >= 2:
    d_corr = results[sizes[-1]]['corr'] - results[sizes[0]]['corr']
    d_var  = results[sizes[-1]]['var_ratio'] - results[sizes[0]]['var_ratio']
    d_mode = results[sizes[-1]]['argmax_shift'] - results[sizes[0]]['argmax_shift']
    print(f"\nΔ from N={sizes[0]} to N={sizes[-1]}:", flush=True)
    print(f"  corr(E[τ], true τ)       : {results[sizes[0]]['corr']:+.3f} → "
          f"{results[sizes[-1]]['corr']:+.3f}   (Δ = {d_corr:+.3f})", flush=True)
    print(f"  Var(E[τ]) / Var(true τ)  : {results[sizes[0]]['var_ratio']:.4f} → "
          f"{results[sizes[-1]]['var_ratio']:.4f}   (Δ = {d_var:+.4f})", flush=True)
    print(f"  Mean argmax shift (bins) : {results[sizes[0]]['argmax_shift']:.2f} → "
          f"{results[sizes[-1]]['argmax_shift']:.2f}   (Δ = {d_mode:+.2f})", flush=True)

print(f"\nDone.  total {time.time() - t0:.1f}s", flush=True)
