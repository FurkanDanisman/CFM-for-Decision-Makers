"""
UWYK vs our model — error to true τ as N_context increases.

UWYK: their full_conditioned_model checkpoint. For each query, call twice
(T_intv=0 then T_intv=1) to get E[Y|do(T=0), X, D] and E[Y|do(T=1), X, D].
Then τ_UWYK = E[Y|T=1] − E[Y|T=0].

Ours: E[τ] from the joint p_mat directly (which by linearity equals same thing
using our joint marginals; the interesting question is whether UWYK's actual
trained weights identify τ more accurately than our paired-head weights).

Report per-SCM and pooled:
  - median |E[τ]_UWYK − τ_true| vs N_context
  - median |E[τ]_ours − τ_true| vs N_context
  - correlation with τ_true vs N_context

Uses cached SCM samples from experiments/scm_cache (same SCMs both models see).
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch
import matplotlib.pyplot as plt

UWYK_SRC   = os.environ.get('UWYK_SRC', '/tmp/g4cfm_uwyk/src')
UWYK_CKPT  = '/tmp/g4cfm_uwyk/experiments/checkpoints/full_conditioned_model'
OUR_CKPT   = 'checkpoints/step_50000_final.pt'
OUT_DIR    = '/Users/furkandanisman/R-PFN/experiments'
SCM_CACHE  = os.path.join(OUT_DIR, 'scm_cache')
N_SCM      = 4
CONTEXT_SIZES = [100, 250, 500, 1000, 2000]
N_QUERIES_PER_SCM = 120     # subsample so total runtime is manageable
DEVICE     = torch.device('cpu')

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, UWYK_SRC); sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, 'MALC'))

from models.InterventionalPFN import InterventionalPFN as OurModel
from losses.BarDistribution2D import unpack_pred
from eval_scm_gen import generate_paired_sample_with_raw

# Load UWYK's models module directly from its file (our `models` package shadows it)
import importlib.util
_uwyk_mod_path = os.path.join(UWYK_SRC, 'models', 'GraphConditionedInterventionalPFN_sklearn.py')
_spec = importlib.util.spec_from_file_location('_uwyk_gcip_sklearn', _uwyk_mod_path)
_uwyk_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_uwyk_mod)
GraphConditionedInterventionalPFNSklearn = _uwyk_mod.GraphConditionedInterventionalPFNSklearn


def _log(m, t0=None):
    t = f"[{time.time() - t0:6.1f}s]" if t0 is not None else "[  0.0s]"
    print(f"{t} {m}", flush=True)


t0 = time.time()

# ── Our model ────────────────────────────────────────────────────────────────
_log(f"Loading OUR model {OUR_CKPT}")
ckpt = torch.load(OUR_CKPT, map_location=DEVICE, weights_only=False)
config = ckpt['config']
J = config['J']; edges_np = ckpt['edges'].to(DEVICE).cpu().numpy()
bin_width = float(edges_np[1] - edges_np[0])
centers = 0.5 * (edges_np[:-1] + edges_np[1:])

our_model = OurModel(
    num_features=config['num_features'], d_model=config['d_model'],
    depth=config['depth'], heads_feat=config['heads'], heads_samp=config['heads'],
    dropout=0.0, output_dim=J*J + 9 + 4, hidden_mult=config['hidden_mult'],
    normalize_features=True, normalize_treatment=False,
    use_treatment_in_query=False, use_checkpoint=False,
).to(DEVICE).eval()
our_model.load_state_dict(ckpt['model_state_dict'])

# ── UWYK model ───────────────────────────────────────────────────────────────
_log(f"Loading UWYK checkpoint from {UWYK_CKPT}")
# UWYK checkpoint pre-dates torch 2.6's weights_only=True default; patch during load.
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load
try:
    uwyk = GraphConditionedInterventionalPFNSklearn(
        config_path=os.path.join(UWYK_CKPT, 'config.yaml'),
        checkpoint_path=os.path.join(UWYK_CKPT, 'model.pt'),
        device='cpu', verbose=True,
    ).load()
finally:
    torch.load = _orig_torch_load
_log("UWYK loaded.", t0)


def our_E_tau(pred_row):
    p_mat, *_ = unpack_pred(pred_row, J, bin_width)
    p = p_mat.detach().cpu().numpy()
    # Convention: p[j0, j1] = p(Y_do0=j0, Y_do1=j1). E[τ] = E[Y_do1] - E[Y_do0].
    E_y0 = (centers[:, None] * p).sum()
    E_y1 = (centers[None, :] * p).sum()
    return float(E_y1 - E_y0)


def get_scm(seed, n_train=2000, n_test=500):
    p = os.path.join(SCM_CACHE, f'scm_seed{seed}_n{n_train}.pt')
    if os.path.exists(p): return torch.load(p, weights_only=False)
    s = generate_paired_sample_with_raw(scm_seed=seed, idx=0, n_train=n_train, n_test=n_test)
    torch.save(s, p); return s


def make_uwyk_adj(n_features):
    """UWYK needs (L+2, L+2) adjacency. Positions 0=T, 1=Y, 2..L+1=X.
    Use all-zeros = 'no known edges' (unconditional-equivalent input)."""
    return np.zeros((n_features + 2, n_features + 2), dtype=np.float32)


# ── Run ──────────────────────────────────────────────────────────────────────
_log("\n### Comparison across N_context sizes ###", t0)

# collect per-(N, scm_seed) errors
per_run_ours = {N: [] for N in CONTEXT_SIZES}
per_run_uwyk = {N: [] for N in CONTEXT_SIZES}
true_all     = {N: [] for N in CONTEXT_SIZES}

for scm_seed in range(N_SCM):
    _log(f"\nSCM seed={scm_seed}", t0)
    s = get_scm(scm_seed)
    true_tau_scaled = (s['Y_do1'] - s['Y_do0']).reshape(-1).numpy()
    rng = np.random.default_rng(scm_seed)
    q_idx = rng.choice(len(true_tau_scaled), size=min(N_QUERIES_PER_SCM, len(true_tau_scaled)),
                       replace=False)
    true_taus_q = true_tau_scaled[q_idx]

    X_obs_full  = s['X_obs']       # (n_train, 50)  torch
    T_obs_full  = s['T_obs']       # (n_train, 1)
    Y_obs_full  = s['Y_obs']       # (n_train, 1)
    X_intv_q    = s['X_intv'][q_idx]      # (Q, 50)

    n_feat = 50   # padded features count
    adj = make_uwyk_adj(n_feat)

    for N in CONTEXT_SIZES:
        if N > X_obs_full.shape[0]:
            _log(f"  N={N} skipped", t0); continue
        # ---- OURS ----
        X_obs = X_obs_full[:N].unsqueeze(0).to(DEVICE)
        T_obs = T_obs_full[:N].unsqueeze(0).to(DEVICE)
        Y_obs = Y_obs_full[:N].unsqueeze(0).to(DEVICE)
        X_intv = X_intv_q.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = our_model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]
        our_Etau = np.array([our_E_tau(pred[i]) for i in range(pred.shape[0])])

        # ---- UWYK ----
        # UWYK wants numpy, no batch dim (batched=False)
        X_obs_np = X_obs_full[:N].numpy()             # (N, 50)
        T_obs_np = T_obs_full[:N].reshape(-1).numpy() # (N,)
        Y_obs_np = Y_obs_full[:N].reshape(-1).numpy() # (N,)
        X_intv_np = X_intv_q.numpy()                  # (Q, 50)
        Q = X_intv_np.shape[0]
        T0 = np.zeros(Q, dtype=np.float32)
        T1 = np.ones(Q, dtype=np.float32)
        q_t0 = time.time()
        try:
            y_pred_T0 = uwyk.predict(
                X_obs_np, T_obs_np, Y_obs_np,
                X_intv_np, T0, adj, prediction_type="mean", batched=False,
            )
            y_pred_T1 = uwyk.predict(
                X_obs_np, T_obs_np, Y_obs_np,
                X_intv_np, T1, adj, prediction_type="mean", batched=False,
            )
            uwyk_Etau = np.asarray(y_pred_T1) - np.asarray(y_pred_T0)
        except Exception as e:
            _log(f"  N={N}: UWYK predict raised {type(e).__name__}: {str(e)[:200]}", t0)
            uwyk_Etau = np.full(Q, np.nan)

        err_ours = np.abs(our_Etau  - true_taus_q)
        err_uwyk = np.abs(uwyk_Etau - true_taus_q) if not np.isnan(uwyk_Etau).any() else np.full(Q, np.nan)

        per_run_ours[N].append((scm_seed, err_ours, our_Etau))
        per_run_uwyk[N].append((scm_seed, err_uwyk, uwyk_Etau))
        true_all[N].append(true_taus_q)

        med_o = float(np.median(err_ours))
        med_u = float(np.nanmedian(err_uwyk)) if not np.isnan(err_uwyk).all() else float('nan')
        _log(f"  N={N:>4d}  median|E[τ]-τ_true|  ours={med_o:.4f}  UWYK={med_u:.4f}  "
             f"UWYK-took {time.time() - q_t0:.1f}s", t0)


# ── Aggregate + plot ─────────────────────────────────────────────────────────
_log("\nAggregating…", t0)
Ns = CONTEXT_SIZES
ours_med  = []; ours_p25 = []; ours_p75 = []; ours_corr = []
uwyk_med  = []; uwyk_p25 = []; uwyk_p75 = []; uwyk_corr = []

for N in Ns:
    if not per_run_ours[N]:
        for lst in (ours_med, ours_p25, ours_p75, ours_corr,
                    uwyk_med, uwyk_p25, uwyk_p75, uwyk_corr):
            lst.append(np.nan)
        continue
    errs_o = np.concatenate([e for (_, e, _) in per_run_ours[N]])
    errs_u = np.concatenate([e for (_, e, _) in per_run_uwyk[N]])
    Et_o   = np.concatenate([Et for (_, _, Et) in per_run_ours[N]])
    Et_u   = np.concatenate([Et for (_, _, Et) in per_run_uwyk[N]])
    true_c = np.concatenate(true_all[N])

    ours_med.append(np.median(errs_o))
    ours_p25.append(np.quantile(errs_o, 0.25)); ours_p75.append(np.quantile(errs_o, 0.75))
    ours_corr.append(float(np.corrcoef(Et_o, true_c)[0, 1]) if Et_o.std() > 1e-9 else np.nan)

    m = ~np.isnan(errs_u)
    if m.any():
        uwyk_med.append(np.median(errs_u[m]))
        uwyk_p25.append(np.quantile(errs_u[m], 0.25)); uwyk_p75.append(np.quantile(errs_u[m], 0.75))
        uwyk_corr.append(float(np.corrcoef(Et_u[m], true_c[m])[0, 1]) if Et_u[m].std() > 1e-9 else np.nan)
    else:
        uwyk_med.append(np.nan); uwyk_p25.append(np.nan); uwyk_p75.append(np.nan); uwyk_corr.append(np.nan)

_log(f"\nSummary:", t0)
_log(f"  N_context   OURS median-err (p25-p75)     UWYK median-err (p25-p75)   OURS corr  UWYK corr", t0)
for i, N in enumerate(Ns):
    _log(f"  {N:>5d}       {ours_med[i]:.4f} ({ours_p25[i]:.3f}-{ours_p75[i]:.3f})     "
         f"{uwyk_med[i]:.4f} ({uwyk_p25[i]:.3f}-{uwyk_p75[i]:.3f})     "
         f"{ours_corr[i]:+.3f}     {uwyk_corr[i]:+.3f}", t0)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(Ns, ours_med, 'o-', color='steelblue', lw=2, label='OURS median error')
axes[0].fill_between(Ns, ours_p25, ours_p75, color='steelblue', alpha=0.2)
axes[0].plot(Ns, uwyk_med, 's--', color='crimson', lw=2, label='UWYK median error')
axes[0].fill_between(Ns, uwyk_p25, uwyk_p75, color='crimson', alpha=0.2)
axes[0].set_xscale('log')
axes[0].set_xlabel('N_context'); axes[0].set_ylabel('|E[τ] − τ_true|  (scaled)')
axes[0].set_title(f'Error vs N_context\n'
                  f'{N_SCM} SCMs × {N_QUERIES_PER_SCM} queries each = {N_SCM*N_QUERIES_PER_SCM} total per point')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(Ns, ours_corr, 'o-', color='steelblue', lw=2, label='OURS corr')
axes[1].plot(Ns, uwyk_corr, 's--', color='crimson', lw=2, label='UWYK corr')
axes[1].axhline(0, color='gray', ls=':', lw=0.8)
axes[1].axhline(1, color='green', ls=':', lw=0.8)
axes[1].set_xscale('log')
axes[1].set_xlabel('N_context'); axes[1].set_ylabel('Pearson correlation with τ_true')
axes[1].set_title('Correlation vs N_context')
axes[1].legend(); axes[1].grid(alpha=0.3)

fig.suptitle('UWYK vs OURS — does either shrink error as N grows?', y=1.02)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/UWYK_vs_OURS.png', dpi=140, bbox_inches='tight')
plt.close(fig)
_log(f"Saved: {OUT_DIR}/UWYK_vs_OURS.png", t0)

# Save arrays
np.savez(f'{OUT_DIR}/UWYK_vs_OURS.npz',
         Ns=np.array(Ns),
         ours_med=np.array(ours_med), ours_p25=np.array(ours_p25), ours_p75=np.array(ours_p75),
         uwyk_med=np.array(uwyk_med), uwyk_p25=np.array(uwyk_p25), uwyk_p75=np.array(uwyk_p75),
         ours_corr=np.array(ours_corr), uwyk_corr=np.array(uwyk_corr),
)
_log(f"\nDone. Total {time.time() - t0:.1f}s", t0)
