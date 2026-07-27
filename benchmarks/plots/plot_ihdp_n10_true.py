"""IHDP demonstration — TRUE potential-outcome and TE distributions.

Loads the raw IHDP NPZ (`ihdp_npci_1-100.{train,test}.npz`) that CausalPFN
ships to get the noiseless response surfaces μ0(x), μ1(x) per test unit and
the noise std σ (estimated from training-side residuals yf − μ_factual).
Under Hill (2011)'s DGP the per-unit potential outcomes are
Y_do0|x ~ N(μ0(x), σ²) and Y_do1|x ~ N(μ1(x), σ²), so the per-unit
treatment effect under the DGP's independent-noise coupling is
τ|x ~ N(μ1(x) − μ0(x), 2σ²).

Uses the same context (first N_CONTEXT rows), the same normaliser
(y_min, y_rng from the training set), and the same query-index selector
(percentile of true_cate_scaled) as the sibling scripts — so the ten
queries here match Ours / UWYK-NoAnc / comono panels exactly.

Output:
  IHDP-TRUE-TE/
    ihdp_n10_marginals.png
    ihdp_n10_te.png
    ihdp_n10_ot.png
    cache.npz
"""
from __future__ import annotations
import os, sys, types
import numpy as np
import matplotlib.pyplot as plt

_HERE   = os.path.dirname(os.path.abspath(__file__))
_BENCH  = os.path.dirname(_HERE)
_REPO   = os.environ.get('REPO', os.path.dirname(_BENCH))
_OUTDIR = os.path.join(_HERE, 'ihdp_n10', 'IHDP-TRUE-TE')
os.makedirs(_OUTDIR, exist_ok=True)

CAUSALPFN   = os.environ.get('CAUSALPFN', '')
DOPFN       = os.environ.get('DOPFN', '')
REALIZATION = int(os.environ.get('REALIZATION', 0))
N_CONTEXT   = int(os.environ.get('N_CONTEXT', 200))
N_QUERIES   = int(os.environ.get('N_QUERIES', 10))

if not (os.path.isdir(CAUSALPFN) and os.path.isdir(DOPFN)):
    print('[skip] IHDP-TRUE-TE needs CAUSALPFN and DOPFN paths')
    print(f'   CAUSALPFN = {CAUSALPFN!r}  exists={os.path.isdir(CAUSALPFN)}')
    print(f'   DOPFN     = {DOPFN!r}      exists={os.path.isdir(DOPFN)}')
    sys.exit(0)

# ── Do-PFN datasets shim + CausalPFN benchmarks ─────────────────────────
sys.path.insert(0, DOPFN)
ds_mod = types.ModuleType('datasets')
with open(os.path.join(DOPFN, 'datasets/__init__.py')) as fp:
    _src = fp.read().split('def load_semi_real')[0]
exec(_src, ds_mod.__dict__)
sys.modules['datasets'] = ds_mod
sys.path.insert(0, CAUSALPFN)

from benchmarks import IHDPDataset
cd, ad = IHDPDataset()[REALIZATION]
X_train_full = cd.X_train.numpy() if hasattr(cd.X_train, 'numpy') else np.asarray(cd.X_train)
t_train_full = cd.t_train.numpy() if hasattr(cd.t_train, 'numpy') else np.asarray(cd.t_train)
y_train_full = cd.y_train.numpy() if hasattr(cd.y_train, 'numpy') else np.asarray(cd.y_train)
true_cate    = cd.true_cate.numpy() if hasattr(cd.true_cate, 'numpy') else np.asarray(cd.true_cate)
true_cate    = true_cate.reshape(-1)

# Match the other scripts' scaling
y_min = float(y_train_full.min()); y_max = float(y_train_full.max())
y_rng = max(y_max - y_min, 1e-6)
true_cate_scaled = true_cate * (2.0 / y_rng)

order = np.argsort(true_cate_scaled)
qs = np.linspace(0.05, 0.95, N_QUERIES)
QUERY_IDXS = order[(qs * (len(true_cate_scaled) - 1)).astype(int)].tolist()
print(f'[data] IHDP r={REALIZATION}  N_context={N_CONTEXT}  '
      f'queries={QUERY_IDXS}', flush=True)


# ── Raw NPZ → μ0(x), μ1(x) per test unit + σ from train residuals ───────
_ihdp_dir = os.path.join(CAUSALPFN, 'benchmarks', 'IHDP')
if not os.path.isdir(_ihdp_dir):
    # CausalPFN installed as a package has benchmarks/ under the package root
    _ihdp_dir = os.path.join(CAUSALPFN, 'IHDP')
train_npz = np.load(os.path.join(_ihdp_dir, 'ihdp_npci_1-100.train.npz'))
test_npz  = np.load(os.path.join(_ihdp_dir, 'ihdp_npci_1-100.test.npz'))

# All arrays are (n_units, 100 realisations); the loader takes realisation
# index REALIZATION.
mu0_test = test_npz['mu0'][..., REALIZATION].astype(np.float32).reshape(-1)
mu1_test = test_npz['mu1'][..., REALIZATION].astype(np.float32).reshape(-1)
yf_train = train_npz['yf'][..., REALIZATION].astype(np.float32).reshape(-1)
t_train  = train_npz['t' ][..., REALIZATION].astype(np.float32).reshape(-1)
mu0_train = train_npz['mu0'][..., REALIZATION].astype(np.float32).reshape(-1)
mu1_train = train_npz['mu1'][..., REALIZATION].astype(np.float32).reshape(-1)

mu_factual_train = np.where(t_train > 0.5, mu1_train, mu0_train)
sigma_orig = float(np.std(yf_train - mu_factual_train, ddof=1))
print(f'[sigma] estimated σ (original units) = {sigma_orig:.3f}', flush=True)

# Scale μ, σ to the model's [-1, 1] Y range
mu0_scaled  = (mu0_test - y_min) / y_rng * 2.0 - 1.0
mu1_scaled  = (mu1_test - y_min) / y_rng * 2.0 - 1.0
sigma_scaled = sigma_orig * (2.0 / y_rng)
print(f'[sigma] σ_scaled = {sigma_scaled:.4f}', flush=True)


# ── Grids matching sibling scripts ──────────────────────────────────────
edges       = np.linspace(-1.5, 1.5, 101)
centers     = 0.5 * (edges[:-1] + edges[1:])
bin_width   = float(centers[1] - centers[0])
tau_edges   = np.linspace(-3.0, 3.0, 601)
tau_centers = 0.5 * (tau_edges[:-1] + tau_edges[1:])


def _gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2.0 * np.pi))


p_y0_true = np.stack([_gauss(centers, mu0_scaled[q], sigma_scaled)
                       for q in QUERY_IDXS])
p_y1_true = np.stack([_gauss(centers, mu1_scaled[q], sigma_scaled)
                       for q in QUERY_IDXS])
p_taus_true = np.stack([
    _gauss(tau_centers, mu1_scaled[q] - mu0_scaled[q],
            np.sqrt(2.0) * sigma_scaled)
    for q in QUERY_IDXS
])


# ── Figure: true marginals ──────────────────────────────────────────────
n_cols = 5 if N_QUERIES == 10 else 3
n_rows = (N_QUERIES + n_cols - 1) // n_cols
palette = {'do0': '#2E7DAF', 'do1': '#7B3E9E'}

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.6 * n_rows),
                          squeeze=False)
for k, q in enumerate(QUERY_IDXS):
    ax = axes[k // n_cols][k % n_cols]
    ax.plot(centers, p_y0_true[k], color=palette['do0'], lw=1.9, label=r'true $p(Y_{do0})$')
    ax.plot(centers, p_y1_true[k], color=palette['do1'], lw=1.9, label=r'true $p(Y_{do1})$')
    ax.plot(mu0_scaled[q], _gauss(mu0_scaled[q], mu0_scaled[q], sigma_scaled),
             'o', color=palette['do0'], markersize=9, markeredgecolor='white',
             markeredgewidth=1.0, zorder=5)
    ax.plot(mu1_scaled[q], _gauss(mu1_scaled[q], mu1_scaled[q], sigma_scaled),
             'o', color=palette['do1'], markersize=9, markeredgecolor='white',
             markeredgewidth=1.0, zorder=5)
    ax.set_title(f'query {q}   $\\tau_{{true}}$={true_cate_scaled[q]:+.2f}',
                  fontsize=10)
    if k // n_cols == n_rows - 1: ax.set_xlabel(r'$Y$  (scaled)')
    if k %  n_cols == 0:          ax.set_ylabel('density')
    ax.grid(alpha=0.25)
    if k == 0: ax.legend(fontsize=9, loc='upper right')
for k in range(N_QUERIES, n_rows * n_cols):
    axes[k // n_cols][k % n_cols].set_visible(False)
fig.suptitle(f'IHDP r={REALIZATION}   TRUE per-query marginals at N={N_CONTEXT}   '
              f'(σ={sigma_scaled:.3f} scaled)',
              fontsize=12, y=0.999)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out = os.path.join(_OUTDIR, 'ihdp_n10_marginals.png')
fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
print(f'[save] {out}')


# ── Figure: true per-query TE distributions ─────────────────────────────
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.6 * n_rows),
                          squeeze=False)
tau_step = tau_centers[1] - tau_centers[0]
for k, q in enumerate(QUERY_IDXS):
    ax = axes[k // n_cols][k % n_cols]
    E_tau = float(mu1_scaled[q] - mu0_scaled[q])
    ax.fill_between(tau_centers, p_taus_true[k], alpha=0.25, color='red')
    ax.plot(tau_centers, p_taus_true[k], color='red', lw=2.0,
             label=f'true $p(\\tau)$  E={E_tau:+.2f}')
    ax.axvline(E_tau, color='red', ls='--', lw=1.4, alpha=0.85)
    ax.plot(true_cate_scaled[q], 0, 'o', color='red', markersize=9,
             clip_on=False, zorder=6)
    ax.set_xlim(-1.5, 1.5)
    ax.set_title(f'query {q}', fontsize=10)
    if k // n_cols == n_rows - 1: ax.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$  (scaled)')
    if k %  n_cols == 0:          ax.set_ylabel(r'$p(\tau)$')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='upper right')
for k in range(N_QUERIES, n_rows * n_cols):
    axes[k // n_cols][k % n_cols].set_visible(False)
fig.suptitle(f'IHDP r={REALIZATION}   TRUE per-query TE distribution at N={N_CONTEXT}   '
              f'(τ|x ~ N(μ₁−μ₀, 2σ²), σ={sigma_scaled:.3f} scaled)',
              fontsize=12, y=0.999)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out = os.path.join(_OUTDIR, 'ihdp_n10_te.png')
fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
print(f'[save] {out}')


# ── Figure: OT barycenter of the true per-query TEs ─────────────────────
sys.path.insert(0, os.path.join(_REPO, 'MALC', 'Optimal_Transport'))
from ot_barycenter import wasserstein_barycenter_1d

bary = wasserstein_barycenter_1d(p_taus_true, tau_centers)
bary_norm = bary / max(bary.sum() * tau_step, 1e-12)
bary_mode = float(tau_centers[int(np.argmax(bary_norm))])
bary_mean = float((tau_centers * bary_norm).sum() * tau_step)
per_q_means = (tau_centers[None, :] * p_taus_true).sum(axis=1) * tau_step
mean_of_means = float(per_q_means.mean())
true_ate_local = float(true_cate_scaled.mean())

fig, ax = plt.subplots(figsize=(9, 4.6))
palette_Q = plt.cm.tab10(np.linspace(0, 0.9, N_QUERIES))
for k in range(N_QUERIES):
    ax.plot(tau_centers, p_taus_true[k], color=palette_Q[k], lw=1.1, alpha=0.35)
ax.fill_between(tau_centers, bary_norm, alpha=0.20, color='#0F8A3C')
ax.plot(tau_centers, bary_norm, color='#0F8A3C', lw=2.6, label='W₂ barycenter (OT)')
ax.axvline(true_ate_local, color='red', ls='--', lw=1.6,
            label=f'true population ATE = {true_ate_local:+.2f}')
ax.axvline(bary_mode, color='#0F8A3C', ls='--', lw=1.6,
            label=f'OT-mode = {bary_mode:+.2f}')
ax.axvline(mean_of_means, color='#F5A623', ls='--', lw=1.6,
            label=f'mean-of-means = {mean_of_means:+.2f}')
ax.set_xlim(-1.5, 1.5)
ax.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$  (scaled)')
ax.set_ylabel('density')
ax.set_title(f'IHDP r={REALIZATION}   TRUE OT aggregation at N={N_CONTEXT}',
              fontsize=12)
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.25)
fig.tight_layout()
out = os.path.join(_OUTDIR, 'ihdp_n10_ot.png')
fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
print(f'[save] {out}')

print(f'\n[summary — IHDP-TRUE-TE]')
print(f'  true population ATE (scaled) = {true_ate_local:+.3f}')
print(f'  mean-of-per-query-means      = {mean_of_means:+.3f}')
print(f'  W2 barycenter mode           = {bary_mode:+.3f}')
print(f'  W2 barycenter mean           = {bary_mean:+.3f}')


# ── Cache ────────────────────────────────────────────────────────────────
_cache = os.path.join(_OUTDIR, 'cache.npz')
np.savez(
    _cache,
    centers=centers, bin_width=np.float32(bin_width),
    edges=edges,
    tau_centers=tau_centers,
    mu0_scaled=mu0_scaled, mu1_scaled=mu1_scaled,
    sigma_scaled=np.float32(sigma_scaled),
    sigma_orig=np.float32(sigma_orig),
    p_y0_true=p_y0_true, p_y1_true=p_y1_true,
    p_taus_true=p_taus_true,
    true_cate_scaled=true_cate_scaled,
    QUERY_IDXS=np.asarray(QUERY_IDXS, dtype=np.int64),
    y_min=np.float32(y_min), y_rng=np.float32(y_rng),
    realization=np.int32(REALIZATION), n_context=np.int32(N_CONTEXT),
)
print(f'[cache] {_cache}')
