"""
Diagnostics for the "E[τ] ≈ 0 for every query" symptom.

Runs three tests:

  A. Sanity test the eval script.
     Feed the model 6 queries, then the SAME 6 queries with X_intv randomly
     shuffled (context unchanged). Compare per-query logits: if identical, the
     model is ignoring X_intv (bug in eval OR training). If different, X_intv
     wiring is fine.

  B. p_mat argmax locations across queries.
     For each of the 6 queries, print the (i, j) grid cell where p_mat peaks.
     If all six argmax at the same cell → model is outputting a near-constant
     joint regardless of X_intv.

  C. Repeat on other SCMs.
     Run the same 6-query E[τ] check on SCM seeds 5, 8, 12. If E[τ] ≈ 0 across
     all queries in every SCM, the collapse is systematic. If some SCMs
     behave, seed=2 is atypical.
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import torch

UWYK_SRC   = os.environ.get('UWYK_SRC', '/tmp/g4cfm_uwyk/src')
CHECKPOINT = os.environ.get('CHECKPOINT', 'checkpoints/step_50000_final.pt')
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, UWYK_SRC)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'MALC'))

from models.InterventionalPFN import InterventionalPFN
from losses.BarDistribution2D import unpack_pred
from eval_scm_gen import generate_paired_sample_with_raw


# ── Load model once ───────────────────────────────────────────────────────────
print(f"Loading checkpoint {CHECKPOINT}…", flush=True)
ckpt   = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
config = ckpt['config']
J      = config['J']
edges  = ckpt['edges'].to(DEVICE)
bin_width = float(edges[1] - edges[0])

model = InterventionalPFN(
    num_features=config['num_features'],
    d_model=config['d_model'],
    depth=config['depth'],
    heads_feat=config['heads'],
    heads_samp=config['heads'],
    dropout=0.0,
    output_dim=J * J + 9 + 4,
    hidden_mult=config['hidden_mult'],
    normalize_features=True,
    normalize_treatment=False,
    use_treatment_in_query=False,
    use_checkpoint=False,
).to(DEVICE).eval()
model.load_state_dict(ckpt['model_state_dict'])
print(f"Model loaded. J={J} bin_width={bin_width:.4f} device={DEVICE}", flush=True)


def run_forward(sample, query_idx):
    """Return pred (Q, J*J+9+4) for the given query indices."""
    X_obs  = sample['X_obs'].unsqueeze(0).to(DEVICE)
    T_obs  = sample['T_obs'].unsqueeze(0).to(DEVICE)
    Y_obs  = sample['Y_obs'].unsqueeze(0).to(DEVICE)
    X_intv = sample['X_intv'][query_idx].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]
    return pred


def E_tau_from_pred(pred, edges_np):
    """E[τ] over the inner-region p_mat (no MALC — just the discrete grid).

    Convention (verified by neg_log_prob_2d):
        p_mat[i, j] = p(Y_do0 = center_i, Y_do1 = center_j)
    so
        E[Y_do0] = Σ_ij p_mat[i,j] * center_i   →  centers[:, None] broadcast
        E[Y_do1] = Σ_ij p_mat[i,j] * center_j   →  centers[None, :] broadcast
    """
    Q = pred.shape[0]
    E_taus = []
    for i in range(Q):
        p_mat, w_region, sL0, sR0, sL1, sR1 = unpack_pred(pred[i], J, bin_width)
        p_mat_np = p_mat.detach().cpu().numpy()  # (J, J)
        centers  = 0.5 * (edges_np[:-1] + edges_np[1:])  # (J,)
        E_y0     = (centers[:, None] * p_mat_np).sum()   # Y_do0 = row axis
        E_y1     = (centers[None, :] * p_mat_np).sum()   # Y_do1 = col axis
        E_taus.append(E_y1 - E_y0)
    return np.array(E_taus)


def p_mat_argmax(pred, i):
    p_mat, *_ = unpack_pred(pred[i], J, bin_width)
    p_mat_np = p_mat.detach().cpu().numpy()
    # p_mat[i, j] = p(Y_do0=i, Y_do1=j) — row = Y_do0, col = Y_do1
    j0, j1 = np.unravel_index(p_mat_np.argmax(), p_mat_np.shape)
    return int(j0), int(j1), float(p_mat_np.max())


edges_np = edges.detach().cpu().numpy()


# ── Generate SCM seed=2, pick 6 spread queries ────────────────────────────────
print(f"\nGenerating SCM seed=2 (this is the same SCM the plots used)…", flush=True)
t0 = time.time()
sample = generate_paired_sample_with_raw(scm_seed=2, idx=0, n_train=1000, n_test=500)
print(f"  affine ymin={sample['ymin']:+.3f} ymax={sample['ymax']:+.3f}", flush=True)

true_TE_scaled = (sample['Y_do1'] - sample['Y_do0']).reshape(-1).numpy()
order = np.argsort(true_TE_scaled)
quantiles = np.linspace(0.05, 0.95, 6)
query_idx = order[(quantiles * (len(true_TE_scaled) - 1)).astype(int)]
print(f"  selected queries: {list(map(int, query_idx))}", flush=True)
print(f"  true τ (scaled) : {true_TE_scaled[query_idx].round(3).tolist()}", flush=True)


# ── CHECK A: shuffled X_intv ──────────────────────────────────────────────────
print(f"\n{'='*70}\nCHECK A: does the model use X_intv?\n{'='*70}", flush=True)

pred_orig = run_forward(sample, query_idx)
E_orig    = E_tau_from_pred(pred_orig, edges_np)

# Shuffle the 6 X_intv rows we're feeding.
shuf = np.random.default_rng(0).permutation(len(query_idx))
query_idx_shuf = query_idx[shuf]
pred_shuf = run_forward(sample, query_idx_shuf)
E_shuf    = E_tau_from_pred(pred_shuf, edges_np)

# For each original position i, its "matching" shuffled position is where the
# same query_idx now lives. If X_intv wiring is correct, E_shuf under-index-back
# should match E_orig exactly.
inv = np.argsort(shuf)
E_shuf_realigned = E_shuf[inv]

max_diff_logits = float((pred_orig - pred_shuf[inv]).abs().max())
max_diff_E_tau  = float(np.abs(E_orig - E_shuf_realigned).max())

print(f"  E[τ] per query (scaled), original order        : {E_orig.round(3).tolist()}",
      flush=True)
print(f"  E[τ] per query (scaled), shuffled+re-aligned   : {E_shuf_realigned.round(3).tolist()}",
      flush=True)
print(f"  |Δlogits|_max (should be ~0 if wiring correct) : {max_diff_logits:.4e}",
      flush=True)
print(f"  |ΔE[τ]|_max                                    : {max_diff_E_tau:.4e}",
      flush=True)

# Sanity: also check that shuffling actually produces DIFFERENT outputs
#         across position. E.g. E_shuf[0] should equal E_orig[shuf[0]].
diff_from_position0 = float((pred_orig[0] - pred_shuf[0]).abs().max())
print(f"  |pred_orig[0] - pred_shuf[0]|_max (should NOT be ~0 unless\n"
      f"      the model outputs are constant across queries)   : {diff_from_position0:.4e}",
      flush=True)


# ── CHECK B: p_mat argmax per query ───────────────────────────────────────────
print(f"\n{'='*70}\nCHECK B: p_mat peak locations across queries\n{'='*70}", flush=True)
for i in range(len(query_idx)):
    j0, j1, height = p_mat_argmax(pred_orig, i)
    y0_peak = 0.5 * (edges_np[j0] + edges_np[j0 + 1])
    y1_peak = 0.5 * (edges_np[j1] + edges_np[j1 + 1])
    print(f"  query {int(query_idx[i]):3d}  argmax (j0={j0:3d}, j1={j1:3d})  "
          f"→ (y0={y0_peak:+.3f}, y1={y1_peak:+.3f})  height={height:.5f}",
          flush=True)


# ── CHECK C: same story on other SCMs ─────────────────────────────────────────
print(f"\n{'='*70}\nCHECK C: is the pattern systematic across SCMs?\n{'='*70}", flush=True)
for scm_seed in (5, 8, 12):
    try:
        s = generate_paired_sample_with_raw(scm_seed=scm_seed, idx=0, n_train=1000, n_test=500)
    except Exception as e:
        print(f"  seed={scm_seed}: FAILED to draw SCM — {type(e).__name__}: {e}",
              flush=True)
        continue

    true_TE_all = (s['Y_do1'] - s['Y_do0']).reshape(-1).numpy()
    ord_ = np.argsort(true_TE_all)
    q = ord_[(np.linspace(0.05, 0.95, 6) * (len(true_TE_all) - 1)).astype(int)]
    pred = run_forward(s, q)
    E_hat = E_tau_from_pred(pred, edges_np)
    true_t = true_TE_all[q]
    corr = float(np.corrcoef(E_hat, true_t)[0, 1]) if E_hat.std() > 1e-6 else float('nan')
    var_ratio = float(E_hat.var() / (true_t.var() + 1e-12))
    print(f"  seed={scm_seed}  "
          f"true τ={true_t.round(3).tolist()}  E[τ]={E_hat.round(3).tolist()}  "
          f"corr={corr:+.3f}  Var(E[τ])/Var(true)={var_ratio:.4f}",
          flush=True)

print(f"\nDone.  total {time.time() - t0:.1f}s", flush=True)
