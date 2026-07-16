"""
Systematic audit for the "Y_do0 and Y_do1 look the same" symptom.

Checks the three failure modes in order:

  (1) Training data itself. Generate many samples. What's the distribution of
      per-query |Y_do1 - Y_do0|? If it's ~0 everywhere, our data pipeline is
      producing degenerate paired outcomes and the model correctly learned
      "the two arms are always equal".

  (2) Loss / convention. Feed the model a batch where the target is
      artificially set to (Y_do0=+0.5, Y_do1=-0.5). Compute the loss.
      Now swap: (Y_do0=-0.5, Y_do1=+0.5). If the losses are identical, the
      loss is arm-symmetric and can't distinguish.

  (3) Output symmetry. For a random query, compute the trained model's p_mat.
      Compare to its transpose. If p_mat ≈ p_mat.T, the head is outputting
      a symmetric joint — either genuinely because that's what training
      produced, or because of an architectural constraint.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch

UWYK_SRC   = os.environ.get('UWYK_SRC', '/tmp/g4cfm_uwyk/src')
CHECKPOINT = os.environ.get('CHECKPOINT', 'checkpoints/step_50000_final.pt')
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, UWYK_SRC); sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, 'MALC'))

from models.InterventionalPFN import InterventionalPFN
from losses.BarDistribution2D import unpack_pred, neg_log_prob_2d
from eval_scm_gen import generate_paired_sample_with_raw


# ── (1) Training data — is τ = Y_do1 - Y_do0 actually non-degenerate? ────────
print(f"\n{'='*70}\n(1) TRAINING DATA: per-query |τ| distribution across SCMs\n{'='*70}",
      flush=True)

t0 = time.time()
all_tau_raw, all_tau_scaled, degenerate_scms = [], [], 0
for scm_seed in range(30):
    try:
        s = generate_paired_sample_with_raw(scm_seed=scm_seed, idx=0, n_train=1000, n_test=500)
    except Exception as e:
        print(f"  seed={scm_seed}: SKIP ({type(e).__name__})", flush=True)
        continue
    tau_r = (s['Y_do1_raw'] - s['Y_do0_raw']).reshape(-1).numpy()
    tau_s = (s['Y_do1']     - s['Y_do0']    ).reshape(-1).numpy()
    all_tau_raw.append(tau_r)
    all_tau_scaled.append(tau_s)
    if float(np.abs(tau_r).max()) < 1e-4:
        degenerate_scms += 1

all_tau_raw    = np.concatenate(all_tau_raw)
all_tau_scaled = np.concatenate(all_tau_scaled)

print(f"  {len(all_tau_raw):>6d} query points across {len(all_tau_raw)//500} SCMs "
      f"({degenerate_scms} degenerate SCMs where max|τ_raw| < 1e-4)", flush=True)
print(f"  τ_scaled : mean={all_tau_scaled.mean():+.4f}  "
      f"std={all_tau_scaled.std():.4f}  "
      f"[min={all_tau_scaled.min():+.3f}, max={all_tau_scaled.max():+.3f}]", flush=True)
print(f"  |τ_scaled| quantiles: "
      f"50%={np.quantile(np.abs(all_tau_scaled), 0.5):.4f}  "
      f"75%={np.quantile(np.abs(all_tau_scaled), 0.75):.4f}  "
      f"90%={np.quantile(np.abs(all_tau_scaled), 0.90):.4f}  "
      f"99%={np.quantile(np.abs(all_tau_scaled), 0.99):.4f}", flush=True)
print(f"  fraction of queries with |τ_scaled| < 0.01: "
      f"{float((np.abs(all_tau_scaled) < 0.01).mean()):.3f}", flush=True)
print(f"  fraction of queries with |τ_scaled| > 0.10: "
      f"{float((np.abs(all_tau_scaled) > 0.10).mean()):.3f}", flush=True)


# ── (2) Loss symmetry — does the loss actually distinguish the two arms? ─────
print(f"\n{'='*70}\n(2) LOSS SYMMETRY: does swapping arms change the loss?\n{'='*70}",
      flush=True)

ckpt   = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
config = ckpt['config']
J      = config['J']
edges  = ckpt['edges'].to(DEVICE)
bin_width = float(edges[1] - edges[0])

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

# Grab one SCM's forward output on 6 queries
sample = generate_paired_sample_with_raw(scm_seed=2, idx=0, n_train=1000, n_test=500)
X_obs  = sample['X_obs'].unsqueeze(0).to(DEVICE)
T_obs  = sample['T_obs'].unsqueeze(0).to(DEVICE)
Y_obs  = sample['Y_obs'].unsqueeze(0).to(DEVICE)
qidx   = np.arange(6)
X_intv = sample['X_intv'][qidx].unsqueeze(0).to(DEVICE)
with torch.no_grad():
    pred = model(X_obs, T_obs, Y_obs, X_intv)['predictions']   # (1, 6, 10013)

# Synthetic targets: half positive-tau, half negative-tau
y_do0_A = torch.tensor([-0.5, -0.5, -0.5,  0.5,  0.5,  0.5], device=DEVICE).view(1, 6)
y_do1_A = torch.tensor([+0.5, +0.5, +0.5, -0.5, -0.5, -0.5], device=DEVICE).view(1, 6)
# Swap arms
y_do0_B = y_do1_A.clone()
y_do1_B = y_do0_A.clone()

with torch.no_grad():
    loss_A = neg_log_prob_2d(pred, y_do0_A, y_do1_A, J, edges).item()
    loss_B = neg_log_prob_2d(pred, y_do0_B, y_do1_B, J, edges).item()

print(f"  Loss with target (y0, y1) = (A, B) : {loss_A:.6f}", flush=True)
print(f"  Loss with target (y0, y1) = (B, A) : {loss_B:.6f}", flush=True)
print(f"  |ΔLoss| under arm swap             : {abs(loss_A - loss_B):.6e}", flush=True)
print(f"  → if |ΔLoss| ≈ 0, loss/output is arm-symmetric; if ≠ 0, arms distinguished.",
      flush=True)


# ── (3) Output symmetry — is the learned p_mat = p_mat.T ? ───────────────────
print(f"\n{'='*70}\n(3) OUTPUT SYMMETRY: |p_mat − p_mat.T| across queries\n{'='*70}",
      flush=True)

asym_L1, asym_argmax_delta = [], []
for i in range(6):
    p_mat, *_ = unpack_pred(pred[0, i], J, bin_width)
    p_mat_np = p_mat.detach().cpu().numpy()
    diff = np.abs(p_mat_np - p_mat_np.T)
    j0, j1 = np.unravel_index(p_mat_np.argmax(), p_mat_np.shape)
    j0T, j1T = np.unravel_index(p_mat_np.T.argmax(), p_mat_np.T.shape)
    print(f"  q{i}  peak p_mat idx=(j0={j0},j1={j1}) "
          f"peak p_mat.T idx=(j0={j0T},j1={j1T})  "
          f"|p - p.T|_L1={diff.sum():.4f}   |p - p.T|_max={diff.max():.5f}",
          flush=True)
    asym_L1.append(diff.sum())
    asym_argmax_delta.append(abs(j0 - j1))

asym_L1 = np.array(asym_L1)
print(f"\n  Mean |p_mat − p_mat.T|_L1 across queries: {asym_L1.mean():.4f}", flush=True)
print(f"  → if ~0, model learned symmetric joint (Y_do0, Y_do1 interchangeable).",
      flush=True)
print(f"  → if substantially > 0, model learned arm-distinct joints.", flush=True)

print(f"\nDone.  total {time.time() - t0:.1f}s", flush=True)
