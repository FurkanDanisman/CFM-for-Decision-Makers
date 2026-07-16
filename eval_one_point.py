"""
One-SCM, one-query-point evaluation of the trained 2D CFM.

Generates 1 SCM (paired-outcome sampling), picks query point 0, runs the
trained model to get raw logits, unpacks into (p_mat, region_weights,
tail_scales), fits MALC (B=100) on the inner p_mat, evaluates the smooth
log-concave density on a 200x200 grid, saves everything.

Outputs go to eval_one_point/:
    true_Y_do0.npy        # scalar  — SCM ground truth at do(T=t0)
    true_Y_do1.npy        # scalar  — SCM ground truth at do(T=t1)
    true_TE.npy           # scalar  — Y_do1 - Y_do0
    p_mat.npy             # (J, J)  — inner-region probability matrix
    region_weights.npy    # (9,)    — 9-region mixture weights
    tail_scales.npy       # (4,)    — sL0, sR0, sL1, sR1
    grid_x.npy            # (J+1,)  — bin edges for Y_do0
    grid_y.npy            # (J+1,)  — bin edges for Y_do1
    malc_grid_x.npy       # (200,)  — eval grid Y_do0
    malc_grid_y.npy       # (200,)  — eval grid Y_do1
    malc_density.npy      # (200,200) — MALC density on the grid
    context.pt            # dict of X_obs, T_obs, Y_obs, X_intv used

Run:  python eval_one_point.py
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch

# ── Config ────────────────────────────────────────────────────────────────────
UWYK_SRC   = os.environ.get('UWYK_SRC', '/tmp/g4cfm/src')
CHECKPOINT = os.environ.get('CHECKPOINT', 'checkpoints/step_50000_final.pt')
OUT_DIR    = os.environ.get('OUT_DIR', 'eval_one_point')
SCM_SEED   = int(os.environ.get('SCM_SEED', 0))
QUERY_IDX  = int(os.environ.get('QUERY_IDX', 0))
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_REPO = os.path.dirname(os.path.abspath(__file__))
# Order matters: repo root FIRST so our models.InterventionalPFN wins over
# UWYK's copy of the same module name.
sys.path.insert(0, UWYK_SRC)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'MALC'))

from data.PairedInterventionalDataset import PairedInterventionalDataset
from models.InterventionalPFN import InterventionalPFN
from losses.BarDistribution2D import unpack_pred, fit_malc_inner
from malc_2d import dmalc_2d

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Generate 1 SCM sample (n_train=1000, n_test=500) ───────────────────────
print(f"[1/5] Generating SCM (seed_base={SCM_SEED})…")
ds = PairedInterventionalDataset(
    n_train=1000, n_test=500, max_features=50,
    seed_base=SCM_SEED, max_outer_attempts=50,
)
sample = ds[0]  # dict of tensors, no batch dim

# Save true paired outcomes at the query point
true_Y_do0 = float(sample['Y_do0'][QUERY_IDX, 0])
true_Y_do1 = float(sample['Y_do1'][QUERY_IDX, 0])
true_TE    = true_Y_do1 - true_Y_do0
np.save(f'{OUT_DIR}/true_Y_do0.npy', np.array(true_Y_do0))
np.save(f'{OUT_DIR}/true_Y_do1.npy', np.array(true_Y_do1))
np.save(f'{OUT_DIR}/true_TE.npy',    np.array(true_TE))
print(f"        true (Y_do0, Y_do1) at query {QUERY_IDX}: "
      f"({true_Y_do0:+.4f}, {true_Y_do1:+.4f}) → TE={true_TE:+.4f}")

# ── 2. Load checkpoint ────────────────────────────────────────────────────────
print(f"[2/5] Loading checkpoint {CHECKPOINT}…")
ckpt   = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
config = ckpt['config']
J = config['J']
NUM_FEATURES = config['num_features']
edges = ckpt['edges'].to(DEVICE)            # (J+1,) bin edges
bin_width = float(edges[1] - edges[0])

model = InterventionalPFN(
    num_features=NUM_FEATURES,
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
print(f"        J={J}  bin_width={bin_width:.4f}  device={DEVICE}")

# ── 3. Run model on this SCM, extract query-point logits ──────────────────────
print(f"[3/5] Forward pass…")
X_obs  = sample['X_obs'].unsqueeze(0).to(DEVICE)     # (1, 1000, 50)
T_obs  = sample['T_obs'].unsqueeze(0).to(DEVICE)     # (1, 1000, 1)
Y_obs  = sample['Y_obs'].unsqueeze(0).to(DEVICE)     # (1, 1000, 1)
X_intv = sample['X_intv'][QUERY_IDX:QUERY_IDX+1].unsqueeze(0).to(DEVICE)  # (1, 1, 50)

with torch.no_grad():
    out = model(X_obs, T_obs, Y_obs, X_intv)
pred = out['predictions'][0, 0]  # (J*J + 9 + 4,)

# Save the raw context for reproducibility
torch.save({
    'X_obs':  sample['X_obs'],
    'T_obs':  sample['T_obs'],
    'Y_obs':  sample['Y_obs'],
    'X_intv': sample['X_intv'][QUERY_IDX:QUERY_IDX+1],
    'query_idx': QUERY_IDX,
    'scm_seed_base': SCM_SEED,
}, f'{OUT_DIR}/context.pt')

# ── 4. Unpack raw logits into (p_mat, region weights, tail scales) ────────────
print(f"[4/5] Unpacking logits…")
p_mat, w_region, sL0, sR0, sL1, sR1 = unpack_pred(pred, J, bin_width)
# unpack_pred returns torch tensors; move to CPU numpy for MALC / disk
p_mat_np       = p_mat.detach().cpu().numpy()            # (J, J) — sums to 1
w_region_np    = w_region.detach().cpu().numpy()         # (9,)
tail_scales_np = np.array([                              # (4,) sL0, sR0, sL1, sR1
    float(sL0), float(sR0), float(sL1), float(sR1),
])
edges_np       = edges.detach().cpu().numpy()            # (J+1,)

np.save(f'{OUT_DIR}/p_mat.npy',           p_mat_np)
np.save(f'{OUT_DIR}/region_weights.npy',  w_region_np)
np.save(f'{OUT_DIR}/tail_scales.npy',     tail_scales_np)
np.save(f'{OUT_DIR}/grid_x.npy',          edges_np)
np.save(f'{OUT_DIR}/grid_y.npy',          edges_np)

print(f"        inner-region mass: {p_mat_np.sum():.4f}")
print(f"        region weights   : {w_region_np.round(4)}")
print(f"        tail scales      : {tail_scales_np.round(4)}")

# ── 5. Fit MALC (B=100) and evaluate smoothed density on a 200x200 grid ───────
print(f"[5/5] Fitting MALC (B_select=B_fit=100)…")
malc_fit = fit_malc_inner(
    p_mat_np, edges_np, edges_np,
    B_select=100, B_fit=100, seed=SCM_SEED,
    parallel=False,  # avoid macOS ProcessPool issue for a single fit
)
print(f"        selected K={malc_fit.K}  pi={np.round(malc_fit.pi, 3)}")

# Density evaluated on a fine grid over the inner region
n_eval = 200
xs = np.linspace(edges_np[0], edges_np[-1], n_eval)
ys = np.linspace(edges_np[0], edges_np[-1], n_eval)
XX, YY = np.meshgrid(xs, ys, indexing='xy')
pts = np.column_stack([XX.ravel(), YY.ravel()])
density = dmalc_2d(malc_fit, pts).reshape(n_eval, n_eval)

np.save(f'{OUT_DIR}/malc_grid_x.npy',   xs)
np.save(f'{OUT_DIR}/malc_grid_y.npy',   ys)
np.save(f'{OUT_DIR}/malc_density.npy',  density)

print(f"\nDone. Saved to {OUT_DIR}/")
print(f"  true_Y_do0={true_Y_do0:+.4f}  true_Y_do1={true_Y_do1:+.4f}  true_TE={true_TE:+.4f}")
