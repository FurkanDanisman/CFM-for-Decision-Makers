"""
Small-scale training for CFM for Decision Makers.

Architecture identical to UWYK (InterventionalPFN) except:
  - T is binary {0, 1}, not continuous
  - Query is X only — no T in query (model fills T slot with learned null token)
  - Output is 2D bar distribution over (Y_do0, Y_do1), not 1D over Y_intv
  - Loss is cross-entropy over J*J = 10,000 bins

Train: samples 0-3
Test:  sample 4

Run:
    /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \\
        /Users/furkandanisman/R-PFN/train_cfm_small.py
"""
import sys
import os
import math
import torch
import torch.optim as optim
import torch.nn.functional as F

REPO_SRC  = '/tmp/g4cfm/src'
R_PFN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, R_PFN_DIR)

from models.InterventionalPFN import InterventionalPFN
from losses.BarDistribution2D import make_edges, neg_log_prob_2d, total_params

# ── Constants ─────────────────────────────────────────────────────────────────
J            = 100
JJ           = J * J
OUTPUT_DIM   = total_params(J)        # J² + 9 region weights + 4 tail scales
Y_MIN        = -1.0
Y_MAX        =  1.0
BIN_WIDTH    = (Y_MAX - Y_MIN) / J    # 0.02
NUM_FEATURES = 50

# ── Model ─────────────────────────────────────────────────────────────────────
D_MODEL  = 64
DEPTH    = 2
HEADS    = 4
DROPOUT  = 0.0

# ── Training ──────────────────────────────────────────────────────────────────
N_CONTEXT_TRAIN = 100
N_QUERY_TRAIN   = 30
N_STEPS         = 200
LR              = 1e-3
LOG_EVERY       = 20

# ── Evaluation ────────────────────────────────────────────────────────────────
N_CONTEXT_EVAL  = 200
SHOW_ROWS       = 5
SHOW_FEATS      = 8
SHOW_QUERIES    = 5     # query points to show in full

DATA_DIR = os.path.join(R_PFN_DIR, 'outputs_paired')

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


# ── Data ──────────────────────────────────────────────────────────────────────

def load_samples(data_dir: str) -> list:
    samples = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith('.pt') and fname.startswith('sample_'):
            samples.append(torch.load(os.path.join(data_dir, fname), weights_only=False))
    if not samples:
        raise RuntimeError(f"No .pt files in {data_dir}")
    return samples


def make_batch(samples: list, n_context: int, n_query: int):
    """
    Pick one SCM, subsample rows.
    Query has no T — model fills the T slot internally.
    """
    s = samples[torch.randint(0, len(samples), (1,)).item()]
    n = min(n_context, s['X_obs'].shape[0])
    m = min(n_query,   s['X_intv'].shape[0])

    ctx = torch.randperm(s['X_obs'].shape[0])[:n]
    qry = torch.randperm(s['X_intv'].shape[0])[:m]

    X_obs  = s['X_obs'][ctx].unsqueeze(0)              # (1, n, 50)
    T_obs  = s['T_obs'][ctx].unsqueeze(0)              # (1, n, 1)
    Y_obs  = s['Y_obs'][ctx].squeeze(-1).unsqueeze(0)  # (1, n)
    X_intv = s['X_intv'][qry].unsqueeze(0)             # (1, m, 50)
    Y_do0  = s['Y_do0'][qry].unsqueeze(0)              # (1, m, 1)
    Y_do1  = s['Y_do1'][qry].unsqueeze(0)              # (1, m, 1)

    return X_obs, T_obs, Y_obs, X_intv, Y_do0, Y_do1  # no T_intv


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: InterventionalPFN, test_sample: dict, edges: torch.Tensor):
    model.eval()
    s = test_sample

    N_total = s['X_obs'].shape[0]
    M_total = s['X_intv'].shape[0]
    n = min(N_CONTEXT_EVAL, N_total)
    ctx = torch.arange(n)

    X_obs_cpu  = s['X_obs'][ctx]    # (n, 50)
    T_obs_cpu  = s['T_obs'][ctx]    # (n, 1)
    Y_obs_cpu  = s['Y_obs'][ctx]    # (n, 1)
    X_intv_cpu = s['X_intv']        # (M, 50)
    Y_do0_cpu  = s['Y_do0']         # (M, 1)
    Y_do1_cpu  = s['Y_do1']         # (M, 1)

    X_obs  = X_obs_cpu.unsqueeze(0).to(DEVICE)
    T_obs  = T_obs_cpu.unsqueeze(0).to(DEVICE)
    Y_obs  = Y_obs_cpu.squeeze(-1).unsqueeze(0).to(DEVICE)
    X_intv = X_intv_cpu.unsqueeze(0).to(DEVICE)

    # ── INPUT ─────────────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  INPUT  (test sample 4)")
    print("═" * 72)

    T_flat = T_obs_cpu.squeeze()
    n0 = (T_flat == 0).sum().item()
    n1 = (T_flat == 1).sum().item()
    Y_flat = Y_obs_cpu.squeeze()

    print(f"\n[ CONTEXT ]  {n} rows\n")
    print(f"  X_obs  {tuple(X_obs_cpu.shape)}  —  standardized covariates")
    print(f"  T_obs  {tuple(T_obs_cpu.shape)}  —  binary {{0,1}}  →  {n0} zeros / {n1} ones")
    print(f"  Y_obs  {tuple(Y_obs_cpu.shape)}  —  "
          f"min={Y_flat.min():.4f}  max={Y_flat.max():.4f}  mean={Y_flat.mean():.4f}")

    print(f"\n  First {SHOW_ROWS} rows  (X features 0–{SHOW_FEATS-1}, then T, then Y):\n")
    hdr = ("  " + f"{'row':>4}  " +
           "  ".join(f"{'X'+str(f):>7}" for f in range(SHOW_FEATS)) +
           f"  {'T':>5}  {'Y':>8}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for r in range(SHOW_ROWS):
        xv = "  ".join(f"{X_obs_cpu[r, f].item():>7.4f}" for f in range(SHOW_FEATS))
        print(f"  {r:>4}  {xv}  {T_obs_cpu[r,0].item():>5.1f}  {Y_obs_cpu[r,0].item():>8.4f}")

    print(f"\n[ QUERY ]  {M_total} rows  —  X only, no T\n")
    print(f"  X_intv  {tuple(X_intv_cpu.shape)}  —  same structure as X_obs")
    print(f"  T_intv  —  NOT provided; model fills T slot with learned null token")

    print(f"\n  First {SHOW_ROWS} query rows  (X features 0–{SHOW_FEATS-1}):\n")
    hdr_q = ("  " + f"{'row':>4}  " +
              "  ".join(f"{'X'+str(f):>7}" for f in range(SHOW_FEATS)))
    print(hdr_q)
    print("  " + "─" * (len(hdr_q) - 2))
    for r in range(SHOW_ROWS):
        xv = "  ".join(f"{X_intv_cpu[r, f].item():>7.4f}" for f in range(SHOW_FEATS))
        print(f"  {r:>4}  {xv}")

    print(f"\n[ TARGET — never seen by model ]\n")
    print(f"  Y_do0  {tuple(Y_do0_cpu.shape)}  —  "
          f"min={Y_do0_cpu.min():.4f}  max={Y_do0_cpu.max():.4f}")
    print(f"  Y_do1  {tuple(Y_do1_cpu.shape)}  —  "
          f"min={Y_do1_cpu.min():.4f}  max={Y_do1_cpu.max():.4f}")

    print(f"\n  First {SHOW_ROWS} target rows:\n")
    print(f"  {'row':>4}  {'Y_do0':>9}  {'Y_do1':>9}  {'CATE':>9}")
    print("  " + "─" * 36)
    for r in range(SHOW_ROWS):
        y0 = Y_do0_cpu[r, 0].item()
        y1 = Y_do1_cpu[r, 0].item()
        print(f"  {r:>4}  {y0:>9.4f}  {y1:>9.4f}  {y1-y0:>9.4f}")

    # ── FORWARD PASS ──────────────────────────────────────────────────────────
    with torch.no_grad():
        out    = model(X_obs, T_obs, Y_obs, X_intv)   # no T_intv
        logits = out['predictions']                    # (1, M, OUTPUT_DIM)
        # Inner bin probabilities: first J² logits → softmax
        inner_logits = logits[0, :, :JJ].float()      # (M, J²)
        inner_probs  = F.softmax(inner_logits, dim=-1) # (M, J²) — conditional inner distribution
        # Region weights: next 9 logits → softmax
        region_weights = F.softmax(logits[0, :, JJ:JJ+9].float(), dim=-1)  # (M, 9)

    # ── OUTPUT ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  OUTPUT  (model predictions on test sample 4)")
    print("═" * 72)

    print(f"\n  logits shape : {tuple(logits.shape)}  (OUTPUT_DIM = J²+9+4 = {OUTPUT_DIM})")
    print(f"\n  Structure per query row:")
    print(f"    [0:{JJ}]         → inner bin logits → softmax → J×J=({J}×{J}) p_mat")
    print(f"    [{JJ}:{JJ+9}]   → 9 region weight logits → softmax → w_region")
    print(f"    [{JJ+9}:{JJ+13}] → 4 raw tail scales → σ_L0, σ_R0, σ_L1, σ_R1")
    print(f"\n  Grid: {J}×{J} bins,  each bin covers {BIN_WIDTH:.3f} units.")
    print(f"  Axis 0 (j0) = Y_do0,  Axis 1 (j1) = Y_do1.\n")

    REGION_NAMES = ['inner-inner', 'L0-inner', 'R0-inner', 'inner-L1', 'inner-R1',
                    'L0-L1', 'L0-R1', 'R0-L1', 'R0-R1']

    for i in range(SHOW_QUERIES):
        row_inner = inner_probs[i].cpu()   # (J²,) conditional inner distribution
        row_wt    = region_weights[i].cpu() # (9,)  region weights
        top5_probs, top5_idx = row_inner.topk(5)

        y0_true = Y_do0_cpu[i, 0].item()
        y1_true = Y_do1_cpu[i, 0].item()
        true_j0 = int((min(max(y0_true, Y_MIN), Y_MAX - 1e-9) - Y_MIN) / BIN_WIDTH)
        true_j1 = int((min(max(y1_true, Y_MIN), Y_MAX - 1e-9) - Y_MIN) / BIN_WIDTH)
        true_flat = true_j0 * J + true_j1

        print(f"  ── Query row {i} ──")
        print(f"     True target: Y_do0={y0_true:.4f} (j0={true_j0}) | "
              f"Y_do1={y1_true:.4f} (j1={true_j1})")
        print(f"     P(true inner bin) = {row_inner[true_flat].item():.6f}")
        print(f"     Region weights: " +
              "  ".join(f"{n}={row_wt[r].item():.3f}" for r, n in enumerate(REGION_NAMES)))
        print(f"     Top-5 inner bins:")
        print(f"     {'rank':>4}  {'flat':>9}  {'j0':>4}  {'j1':>4}  {'p_inner':>10}")
        print("     " + "─" * 38)
        for rank, (p_val, idx) in enumerate(zip(top5_probs, top5_idx)):
            j0 = idx.item() // J
            j1 = idx.item() %  J
            marker = " ← true" if idx.item() == true_flat else ""
            print(f"     {rank+1:>4}  {idx.item():>9}  {j0:>4}  {j1:>4}  "
                  f"{p_val.item():>10.6f}{marker}")
        print()

    model.train()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device:     {DEVICE}")
    print(f"J={J},  J²={JJ},  output_dim={OUTPUT_DIM}  (J²+9+4)")
    print(f"Bin width={BIN_WIDTH:.3f},  Y range [{Y_MIN}, {Y_MAX}]")
    print(f"Model:    d_model={D_MODEL}, depth={DEPTH}, heads={HEADS}")
    print(f"Training: n_context={N_CONTEXT_TRAIN}, n_query={N_QUERY_TRAIN}, "
          f"steps={N_STEPS}, lr={LR}")
    # Expected loss at uniform-softmax init: log(9 regions) + 2·log(J²) - 2·log(J²)
    #   = log(9) + 2·log(J) + 2·log(bw) = log(9) + 2·log(1) = log(9) + 0
    # More precisely: -(log(1/9) + log(1/J²) - 2·log(bw)) ≈ log(9) + 2·log(2) ≈ 3.58
    print(f"Expected loss at init (uniform softmax): ~{math.log(9) + 2*math.log(2):.4f} nats\n")

    all_samples = load_samples(DATA_DIR)
    train_samples = all_samples[:4]
    test_sample   = all_samples[4]

    edges = make_edges(J, Y_MIN, Y_MAX)

    model = InterventionalPFN(
        num_features=NUM_FEATURES,
        d_model=D_MODEL,
        depth=DEPTH,
        heads_feat=HEADS,
        heads_samp=HEADS,
        dropout=DROPOUT,
        output_dim=OUTPUT_DIM,         # J² + 9 + 4 = total_params(J)
        normalize_features=True,       # per-context quantile transform on X (identical to UWYK)
        normalize_treatment=False,     # binary T={0,1} passed through unchanged
        use_treatment_in_query=False,  # query is X only; model fills T slot with learned null token
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Train: samples 0–3  |  Test: sample 4\n")

    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"{'Step':>5}  {'Loss':>8}")
    print("─" * 16)

    losses = []
    model.train()
    for step in range(1, N_STEPS + 1):
        X_obs, T_obs, Y_obs, X_intv, Y_do0, Y_do1 = make_batch(
            train_samples, N_CONTEXT_TRAIN, N_QUERY_TRAIN
        )
        X_obs  = X_obs.to(DEVICE)
        T_obs  = T_obs.to(DEVICE)
        Y_obs  = Y_obs.to(DEVICE)
        X_intv = X_intv.to(DEVICE)
        Y_do0  = Y_do0.to(DEVICE)
        Y_do1  = Y_do1.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_obs, T_obs, Y_obs, X_intv)['predictions']  # no T_intv
        loss   = neg_log_prob_2d(logits, Y_do0, Y_do1, J, edges)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Step {step}: NaN/Inf — stopping")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        if step % LOG_EVERY == 0 or step == 1:
            print(f"{step:>5}  {loss.item():>8.4f}")

    print()
    print(f"First: {losses[0]:.4f}  →  Last: {losses[-1]:.4f}  "
          f"({losses[-1]-losses[0]:+.4f})")

    evaluate(model, test_sample, edges)


if __name__ == '__main__':
    main()
