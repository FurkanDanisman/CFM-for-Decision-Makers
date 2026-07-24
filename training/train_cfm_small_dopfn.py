"""
Small-scale training for CFM for Decision Makers — Do-PFN prior variant.

Twin of ``train_cfm_small.py``. Same model / loss / eval, but the SCM prior is
Do-PFN's original prior (streamed via ``PairedDoPFNDataset``) instead of UWYK's
prior. Because Do-PFN dictates the feature count per-task, no pad-to-50 step is
needed — the model's ``num_features`` is set to the dataset's ``NUM_FEATURES``.

This is the small/debug counterpart to ``train_cfm_dopfn.py`` (the production
trainer): tiny model, tiny task shape, ~200 steps, plain Adam (no cosine / no
bf16 / no grad accum) — so it fits on a laptop CPU/MPS or one GPU in a few
minutes.

Streaming is the only data source here (there are no cached ``.pt`` files for
the Do-PFN prior).

Run:
    DOPFN_SRC=/path/to/Do-PFN \\
    python training/train_cfm_small_dopfn.py
"""
import sys
import os
import math
import torch
import torch.optim as optim
import torch.nn.functional as F

# We live in R-PFN/training/ — parent dir is the repo root.
R_PFN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, R_PFN_DIR)

from models.InterventionalPFN import InterventionalPFN
from losses.BarDistribution2D import make_edges, fit_edges_2d, neg_log_prob_2d, total_params
from training.data.PairedDoPFNDataset import make_dopfn_streaming_loader

# ── Constants ─────────────────────────────────────────────────────────────────
J            = 100
JJ           = J * J
OUTPUT_DIM   = total_params(J)        # J² + 9 region weights + 4 tail scales
Y_MIN        = -1.0
Y_MAX        =  1.0
BIN_WIDTH    = (Y_MAX - Y_MIN) / J    # 0.02
# Fixed by the Do-PFN prior — the dataset emits exactly this many covariates
# and the model is built to match (no pad-to-50 step, unlike UWYK).
NUM_FEATURES = int(os.environ.get('NUM_FEATURES', 10))

# ── Model ─────────────────────────────────────────────────────────────────────
D_MODEL  = int(os.environ.get('D_MODEL', 64))
DEPTH    = int(os.environ.get('DEPTH', 2))
HEADS    = int(os.environ.get('HEADS', 4))
DROPOUT  = float(os.environ.get('DROPOUT', 0.0))

# ── Training ──────────────────────────────────────────────────────────────────
N_CONTEXT_TRAIN = int(os.environ.get('N_CONTEXT_TRAIN', 100))
N_QUERY_TRAIN   = int(os.environ.get('N_QUERY_TRAIN', 30))
N_STEPS         = int(os.environ.get('N_STEPS', 200))
LR              = float(os.environ.get('LR', 1e-3))
LOG_EVERY       = int(os.environ.get('LOG_EVERY', 20))

# ── Per-task SCM sizes emitted by the dataset ─────────────────────────────────
# Kept >= the context/query subsample above. Exposed so a smoke test can
# shrink SCM-generation cost.
N_TRAIN         = int(os.environ.get('N_TRAIN', 200))
N_TEST          = int(os.environ.get('N_TEST', 100))

# ── Streaming ─────────────────────────────────────────────────────────────────
STREAM_WORKERS  = int(os.environ.get('STREAM_WORKERS', 2))
STREAM_SEED     = int(os.environ.get('STREAM_SEED', 42))
STREAM_WARMUP   = int(os.environ.get('STREAM_WARMUP', 4))

# ── Evaluation ────────────────────────────────────────────────────────────────
N_CONTEXT_EVAL  = int(os.environ.get('N_CONTEXT_EVAL', 200))
SHOW_ROWS       = 5
SHOW_FEATS      = 8
SHOW_QUERIES    = 5

# Do-PFN source (surfaced here for a friendly check; the dataset reads it too)
DOPFN_SRC     = os.environ.get('DOPFN_SRC', '/tmp/dopfn')

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


# ── Data ──────────────────────────────────────────────────────────────────────

def make_batch_streaming(stream_iter, n_context: int, n_query: int):
    """
    Pull one fresh SCM task from the streaming DataLoader, subsample rows.
    DataLoader here is built with batch_size=1, so we drop the batch dim of 1
    before subsampling — same signature as train_cfm_small.py's helper.
    """
    s = next(stream_iter)
    s = {k: v.squeeze(0) for k, v in s.items()}
    n = min(n_context, s['X_obs'].shape[0])
    m = min(n_query,   s['X_intv'].shape[0])

    ctx = torch.randperm(s['X_obs'].shape[0])[:n]
    qry = torch.randperm(s['X_intv'].shape[0])[:m]

    X_obs  = s['X_obs'][ctx].unsqueeze(0)
    T_obs  = s['T_obs'][ctx].unsqueeze(0)
    Y_obs  = s['Y_obs'][ctx].squeeze(-1).unsqueeze(0)
    X_intv = s['X_intv'][qry].unsqueeze(0)
    Y_do0  = s['Y_do0'][qry].unsqueeze(0)
    Y_do1  = s['Y_do1'][qry].unsqueeze(0)

    return X_obs, T_obs, Y_obs, X_intv, Y_do0, Y_do1


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: InterventionalPFN, test_sample: dict, edges: torch.Tensor):
    model.eval()
    s = test_sample

    N_total = s['X_obs'].shape[0]
    M_total = s['X_intv'].shape[0]
    n = min(N_CONTEXT_EVAL, N_total)
    ctx = torch.arange(n)

    X_obs_cpu  = s['X_obs'][ctx]
    T_obs_cpu  = s['T_obs'][ctx]
    Y_obs_cpu  = s['Y_obs'][ctx]
    X_intv_cpu = s['X_intv']
    Y_do0_cpu  = s['Y_do0']
    Y_do1_cpu  = s['Y_do1']

    X_obs  = X_obs_cpu.unsqueeze(0).to(DEVICE)
    T_obs  = T_obs_cpu.unsqueeze(0).to(DEVICE)
    Y_obs  = Y_obs_cpu.squeeze(-1).unsqueeze(0).to(DEVICE)
    X_intv = X_intv_cpu.unsqueeze(0).to(DEVICE)

    # ── INPUT ─────────────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  INPUT  (Do-PFN test task)")
    print("═" * 72)

    T_flat = T_obs_cpu.squeeze()
    n0 = (T_flat == 0).sum().item()
    n1 = (T_flat == 1).sum().item()
    Y_flat = Y_obs_cpu.squeeze()

    n_show_feats = min(SHOW_FEATS, X_obs_cpu.shape[1])

    print(f"\n[ CONTEXT ]  {n} rows\n")
    print(f"  X_obs  {tuple(X_obs_cpu.shape)}  —  standardized covariates")
    print(f"  T_obs  {tuple(T_obs_cpu.shape)}  —  binary {{0,1}}  →  {n0} zeros / {n1} ones")
    print(f"  Y_obs  {tuple(Y_obs_cpu.shape)}  —  "
          f"min={Y_flat.min():.4f}  max={Y_flat.max():.4f}  mean={Y_flat.mean():.4f}")

    print(f"\n  First {SHOW_ROWS} rows  (X features 0–{n_show_feats-1}, then T, then Y):\n")
    hdr = ("  " + f"{'row':>4}  " +
           "  ".join(f"{'X'+str(f):>7}" for f in range(n_show_feats)) +
           f"  {'T':>5}  {'Y':>8}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for r in range(SHOW_ROWS):
        xv = "  ".join(f"{X_obs_cpu[r, f].item():>7.4f}" for f in range(n_show_feats))
        print(f"  {r:>4}  {xv}  {T_obs_cpu[r,0].item():>5.1f}  {Y_obs_cpu[r,0].item():>8.4f}")

    print(f"\n[ QUERY ]  {M_total} rows  —  X only, no T\n")
    print(f"  X_intv  {tuple(X_intv_cpu.shape)}  —  same structure as X_obs")
    print(f"  T_intv  —  NOT provided; model fills T slot with learned null token")

    print(f"\n  First {SHOW_ROWS} query rows  (X features 0–{n_show_feats-1}):\n")
    hdr_q = ("  " + f"{'row':>4}  " +
              "  ".join(f"{'X'+str(f):>7}" for f in range(n_show_feats)))
    print(hdr_q)
    print("  " + "─" * (len(hdr_q) - 2))
    for r in range(SHOW_ROWS):
        xv = "  ".join(f"{X_intv_cpu[r, f].item():>7.4f}" for f in range(n_show_feats))
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
        inner_logits   = logits[0, :, :JJ].float()
        inner_probs    = F.softmax(inner_logits, dim=-1)
        region_weights = F.softmax(logits[0, :, JJ:JJ+9].float(), dim=-1)

    # ── OUTPUT ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  OUTPUT  (model predictions on Do-PFN test task)")
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

    for i in range(min(SHOW_QUERIES, M_total)):
        row_inner = inner_probs[i].cpu()
        row_wt    = region_weights[i].cpu()
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
    print(f"Data source: Do-PFN prior  (DOPFN_SRC={DOPFN_SRC})")
    if not os.path.isdir(DOPFN_SRC):
        print(f"[warn] DOPFN_SRC={DOPFN_SRC!r} is not a directory — set it to your "
              f"Do-PFN repo root or the dataset import will fail.")
    print(f"Device:     {DEVICE}")
    print(f"J={J},  J²={JJ},  output_dim={OUTPUT_DIM}  (J²+9+4)")
    print(f"Bin width={BIN_WIDTH:.3f},  Y range [{Y_MIN}, {Y_MAX}]")
    print(f"Features:   num_features={NUM_FEATURES}  (fixed by Do-PFN prior, no padding)")
    print(f"Model:      d_model={D_MODEL}, depth={DEPTH}, heads={HEADS}")
    print(f"Training:   n_context={N_CONTEXT_TRAIN}, n_query={N_QUERY_TRAIN}, "
          f"steps={N_STEPS}, lr={LR}")
    print(f"Per-task:   n_train={N_TRAIN}, n_test={N_TEST}")
    print(f"Streaming:  workers={STREAM_WORKERS}, seed_base={STREAM_SEED}, warmup={STREAM_WARMUP}")
    # Same expected-loss note as train_cfm_small.py.
    print(f"Expected loss at init (uniform softmax): ~{math.log(9) + 2*math.log(2):.4f} nats\n")

    print(f"[stream] using PairedDoPFNDataset (workers={STREAM_WORKERS}, seed_base={STREAM_SEED})")
    train_loader = make_dopfn_streaming_loader(
        batch_size=1,
        num_workers=STREAM_WORKERS,
        seed_base=STREAM_SEED,
        num_features=NUM_FEATURES,
        n_train=N_TRAIN,
        n_test=N_TEST,
    )
    train_iter = iter(train_loader)

    # Warm-up pulls for edge fitting (mirrors UWYK BarDistribution.fit()).
    # The last warmup task doubles as the evaluation task, so no separate
    # test split is needed — every draw is a fresh SCM anyway.
    warmup = []
    print(f"[stream] drawing {STREAM_WARMUP} warm-up tasks for edge fitting…")
    for _ in range(STREAM_WARMUP):
        s = next(train_iter)
        warmup.append({k: v.squeeze(0) for k, v in s.items()})
    edges = fit_edges_2d(warmup, J)
    test_sample = warmup[-1]

    model = InterventionalPFN(
        num_features=NUM_FEATURES,
        d_model=D_MODEL,
        depth=DEPTH,
        heads_feat=HEADS,
        heads_samp=HEADS,
        dropout=DROPOUT,
        output_dim=OUTPUT_DIM,
        normalize_features=True,
        normalize_treatment=False,
        use_treatment_in_query=False,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")

    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"{'Step':>5}  {'Loss':>8}")
    print("─" * 16)

    losses = []
    model.train()
    for step in range(1, N_STEPS + 1):
        X_obs, T_obs, Y_obs, X_intv, Y_do0, Y_do1 = make_batch_streaming(
            train_iter, N_CONTEXT_TRAIN, N_QUERY_TRAIN
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
