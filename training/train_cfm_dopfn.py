"""
Production training for CFM for Decision Makers — Do-PFN prior variant.

Twin of ``train_cfm.py`` (the g4cfm/UWYK trainer). The model
(``InterventionalPFN``) and the joint head/loss (``BarDistribution2D``) are
SCM-source-agnostic and are reused **unchanged**; the only thing that differs
from the UWYK trainer is the data source:

  - streams ``PairedDoPFNDataset`` (Do-PFN's original SCM prior) instead of
    ``PairedInterventionalDataset`` (UWYK's prior), and
  - the feature count is **fixed** by Do-PFN's prior (``NUM_FEATURES``), so no
    pad-to-50 step is needed — the model is simply built with the same
    ``num_features`` the dataset emits.

Everything else — optimizer / cosine schedule / bf16 / gradient accumulation /
checkpointing / edge fitting / 2D NLL — is identical to ``train_cfm.py``.

Requires Do-PFN's repo on disk; point ``DOPFN_SRC`` at it (the dataset reads
that env var). All config below is env-overridable; run a smoke test by
shrinking N_STEPS / MICROBATCH / D_MODEL / N_TRAIN etc.

    DOPFN_SRC=/path/to/Do-PFN \
    N_STEPS=20 MICROBATCH=2 GRAD_ACCUM=1 D_MODEL=64 DEPTH=2 \
    NUM_FEATURES=5 N_TRAIN=200 N_TEST=100 STREAM_WORKERS=2 \
    python training/train_cfm_dopfn.py
"""
# Enable faulthandler FIRST, before any heavyweight imports, so a C-level
# crash (segfault/abort) dumps a full stack instead of dying silently.
import faulthandler
faulthandler.enable()

import sys
import os
import math
import time
import glob
import signal
import contextlib
import torch
import torch.optim as optim
import torch.nn.functional as F

# We live in R-PFN/training/ — parent dir is the repo root that holds
# models/, losses/, data/, MALC/, checkpoints/.
REPO_SRC  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_SRC)

from models.InterventionalPFN import InterventionalPFN
from losses.BarDistribution2D import (
    make_edges, fit_edges_2d, neg_log_prob_2d, total_params,
)
from training.data.PairedDoPFNDataset import make_dopfn_streaming_loader


# ── CONFIG (env-overridable) ──────────────────────────────────────────────────

# Grid / output
J             = int(os.environ.get('J', 100))
JJ            = J * J
OUTPUT_DIM    = total_params(J)
# Feature count is dictated by the Do-PFN prior (fixed per task, no padding).
# The dataset emits exactly this many covariates and the model is built to match.
NUM_FEATURES  = int(os.environ.get('NUM_FEATURES', 10))

# Model — UWYK Appendix G defaults (architecture held identical across priors)
D_MODEL       = int(os.environ.get('D_MODEL', 256))
DEPTH         = int(os.environ.get('DEPTH', 8))
HEADS         = int(os.environ.get('HEADS', 8))
DROPOUT       = float(os.environ.get('DROPOUT', 0.0))
HIDDEN_MULT   = int(os.environ.get('HIDDEN_MULT', 4))

# Optimizer — UWYK Appendix G defaults
LR            = float(os.environ.get('LR', 1e-4))
WEIGHT_DECAY  = float(os.environ.get('WEIGHT_DECAY', 1e-5))
WARMUP_FRAC   = float(os.environ.get('WARMUP_FRAC', 0.1))
MIN_LR_RATIO  = float(os.environ.get('MIN_LR_RATIO', 0.1))
GRAD_CLIP     = float(os.environ.get('GRAD_CLIP', 1.0))

# Training — 50K steps, eff. batch 32 (microbatch 8 × accum 4)
N_STEPS         = int(os.environ.get('N_STEPS', 50000))
MICROBATCH      = int(os.environ.get('MICROBATCH', 8))
GRAD_ACCUM      = int(os.environ.get('GRAD_ACCUM', 4))
N_CONTEXT_TRAIN = int(os.environ.get('N_CONTEXT_TRAIN', 1000))
N_QUERY_TRAIN   = int(os.environ.get('N_QUERY_TRAIN', 250))

# Per-task SCM sizes emitted by the dataset (kept >= the context/query subsample
# above). Exposed so a smoke test can shrink SCM-generation cost.
N_TRAIN         = int(os.environ.get('N_TRAIN', 1000))
N_TEST          = int(os.environ.get('N_TEST', 500))

# Precision
USE_BF16        = os.environ.get('USE_BF16', '1') == '1'

# Activation checkpointing — recomputes block forward during backward to
# halve activation memory (needed at UWYK scale under autocast).
USE_CHECKPOINT  = os.environ.get('USE_CHECKPOINT', '1') == '1'

# Streaming
STREAM_WORKERS  = int(os.environ.get('STREAM_WORKERS', 8))
STREAM_SEED     = int(os.environ.get('STREAM_SEED', 42))
STREAM_WARMUP   = int(os.environ.get('STREAM_WARMUP', 4))

# Checkpoints
CHECKPOINT_DIR    = os.environ.get('CHECKPOINT_DIR', './checkpoints_dopfn')
CHECKPOINT_EVERY  = int(os.environ.get('CHECKPOINT_EVERY', 5000))
RESUME            = os.environ.get('RESUME', '1') == '1'

# Logging
LOG_EVERY     = int(os.environ.get('LOG_EVERY', 100))
LOSS_WARN_THRESH = float(os.environ.get('LOSS_WARN_THRESH', 1e3))

# Do-PFN source (the dataset reads this too; surfaced here for a friendly check)
DOPFN_SRC     = os.environ.get('DOPFN_SRC', '/tmp/dopfn')

# Device
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


# ── Helpers ───────────────────────────────────────────────────────────────────

def _range_str(t):
    """min/max/absmax/nonfinite summary of a tensor, robust to NaN/Inf."""
    tf = t.detach().float()
    nb = int((~torch.isfinite(tf)).sum())
    fin = tf[torch.isfinite(tf)]
    if fin.numel() == 0:
        return f"all-nonfinite (n={tf.numel()})"
    return (f"min={fin.min().item():.3g} max={fin.max().item():.3g} "
            f"absmax={fin.abs().max().item():.3g} nonfinite={nb}/{tf.numel()}")


def log_bad_batch(step, loss, batch, logits, reason):
    """Dump the target/logit ranges for a batch that produced a bad loss."""
    print(f"  [LOSS-WARN] step {step}: {reason} (loss={loss.item():.4g})")
    for k in ('Y_obs', 'Y_do0', 'Y_do1'):
        if k in batch:
            print(f"      {k:>6}: {_range_str(batch[k])}")
    print(f"      logits: {_range_str(logits)}")
    sys.stdout.flush()


def print_config():
    print("─" * 72)
    print(f"Data source:   Do-PFN prior  (DOPFN_SRC={DOPFN_SRC})")
    print(f"Device:        {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU:           {torch.cuda.get_device_name(0)}")
        print(f"Free memory:   {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")
    print(f"Precision:     {'bf16 autocast' if USE_BF16 else 'fp32'}")
    print(f"Checkpoint:    activation checkpointing {'on' if USE_CHECKPOINT else 'off'}")
    print(f"J:             {J}  (output_dim = {OUTPUT_DIM})")
    print(f"Features:      num_features={NUM_FEATURES}  (fixed by prior, no padding)")
    print(f"Model:         d_model={D_MODEL}  depth={DEPTH}  heads={HEADS}  hidden_mult={HIDDEN_MULT}")
    print(f"Optimizer:     Adam(lr={LR:.0e}, wd={WEIGHT_DECAY:.0e})  grad_clip={GRAD_CLIP}")
    print(f"Schedule:      cosine  warmup_frac={WARMUP_FRAC}  min_lr_ratio={MIN_LR_RATIO}")
    print(f"Training:      steps={N_STEPS}  microbatch={MICROBATCH}  grad_accum={GRAD_ACCUM}")
    print(f"                effective_batch = {MICROBATCH * GRAD_ACCUM}")
    print(f"                N_context={N_CONTEXT_TRAIN}  N_query={N_QUERY_TRAIN}")
    print(f"                per-task SCM: n_train={N_TRAIN}  n_test={N_TEST}")
    print(f"Streaming:     workers={STREAM_WORKERS}  seed_base={STREAM_SEED}  warmup={STREAM_WARMUP}")
    print(f"Checkpoint:    dir={CHECKPOINT_DIR}  every={CHECKPOINT_EVERY}  resume={RESUME}")
    print(f"Logging:       every {LOG_EVERY} steps")
    print("─" * 72)


def make_scheduler(optimizer, n_steps, warmup_frac, min_lr_ratio):
    """Linear warmup → cosine decay to min_lr_ratio·LR."""
    warmup_steps = max(1, int(warmup_frac * n_steps))
    decay_steps  = max(1, n_steps - warmup_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / decay_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(path, step, model, optimizer, scheduler, edges):
    tmp_path = path + '.tmp'
    torch.save({
        'step':             step,
        'model_state_dict': model.state_dict(),
        'optimizer_state':  optimizer.state_dict(),
        'scheduler_state':  scheduler.state_dict(),
        'edges':            edges.cpu(),
        'config': {
            'J': J, 'd_model': D_MODEL, 'depth': DEPTH, 'heads': HEADS,
            'num_features': NUM_FEATURES, 'hidden_mult': HIDDEN_MULT,
            'prior': 'dopfn',
        },
    }, tmp_path)
    os.replace(tmp_path, path)


def latest_checkpoint(ckpt_dir):
    files = glob.glob(os.path.join(ckpt_dir, 'step_*.pt'))
    if not files:
        return None
    files.sort(key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0]))
    return files[-1]


def load_checkpoint(path, model, optimizer, scheduler):
    print(f"Resuming from {path}")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    edges = ckpt['edges'].to(DEVICE)
    return ckpt['step'], edges


def subsample_task(batch, n_context, n_query):
    """
    batch: dict of (B, N, ...) or (B, M, ...) tensors from DataLoader.
    Subsample to (B, n_context, ...) for context and (B, n_query, ...) for query.
    """
    N = batch['X_obs'].shape[1]
    M = batch['X_intv'].shape[1]

    n = min(n_context, N)
    m = min(n_query,   M)

    ctx = torch.randperm(N)[:n]
    qry = torch.randperm(M)[:m]

    return {
        'X_obs':  batch['X_obs'][:, ctx],
        'T_obs':  batch['T_obs'][:, ctx],
        'Y_obs':  batch['Y_obs'][:, ctx].squeeze(-1),    # (B, n)
        'X_intv': batch['X_intv'][:, qry],
        'Y_do0':  batch['Y_do0'][:, qry].squeeze(-1),    # (B, m)
        'Y_do1':  batch['Y_do1'][:, qry].squeeze(-1),    # (B, m)
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print_config()
    if not os.path.isdir(DOPFN_SRC):
        print(f"[warn] DOPFN_SRC={DOPFN_SRC!r} is not a directory — set it to your "
              f"Do-PFN repo root or the dataset import will fail.")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # SIGTERM handler — SLURM sends SIGTERM ~60s before SIGKILL on wall-clock
    # timeout. Catch it, finish the current step, save a checkpoint, exit clean.
    interrupted = {'flag': False}
    def _sigterm_handler(signum, frame):
        print(f"\n[signal] Received signal {signum} — will save checkpoint and exit at next step boundary")
        interrupted['flag'] = True
    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigterm_handler)

    # Streaming data — Do-PFN prior. num_features/n_train/n_test flow through to
    # PairedDoPFNDataset, so the dataset emits exactly NUM_FEATURES covariates
    # and the model (built below) matches without any padding step.
    print(f"[stream] starting DataLoader (workers={STREAM_WORKERS}, microbatch={MICROBATCH})")
    train_loader = make_dopfn_streaming_loader(
        batch_size=MICROBATCH,
        num_workers=STREAM_WORKERS,
        seed_base=STREAM_SEED,
        num_features=NUM_FEATURES,
        n_train=N_TRAIN,
        n_test=N_TEST,
    )
    train_iter = iter(train_loader)

    # Warm-up draws for edge fitting (fit on Y_obs only — matches the dataset's
    # Y_obs-only scaling and inference-time scaling).
    print(f"[stream] drawing {STREAM_WARMUP} warm-up tasks for edge fitting…")
    warmup_samples = []
    for _ in range(STREAM_WARMUP):
        b = next(train_iter)
        for i in range(MICROBATCH):
            warmup_samples.append({k: v[i] for k, v in b.items()})
    edges = fit_edges_2d(warmup_samples, J).to(DEVICE)

    # Model — identical architecture to the UWYK trainer, only num_features differs.
    model = InterventionalPFN(
        num_features=NUM_FEATURES,
        d_model=D_MODEL,
        depth=DEPTH,
        heads_feat=HEADS,
        heads_samp=HEADS,
        dropout=DROPOUT,
        output_dim=OUTPUT_DIM,
        hidden_mult=HIDDEN_MULT,
        normalize_features=True,
        normalize_treatment=False,
        use_treatment_in_query=False,
        use_checkpoint=USE_CHECKPOINT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer, N_STEPS, WARMUP_FRAC, MIN_LR_RATIO)

    # Resume?
    start_step = 0
    if RESUME:
        ckpt = latest_checkpoint(CHECKPOINT_DIR)
        if ckpt:
            start_step, edges = load_checkpoint(ckpt, model, optimizer, scheduler)

    # bf16 autocast context
    use_amp = USE_BF16 and DEVICE.type == 'cuda'
    autocast_ctx = (lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16)) \
        if use_amp else (lambda: contextlib.nullcontext())

    print(f"\n{'step':>7}  {'loss':>10}  {'lr':>10}  {'wall':>8}")
    print("─" * 42)
    model.train()
    t0 = time.time()

    for step in range(start_step + 1, N_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(GRAD_ACCUM):
            batch = next(train_iter)
            batch = subsample_task(batch, N_CONTEXT_TRAIN, N_QUERY_TRAIN)
            for k in batch:
                batch[k] = batch[k].to(DEVICE, non_blocking=True)

            with autocast_ctx():
                logits = model(
                    batch['X_obs'], batch['T_obs'], batch['Y_obs'], batch['X_intv'],
                )['predictions']
                loss = neg_log_prob_2d(
                    logits.float(), batch['Y_do0'], batch['Y_do1'], J, edges,
                )

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Step {step}: NaN/Inf loss — skipping microbatch")
                log_bad_batch(step, loss, batch, logits, "non-finite loss")
                continue

            if loss.item() > LOSS_WARN_THRESH:
                log_bad_batch(step, loss, batch, logits, f"loss > {LOSS_WARN_THRESH:.0e}")

            (loss / GRAD_ACCUM).backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if step % LOG_EVERY == 0 or step == 1:
            wall = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            avg_loss = accum_loss / GRAD_ACCUM
            print(f"{step:>7}  {avg_loss:>10.4f}  {lr_now:>10.2e}  {wall:>7.1f}s")

        if step % CHECKPOINT_EVERY == 0:
            path = os.path.join(CHECKPOINT_DIR, f'step_{step}.pt')
            save_checkpoint(path, step, model, optimizer, scheduler, edges)
            print(f"  → checkpoint saved: {path}")

        # SIGTERM-driven graceful shutdown (SLURM time-out approaching)
        if interrupted['flag']:
            path = os.path.join(CHECKPOINT_DIR, f'step_{step}_interrupt.pt')
            save_checkpoint(path, step, model, optimizer, scheduler, edges)
            print(f"  → interrupt checkpoint saved: {path}")
            print(f"  Exiting cleanly. Resubmit to resume.")
            sys.exit(0)

    # Final checkpoint
    final_path = os.path.join(CHECKPOINT_DIR, f'step_{N_STEPS}_final.pt')
    save_checkpoint(final_path, N_STEPS, model, optimizer, scheduler, edges)
    print(f"\nFinal checkpoint: {final_path}")
    print(f"Total wall: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
