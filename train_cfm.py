"""
Production training for CFM for Decision Makers — UWYK-scale.

Configurable via env vars (see CONFIG section below). Defaults match UWYK's
Appendix G config (50K steps, batch 32 via 8×4 grad accumulation, Adam @ 1e-4,
cosine + warmup, bf16, weight decay 1e-5) except for the 2D BarDistribution
head (output_dim = J² + 9 + 4 = 10,013 at J=100).

Run on Trillium via submit_train.sh.

Features:
  - Streaming PairedInterventionalDataset
  - bf16 mixed precision via torch.autocast
  - Gradient accumulation (effective batch = MICROBATCH × GRAD_ACCUM)
  - Cosine LR schedule with linear warmup
  - Checkpoints every CHECKPOINT_EVERY steps (resumes automatically)
"""
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

REPO_SRC  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_SRC)

from models.InterventionalPFN import InterventionalPFN
from losses.BarDistribution2D import (
    make_edges, fit_edges_2d, neg_log_prob_2d, total_params,
)
from data.PairedInterventionalDataset import make_streaming_loader


# ── CONFIG (env-overridable) ──────────────────────────────────────────────────

# Grid / output
J             = int(os.environ.get('J', 100))
JJ            = J * J
OUTPUT_DIM    = total_params(J)
NUM_FEATURES  = int(os.environ.get('NUM_FEATURES', 50))

# Model — UWYK Appendix G defaults
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

# Training — UWYK: 50K steps, eff. batch 32 (microbatch 8 × accum 4), N=1000, M~U[1,500]
N_STEPS         = int(os.environ.get('N_STEPS', 50000))
MICROBATCH      = int(os.environ.get('MICROBATCH', 8))
GRAD_ACCUM      = int(os.environ.get('GRAD_ACCUM', 4))
N_CONTEXT_TRAIN = int(os.environ.get('N_CONTEXT_TRAIN', 1000))
N_QUERY_TRAIN   = int(os.environ.get('N_QUERY_TRAIN', 250))

# Precision
USE_BF16        = os.environ.get('USE_BF16', '1') == '1'

# Activation checkpointing — recomputes block forward during backward to
# halve activation memory. Needed at UWYK scale because nn.MultiheadAttention
# under autocast can't dispatch to Flash Attention.
USE_CHECKPOINT  = os.environ.get('USE_CHECKPOINT', '1') == '1'

# Streaming
STREAM_WORKERS  = int(os.environ.get('STREAM_WORKERS', 8))
STREAM_SEED     = int(os.environ.get('STREAM_SEED', 42))
STREAM_WARMUP   = int(os.environ.get('STREAM_WARMUP', 4))

# Disk corpus mode — if set, read pre-generated tasks from disk instead of
# streaming fresh SCMs. Avoids the SCM-mutation memory-corruption bug and
# the ~31 s/step bottleneck of single-process streaming.
CORPUS_DIR      = os.environ.get('CORPUS_DIR', '')

# Checkpoints
CHECKPOINT_DIR    = os.environ.get('CHECKPOINT_DIR', './checkpoints')
CHECKPOINT_EVERY  = int(os.environ.get('CHECKPOINT_EVERY', 5000))
RESUME            = os.environ.get('RESUME', '1') == '1'

# Logging
LOG_EVERY     = int(os.environ.get('LOG_EVERY', 100))

# Device
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_config():
    print("─" * 72)
    print(f"Device:        {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU:           {torch.cuda.get_device_name(0)}")
        print(f"Free memory:   {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")
    print(f"Precision:     {'bf16 autocast' if USE_BF16 else 'fp32'}")
    print(f"Checkpoint:    activation checkpointing {'on' if USE_CHECKPOINT else 'off'}")
    print(f"J:             {J}  (output_dim = {OUTPUT_DIM})")
    print(f"Model:         d_model={D_MODEL}  depth={DEPTH}  heads={HEADS}  hidden_mult={HIDDEN_MULT}")
    print(f"Optimizer:     Adam(lr={LR:.0e}, wd={WEIGHT_DECAY:.0e})  grad_clip={GRAD_CLIP}")
    print(f"Schedule:      cosine  warmup_frac={WARMUP_FRAC}  min_lr_ratio={MIN_LR_RATIO}")
    print(f"Training:      steps={N_STEPS}  microbatch={MICROBATCH}  grad_accum={GRAD_ACCUM}")
    print(f"                effective_batch = {MICROBATCH * GRAD_ACCUM}")
    print(f"                N_context={N_CONTEXT_TRAIN}  N_query={N_QUERY_TRAIN}")
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
    Uses a single random permutation across the batch — cheaper than per-task,
    and the data within each task is already randomly ordered by the SCM sampler.
    """
    Bsize = batch['X_obs'].shape[0]
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
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # SIGTERM handler — SLURM sends SIGTERM ~60s before SIGKILL on wall-clock
    # timeout. Catch it, finish the current step, save a checkpoint, and exit
    # cleanly. Avoids losing up to CHECKPOINT_EVERY steps of progress.
    interrupted = {'flag': False}
    def _sigterm_handler(signum, frame):
        print(f"\n[signal] Received signal {signum} — will save checkpoint and exit at next step boundary")
        interrupted['flag'] = True
    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigterm_handler)

    # Data: disk corpus if CORPUS_DIR set, else stream fresh SCMs
    if CORPUS_DIR:
        from torch.utils.data import Dataset as _DS, DataLoader as _DL
        import glob as _glob

        class _CorpusDataset(_DS):
            def __init__(self, root):
                self.files = sorted(_glob.glob(os.path.join(root, 'sample_*.pt')))
                if not self.files:
                    raise RuntimeError(f"No sample_*.pt files in {root}")
            def __len__(self): return len(self.files)
            def __getitem__(self, idx):
                return torch.load(self.files[idx], weights_only=False)

        ds = _CorpusDataset(CORPUS_DIR)
        print(f"[corpus] loaded {len(ds)} tasks from {CORPUS_DIR}", flush=True)
        train_loader = _DL(
            ds,
            batch_size=MICROBATCH,
            shuffle=True,
            num_workers=STREAM_WORKERS,
            pin_memory=True,
            persistent_workers=STREAM_WORKERS > 0,
            drop_last=True,
        )
        train_iter = iter(train_loader)
    else:
        print(f"[stream] starting DataLoader (workers={STREAM_WORKERS}, microbatch={MICROBATCH})", flush=True)
        train_loader = make_streaming_loader(
            batch_size=MICROBATCH,
            num_workers=STREAM_WORKERS,
            seed_base=STREAM_SEED,
        )
        train_iter = iter(train_loader)

    # Warm-up draws for edge fitting
    print(f"[stream] drawing {STREAM_WARMUP} warm-up tasks for edge fitting…", flush=True)
    warmup_samples = []
    warm_t0 = time.time()
    for i_warmup in range(STREAM_WARMUP):
        t0 = time.time()
        b = next(train_iter)
        print(f"  [stream] warmup {i_warmup+1}/{STREAM_WARMUP} fetched in {time.time()-t0:.1f}s", flush=True)
        # Each batch has MICROBATCH tasks; unpack so fit_edges_2d sees per-task dicts
        for i in range(MICROBATCH):
            warmup_samples.append({k: v[i] for k, v in b.items()})
    print(f"[stream] warmup total: {time.time()-warm_t0:.1f}s for {STREAM_WARMUP*MICROBATCH} SCMs", flush=True)
    edges = fit_edges_2d(warmup_samples, J).to(DEVICE)

    # Model
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
    print(f"Model parameters: {n_params:,}", flush=True)

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

    print(f"\n{'step':>7}  {'loss':>10}  {'lr':>10}  {'wall':>8}", flush=True)
    print("─" * 42, flush=True)
    model.train()
    t0 = time.time()

    for step in range(start_step + 1, N_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(GRAD_ACCUM):
            try:
                batch = next(train_iter)
            except StopIteration:
                # Finite corpus exhausted — start a new epoch
                train_iter = iter(train_loader)
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
                continue

            (loss / GRAD_ACCUM).backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if step % LOG_EVERY == 0 or step == 1:
            wall = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            avg_loss = accum_loss / GRAD_ACCUM
            print(f"{step:>7}  {avg_loss:>10.4f}  {lr_now:>10.2e}  {wall:>7.1f}s", flush=True)

        if step % CHECKPOINT_EVERY == 0:
            path = os.path.join(CHECKPOINT_DIR, f'step_{step}.pt')
            save_checkpoint(path, step, model, optimizer, scheduler, edges)
            print(f"  → checkpoint saved: {path}")

        # SIGTERM-driven graceful shutdown (SLURM time-out approaching)
        if interrupted['flag']:
            path = os.path.join(CHECKPOINT_DIR, f'step_{step}_interrupt.pt')
            save_checkpoint(path, step, model, optimizer, scheduler, edges)
            print(f"  → interrupt checkpoint saved: {path}")
            print(f"  Exiting cleanly. Resubmit submit_train.sh to resume.")
            sys.exit(0)

    # Final checkpoint
    final_path = os.path.join(CHECKPOINT_DIR, f'step_{N_STEPS}_final.pt')
    save_checkpoint(final_path, N_STEPS, model, optimizer, scheduler, edges)
    print(f"\nFinal checkpoint: {final_path}")
    print(f"Total wall: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
