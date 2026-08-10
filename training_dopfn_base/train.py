"""Training loop for the DoPFN-backbone R-PFN.

Twin of training/train_cfm_dopfn.py — everything is the same (edge fitting,
streaming loader, optimizer, cosine schedule, grad accumulation, activation
checkpointing, bf16 autocast, resume, SIGTERM graceful shutdown) EXCEPT the
model class. Instead of building InterventionalPFN, we build
DoPFNBackboneWith2DHead which loads Do-PFN's TabPFN backbone and swaps in a
2D decoder head.

Config vars unchanged from the sibling trainer; two new ones:
    DOPFN_ROOT   path to the Do-PFN repo (must contain artifacts/dopfn_model.pkl)
    HEAD_ONLY    '1' (default) freezes the transformer backbone and trains only
                   the new 2D head + Y-adapter; '0' unfreezes everything.

Usage (smoke test):
    DOPFN_ROOT=/path/to/Do-PFN DOPFN_SRC=/path/to/Do-PFN \\
    N_STEPS=20 MICROBATCH=2 GRAD_ACCUM=1 \\
    NUM_FEATURES=5 N_TRAIN=200 N_TEST=100 STREAM_WORKERS=2 \\
    python training_dopfn_base/train.py
"""
import faulthandler
faulthandler.enable()

import contextlib
import glob
import math
import os
import signal
import sys
import time

import torch
import torch.optim as optim
import torch.nn.functional as F

# Repo root
REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_SRC)

from losses.BarDistribution2D import (
    make_edges, fit_edges_2d, neg_log_prob_2d, total_params,
)
from training.data.PairedDoPFNDataset import make_dopfn_streaming_loader
from training_dopfn_base.dopfn_backbone_head import DoPFNBackboneWith2DHead


# ── CONFIG (env-overridable, mirrors train_cfm_dopfn.py) ───────────────────

J             = int(os.environ.get('J', 100))
OUTPUT_DIM    = total_params(J)
NUM_FEATURES  = int(os.environ.get('NUM_FEATURES', 10))

# New env vars for the DoPFN backbone
DOPFN_ROOT    = os.environ.get('DOPFN_ROOT',
                               os.environ.get('DOPFN_SRC', '/tmp/dopfn'))
HEAD_ONLY     = os.environ.get('HEAD_ONLY', '1') == '1'

# Optimizer / training — same defaults as train_cfm_dopfn.py
LR            = float(os.environ.get('LR', 1e-4))
WEIGHT_DECAY  = float(os.environ.get('WEIGHT_DECAY', 1e-5))
WARMUP_FRAC   = float(os.environ.get('WARMUP_FRAC', 0.1))
MIN_LR_RATIO  = float(os.environ.get('MIN_LR_RATIO', 0.1))
GRAD_CLIP     = float(os.environ.get('GRAD_CLIP', 1.0))

N_STEPS         = int(os.environ.get('N_STEPS', 50000))
MICROBATCH      = int(os.environ.get('MICROBATCH', 8))
GRAD_ACCUM      = int(os.environ.get('GRAD_ACCUM', 4))
N_CONTEXT_TRAIN = int(os.environ.get('N_CONTEXT_TRAIN', 1000))
N_QUERY_TRAIN   = int(os.environ.get('N_QUERY_TRAIN', 250))
N_TRAIN         = int(os.environ.get('N_TRAIN', 1000))
N_TEST          = int(os.environ.get('N_TEST', 500))

USE_BF16        = os.environ.get('USE_BF16', '1') == '1'
STREAM_WORKERS  = int(os.environ.get('STREAM_WORKERS', 8))
STREAM_SEED     = int(os.environ.get('STREAM_SEED', 42))
STREAM_WARMUP   = int(os.environ.get('STREAM_WARMUP', 4))

CHECKPOINT_DIR    = os.environ.get(
    'CHECKPOINT_DIR', './checkpoints_dopfn_backbone')
CHECKPOINT_EVERY  = int(os.environ.get('CHECKPOINT_EVERY', 5000))
RESUME            = os.environ.get('RESUME', '1') == '1'

LOG_EVERY        = int(os.environ.get('LOG_EVERY', 100))
LOSS_WARN_THRESH = float(os.environ.get('LOSS_WARN_THRESH', 1e3))

DEVICE = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))


# ── Helpers — vendored from train_cfm_dopfn.py so we don't have to import ──

def make_scheduler(optimizer, n_steps, warmup_frac, min_lr_ratio):
    warmup_steps = int(warmup_frac * n_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return max(min_lr_ratio,
                   0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def latest_checkpoint(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = glob.glob(os.path.join(ckpt_dir, 'step_*.pt'))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(
        os.path.basename(p).split('_')[1].split('.')[0]))


def save_checkpoint(path, step, model, optimizer, scheduler, edges):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'edges': edges.cpu(),
        'config': {
            'J': J, 'num_features': NUM_FEATURES,
            'head_only': HEAD_ONLY, 'dopfn_root': DOPFN_ROOT,
            'backbone': 'DoPFN-TabPFN',
        },
    }, path)


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt['step'], ckpt['edges'].to(DEVICE)


def subsample_task(batch, n_ctx, n_q):
    """Randomly subsample context / query rows from each task in the batch."""
    N = batch['X_obs'].shape[1]
    M = batch['X_intv'].shape[1]
    ctx_ix = torch.randperm(N)[:min(n_ctx, N)]
    q_ix   = torch.randperm(M)[:min(n_q, M)]
    out = {}
    for k, v in batch.items():
        if v.shape[1] == N:
            out[k] = v[:, ctx_ix]
        elif v.shape[1] == M:
            out[k] = v[:, q_ix]
        else:
            out[k] = v
    return out


# ── main ───────────────────────────────────────────────────────────────────

def main():
    # SIGTERM handling for SLURM graceful shutdown
    interrupted = {'flag': False}
    def _sigterm_handler(signum, frame):
        print(f"\n[signal] {signum} — will save checkpoint at next step boundary")
        interrupted['flag'] = True
    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigterm_handler)

    print("─" * 72)
    print(f"BACKBONE:      Do-PFN TabPFN (loaded from {DOPFN_ROOT})")
    print(f"HEAD:          new 2D BarDistribution decoder ({OUTPUT_DIM} outputs)")
    print(f"Training mode: {'HEAD-ONLY (backbone frozen)' if HEAD_ONLY else 'FULL fine-tune'}")
    print(f"J:             {J}   NUM_FEATURES: {NUM_FEATURES}")
    print(f"Device:        {DEVICE}")
    print("─" * 72)

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

    print(f"[stream] drawing {STREAM_WARMUP} warm-up tasks for edge fitting…")
    warmup_samples = []
    for _ in range(STREAM_WARMUP):
        b = next(train_iter)
        for i in range(MICROBATCH):
            warmup_samples.append({k: v[i] for k, v in b.items()})
    edges = fit_edges_2d(warmup_samples, J).to(DEVICE)

    # NEW: build the DoPFN-backbone model with our 2D head
    model = DoPFNBackboneWith2DHead(
        dopfn_root=DOPFN_ROOT,
        K=J,
        head_only=HEAD_ONLY,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters (total)     : {n_params:,}")
    print(f"Model parameters (trainable) : {n_trainable:,}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer, N_STEPS, WARMUP_FRAC, MIN_LR_RATIO)

    start_step = 0
    if RESUME:
        ckpt = latest_checkpoint(CHECKPOINT_DIR)
        if ckpt:
            start_step, edges = load_checkpoint(ckpt, model, optimizer, scheduler)
            print(f"[resume] step {start_step} from {ckpt}")

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

            # Context = (X_obs, T_obs, Y_obs). Y_obs is the factual outcome
            # only (Y under T_obs); Y_do0/Y_do1 are TARGETS for the joint
            # loss over query units, NOT inputs to the model.
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

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=GRAD_CLIP)
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

        if interrupted['flag']:
            path = os.path.join(CHECKPOINT_DIR, f'step_{step}_interrupt.pt')
            save_checkpoint(path, step, model, optimizer, scheduler, edges)
            print(f"  → interrupt checkpoint saved: {path}")
            sys.exit(0)

    final_path = os.path.join(CHECKPOINT_DIR, f'step_{N_STEPS}_final.pt')
    save_checkpoint(final_path, N_STEPS, model, optimizer, scheduler, edges)
    print(f"\nFinal checkpoint: {final_path}")
    print(f"Total wall: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
