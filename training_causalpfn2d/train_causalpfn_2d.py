"""Train CausalPFN's transformer body with our 2D joint BarDistribution head.

Uses CausalPFN's own data pipeline (`BackdoorDGPMetaDataset`) and mirrors
the per-batch train-step logic of `causalpfn.training.trainer` — the
only difference is the loss: our joint 2D neg-log-prob on paired
(E_y0, E_y1) instead of their single-arm HL-Gauss cross-entropy.

Env vars mirror `training_graph2d/train_graph_2d.py`. Key CausalPFN
config that matters here:
  - MIN/MAX_TRAIN_SPLIT: random context/query split fraction per batch
  - OVERLAP_THRESHOLD: drop tables where treated or control count is
    below this fraction of split_pos (mirrors CausalPFN's valid_mask)
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
from torch.utils.data import DataLoader

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_SRC)

# Make sure CAUSALPFN_ROOT is on sys.path before importing the dataset.
from training_causalpfn2d.model_causalpfn_2d import (      # noqa: E402  (side-effects wire paths)
    CausalPFN2DHead, _wire_causalpfn_paths,
)
_wire_causalpfn_paths()

from causalpfn.training.priors import BackdoorDGPMetaDataset  # noqa: E402
from losses.BarDistribution2D import fit_edges_2d, total_params  # noqa: E402


# ── CONFIG (env-overridable) ──────────────────────────────────────────────────

# Grid / output. J=25 matches CausalPFN's default nbins=500 in param budget
# (25^2 + 13 = 638 vs 500).
J             = int(os.environ.get('J', 25))
OUTPUT_DIM    = total_params(J)
NUM_FEATURES  = int(os.environ.get('NUM_FEATURES', 100))

# Backbone
NINP          = int(os.environ.get('NINP', 256))
NHID_FACTOR   = int(os.environ.get('NHID_FACTOR', 4))
NHID          = NINP * NHID_FACTOR
NHEAD         = int(os.environ.get('NHEAD', 8))
NLAYERS       = int(os.environ.get('NLAYERS', 8))
DROPOUT       = float(os.environ.get('DROPOUT', 0.0))
N_OUT         = int(os.environ.get('N_OUT', 10))

# Optimizer
LR            = float(os.environ.get('LR', 1e-4))
WEIGHT_DECAY  = float(os.environ.get('WEIGHT_DECAY', 1e-5))
WARMUP_FRAC   = float(os.environ.get('WARMUP_FRAC', 0.1))
MIN_LR_RATIO  = float(os.environ.get('MIN_LR_RATIO', 0.1))
GRAD_CLIP     = float(os.environ.get('GRAD_CLIP', 1.0))

# Training length + batching. Default matches graph2d/fn=50 (50k steps).
N_STEPS       = int(os.environ.get('N_STEPS', 50000))
MICROBATCH    = int(os.environ.get('MICROBATCH', 2))
GRAD_ACCUM    = int(os.environ.get('GRAD_ACCUM', 16))

# CausalPFN train-step config
MIN_TRAIN_SPLIT   = float(os.environ.get('MIN_TRAIN_SPLIT', 0.3))
MAX_TRAIN_SPLIT   = float(os.environ.get('MAX_TRAIN_SPLIT', 0.9))
OVERLAP_THRESHOLD = float(os.environ.get('OVERLAP_THRESHOLD', 0.05))

# CausalPFN meta-dataset config (BackdoorDGPMetaDataset defaults are set
# via its hydra config; we pass a minimum set here and let the class
# fill any remaining defaults). Override individual knobs with env vars
# if you need to.
N_SAMPLES_PER_TASK = int(os.environ.get('N_SAMPLES_PER_TASK', 1024))
MAX_N_COVARIATES   = int(os.environ.get('MAX_N_COVARIATES', 50))

USE_BF16          = os.environ.get('USE_BF16', '1') == '1'

# Streaming
STREAM_WORKERS  = int(os.environ.get('STREAM_WORKERS', 8))
STREAM_WARMUP   = int(os.environ.get('STREAM_WARMUP', 4))

# Checkpoints
CHECKPOINT_DIR    = os.environ.get('CHECKPOINT_DIR', './checkpoints_causalpfn2d')
CHECKPOINT_EVERY  = int(os.environ.get('CHECKPOINT_EVERY', 2000))
RESUME            = os.environ.get('RESUME', '1') == '1'

# Logging
LOG_EVERY        = int(os.environ.get('LOG_EVERY', 100))
LOSS_WARN_THRESH = float(os.environ.get('LOSS_WARN_THRESH', 1e3))

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


def print_config():
    print('─' * 72)
    print(f'Device:        {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'GPU:           {torch.cuda.get_device_name(0)}')
    print(f'Precision:     {"bf16 autocast" if USE_BF16 else "fp32"}')
    print(f'J:             {J}  (output_dim = {OUTPUT_DIM})')
    print(f'Backbone:      TabDPTLongContext  ninp={NINP}  nhid={NHID}  '
          f'nlayers={NLAYERS}  nhead={NHEAD}')
    print(f'Optimizer:     Adam(lr={LR:.0e}, wd={WEIGHT_DECAY:.0e})  grad_clip={GRAD_CLIP}')
    print(f'Schedule:      cosine  warmup_frac={WARMUP_FRAC}  min_lr_ratio={MIN_LR_RATIO}')
    print(f'Training:      steps={N_STEPS}  microbatch={MICROBATCH}  grad_accum={GRAD_ACCUM}')
    print(f'                effective_batch = {MICROBATCH * GRAD_ACCUM}')
    print(f'Data:          BackdoorDGPMetaDataset  n_samples={N_SAMPLES_PER_TASK}  '
          f'max_covariates={MAX_N_COVARIATES}')
    print(f'Split:         [{MIN_TRAIN_SPLIT}, {MAX_TRAIN_SPLIT}]  '
          f'overlap_threshold={OVERLAP_THRESHOLD}')
    print(f'Streaming:     workers={STREAM_WORKERS}  warmup={STREAM_WARMUP}')
    print(f'Checkpoint:    dir={CHECKPOINT_DIR}  every={CHECKPOINT_EVERY}  resume={RESUME}')
    print(f'Logging:       every {LOG_EVERY} steps')
    print('─' * 72)


def make_scheduler(optimizer, n_steps, warmup_frac, min_lr_ratio):
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
    tmp = path + '.tmp'
    torch.save({
        'step':             step,
        'model_state_dict': model.state_dict(),
        'optimizer_state':  optimizer.state_dict(),
        'scheduler_state':  scheduler.state_dict(),
        'edges':            edges.cpu(),
        'config': {
            'J': J, 'ninp': NINP, 'nhid': NHID, 'nlayers': NLAYERS, 'nhead': NHEAD,
            'num_features': NUM_FEATURES, 'n_out': N_OUT, 'dropout': DROPOUT,
            'backbone': 'causalpfn.TabDPTLongContextModel',
        },
    }, tmp)
    os.replace(tmp, path)


def latest_checkpoint(ckpt_dir):
    files = glob.glob(os.path.join(ckpt_dir, 'step_*.pt'))
    if not files:
        return None
    files.sort(key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0]))
    return files[-1]


def build_meta_dataset() -> BackdoorDGPMetaDataset:
    """Construct BackdoorDGPMetaDataset. Passes only the two knobs we want
    to expose via env vars (n_samples, max_covariates); everything else
    uses the class's own defaults."""
    return BackdoorDGPMetaDataset(
        name='backdoor',
        n_samples=N_SAMPLES_PER_TASK,
        max_n_covariates=MAX_N_COVARIATES,
        post_padding_n_cols=NUM_FEATURES,
    )


def _train_step_data(batch, device):
    """Mirror CausalPFN's trainer._train_step batch unpacking and split."""
    X = batch['X'].to(device, non_blocking=True)                # (B, N, F)
    # column permutation for invariance (same as CausalPFN)
    idx = torch.randperm(X.shape[-1], device=device)
    X = X[:, :, idx]
    t = batch['t'].to(device, non_blocking=True)                # (B, N)
    y = batch['y'].to(device, non_blocking=True)                # (B, N)
    E_y0 = batch['E_y0'].to(device, non_blocking=True)          # (B, N)
    E_y1 = batch['E_y1'].to(device, non_blocking=True)          # (B, N)

    split_frac = (torch.rand(()) * (MAX_TRAIN_SPLIT - MIN_TRAIN_SPLIT) + MIN_TRAIN_SPLIT).item()
    split_pos = max(1, int(X.shape[1] * split_frac))

    return {
        'X_context':  X[:, :split_pos],
        't_context':  t[:, :split_pos],
        'y_context':  y[:, :split_pos],
        'X_query':    X[:, split_pos:],
        'E_y0_query': E_y0[:, split_pos:],
        'E_y1_query': E_y1[:, split_pos:],
        't_full':     t,
        'split_pos':  split_pos,
    }


def _valid_mask(t_full, split_pos, overlap_thr):
    """Same mask CausalPFN's trainer uses: drop tasks whose context has
    too few treated or too few control units."""
    t_ctx = t_full[:, :split_pos]
    treated = (t_ctx == 1).long().sum(dim=1)
    control = (t_ctx == 0).long().sum(dim=1)
    thr = overlap_thr * split_pos
    return (treated >= thr) & (control >= thr)


def main():
    print_config()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    interrupted = {'flag': False}
    def _sig(sig, frame):
        print(f'\n[signal] {sig} — will save on next step boundary')
        interrupted['flag'] = True
    signal.signal(signal.SIGTERM, _sig); signal.signal(signal.SIGINT, _sig)

    # ── data ──
    meta = build_meta_dataset()
    loader = DataLoader(
        meta,
        batch_size=MICROBATCH,
        num_workers=STREAM_WORKERS,
        pin_memory=True,
    )
    it = iter(loader)

    # ── edges from warmup batches (uses standardised y_context per task) ──
    warmup = []
    for _ in range(STREAM_WARMUP):
        b = next(it)
        for i in range(MICROBATCH):
            y = b['y'][i].float()
            y_std = (y - y.mean()) / (y.std() + 1e-6)
            warmup.append({'Y_obs': y_std})
    edges = fit_edges_2d(warmup, J).to(DEVICE)

    # ── model ──
    model = CausalPFN2DHead(
        J=J, num_features=NUM_FEATURES,
        ninp=NINP, nhid=NHID, nhead=NHEAD, nlayers=NLAYERS,
        dropout=DROPOUT, n_out=N_OUT,
    ).to(DEVICE)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = make_scheduler(opt, N_STEPS, WARMUP_FRAC, MIN_LR_RATIO)

    start = 0
    if RESUME:
        cp = latest_checkpoint(CHECKPOINT_DIR)
        if cp:
            print(f'Resuming from {cp}')
            ck = torch.load(cp, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ck['model_state_dict'])
            opt.load_state_dict(ck['optimizer_state'])
            sched.load_state_dict(ck['scheduler_state'])
            edges = ck['edges'].to(DEVICE)
            start = ck['step']

    use_amp = USE_BF16 and DEVICE.type == 'cuda'
    ac = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if use_amp else (lambda: contextlib.nullcontext())

    print(f'\n{"step":>7}  {"loss":>10}  {"lr":>10}  {"wall":>8}')
    print('─' * 42)
    model.train()
    t0 = time.time()

    for step in range(start + 1, N_STEPS + 1):
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(GRAD_ACCUM):
            batch = next(it)
            b = _train_step_data(batch, DEVICE)
            with ac():
                losses = model(
                    X_context=b['X_context'],
                    t_context=b['t_context'],
                    y_context=b['y_context'],
                    X_query=b['X_query'],
                    E_y0_query=b['E_y0_query'],
                    E_y1_query=b['E_y1_query'],
                    edges=edges,
                )
                mask = _valid_mask(b['t_full'], b['split_pos'], OVERLAP_THRESHOLD)
                mask = mask & torch.isfinite(losses)
                valid = losses[mask]
                if valid.numel() == 0:
                    # Attach zero to every parameter so DDP-alike training doesn't hang.
                    loss = losses.new_zeros(())
                    for p in model.parameters():
                        if p.requires_grad:
                            loss = loss + 0.0 * p.sum()
                else:
                    loss = valid.mean()
            if not torch.isfinite(loss):
                print(f'step {step}: non-finite loss, skip')
                continue
            (loss / GRAD_ACCUM).backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step(); sched.step()

        if step % LOG_EVERY == 0 or step == 1:
            print(f'{step:>7}  {accum_loss/GRAD_ACCUM:>10.4f}  '
                  f'{sched.get_last_lr()[0]:>10.2e}  {time.time()-t0:>7.1f}s')

        if step % CHECKPOINT_EVERY == 0:
            save_checkpoint(os.path.join(CHECKPOINT_DIR, f'step_{step}.pt'),
                            step, model, opt, sched, edges)

        if interrupted['flag']:
            save_checkpoint(os.path.join(CHECKPOINT_DIR, f'step_{step}_interrupt.pt'),
                            step, model, opt, sched, edges)
            sys.exit(0)

    save_checkpoint(os.path.join(CHECKPOINT_DIR, f'step_{N_STEPS}_final.pt'),
                    N_STEPS, model, opt, sched, edges)


if __name__ == '__main__':
    main()
