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

# Grid / output. K=32 matches CausalPFN's default nbins=1024 in parameter
# budget (32^2 + 13 = 1037 ≈ 1024).
J             = int(os.environ.get('J', 32))
OUTPUT_DIM    = total_params(J)
NUM_FEATURES  = int(os.environ.get('NUM_FEATURES', 100))

# Backbone -- matches conf/model/tabdpt_long_context.yaml exactly.
NINP          = int(os.environ.get('NINP', 384))
NHID          = int(os.environ.get('NHID', 768))
NHEAD         = int(os.environ.get('NHEAD', 6))
NLAYERS       = int(os.environ.get('NLAYERS', 20))
DROPOUT       = float(os.environ.get('DROPOUT', 0.0))
N_OUT         = int(os.environ.get('N_OUT', 10))

# Optimizer -- matches conf/optimizer/schedulefree_adamw.yaml.
LR            = float(os.environ.get('LR', 5e-4))
WEIGHT_DECAY  = float(os.environ.get('WEIGHT_DECAY', 0.05))
WARMUP_STEPS  = int(os.environ.get('WARMUP_STEPS', 1000))   # absolute, not fraction
BETA1         = float(os.environ.get('BETA1', 0.98))
BETA2         = float(os.environ.get('BETA2', 0.999))
GRAD_CLIP     = float(os.environ.get('GRAD_CLIP', 1.0))

# Training length + batching. Matches CausalPFN's max_epochs=2048 x
# num_model_updates=128 = 262 144 optimizer steps, with effective batch
# size batch_size x num_agg = 32 x 8 = 256.
N_STEPS       = int(os.environ.get('N_STEPS', 262144))
MICROBATCH    = int(os.environ.get('MICROBATCH', 32))
GRAD_ACCUM    = int(os.environ.get('GRAD_ACCUM', 8))

# CausalPFN train-step config -- matches conf/trainer/default.yaml.
MIN_TRAIN_SPLIT   = float(os.environ.get('MIN_TRAIN_SPLIT', 0.333))
MAX_TRAIN_SPLIT   = float(os.environ.get('MAX_TRAIN_SPLIT', 0.666))
OVERLAP_THRESHOLD = float(os.environ.get('OVERLAP_THRESHOLD', 0.01))

# CausalPFN meta-dataset config -- matches conf/meta_dataset/synthetic_backdoor.yaml.
N_SAMPLES_PER_TASK = int(os.environ.get('N_SAMPLES_PER_TASK', 2048))
MAX_N_COVARIATES   = int(os.environ.get('MAX_N_COVARIATES', 98))

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
    print(f'Optimizer:     AdamWScheduleFree(lr={LR:.2e}, wd={WEIGHT_DECAY:.2e}, '
          f'betas=({BETA1}, {BETA2}), warmup_steps={WARMUP_STEPS})  grad_clip={GRAD_CLIP}')
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


def build_optimizer(params):
    """CausalPFN's optimizer: schedulefree.AdamWScheduleFree with an
    absolute warmup. Fall back to plain AdamW if the package isn't
    installed (so smoke tests still pass on machines without it)."""
    try:
        from schedulefree import AdamWScheduleFree
    except ImportError:
        print('[optimizer] schedulefree not installed; falling back to torch.AdamW')
        return torch.optim.AdamW(
            params, lr=LR, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY,
        )
    return AdamWScheduleFree(
        params, lr=LR, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
    )


def save_checkpoint(path, step, model, optimizer, edges):
    tmp = path + '.tmp'
    torch.save({
        'step':             step,
        'model_state_dict': model.state_dict(),
        'optimizer_state':  optimizer.state_dict(),
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

    opt = build_optimizer(model.parameters())
    sched = None                       # schedulefree handles its own LR

    start = 0
    if RESUME:
        cp = latest_checkpoint(CHECKPOINT_DIR)
        if cp:
            print(f'Resuming from {cp}')
            ck = torch.load(cp, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ck['model_state_dict'])
            opt.load_state_dict(ck['optimizer_state'])
            edges = ck['edges'].to(DEVICE)
            start = ck['step']

    # schedulefree's AdamWScheduleFree needs .train() before training loops
    # and .eval() before checkpoint saves / eval. Guard both.
    if hasattr(opt, 'train'):
        opt.train()

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
        opt.step()

        if step % LOG_EVERY == 0 or step == 1:
            print(f'{step:>7}  {accum_loss/GRAD_ACCUM:>10.4f}  '
                  f'{opt.param_groups[0]["lr"]:>10.2e}  {time.time()-t0:>7.1f}s')

        if step % CHECKPOINT_EVERY == 0:
            if hasattr(opt, 'eval'): opt.eval()
            save_checkpoint(os.path.join(CHECKPOINT_DIR, f'step_{step}.pt'),
                            step, model, opt, edges)
            if hasattr(opt, 'train'): opt.train()

        if interrupted['flag']:
            if hasattr(opt, 'eval'): opt.eval()
            save_checkpoint(os.path.join(CHECKPOINT_DIR, f'step_{step}_interrupt.pt'),
                            step, model, opt, edges)
            sys.exit(0)

    if hasattr(opt, 'eval'): opt.eval()
    save_checkpoint(os.path.join(CHECKPOINT_DIR, f'step_{N_STEPS}_final.pt'),
                    N_STEPS, model, opt, edges)


if __name__ == '__main__':
    main()
