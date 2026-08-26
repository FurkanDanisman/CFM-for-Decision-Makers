"""Train CausalPFN's transformer body with our 2D joint BarDistribution head.

Same training pattern as ``training_graph2d/train_graph_2d.py``:

  - Streaming ``PairedInterventionalDataset`` for paired (Y_do0, Y_do1) labels
  - bf16 autocast
  - Gradient accumulation
  - Cosine LR with linear warmup
  - Checkpoint save + resume + SIGTERM-safe interrupt save

Env vars mirror the graph2d trainer. Only differences:
  - Model class is ``CausalPFN2DHead`` (no ancestral matrix input).
  - Loss target: raw ``(Y_do0, Y_do1)`` in the scaled [-1, 1] domain that
    ``PairedInterventionalDataset`` already emits.
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

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_SRC)

from losses.BarDistribution2D import fit_edges_2d, neg_log_prob_2d, total_params
from training.data.PairedInterventionalDataset import make_streaming_loader
from training_causalpfn2d.model_causalpfn_2d import CausalPFN2DHead


# ── CONFIG (env-overridable) ──────────────────────────────────────────────────

J             = int(os.environ.get('J', 100))
OUTPUT_DIM    = total_params(J)
NUM_FEATURES  = int(os.environ.get('NUM_FEATURES', 50))

# CausalPFN's TabDPTLongContextModel defaults from the codex branch.
NINP          = int(os.environ.get('NINP', 256))
NHID_FACTOR   = int(os.environ.get('NHID_FACTOR', 4))
NHID          = NINP * NHID_FACTOR
NHEAD         = int(os.environ.get('NHEAD', 8))
NLAYERS       = int(os.environ.get('NLAYERS', 8))
DROPOUT       = float(os.environ.get('DROPOUT', 0.0))
N_OUT         = int(os.environ.get('N_OUT', 10))   # unused for regression, kept for head layout

# Optimizer
LR            = float(os.environ.get('LR', 1e-4))
WEIGHT_DECAY  = float(os.environ.get('WEIGHT_DECAY', 1e-5))
WARMUP_FRAC   = float(os.environ.get('WARMUP_FRAC', 0.1))
MIN_LR_RATIO  = float(os.environ.get('MIN_LR_RATIO', 0.1))
GRAD_CLIP     = float(os.environ.get('GRAD_CLIP', 1.0))

# Training length + batching. Default matches graph2d/fn=50 (50k steps).
N_STEPS         = int(os.environ.get('N_STEPS', 50000))
MICROBATCH      = int(os.environ.get('MICROBATCH', 2))
GRAD_ACCUM      = int(os.environ.get('GRAD_ACCUM', 16))
N_CONTEXT_TRAIN = int(os.environ.get('N_CONTEXT_TRAIN', 1000))
N_QUERY_TRAIN   = int(os.environ.get('N_QUERY_TRAIN', 250))

USE_BF16        = os.environ.get('USE_BF16', '1') == '1'

# Streaming
STREAM_WORKERS  = int(os.environ.get('STREAM_WORKERS', 8))
STREAM_SEED     = int(os.environ.get('STREAM_SEED', 42))
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


def _range_str(t):
    tf = t.detach().float()
    nb = int((~torch.isfinite(tf)).sum())
    fin = tf[torch.isfinite(tf)]
    if fin.numel() == 0:
        return f'all-nonfinite (n={tf.numel()})'
    return (f'min={fin.min().item():.3g} max={fin.max().item():.3g} '
            f'absmax={fin.abs().max().item():.3g} nonfinite={nb}/{tf.numel()}')


def log_bad_batch(step, loss, batch, logits, reason):
    print(f'  [LOSS-WARN] step {step}: {reason} (loss={loss.item():.4g})')
    for k in ('Y_obs', 'Y_do0', 'Y_do1'):
        if k in batch:
            print(f'      {k:>6}: {_range_str(batch[k])}')
    print(f'      logits: {_range_str(logits)}')
    sys.stdout.flush()


def print_config():
    print('─' * 72)
    print(f'Device:        {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'GPU:           {torch.cuda.get_device_name(0)}')
        print(f'Free memory:   {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB')
    print(f'Precision:     {"bf16 autocast" if USE_BF16 else "fp32"}')
    print(f'J:             {J}  (output_dim = {OUTPUT_DIM})')
    print(f'Backbone:      TabDPTLongContext  ninp={NINP}  nhid={NHID}  '
          f'nlayers={NLAYERS}  nhead={NHEAD}')
    print(f'Optimizer:     Adam(lr={LR:.0e}, wd={WEIGHT_DECAY:.0e})  grad_clip={GRAD_CLIP}')
    print(f'Schedule:      cosine  warmup_frac={WARMUP_FRAC}  min_lr_ratio={MIN_LR_RATIO}')
    print(f'Training:      steps={N_STEPS}  microbatch={MICROBATCH}  grad_accum={GRAD_ACCUM}')
    print(f'                effective_batch = {MICROBATCH * GRAD_ACCUM}')
    print(f'                N_context={N_CONTEXT_TRAIN}  N_query={N_QUERY_TRAIN}')
    print(f'Streaming:     workers={STREAM_WORKERS}  seed_base={STREAM_SEED}  warmup={STREAM_WARMUP}')
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
    tmp_path = path + '.tmp'
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
    }, tmp_path)
    os.replace(tmp_path, path)


def latest_checkpoint(ckpt_dir):
    files = glob.glob(os.path.join(ckpt_dir, 'step_*.pt'))
    if not files:
        return None
    files.sort(key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0]))
    return files[-1]


def load_checkpoint(path, model, optimizer, scheduler):
    print(f'Resuming from {path}')
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    edges = ckpt['edges'].to(DEVICE)
    return ckpt['step'], edges


def subsample_task(batch, n_context, n_query):
    """Subsample N and M axes; anc_matrix (if emitted by the dataset) is
    dropped -- CausalPFN's body doesn't consume it."""
    N = batch['X_obs'].shape[1]
    M = batch['X_intv'].shape[1]
    n = min(n_context, N)
    m = min(n_query,   M)
    ctx = torch.randperm(N)[:n]
    qry = torch.randperm(M)[:m]
    return {
        'X_obs':  batch['X_obs'][:, ctx],
        'T_obs':  batch['T_obs'][:, ctx],
        'Y_obs':  batch['Y_obs'][:, ctx].squeeze(-1),
        'X_intv': batch['X_intv'][:, qry],
        'Y_do0':  batch['Y_do0'][:, qry].squeeze(-1),
        'Y_do1':  batch['Y_do1'][:, qry].squeeze(-1),
    }


def main():
    print_config()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    interrupted = {'flag': False}
    def _sigterm_handler(signum, frame):
        print(f'\n[signal] Received signal {signum} — will save on next step boundary')
        interrupted['flag'] = True
    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT,  _sigterm_handler)

    print(f'[stream] starting DataLoader (workers={STREAM_WORKERS}, microbatch={MICROBATCH})')
    train_loader = make_streaming_loader(
        batch_size=MICROBATCH,
        num_workers=STREAM_WORKERS,
        seed_base=STREAM_SEED,
    )
    train_iter = iter(train_loader)

    print(f'[stream] drawing {STREAM_WARMUP} warm-up tasks for edge fitting…')
    warmup_samples = []
    for _ in range(STREAM_WARMUP):
        b = next(train_iter)
        for i in range(MICROBATCH):
            warmup_samples.append({k: v[i] for k, v in b.items()})
    edges = fit_edges_2d(warmup_samples, J).to(DEVICE)

    model = CausalPFN2DHead(
        J=J,
        num_features=NUM_FEATURES,
        ninp=NINP,
        nhid=NHID,
        nhead=NHEAD,
        nlayers=NLAYERS,
        dropout=DROPOUT,
        n_out=N_OUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params:,}')

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer, N_STEPS, WARMUP_FRAC, MIN_LR_RATIO)

    start_step = 0
    if RESUME:
        ckpt = latest_checkpoint(CHECKPOINT_DIR)
        if ckpt:
            start_step, edges = load_checkpoint(ckpt, model, optimizer, scheduler)

    use_amp = USE_BF16 and DEVICE.type == 'cuda'
    autocast_ctx = (lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16)) \
        if use_amp else (lambda: contextlib.nullcontext())

    print(f'\n{"step":>7}  {"loss":>10}  {"lr":>10}  {"wall":>8}')
    print('─' * 42)
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
                    batch['X_obs'], batch['T_obs'], batch['Y_obs'],
                    batch['X_intv'],
                )
                loss = neg_log_prob_2d(
                    logits.float(), batch['Y_do0'], batch['Y_do1'], J, edges,
                )

            if not torch.isfinite(loss):
                print(f'step {step}: non-finite loss — skipping microbatch')
                log_bad_batch(step, loss, batch, logits, 'non-finite loss')
                continue

            if loss.item() > LOSS_WARN_THRESH:
                log_bad_batch(step, loss, batch, logits, f'loss > {LOSS_WARN_THRESH:.0e}')

            (loss / GRAD_ACCUM).backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if step % LOG_EVERY == 0 or step == 1:
            wall = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            avg_loss = accum_loss / GRAD_ACCUM
            print(f'{step:>7}  {avg_loss:>10.4f}  {lr_now:>10.2e}  {wall:>7.1f}s')

        if step % CHECKPOINT_EVERY == 0:
            path = os.path.join(CHECKPOINT_DIR, f'step_{step}.pt')
            save_checkpoint(path, step, model, optimizer, scheduler, edges)
            print(f'  → checkpoint saved: {path}')

        if interrupted['flag']:
            path = os.path.join(CHECKPOINT_DIR, f'step_{step}_interrupt.pt')
            save_checkpoint(path, step, model, optimizer, scheduler, edges)
            print(f'  → interrupt checkpoint saved: {path}')
            print('  Exiting cleanly. Resubmit sbatch to resume.')
            sys.exit(0)

    final_path = os.path.join(CHECKPOINT_DIR, f'step_{N_STEPS}_final.pt')
    save_checkpoint(final_path, N_STEPS, model, optimizer, scheduler, edges)
    print(f'\nFinal checkpoint: {final_path}')
    print(f'Total wall: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
