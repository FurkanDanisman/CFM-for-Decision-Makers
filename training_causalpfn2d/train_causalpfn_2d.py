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
import datetime
import glob
import math
import os
import signal
import sys
import time

import torch
import torch.distributed as dist
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
from losses.BarDistribution2D import make_edges, total_params  # noqa: E402


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

# Loss + scaling knobs (default = current CausalPFN-style behaviour).
#   Y_SCALING_MODE ∈ {pooled_std, uwyk_minmax}
#     pooled_std  → (y - pooled_mean) / pooled_std, edges [-10, +10]
#     uwyk_minmax → per-task min/max scale to [-1, +1], edges [-1, +1]
#   LOSS_TYPE ∈ {density, hlgauss}
#     density → neg_log_prob_2d (bar-distribution density on full 9 regions)
#     hlgauss → neg_log_prob_2d_hlgauss (2D CE with diagonal Gaussian target
#               on the inner K² region; 8 tail regions unchanged)
#   HLGAUSS_SIGMA → sigma of the target Gaussian (only used if LOSS_TYPE=hlgauss)
Y_SCALING_MODE = os.environ.get('Y_SCALING_MODE', 'pooled_std')
LOSS_TYPE      = os.environ.get('LOSS_TYPE',      'density')
HLGAUSS_SIGMA  = float(os.environ.get('HLGAUSS_SIGMA', 0.2))
assert Y_SCALING_MODE in ('pooled_std', 'uwyk_minmax'), Y_SCALING_MODE
assert LOSS_TYPE      in ('density',    'hlgauss'),     LOSS_TYPE

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
STREAM_WORKERS  = int(os.environ.get('STREAM_WORKERS', 32))
STREAM_WARMUP   = int(os.environ.get('STREAM_WARMUP', 4))
PREFETCH_FACTOR = int(os.environ.get('PREFETCH_FACTOR', 4))

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


def _strip_compile_prefix(sd):
    """Remove torch.compile's '_orig_mod.' segment from anywhere in each key.

    When only a SUBMODULE is compiled (e.g. `inner.backbone = torch.compile(inner.backbone)`),
    the resulting state_dict keys look like `backbone._orig_mod.transformer_encoder.0.weight`
    — the prefix is MID-PATH, not at the start. A naive `k.startswith('_orig_mod.')`
    strip misses these entirely, and downstream loaders silently fail to bind 148/149
    backbone parameters (verified: step_28000.pt had this exact shape). Use a global
    replace so both start-of-key and mid-path prefixes are handled.
    """
    if any('_orig_mod.' in k for k in sd):
        return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    return sd


def save_checkpoint(path, step, model, optimizer, edges):
    tmp = path + '.tmp'
    torch.save({
        'step':             step,
        'model_state_dict': _strip_compile_prefix(model.state_dict()),
        'optimizer_state':  optimizer.state_dict(),
        'edges':            edges.cpu(),
        'config': {
            'J': J, 'ninp': NINP, 'nhid': NHID, 'nlayers': NLAYERS, 'nhead': NHEAD,
            'num_features': NUM_FEATURES, 'n_out': N_OUT, 'dropout': DROPOUT,
            'backbone': 'causalpfn.TabDPTLongContextModel',
            # Loss/scaling knobs — eval needs these to reconstruct the model
            # with matching scaling behaviour.
            'y_scaling_mode': Y_SCALING_MODE,
            'loss_type':      LOSS_TYPE,
            'hlgauss_sigma':  HLGAUSS_SIGMA,
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
    """Instantiate BackdoorDGPMetaDataset the SAME way CausalPFN does --
    via Hydra on their published conf/meta_dataset/synthetic_backdoor.yaml.

    That yaml wires the two SyntheticTableGenerator objects plus the sampler
    distributions the constructor requires. Rather than reimplement them, we
    load the yaml with OmegaConf and let hydra.utils.instantiate do the work,
    matching CausalPFN's own train.py path.
    """
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    yaml_path = os.path.join(
        os.environ.get('CAUSALPFN_ROOT',
                        os.path.abspath(os.path.join(
                            os.path.dirname(__file__), '..', '..', 'external', 'causalpfn'))),
        'conf', 'meta_dataset', 'synthetic_backdoor.yaml',
    )
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(
            f'meta-dataset yaml not found at {yaml_path}. Is the '
            'codex/add-training-code branch of CausalPFN checked out?'
        )
    cfg = OmegaConf.load(yaml_path)
    # Override only the two knobs we want env-controllable; everything else
    # (samplers, distributions) inherits from the yaml.
    cfg.n_samples          = N_SAMPLES_PER_TASK
    cfg.max_n_covariates   = MAX_N_COVARIATES
    cfg.post_padding_n_cols = NUM_FEATURES
    return instantiate(cfg)


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
    # ── OPTIONAL: DDP init (torchrun sets LOCAL_RANK/RANK/WORLD_SIZE) ─────
    # Follows CausalPFN's own src/causalpfn/training/distributed.py pattern
    # verbatim. When these env vars aren't set (plain `python -m ...`), we
    # fall through to single-GPU as before — no behaviour change.
    global DEVICE
    using_dist = 'LOCAL_RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if using_dist:
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        DEVICE = torch.device(f'cuda:{local_rank}')
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=datetime.timedelta(seconds=1800),   # 30 min for NCCL init
            world_size=world_size, rank=rank,
        )
        dist.barrier(device_ids=[local_rank])
    else:
        local_rank = 0
        rank = 0
        world_size = 1
    is_main = (rank == 0)

    def _log(msg):
        if is_main: print(msg, flush=True)

    if is_main:
        print_config()
        if using_dist:
            print(f'[dist] using DDP: rank {rank}/{world_size}  device {DEVICE}', flush=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── H100 perf knobs. All are free wins on any modern GPU ─────────────
    # TF32 on matmul: ~2x on H100 for fp32 matmuls (bf16 autocast skips
    # most of these but the residual fp32 layers still benefit).
    torch.set_float32_matmul_precision('high')
    # Force Flash / Memory-efficient SDPA. Attention on seq=2048 is O(N²)
    # naïve → O(N) with Flash, which is 8-15x faster on H100. If the
    # CausalPFN backbone routes through F.scaled_dot_product_attention,
    # these hints stick; if it uses a hand-rolled matmul-softmax path,
    # they're inert (still safe to set).
    if DEVICE.type == 'cuda':
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)   # fall back only if the others reject
            torch.backends.cudnn.benchmark = True
        except Exception as e:
            print(f'[perf] SDPA hint failed: {e}')

    interrupted = {'flag': False}
    def _sig(sig, frame):
        print(f'\n[signal] {sig} — will save on next step boundary')
        interrupted['flag'] = True
    signal.signal(signal.SIGTERM, _sig); signal.signal(signal.SIGINT, _sig)

    # ── data ──
    # Match CausalPFN's conf/train.yaml verbatim: num_workers=32,
    # prefetch_factor=4, persistent_workers when workers>0. Without
    # persistent_workers, PyTorch tears down worker processes on iterator
    # restart, incurring Python import + SCM sampler warmup on every
    # relaunch and starving the GPU. Without prefetch_factor>=4, the
    # per-worker queue depth is only 2 (PyTorch default), which is not
    # deep enough to absorb DGP sampling latency variance.
    meta = build_meta_dataset()
    loader_kwargs = dict(
        batch_size=MICROBATCH,
        num_workers=STREAM_WORKERS,
        pin_memory=True,
    )
    if STREAM_WORKERS > 0:
        loader_kwargs['prefetch_factor'] = PREFETCH_FACTOR
        loader_kwargs['persistent_workers'] = True
    loader = DataLoader(meta, **loader_kwargs)
    it = iter(loader)

    # ── Edges depend on Y_SCALING_MODE ────────────────────────────────
    # pooled_std  → y_scaled = (y - pooled_mean) / pooled_std, edges [-10, +10]
    # uwyk_minmax → y_scaled = (y - shift) / scale ∈ [-1, +1],  edges [-1, +1]
    if Y_SCALING_MODE == 'uwyk_minmax':
        _edge_lo, _edge_hi = -1.0, 1.0
    else:
        _edge_lo, _edge_hi = -10.0, 10.0
    edges = make_edges(J, y_min=_edge_lo, y_max=_edge_hi).to(DEVICE)
    _log(f'[edges] Y_SCALING_MODE={Y_SCALING_MODE}  LOSS_TYPE={LOSS_TYPE}'
         f'{"  HLGAUSS_SIGMA=" + str(HLGAUSS_SIGMA) if LOSS_TYPE == "hlgauss" else ""}')
    _log(f'[edges] fixed to [{_edge_lo}, {_edge_hi}]  J+1={J+1}  '
         f'bin_width={(_edge_hi - _edge_lo)/J:.4f}')

    # Empirical sanity: consume warmup batches and log how far Y_scaled
    # actually ranges under the chosen mode. If typical values fall well
    # inside [_edge_lo, _edge_hi] the fixed edges cover the density.
    ymin_seen, ymax_seen = float('inf'), float('-inf')
    for _ in range(STREAM_WARMUP):
        b = next(it)
        for i in range(MICROBATCH):
            y = b['y'][i].float()
            if Y_SCALING_MODE == 'uwyk_minmax':
                y_lo, y_hi = y.amin(), y.amax()
                shift = 0.5 * (y_lo + y_hi)
                scale = (0.5 * (y_hi - y_lo)).clamp(min=1e-6)
                y_scaled = (y - shift) / scale
            else:
                y_scaled = (y - y.mean()) / (y.std() + 1e-6)
            ymin_seen = min(ymin_seen, y_scaled.min().item())
            ymax_seen = max(ymax_seen, y_scaled.max().item())
    _log(f'[edges] warmup empirical y_scaled range: [{ymin_seen:.3f}, {ymax_seen:.3f}]  '
         f'(fixed edges cover: {(ymin_seen >= _edge_lo and ymax_seen <= _edge_hi)})')

    # ── model ──
    model = CausalPFN2DHead(
        J=J, num_features=NUM_FEATURES,
        ninp=NINP, nhid=NHID, nhead=NHEAD, nlayers=NLAYERS,
        dropout=DROPOUT, n_out=N_OUT,
        y_scaling_mode=Y_SCALING_MODE,
        loss_type=LOSS_TYPE,
        hlgauss_sigma=HLGAUSS_SIGMA,
    ).to(DEVICE)
    if is_main:
        print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # ── OPTIONAL: warm-start the backbone from CausalPFN's pretrained TabDPT ──
    # CausalPFN's REPRODUCE.md explicitly says: "We specifically warm-start the
    # training with a predictive TabDPT checkpoint." Their tabdpt_long_context_
    # pretrained.yaml loads vdblm/causalpfn/tabdpt_long_context.ckpt from HF Hub
    # (revision 83aad07d...). Without this we train the backbone from random init
    # which loses their pretraining advantage and gives worse convergence.
    #
    # The head weights of our 2D BarDist head are NOT loaded — those are ours,
    # random-init. Only the backbone (attention + FFN + input embeddings) is
    # warm-started.
    #
    # WARMSTART_CKPT: HuggingFace model path (repo/filename) OR a local .pt path
    #   default: vdblm/causalpfn/tabdpt_long_context.ckpt (their default)
    # WARMSTART_REVISION: HF revision hash (default: 83aad07d..., their pinned commit)
    # WARMSTART=0 disables (train from scratch, matches previous behavior).
    # Only warmstart when there's no checkpoint to resume from — else the RESUME
    # block below would overwrite the warmstarted backbone with our own past step.
    _has_resume_ckpt = RESUME and latest_checkpoint(CHECKPOINT_DIR) is not None
    if os.environ.get('WARMSTART', '1') == '1' and not _has_resume_ckpt:
        try:
            hf_repo = os.environ.get('WARMSTART_REPO', 'vdblm/causalpfn')
            hf_file = os.environ.get('WARMSTART_FILE', 'tabdpt_long_context.ckpt')
            hf_rev  = os.environ.get(
                'WARMSTART_REVISION',
                '83aad07da1cb077cfda4236878a1b07dc9f72a54',
            )
            local_ckpt = os.environ.get('WARMSTART_LOCAL')
            if local_ckpt and os.path.isfile(local_ckpt):
                ckpt_path = local_ckpt
                print(f'[warmstart] loading local pretrained TabDPT ckpt: {ckpt_path}')
            else:
                from huggingface_hub import hf_hub_download
                ckpt_path = hf_hub_download(
                    repo_id=hf_repo, filename=hf_file, revision=hf_rev,
                )
                # Defensive: our shim for huggingface_hub returns a MagicMock,
                # not a path. Reject non-string returns loudly instead of
                # letting torch.load crash with a cryptic OSError.
                if not isinstance(ckpt_path, str):
                    raise RuntimeError(
                        f'hf_hub_download returned a {type(ckpt_path).__name__} '
                        f'({ckpt_path!r}), not a path — huggingface_hub is likely '
                        f'shimmed on this env. Set WARMSTART_LOCAL to a locally-'
                        f'downloaded copy of {hf_repo}/{hf_file} instead.'
                    )
                print(f'[warmstart] downloaded pretrained TabDPT ckpt from '
                      f'{hf_repo}/{hf_file}@{hf_rev[:8]}: {ckpt_path}')
            if not os.path.isfile(ckpt_path):
                raise RuntimeError(f'warmstart ckpt path does not exist: {ckpt_path}')
            sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            # Filter shape-mismatched keys BEFORE load_state_dict, else it
            # aborts wholesale and we lose the entire warmstart. Job 5037909
            # showed encoder.weight is [384, 100] vs our [384, 101] (we
            # prepend T) and head.2.weight is [1034, 768] vs [1047, 768]
            # (our 2D joint head has more output bins). Both are architectural
            # differences we chose; they must stay random-init. Every other
            # layer (the 20 transformer blocks, LNs, y_encoder) loads cleanly.
            _model_sd_ref = model.backbone.state_dict()
            _skipped_shape = []
            # CausalPFN's tabdpt_long_context.ckpt wraps the real state_dict
            # under the key 'model' (top-level = {'model', 'opt', 'cfg',
            # 'stats'}). Also handle the other common wrappers. Verified via
            # the missing=148/unexpected=4 log on job 5037726.
            if isinstance(sd, dict):
                for wrap_key in ('model_state_dict', 'state_dict', 'model'):
                    if wrap_key in sd and isinstance(sd[wrap_key], dict):
                        sd = sd[wrap_key]
                        print(f'[warmstart] unwrapped state_dict from top-level key {wrap_key!r}')
                        break
            # Strip `_orig_mod.` prefix. When a model was wrapped with
            # torch.compile(...) before saving, PyTorch prepends this prefix
            # to every state_dict key. CausalPFN saves compiled models, so
            # every key looks like `_orig_mod.transformer_encoder.0.kv_proj.
            # weight` — mismatched against our uncompiled model's keys.
            # Verified via missing=148/unexpected=148 log on job 5037860.
            if isinstance(sd, dict) and any(k.startswith('_orig_mod.') for k in sd):
                _prefix = '_orig_mod.'
                sd = {(k[len(_prefix):] if k.startswith(_prefix) else k): v
                      for k, v in sd.items()}
                print(f'[warmstart] stripped {_prefix!r} prefix from state_dict keys '
                      '(torch.compile wrapping)')
            # Load into backbone only. Head keys (BarDist.borders / last-K logits)
            # are ours and stay at their random init.
            # Filter out any key whose shape doesn't match ours — otherwise
            # load_state_dict raises and we lose the entire warmstart.
            _filtered_sd = {}
            for _k, _v in sd.items():
                if _k in _model_sd_ref and _model_sd_ref[_k].shape != _v.shape:
                    _skipped_shape.append(
                        (_k, tuple(_v.shape), tuple(_model_sd_ref[_k].shape))
                    )
                    continue
                _filtered_sd[_k] = _v
            if _skipped_shape:
                print(f'[warmstart] skipped {len(_skipped_shape)} shape-mismatched key(s) '
                      f'(kept random init on these — architectural difference vs CausalPFN):')
                for _k, _ck, _my in _skipped_shape:
                    print(f'  {_k}: ckpt {_ck} vs ours {_my}')
            missing, unexpected = model.backbone.load_state_dict(_filtered_sd, strict=False)
            print(f'[warmstart] backbone loaded  '
                  f'missing={len(missing)}  unexpected={len(unexpected)}  '
                  f'(missing includes the {len(_skipped_shape)} skipped mismatches above)')

            # HARD verification 1: reject silent load failures. Anything beyond the
            # intentionally-skipped head keys means our load path is wrong; refuse to
            # train on random init.
            _unexpected_missing = len(missing) - len(_skipped_shape)
            if _unexpected_missing > 5 or len(unexpected) > 5:
                print(f'[warmstart]   first missing:    {list(missing)[:10]}')
                print(f'[warmstart]   first unexpected: {list(unexpected)[:10]}')
                raise RuntimeError(
                    f'[warmstart] ABORT: {_unexpected_missing} unexpected missing '
                    f'and {len(unexpected)} unexpected extra keys after load_state_dict '
                    f'(beyond the {len(_skipped_shape)} intentional shape-mismatches). '
                    f'Backbone weights are mostly at random init — refusing to waste '
                    f'GPU days training a random model. Fix the load path or set '
                    f'WARMSTART=0 explicitly to skip warmstart.'
                )

            # HARD verification 2: bytes match. Pick a probe parameter and verify
            # its post-load value is bit-identical to the checkpoint's. Catches
            # cases where load_state_dict silently no-ops (empty _filtered_sd, all
            # keys skipped, wrong prefix).
            _probe_key = next(
                (k for k in _filtered_sd
                 if 'weight' in k and hasattr(_filtered_sd[k], 'dim')
                    and _filtered_sd[k].dim() >= 2),
                None,
            )
            if _probe_key is None:
                raise RuntimeError(
                    '[warmstart] ABORT: no probe key found in filtered state_dict — '
                    'nothing was loaded. Check the checkpoint format.'
                )
            _got = model.backbone.state_dict()[_probe_key].detach().cpu()
            _want = _filtered_sd[_probe_key].detach().cpu()
            if not torch.equal(_got, _want):
                raise RuntimeError(
                    f'[warmstart] ABORT: probe key {_probe_key!r} bytes do NOT match '
                    f'the checkpoint after load_state_dict. Load silently no-op\'d. '
                    f'ckpt_norm={_want.norm().item():.6f} vs loaded_norm={_got.norm().item():.6f}'
                )
            print(f'[warmstart] VERIFIED: backbone.{_probe_key} matches ckpt exactly  '
                  f'(L2={_want.norm().item():.4f}, n_elems={_want.numel()})')
        except Exception as e:
            # RuntimeError from our own abort should NOT be swallowed — that
            # defeats the whole point of the verification. Only swallow
            # transient IO / hf-hub errors.
            if isinstance(e, RuntimeError) and '[warmstart] ABORT' in str(e):
                raise
            print(f'[warmstart] FAILED ({e}); continuing from scratch')

    # torch.compile matches the CausalPFN trainer (their conf/model/*.yaml
    # sets compile: true and their trainer wraps model = torch.compile(model,
    # dynamic=True)). On a 20-layer transformer at bf16 this fuses
    # attention + MLP kernels and picks the Flash-Attention path where
    # available — typically 2-3x per-step on H100. `dynamic=True` because
    # our context/query split varies per batch.
    # ── torch.compile strategy ────────────────────────────────────────
    # Full-model compile (previous attempt) hit an Inductor tiling assert
    # (InductorError in tiling_utils.py get_pw_red_splits, job 5032910)
    # — almost certainly triggered by the loss's 8× data-dependent
    # `region_idx[bool_mask] = R_X` scatter writes + `.item()` calls.
    # The transformer backbone is where the compute is anyway, so we
    # compile just that. Loss stays eager, no compile-hostile patterns
    # in the compiled subgraph.
    #
    # Wrap model with DDP after warmstart so state_dict loading works cleanly
    # (DDP-wrapped state_dict has `.module.` prefix which would complicate the
    # shape-filter code above).
    if using_dist:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
        _log(f'[dist] model wrapped in DDP  device_ids={[local_rank]}')

    # COMPILE_MODE: 'backbone' (default) | 'full' | 'off'
    compile_mode = os.environ.get('COMPILE_MODE', 'backbone' if os.environ.get('COMPILE', '1') == '1' else 'off')
    if compile_mode != 'off' and DEVICE.type == 'cuda':
        try:
            import torch._dynamo as _dynamo
            _dynamo.config.suppress_errors = True
            if compile_mode == 'backbone':
                # Reach through DDP.module -> CausalPFN2DHead -> backbone.
                inner = model.module if hasattr(model, 'module') else model
                inner.backbone = torch.compile(inner.backbone, dynamic=True)
                _log('[compile] torch.compile(model.backbone, dynamic=True) enabled '
                     '(compile_mode=backbone — loss stays eager)')
            elif compile_mode == 'full':
                model = torch.compile(model, dynamic=True)
                _log('[compile] torch.compile(model, dynamic=True) enabled '
                     '(compile_mode=full — may hit Inductor tiling bug in loss)')
        except Exception as e:
            _log(f'[compile] torch.compile failed ({e}); running eager')
    else:
        _log(f'[compile] disabled (compile_mode={compile_mode})')

    # For the optimizer: pass the unwrapped model's params. DDP wraps
    # `model` but `model.parameters()` still returns the underlying params,
    # so this works either way. `.module` unwrap keeps things clearer for
    # future readers.
    _params_owner = model.module if using_dist and hasattr(model, 'module') else model
    opt = build_optimizer(_params_owner.parameters())
    sched = None                       # schedulefree handles its own LR

    start = 0
    if RESUME:
        cp = latest_checkpoint(CHECKPOINT_DIR)
        if cp:
            _log(f'Resuming from {cp}')
            ck = torch.load(cp, map_location=DEVICE, weights_only=False)
            # Checkpoints are saved as the UNWRAPPED model state (see
            # save_checkpoint below), so load into the unwrapped model
            # regardless of DDP-wrapping.
            _load_target = model.module if using_dist and hasattr(model, 'module') else model
            # Strip torch.compile's '_orig_mod.' prefix (potentially mid-path)
            # so old checkpoints saved before _strip_compile_prefix load cleanly.
            _load_target.load_state_dict(_strip_compile_prefix(ck['model_state_dict']))
            opt.load_state_dict(ck['optimizer_state'])
            edges = ck['edges'].to(DEVICE)
            start = ck['step']
            if using_dist:
                dist.barrier(device_ids=[local_rank])

    # schedulefree's AdamWScheduleFree needs .train() before training loops
    # and .eval() before checkpoint saves / eval. Guard both.
    if hasattr(opt, 'train'):
        opt.train()

    use_amp = USE_BF16 and DEVICE.type == 'cuda'
    ac = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if use_amp else (lambda: contextlib.nullcontext())

    if is_main:
        print(f'\n{"step":>7}  {"loss":>10}  {"lr":>10}  {"wall":>8}')
        print('─' * 42)
    model.train()
    t0 = time.time()

    # Profile the first 3 steps to see where wall-clock actually goes.
    # Toggle with PROFILE_STEPS env (default 3; 0 disables).
    profile_steps = int(os.environ.get('PROFILE_STEPS', 3))

    def _cuda_sync():
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()

    for step in range(start + 1, N_STEPS + 1):
        opt.zero_grad(set_to_none=True)
        # Tensor accumulator so we don't sync every microbatch. Only .item()
        # the sum once per LOG_EVERY, when we actually print it.
        accum_loss_t = torch.zeros((), device=DEVICE)
        do_profile = (step - start) <= profile_steps
        prof = {'data': 0.0, 'fwd': 0.0, 'bwd': 0.0}

        for _ in range(GRAD_ACCUM):
            if do_profile: _cuda_sync(); t_data = time.time()
            batch = next(it)
            b = _train_step_data(batch, DEVICE)
            if do_profile: _cuda_sync(); prof['data'] += time.time() - t_data; t_fwd = time.time()
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
                # Don't index by bool mask + branch on .numel() — that
                # forces a CPU sync every microbatch. Zero out invalid
                # entries and average over the valid count as a scalar.
                mask_f = mask.float()
                valid_count = mask_f.sum().clamp(min=1.0)
                loss = (losses * mask_f).sum() / valid_count
            if do_profile: _cuda_sync(); prof['fwd'] += time.time() - t_fwd; t_bwd = time.time()
            (loss / GRAD_ACCUM).backward()
            if do_profile: _cuda_sync(); prof['bwd'] += time.time() - t_bwd
            accum_loss_t = accum_loss_t + loss.detach()

        if do_profile: _cuda_sync(); t_opt = time.time()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        if do_profile:
            _cuda_sync()
            opt_dt = time.time() - t_opt
            total = prof['data'] + prof['fwd'] + prof['bwd'] + opt_dt
            if is_main:
                print(f'[profile step={step}]  data={prof["data"]:.2f}s  '
                      f'fwd={prof["fwd"]:.2f}s  bwd={prof["bwd"]:.2f}s  '
                      f'opt={opt_dt:.2f}s  total={total:.2f}s  '
                      f'(per microbatch: fwd={prof["fwd"]/GRAD_ACCUM*1000:.0f}ms  '
                      f'bwd={prof["bwd"]/GRAD_ACCUM*1000:.0f}ms)', flush=True)

        if step % LOG_EVERY == 0 or step == 1:
            avg_loss_t = (accum_loss_t / GRAD_ACCUM)
            # In DDP, average loss across ranks for the printed value.
            if using_dist:
                dist.all_reduce(avg_loss_t, op=dist.ReduceOp.SUM)
                avg_loss_t = avg_loss_t / world_size
            avg_loss = avg_loss_t.item()
            if is_main:
                print(f'{step:>7}  {avg_loss:>10.4f}  '
                      f'{opt.param_groups[0]["lr"]:>10.2e}  {time.time()-t0:>7.1f}s', flush=True)

        # Checkpointing: only rank 0 saves. All ranks read on resume.
        if step % CHECKPOINT_EVERY == 0:
            if is_main:
                if hasattr(opt, 'eval'): opt.eval()
                _save_model = model.module if using_dist and hasattr(model, 'module') else model
                save_checkpoint(os.path.join(CHECKPOINT_DIR, f'step_{step}.pt'),
                                step, _save_model, opt, edges)
                if hasattr(opt, 'train'): opt.train()
            if using_dist:
                dist.barrier(device_ids=[local_rank])

        if interrupted['flag']:
            if is_main:
                if hasattr(opt, 'eval'): opt.eval()
                _save_model = model.module if using_dist and hasattr(model, 'module') else model
                save_checkpoint(os.path.join(CHECKPOINT_DIR, f'step_{step}_interrupt.pt'),
                                step, _save_model, opt, edges)
            if using_dist:
                dist.barrier(device_ids=[local_rank])
                dist.destroy_process_group()
            sys.exit(0)

    if is_main:
        if hasattr(opt, 'eval'): opt.eval()
        _save_model = model.module if using_dist and hasattr(model, 'module') else model
        save_checkpoint(os.path.join(CHECKPOINT_DIR, f'step_{N_STEPS}_final.pt'),
                        N_STEPS, _save_model, opt, edges)
    if using_dist:
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
