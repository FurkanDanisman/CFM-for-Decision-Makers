"""Diagnostic: does the 9-region head activate under tightened inner edges?

Loads a cpfn2d checkpoint, forwards N_TASKS synthetic tasks from the same
training generator, extracts the mean softmax'd region weights per query,
reports the distribution across regions.

If tails are learning:  w_inner < 1.0 and tail regions carry non-trivial mass
If tails are dead:      w_inner ≈ 1.0 and tail regions ≈ 0 each

Env:
  CKPT          (required) path to a cpfn2d step_XXXX.pt checkpoint
  CAUSALPFN     (required) external/causalpfn root
  N_TASKS       default 20
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch


CKPT      = os.environ['CKPT']
CAUSALPFN = os.environ['CAUSALPFN']
N_TASKS   = int(os.environ.get('N_TASKS', 20))

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, CAUSALPFN)
sys.path.insert(0, CAUSALPFN + '/src')

# So `full_mixture_mean` is importable from the same dir
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

# Env vars the training-side meta-dataset builder expects
os.environ.setdefault('CAUSALPFN_ROOT', CAUSALPFN)
os.environ.setdefault('N_SAMPLES_PER_TASK', '1024')
os.environ.setdefault('MAX_N_COVARIATES', '50')
os.environ.setdefault('NUM_FEATURES', '99')

from training_causalpfn2d.model_causalpfn_2d import CausalPFN2DHead  # noqa: E402
from full_mixture_mean import unpack_2d_head  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

REGION_NAMES = [
    'inner',
    'L0',    'R0',    'L1',    'R1',           # edges: one axis outside
    'L0L1',  'L0R1',  'R0L1',  'R0R1',         # corners: both outside
]


def build_meta_dataset():
    yaml_path = os.path.join(CAUSALPFN, 'conf', 'meta_dataset', 'synthetic_backdoor.yaml')
    cfg = OmegaConf.load(yaml_path)
    cfg.n_samples           = 1024
    cfg.max_n_covariates    = 50
    cfg.post_padding_n_cols = 99
    return instantiate(cfg)


def load_model_from_ckpt(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'config' in ck:
        cfg = ck['config']
    else:
        mc = ck['model_config']
        cfg = dict(mc['model'])
        cfg['y_scaling_mode'] = mc.get('y_scaling_mode', 'pooled_std')
        cfg['loss_type']      = mc.get('loss_type',      'density')
        cfg['hlgauss_sigma']  = float(mc.get('hlgauss_sigma', 0.2))

    edge_lo = cfg.get('edge_lo', None)
    edge_hi = cfg.get('edge_hi', None)
    model = CausalPFN2DHead(
        J=cfg['J'], num_features=cfg['num_features'],
        ninp=cfg['ninp'], nhid=cfg['nhid'], nhead=cfg['nhead'],
        nlayers=cfg['nlayers'], dropout=cfg.get('dropout', 0.0),
        n_out=cfg.get('n_out', 10),
        y_scaling_mode=cfg.get('y_scaling_mode', 'pooled_std'),
        loss_type=cfg.get('loss_type', 'density'),
        hlgauss_sigma=float(cfg.get('hlgauss_sigma', 0.2)),
        edge_lo=edge_lo, edge_hi=edge_hi,
    ).to(DEVICE)

    sd = ck['model_state_dict']
    if any('_orig_mod.' in k for k in sd):
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    ref = model.state_dict()
    kept = {k: v for k, v in sd.items() if k in ref and ref[k].shape == v.shape}
    model.load_state_dict(kept, strict=False)
    model.eval()

    edges_np = model.edges.detach().cpu().numpy().astype(np.float64)
    return model, cfg, edges_np, ck.get('actual_step', ck.get('step', '?'))


@torch.no_grad()
def forward_logits(model, X_ctx, T_ctx, y_ctx, X_q):
    # match _forward_logits shape conventions
    X_ctx_t = torch.from_numpy(X_ctx.astype(np.float32)).unsqueeze(0).to(DEVICE)
    T_ctx_t = torch.from_numpy(T_ctx.astype(np.float32)).unsqueeze(0).to(DEVICE)
    Y_ctx_t = torch.from_numpy(y_ctx.astype(np.float32)).unsqueeze(0).to(DEVICE)
    X_q_t   = torch.from_numpy(X_q.astype(np.float32)).unsqueeze(0).to(DEVICE)
    # Standardise Y per the model's own scaling mode
    if model.y_scaling_mode == 'uwyk_minmax':
        y_lo = Y_ctx_t.amin(dim=1, keepdim=True); y_hi = Y_ctx_t.amax(dim=1, keepdim=True)
        sh = 0.5 * (y_lo + y_hi); sc = (0.5 * (y_hi - y_lo)).clamp(min=1e-6)
    else:
        sh = Y_ctx_t.mean(dim=1, keepdim=True); sc = Y_ctx_t.std(dim=1, keepdim=True).clamp(min=1e-6)
    y_std = (Y_ctx_t - sh) / sc
    logits = model._forward_logits(X_ctx_t, T_ctx_t, y_std, X_q_t)  # (1, N_q, J²+9+4)
    return logits.squeeze(0).float().cpu().numpy(), float(sc.item())


def main():
    print(f'[bootstrap] loading {CKPT}', flush=True)
    model, cfg, edges_np, step = load_model_from_ckpt(CKPT)
    J = cfg['J']
    print(f'[bootstrap] J={J}  edges=[{edges_np[0]:.3f}, {edges_np[-1]:.3f}]  step={step}  '
          f'y_scaling={model.y_scaling_mode}', flush=True)

    ds = build_meta_dataset()
    loader = DataLoader(ds, batch_size=1, num_workers=0)
    it = iter(loader)

    all_w = []
    all_tail_scales = []
    all_yctx_max_abs_std = []
    all_yq_frac_outside = []
    for t in range(N_TASKS):
        batch = next(it)
        X = np.asarray(batch['X']).astype(np.float32).reshape(-1, batch['X'].shape[-1])
        T = np.asarray(batch['t']).astype(np.float32).reshape(-1)
        y = np.asarray(batch['y']).astype(np.float32).reshape(-1)

        # Use half as ctx, other half as query (rough split)
        split = X.shape[0] // 2
        X_ctx = X[:split]; T_ctx = T[:split]; y_ctx = y[:split]
        X_q   = X[split:]

        # Pad X to num_features
        F = cfg['num_features']
        if X_ctx.shape[1] < F:
            X_ctx = np.hstack([X_ctx, np.zeros((X_ctx.shape[0], F - X_ctx.shape[1]), dtype=np.float32)])
            X_q   = np.hstack([X_q,   np.zeros((X_q.shape[0],   F - X_q.shape[1]),   dtype=np.float32)])

        logits_np, y_scale = forward_logits(model, X_ctx, T_ctx, y_ctx, X_q)
        # region weights
        unpacked = unpack_2d_head(logits_np, J)
        w = unpacked['w_reg']                 # (N_q, 9)
        all_w.append(w)

        # Diagnostic: how many context Y_std land outside inner region?
        y_ctx_std = (y_ctx - y_ctx.mean()) / max(y_ctx.std(), 1e-6)
        all_yctx_max_abs_std.append(float(np.abs(y_ctx_std).max()))
        # (query Y is unknown at inference; we just report ctx to show whether
        # the task falls in the "typical" or "tail" regime.)

    all_w = np.concatenate(all_w, axis=0)      # (Σ_N_q, 9)

    print(f'\n══ mean region weight over {all_w.shape[0]:,d} synthetic queries ({N_TASKS} tasks) ══')
    mean_w = all_w.mean(axis=0)
    for i, name in enumerate(REGION_NAMES):
        print(f'  {i} {name:<7}  w̄ = {mean_w[i]:8.5f}   frac_dominant = {float(np.mean(all_w.argmax(axis=-1) == i)):8.4f}')

    print(f'\n══ tail activation summary ══')
    tail_mass = mean_w[1:].sum()
    print(f'  total tail (regions 1..8) w̄     = {tail_mass:.5f}   ({tail_mass*100:5.2f}%)')
    print(f'  edge (regions 1..4)  w̄          = {mean_w[1:5].sum():.5f}')
    print(f'  corner (regions 5..8) w̄         = {mean_w[5:9].sum():.5f}')
    print(f'\n══ context |y_std| max — per-task distribution ══')
    y_max = np.array(all_yctx_max_abs_std)
    for q in (0.5, 0.75, 0.9, 1.0):
        print(f'  quantile {q:.2f}:  {np.quantile(y_max, q):5.2f}')
    print(f'  # tasks whose ctx has |y_std| > {abs(edges_np[0]):.1f}:  '
          f'{int(np.sum(y_max > abs(edges_np[0])))}/{N_TASKS}')


if __name__ == '__main__':
    main()
