"""Evaluate a graph-conditioned 2D-head checkpoint on the RealCause benchmark
suite (IHDP, ACIC, Lalonde CPS, Lalonde PSID, Lalonde PSID-balanced).

Per realization we run inference under TWO adjacency modes and derive CATE
under TWO estimators, giving 4 numbers per realization:

  * adjacency modes (both built by build_adjacency_matrix, which mirrors UWYK's
    dofm_full_conditioning.py::build_adjacency_matrix verbatim)
      - anc:   graph_mode=ANC_MODE, default "full_graph"
               (T→Y, all real X→T, all real X→Y); padded slots -1
      - noanc: graph_mode="all_unknown"
               (adjacency zeroed everywhere; padded slots still -1)
  * estimators (both on the marginals p_y0 = p.sum(-1), p_y1 = p.sum(-2))
      - raw: E[Y] = Σ_j centres[j] · p[j]
      - em:  fixed-point Gaussian correction (see MALC/malc_2d.py::_em_mean_2d;
             identical port lives in benchmarks/eval_causalpfn2d/eval_cpfn2d_ihdp_em.py)

PSID-balanced follows the recipe from
  ArikReuter/Graphs4CausalFoundationModels @ reproduce-realcause-results/
  RealCauseEval/run_baselines/dofm_psid_balanced.py
namely: keep all T=1 rows, then sample min(500, n_control) T=0 rows using
np.random.seed(42), concat + shuffle with RandomState(42). Only applied
when --dataset=PSID_bal.

Usage:
    CKPT=/scratch/.../checkpoints_graph2d/step_50000.pt \
    OUT=/scratch/.../results_graph2d_realcause \
    UWYK=/scratch/.../external/uwyk \
    CAUSALPFN=/scratch/.../external/causalpfn \
    DATASET=IHDP  (or ACIC / CPS / PSID / PSID_bal)
    python -u benchmarks/eval_graph2d/eval_graph2d_realcause.py
"""
from __future__ import annotations
import argparse
import math
import os
import sys
import time
import numpy as np
import torch
from scipy.stats import norm


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default=os.environ.get('DATASET', 'IHDP'),
                    choices=('IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal'))
args, _ = parser.parse_known_args()
DATASET = args.dataset

CKPT      = os.environ['CKPT']
OUT       = os.environ.get('OUT', f'./results_graph2d_realcause_{DATASET}')
UWYK      = os.environ['UWYK']
CAUSALPFN = os.environ['CAUSALPFN']

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, UWYK)
sys.path.insert(0, UWYK + '/src')
sys.path.insert(0, CAUSALPFN)
sys.path.insert(0, CAUSALPFN + '/src')

from benchmarks import (  # noqa: E402
    IHDPDataset,
    ACIC2016Dataset,
    RealCauseLalondeCPSDataset,
    RealCauseLalondePSIDDataset,
)
from training_graph2d.model_graph_2d import GraphConditioned2DHead  # noqa: E402
from losses.BarDistribution2D import (  # noqa: E402
    N_REGIONS, SOFTPLUS_FLOOR,
    R_INNER, R_L0, R_R0, R_L1, R_R1, R_L0L1, R_L0R1, R_R0L1, R_R0R1,
)

# ── Tail mass ─────────────────────────────────────────────────────────────
# The head is a 9-region mixture (see losses/BarDistribution2D.py):
#   logits[      : J*J    ] -> softmax -> p_mat : density SHAPE INSIDE the grid
#   logits[ J*J  : J*J+9  ] -> softmax -> w     : mass in each of the 9 regions
#   logits[ J*J+9:        ] -> 4 half-Gaussian tail scales (sL0,sR0,sL1,sR1)
#
# p_mat is a softmax, so it sums to 1 BY CONSTRUCTION -- it is the density
# conditional on landing inside the grid, NOT the marginal. The probability of
# actually being inside is w[R_INNER], a separate output. Reading only p_mat
# asserts w[R_INNER]==1, silently deletes all out-of-grid mass, and truncates
# E[Y] toward the centre -- worst exactly for large-|CATE| queries, which are
# the ones PEHE weights most.
#
# TAIL_MASS=1 (default) reconstructs the full mixture marginal.
# TAIL_MASS=0 restores the old interior-only behaviour for A/B comparison.
TAIL_MASS = os.environ.get('TAIL_MASS', '1') == '1'

# ── Real-block diagonal ───────────────────────────────────────────────────
# The model adds the identity to the adjacency before turning it into
# attention biases (PartialGraphConditionedInterventionalPFN.py:984):
#     A = clamp(A + I, -1, 1)
# so the diagonal you supply is NOT read literally:
#     you write -1  ->  +I -> 0   -> no self-bias      <- what training emits
#     you write  0  ->  +I -> +1  -> +bias_edge        <- what we emit
# Training always ends with propagate_ancestor_knowledge, which forces the
# diagonal to -1 (graph_utils.py:220,280), so the model has never seen a
# self-bias. Leaving the real-block diagonal at 0 gives every one of the
# n_real+2 real tokens an out-of-distribution positive boost on its own
# attention, in every feature-attention layer and head.
#
# UWYK's own eval scripts have this same gap (dofm_no_clustering.py:29-53),
# so DEFAULT IS OFF: the replica stays bit-identical to what the reproduce
# branch feeds its model. FIX_DIAG=1 writes the training-correct -1 instead.
# It is applied to BOTH graph modes, so the control arm stays clean.
FIX_DIAG = os.environ.get('FIX_DIAG', '0') == '1'

# Scale the learned soft-attention-bias params at inference. Smaller values
# soften the anc-induced attention shift: bias_edge=learned*scale means the
# +boost applied at anc edges is scale× smaller. Purpose: preserve some
# Y-self and T-self attention mass that anc otherwise crushes to near-zero,
# which the diagnostic identified as the mechanism through which anc
# degrades PEHE on ACIC/CPS. scale=1.0 = no change (default).
BIAS_EDGE_SCALE = float(os.environ.get('BIAS_EDGE_SCALE', '1.0'))

# Override the null_t_intv value at query time. Default '' = use the learned
# null token (current behavior). If set to a numeric string like '0.5', the
# query T slot uses that constant value instead of the learned null. Purpose:
# test whether providing a specific T value at query (mimicking the 1D head's
# T_intv-conditioning) restores information that the null_t_intv doesn't
# carry.
T_INTV_OVERRIDE = os.environ.get('T_INTV_OVERRIDE', '')

# 99% quantile outlier clipping on X features at eval, matching the
# reproduce-branch preprocessing (best_model_config.yaml: remove_outliers=true,
# outlier_quantile=0.99). Their training AND eval both clip; ours does
# neither. This eval-time clipping is a partial compensation attempt —
# brings our eval input distribution closer to what their model sees at
# eval, though our model wasn't trained with clipping. Default None (off).
X_CLIP_QUANTILE = os.environ.get('X_CLIP_QUANTILE', '')  # e.g. '0.99'

# Random-subsample context to this size if it exceeds. Matches reproduce
# branch's PreprocessingGraphConditionedPFN._pad_or_truncate_samples
# (max_number_train_samples_per_dataset=1000 in their config, is_train=True
# path uses np.random.choice with replace=False). Our training used
# N_CONTEXT_TRAIN=1000; feeding full context on ACIC (4321)/CPS (14500)/
# PSID (14500) at eval means 4-14× more context than training saw.
# Default '' = no subsampling (current behavior). Set e.g. '1000' to
# match training context size.
EVAL_MAX_CONTEXT = os.environ.get('EVAL_MAX_CONTEXT', '')
EVAL_CONTEXT_SEED = int(os.environ.get('EVAL_CONTEXT_SEED', '42'))

# Graph-knowledge mode for the positive ("anc") condition. Takes the same mode
# names as UWYK's dofm_full_conditioning.py: 'full_graph' (default),
# 't_to_y_only', 'x_to_t_only', 'x_to_y_only', 'all_unknown'. The baseline
# ("noanc") condition is always 'all_unknown'. Result keys stay `*_anc` /
# `*_noanc` regardless of which positive mode is selected.
ANC_MODE = os.environ.get('ANC_MODE', 'full_graph')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_dataset(name):
    if name == 'IHDP':
        return IHDPDataset()
    if name == 'ACIC':
        return ACIC2016Dataset()
    if name == 'CPS':
        return RealCauseLalondeCPSDataset()
    if name in ('PSID', 'PSID_bal'):
        return RealCauseLalondePSIDDataset()
    raise ValueError(name)


def _pad_features(X: np.ndarray, F: int) -> np.ndarray:
    if X.shape[1] == F:
        return X
    if X.shape[1] > F:
        return X[:, :F]
    pad = np.zeros((X.shape[0], F - X.shape[1]), dtype=X.dtype)
    return np.hstack([X, pad])


def _standardize_train_test(X_train, X_test, eps=1e-8):
    # Optional 99% quantile clipping (matches their preprocessing_config.
    # remove_outliers=true, outlier_quantile=0.99 in
    # best_model_config.yaml). Bounds computed on X_train per-column,
    # applied to BOTH train and test (same as their pipeline).
    if X_CLIP_QUANTILE:
        q = float(X_CLIP_QUANTILE)
        lo = np.quantile(X_train, 1.0 - q, axis=0, keepdims=True)
        hi = np.quantile(X_train, q,       axis=0, keepdims=True)
        X_train = np.clip(X_train, lo, hi)
        X_test  = np.clip(X_test,  lo, hi)
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, eps, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd


def _scale_y(y):
    ymin = float(y.min())
    ymax = float(y.max())
    yrange = max(ymax - ymin, 1e-9)
    return (2.0 * (y - ymin) / yrange - 1.0).astype(np.float32), ymin, yrange


def build_adjacency_matrix(model_n_features, n_real_features, graph_mode="full_graph"):
    """Build adjacency matrix based on graph knowledge mode."""
    # Initialize all as unknown (0)
    adjacency_matrix = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)

    T_idx = 0
    Y_idx = 1
    feature_offset = 2  # Features start at position 2

    if graph_mode == "all_unknown":
        msg = "ALL UNKNOWN (no graph information provided)"
    elif graph_mode == "full_graph":
        adjacency_matrix[T_idx, Y_idx] = 1.0
        for i in range(n_real_features):
            adjacency_matrix[feature_offset + i, T_idx] = 1.0
            adjacency_matrix[feature_offset + i, Y_idx] = 1.0
        msg = "T->Y=1, X->T=1, X->Y=1 (full graph)"
    else:
        raise ValueError(f"Unknown graph_mode: {graph_mode}")

    # PADDED features: Set all edges to -1 (no edge)
    for i in range(n_real_features, model_n_features):
        feat_idx = feature_offset + i
        adjacency_matrix[feat_idx, :] = -1.0
        adjacency_matrix[:, feat_idx] = -1.0
        adjacency_matrix[feat_idx, feat_idx] = -1.0

    # REAL block diagonal: -1 ("not its own ancestor") is what training emits,
    # and is the only value that yields no self-attention bias. See FIX_DIAG.
    # Safe after the padded loop: that loop only touches padded indices.
    if FIX_DIAG:
        for i in range(feature_offset + n_real_features):
            adjacency_matrix[i, i] = -1.0

    # Called once per realization per mode; log each distinct mode only once.
    _seen = getattr(build_adjacency_matrix, '_seen_modes', set())
    if graph_mode not in _seen:
        _seen.add(graph_mode)
        build_adjacency_matrix._seen_modes = _seen
        _diag = 'real-diag=-1 (FIX_DIAG)' if FIX_DIAG else 'real-diag=0 (UWYK-faithful)'
        print(f"Graph knowledge: {msg}  [{_diag}]", flush=True)

    return adjacency_matrix


# ── PSID-balanced subsample (mirrors dofm_psid_balanced.py verbatim) ────
def psid_balance_subsample(X_train, t_train, y_train):
    """all T=1 + up to 500 T=0 sampled with np.random.seed(42), shuffle with
    RandomState(42). Matches ArikReuter reproduce-realcause-results branch.
    """
    t_flat = t_train.reshape(-1)
    treated = (t_flat == 1)
    control = (t_flat == 0)

    X_tr = X_train[treated]; t_tr = t_train[treated]; y_tr = y_train[treated]
    X_ct = X_train[control]; t_ct = t_train[control]; y_ct = y_train[control]

    n_control = X_ct.shape[0]
    n_keep = min(500, n_control)
    if n_control > n_keep:
        np.random.seed(42)
        idx = np.random.choice(n_control, n_keep, replace=False)
        X_ct = X_ct[idx]; t_ct = t_ct[idx]; y_ct = y_ct[idx]
        print(f'[PSID-bal] kept {X_tr.shape[0]} treated, sampled {n_keep}/{n_control} controls',
              flush=True)
    else:
        print(f'[PSID-bal] kept {X_tr.shape[0]} treated, all {n_control} controls',
              flush=True)

    X = np.vstack([X_tr, X_ct]); t = np.concatenate([t_tr, t_ct]); y = np.concatenate([y_tr, y_ct])
    perm = np.random.RandomState(42).permutation(X.shape[0])
    return X[perm], t[perm], y[perm]


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ck['config']

    sd = ck['model_state_dict']
    if any('_orig_mod.' in k for k in sd):
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}

    def _sink_count(prefix):
        # Buffer shapes in the saved state_dict:
        #   sink_rows_x: (1, num_sinks, F+2, d_model)
        #   sink_rows_y: (1, num_sinks, d_model)
        # Leading dim is a batch broadcast (fixed at 1). The actual sink count
        # lives at dim 1. Reading shape[0] gave us 1 → model was built with
        # n_sample_attention_sink_rows=1 → sink buffers had a different shape
        # than the ckpt's → load_state_dict SILENTLY DROPPED them due to the
        # shape filter → the model ran with sink buffers at random init.
        # Fix: if there's a leading batch dim (shape[0]==1) use shape[1],
        # otherwise fall back to shape[0] (older ckpts without batch dim).
        for suffix in ('_x', '_y'):
            k = prefix + suffix
            if k not in sd or sd[k].dim() < 2:
                continue
            t = sd[k]
            return int(t.shape[1] if t.shape[0] == 1 else t.shape[0])
        return 0

    model = GraphConditioned2DHead(
        num_features=cfg['num_features'],
        d_model=cfg['d_model'],
        depth=cfg['depth'],
        heads_feat=cfg['heads'],
        heads_samp=cfg['heads'],
        dropout=0.0,
        hidden_mult=cfg['hidden_mult'],
        normalize_features=True,
        J=cfg['J'],
        n_sample_attention_sink_rows=_sink_count('sink_rows'),
        n_feature_attention_sink_cols=_sink_count('sink_cols'),
    ).to(DEVICE)

    ref = model.state_dict()
    kept = {k: v for k, v in sd.items() if k in ref and ref[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(kept, strict=False)
    print(f'[load_model] missing={len(missing)} unexpected={len(unexpected)}  '
          f'loaded={len(kept)}/{len(ref)}  step={ck.get("step")}', flush=True)
    if len(missing) > 5:
        raise RuntimeError(f'[load_model] ABORT: {len(missing)} missing keys')

    # Apply BIAS_EDGE_SCALE at eval — multiply learned soft-attention-bias
    # params in-place. Both edge and no_edge biases scaled together so the
    # ratio between them is preserved (only the overall magnitude of the
    # anc-induced attention shift changes).
    if abs(BIAS_EDGE_SCALE - 1.0) > 1e-6:
        n_scaled = 0
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.endswith('bias_edge') or name.endswith('bias_no_edge'):
                    p.mul_(BIAS_EDGE_SCALE)
                    n_scaled += 1
        print(f'[load_model] scaled {n_scaled} bias_edge/bias_no_edge params '
              f'by {BIAS_EDGE_SCALE}', flush=True)

    model.eval()

    # Bin edges are part of the model's meaning: they map bucket indices back
    # to outcome values, and they set bin_width, which scales the tail sigmas.
    # Training fits them once and saves them (train_graph_2d.py::save_checkpoint),
    # so read them rather than reconstructing.
    _e = ck.get('edges', None)
    if _e is None:
        edges = np.linspace(-1.0, 1.0, cfg['J'] + 1, dtype=np.float64)
        print(f"[load_model] WARNING: checkpoint has no 'edges' key; "
              f"falling back to linspace(-1, 1, {cfg['J'] + 1})", flush=True)
    else:
        edges = np.asarray(_e.cpu() if hasattr(_e, 'cpu') else _e, dtype=np.float64)
        print(f'[load_model] edges from ckpt: [{edges[0]:.6f}, {edges[-1]:.6f}]  '
              f'n={len(edges)}  bin_width={(edges[-1] - edges[0]) / cfg["J"]:.6g}', flush=True)

    return model, cfg, edges


# ── EM-mean (ported from eval_cpfn2d_ihdp_em.py::_em_mean_1d) ───────────
def _em_mean_1d(props, grid, sigma, start,
                max_step=1000, eps2=1e-10, eps1=1e-5):
    pn = props / max(props.sum(), 1e-45)
    mu = start
    for _ in range(max_step):
        a = (grid - mu) / sigma
        G1 = norm.cdf(a); G2 = norm.pdf(a)
        temp = (np.diff(G2) + eps2) / (np.diff(G1) + eps2)
        mu_new = mu - sigma * float(np.sum(pn * temp))
        if abs(mu_new - mu) < eps1:
            return float(mu_new)
        mu = mu_new
    return float(mu)


def _marginal_stats(p, grid):
    """Seed (mu, sigma) for the EM fixed-point on a 1D marginal."""
    delta = grid[1] - grid[0]
    centres = 0.5 * (grid[:-1] + grid[1:])
    mu_low = float(np.sum(p * grid[:-1]))
    mu_mid = 0.5 * (mu_low + float(np.sum(p * grid[1:])))
    sigma = float(np.sqrt(np.sum(p * (centres - mu_mid) ** 2) + delta ** 2 / 12.0))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = delta
    return mu_mid, sigma



# ── 9-region mixture -> per-arm marginal decomposition ────────────────────
_HALFNORM_MEAN = math.sqrt(2.0 / math.pi)   # E|N(0,1)|


def _norm_rows(v):
    """Normalise the last axis to sum to 1."""
    return v / v.sum(-1, keepdim=True).clamp_min(1e-45)


def _decompose_head(logits, J, edges):
    """Split raw head output into a per-arm marginal decomposition.

    Returns (arm0, arm1) for Y_do0 / Y_do1. Each is a dict of numpy arrays
    with leading query dim:

        p     (M, J)  in-grid density shape, normalised to sum 1
        w_in  (M,)    P(this arm's outcome lands inside the grid)
        w_lo  (M,)    P(below grid)   m_lo (M,)  its conditional mean
        w_hi  (M,)    P(above grid)   m_hi (M,)  its conditional mean

    with w_in + w_lo + w_hi == 1, so

        E[Y] = w_in * E[Y | inside] + w_lo * m_lo + w_hi * m_hi

    Region -> arm mapping (see losses/BarDistribution2D.py region table):
      Y_do0 inside : R_INNER, R_L1, R_R1     below : R_L0, R_L0L1, R_L0R1
      Y_do1 inside : R_INNER, R_L0, R_R0     below : R_L1, R_L0L1, R_R0L1
    In the mixed regions the in-grid shape is the boundary row/column of
    p_mat, matching how neg_log_prob_2d builds those conditionals.

    Approximation: in the four corner regions the loss uses a rho-coupled
    bivariate half-Gaussian, whose per-axis marginal is not exactly a
    half-normal. We use the half-normal mean anyway -- exact at rho=0
    (verified against Monte Carlo), and corner weights are typically small.
    """
    JJ = J * J
    p_mat = torch.softmax(logits[..., :JJ].float(), dim=-1)
    p_mat = p_mat.reshape(*logits.shape[:-1], J, J)

    def _np(t):
        return t.squeeze(0).cpu().numpy()

    if not TAIL_MASS:
        # Legacy: assert w[R_INNER] == 1 and drop everything outside the grid.
        ones = torch.ones(p_mat.shape[:-2], device=p_mat.device)
        zeros = torch.zeros_like(ones)
        legacy = lambda pg: dict(p=_np(pg), w_in=_np(ones),
                                 w_lo=_np(zeros), m_lo=_np(zeros),
                                 w_hi=_np(zeros), m_hi=_np(zeros))
        return legacy(p_mat.sum(-1)), legacy(p_mat.sum(-2))

    w = torch.softmax(logits[..., JJ:JJ + N_REGIONS].float(), dim=-1)
    tail_raw = logits[..., JJ + N_REGIONS:].float()

    lo, hi = float(edges[0]), float(edges[-1])
    bin_width = (hi - lo) / J
    sig = bin_width * (torch.nn.functional.softplus(tail_raw) + SOFTPLUS_FLOOR)
    sL0, sR0, sL1, sR1 = sig[..., 0], sig[..., 1], sig[..., 2], sig[..., 3]

    # ---- Y_do0: sum over the Y_do1 axis ----
    p0 = (w[..., R_INNER, None] * p_mat.sum(-1)
          + w[..., R_L1, None] * _norm_rows(p_mat[..., :, 0])
          + w[..., R_R1, None] * _norm_rows(p_mat[..., :, -1]))
    arm0 = dict(
        p=_np(_norm_rows(p0)),
        w_in=_np(w[..., R_INNER] + w[..., R_L1] + w[..., R_R1]),
        w_lo=_np(w[..., R_L0] + w[..., R_L0L1] + w[..., R_L0R1]),
        w_hi=_np(w[..., R_R0] + w[..., R_R0L1] + w[..., R_R0R1]),
        m_lo=_np(lo - sL0 * _HALFNORM_MEAN),
        m_hi=_np(hi + sR0 * _HALFNORM_MEAN),
    )

    # ---- Y_do1: sum over the Y_do0 axis ----
    p1 = (w[..., R_INNER, None] * p_mat.sum(-2)
          + w[..., R_L0, None] * _norm_rows(p_mat[..., 0, :])
          + w[..., R_R0, None] * _norm_rows(p_mat[..., -1, :]))
    arm1 = dict(
        p=_np(_norm_rows(p1)),
        w_in=_np(w[..., R_INNER] + w[..., R_L0] + w[..., R_R0]),
        w_lo=_np(w[..., R_L1] + w[..., R_L0L1] + w[..., R_R0L1]),
        w_hi=_np(w[..., R_R1] + w[..., R_L0R1] + w[..., R_R0R1]),
        m_lo=_np(lo - sL1 * _HALFNORM_MEAN),
        m_hi=_np(hi + sR1 * _HALFNORM_MEAN),
    )
    return arm0, arm1


def _slice_arm(a, n):
    return {k: v[:n] for k, v in a.items()}


def _cat_arms(parts):
    return {k: np.concatenate([a[k] for a in parts], axis=0) for k in parts[0]}


def _log_tail_mass(arm0, arm1, adj):
    """Once per (dataset, mode): report how much mass sits outside the grid.

    If the two adjacency modes differ here, the interior-only bug was a
    DIFFERENTIAL error between them, not a common-mode one.
    """
    mode = 'anc' if bool((adj > 0).any()) else 'noanc'
    seen = getattr(_log_tail_mass, '_seen', set())
    key = (DATASET, mode, TAIL_MASS)
    if key in seen:
        return
    seen.add(key)
    _log_tail_mass._seen = seen
    print(f'[tail-mass] {DATASET:<9} mode={mode:<5} TAIL_MASS={int(TAIL_MASS)}  '
          f'mean w_in: y0={arm0["w_in"].mean():.4f} y1={arm1["w_in"].mean():.4f}  |  '
          f'out-of-grid: y0={1 - arm0["w_in"].mean():.4f} y1={1 - arm1["w_in"].mean():.4f}',
          flush=True)


@torch.no_grad()
def marginals_from_forward(model, X_train, T_train, Y_train_scaled, X_test, adj, J, edges):
    """Run one forward pass; return the per-arm marginal decomposition
    (arm0, arm1) described in _decompose_head."""
    B = 1
    X_obs = torch.from_numpy(X_train.astype(np.float32)).unsqueeze(0).to(DEVICE)
    T_obs = torch.from_numpy(T_train.astype(np.float32)).reshape(1, -1, 1).to(DEVICE)
    Y_obs = torch.from_numpy(Y_train_scaled).unsqueeze(0).to(DEVICE)
    X_intv = torch.from_numpy(X_test.astype(np.float32)).unsqueeze(0).to(DEVICE)
    adj_t = torch.from_numpy(adj).unsqueeze(0).to(DEVICE)

    # ── EVAL DIAGNOSTIC ── dump anc-matrix + tensor stats on first forward
    # per dataset (opt-in via EVAL_DIAG=1). Verifies that what the model
    # actually receives matches the training-time distribution: values
    # strictly in {-1, 0, +1}, correct shape, propagate applied, X in the
    # right scale, Y in the right scale.
    if os.environ.get('EVAL_DIAG', '0') == '1':
        with torch.no_grad():
            a = adj_t.detach().cpu()
            _mode_tag = 'anc' if bool((a > 0).any()) else 'noanc'
            _seen = getattr(marginals_from_forward, '_diag_seen', set())
            _key = (DATASET, _mode_tag)
            if _key not in _seen:
                _seen.add(_key)
                marginals_from_forward._diag_seen = _seen
                uniq, counts = torch.unique(a, return_counts=True)
                uniq_dict = {float(v): int(c) for v, c in zip(uniq.tolist(), counts.tolist())}
                other_mask = ~((a == -1) | (a == 0) | (a == +1))
                print(f'\n[eval-diag] ══ DATASET={DATASET}  MODE={_mode_tag} ══', flush=True)
                print(f'[eval-diag] adj shape={tuple(a.shape)}  dtype={a.dtype}', flush=True)
                print(f'[eval-diag]   min={float(a.min()):.4f}  max={float(a.max()):.4f}', flush=True)
                print(f'[eval-diag]   frac_neg1={float((a==-1).float().mean()):.4f}  '
                      f'frac_zero={float((a==0).float().mean()):.4f}  '
                      f'frac_pos1={float((a==+1).float().mean()):.4f}', flush=True)
                print(f'[eval-diag]   OUT_OF_SET_count={int(other_mask.sum())}   '
                      f'(should be 0; if >0 we have a bug)', flush=True)
                print(f'[eval-diag]   unique-values-counts={uniq_dict}', flush=True)
                _r = min(a.shape[-1], 8)
                print(f'[eval-diag]   adj[:{_r},:{_r}] =\n{a[0, :_r, :_r].numpy()}', flush=True)
                print(f'[eval-diag] X_obs  shape={tuple(X_obs.shape)}  '
                      f'min={float(X_obs.min()):.3f} max={float(X_obs.max()):.3f} '
                      f'mean={float(X_obs.mean()):.3f} std={float(X_obs.std()):.3f}', flush=True)
                print(f'[eval-diag] Y_obs  shape={tuple(Y_obs.shape)}  '
                      f'min={float(Y_obs.min()):.3f} max={float(Y_obs.max()):.3f} '
                      f'mean={float(Y_obs.mean()):.3f} std={float(Y_obs.std()):.3f}', flush=True)
                print(f'[eval-diag] T_obs  shape={tuple(T_obs.shape)}  '
                      f'unique={torch.unique(T_obs).tolist()}', flush=True)
                print(f'[eval-diag] X_intv shape={tuple(X_intv.shape)}  '
                      f'min={float(X_intv.min()):.3f} max={float(X_intv.max()):.3f}', flush=True)

    # ── MATCH_UWYK_PADDING ── replicate UWYK's _pad_or_truncate_samples
    # (is_train=False) behavior at eval: batch queries to max_n_test=1000,
    # pad each batch's tail with zero rows, forward, drop padded predictions.
    # UWYK's reproduce config uses max_number_test_samples_per_dataset=1000.
    _max_n_test = int(os.environ.get('MATCH_UWYK_PADDING', '0'))
    if _max_n_test > 0:
        M_real = X_intv.shape[1]
        F_dim = X_intv.shape[2]
        p_y0_all, p_y1_all = [], []
        for start in range(0, M_real, _max_n_test):
            end = min(start + _max_n_test, M_real)
            n_batch_real = end - start
            X_intv_batch = X_intv[:, start:end, :]
            if n_batch_real < _max_n_test:
                pad_rows = _max_n_test - n_batch_real
                zero_pad = torch.zeros((1, pad_rows, F_dim), dtype=X_intv.dtype, device=DEVICE)
                X_intv_batch = torch.cat([X_intv_batch, zero_pad], dim=1)
            if T_INTV_OVERRIDE:
                t_val = float(T_INTV_OVERRIDE)
                T_intv_batch = torch.full((1, _max_n_test, 1), t_val,
                                          dtype=X_intv.dtype, device=DEVICE)
                out = model(X_obs, T_obs, Y_obs, X_intv_batch, adj_t, T_intv=T_intv_batch)
            else:
                out = model(X_obs, T_obs, Y_obs, X_intv_batch, adj_t)
            logits = out['predictions'] if isinstance(out, dict) else out
            a0_batch, a1_batch = _decompose_head(logits, J, edges)
            # Drop padded rows before appending
            p_y0_all.append(_slice_arm(a0_batch, n_batch_real))
            p_y1_all.append(_slice_arm(a1_batch, n_batch_real))
        arm0, arm1 = _cat_arms(p_y0_all), _cat_arms(p_y1_all)
        _log_tail_mass(arm0, arm1, adj)
        return arm0, arm1

    # T_INTV_OVERRIDE: if set, feed a specific T_intv value at query instead
    # of the learned null_t_intv. Otherwise (default) the model's forward
    # fills in null_t_intv itself.
    if T_INTV_OVERRIDE:
        t_val = float(T_INTV_OVERRIDE)
        M = X_intv.shape[1]
        T_intv = torch.full((1, M, 1), t_val, dtype=X_intv.dtype, device=DEVICE)
        out = model(X_obs, T_obs, Y_obs, X_intv, adj_t, T_intv=T_intv)
    else:
        out = model(X_obs, T_obs, Y_obs, X_intv, adj_t)
    logits = out['predictions'] if isinstance(out, dict) else out
    arm0, arm1 = _decompose_head(logits, J, edges)
    _log_tail_mass(arm0, arm1, adj)
    return arm0, arm1


def _arm_mean(arm, edges, centres, use_em):
    """E[Y] for one arm under the full 9-region mixture.

    Both estimators only ever supply E[Y | inside the grid]; the tail terms
    are added identically on top, so 'raw' vs 'em' stays a like-for-like
    comparison of how the in-grid mean is computed.
    """
    p = arm['p']
    if use_em:
        e_in = np.empty(p.shape[0])
        for q in range(p.shape[0]):
            mu, sg = _marginal_stats(p[q], edges)
            e_in[q] = _em_mean_1d(p[q], edges, sg, mu)
    else:
        e_in = (p * centres[None, :]).sum(axis=-1)
    return arm['w_in'] * e_in + arm['w_lo'] * arm['m_lo'] + arm['w_hi'] * arm['m_hi']


def cate_from_marginals(arm0, arm1, J, edges):
    """Return (cate_raw, cate_em) on the model's (scaled) outcome axis."""
    edges   = np.asarray(edges, dtype=np.float64)
    centres = 0.5 * (edges[:-1] + edges[1:])

    cate_raw = _arm_mean(arm1, edges, centres, False) - _arm_mean(arm0, edges, centres, False)
    cate_em  = _arm_mean(arm1, edges, centres, True)  - _arm_mean(arm0, edges, centres, True)

    return cate_raw.astype(np.float32), cate_em.astype(np.float32)


def evaluate(realization, ds, model, J, F, apply_psid_balance, edges):
    cate_ds = ds[realization][0]
    X_tr_raw = np.asarray(cate_ds.X_train, dtype=np.float32)
    T_tr     = np.asarray(cate_ds.t_train, dtype=np.float32).reshape(-1)
    y_tr_raw = np.asarray(cate_ds.y_train, dtype=np.float32).reshape(-1)
    X_te_raw = np.asarray(cate_ds.X_test,  dtype=np.float32)
    true_cate = np.asarray(cate_ds.true_cate, dtype=np.float32)
    true_ate  = float(true_cate.mean())

    if apply_psid_balance:
        X_tr_raw, T_tr, y_tr_raw = psid_balance_subsample(X_tr_raw, T_tr, y_tr_raw)

    # Match reproduce branch's max_number_train_samples_per_dataset by
    # random-subsampling context if it exceeds EVAL_MAX_CONTEXT. Deterministic
    # per-realization seed so the same context is picked every run.
    if EVAL_MAX_CONTEXT:
        cap = int(EVAL_MAX_CONTEXT)
        n_ctx = X_tr_raw.shape[0]
        if n_ctx > cap:
            rng = np.random.default_rng(EVAL_CONTEXT_SEED + realization)
            idx = rng.choice(n_ctx, cap, replace=False)
            X_tr_raw = X_tr_raw[idx]; T_tr = T_tr[idx]; y_tr_raw = y_tr_raw[idx]
            print(f'  [context-subsample] r={realization}  {n_ctx} → {cap} '
                  f'(seed={EVAL_CONTEXT_SEED + realization})', flush=True)

    # Clamp n_real to F: _pad_features TRUNCATES when the dataset has more
    # covariates than the model was trained on (ACIC: 55 vs F=50). Everything
    # past index F is dropped, so it never enters the adjacency matrix.
    n_real = min(X_tr_raw.shape[1], F)
    X_tr_std, X_te_std = _standardize_train_test(X_tr_raw, X_te_raw)
    X_tr = _pad_features(X_tr_std, F)
    X_te = _pad_features(X_te_std, F)
    y_scaled, ymin, yrange = _scale_y(y_tr_raw)
    Y_obs = y_scaled.reshape(-1, 1)

    results = {}
    _mode_list = (('anc',   build_adjacency_matrix(F, n_real, ANC_MODE)),
                  ('noanc', build_adjacency_matrix(F, n_real, 'all_unknown')))
    for mode, adj in _mode_list:
        arm0, arm1 = marginals_from_forward(model, X_tr, T_tr, Y_obs, X_te, adj, J, edges)
        cate_raw_scaled, cate_em_scaled = cate_from_marginals(arm0, arm1, J, edges)
        # Un-scale to raw Y units. (2 * cate_scaled / 2) * yrange / 2 = cate_scaled * yrange / 2.
        for method, cate_scaled in (('raw', cate_raw_scaled), ('em', cate_em_scaled)):
            cate = cate_scaled * yrange / 2.0
            pehe = float(np.sqrt(np.mean((cate - true_cate) ** 2)))
            ate_hat = float(cate.mean())
            err_ate = abs(ate_hat - true_ate) / max(abs(true_ate), 1e-9)
            results[f'pehe_{method}_{mode}'] = pehe
            results[f'err_{method}_{mode}']  = err_ate
            results[f'ate_{method}_{mode}']  = ate_hat

    return {
        'dataset': DATASET,
        'realization': realization,
        'true_ate': true_ate,
        'n_queries': int(true_cate.size),
        'n_context': int(X_tr_raw.shape[0]),
        **results,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f'[bootstrap] dataset={DATASET}  device={DEVICE}  ckpt={CKPT}  out={OUT}',
          flush=True)

    ds = get_dataset(DATASET)
    apply_psid_balance = (DATASET == 'PSID_bal')
    print(f'[bootstrap] {DATASET} n_tables={ds.n_tables}  psid_bal={apply_psid_balance}',
          flush=True)

    model, cfg, edges = load_model(CKPT)
    J = cfg['J']; F = cfg['num_features']
    print(f'[bootstrap] J={J}  F={F}', flush=True)

    rows = []
    t0 = time.time()
    # MAX_REAL: cap number of realizations (e.g. MAX_REAL=1 for a fast
    # diagnostic run of a single realization per dataset).
    _cap = int(os.environ.get('MAX_REAL', ds.n_tables))
    for r in range(min(ds.n_tables, _cap)):
        row = evaluate(r, ds, model, J, F, apply_psid_balance, edges)
        rows.append(row)
        np.savez(os.path.join(OUT, f'{DATASET}_r{r:03d}.npz'),
                 **{k: np.array(v) for k, v in row.items()})
        print(
            f'r={r:03d}  '
            f'raw-anc: pehe={row["pehe_raw_anc"]:6.3f} err={row["err_raw_anc"]:5.3f}  |  '
            f'em-anc: pehe={row["pehe_em_anc"]:6.3f} err={row["err_em_anc"]:5.3f}  |  '
            f'raw-noanc: pehe={row["pehe_raw_noanc"]:6.3f} err={row["err_raw_noanc"]:5.3f}  |  '
            f'em-noanc: pehe={row["pehe_em_noanc"]:6.3f} err={row["err_em_noanc"]:5.3f}  '
            f'({time.time()-t0:.0f}s)',
            flush=True,
        )

    def ms(k):
        v = np.array([r[k] for r in rows if np.isfinite(r[k])])
        if v.size < 2: return float('nan'), float('nan'), int(v.size)
        return v.mean(), v.std(ddof=1) / np.sqrt(len(v)), int(v.size)

    print(f'\n══ {DATASET} summary (n={len(rows)}) ══')
    keys = ['pehe_raw_anc', 'err_raw_anc',
            'pehe_em_anc',  'err_em_anc',
            'pehe_raw_noanc', 'err_raw_noanc',
            'pehe_em_noanc',  'err_em_noanc']
    for k in keys:
        m, s, n = ms(k)
        print(f'  {k:20s} = {m:8.3f} ± {s:6.3f}   (n={n})')


if __name__ == '__main__':
    main()
