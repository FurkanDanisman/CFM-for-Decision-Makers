"""Evaluate a graph-conditioned 2D-head checkpoint on the RealCause benchmark
suite (IHDP, ACIC, Lalonde CPS, Lalonde PSID, Lalonde PSID-balanced).

Per realization we run inference under TWO adjacency modes and derive CATE
under TWO estimators, giving 4 numbers per realization:

  * adjacency modes
      - anc:   true graph (T→Y, all real X→T, all real X→Y); padded slots -1
      - noanc: adjacency zeroed everywhere (padded slots still -1)
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
import os
import sys
import time
import numpy as np
import torch
from scipy.stats import norm


parser = argparse.ArgumentParser()
_SCM_CASES = ('Observed_Confounder', 'Observed_Mediator',
              'Observed_Mediator_and_Confounder', 'Unobserved_Confounder',
              'Frontdoor_Criterion', 'Backdoor_Criterion')
parser.add_argument('--dataset', type=str,
                    default=os.environ.get('DATASET', 'IHDP'),
                    choices=('IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal') + _SCM_CASES)
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
from utils.graph_utils import propagate_ancestor_knowledge  # noqa: E402  # UWYK

# Match training-time anc matrix distribution: after building the +1
# ancestor edges we assert, run propagate_ancestor_knowledge to fill in
# the -1s at (a) diagonal (irreflexive), (b) antisymmetric reverse edges,
# and derive any transitive +1s. This is the SAME function training uses
# (PairedInterventionalDataset.py:754). If we skip this step, our eval
# matrix has 0 at positions where training would have -1, which is a
# distribution shift on the input the soft-attention-bias params were
# calibrated for. Toggle via env var; default ON (match training).
PROPAGATE_ANC = os.environ.get('PROPAGATE_ANC', '1') == '1'
# Assume features are independent confounders: fill in -1 at X_i ↔ X_j
# for i != j. Propagate can't derive these (no +1 chain triggers them).
# Training SCMs typically produce mostly independent covariates so the
# anc matrix has -1s there — leaving 0 at eval creates a distribution
# shift proportional to n_real^2 (small for Lalonde n_real=8, huge for
# ACIC n_real=50).
INDEP_FEATURES = os.environ.get('INDEP_FEATURES', '0') == '1'

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
EVAL_CONTEXT_SEED = int(os.environ.get('EVAL_CONTEXT_SEED', '1'))
# PSID-bal subsampling seed. Kept at 42 (matches reproduce branch).
# Env-controllable but default preserves reproduce-branch consistency.
PSID_BAL_SEED = int(os.environ.get('PSID_BAL_SEED', '42'))

# Anc-content probe. Default 'full' = original T→Y + X→T + X→Y +1 edges.
# 'ty_only' = only T→Y = +1; X→T and X→Y left as 0 (unknown). Tests whether
# the model's degradation is caused by over-attending to the X→T/X→Y edges
# specifically. In 'ty_only' mode the results dict uses key `pehe_raw_ty`
# (and `pehe_em_ty`, etc.) instead of `pehe_raw_anc`.
ANC_MODE = os.environ.get('ANC_MODE', 'full')


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
    if name in _SCM_CASES:
        import sys as _sys
        _rp_bench = os.path.join(REPO_SRC, 'benchmarks')
        if _rp_bench not in _sys.path: _sys.path.insert(0, _rp_bench)
        from scm_case_study_dataset import SCMCaseStudyDataset
        return SCMCaseStudyDataset(name)
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


def build_anc_full(F, n_real):
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    T_idx, Y_idx, feat_off = 0, 1, 2
    A[T_idx, Y_idx] = 1.0
    for i in range(n_real):
        A[feat_off + i, T_idx] = 1.0
        A[feat_off + i, Y_idx] = 1.0
    for i in range(n_real, F):
        A[feat_off + i, :] = -1.0
        A[:, feat_off + i] = -1.0
        A[feat_off + i, feat_off + i] = -1.0

    if INDEP_FEATURES:
        # X_i and X_j (i != j) are assumed independent → neither is ancestor
        # of the other → both A[2+i, 2+j] and A[2+j, 2+i] = -1. Set BEFORE
        # propagate so it can chain from these too.
        feat_off = 2
        for i in range(n_real):
            for j in range(n_real):
                if i != j:
                    A[feat_off + i, feat_off + j] = -1.0

    if PROPAGATE_ANC:
        # Fill in -1s at antisymmetric reverse edges + diagonal, and any
        # transitive +1s. Restricted to the REAL submatrix so we don't
        # perturb the padded -1s.
        import torch as _torch
        real_n = 2 + n_real
        real_block = _torch.from_numpy(A[:real_n, :real_n].copy())
        real_block = propagate_ancestor_knowledge(real_block, raise_on_inconsistent=False)
        A[:real_n, :real_n] = real_block.numpy().astype(np.float32)
    return A


def build_anc_none(F, n_real):
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    feat_off = 2
    for i in range(n_real, F):
        A[feat_off + i, :] = -1.0
        A[:, feat_off + i] = -1.0
        A[feat_off + i, feat_off + i] = -1.0
    return A


def build_anc_ty_only(F, n_real):
    """T→Y only anc. Real block is 0 except A[T,Y]=+1. Padded features −1
    on their rows/cols like the other modes. Probes whether the model's
    trouble is over-attending to X→T / X→Y edges specifically."""
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    T_idx, Y_idx, feat_off = 0, 1, 2
    A[T_idx, Y_idx] = 1.0
    for i in range(n_real, F):
        A[feat_off + i, :] = -1.0
        A[:, feat_off + i] = -1.0
        A[feat_off + i, feat_off + i] = -1.0
    if PROPAGATE_ANC:
        from utils.graph_utils import propagate_ancestor_knowledge  # noqa: E402
        real_n = 2 + n_real
        real_block = torch.from_numpy(A[:real_n, :real_n].copy())
        real_block = propagate_ancestor_knowledge(real_block, raise_on_inconsistent=False)
        A[:real_n, :real_n] = real_block.numpy().astype(np.float32)
    return A


def build_anc_ty_antisym(F, n_real):
    """T→Y = +1 AND Y→T = -1 explicitly. Diagonal 0, all X↔X 0, all X↔T
    and X↔Y 0. Padded features -1 on their rows/cols. No propagation
    (independent of PROPAGATE_ANC env var)."""
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    T_idx, Y_idx, feat_off = 0, 1, 2
    A[T_idx, Y_idx] = 1.0
    A[Y_idx, T_idx] = -1.0
    for i in range(n_real, F):
        A[feat_off + i, :] = -1.0
        A[:, feat_off + i] = -1.0
        A[feat_off + i, feat_off + i] = -1.0
    return A


def _padded_neg1_only(F, n_real):
    """Common helper: return a matrix with padded rows/cols all -1 and
    the real block untouched (all 0). Caller adds their real-block edges."""
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    feat_off = 2
    for i in range(n_real, F):
        A[feat_off + i, :] = -1.0
        A[:, feat_off + i] = -1.0
        A[feat_off + i, feat_off + i] = -1.0
    return A


# ── Variant builders (all no-propagate, all padded region -1) ──────────
def build_anc_v1a(F, n_real):
    """v1a: T→Y = +1. Rest of real block 0."""
    A = _padded_neg1_only(F, n_real)
    A[0, 1] = 1.0
    return A


def build_anc_v1b(F, n_real):
    """v1b: T→Y = +1, Y→T = -1. Rest of real block 0."""
    A = _padded_neg1_only(F, n_real)
    A[0, 1] = 1.0
    A[1, 0] = -1.0
    return A


def build_anc_v2a(F, n_real):
    """v2a: T→Y = +1, all X→T = +1. Rest of real block 0."""
    A = _padded_neg1_only(F, n_real)
    A[0, 1] = 1.0
    for i in range(n_real):
        A[2 + i, 0] = 1.0
    return A


def build_anc_v2b(F, n_real):
    """v2b: T→Y = +1, all X→T = +1, Y→T = -1, all T→X = -1. Rest 0."""
    A = _padded_neg1_only(F, n_real)
    A[0, 1] = 1.0
    A[1, 0] = -1.0
    for i in range(n_real):
        A[2 + i, 0] = 1.0
        A[0, 2 + i] = -1.0
    return A


def build_anc_v3a(F, n_real):
    """v3a: T→Y = +1, all X→T = +1, all X→Y = +1. Rest of real block 0.
    (= build_anc_full without propagation.)"""
    A = _padded_neg1_only(F, n_real)
    A[0, 1] = 1.0
    for i in range(n_real):
        A[2 + i, 0] = 1.0
        A[2 + i, 1] = 1.0
    return A


def build_anc_v3b(F, n_real):
    """v3b: v3a + all reverses = -1 (Y→T=-1, T→X=-1, Y→X=-1). Rest 0.
    Same +1 layout as build_anc_full; -1s asserted explicitly (no propagate)."""
    A = _padded_neg1_only(F, n_real)
    A[0, 1] = 1.0
    A[1, 0] = -1.0
    for i in range(n_real):
        A[2 + i, 0] = 1.0    # X→T
        A[2 + i, 1] = 1.0    # X→Y
        A[0, 2 + i] = -1.0   # T→X (reverse)
        A[1, 2 + i] = -1.0   # Y→X (reverse)
    return A


def build_anc_v3c(F, n_real):
    """v3c: v3b + diagonal -1 on the real block (T,Y,X_real all self-loop -1).
    Equivalent to what build_anc_full + propagate produces on unconfoundedness
    edges — but constructed explicitly here without calling propagate."""
    A = build_anc_v3b(F, n_real)
    real_n = 2 + n_real
    for i in range(real_n):
        A[i, i] = -1.0
    return A


def build_anc_diag(F, n_real):
    """diag-only: diagonal of the real block is -1. Rest of real block 0.
    Padded rows/cols still -1."""
    A = _padded_neg1_only(F, n_real)
    real_n = 2 + n_real
    for i in range(real_n):
        A[i, i] = -1.0
    return A


def build_anc_v4a(F, n_real):
    """v4a: v3a MINUS T→Y edge. So X→T = +1 and X→Y = +1, but A[T,Y] = 0.
    Rest of real block 0. Probes whether our model is degraded specifically
    by the T→Y=+1 assertion, or by the X→T / X→Y edges."""
    A = _padded_neg1_only(F, n_real)
    for i in range(n_real):
        A[2 + i, 0] = 1.0
        A[2 + i, 1] = 1.0
    return A


def build_anc_v5a(F, n_real):
    """v5a: X→Y = +1 only. T untouched: T→Y = 0. Rest of real block 0.
    Padded -1. Probes whether asserting only the X→Y (outcome regression)
    edges helps, without asserting any T-related edges."""
    A = _padded_neg1_only(F, n_real)
    for i in range(n_real):
        A[2 + i, 1] = 1.0
    return A


def build_anc_v5b(F, n_real):
    """v5b: v3a but with X↔T swapped: X→T = 0 (normal direct edge removed),
    T→X = -1 (assert reverse edge). Keeps T→Y=+1 and X→Y=+1. Rest of real
    block 0. Padded -1. Probes whether asserting non-directionality of X↔T
    via reverse instead of forward edge changes behavior."""
    A = _padded_neg1_only(F, n_real)
    A[0, 1] = 1.0
    for i in range(n_real):
        # NO X→T = +1 (v3a would have this)
        A[2 + i, 1] = 1.0    # X→Y = +1 (kept from v3a)
        A[0, 2 + i] = -1.0   # T→X = -1 (asymmetric assertion)
    return A


def build_anc_v6a(F, n_real):
    """v6a: no +1 edges anywhere. All -1s from unconfoundedness:
      - Diagonal (self-loops) = -1
      - Y→T = -1 (T causes Y, not the other way)
      - Y→X_i = -1 for all real X (Y is downstream)
      - T→X_i = -1 for all real X (T is downstream)
    Everything else in real block = 0. Padded region -1."""
    A = _padded_neg1_only(F, n_real)
    real_n = 2 + n_real
    for i in range(real_n):
        A[i, i] = -1.0
    A[1, 0] = -1.0   # Y→T = -1
    for i in range(n_real):
        A[1, 2 + i] = -1.0   # Y→X_i
        A[0, 2 + i] = -1.0   # T→X_i
    return A


def build_anc_combo(F, n_real, code, diag_neg=False):
    """Parameterized anc builder. code = 3-char string of {P, N, B, O} for
    edges [T→Y, X→T, X→Y]:
      - P: assert +1 in ancestor direction only
      - N: assert -1 in reverse direction only
      - B: both (+1 in ancestor direction, -1 in reverse)
      - O: omit (leave as 0)
    diag_neg: if True, set real-block diagonal to -1. Else 0.
    Padded region: always -1."""
    assert len(code) == 3 and all(c in 'PNBO' for c in code), f'bad code: {code}'
    A = _padded_neg1_only(F, n_real)
    T_idx, Y_idx = 0, 1
    # Edge 1: T→Y
    c = code[0]
    if c in ('P', 'B'):
        A[T_idx, Y_idx] = 1.0
    if c in ('N', 'B'):
        A[Y_idx, T_idx] = -1.0
    # Edge 2: X_i → T for all real i
    c = code[1]
    for i in range(n_real):
        if c in ('P', 'B'):
            A[2 + i, T_idx] = 1.0
        if c in ('N', 'B'):
            A[T_idx, 2 + i] = -1.0
    # Edge 3: X_i → Y for all real i
    c = code[2]
    for i in range(n_real):
        if c in ('P', 'B'):
            A[2 + i, Y_idx] = 1.0
        if c in ('N', 'B'):
            A[Y_idx, 2 + i] = -1.0
    if diag_neg:
        for i in range(2 + n_real):
            A[i, i] = -1.0
    return A


def _three_edge_54_tags_and_codes(F, n_real):
    """Return list of (tag, adj) for all 27 three-edge codes × 2 diag choices.
    Tag format: {code}{diag_suffix} where suffix = '' for diag=0 or 'n' for diag=-1.
    Example: 'PPP' (diag=0), 'PPPn' (diag=-1). 54 total."""
    codes3 = [a + b + c for a in 'PNB' for b in 'PNB' for c in 'PNB']  # 27
    out = []
    for code in codes3:
        out.append((code,      build_anc_combo(F, n_real, code, diag_neg=False)))
        out.append((code + 'n', build_anc_combo(F, n_real, code, diag_neg=True)))
    return out


def build_anc_v6b(F, n_real):
    """v6b: same as v6a (all -1s from unconfoundedness) but with diagonal = 0.
    Only Y→T=-1, Y→X_i=-1, T→X_i=-1. No self-loop assertion. No +1 edges.
    Rest of real block 0. Padded -1."""
    A = _padded_neg1_only(F, n_real)
    A[1, 0] = -1.0   # Y→T = -1
    for i in range(n_real):
        A[1, 2 + i] = -1.0   # Y→X_i
        A[0, 2 + i] = -1.0   # T→X_i
    return A


def build_anc_v7a(F, n_real):
    """v7a: T→Y=+1 AND diagonal=-1. Rest of real block 0. Padded -1."""
    A = _padded_neg1_only(F, n_real)
    A[0, 1] = 1.0
    real_n = 2 + n_real
    for i in range(real_n):
        A[i, i] = -1.0
    return A


def build_anc_v7b(F, n_real):
    """v7b: T→Y=+1, Y→T=-1, diagonal=-1. Rest of real block 0. Padded -1."""
    A = _padded_neg1_only(F, n_real)
    A[0, 1] = 1.0
    A[1, 0] = -1.0
    real_n = 2 + n_real
    for i in range(real_n):
        A[i, i] = -1.0
    return A


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
        np.random.seed(PSID_BAL_SEED)
        idx = np.random.choice(n_control, n_keep, replace=False)
        X_ct = X_ct[idx]; t_ct = t_ct[idx]; y_ct = y_ct[idx]
        print(f'[PSID-bal] kept {X_tr.shape[0]} treated, sampled {n_keep}/{n_control} controls '
              f'(seed={PSID_BAL_SEED})', flush=True)
    else:
        print(f'[PSID-bal] kept {X_tr.shape[0]} treated, all {n_control} controls',
              flush=True)

    X = np.vstack([X_tr, X_ct]); t = np.concatenate([t_tr, t_ct]); y = np.concatenate([y_tr, y_ct])
    perm = np.random.RandomState(PSID_BAL_SEED).permutation(X.shape[0])
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
    return model, cfg


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


@torch.no_grad()
def marginals_from_forward(model, X_train, T_train, Y_train_scaled, X_test, adj, J):
    """Run one forward pass; return per-query (p_y0, p_y1, logits_np).

    - p_y0, p_y1: (N_q, J)  inner-marginals (softmax over J² then marginalise)
    - logits_np:  (N_q, J²+9+4)  FULL head output, needed by full-mixture mean
    """
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
        p_y0_all, p_y1_all, logits_all = [], [], []
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
            interior = logits[..., : J * J]
            p = torch.softmax(interior, dim=-1).reshape(B, -1, J, J)
            p_y0_batch = p.sum(dim=-1).squeeze(0).cpu().numpy()   # (max_n_test, J)
            p_y1_batch = p.sum(dim=-2).squeeze(0).cpu().numpy()
            logits_batch = logits.squeeze(0).float().cpu().numpy()  # (max_n_test, J²+9+4)
            # Drop padded rows before appending
            p_y0_all.append(p_y0_batch[:n_batch_real])
            p_y1_all.append(p_y1_batch[:n_batch_real])
            logits_all.append(logits_batch[:n_batch_real])
        return (np.concatenate(p_y0_all, axis=0),
                np.concatenate(p_y1_all, axis=0),
                np.concatenate(logits_all, axis=0))

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
    interior = logits[..., : J * J]
    p = torch.softmax(interior, dim=-1).reshape(B, -1, J, J)
    p_y0 = p.sum(dim=-1).squeeze(0).cpu().numpy()
    p_y1 = p.sum(dim=-2).squeeze(0).cpu().numpy()
    logits_np = logits.squeeze(0).float().cpu().numpy()   # (N_q, J²+9+4)
    return p_y0, p_y1, logits_np


def cate_from_marginals(p_y0, p_y1, J, logits_np=None):
    """Return (cate_raw, cate_em, cate_full) on the [-1, 1] scale.

    - raw:  inner-only marginal center-of-mass  (drops tail region mass)
    - em:   Gaussian-corrected inner marginal fixed-point mean
    - full: mean over the full 9-region mixture density   (requires logits_np;
            returns nan array if logits_np is None)
    """
    edges   = np.linspace(-1.0, 1.0, J + 1, dtype=np.float64)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Raw mean: center-of-mass.
    e_y0_raw = (p_y0 * centres[None, :]).sum(axis=-1)
    e_y1_raw = (p_y1 * centres[None, :]).sum(axis=-1)
    cate_raw = e_y1_raw - e_y0_raw

    # EM mean: per-query per-arm fixed-point Gaussian correction.
    # Skipped if SKIP_EM=1 (default fast path for SCM case studies).
    if os.environ.get('SKIP_EM', '0') == '1':
        e_y0_em = e_y0_raw.copy(); e_y1_em = e_y1_raw.copy()
        cate_em = cate_raw.copy()
    else:
        N_q = p_y0.shape[0]
        e_y0_em = np.empty(N_q); e_y1_em = np.empty(N_q)
        for q in range(N_q):
            mu0, s0 = _marginal_stats(p_y0[q], edges)
            mu1, s1 = _marginal_stats(p_y1[q], edges)
            e_y0_em[q] = _em_mean_1d(p_y0[q], edges, s0, mu0)
            e_y1_em[q] = _em_mean_1d(p_y1[q], edges, s1, mu1)
        cate_em = e_y1_em - e_y0_em

    # Full 9-region mixture mean (integrates over ℝ², not just inner).
    if logits_np is not None:
        # Local sys.path hack — full_mixture_mean lives in benchmarks/eval_causalpfn2d/
        import sys, os as _os
        _fm_dir = _os.path.abspath(_os.path.join(
            _os.path.dirname(__file__), '..', 'eval_causalpfn2d'))
        if _fm_dir not in sys.path:
            sys.path.insert(0, _fm_dir)
        from full_mixture_mean import full_mixture_mean
        e_y0_full, e_y1_full = full_mixture_mean(logits_np, J, edges)
        cate_full = e_y1_full - e_y0_full
    else:
        cate_full = np.full(N_q, np.nan)

    return (cate_raw.astype(np.float32),
            cate_em.astype(np.float32),
            cate_full.astype(np.float32))


def build_mode_list(F, n_real, anc_mode=None):
    """Map ANC_MODE → ((tag, adjacency), ...). Tags become the npz key suffixes
    (`pehe_raw_<tag>` etc.). Shared with eval_uwyk1d_realcause.py so the control
    run emits the exact same adjacency matrices and the same npz schema."""
    anc_mode = ANC_MODE if anc_mode is None else anc_mode
    if anc_mode == 'ty_only':
        return (('ty',    build_anc_ty_only(F, n_real)),
                ('noanc', build_anc_none(F, n_real)))
    if anc_mode == 'ty_antisym':
        return (('tyx',   build_anc_ty_antisym(F, n_real)),
                ('noanc', build_anc_none(F, n_real)))
    if anc_mode == 'v4a_only':
        return (('v4a',   build_anc_v4a(F, n_real)),
                ('noanc', build_anc_none(F, n_real)))
    if anc_mode == 'v5a_only':
        return (('v5a',   build_anc_v5a(F, n_real)),
                ('noanc', build_anc_none(F, n_real)))
    if anc_mode == 'v5b_only':
        return (('v5b',   build_anc_v5b(F, n_real)),
                ('noanc', build_anc_none(F, n_real)))
    if anc_mode == 'v6b_only':
        return (('v6b',   build_anc_v6b(F, n_real)),
                ('noanc', build_anc_none(F, n_real)))
    if anc_mode == 'v6a_only':
        return (('v6a',   build_anc_v6a(F, n_real)),
                ('noanc', build_anc_none(F, n_real)))
    if anc_mode == 'all_combos':
        # 4^3 = 64 combinations of (T→Y, X→T, X→Y) encoded as P|N|B|O.
        # Diagonal always 0; padded region always -1.
        _codes = [a + b + c for a in 'PNBO' for b in 'PNBO' for c in 'PNBO']
        return tuple((code, build_anc_combo(F, n_real, code)) for code in _codes)
    if anc_mode == 'three_edge_all':
        # All 27 three-edge combos (P|N|B for each of T→Y, X→T, X→Y) × 2 diag
        # choices = 54 variants. All 3 edges asserted (no O). Tags use 'n'
        # suffix for diag=-1 variant (e.g., PPP vs PPPn).
        return tuple(_three_edge_54_tags_and_codes(F, n_real))
    if anc_mode == 'focus3':
        return (
            ('v7a',   build_anc_v7a(F, n_real)),   # T→Y + diag
            ('v7b',   build_anc_v7b(F, n_real)),   # T→Y + Y→T=-1 + diag
            ('v6a',   build_anc_v6a(F, n_real)),   # all -1s, no +1s
            ('noanc', build_anc_none(F, n_real)),
        )
    if anc_mode == 'focus4':
        return (
            ('v7a',   build_anc_v7a(F, n_real)),   # T→Y + diag
            ('v7b',   build_anc_v7b(F, n_real)),   # T→Y + Y→T=-1 + diag
            ('v6a',   build_anc_v6a(F, n_real)),   # all -1s, no +1s, diag -1
            ('v6b',   build_anc_v6b(F, n_real)),   # all -1s, no +1s, diag 0
            ('noanc', build_anc_none(F, n_real)),
        )
    if anc_mode == 'all_variants':
        return (
            ('v1a',   build_anc_v1a(F, n_real)),
            ('v1b',   build_anc_v1b(F, n_real)),
            ('v2a',   build_anc_v2a(F, n_real)),
            ('v2b',   build_anc_v2b(F, n_real)),
            ('v3a',   build_anc_v3a(F, n_real)),
            ('v3b',   build_anc_v3b(F, n_real)),
            ('v3c',   build_anc_v3c(F, n_real)),
            ('v4a',   build_anc_v4a(F, n_real)),
            ('v5a',   build_anc_v5a(F, n_real)),
            ('v5b',   build_anc_v5b(F, n_real)),
            ('noanc', build_anc_none(F, n_real)),
            ('diag',  build_anc_diag(F, n_real)),
        )
    return (('anc',   build_anc_full(F, n_real)),
            ('noanc', build_anc_none(F, n_real)))


def evaluate(realization, ds, model, J, F, apply_psid_balance):
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
    _mode_list = build_mode_list(F, n_real)
    for mode, adj in _mode_list:
        p_y0, p_y1, logits_np = marginals_from_forward(model, X_tr, T_tr, Y_obs, X_te, adj, J)
        cate_raw_scaled, cate_em_scaled, cate_full_scaled = cate_from_marginals(
            p_y0, p_y1, J, logits_np=logits_np,
        )
        # Un-scale to raw Y units. (2 * cate_scaled / 2) * yrange / 2 = cate_scaled * yrange / 2.
        for method, cate_scaled in (('raw',  cate_raw_scaled),
                                     ('em',   cate_em_scaled),
                                     ('full', cate_full_scaled)):
            cate = cate_scaled * yrange / 2.0
            pehe = float(np.sqrt(np.mean((cate - true_cate) ** 2)))
            ate_hat = float(cate.mean())
            err_ate = abs(ate_hat - true_ate) / max(abs(true_ate), 0.1)
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

    model, cfg = load_model(CKPT)
    J = cfg['J']; F = cfg['num_features']
    print(f'[bootstrap] J={J}  F={F}', flush=True)

    rows = []
    t0 = time.time()
    # MAX_REAL: cap number of realizations (e.g. MAX_REAL=1 for a fast
    # diagnostic run of a single realization per dataset).
    _cap = int(os.environ.get('MAX_REAL', ds.n_tables))
    for r in range(min(ds.n_tables, _cap)):
        row = evaluate(r, ds, model, J, F, apply_psid_balance)
        rows.append(row)
        np.savez(os.path.join(OUT, f'{DATASET}_r{r:03d}.npz'),
                 **{k: np.array(v) for k, v in row.items()})
        # Mode-agnostic printing: 'anc' | 'ty' | 'tyx' | 'v*' depending on ANC_MODE.
        if ANC_MODE == 'all_variants':
            # Compact one-line summary for all variants
            parts = []
            for tag in ('v1a','v1b','v2a','v2b','v3a','v3b','v3c','v4a','v5a','v5b','noanc','diag'):
                p = row.get(f'pehe_raw_{tag}', float('nan'))
                parts.append(f'{tag}={p:6.3f}')
            print(f'r={r:03d}  ' + '  '.join(parts) + f'  ({time.time()-t0:.0f}s)', flush=True)
        elif ANC_MODE == 'focus3':
            parts = []
            for tag in ('v7a','v7b','v6a','noanc'):
                p = row.get(f'pehe_raw_{tag}', float('nan'))
                parts.append(f'{tag}={p:6.3f}')
            print(f'r={r:03d}  ' + '  '.join(parts) + f'  ({time.time()-t0:.0f}s)', flush=True)
        elif ANC_MODE == 'focus4':
            parts = []
            for tag in ('v7a','v7b','v6a','v6b','noanc'):
                p = row.get(f'pehe_raw_{tag}', float('nan'))
                parts.append(f'{tag}={p:6.3f}')
            print(f'r={r:03d}  ' + '  '.join(parts) + f'  ({time.time()-t0:.0f}s)', flush=True)
        elif ANC_MODE == 'all_combos':
            _tags_all = [a + b + c for a in 'PNBO' for b in 'PNBO' for c in 'PNBO']
            # 64 tags is too long for one line — print just the min-PEHE tag
            best = min(_tags_all, key=lambda t: row.get(f'pehe_raw_{t}', float('inf')))
            print(f'r={r:03d}  best={best} pehe={row[f"pehe_raw_{best}"]:6.3f}  '
                  f'OOO(noanc) pehe={row["pehe_raw_OOO"]:6.3f}  ({time.time()-t0:.0f}s)', flush=True)
        elif ANC_MODE == 'three_edge_all':
            _tags_all = [c + s for c in [a+b+d for a in 'PNB' for b in 'PNB' for d in 'PNB']
                                for s in ('', 'n')]
            best = min(_tags_all, key=lambda t: row.get(f'pehe_raw_{t}', float('inf')))
            print(f'r={r:03d}  best={best} pehe={row[f"pehe_raw_{best}"]:6.3f}  '
                  f'({time.time()-t0:.0f}s)', flush=True)
        else:
            _pos_tag = ('ty' if ANC_MODE == 'ty_only' else
                        'tyx' if ANC_MODE == 'ty_antisym' else
                        'v4a' if ANC_MODE == 'v4a_only' else
                        'v5a' if ANC_MODE == 'v5a_only' else
                        'v5b' if ANC_MODE == 'v5b_only' else
                        'v6a' if ANC_MODE == 'v6a_only' else
                        'v6b' if ANC_MODE == 'v6b_only' else 'anc')
            print(
                f'r={r:03d}  '
                f'raw-{_pos_tag}: pehe={row[f"pehe_raw_{_pos_tag}"]:6.3f} err={row[f"err_raw_{_pos_tag}"]:5.3f}  |  '
                f'em-{_pos_tag}: pehe={row[f"pehe_em_{_pos_tag}"]:6.3f} err={row[f"err_em_{_pos_tag}"]:5.3f}  |  '
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
    if ANC_MODE == 'all_variants':
        keys = []
        for tag in ('v1a','v1b','v2a','v2b','v3a','v3b','v3c','v4a','v5a','v5b','noanc','diag'):
            keys += [f'pehe_raw_{tag}', f'err_raw_{tag}', f'pehe_em_{tag}', f'err_em_{tag}']
    elif ANC_MODE == 'focus3':
        keys = []
        for tag in ('v7a','v7b','v6a','noanc'):
            keys += [f'pehe_raw_{tag}', f'err_raw_{tag}', f'pehe_em_{tag}', f'err_em_{tag}']
    elif ANC_MODE == 'focus4':
        keys = []
        for tag in ('v7a','v7b','v6a','v6b','noanc'):
            keys += [f'pehe_raw_{tag}', f'err_raw_{tag}', f'pehe_em_{tag}', f'err_em_{tag}']
    elif ANC_MODE == 'all_combos':
        keys = []
        for tag in [a + b + c for a in 'PNBO' for b in 'PNBO' for c in 'PNBO']:
            keys += [f'pehe_raw_{tag}', f'pehe_em_{tag}']
    elif ANC_MODE == 'three_edge_all':
        keys = []
        for tag in [c + s for c in [a+b+d for a in 'PNB' for b in 'PNB' for d in 'PNB']
                          for s in ('', 'n')]:
            keys += [f'pehe_raw_{tag}', f'pehe_em_{tag}']
    else:
        _pos_tag = ('ty' if ANC_MODE == 'ty_only' else
                    'tyx' if ANC_MODE == 'ty_antisym' else
                    'v4a' if ANC_MODE == 'v4a_only' else
                    'v5a' if ANC_MODE == 'v5a_only' else
                    'v5b' if ANC_MODE == 'v5b_only' else 'anc')
        keys = [f'pehe_raw_{_pos_tag}', f'err_raw_{_pos_tag}',
                f'pehe_em_{_pos_tag}',  f'err_em_{_pos_tag}',
                'pehe_raw_noanc', 'err_raw_noanc',
                'pehe_em_noanc',  'err_em_noanc']
    for k in keys:
        m, s, n = ms(k)
        print(f'  {k:20s} = {m:8.3f} ± {s:6.3f}   (n={n})')


if __name__ == '__main__':
    main()
