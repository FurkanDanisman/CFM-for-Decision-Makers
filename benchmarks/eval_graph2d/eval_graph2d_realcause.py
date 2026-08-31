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
    """Three-state ancestor matrix matching the training convention (verified
    against Graphs4CausalFoundationModels reproduce-realcause-results branch,
    src/priordata_processing/Datasets/InterventionalDataset.py L998-1002).

    Training-time anc matrices have entries in {-1, 0, +1}:
      +1: known ancestor
      -1: known NOT ancestor
       0: unknown (only set when hide_fraction_matrix > 0)

    Under 'full-graph' knowledge (Lalonde/ACIC/IHDP downstream), the model
    should see -1 wherever we KNOW there's no ancestral relation. The prior
    version left those entries at 0 which reads as "I don't know" — that's
    what's been hurting anc-mode results on Lalonde/ACIC.

    Assumed structure: T→Y, X_i→T, X_i→Y for i in real features. Then:
      * Y↛T (Y not ancestor of T)
      * T↛X_i, Y↛X_i (T, Y not ancestors of features)
      * self-loops → -1 (irreflexive ancestor)
      * feature↔feature: TRULY UNKNOWN → 0 (we have no prior on covariate DAG)
      * padded slots → -1 (matches training)
    """
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    T_idx, Y_idx, feat_off = 0, 1, 2

    # ── Known ancestor edges (+1) ────────────────────────────────
    A[T_idx, Y_idx] = 1.0
    for i in range(n_real):
        A[feat_off + i, T_idx] = 1.0
        A[feat_off + i, Y_idx] = 1.0

    # ── Known NON-ancestor edges (-1) ────────────────────────────
    A[Y_idx, T_idx] = -1.0                       # Y is not ancestor of T
    for i in range(n_real):
        A[T_idx, feat_off + i] = -1.0            # T is not ancestor of X_i
        A[Y_idx, feat_off + i] = -1.0            # Y is not ancestor of X_i
    # Irreflexive: self is not own ancestor.
    A[T_idx, T_idx] = -1.0
    A[Y_idx, Y_idx] = -1.0
    for i in range(n_real):
        A[feat_off + i, feat_off + i] = -1.0

    # feature-to-feature (X_i, X_j for i != j) intentionally left at 0
    # (unknown) — we have no domain prior on the covariate DAG.

    # ── Padded feature slots (masked out) ───────────────────────
    for i in range(n_real, F):
        A[feat_off + i, :] = -1.0
        A[:, feat_off + i] = -1.0
        A[feat_off + i, feat_off + i] = -1.0

    return A


def build_anc_none(F, n_real):
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    feat_off = 2
    for i in range(n_real, F):
        A[feat_off + i, :] = -1.0
        A[:, feat_off + i] = -1.0
        A[feat_off + i, feat_off + i] = -1.0
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
        for suffix in ('_x', '_y'):
            k = prefix + suffix
            if k in sd and sd[k].dim() >= 1:
                return int(sd[k].shape[0])
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
    """Run one forward pass; return per-query (p_y0, p_y1) numpy arrays of
    shape (N_q, J)."""
    B = 1
    X_obs = torch.from_numpy(X_train.astype(np.float32)).unsqueeze(0).to(DEVICE)
    T_obs = torch.from_numpy(T_train.astype(np.float32)).reshape(1, -1, 1).to(DEVICE)
    Y_obs = torch.from_numpy(Y_train_scaled).unsqueeze(0).to(DEVICE)
    X_intv = torch.from_numpy(X_test.astype(np.float32)).unsqueeze(0).to(DEVICE)
    adj_t = torch.from_numpy(adj).unsqueeze(0).to(DEVICE)

    out = model(X_obs, T_obs, Y_obs, X_intv, adj_t)
    logits = out['predictions'] if isinstance(out, dict) else out
    interior = logits[..., : J * J]
    p = torch.softmax(interior, dim=-1).reshape(B, -1, J, J)
    p_y0 = p.sum(dim=-1).squeeze(0).cpu().numpy()
    p_y1 = p.sum(dim=-2).squeeze(0).cpu().numpy()
    return p_y0, p_y1


def cate_from_marginals(p_y0, p_y1, J):
    """Return (cate_raw, cate_em) on the [-1, 1] scale."""
    edges   = np.linspace(-1.0, 1.0, J + 1, dtype=np.float64)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Raw mean: center-of-mass.
    e_y0_raw = (p_y0 * centres[None, :]).sum(axis=-1)
    e_y1_raw = (p_y1 * centres[None, :]).sum(axis=-1)
    cate_raw = e_y1_raw - e_y0_raw

    # EM mean: per-query per-arm fixed-point Gaussian correction.
    N_q = p_y0.shape[0]
    e_y0_em = np.empty(N_q); e_y1_em = np.empty(N_q)
    for q in range(N_q):
        mu0, s0 = _marginal_stats(p_y0[q], edges)
        mu1, s1 = _marginal_stats(p_y1[q], edges)
        e_y0_em[q] = _em_mean_1d(p_y0[q], edges, s0, mu0)
        e_y1_em[q] = _em_mean_1d(p_y1[q], edges, s1, mu1)
    cate_em = e_y1_em - e_y0_em

    return cate_raw.astype(np.float32), cate_em.astype(np.float32)


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
    for mode, adj in (('anc',   build_anc_full(F, n_real)),
                       ('noanc', build_anc_none(F, n_real))):
        p_y0, p_y1 = marginals_from_forward(model, X_tr, T_tr, Y_obs, X_te, adj, J)
        cate_raw_scaled, cate_em_scaled = cate_from_marginals(p_y0, p_y1, J)
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

    model, cfg = load_model(CKPT)
    J = cfg['J']; F = cfg['num_features']
    print(f'[bootstrap] J={J}  F={F}', flush=True)

    rows = []
    t0 = time.time()
    for r in range(ds.n_tables):
        row = evaluate(r, ds, model, J, F, apply_psid_balance)
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
    for k in ('pehe_raw_anc', 'err_raw_anc',
              'pehe_em_anc',  'err_em_anc',
              'pehe_raw_noanc', 'err_raw_noanc',
              'pehe_em_noanc',  'err_em_noanc'):
        m, s, n = ms(k)
        print(f'  {k:20s} = {m:8.3f} ± {s:6.3f}   (n={n})')


if __name__ == '__main__':
    main()
