"""Evaluate our fn=50 checkpoint on one RealCause dataset under the same
inference protocol UWYK's reproduce-realcause-results branch uses for its
No/Ancestral rows:

  - single forward pass over the FULL train context (no clustering),
  - X standardised per-task (mean/std of X_train),
  - Y scaled to [-1, 1] per-task,
  - CATE extracted as the difference of the 2D joint's marginal means,
  - inverse-scaled back to raw Y units.

Two things we do NOT copy from UWYK's DOFM path, because our checkpoint
wasn't trained under those conditions:
  - target-encoded T (our fn=50 sees T in {0, 1} at every training step),
  - explicit T_intv at query (our joint head fills it with null_t_intv).

The output per realization matches UWYK's format (pickle with
{model, dataset, realization, cate_preds, pehe, ate_rel_err}) so the same
aggregate_results.py works.

Usage:
    OURS_CKPT=$REPO/checkpoints/step_50000_final.pt \\
    CAUSALPFN_ROOT=$DEPLOY_ROOT/external/causalpfn \\
    python -u benchmarks/uwyk_table1/ours_fn50_no_clustering.py \\
        --dataset IHDP --model ours_fn50 --exp_name table1_ours_fn50
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch


# ── path plumbing ───────────────────────────────────────────────────────
_HERE     = os.path.abspath(os.path.dirname(__file__))
_R_PFN    = os.path.abspath(os.path.join(_HERE, '..', '..'))
CAUSALPFN = os.environ.get(
    'CAUSALPFN_ROOT', '/scratch/furkanbd/rpfn_bench_kit/external/causalpfn')
sys.path.insert(0, _R_PFN)
sys.path.insert(0, CAUSALPFN)
# benchmarks/__init__.py at CAUSALPFN root imports `causalpfn.synthetic`,
# which lives at CAUSALPFN/src/causalpfn/, so add both dirs.
_CAUSALPFN_SRC = os.path.join(CAUSALPFN, 'src')
if os.path.isdir(_CAUSALPFN_SRC):
    sys.path.insert(0, _CAUSALPFN_SRC)

# Shim faiss / huggingface_hub etc. so `causalpfn` can be imported.
_SHIMS = os.path.join(_HERE, 'shims')
if os.path.isdir(_SHIMS) and _SHIMS not in sys.path:
    sys.path.insert(0, _SHIMS)

from benchmarks import (          # noqa: E402  from CausalPFN's benchmarks
    IHDPDataset, ACIC2016Dataset,
    RealCauseLalondeCPSDataset, RealCauseLalondePSIDDataset,
)
from models.InterventionalPFN import InterventionalPFN   # noqa: E402
from losses.BarDistribution2D import total_params         # noqa: E402


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASETS = {
    'IHDP':  IHDPDataset,
    'ACIC':  ACIC2016Dataset,
    'CPS':   RealCauseLalondeCPSDataset,
    'PSID':  RealCauseLalondePSIDDataset,
}


def _pad_features(X: np.ndarray, F: int) -> np.ndarray:
    if X.shape[1] == F:
        return X
    if X.shape[1] > F:
        return X[:, :F]
    return np.hstack([X, np.zeros((X.shape[0], F - X.shape[1]), dtype=X.dtype)])


def _standardize_train_test(X_train: np.ndarray, X_test: np.ndarray):
    """Same recipe as `PairedInterventionalDataset._standardize` we trained
    against (mean/std on train, apply to both). No outlier clipping — our
    training pipeline didn't do it."""
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0,  keepdims=True)
    sd = np.where(sd < 1e-8, 1e-8, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd


def _scale_y(y: np.ndarray):
    ymin  = float(y.min())
    yrange = float(max(y.max() - ymin, 1e-9))
    return (2.0 * (y - ymin) / yrange - 1.0).astype(np.float32), ymin, yrange


def load_model(ckpt_path: str) -> tuple[InterventionalPFN, dict]:
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ck['config']
    model = InterventionalPFN(
        num_features=cfg['num_features'],
        d_model=cfg['d_model'],
        depth=cfg['depth'],
        heads_feat=cfg['heads'],
        heads_samp=cfg['heads'],
        dropout=0.0,
        output_dim=total_params(cfg['J']),
        hidden_mult=cfg['hidden_mult'],
        normalize_features=True,
        normalize_treatment=False,
        use_treatment_in_query=False,
        use_checkpoint=False,
    ).to(DEVICE)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    return model, cfg


@torch.no_grad()
def _cate_from_joint(model, X_train, T_train, Y_train_scaled, X_test, J):
    """Single forward → 2D joint → per-query marginals → E[y1] - E[y0]."""
    B = 1
    X_obs  = torch.from_numpy(X_train.astype(np.float32)).unsqueeze(0).to(DEVICE)
    T_obs  = torch.from_numpy(T_train.astype(np.float32)).reshape(1, -1, 1).to(DEVICE)
    Y_obs  = torch.from_numpy(Y_train_scaled).unsqueeze(0).to(DEVICE)
    X_intv = torch.from_numpy(X_test.astype(np.float32)).unsqueeze(0).to(DEVICE)

    out = model(X_obs, T_obs, Y_obs, X_intv)          # T_intv=None → null_t_intv
    logits = out['predictions'] if isinstance(out, dict) else out
    p = torch.softmax(logits[..., : J * J], dim=-1).reshape(B, -1, J, J)

    centres = torch.linspace(-1.0 + 1.0 / J, 1.0 - 1.0 / J, J, device=logits.device)
    p_y0 = p.sum(dim=-1)
    p_y1 = p.sum(dim=-2)
    e_y0 = (p_y0 * centres.view(1, 1, J)).sum(dim=-1)
    e_y1 = (p_y1 * centres.view(1, 1, J)).sum(dim=-1)
    return (e_y1 - e_y0).squeeze(0).cpu().numpy()


@torch.no_grad()
def _cluster_and_average_cate(
    model, X_train, T_train, Y_train_scaled, X_test, J,
    max_n_train: int = 1000, random_state: int = 0,
):
    """KMeans-partition-and-average inference. Kept for reference / A-B
    comparison. Empirically WORSE than _subsample_and_average_cate on CPS
    (12727 vs single-pass 12688 — no gain) because KMeans forces each
    cluster to have a narrow, homogeneous X range: per-cluster CATE(x_te)
    is unreliable for test queries outside that cluster's X domain, and
    averaging 15 mostly-unreliable local-expert predictions doesn't
    approximate the true full-data CATE. Use `_subsample_and_average_cate`
    instead for large datasets — random subsets preserve the marginal X
    distribution so every subset's CATE is unbiased.
    """
    from sklearn.cluster import KMeans
    n = X_train.shape[0]
    k = max(1, (n + max_n_train - 1) // max_n_train)
    if k == 1:
        return _cate_from_joint(model, X_train, T_train, Y_train_scaled, X_test, J)
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X_train)
    per_cluster_cate = []
    for c in range(k):
        m = (labels == c)
        if m.sum() < 8:                            # skip degenerate clusters
            continue
        cate_c = _cate_from_joint(
            model,
            X_train[m], T_train[m], Y_train_scaled[m],
            X_test, J,
        )
        per_cluster_cate.append(cate_c)
    if not per_cluster_cate:
        return _cate_from_joint(model, X_train, T_train, Y_train_scaled, X_test, J)
    return np.mean(np.stack(per_cluster_cate, axis=0), axis=0)


@torch.no_grad()
def _subsample_and_average_cate(
    model, X_train, T_train, Y_train_scaled, X_test, J,
    max_n_train: int = 1000, n_repeats: int = 20,
    stratify_by_t: bool = True, random_state: int = 0,
):
    """Bagged in-context inference: draw n_repeats random subsets of size
    ≤ max_n_train from the training set, run one forward per subset, and
    average per-query CATE.

    Unlike _cluster_and_average_cate this KEEPS the marginal X distribution
    intact in every subset (up to sampling noise), so each per-subset CATE
    is a valid unbiased estimator of the full-data CATE. The average
    reduces variance without introducing the "local expert" bias that KMeans
    partitioning creates.

    When `stratify_by_t=True`, each subset preserves the T=0/T=1 ratio of
    the full training set — matters when T is imbalanced (e.g. PSID has
    ~85 controls per treated), where a raw random subset can end up with
    almost no treated units and give a degenerate CATE.
    """
    rng = np.random.default_rng(random_state)
    n = X_train.shape[0]
    if n <= max_n_train:
        return _cate_from_joint(model, X_train, T_train, Y_train_scaled, X_test, J)

    T_flat = T_train.reshape(-1)
    if stratify_by_t:
        idx_t = np.where(T_flat == 1)[0]
        idx_c = np.where(T_flat == 0)[0]
        # Preserve empirical p(T=1); cap total at max_n_train.
        frac_t = idx_t.size / n
        n_t = max(1, int(round(max_n_train * frac_t)))
        n_c = max(1, max_n_train - n_t)
        # Guard against under-population on tiny arms.
        n_t = min(n_t, idx_t.size)
        n_c = min(n_c, idx_c.size)

    per_run_cate = []
    for r in range(n_repeats):
        if stratify_by_t:
            pick = np.concatenate([
                rng.choice(idx_t, size=n_t, replace=False),
                rng.choice(idx_c, size=n_c, replace=False),
            ])
        else:
            pick = rng.choice(n, size=max_n_train, replace=False)
        rng.shuffle(pick)
        cate_r = _cate_from_joint(
            model,
            X_train[pick], T_train[pick], Y_train_scaled[pick],
            X_test, J,
        )
        per_run_cate.append(cate_r)
    return np.mean(np.stack(per_run_cate, axis=0), axis=0)


def _psid_balanced_subsample(X, T, Y, n_controls: int = 500, seed: int = 42):
    """Mirror `dofm_psid_balanced.py`: keep every treated unit, subsample
    up to n_controls control units. Returns (X, T, Y) shuffled."""
    T_flat = T.flatten()
    idx_t = np.where(T_flat == 1)[0]
    idx_c = np.where(T_flat == 0)[0]
    rng = np.random.default_rng(seed)
    n_keep = min(n_controls, idx_c.size)
    idx_c_keep = rng.choice(idx_c, size=n_keep, replace=False) if n_keep < idx_c.size else idx_c
    keep = np.concatenate([idx_t, idx_c_keep])
    rng.shuffle(keep)
    return X[keep], T[keep], Y[keep]


def evaluate_realization(model, cfg, dataset_cls, r: int,
                          psid_balanced: bool = False,
                          cluster: bool = False,
                          subsample: bool = False,
                          n_repeats: int = 20,
                          stratify_by_t: bool = True,
                          max_n_train: int = 1000):
    ds = dataset_cls()
    cate_ds = ds[r][0]
    X_tr_raw = cate_ds.X_train.astype(np.float32)
    T_tr     = cate_ds.t_train.astype(np.float32).reshape(-1)
    y_tr_raw = cate_ds.y_train.astype(np.float32).reshape(-1)
    X_te_raw = cate_ds.X_test.astype(np.float32)
    true_cate = cate_ds.true_cate.astype(np.float32)
    true_ate  = float(true_cate.mean())

    if psid_balanced:
        X_tr_raw, T_tr, y_tr_raw = _psid_balanced_subsample(X_tr_raw, T_tr, y_tr_raw)

    F = cfg['num_features']
    J = cfg['J']

    # Same preprocessing as our training pipeline (matches
    # PairedInterventionalDataset). This is the single-most-important
    # thing to get right; see the graph2d IHDP eval bug I chased earlier.
    X_tr_std, X_te_std = _standardize_train_test(X_tr_raw, X_te_raw)
    X_tr = _pad_features(X_tr_std, F)
    X_te = _pad_features(X_te_std, F)
    y_scaled, ymin, yrange = _scale_y(y_tr_raw)
    Y_obs = y_scaled.reshape(-1, 1)

    if subsample:
        cate_scaled = _subsample_and_average_cate(
            model, X_tr, T_tr, Y_obs, X_te, J,
            max_n_train=max_n_train, n_repeats=n_repeats,
            stratify_by_t=stratify_by_t, random_state=r,
        )
    elif cluster:
        cate_scaled = _cluster_and_average_cate(
            model, X_tr, T_tr, Y_obs, X_te, J, max_n_train=max_n_train,
        )
    else:
        cate_scaled = _cate_from_joint(model, X_tr, T_tr, Y_obs, X_te, J)
    cate = cate_scaled * yrange / 2.0                         # inverse-scale

    pehe = float(np.sqrt(np.mean((cate - true_cate) ** 2)))
    ate_hat = float(cate.mean())
    ate_rel_err = abs(ate_hat - true_ate) / max(abs(true_ate), 1e-9)
    return dict(
        pehe=pehe,
        ate_rel_err=ate_rel_err,
        cate_preds=cate,
        true_ate=true_ate,
        n_queries=int(true_cate.size),
        n_context=int(X_tr_raw.shape[0]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset',  required=True, choices=list(DATASETS) + ['all'])
    ap.add_argument('--model',    default='ours_fn50')
    ap.add_argument('--exp_name', required=True)
    ap.add_argument('--ckpt',     default=os.environ.get(
        'OURS_CKPT', '/scratch/furkanbd/rpfn_bench_kit/R-PFN/checkpoints/step_50000_final.pt'))
    ap.add_argument('--out',      default=os.environ.get(
        'TABLE1_OUT_ROOT', '/scratch/furkanbd/rpfn_bench_kit/results_table1_ours_fn50'))
    ap.add_argument('--psid-balanced', action='store_true',
                     help='Mirror dofm_psid_balanced.py subsampling: keep all T=1 '
                          '+ 500 random T=0 per realization.')
    ap.add_argument('--cluster', action='store_true',
                     help='(Legacy / A-B baseline.) KMeans-partition train context '
                          'and average per-query CATE across clusters. Empirically '
                          'no better than single-pass on CPS because KMeans '
                          'creates narrow-X clusters whose per-cluster CATE is '
                          'unreliable outside that X domain. Use --subsample.')
    ap.add_argument('--subsample', action='store_true',
                     help='Bagged in-context inference: draw N random subsets of '
                          'size <= max_n_train from the train set, run one forward '
                          'per subset, average per-query CATE. Preserves the '
                          'marginal X distribution in every subset (unlike --cluster), '
                          'so each per-subset CATE is unbiased. Recommended for '
                          'large-context datasets (CPS, PSID).')
    ap.add_argument('--n-repeats', type=int, default=20,
                     help='Number of random subsets to draw in --subsample mode '
                          '(default 20). More = lower variance, linearly more compute.')
    ap.add_argument('--no-stratify-t', action='store_true',
                     help='Disable T=0/T=1 stratification in --subsample mode. '
                          'Only relevant when T is imbalanced (PSID); on balanced '
                          'datasets stratification is a no-op.')
    ap.add_argument('--max-n-train', type=int, default=1000,
                     help='Per-subset (or per-cluster) max context size (default '
                          '1000, matches CausalPFN training-time context bound).')
    args = ap.parse_args()

    assert os.path.isfile(args.ckpt), f'checkpoint not found: {args.ckpt}'
    print(f'[bootstrap] device={DEVICE}  ckpt={args.ckpt}  out={args.out}')

    model, cfg = load_model(args.ckpt)
    print(f'[bootstrap] J={cfg["J"]}  F={cfg["num_features"]}  '
          f'params={sum(p.numel() for p in model.parameters()):,}')

    datasets_to_run = list(DATASETS) if args.dataset == 'all' else [args.dataset]
    for ds_name in datasets_to_run:
        ds_cls = DATASETS[ds_name]
        n_tables = ds_cls().n_tables
        exp_dir = os.path.join(args.out, args.exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        print(f'\n══ {ds_name}  (n_realizations={n_tables}) ══')

        pehes, atrerrs = [], []
        t0 = time.time()
        for r in range(n_tables):
            res = evaluate_realization(
                model, cfg, ds_cls, r,
                psid_balanced=args.psid_balanced,
                cluster=args.cluster,
                subsample=args.subsample,
                n_repeats=args.n_repeats,
                stratify_by_t=not args.no_stratify_t,
                max_n_train=args.max_n_train,
            )
            pehes.append(res['pehe']); atrerrs.append(res['ate_rel_err'])
            with open(os.path.join(exp_dir, f'{args.model}_{ds_name}_{r}'), 'wb') as f:
                pickle.dump(dict(model=args.model, dataset=ds_name, realization=r,
                                 **res), f)
            print(f'  r={r:03d}  pehe={res["pehe"]:>10.3f}  '
                  f'ate_rel_err={res["ate_rel_err"]:>6.3f}  '
                  f'({time.time()-t0:.0f}s)', flush=True)

        pehes = np.array(pehes); atrerrs = np.array(atrerrs)
        sem_pehe = pehes.std(ddof=1) / np.sqrt(len(pehes)) if len(pehes) > 1 else 0.0
        sem_ate  = atrerrs.std(ddof=1) / np.sqrt(len(atrerrs)) if len(atrerrs) > 1 else 0.0
        print(f'\n{ds_name} summary   PEHE = {pehes.mean():>10.3f} ± {sem_pehe:>6.3f}   '
              f'ATE_rel_err = {atrerrs.mean():.3f} ± {sem_ate:.3f}')


if __name__ == '__main__':
    main()
