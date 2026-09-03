"""Evaluate our fn=50 checkpoint mirroring UWYK's predictive S-learner
inference pipeline (predmodel_Slearner_full_context.py) as literally as
architectural differences allow.

Deltas vs predictive we CANNOT eliminate without retraining:
  - Head: predictive is a single-arm K+4 regression head; ours is a
    K^2 + 9 + 4 joint head over (Y_do0, Y_do1). We emulate their
    "predict mean per arm" by taking the arm-marginal mean from our
    joint output for the corresponding pass.
  - T at query: predictive was trained with target-encoded T (mean_y|T)
    in its X-input concat; our fn=50 was trained with null_t_intv only.
    We use the target-encoded T here anyway to match the protocol,
    knowing this pushes our model out of distribution.

Everything else follows predictive's script step-for-step:
  1. Target encoding: t_train_encoded = mean(y|t=0 or t=1); same for
     t_intv_0 / t_intv_1.
  2. Outlier clip X at q=0.99 (train quantiles applied to both).
  3. Standardize X per-column (train mean/std).
  4. Scale Y to [-1, 1] with train min/range.
  5. Two forward passes: one with T_intv = mean_y_t1, one with
     T_intv = mean_y_t0.
  6. Extract single-arm mean per pass from the joint's matching marginal.
  7. CATE_scaled = mean_arm1 - mean_arm0.
  8. Inverse-scale: CATE = CATE_scaled * y_range / 2.

Output pkl format matches UWYK's evaluate_pipeline so the shared
aggregator works.
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch


_HERE     = os.path.abspath(os.path.dirname(__file__))
_R_PFN    = os.path.abspath(os.path.join(_HERE, '..', '..'))
CAUSALPFN = os.environ.get(
    'CAUSALPFN_ROOT', '/scratch/furkanbd/rpfn_bench_kit/external/causalpfn')
sys.path.insert(0, _R_PFN)
sys.path.insert(0, CAUSALPFN)
_CAUSALPFN_SRC = os.path.join(CAUSALPFN, 'src')
if os.path.isdir(_CAUSALPFN_SRC):
    sys.path.insert(0, _CAUSALPFN_SRC)
_SHIMS = os.path.join(_HERE, 'shims')
if os.path.isdir(_SHIMS) and _SHIMS not in sys.path:
    sys.path.insert(0, _SHIMS)

from benchmarks import (          # noqa: E402
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


def _pad_features(X: np.ndarray, F: int) -> np.ndarray:
    if X.shape[1] == F:
        return X
    if X.shape[1] > F:
        return X[:, :F]
    return np.hstack([X, np.zeros((X.shape[0], F - X.shape[1]), dtype=X.dtype)])


def _predictive_preprocess(X_train, X_test, y_train, q: float = 0.99):
    """Line-for-line copy of predmodel_Slearner_full_context.py's X-and-Y
    preprocessing: 0.99-quantile clip on X, then z-score, then scale Y."""
    # X: outlier clip
    lo = np.quantile(X_train, 1.0 - q, axis=0)
    hi = np.quantile(X_train, q,       axis=0)
    Xtr = np.clip(X_train, lo, hi)
    Xte = np.clip(X_test,  lo, hi)
    # X: z-score
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0,  keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    # Y: scale to [-1, 1]
    ymin  = float(y_train.min())
    yrange = float(max(y_train.max() - ymin, 1e-9))
    ytr_scaled = (2.0 * (y_train - ymin) / yrange - 1.0).astype(np.float32)
    return Xtr, Xte, ytr_scaled, ymin, yrange


@torch.no_grad()
def _one_pass_arm_mean(model, X_train_std, T_train_encoded, Y_train_scaled,
                         X_test_std, T_intv_encoded_scalar, arm: int, J: int) -> np.ndarray:
    """One forward pass with T_intv broadcast to `T_intv_encoded_scalar`;
    return E[y_arm] under the joint's matching marginal."""
    B = 1
    X_obs = torch.from_numpy(X_train_std.astype(np.float32)).unsqueeze(0).to(DEVICE)
    T_obs = torch.from_numpy(T_train_encoded.astype(np.float32)).reshape(1, -1, 1).to(DEVICE)
    Y_obs = torch.from_numpy(Y_train_scaled).unsqueeze(0).to(DEVICE)
    X_intv = torch.from_numpy(X_test_std.astype(np.float32)).unsqueeze(0).to(DEVICE)

    # Broadcast the (target-encoded) scalar to every query row.
    T_intv = torch.full(
        (B, X_intv.shape[1], 1), fill_value=float(T_intv_encoded_scalar),
        device=DEVICE, dtype=X_intv.dtype,
    )

    out = model(X_obs, T_obs, Y_obs, X_intv, T_intv=T_intv)
    logits = out['predictions'] if isinstance(out, dict) else out
    p = torch.softmax(logits[..., : J * J], dim=-1).reshape(B, -1, J, J)

    centres = torch.linspace(-1.0 + 1.0 / J, 1.0 - 1.0 / J, J, device=logits.device)
    if arm == 0:
        p_arm = p.sum(dim=-1)
    else:
        p_arm = p.sum(dim=-2)
    e_arm = (p_arm * centres.view(1, 1, J)).sum(dim=-1)
    return e_arm.squeeze(0).cpu().numpy()


def evaluate_realization(model, cfg, dataset_cls, r: int):
    ds = dataset_cls()
    cate_ds = ds[r][0]
    X_tr_raw = cate_ds.X_train.astype(np.float32)
    T_tr_raw = cate_ds.t_train.astype(np.float32).reshape(-1)
    y_tr_raw = cate_ds.y_train.astype(np.float32).reshape(-1)
    X_te_raw = cate_ds.X_test.astype(np.float32)
    true_cate = cate_ds.true_cate.astype(np.float32)
    true_ate  = float(true_cate.mean())

    F = cfg['num_features']
    J = cfg['J']

    # (1) Target-encode T (predictive's step 1)
    mean_y_t0 = y_tr_raw[T_tr_raw == 0].mean()
    mean_y_t1 = y_tr_raw[T_tr_raw == 1].mean()
    T_tr_encoded = np.where(T_tr_raw == 0, mean_y_t0, mean_y_t1).astype(np.float32)

    # (2)-(4) X clip + z-score, Y scale (predictive's preprocessing)
    X_tr_std, X_te_std, y_scaled, ymin, yrange = _predictive_preprocess(
        X_tr_raw, X_te_raw, y_tr_raw
    )
    X_tr = _pad_features(X_tr_std, F)
    X_te = _pad_features(X_te_std, F)

    # (5)-(6) Two forward passes, extract arm marginal per pass
    e_y1 = _one_pass_arm_mean(model, X_tr, T_tr_encoded, y_scaled.reshape(-1, 1),
                                X_te, float(mean_y_t1), arm=1, J=J)
    e_y0 = _one_pass_arm_mean(model, X_tr, T_tr_encoded, y_scaled.reshape(-1, 1),
                                X_te, float(mean_y_t0), arm=0, J=J)

    # (7)-(8) CATE + inverse scale
    cate_scaled = e_y1 - e_y0
    cate = cate_scaled * yrange / 2.0

    pehe = float(np.sqrt(np.mean((cate - true_cate) ** 2)))
    ate_hat = float(cate.mean())
    ate_rel_err = abs(ate_hat - true_ate) / max(abs(true_ate), 0.1)
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
    ap.add_argument('--model',    default='ours_fn50_predstyle')
    ap.add_argument('--exp_name', required=True)
    ap.add_argument('--ckpt',     default=os.environ.get(
        'OURS_CKPT', '/scratch/furkanbd/rpfn_bench_kit/R-PFN/checkpoints/step_50000_final.pt'))
    ap.add_argument('--out',      default=os.environ.get(
        'TABLE1_OUT_ROOT', '/scratch/furkanbd/rpfn_bench_kit/results_table1_ours_fn50'))
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
        print(f'\n══ {ds_name}  (n_realizations={n_tables})  [predictive-mirror] ══')

        pehes, atrerrs = [], []
        t0 = time.time()
        for r in range(n_tables):
            res = evaluate_realization(model, cfg, ds_cls, r)
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
