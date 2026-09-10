"""Loader + PEHE / eps_ATE metrics for the UWYK Fig-3 / Fig-4 PEHE benchmark.

Companion to ``generate_pehe_benchmark.py``. A realization npz carries matched
potential outcomes, so both metrics follow UWYK's own definitions (paper F.2.1):

    sqrt(PEHE) = sqrt( mean_i ( tau_hat(x_i) - tau_i )^2 ),   tau_i = y1_i - y0_i
    eps_ATE    = | ATE_hat - ATE | / | ATE |

Caveats that matter when reading the numbers
--------------------------------------------
* ``tau_i`` carries unit-level noise that ``x_i`` does not determine, so even a
  perfect model has PEHE > 0. That floor is method-independent (it is the same
  labels for everyone) but it is not zero. ``oracle_pehe_floor`` reports it.
* For ``path_YT`` and ``path_independent_TY`` the true CATE is identically 0, so
  ATE == 0 and ``eps_ATE`` is undefined (division by zero). Use ``rmse_cate``
  there instead -- it reduces to the RMS of the predicted effect, i.e. a direct
  false-positive-effect measure. ``eps_ate`` returns NaN rather than inf.
"""
from __future__ import annotations

import glob
import os

import numpy as np

REGIMES = ("path_TY", "path_YT", "path_independent_TY")
# Regimes with no directed T -> Y path: the true CATE is identically zero.
NULL_EFFECT_REGIMES = ("path_YT", "path_independent_TY")

_ARRAY_KEYS = ("X_train", "T_train", "Y_train", "X_test", "Y_do0", "Y_do1",
               "true_cate", "anc_matrix", "adj_matrix")


def load_realization(path: str) -> dict:
    """Load one r<idx>.npz into a plain dict (0-d arrays unwrapped to scalars)."""
    with np.load(path, allow_pickle=True) as z:
        out = {}
        for k in z.files:
            v = z[k]
            out[k] = v if k in _ARRAY_KEYS else (v.item() if v.ndim == 0 else v)
    return out


def load_cell(data_root: str, prior: str, n_nodes: int, regime: str,
              hide: float) -> list[dict]:
    """Load every realization for one (prior, n, regime, hide) cell, in r-order."""
    cell = os.path.join(data_root, prior, f"{n_nodes}node", regime, f"hide_{hide}")
    paths = sorted(glob.glob(os.path.join(cell, "r*.npz")),
                   key=lambda p: int(os.path.basename(p)[1:-4]))
    if not paths:
        raise FileNotFoundError(f"no realizations under {cell}")
    return [load_realization(p) for p in paths]


def rmse_cate(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """sqrt(PEHE). Named for what it is, since 'PEHE' is used for both in the wild."""
    tau_hat = np.asarray(tau_hat, dtype=np.float64).reshape(-1)
    tau_true = np.asarray(tau_true, dtype=np.float64).reshape(-1)
    if tau_hat.shape != tau_true.shape:
        raise ValueError(f"shape mismatch: {tau_hat.shape} vs {tau_true.shape}")
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))


# UWYK's Table 1 header writes this as sqrt(PEHE); keep both names bound.
sqrt_pehe = rmse_cate


def eps_ate(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """Relative ATE error. NaN when the true ATE is 0 (the null-effect regimes)."""
    ate_true = float(np.mean(np.asarray(tau_true, dtype=np.float64)))
    ate_hat = float(np.mean(np.asarray(tau_hat, dtype=np.float64)))
    if ate_true == 0.0:
        return float("nan")
    return abs(ate_hat - ate_true) / abs(ate_true)


def oracle_pehe_floor(tau_true: np.ndarray) -> float:
    """sqrt(PEHE) of a constant predictor: the within-dataset sd of the true ITE.

    NOT the best any model can reach -- that is sd(tau - E[tau|X]), which is <=
    this. This is the score of a model that ignores X entirely, so it is the bar
    a covariate-using method must beat, not a floor it cannot pass.

    A model with no usable covariate information (e.g. data generated with
    ``--test-feature-mask-fraction 1.0``) cannot beat this. Comparing a method's
    score against it tells you whether it is exploiting heterogeneity at all.
    """
    return float(np.std(np.asarray(tau_true, dtype=np.float64)))


def summarize(records: list[dict], tau_hats: list[np.ndarray]) -> dict:
    """Aggregate one cell: per-realization metrics + mean/std across realizations."""
    if len(records) != len(tau_hats):
        raise ValueError(f"{len(records)} records vs {len(tau_hats)} predictions")
    per_pehe, per_eps, per_floor = [], [], []
    for rec, th in zip(records, tau_hats):
        tt = rec["true_cate"]
        per_pehe.append(rmse_cate(th, tt))
        per_eps.append(eps_ate(th, tt))
        per_floor.append(oracle_pehe_floor(tt))
    per_pehe = np.asarray(per_pehe)
    per_eps = np.asarray(per_eps)
    return {
        "n_realizations": len(records),
        "sqrt_pehe_mean": float(per_pehe.mean()),
        "sqrt_pehe_std": float(per_pehe.std()),
        "eps_ate_mean": float(np.nanmean(per_eps)) if not np.all(np.isnan(per_eps)) else float("nan"),
        "eps_ate_std": float(np.nanstd(per_eps)) if not np.all(np.isnan(per_eps)) else float("nan"),
        "oracle_floor_mean": float(np.mean(per_floor)),
        "n_descendant_free": sum(1 for r in records
                                 if int(r["n_descendant_features"]) == 0),
        "per_realization_sqrt_pehe": per_pehe.tolist(),
    }
