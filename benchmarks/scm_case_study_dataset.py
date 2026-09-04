"""Adapter that exposes DoPFN's 6 synthetic case-study pkls as an
IHDPDataset-compatible interface, so any of our existing realcause eval
scripts (eval_cpfn2d_realcause.py, eval_graph2d_realcause.py, etc.) can
consume them without further changes.

Each pkl is a DoPFN `InterventionalDataset`:
  .x    — features INCLUDING T in the first column
  .y    — outcomes
  .cate — per-query ground-truth CATE (available after .generate_valid_split)

We split each realization into train (200 rows, matches DoPFN's default) and
test (rest), then adopt the IHDPDataset convention: T is a separate array
(t_train), features exclude T (X_train), true_cate is per test query.

Case study names (per DoPFN paper Table 1 / Fig 2):
  Observed_Confounder
  Observed_Mediator
  Observed_Mediator_and_Confounder
  Unobserved_Confounder
  Frontdoor_Criterion
  Backdoor_Criterion

Env vars driving path lookup:
  DOPFN_DATA_ROOT (default: /scratch/furkanbd/rpfn_bench_kit/external/dopfn/data/prior_sampling)
"""
from __future__ import annotations
import os
import pickle
import numpy as np
from dataclasses import dataclass


_CASE_STUDIES = (
    'Observed_Confounder',
    'Observed_Mediator',
    'Observed_Mediator_and_Confounder',
    'Unobserved_Confounder',
    'Frontdoor_Criterion',
    'Backdoor_Criterion',
)


@dataclass
class _CATE_Slice:
    """Mirrors the fields our realcause eval scripts read off IHDPDataset[r][0].

    New-format-only fields (populated when the pkl carries them; else None):
      mu_0        per-test-query structural mean E[Y | do(T=0), X=x_i]
      mu_1        per-test-query structural mean E[Y | do(T=1), X=x_i]
      sigma_eps   scalar Y-noise std used by the SCM
      rho_y_noise correlation between the Y-node's obs / int noise draws
      test_row_indices  which rows of ds.x are the interventional queries
                        (aligned with mu_0 / mu_1 / true_cate)
    """
    X_train: np.ndarray
    t_train: np.ndarray
    y_train: np.ndarray
    X_test:  np.ndarray
    true_cate: np.ndarray
    mu_0: np.ndarray | None = None
    mu_1: np.ndarray | None = None
    sigma_eps: float | None = None
    rho_y_noise: float | None = None
    test_row_indices: np.ndarray | None = None


class SCMCaseStudyDataset:
    """DoPFN 6-case-study pkl → IHDPDataset-shaped API.

    Usage mirrors IHDPDataset:
        ds = SCMCaseStudyDataset('Observed_Confounder')
        cate_ds, _ = ds[r]     # r in [0, ds.n_tables)
        X_train = cate_ds.X_train     # (N_train, d) — T stripped
        t_train = cate_ds.t_train     # (N_train,) — 0/1
        y_train = cate_ds.y_train     # (N_train,)
        X_test  = cate_ds.X_test      # (N_test,  d) — T stripped
        true_cate = cate_ds.true_cate # (N_test,)
    """

    n_train: int = 200      # matches DoPFN's default get_splits()

    def __init__(self, case_study: str,
                 data_root: str | None = None,
                 n_train: int | None = None,
                 true_ate_shift: float | None = None):
        if case_study not in _CASE_STUDIES:
            raise ValueError(f'{case_study!r} not in {_CASE_STUDIES}')
        self.case_study = case_study
        # DGP-level shift: add `true_ate_shift` to Y wherever T=1 (observational)
        # AND to true_cate. Result: ATE_true shifts by +true_ate_shift on every
        # realization → never near zero → rel-err denominator well-defined.
        # Controlled by env var SCM_TRUE_ATE_SHIFT (default 0 = no shift).
        if true_ate_shift is None:
            true_ate_shift = float(os.environ.get('SCM_TRUE_ATE_SHIFT', '0.0'))
        self.true_ate_shift = float(true_ate_shift)
        self.data_root = data_root or os.environ.get(
            'DOPFN_DATA_ROOT',
            '/scratch/furkanbd/rpfn_bench_kit/external/dopfn/data/prior_sampling',
        )
        self.case_dir = os.path.join(self.data_root, case_study)
        if not os.path.isdir(self.case_dir):
            raise FileNotFoundError(
                f'case-study dir not found: {self.case_dir}\n'
                f'Set DOPFN_DATA_ROOT to the parent of prior_sampling/'
            )
        # Sort by realization number (Observed_Confounder_1.pkl, _2, ..., _100)
        self.pkl_paths = sorted(
            [f for f in os.listdir(self.case_dir) if f.endswith('.pkl')],
            key=lambda p: int(p.rsplit('_', 1)[1].split('.')[0]),
        )
        self.pkl_paths = [os.path.join(self.case_dir, p) for p in self.pkl_paths]
        if n_train is not None:
            self.n_train = int(n_train)

    @property
    def n_tables(self):
        return len(self.pkl_paths)

    def __len__(self):
        return len(self.pkl_paths)

    def _load_one(self, r: int) -> _CATE_Slice:
        ds = _pickle_load_with_dopfn_shim(self.pkl_paths[r])
        x = np.asarray(ds.x, dtype=np.float32)      # (N, d+1), T in col 0
        y = np.asarray(ds.y, dtype=np.float32).reshape(-1)
        N = x.shape[0]

        # NEW-FORMAT DETECTION — the bivariate-Y-noise regenerated pkls set:
        #   ds.mu_0_per_query, ds.mu_1_per_query, ds.sigma_eps, ds.rho_y_noise
        #   ds.int_row_indices  (rows of ds.x that are interventional queries)
        # For these we use a DETERMINISTIC train/test split that respects the
        # obs/int row separation: train from obs rows, test from int rows.
        # For old-format pkls we fall back to the legacy random permutation.
        mu_0_full = getattr(ds, 'mu_0_per_query', None)
        mu_1_full = getattr(ds, 'mu_1_per_query', None)
        sigma_eps = getattr(ds, 'sigma_eps', None)
        rho_y     = getattr(ds, 'rho_y_noise', None)
        int_idcs  = getattr(ds, 'int_row_indices', None)
        n_int_rows = getattr(ds, 'n_int_rows', None)
        n_obs_rows = getattr(ds, 'n_obs_rows', None)
        is_new_format = (mu_0_full is not None and mu_1_full is not None
                         and int_idcs is not None and n_obs_rows is not None)

        if is_new_format:
            int_idcs = np.asarray(int_idcs, dtype=np.int64)
            obs_idcs = np.setdiff1d(np.arange(N), int_idcs, assume_unique=False)
            rng = np.random.default_rng(42 + r)
            obs_perm = rng.permutation(len(obs_idcs))
            n_tr = min(self.n_train, len(obs_idcs))
            tr_idx = obs_idcs[obs_perm[:n_tr]]
            te_idx = int_idcs                         # ALL interventional queries
        else:
            # Legacy random-permutation split (matches previous behaviour).
            rng = np.random.default_rng(42 + r)
            perm = rng.permutation(N)
            n_tr = min(self.n_train, N - 1)
            tr_idx, te_idx = perm[:n_tr], perm[n_tr:]

        # For CATE truth on the test rows we need per-x counterfactual outcomes.
        # DoPFN's `.cate` on an InterventionalDataset is populated by
        # generate_valid_split. We re-derive it here to avoid pulling in
        # `datasets/__init__.py` (which needs pandas etc.):
        # `.do_scm` samples y under do(T=t). Following inference_example.py,
        # the test data supplies (x_int, y_int) with T=0 vs T=1 in the first
        # column. Query CATE per unique X_test-with-covariates:
        #   – ds.x has shape (N, d+1) with T in col 0 for BOTH obs and int
        #     (they're concatenated in generate_valid_split); the loader here
        #     just receives the concatenated tensor.
        # If ds already carries a .cate attribute (post-split), use it.
        cate_true = None
        if hasattr(ds, 'cate') and ds.cate is not None:
            cate_arr = np.asarray(ds.cate, dtype=np.float32).reshape(-1)
            if is_new_format:
                # For new-format pkls, ds.cate is per-row over the full x
                # (obs || int concatenation); pick the int rows.
                if cate_arr.shape[0] == N:
                    cate_true = cate_arr[te_idx]
                elif cate_arr.shape[0] == len(int_idcs):
                    cate_true = cate_arr
            else:
                if cate_arr.shape[0] == N:
                    cate_true = cate_arr[te_idx]
        if cate_true is None:
            # Derive per-test-query CATE by querying the SCM directly.
            # do_scm.forward(x_features_without_t, do_t=1/0) returns y — but
            # signature varies. Fall back to a diff of neighbouring T=0/T=1
            # rows if we can't call the SCM safely.
            # Simplest robust path: assume every test row pairs with a
            # counterfactual row (DoPFN pkls are usually duplicated).
            X_te = x[te_idx, 1:]
            T_te = x[te_idx, 0]
            Y_te = y[te_idx]
            cate_true = _cate_from_paired_rows(X_te, T_te, Y_te)

        # Align mu_0 / mu_1 with te_idx for the new-format pkls.
        mu_0 = mu_1 = None
        if is_new_format:
            mu0_arr = np.asarray(mu_0_full, dtype=np.float32).reshape(-1)
            mu1_arr = np.asarray(mu_1_full, dtype=np.float32).reshape(-1)
            # mu_*_per_query is stored aligned to the INT rows (length n_int_rows).
            # te_idx are absolute row indices in ds.x — subtract n_obs_rows to get
            # positions within the int block.
            local_pos = te_idx - int(n_obs_rows)
            if (local_pos.min() >= 0) and (local_pos.max() < len(mu0_arr)):
                mu_0 = mu0_arr[local_pos]
                mu_1 = mu1_arr[local_pos]

        X_train = x[tr_idx, 1:]; t_train = x[tr_idx, 0]; y_train = y[tr_idx]
        X_test  = x[te_idx, 1:]

        # Apply DGP shift: treated units get +shift in Y (observational);
        # true CATE = Y1 - Y0 also gets +shift (since we shift Y1 branch).
        if self.true_ate_shift != 0.0:
            treated_mask = t_train > 0.5
            y_train = y_train.copy()
            y_train[treated_mask] += self.true_ate_shift
            cate_true = cate_true + self.true_ate_shift

        return _CATE_Slice(
            X_train=X_train.astype(np.float32),
            t_train=t_train.astype(np.float32),
            y_train=y_train.astype(np.float32),
            X_test=X_test.astype(np.float32),
            true_cate=cate_true.astype(np.float32),
            mu_0=None if mu_0 is None else mu_0.astype(np.float32),
            mu_1=None if mu_1 is None else mu_1.astype(np.float32),
            sigma_eps=None if sigma_eps is None else float(sigma_eps),
            rho_y_noise=None if rho_y is None else float(rho_y),
            test_row_indices=te_idx.astype(np.int64),
        )

    def __getitem__(self, r: int):
        """Return (cate_ds, ate_ds). ate_ds unused by our eval scripts."""
        cate = self._load_one(r)
        return cate, None


def _pickle_load_with_dopfn_shim(pkl_path: str):
    """The pkl unpickles a DoPFN `datasets.InterventionalDataset` object.
    Our shims stub `datasets` with MagicMock so causalpfn imports work — that
    breaks pickle here. Temporarily install the REAL DoPFN datasets module,
    load, then restore the shim.
    """
    import sys as _sys
    dopfn_root = os.environ.get('DOPFN_ROOT') or os.environ.get('DOPFN')
    if not dopfn_root:
        # Try to infer from DOPFN_DATA_ROOT (which is dopfn/data/prior_sampling)
        d = os.environ.get('DOPFN_DATA_ROOT', '')
        if 'data/prior_sampling' in d:
            dopfn_root = d.split('/data/prior_sampling')[0]
    if not dopfn_root or not os.path.isdir(dopfn_root):
        raise RuntimeError(
            'DOPFN_ROOT env var must point to the DoPFN repo root to unpickle '
            'case-study data (has a `datasets/` package). Got: '
            f'{dopfn_root!r}'
        )

    # DoPFN's pickle chain pulls in these top-level packages. We must save
    # any pre-existing ones (usually UWYK's `utils` or a shimmed `datasets`),
    # let the real DoPFN modules take over during unpickling, then restore.
    shadowed = ('datasets', 'utils', 'priors', 'models', 'scripts')

    saved = {}
    for name in list(_sys.modules):
        if name in shadowed or any(name.startswith(p + '.') for p in shadowed):
            saved[name] = _sys.modules.pop(name)
    inserted = False
    if dopfn_root not in _sys.path:
        _sys.path.insert(0, dopfn_root); inserted = True
    try:
        import importlib
        # Prime the real DoPFN packages so pickle finds their classes
        for p in shadowed:
            try: importlib.import_module(p)
            except Exception: pass
        with open(pkl_path, 'rb') as f:
            obj = pickle.load(f)
    finally:
        for name in list(_sys.modules):
            if name in shadowed or any(name.startswith(p + '.') for p in shadowed):
                _sys.modules.pop(name)
        if inserted and dopfn_root in _sys.path:
            _sys.path.remove(dopfn_root)
        _sys.modules.update(saved)
    return obj


def _cate_from_paired_rows(X_te: np.ndarray, T_te: np.ndarray, Y_te: np.ndarray) -> np.ndarray:
    """DoPFN's InterventionalDataset stores test rows as duplicated X with
    T=0 and T=1. Pair rows by exact X match, return CATE per unique X.

    If no pairing structure is found, fall back to per-row Y (CATE undefined,
    just returns zeros so the eval still runs — will report as PEHE = |y|)."""
    n = X_te.shape[0]
    # Fast path: consecutive rows come in (T=0, T=1) pairs
    if n % 2 == 0:
        even_X, odd_X = X_te[0::2], X_te[1::2]
        even_T, odd_T = T_te[0::2], T_te[1::2]
        if (np.abs(even_X - odd_X).max() < 1e-6
                and np.all(even_T < 0.5) and np.all(odd_T > 0.5)):
            return Y_te[1::2] - Y_te[0::2]
    # Fallback — no pairing detected
    return np.zeros(n, dtype=np.float32)
