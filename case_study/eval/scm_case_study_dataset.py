"""Drop-in `SCMCaseStudyDataset` backed by OUR generated case-study npz.

The RealCause eval scripts under `benchmarks/eval_*/` already know how to
evaluate the six synthetic case studies: each carries a `_SCM_CASES` tuple and
does

    if name in _SCM_CASES:
        from scm_case_study_dataset import SCMCaseStudyDataset
        return SCMCaseStudyDataset(name)

and switches the ATE metric to L1 `|ate_hat - true_ate|`. That import is *bare*,
so putting THIS folder first on PYTHONPATH shadows the DoPFN-pkl loader at
`benchmarks/scm_case_study_dataset.py` with the one below — which instead reads
the `.npz` files produced by `case_study/generation.py`. No edits to any model
script are required.

It presents the exact interface those scripts consume:
    ds = SCMCaseStudyDataset(name)   # name ∈ the six case studies
    ds.n_tables                       # number of realizations
    cate_ds, _ = ds[r]                # r in [0, n_tables)
    cate_ds.X_train / .t_train / .y_train   # observational context
    cate_ds.X_test / .true_cate             # CATE query covariates + ground truth

Which npz cell to read is selected by environment variables (the model scripts
only ever pass the case-study name):
    CASE_STUDY_DATA_ROOT   root written by generation.py, e.g.
                           case_study/data_shift+2   (REQUIRED)
    CASE_STUDY_N           context size subdir, e.g. 200 / 500 / 1000  (REQUIRED)

Optional knobs (mirror the DoPFN-pkl loader's env API so context/query sweeps
work unchanged):
    SCM_N_TRAIN   cap the observational context to the first n rows (default:
                  all N rows — the whole realization is the context).
    SCM_N_QUERY   cap the CATE query set to the first q rows (default: all rows).

Split convention: each realization is N observational rows, every row carrying
its own ground-truth CATE. We therefore evaluate the population CATE in-sample —
the context is the observational rows and the query set is those same units'
covariates (X_test = X_train covariates, true_cate = per-row cate). This matches
how the DoPFN case studies are scored (predict CATE for the observed units).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

CASE_STUDIES = (
    "Observed_Confounder",
    "Backdoor_Criterion",
    "Observed_Mediator",
    "Observed_Mediator_and_Confounder",
    "Unobserved_Confounder",
    "Frontdoor_Criterion",
)


@dataclass
class _CATE_Slice:
    """Mirrors the fields the RealCause eval scripts read off `ds[r][0]`.
    `mu_0` / `mu_1` are carried through for any density/oracle use; the point
    metrics only need X_train / t_train / y_train / X_test / true_cate."""
    X_train: np.ndarray
    t_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    true_cate: np.ndarray
    mu_0: np.ndarray | None = None
    mu_1: np.ndarray | None = None
    sigma_eps: float | None = None
    rho_y_noise: float | None = None
    test_row_indices: np.ndarray | None = None


class SCMCaseStudyDataset:
    def __init__(self, case_study: str, data_root: str | None = None,
                 n_context: int | None = None):
        if case_study not in CASE_STUDIES:
            raise ValueError(f"{case_study!r} not in {CASE_STUDIES}")
        self.case_study = case_study

        self.data_root = data_root or os.environ.get("CASE_STUDY_DATA_ROOT")
        if not self.data_root:
            raise RuntimeError(
                "CASE_STUDY_DATA_ROOT is unset — point it at a generation.py "
                "output root (e.g. case_study/data_shift+2).")

        n = n_context if n_context is not None else os.environ.get("CASE_STUDY_N")
        if n is None:
            raise RuntimeError("CASE_STUDY_N is unset — set it to the context "
                               "size subdir to evaluate (e.g. 200).")
        self.n = int(n)

        self.case_dir = os.path.join(self.data_root, case_study, f"N{self.n}")
        if not os.path.isdir(self.case_dir):
            raise FileNotFoundError(f"case-study cell not found: {self.case_dir}")
        self.pkl_paths = sorted(
            glob.glob(os.path.join(self.case_dir, f"{case_study}_*.npz")),
            key=lambda p: int(os.path.basename(p).rsplit("_", 1)[1].split(".")[0]))
        if not self.pkl_paths:
            raise FileNotFoundError(f"no .npz realizations under {self.case_dir}")

        _ntr = os.environ.get("SCM_N_TRAIN")
        self.n_train = int(_ntr) if _ntr else None
        _nq = os.environ.get("SCM_N_QUERY")
        self.n_query = int(_nq) if _nq else None

    @property
    def n_tables(self) -> int:
        return len(self.pkl_paths)

    def __len__(self) -> int:
        return len(self.pkl_paths)

    def _load_one(self, r: int) -> _CATE_Slice:
        d = np.load(self.pkl_paths[r], allow_pickle=True)
        X = np.asarray(d["X"], dtype=np.float32)
        T = np.asarray(d["T"], dtype=np.float32).reshape(-1)
        Y = np.asarray(d["Y"], dtype=np.float32).reshape(-1)
        cate = np.asarray(d["cate"], dtype=np.float32).reshape(-1)
        mu_0 = np.asarray(d["mu_0"], dtype=np.float32).reshape(-1)
        mu_1 = np.asarray(d["mu_1"], dtype=np.float32).reshape(-1)

        n_tr = min(self.n_train, X.shape[0]) if self.n_train else X.shape[0]
        n_q = min(self.n_query, X.shape[0]) if self.n_query else X.shape[0]

        return _CATE_Slice(
            X_train=X[:n_tr], t_train=T[:n_tr], y_train=Y[:n_tr],
            X_test=X[:n_q], true_cate=cate[:n_q],
            mu_0=mu_0[:n_q], mu_1=mu_1[:n_q],
            test_row_indices=np.arange(n_q, dtype=np.int64),
        )

    def __getitem__(self, r: int):
        """Return (cate_ds, ate_ds). ate_ds is unused by the eval scripts."""
        return self._load_one(r), None
