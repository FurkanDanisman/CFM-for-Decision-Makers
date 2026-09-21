"""Present Do-PFN's semi-real known-graph datasets (sales, law_race) to every
harness, using the same shim the ComplexMech benchmark uses.

Every eval harness already accepts a dataset object exposing

    ds.n_tables        how many "realizations" there are
    ds[r]              -> (cate_slice, adj_slice)

with X_train / t_train / y_train / X_test / true_cate on the slice. ComplexMech
reaches all of them through benchmarks/uwyk_fig34_dataset.py on exactly that
contract, so wiring these two datasets in needs an adapter, not thirteen edits.

Dataset names
-------------
    SEMIREAL_sales
    SEMIREAL_law_race

A REALIZATION HERE IS A SPLIT, and there are only five of them --
`generate_valid_split(split_number=r+1, n_splits=5)`, the same protocol
inference_example.py uses, so numbers stay comparable with Do-PFN's published
ones. That is a much smaller denominator than the 100 realizations RealCause and
the case studies use, and per-realization coverage over five points has wide
error bars. Report n alongside any coverage from this benchmark.

The loader needs Do-PFN's repo on sys.path and its cwd, because its datasets
module resolves artifacts by relative path. DOPFN_ROOT or --dopfn supplies it.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

import numpy as np

_NAME_RE = re.compile(r"^SEMIREAL_(sales|law_race)$")
_DEFAULT_N_SPLITS = int(os.environ.get("SEMIREAL_N_SPLITS", "5"))


def dataset_names() -> tuple[str, ...]:
    return ("SEMIREAL_sales", "SEMIREAL_law_race")


def parse_name(name: str):
    """-> the Do-PFN dataset name, or None if this is not a semi-real name."""
    m = _NAME_RE.match(str(name))
    return m.group(1) if m else None


@dataclass
class _CATESlice:
    X_train: np.ndarray
    t_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    true_cate: np.ndarray
    # Provenance, so a table row can be traced back to its split.
    source_realization: int = -1
    n_real_features: int = -1
    n_context: int = -1


def _to_np(a):
    if hasattr(a, "detach"):
        a = a.detach().cpu()
    return np.asarray(a)


def _dopfn_root() -> str:
    root = os.environ.get("DOPFN_ROOT", "")
    if root and os.path.isdir(root):
        return root
    raise RuntimeError(
        "DOPFN_ROOT is not set or does not exist. The semi-real datasets are "
        "loaded by Do-PFN's own datasets module, which resolves its artifacts by "
        "RELATIVE path, so both the import path and the working directory must "
        "point at the Do-PFN repo."
    )


class DoPFNSemiRealDataset:
    """One Do-PFN semi-real dataset, indexed by split.

    Splits are generated lazily and cached: generate_valid_split is
    deterministic in split_number, so indexing twice returns identical data --
    which matters because the density dump and the point estimate are separate
    passes over the same ds[r].
    """

    def __init__(self, name: str, n_splits: int = _DEFAULT_N_SPLITS,
                 max_context: int = 0):
        ds_name = parse_name(name)
        if ds_name is None:
            raise ValueError(f"{name!r} is not a semi-real dataset name")
        self.name = name
        self.ds_name = ds_name
        self.n_splits = int(n_splits)
        self.n_tables = self.n_splits
        self.max_context = int(max_context or
                               os.environ.get("SEMIREAL_MAX_CONTEXT", "0"))
        self._raw = None
        self._cache: dict = {}

    # ── loading ─────────────────────────────────────────────────────────────
    def _load(self):
        if self._raw is not None:
            return self._raw
        root = _dopfn_root()
        cwd = os.getcwd()
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            os.chdir(root)
            from datasets import load_dataset as _dopfn_load
            self._raw = _dopfn_load(ds_name=self.ds_name)
        finally:
            os.chdir(cwd)
        return self._raw

    def _split(self, r: int):
        """Do-PFN's own split, so the protocol matches its published numbers."""
        ds = self._load()
        root = _dopfn_root()
        cwd = os.getcwd()
        try:
            os.chdir(root)
            # split_number is 1-based upstream; r is 0-based here.
            return ds.generate_valid_split(split_number=r + 1,
                                           n_splits=self.n_splits)
        finally:
            os.chdir(cwd)

    # ── the contract every harness expects ──────────────────────────────────
    def __len__(self) -> int:
        return self.n_tables

    def __getitem__(self, r: int):
        if r in self._cache:
            return self._cache[r]
        if not (0 <= r < self.n_tables):
            raise IndexError(f"{self.name}: split {r} out of range "
                             f"(n_tables={self.n_tables})")
        train_ds, test_ds = self._split(r)
        # Do-PFN's TabularDataset keeps T in column 0 of .x and the outcome in
        # .y; .cate on the test split is the per-unit true effect.
        xt, yt = _to_np(train_ds.x), _to_np(train_ds.y)
        xe = _to_np(test_ds.x)
        X_train = xt[:, 1:].astype(np.float32)
        t_train = xt[:, 0].astype(np.float32)
        y_train = np.asarray(yt, dtype=np.float32).reshape(-1)
        X_test = xe[:, 1:].astype(np.float32)
        true_cate = _to_np(test_ds.cate).reshape(-1).astype(np.float32)

        if self.max_context and X_train.shape[0] > self.max_context:
            X_train, t_train, y_train = self._subsample(
                X_train, t_train, y_train, r)

        sl = _CATESlice(
            X_train=X_train, t_train=t_train, y_train=y_train,
            X_test=X_test, true_cate=true_cate,
            source_realization=r,
            n_real_features=int(X_train.shape[1]),
            n_context=int(X_train.shape[0]),
        )
        # adj slice: these datasets have a known graph, but no harness consumes
        # it through this path, so None keeps the tuple shape without implying
        # a structure we are not actually supplying.
        out = (sl, None)
        self._cache[r] = out
        return out

    def _subsample(self, X, t, y, r: int):
        """Cap the context, stratified by arm so the treated fraction is kept.

        Seeded on the split index, so the same split always yields the same
        rows -- every model must see identical context for the comparison to
        mean anything.
        """
        rng = np.random.default_rng(20180621 + r)
        idx_t = np.flatnonzero(t > 0.5)
        idx_c = np.flatnonzero(t <= 0.5)
        frac = len(idx_t) / max(len(t), 1)
        n_t = int(round(self.max_context * frac))
        n_t = min(n_t, len(idx_t))
        n_c = min(self.max_context - n_t, len(idx_c))
        keep = np.concatenate([rng.choice(idx_t, n_t, replace=False),
                               rng.choice(idx_c, n_c, replace=False)])
        rng.shuffle(keep)
        return X[keep], t[keep], y[keep]


def get_dataset(name: str, **kw) -> DoPFNSemiRealDataset:
    return DoPFNSemiRealDataset(name, **kw)
