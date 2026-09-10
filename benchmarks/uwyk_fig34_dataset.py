"""Exposes the UWYK_Fig3_4 ComplexMech benchmark as an IHDPDataset-compatible
dataset, so every existing RealCause eval script consumes it unchanged.

Same trick as `scm_case_study_dataset.py`: the eval harnesses only ever touch
`ds.n_tables` and `ds[r] -> (cate_slice, adj_slice)` with
`X_train / t_train / y_train / X_test / true_cate`, so a shim is all it takes.

Dataset names
-------------
    CMECH_n<N>            all test queries
    CMECH_n<N>_nonzero    queries whose true effect is != 0
    CMECH_n<N>_zero       queries whose true effect is exactly 0

N in {5, 20, 30, 40, 50}. Regime is always `path_TY` (the other two UWYK
regimes have tau identically zero, so PEHE there is not informative).

Why the zero / nonzero split
----------------------------
12-27% of ComplexMech queries have an *exactly* zero treatment effect: tree and
saturating mechanisms often do not respond at all to flipping T. Those queries
ask "can the model report no effect?", which is a different skill from grading a
non-zero effect, and pooling them hides which one a method is good at.

The split is bimodal at the dataset level, not spread evenly across queries --
at n=5, 72 of 100 realizations contain no zero-effect query at all. So a subset
does not exist for every realization. This class exposes ONLY the realizations
where the requested subset is non-empty; `n_tables` reflects that, and
`source_realizations[i]` maps back to the generating realization index so a
result can be traced to its dataset.

Because the split is over disjoint query sets, the pooled PEHE is recoverable
exactly from the two parts:

    pehe_all^2 = (n0 * pehe_zero^2 + n1 * pehe_nonzero^2) / (n0 + n1)

so running the two subsets is sufficient; `CMECH_n<N>` is provided for checking.

Units
-----
Y and tau are on the generator's [-1, 1] target scale (UWYK's own
`target_negative_one_one_scaling`). PEHE is therefore in those units and is
comparable across methods and node counts, but NOT to RealCause PEHE numbers,
which are in each dataset's own outcome units.

Env
---
    UWYK_FIG34_DATA   root holding <prior>/<N>node/<regime>/hide_<h>/r*.npz
                      (default: <repo>/UWYK_Fig3_4/data)
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DATA = os.path.join(_REPO_ROOT, "UWYK_Fig3_4", "data")

NODE_COUNTS = (5, 10, 20, 30, 40, 50)
SUBSETS = ("all", "nonzero", "zero")

_NAME_RE = re.compile(r"^CMECH_n(\d+)(?:_(nonzero|zero|all))?$")


def dataset_names() -> tuple[str, ...]:
    """Every name this adapter answers to — for eval-script `choices` lists."""
    out = []
    for n in NODE_COUNTS:
        out.append(f"CMECH_n{n}")
        out.extend(f"CMECH_n{n}_{s}" for s in ("nonzero", "zero"))
    return tuple(out)


def parse_name(name: str) -> tuple[int, str] | None:
    """'CMECH_n20_zero' -> (20, 'zero'); None if not one of ours."""
    m = _NAME_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), (m.group(2) or "all")


@dataclass
class _CATESlice:
    X_train: np.ndarray
    t_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    true_cate: np.ndarray
    # Provenance, so a row in the results table can be traced to its dataset.
    source_realization: int = -1
    n_real_features: int = -1
    n_zero_queries: int = -1
    n_nonzero_queries: int = -1
    n_context: int = -1


class UWYKFig34Dataset:
    """IHDPDataset-shaped view over one (node count, subset) cell."""

    def __init__(self, name: str, data_root: str | None = None,
                 prior: str = "complexmech", regime: str = "path_TY",
                 hide: float = 0.0, max_context: int | None = None):
        """max_context: cap the training context to this many rows.

        Subsampling happens HERE rather than via each harness's own
        --eval-max-context flag, because those flags do not share a default
        (graph2d/uwyk1d use 1000, cpfn1d/cpfn2d use 0 = uncapped) and
        dopfn_native/dopfn_bb have no such flag at all. Doing it in one place
        guarantees every method sees the identical rows for a given
        (realization, cap), which is the whole point of the comparison.

        The draw is seeded from (realization, cap) only, so it is reproducible
        and does not depend on which method is running or in what order.
        Composes safely with the harness caps: those are >= this one, so they
        become no-ops.

        Reads UWYK_FIG34_MAX_CONTEXT when not passed explicitly.
        """
        parsed = parse_name(name)
        if parsed is None:
            raise ValueError(f"not a UWYK_Fig3_4 dataset name: {name!r}")
        self.name = name
        self.n_nodes, self.subset = parsed
        self.prior, self.regime, self.hide = prior, regime, hide

        if max_context is None:
            env = os.environ.get("UWYK_FIG34_MAX_CONTEXT", "").strip()
            max_context = int(env) if env else None
        self.max_context = max_context if (max_context or 0) > 0 else None

        root = data_root or os.environ.get("UWYK_FIG34_DATA", _DEFAULT_DATA)
        self.cell_dir = os.path.join(root, prior, f"{self.n_nodes}node",
                                     regime, f"hide_{hide}")
        paths = sorted(
            glob.glob(os.path.join(self.cell_dir, "r*.npz")),
            key=lambda p: int(os.path.basename(p)[1:-4]),
        )
        if not paths:
            raise FileNotFoundError(
                f"no realizations under {self.cell_dir}. Generate them first:\n"
                f"  python UWYK_Fig3_4/generate_pehe_benchmark.py --prior {prior} "
                f"--nodes {self.n_nodes} --regimes {regime} --hide-fractions {hide}"
            )

        # Keep only realizations where the requested subset has queries.
        self._paths: list[str] = []
        self.source_realizations: list[int] = []
        for p in paths:
            with np.load(p) as z:
                tau = z["true_cate"]
            if self._mask_for(tau).sum() > 0:
                self._paths.append(p)
                self.source_realizations.append(int(os.path.basename(p)[1:-4]))
        self.n_tables = len(self._paths)
        self.n_skipped = len(paths) - self.n_tables
        if self.n_tables == 0:
            raise ValueError(
                f"{name}: no realization has any '{self.subset}' query "
                f"(scanned {len(paths)} under {self.cell_dir})"
            )

    def _stratified_subsample(self, t_train: np.ndarray, r: int) -> np.ndarray:
        """Pick `max_context` rows, stratified by treatment arm.

        Sampling uniformly can empty an arm: at N=50 that happened in ~1 of 91
        realizations. Any harness taking a per-arm statistic then gets nan --
        uwyk1d's default T_ENCODING='target' feeds the model mean(Y|T=t), so a
        single collapsed realization poisons its whole run.

        Each arm keeps its share of the cap, rounded down, with a floor of
        min(2, arm size) so a per-arm mean and variance both stay defined. Any
        rounding remainder goes to the larger arm, which keeps the treated
        fraction close to the full-context value.

        Seeded from (realization, cap) only, so every method sees identical rows.
        """
        rng = np.random.default_rng([self.source_realizations[r],
                                     int(self.max_context)])
        cap = int(self.max_context)
        arms = [np.flatnonzero(t_train == v) for v in (0.0, 1.0)]
        if any(a.size == 0 for a in arms):
            # Single-arm to begin with — nothing to preserve; fall back to plain.
            idx = rng.choice(t_train.shape[0], cap, replace=False)
            idx.sort()
            return idx

        n_tot = sum(a.size for a in arms)
        take = [min(a.size, max(min(2, a.size), int(a.size * cap // n_tot)))
                for a in arms]
        # Hand any leftover to the larger arm, then trim if the floors overshot.
        leftover = cap - sum(take)
        order = sorted((0, 1), key=lambda i: -arms[i].size)
        for i in order:
            if leftover <= 0:
                break
            add = min(leftover, arms[i].size - take[i])
            take[i] += add
            leftover -= add
        while sum(take) > cap:
            i = max(order, key=lambda j: take[j])
            if take[i] <= min(2, arms[i].size):
                break
            take[i] -= 1

        idx = np.concatenate([rng.choice(a, k, replace=False)
                              for a, k in zip(arms, take)])
        idx.sort()
        return idx

    def _mask_for(self, tau: np.ndarray) -> np.ndarray:
        if self.subset == "zero":
            return tau == 0
        if self.subset == "nonzero":
            return tau != 0
        return np.ones_like(tau, dtype=bool)

    def __len__(self) -> int:
        return self.n_tables

    def __getitem__(self, r: int):
        with np.load(self._paths[r]) as z:
            n_real = int(z["n_real_features"])
            # Trailing columns are zero padding from the generator; keeping them
            # would feed the eval harnesses constant features and make their
            # standardisation divide by ~0.
            X_train = np.asarray(z["X_train"][:, :n_real], dtype=np.float32)
            X_test_full = np.asarray(z["X_test"][:, :n_real], dtype=np.float32)
            t_train = np.asarray(z["T_train"], dtype=np.float32).reshape(-1)
            y_train = np.asarray(z["Y_train"], dtype=np.float32).reshape(-1)
            tau = np.asarray(z["true_cate"], dtype=np.float32).reshape(-1)

        if self.max_context is not None and X_train.shape[0] > self.max_context:
            idx = self._stratified_subsample(t_train, r)
            X_train, t_train, y_train = X_train[idx], t_train[idx], y_train[idx]

        mask = self._mask_for(tau)
        sl = _CATESlice(
            X_train=X_train,
            t_train=t_train,
            y_train=y_train,
            X_test=X_test_full[mask],
            true_cate=tau[mask],
            source_realization=self.source_realizations[r],
            n_real_features=n_real,
            n_zero_queries=int((tau == 0).sum()),
            n_nonzero_queries=int((tau != 0).sum()),
            n_context=int(X_train.shape[0]),
        )
        return sl, None


def get_dataset(name: str, **kw) -> UWYKFig34Dataset:
    return UWYKFig34Dataset(name, **kw)
