"""
Streaming Dataset for paired potential outcomes (Y_do0, Y_do1) from
fresh SCMs at every step.

Refactor of ``generate_paired_samples.py``:
  - Same SCM config, same binarisation, same paired-noise propagation, same
    preprocessing (X standardised, Y scaled to [-1,1], NO clamp).
  - Each ``__getitem__(idx)`` produces a fresh task with `idx` as the
    deterministic seed offset. Designed for ``DataLoader(num_workers=K)`` —
    each worker maintains its own ``SCMSampler``.

Required external code:
  Requires UWYK's ``src/`` on the Python path so we can import:
    - ``priors.causal_prior.scm.SCMSampler``
    - ``priors.causal_prior.mechanisms.BinarizingMechanism``
  Set ``UWYK_SRC`` env var to the path (default: ``/tmp/g4cfm/src``).

Returned tensors per item (no batch dim — DataLoader stacks):
    X_obs   : (n_train, max_features) float
    T_obs   : (n_train, 1)            float in {0,1}
    Y_obs   : (n_train, 1)            float in [-1,1]
    X_intv  : (n_test,  max_features) float
    Y_do0   : (n_test,  1)            float
    Y_do1   : (n_test,  1)            float
    anc_matrix : (max_features+2, max_features+2) float in {-1,0,1}
"""
from __future__ import annotations

import os
import sys
import time
import traceback as _tb
from copy import deepcopy
from typing import Any

import torch
from torch.utils.data import Dataset


# --- Work around a PyTorch refcount bug (confirmed torch 2.6.0+cu124) ----------
# copy.deepcopy / copy.copy of a torch.Generator over-decrements the refcount of
# the None singleton by exactly 1 per call. Each sampled SCM holds one
# torch.Generator per mechanism (tens of them), so `deepcopy(scm)` in
# _generate_one leaked ~30-40 None-refs per task. After a few thousand tasks a
# DataLoader worker drove None's refcount to zero and the process hard-aborted
# with `Fatal Python error: none_dealloc` (killing training ~step 500).
#
# copy._deepcopy_dispatch is consulted *before* any __deepcopy__ hook, and works
# on torch.Generator (a C type that cannot take a monkeypatched __deepcopy__), so
# registering a state-based clone here fixes every Generator anywhere in the SCM
# object graph, in every process that imports this module. A fresh Generator
# seeded via set_state(get_state()) is verified leak-free (None delta 0/call).
import copy as _copy


def _deepcopy_generator(g: torch.Generator, memo):
    ng = torch.Generator(device=g.device)
    ng.set_state(g.get_state())
    memo[id(g)] = ng
    return ng


if torch.Generator not in _copy._deepcopy_dispatch:
    _copy._deepcopy_dispatch[torch.Generator] = _deepcopy_generator


# --- UWYK source path (set UWYK_SRC env var to override) ----------------------
_UWYK_SRC = os.environ.get("UWYK_SRC", "/tmp/g4cfm/src")


# --- Diagnostics (env-gated) --------------------------------------------------
# These make the data pipeline self-reporting so we can pin down (a) which exact
# sample/seed a worker dies on (the `none_dealloc` C abort) and (b) whether that
# sample is also numerically broken. All cheap relative to ~0.2s/sample SCM gen.
#
#   DATA_VALIDATE=1    check each returned tensor for NaN/Inf + target blow-ups
#   DATA_BREADCRUMB=1  write last (idx,seed) per worker to logs/ so a hard C
#                      abort (which kills the process with no Python traceback)
#                      still tells us the culprit sample
#   DATA_VERBOSE=0     log every sample as it's generated (noisy; off by default)
#   DATA_TARGET_ABSMAX=10.0  |target| above this is flagged as a scaling blow-up
#   DATA_REFCOUNT=0    track sys.getrefcount(None) across each __getitem__ and its
#                      sub-phases. The `none_dealloc` abort is a C-extension
#                      over-decref of the None singleton that accumulates over
#                      many calls; this is the only diagnostic that localizes it.
#                      Logs a [REFCOUNT] line for any phase with a nonzero delta.
_DATA_VALIDATE   = os.environ.get("DATA_VALIDATE", "1") == "1"
_DATA_BREADCRUMB = os.environ.get("DATA_BREADCRUMB", "1") == "1"
_DATA_VERBOSE    = os.environ.get("DATA_VERBOSE", "0") == "1"
_DATA_REFCOUNT   = os.environ.get("DATA_REFCOUNT", "0") == "1"
_TARGET_ABSMAX   = float(os.environ.get("DATA_TARGET_ABSMAX", "10.0"))
_REPO_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BREADCRUMB_DIR  = os.environ.get("DATA_BREADCRUMB_DIR", os.path.join(_REPO_ROOT, "logs"))


def _breadcrumb(worker_id: int, idx: int, seed: int, phase: str) -> None:
    """Overwrite a tiny per-worker file with the sample currently in flight.

    After a fatal C abort, the breadcrumb whose phase is still ENTER (no
    matching DONE) is the sample that killed the worker. flush() is enough to
    survive an abort — the page stays in the OS cache.
    """
    path = os.path.join(_BREADCRUMB_DIR, f"databreadcrumb_w{worker_id}_pid{os.getpid()}.txt")
    try:
        with open(path, "w") as f:
            f.write(f"{phase} worker={worker_id} idx={idx} seed={seed} t={time.time():.3f}\n")
            f.flush()
    except OSError:
        pass


class _NoneRefcountTracker:
    """Localizes the `none_dealloc` C abort by watching ``sys.getrefcount(None)``.

    The abort is caused by a C extension over-decrementing the None singleton's
    refcount; it accumulates silently over thousands of calls until None hits
    zero and the process aborts. A normal sub-step nets a delta of 0 (it INCREFs
    and DECREFs None the same number of times). A buggy sub-step nets a *negative*
    delta on at least some calls — that's the leak. We measure the delta across
    each named phase of ``_generate_one`` and emit a [REFCOUNT] line whenever a
    phase nets nonzero, plus a running cumulative total per worker so a slow
    monotonic drift is visible even if no single call looks dramatic.

    sys.getrefcount itself adds a temporary +1 (its own argument ref), but that
    is constant across both reads so it cancels in the delta — deltas are exact.
    """

    def __init__(self, worker_id: int, idx: int, seed: int):
        self.worker_id = worker_id
        self.idx = idx
        self.seed = seed
        self._last = sys.getrefcount(None)
        self._start = self._last

    def mark(self, phase: str) -> None:
        cur = sys.getrefcount(None)
        delta = cur - self._last
        self._last = cur
        if delta != 0:
            sys.stderr.write(
                f"[REFCOUNT] worker={self.worker_id} idx={self.idx} seed={self.seed} "
                f"phase={phase}: None refcount delta={delta:+d} (now={cur})\n"
            )
            sys.stderr.flush()

    def finish(self) -> None:
        net = sys.getrefcount(None) - self._start
        if net != 0:
            sys.stderr.write(
                f"[REFCOUNT] worker={self.worker_id} idx={self.idx} seed={self.seed} "
                f"NET None refcount delta={net:+d} over this sample — "
                f"a negative drift here is the source of the eventual none_dealloc abort\n"
            )
            sys.stderr.flush()


class _NullTracker:
    """No-op tracker so the hot path stays branch-light when DATA_REFCOUNT=0."""

    def mark(self, phase: str) -> None:  # noqa: D401
        pass

    def finish(self) -> None:
        pass


def sample_metrics(out: dict) -> dict:
    """Aggregate finiteness/magnitude metrics for one generated sample."""
    n_nonfinite = 0
    target_absmax = 0.0
    per_field: dict[str, str] = {}
    for k, v in out.items():
        if not torch.is_tensor(v) or v.numel() == 0:
            continue
        tf = v.detach().float()
        nb = int((~torch.isfinite(tf)).sum())
        n_nonfinite += nb
        fin = tf[torch.isfinite(tf)]
        amax = float(fin.abs().max()) if fin.numel() else float("inf")
        if k in ("Y_obs", "Y_do0", "Y_do1"):
            target_absmax = max(target_absmax, amax)
        if nb or amax > _TARGET_ABSMAX:
            per_field[k] = f"absmax={amax:.3g} nonfinite={nb}/{tf.numel()}"
    return {"n_nonfinite": n_nonfinite, "target_absmax": target_absmax, "per_field": per_field}


def _contains_nan_or_inf(sample: dict) -> bool:
    """True if any tensor in ``sample`` has NaN or Inf — mirrors UWYK's
    InterventionalDataset._contains_nan (we additionally reject Inf since our
    BarDistribution2D NLL diverges on either)."""
    for v in sample.values():
        if torch.is_tensor(v) and not torch.isfinite(v).all():
            return True
    return False


def _sample_passes_thresholds(
    sample: dict,
    min_var: float,
    min_unique_frac: float,
) -> tuple[bool, str]:
    """UWYK-style post-hoc filter: reject samples where any of Y_obs / Y_do0 /
    Y_do1 collapse to near-constant. Mirrors UWYK's min_target_variance and
    min_unique_target_fraction checks (InterventionalDataset.py lines 1067–1114),
    applied across all three Y arms instead of (Y_obs, Y_intv)."""
    for k in ("Y_obs", "Y_do0", "Y_do1"):
        v = sample.get(k)
        if v is None or not torch.is_tensor(v):
            continue
        flat = v.reshape(-1).float()
        if min_var is not None and float(flat.var()) < min_var:
            return False, f"{k} var={float(flat.var()):.3g} < {min_var}"
        if min_unique_frac is not None and flat.numel() > 0:
            frac = float(torch.unique(flat).numel()) / flat.numel()
            if frac < min_unique_frac:
                return False, f"{k} unique-frac={frac:.3g} < {min_unique_frac}"
    return True, ""


def _validate_sample(out: dict, idx: int, seed: int, worker_id: int) -> list[str]:
    """Log [DATA-WARN] for any non-finite values or target scaling blow-ups."""
    m = sample_metrics(out)
    problems: list[str] = []
    if m["n_nonfinite"]:
        problems.append(f"{m['n_nonfinite']} non-finite values")
    if m["target_absmax"] > _TARGET_ABSMAX:
        problems.append(f"target |max|={m['target_absmax']:.3g} (≫1 — scaling blow-up)")
    if problems:
        fields = "  ".join(f"{k}[{v}]" for k, v in m["per_field"].items())
        sys.stderr.write(
            f"[DATA-WARN] worker={worker_id} idx={idx} seed={seed}: "
            f"{'; '.join(problems)}  | {fields}\n"
        )
        sys.stderr.flush()
    return problems


# --- Default SCM config (verbatim from generate_paired_samples.py) ------------

DEFAULT_SCM_CONFIG = {
    "num_nodes": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 2, "high": 51}},
    "graph_edge_prob": {"distribution": "beta", "distribution_parameters": {"alpha": 2.0, "beta": 3.0}},
    "graph_seed": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 0, "high": 100000}},
    "xgboost_prob": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [0.0, 0.1, 0.2, 0.3], "probabilities": [1.0, 0.0, 0.0, 0.0]},
    },
    "mechanism_seed": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 0, "high": 100000}},
    "mlp_nonlins": {"value": "tabicl"},
    "mlp_num_hidden_layers": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [0, 1, 2, 3], "probabilities": [0.875, 0.1, 0.025, 0.01]},
    },
    "mlp_hidden_dim": {
        "distribution": "categorical",
        "distribution_parameters": {
            "choices": [1, 2, 4, 6, 8, 10, 12, 14, 16, 32],
            "probabilities": [0.7, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01],
        },
    },
    "mlp_activation_mode": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": ["pre", "post", "mixed_in"], "probabilities": [0.3, 0.3, 0.3]},
    },
    "mlp_use_batch_norm": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [True, False], "probabilities": [0.5, 0.5]},
    },
    "mlp_node_shape": {"value": (1,)},
    "xgb_node_shape": {"value": (1,)},
    "xgb_num_hidden_layers": {"value": 0},
    "xgb_hidden_dim": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [0, 16, 32, 64]},
    },
    "xgb_activation_mode": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": ["pre", "post", "mixed_in"], "probabilities": [0.33, 0.33, 0.34]},
    },
    "xgb_use_batch_norm": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [True, False], "probabilities": [0.5, 0.5]},
    },
    "xgb_n_training_samples": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [10, 50, 100, 200, 500], "probabilities": [0.1, 0.1, 0.3, 0.4, 0.5]},
    },
    "xgb_add_noise": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [True, False], "probabilities": [0.5, 0.5]},
    },
    "random_additive_std": {"value": True},
    "exo_std_distribution": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": ["gamma", "pareto"], "probabilities": [1.0, 0.0]},
    },
    "endo_std_distribution": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": ["gamma", "pareto"], "probabilities": [1.0, 0.0]},
    },
    "exo_std_mean": {"distribution": "lognormal", "distribution_parameters": {"mean": 1.0, "std": 1.0}},
    "exo_std_std": {"distribution": "uniform", "distribution_parameters": {"low": 0.1, "high": 0.4}},
    "endo_std_mean": {"distribution": "lognormal", "distribution_parameters": {"mean": -3.0, "std": 0.6}},
    "endo_std_std": {"distribution": "uniform", "distribution_parameters": {"low": 0.0, "high": 0.5}},
    "endo_p_zero": {"value": 0.0},
    "noise_mixture_proportions": {"value": [0.33, 0.33, 0.34]},
    "use_exogenous_mechanisms": {"value": True},
    "mechanism_generator_seed": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 0, "high": 100000}},
}


# --- Helpers (verbatim from generate_paired_samples.py) ------------------------

def _standardize(X_train, X_test=None, eps=1e-8):
    mu = X_train.mean(0, keepdim=True)
    std = X_train.std(0, keepdim=True).clamp(min=eps)
    X_train_s = (X_train - mu) / std
    if X_test is None:
        return X_train_s
    return X_train_s, (X_test - mu) / std


def _scale_to_neg1_pos1(Y_train, Y_test0, Y_test1, eps=1e-8):
    lo = Y_train.min()
    hi = Y_train.max()
    rng = (hi - lo).clamp(min=eps)
    def _scale(y):
        return 2.0 * (y - lo) / rng - 1.0
    return _scale(Y_train), _scale(Y_test0), _scale(Y_test1)


def _clip_outliers(t, q=0.99):
    lo = torch.quantile(t.reshape(-1).float(), 1.0 - q)
    hi = torch.quantile(t.reshape(-1).float(), q)
    return t.clamp(lo, hi)


def _propagate_paired(obs_scm, intv_scm, treatment_node, n_test, t0_value, t1_value):
    """
    Propagate intv_scm for both do(T=t0_value) and do(T=t1_value) in one
    doubled batch.

    t0_value / t1_value are the SCM's two binary treatment levels chosen by
    BinarizingMechanism.from_observational_data — sampled from observed T
    quantiles so downstream MLPs see in-distribution inputs. The caller
    remaps T_obs to model-facing {0,1} after propagation.

    Matches UWYK ``sample_exogenous`` / ``sample_endogenous`` memory layout:
    the dict entries in ``_fixed_exogenous`` / ``_fixed_endogenous`` are
    **views** into the corresponding ``_fixed_exogenous_vec`` /
    ``_fixed_endogenous_vec`` buffers (see SCM.py for the pattern). The
    earlier version of this function assigned standalone tensors to the
    dicts; over many calls that produced a fatal `none_dealloc`
    refcount error in a C extension.
    """
    B2 = 2 * n_test

    # ── Exogenous: allocate buffer first, fill slices, then build views ──
    total_exo = intv_scm._total_exo_dim
    fixed_exo_vec = torch.zeros(B2, total_exo, dtype=torch.float32)

    for v in intv_scm._exo_order:
        s, e = intv_scm._exo_slices[v]
        d = e - s
        if v == treatment_node:
            # First half do(T=t0_value), second half do(T=t1_value).
            t_vals = torch.cat([
                torch.full((n_test,), t0_value),
                torch.full((n_test,), t1_value),
            ])
            fixed_exo_vec[:, s:e] = t_vals.reshape(B2, d)
        else:
            # Share noise with obs by tiling each obs sample twice
            old = obs_scm._fixed_exogenous[v]
            tiled = old.repeat(2) if old.dim() == 1 else old.repeat(2, 1)
            fixed_exo_vec[:, s:e] = tiled.reshape(B2, d)

    # Build dict entries as VIEWS into the buffer — same pattern as UWYK's
    # SCM.sample_exogenous so the library's invariants stay intact.
    fixed_exo: dict[str, torch.Tensor] = {}
    for v in intv_scm._exo_order:
        s, e = intv_scm._exo_slices[v]
        flat = fixed_exo_vec[:, s:e]
        if intv_scm.use_exogenous_mechanisms:
            fixed_exo[v] = flat.reshape(B2)
        else:
            shp = intv_scm._node_shape.get(v, ())
            fixed_exo[v] = flat.reshape(B2, *shp) if shp else flat.reshape(B2)

    intv_scm._fixed_exogenous_vec = fixed_exo_vec
    intv_scm._fixed_exogenous = fixed_exo
    intv_scm._fixed_batch = B2

    total_endo = intv_scm._total_endo_dim
    if total_endo == 0:
        fixed_endo_vec = torch.empty(B2, 0)
    else:
        fixed_endo_vec = torch.zeros(B2, total_endo, dtype=torch.float32)
        for v in intv_scm._endo_order:
            s, e = intv_scm._endo_slices[v]
            d = e - s
            old = obs_scm._fixed_endogenous.get(v) if obs_scm._fixed_endogenous else None
            if old is not None:
                old_flat = old.reshape(n_test, d)
                fixed_endo_vec[:, s:e] = old_flat.repeat(2, 1)

    fixed_endo: dict[str, torch.Tensor] = {}
    for v in intv_scm._endo_order:
        s, e = intv_scm._endo_slices[v]
        flat = fixed_endo_vec[:, s:e]
        shp = intv_scm._node_shape.get(v, ())
        fixed_endo[v] = flat.reshape(B2, *shp) if shp else flat.reshape(B2)

    intv_scm._fixed_endogenous_vec = fixed_endo_vec
    intv_scm._fixed_endogenous = fixed_endo

    res_full = intv_scm.propagate(B2)
    res0 = {v: t[:n_test] for v, t in res_full.items()}
    res1 = {v: t[n_test:] for v, t in res_full.items()}
    return res0, res1


def _pad_x(x: torch.Tensor, max_f: int) -> torch.Tensor:
    if x.shape[1] >= max_f:
        return x[:, :max_f]
    pad = torch.zeros(x.shape[0], max_f - x.shape[1])
    return torch.cat([x, pad], dim=1)


# --- Dataset class ------------------------------------------------------------

class PairedInterventionalDataset(Dataset):
    """
    Streaming dataset; each __getitem__(idx) samples a fresh SCM task.

    Args:
        scm_config: SCM prior config (defaults to the same one used in
            ``generate_paired_samples.py``).
        n_train: observational context size per task.
        n_test:  query (paired-outcome) count per task.
        max_features: pad/truncate X to this width.
        seed_base: per-process seed offset (the actual per-sample seed is
            ``seed_base + idx + 997·worker_id``).
        max_outer_attempts: retry count when SCM sampling produces a
            degenerate task (no T→Y path, binarisation failure, near-constant
            Y, etc.). Generally 50 is enough.
        outlier_q: Y clipping quantile (0.99).
        epsilon: numerical floor for normalisation.
    """

    def __init__(
        self,
        scm_config: dict | None = None,
        n_train: int = 1000,
        n_test: int = 500,
        max_features: int = 50,
        seed_base: int = 42,
        max_outer_attempts: int = 50,
        outlier_q: float = 0.99,
        epsilon: float = 1e-8,
        infinite_len: int = 10**9,
        # UWYK post-hoc reject thresholds (mirror their defaults).
        # The retry loop in __getitem__ resamples the SCM with an offset seed
        # whenever a sample fails any of these checks.
        max_nan_retries: int = 10,
        min_target_variance: float | None = 1e-2,
        min_unique_target_fraction: float | None = 0.2,
    ):
        super().__init__()
        if _UWYK_SRC not in sys.path:
            sys.path.insert(0, _UWYK_SRC)
        # Lazy imports — only happen when UWYK source is on the path
        from priors.causal_prior.scm.SCMSampler import SCMSampler          # noqa: F401
        from priors.causal_prior.mechanisms.BinarizingMechanism import BinarizingMechanism  # noqa: F401
        self._SCMSampler = SCMSampler
        self._BinarizingMechanism = BinarizingMechanism

        self.scm_config = scm_config if scm_config is not None else DEFAULT_SCM_CONFIG
        self.n_train = n_train
        self.n_test = n_test
        self.max_features = max_features
        self.seed_base = seed_base
        self.max_outer_attempts = max_outer_attempts
        self.outlier_q = outlier_q
        self.epsilon = epsilon
        self.infinite_len = infinite_len
        self.max_nan_retries = max_nan_retries
        self.min_target_variance = min_target_variance
        self.min_unique_target_fraction = min_unique_target_fraction

        # One sampler per process. DataLoader workers each get their own via
        # ``worker_init_fn`` (set in :func:`make_streaming_loader`).
        self._sampler = self._SCMSampler(self.scm_config, seed=seed_base * 31 + 17)

    def __len__(self) -> int:
        return self.infinite_len

    def __getitem__(self, idx: int) -> dict[str, Any]:
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else -1
        seed = self.seed_base + idx

        if _DATA_BREADCRUMB:
            _breadcrumb(worker_id, idx, seed, "ENTER")
        if _DATA_VERBOSE:
            sys.stderr.write(f"[DATA] worker={worker_id} idx={idx} seed={seed} generating…\n")
            sys.stderr.flush()

        tracker = _NoneRefcountTracker(worker_id, idx, seed) if _DATA_REFCOUNT else _NullTracker()

        # Retry-with-offset-seed loop, mirroring UWYK's InterventionalDataset
        # __getitem__ (lines 551–568). On each retry the SCM is resampled via a
        # 1_000_000-stride seed offset, so we get a genuinely different SCM each
        # time rather than the same degenerate one.
        #
        # UWYK's fallback on retry exhaustion (their lines 1124–1136) is "use
        # last sample" — they accept the imperfect sample rather than crash.
        # We follow the same policy, with one safety twist: we only accept the
        # last *NaN/Inf-free* sample, since NaN would explode the model loss.
        out = None
        out_last_finite = None
        last_reject_reason = ""
        for attempt in range(self.max_nan_retries):
            try:
                out = self._generate_one(idx, tracker, attempt=attempt)
            except Exception:
                sys.stderr.write(
                    f"[DATA-ERR] worker={worker_id} idx={idx} seed={seed} attempt={attempt} raised:\n{_tb.format_exc()}\n"
                )
                sys.stderr.flush()
                raise

            if _contains_nan_or_inf(out):
                sys.stderr.write(
                    f"[DATA-WARN] worker={worker_id} idx={idx}: NaN/Inf in sample "
                    f"(attempt {attempt + 1}/{self.max_nan_retries}). Resampling.\n"
                )
                sys.stderr.flush()
                continue

            # Remember the last finite (NaN-free) sample — this is what we'll
            # fall back to if all attempts fail thresholds.
            out_last_finite = out

            ok, reason = _sample_passes_thresholds(
                out,
                self.min_target_variance,
                self.min_unique_target_fraction,
            )
            if not ok:
                last_reject_reason = reason
                sys.stderr.write(
                    f"[DATA-WARN] worker={worker_id} idx={idx}: threshold reject "
                    f"(attempt {attempt + 1}/{self.max_nan_retries}): {reason}. Resampling.\n"
                )
                sys.stderr.flush()
                continue

            break
        else:
            # All attempts exhausted — fall back to UWYK's "use last sample"
            # behaviour for thresholds, but refuse to ship a NaN/Inf sample.
            if out_last_finite is not None:
                sys.stderr.write(
                    f"[DATA-WARN] worker={worker_id} idx={idx}: max retries "
                    f"({self.max_nan_retries}) exhausted; thresholds never satisfied "
                    f"(last={last_reject_reason}) — using last NaN/Inf-free sample.\n"
                )
                sys.stderr.flush()
                out = out_last_finite
            else:
                raise RuntimeError(
                    f"PairedInterventionalDataset: idx={idx} produced NaN/Inf in all "
                    f"{self.max_nan_retries} attempts (no finite fallback available)"
                )

        if _DATA_VALIDATE:
            _validate_sample(out, idx, seed, worker_id)
        if _DATA_BREADCRUMB:
            _breadcrumb(worker_id, idx, seed, "DONE")
        return out

    # ----- internal -----------------------------------------------------------

    def _generate_one(self, idx: int, tracker=None, attempt: int = 0) -> dict[str, Any]:
        if tracker is None:
            tracker = _NullTracker()
        # UWYK uses a 1_000_000 stride between attempts so each retry draws a
        # genuinely different SCM (InterventionalDataset.py line 574).
        seed = self.seed_base + idx + attempt * 1_000_000
        torch.manual_seed(seed)

        scm = None
        treatment_node = None
        target_node = None
        feature_nodes: list = []
        obs = None
        T_obs_raw: torch.Tensor | None = None
        Y_obs_raw: torch.Tensor | None = None
        X_obs_raw: torch.Tensor | None = None

        for outer_attempt in range(self.max_outer_attempts):
            attempt_seed = seed + outer_attempt * 997
            scm = self._sampler.sample(seed=attempt_seed)

            all_nodes = sorted(scm.dag.nodes())
            n_nodes = len(all_nodes)
            if n_nodes < 3:
                continue

            rng = torch.Generator()
            rng.manual_seed(attempt_seed)
            found_pair = False
            for _ in range(30):
                t_idx = torch.randint(0, n_nodes, (1,), generator=rng).item()
                treatment_node = all_nodes[t_idx]
                available = [n for n in all_nodes if n != treatment_node]
                y_idx = torch.randint(0, len(available), (1,), generator=rng).item()
                target_node = available[y_idx]
                if scm.exists_treatment_outcome_path(treatment_node, target_node):
                    found_pair = True
                    break
            if not found_pair:
                continue

            feature_nodes = [n for n in all_nodes if n != treatment_node and n != target_node]
            original_mech = scm.mechanisms[treatment_node]

            # Binarise treatment until we get both classes
            binarised_ok = False
            for bin_try in range(10):
                scm.sample_exogenous(self.n_train)
                scm._fixed_endogenous_vec = None
                scm.sample_endogenous(self.n_train)
                obs_cont = scm.propagate(self.n_train)
                t_cont = obs_cont[treatment_node].reshape(-1).float()
                # UWYK's factory samples threshold + t0/t1 from observed T
                # quantiles so downstream MLPs see in-distribution T values.
                try:
                    bin_mech = self._BinarizingMechanism.from_observational_data(
                        wrapped_mechanism=original_mech, obs_values=t_cont,
                    )
                except ValueError:
                    # Factory raises when sampled t0 == t1 (SCM emitted
                    # constant T). Retry; outer loop rejects if all fail.
                    continue
                scm.mechanisms[treatment_node] = bin_mech
                t0_value = bin_mech.t0
                t1_value = bin_mech.t1

                scm.sample_exogenous(self.n_train)
                scm._fixed_endogenous_vec = None
                scm.sample_endogenous(self.n_train)
                obs = scm.propagate(self.n_train)

                T_obs_raw = obs[treatment_node].reshape(-1, 1).float()
                if T_obs_raw.unique().numel() >= 2:
                    binarised_ok = True
                    break
                scm.mechanisms[treatment_node] = original_mech

            if not binarised_ok:
                continue

            Y_obs_raw = obs[target_node].reshape(-1, 1).float()
            X_obs_raw = (
                torch.cat([obs[n].reshape(self.n_train, -1).float() for n in feature_nodes], dim=1)
                if feature_nodes
                else torch.zeros(self.n_train, 0)
            )

            if Y_obs_raw.var() < 1e-3:
                continue
            if torch.unique(Y_obs_raw).numel() < max(5, int(0.1 * self.n_train)):
                continue
            break
        else:
            # Couldn't find a usable SCM in max_outer_attempts tries
            raise RuntimeError(f"PairedInterventionalDataset: gave up after {self.max_outer_attempts} attempts at idx={idx}")

        tracker.mark("scm_search")  # sampler.sample + binarise + obs propagate loop

        # Interventional propagation with the paired-noise trick
        intv_scm = deepcopy(scm)
        intv_scm.intervene(treatment_node)
        tracker.mark("deepcopy_intervene")

        scm.sample_exogenous(self.n_test)
        scm._fixed_endogenous_vec = None
        scm.sample_endogenous(self.n_test)
        obs_test = scm.propagate(self.n_test)
        tracker.mark("obs_test_propagate")

        res0, res1 = _propagate_paired(scm, intv_scm, treatment_node, self.n_test, t0_value, t1_value)
        tracker.mark("propagate_paired")
        Y_do0_raw = res0[target_node].reshape(-1, 1).float()
        Y_do1_raw = res1[target_node].reshape(-1, 1).float()

        X_intv_raw = (
            torch.cat([obs_test[n].reshape(self.n_test, -1).float() for n in feature_nodes], dim=1)
            if feature_nodes
            else torch.zeros(self.n_test, 0)
        )

        # --- preprocess: UWYK-style joint [-1,1] affine over (Y_obs, Y_do0, Y_do1)
        # Mirrors UWYK's process_from_splits → _fit_apply_target_pipeline: the
        # affine is computed from the *concatenation* of all Y arms, so any
        # extreme value in Y_do0/Y_do1 pulls the affine wide enough to contain
        # the whole sample inside [-1,1]. Replaces the prior "clip Y_obs at q99
        # then scale with Y_obs-only min/max" path, which left Y_do unbounded.
        y_all = torch.cat([
            Y_obs_raw.reshape(-1),
            Y_do0_raw.reshape(-1),
            Y_do1_raw.reshape(-1),
        ])
        ymin = y_all.min()
        ymax = y_all.max()
        rng = (ymax - ymin).clamp(min=self.epsilon)
        Y_obs = 2.0 * (Y_obs_raw - ymin) / rng - 1.0
        Y_do0 = 2.0 * (Y_do0_raw - ymin) / rng - 1.0
        Y_do1 = 2.0 * (Y_do1_raw - ymin) / rng - 1.0

        if X_obs_raw.shape[1] > 0:
            X_obs_s, X_intv_s = _standardize(X_obs_raw, X_intv_raw, eps=self.epsilon)
        else:
            X_obs_s = X_obs_raw
            X_intv_s = X_intv_raw

        X_obs = _pad_x(X_obs_s, self.max_features)
        X_intv = _pad_x(X_intv_s, self.max_features)

        # T_obs_raw holds the SCM's two binary values (t0_value, t1_value sampled
        # from observed T quantiles, NOT necessarily {0,1}). Remap to model-facing
        # {0,1} using the midpoint for float-safe comparison.
        T_obs = (T_obs_raw > (t0_value + t1_value) / 2.0).float()

        # --- ancestor matrix -------------------------------------------------
        try:
            from utils.graph_utils import (  # type: ignore
                adjacency_to_ancestor_matrix,
                propagate_ancestor_knowledge,
            )
            ordered_nodes = [treatment_node, target_node] + feature_nodes
            adj_raw = scm.get_adjacency_matrix(node_order=ordered_nodes)
            anc_raw = adjacency_to_ancestor_matrix(adj_raw)
            anc = 2.0 * anc_raw.float() - 1.0

            hide_frac = torch.rand(1).item()
            L = len(feature_nodes)
            real_n = 2 + L
            rand_mat = torch.rand(real_n, real_n)
            hide_mask = rand_mat < hide_frac
            anc[:real_n, :real_n][hide_mask] = 0.0
            anc = propagate_ancestor_knowledge(anc)

            target_size = self.max_features + 2
            if anc.shape[0] < target_size:
                padded = torch.full((target_size, target_size), -1.0)
                padded[:anc.shape[0], :anc.shape[1]] = anc
                anc = padded
        except Exception:
            anc = torch.full((self.max_features + 2, self.max_features + 2), 0.0)
        tracker.mark("ancestor_matrix")

        tracker.finish()
        return {
            "X_obs":   X_obs,
            "T_obs":   T_obs,
            "Y_obs":   Y_obs,
            "X_intv":  X_intv,
            "Y_do0":   Y_do0,
            "Y_do1":   Y_do1,
            "anc_matrix": anc,
        }


# --- DataLoader factory with worker-aware seeding ------------------------------

def _worker_init_fn(worker_id: int):
    """Reseed each worker's sampler so we don't repeat the same SCMs."""
    # Enable faulthandler in each forked worker process so a C-level crash
    # here gets a real stack trace, not just the Python frames. fork
    # inherits state from the parent but per-process handlers must still be
    # registered.
    import faulthandler
    faulthandler.enable()

    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    if isinstance(ds, PairedInterventionalDataset):
        # New sampler with a worker-distinct seed
        ds._sampler = ds._SCMSampler(ds.scm_config, seed=(ds.seed_base + worker_id) * 31 + 17)


def make_streaming_loader(
    batch_size: int = 1,
    num_workers: int = 4,
    scm_config: dict | None = None,
    **dataset_kwargs,
) -> torch.utils.data.DataLoader:
    """
    Convenience constructor. Returns a DataLoader yielding dicts of stacked
    tensors with leading batch dim ``batch_size``.
    """
    ds = PairedInterventionalDataset(scm_config=scm_config, **dataset_kwargs)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
