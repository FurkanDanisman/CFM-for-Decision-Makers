"""PEHE-able eval sets from UWYK's own Fig-3 (LinGaus) and Fig-4 (ComplexMech) priors.

Goal: keep UWYK's Figure 3 / Figure 4 setup intact and change only what is
strictly required to make PEHE well defined. This module loads UWYK's own
benchmark YAMLs, hands ``scm_config`` untouched to their ``SCMSampler``, and
runs their own ``BasicProcessing`` for every preprocessing step.

Why Fig 3 / Fig 4 cannot yield PEHE as shipped
----------------------------------------------
Verified against UWYK @ c27fba6, not inferred from the paper:

1. ``binarize_treatment_prob`` defaults to 0.0 and is absent from all 24 LinGaus
   configs and every ComplexMechIDK config, so T is continuous and do-values come
   from ``ScaledUniformResamplingDist`` -- never do(0)/do(1).
2. The interventional pass resamples the noise
   (``InterventionalDataset.py``: ``sample_exogenous`` / ``sample_endogenous``
   before ``propagate``), so there is no matched (y0_i, y1_i) per unit.
3. ``test_feature_mask_fraction: 1.0`` in every one of those configs, and
   ``BasicProcessing._apply_test_feature_masking`` zeroes *every* non-zero test
   feature column. In Fig 3 / Fig 4 the model therefore sees X_intv == 0 at test
   time and predicts p(y | do(t), D) with no covariate to be heterogeneous in.

The five deviations, and why each is forced
-------------------------------------------
D1. ``binarize_treatment_prob``: 0.0 -> 1.0.
    The sanctioned change. This is UWYK's own switch and their own
    ``BinarizingMechanism.from_observational_data``; t0/t1 are drawn from
    observed T quantiles so downstream mechanisms see in-distribution values.

D2. Exogenous *and* endogenous noise shared across the do(t0) and do(t1) passes.
    Without this there are no matched potential outcomes and PEHE is undefined.
    Implemented by the training pipeline's proven ``_propagate_paired``.

D3. X is read off the *observational* test pass and tiled across both arms.
    UWYK reads X from the intervened pass. That cannot survive paired sampling:
    a descendant of T takes different values under do(t0) and do(t1), so there
    would be two different X matrices and no single x to condition on. Every
    non-descendant of T is bit-identical across all three passes, so this is a
    no-op for SCMs with no descendant features -- see ``n_descendant_features``,
    recorded per realization, to restrict to that exactly-faithful subset.

D4. ``test_feature_mask_fraction`` defaults to 0.0 here, vs 1.0 in UWYK's configs.
    At 1.0 every test covariate is zeroed, so the best possible tau-hat(x) is
    constant across units and PEHE collapses to an ATE error with an irreducible
    floor of sd(true ITE). Pass ``--test-feature-mask-fraction 1.0`` to reproduce
    UWYK's setting exactly.

D5. The outcome node is pinned to the node the path constraint was checked against.
    UWYK checks ``ensure_*_path`` against a uniformly drawn ``target_node``, then
    lets ``BasicProcessing._select_target_feature`` pick the real outcome *by
    variance* (90% from the top-variance quintile) and reassigns
    ``target_node = processor.selected_target_feature``
    (``InterventionalDataset.py:905``). The two coincide only by luck, so the
    path-regime labels are not actually enforced upstream. We pass
    ``target_feature=y_node`` so ``path_TY`` / ``path_YT`` /
    ``path_independent_TY`` mean what they say.

Degenerate-by-design regimes
----------------------------
``path_YT`` and ``path_independent_TY`` have no directed path T -> Y, so
intervening on T cannot move Y: Y_do0 == Y_do1 exactly and the true CATE is
identically 0. Those cells measure whether a model correctly reports *no*
effect. ``--self-test`` asserts it.

Environment
-----------
    UWYK_SRC   path to UWYK's ``src/``  (default /tmp/g4cfm/src)
    UWYK_ROOT  path to UWYK's repo root (default: parent of UWYK_SRC)

On Killarney both live under ``$DEPLOY_ROOT/external/uwyk``.

Usage
-----
    python UWYK_Fig3_4/generate_pehe_benchmark.py \
        --prior lingaus --nodes 2 5 20 35 50 --n-realizations 100 \
        --out-dir UWYK_Fig3_4/data

    python UWYK_Fig3_4/generate_pehe_benchmark.py \
        --prior complexmech --nodes 20 --hide-fractions 0.0 0.25 0.5 0.75 1.0 \
        --n-realizations 100 --out-dir UWYK_Fig3_4/data

Output: <out-dir>/<prior>/<n>node/<regime>/hide_<h>/r<idx>.npz  (+ manifest.json)
"""
from __future__ import annotations

# macOS: torch and XGBoost each ship their own OpenMP runtime, and having both
# live in one process segfaults inside XGBoost-backed mechanisms (a hard heap
# corruption -- faulthandler cannot even finish printing). Pinning OpenMP to one
# thread before torch is imported avoids it. Reproduced on the ComplexMech prior
# at 20+ nodes; harmless for the LinGaus prior, which uses no XGBoost.
# Override by exporting OMP_NUM_THREADS yourself.
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import yaml

# Repo root first: reuse the *proven* paired-noise helper from the training
# pipeline rather than forking it. Importing that module also installs its
# copy._deepcopy_dispatch patch for torch.Generator, which deepcopy(scm) below
# depends on (see PairedInterventionalDataset for the none_dealloc story).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from training.data.PairedInterventionalDataset import _propagate_paired  # noqa: E402

_UWYK_SRC = os.environ.get("UWYK_SRC", "/tmp/g4cfm/src")
_UWYK_ROOT = os.environ.get("UWYK_ROOT", os.path.dirname(_UWYK_SRC.rstrip("/")))

from pehe_metrics import NULL_EFFECT_REGIMES, REGIMES  # noqa: E402,F401


REJECT_TALLY = {
    "seen": 0, "accepted": 0, "saturation": 0, "ate_ratio": 0, "no_het": 0, "hi_het": 0,
    "pinned_vals": [], "ratio_vals": [], "accepted_pinned": [], "abs_vals": [],
    "tau_cv": [], "tau_rel": [], "tau_sd": [], "acc_tau_rel": [],
}


def _reset_tally():
    REJECT_TALLY.update(seen=0, accepted=0, saturation=0, ate_ratio=0, no_het=0, hi_het=0)
    REJECT_TALLY["pinned_vals"] = []
    REJECT_TALLY["ratio_vals"] = []
    REJECT_TALLY["accepted_pinned"] = []
    REJECT_TALLY["abs_vals"] = []
    REJECT_TALLY["tau_cv"] = []
    REJECT_TALLY["tau_rel"] = []
    REJECT_TALLY["tau_sd"] = []
    REJECT_TALLY["acc_tau_rel"] = []


def _q(vals, q):
    a = np.asarray(vals, dtype=float)
    return float(np.quantile(a, q)) if a.size else float("nan")


def _below(vals, thr):
    a = np.asarray(vals, dtype=float)
    return int(np.sum(a < thr)) if a.size else 0


def _tally_line():
    """One line describing what the saturation filter actually did.

    A rejection resamples, so it leaves no trace in n_ok / n_fail: "100 ok / 0
    fail" cannot distinguish a filter that fired 42 times from one that never
    fired at all. This reports the DISTRIBUTION of the pinned fraction as well as
    the reject counts, so the threshold can be chosen from the data rather than
    guessed, and so "nothing to reject" is visibly different from "not reached".
    """
    t = REJECT_TALLY
    if not t["seen"]:
        return "saturation filter: NEVER REACHED (no SCM got as far as the check)"
    pv = np.asarray(t["pinned_vals"], dtype=float)
    rv = np.asarray(t["ratio_vals"], dtype=float)
    q = np.quantile(pv, [0.5, 0.9, 0.99]) if pv.size else [float("nan")] * 3
    ap = np.asarray(t["accepted_pinned"], dtype=float)
    return ("  [filter] seen={seen} accepted={acc} rejected_pinned={sat} "
            "rejected_ratio={rat} rejected_no_het={nh} rejected_hi_het={hh}\n"
            "  [filter] pinned  p50={p50:.4f} p90={p90:.4f} p99={p99:.4f} "
            "max={mx:.4f}   accepted_max={amx:.4f}\n"
            "  [filter] pinned_abs (frac |y|>1-eps, only meaningful if y is "
            "bounded) p50={a50:.4f} max={amax:.4f}\n"
            "  [filter] |ATE|/sd  p50={r50:.3f} p90={r90:.3f} max={rmx:.3f}\n"
            "  [filter] sd(tau)/|E tau|  p10={c10:.4f} p50={c50:.4f} p90={c90:.4f}\n"
            "  [filter] sd(tau)/sd(y0)   p10={t10:.4f} p50={t50:.4f} p90={t90:.4f}\n"
            "  [filter] sd(tau) abs      p10={s10:.4g} p50={s50:.4g}\n"
            "  [filter] would-reject at sd(tau)/sd(y0) < : "
            "0.01 -> {b1}/{n}  0.05 -> {b5}/{n}  0.10 -> {b10}/{n}  0.20 -> {b20}/{n}\n"
            "  [filter] would-reject at pinned > 0.50: {bp}/{n}\n"
            "  [filter] ACCEPTED sd(tau)/sd(y0)  min={am:.4g} p10={ap10:.4g} "
            "p50={ap50:.4g}   (must all clear --min-tau-het)").format(
        seen=t["seen"], acc=t["accepted"], sat=t["saturation"], rat=t["ate_ratio"],
        nh=t["no_het"], hh=t["hi_het"],
        p50=q[0], p90=q[1], p99=q[2],
        mx=float(pv.max()) if pv.size else float("nan"),
        amx=float(ap.max()) if ap.size else float("nan"),
        a50=float(np.quantile(np.asarray(t["abs_vals"], dtype=float), 0.5))
        if t["abs_vals"] else float("nan"),
        amax=float(np.max(np.asarray(t["abs_vals"], dtype=float)))
        if t["abs_vals"] else float("nan"),
        c10=_q(t["tau_cv"], 0.10), c50=_q(t["tau_cv"], 0.50), c90=_q(t["tau_cv"], 0.90),
        t10=_q(t["tau_rel"], 0.10), t50=_q(t["tau_rel"], 0.50), t90=_q(t["tau_rel"], 0.90),
        s10=_q(t["tau_sd"], 0.10), s50=_q(t["tau_sd"], 0.50),
        n=t["seen"],
        b1=_below(t["tau_rel"], 0.01), b5=_below(t["tau_rel"], 0.05),
        b10=_below(t["tau_rel"], 0.10), b20=_below(t["tau_rel"], 0.20),
        bp=int(np.sum(np.asarray(t["pinned_vals"], dtype=float) > 0.5))
        if t["pinned_vals"] else 0,
        am=(float(np.min(np.asarray(t["acc_tau_rel"], dtype=float)))
            if t["acc_tau_rel"] else float("nan")),
        ap10=_q(t["acc_tau_rel"], 0.10), ap50=_q(t["acc_tau_rel"], 0.50),
        r50=float(np.quantile(rv, 0.5)) if rv.size else float("nan"),
        r90=float(np.quantile(rv, 0.9)) if rv.size else float("nan"),
        rmx=float(rv.max()) if rv.size else float("nan"))


def _import_uwyk():
    """Import UWYK's prior/preprocessing classes (lazy: needs UWYK_SRC present)."""
    if _UWYK_SRC not in sys.path:
        sys.path.insert(0, _UWYK_SRC)
    from priors.causal_prior.scm.SCMSampler import SCMSampler
    from priors.causal_prior.mechanisms.BinarizingMechanism import BinarizingMechanism
    from priordata_processing.BasicProcessing import BasicProcessing
    from utils.graph_utils import (
        adjacency_to_ancestor_matrix,
        propagate_ancestor_knowledge,
    )
    return (SCMSampler, BinarizingMechanism, BasicProcessing,
            adjacency_to_ancestor_matrix, propagate_ancestor_knowledge)


# ── UWYK config discovery ─────────────────────────────────────────────────────

def config_path(prior: str, n_nodes: int, regime: str, hide: float) -> str:
    base = os.path.join(_UWYK_ROOT, "experiments", "GraphConditioning", "Benchmarks")
    if prior == "lingaus":
        # Fig 3. UWYK has no hide-fraction axis here; the grid is <n>node_<regime>.yaml.
        return os.path.join(base, "LinGaus", f"{n_nodes}node_{regime}.yaml")
    if prior == "complexmech":
        # Fig 4. One YAML per (n, regime, hide-fraction).
        return os.path.join(base, "ComplexMechIDK", "configs",
                            f"{n_nodes}node", regime, f"hide_{hide}.yaml")
    raise ValueError(f"unknown prior {prior!r}")


def _val(cfg: dict, key: str, default=None):
    """Read a `{key: {value: v}}` entry from a UWYK YAML block."""
    entry = cfg.get(key)
    if entry is None:
        return default
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return default


# UWYK ships benchmark YAMLs only for these node counts.
UWYK_NODE_COUNTS = (2, 5, 10, 20, 35, 50)


def resolve_config(prior: str, n_nodes: int, regime: str, hide: float) -> tuple:
    """Return (cfg, source_path, synthesized).

    For a node count UWYK does not ship (e.g. 30, 40) we clone the nearest
    shipped config and override `num_nodes`.

    Diffing UWYK's per-n YAMLs shows exactly two numeric differences:
    `scm_config.num_nodes` and `model_config.num_features` (= num_nodes - 3).
    The second is a transformer input width; this module reads only
    scm_config / dataset_config / preprocessing_config, so overriding num_nodes
    reproduces the prior exactly. We set num_features too, for consistency if
    anything downstream ever reads the config.
    """
    exact = config_path(prior, n_nodes, regime, hide)
    if os.path.exists(exact):
        return load_config(exact), exact, False
    base_n = min(UWYK_NODE_COUNTS, key=lambda k: (abs(k - n_nodes), k))
    base = config_path(prior, base_n, regime, hide)
    if not os.path.exists(base):
        raise FileNotFoundError(f"no config for {prior} n={n_nodes} (and no base at {base})")
    cfg = load_config(base)
    cfg["scm_config"]["num_nodes"] = {"value": int(n_nodes)}
    if "model_config" in cfg and "num_features" in cfg["model_config"]:
        cfg["model_config"]["num_features"] = {"value": max(1, int(n_nodes) - 3)}
    return cfg, base, True


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for block in ("scm_config", "dataset_config", "preprocessing_config"):
        if block not in cfg:
            raise KeyError(f"{path} has no {block} block")
    # YAML gives node shapes as lists; SCMSampler forwards them to torch shapes.
    for key in ("mlp_node_shape", "xgb_node_shape"):
        entry = cfg["scm_config"].get(key)
        if isinstance(entry, dict) and isinstance(entry.get("value"), list):
            entry["value"] = tuple(entry["value"])
    return cfg


# ── One realization ───────────────────────────────────────────────────────────

_WARMED_UP = False


def _warm_up(cfg: dict, uwyk, regime: str, kwargs: dict) -> None:
    """Burn the process's one-time initialisation with a throwaway realization.

    The first realization involving XGBoost mechanisms does not reproduce; every
    one after it does (measured: call1 != call2 with max|d tau| 0.67, call2 ==
    call3 bitwise). The one-off state is somewhere in XGBoost's first fit and is
    not cleared by fitting a bare XGBRegressor, so rather than guess at it we
    simply spend one discarded generation. Every run burns exactly one, so real
    realizations always start from the same steady state and reproduce across
    runs. Uses its own sampler, and generate_realization reseeds the global torch
    RNG per attempt, so this cannot perturb the realizations that follow.
    """
    try:
        generate_realization(cfg, uwyk[0](cfg["scm_config"], seed=0), uwyk,
                             regime, seed=0, **kwargs)
    except Exception:  # noqa: BLE001 - warm-up is best-effort
        pass


def _to_binary(t, t0: float, t1: float):
    """Map the SCM's two treatment levels to {0, 1}: t1 -> 1, t0 -> 0.

    BinarizingMechanism draws t0/t1 from observed T quantiles, so the treatment
    column leaves BasicProcessing on those raw levels (e.g. {70.14, 76.29}), not
    {0, 1}. Every eval harness slices on `T == 1`, which on raw levels selects
    nothing and yields nan metrics -- so the remap has to happen here.

    Uses nearest-level assignment rather than a midpoint threshold so arm labels
    stay tied to (t0, t1) even if t1 < t0; Y_do0/Y_do1 are defined by those
    levels, so an inverted mapping would silently swap the arms.
    """
    import numpy as _np
    t = _np.asarray(t, dtype=_np.float64)
    return (_np.abs(t - t1) <= _np.abs(t - t0)).astype(_np.float32)


def generate_realization(
    cfg: dict,
    sampler,
    uwyk,
    regime: str,
    *,
    seed: int,
    hide_fraction: float,
    test_feature_mask_fraction: float,
    max_resample_attempts: int = 200,
    # Saturation rejection. ComplexMech's prior does not bound the
    # pre-activation scale (unlike the case-study generator's fan-in bounded
    # weights), so a tanh-like activation can pin the outcome at its boundary:
    # measured up to 99.2% of units at |Y| = 1, and a control arm with
    # sd(Y_do0) = 0.005 against a claimed ATE of 1.96. Such a realization has no
    # estimable effect -- every model correctly returns ~0 -- and since
    # saturation is a property of the REALIZATION, all of its queries fail
    # together, which is what collapses per-realization coverage to 0/1.
    #
    # Rejecting and resampling is UWYK's own idiom here: this loop already
    # continues on constant T, collapsed binarisation and low target variance.
    # Defaults are permissive (no rejection) so existing behaviour is unchanged.
    max_pinned: float = 1.0,
    min_tau_het: float = 0.0,
    max_tau_het: float = float('inf'),
    max_ate_sd_ratio: float = float("inf"),
    sat_eps: float = 1e-3,
    n_test_override: int | None = None,
    # Repeated observational samples for ONE accepted realization: same SCM, same
    # queries, same potential outcomes, a fresh training set each time. Returns a
    # LIST when > 1 and the single record when 1, so existing callers are untouched.
    # This is what makes frequentist coverage measurable here -- the truth has to be
    # one fixed number while only the sample varies.
    n_train_draws: int = 1,
) -> dict[str, Any]:
    """Sample one SCM and return a PEHE-ready realization.

    Follows InterventionalDataset.__getitem__ order of operations: sample SCM ->
    draw the intervention node -> binarise -> draw the outcome node -> check the
    path regime -> resample the whole SCM on failure (UWYK never re-picks (T, Y)
    within an SCM, and doing so would bias the graph distribution).
    """
    import networkx as nx

    (_SCMSampler, BinarizingMechanism, BasicProcessing,
     adjacency_to_ancestor_matrix, propagate_ancestor_knowledge) = uwyk

    global _WARMED_UP
    if not _WARMED_UP:
        _WARMED_UP = True   # set first: the warm-up call must not recurse
        _warm_up(cfg, uwyk, regime,
                 {"hide_fraction": hide_fraction,
                  "test_feature_mask_fraction": test_feature_mask_fraction})

    ds, pp = cfg["dataset_config"], cfg["preprocessing_config"]
    n_train = int(_val(ds, "max_number_train_samples_per_dataset"))
    n_test = int(_val(ds, "max_number_test_samples_per_dataset"))
    if n_test_override:
        n_test = int(n_test_override)
    max_features = int(_val(ds, "max_number_features"))
    min_target_variance = float(_val(ds, "min_target_variance", 1e-3))

    for retry in range(max_resample_attempts):
        attempt_seed = seed + retry
        # Reseed the GLOBAL torch RNG on every attempt. SCMSampler.sample(seed=)
        # seeds only the graph and the hyperparameter draws; mechanism weight
        # init and every noise draw read the global RNG instead. Without this the
        # benchmark is not regenerable -- two identical runs differed by
        # max|d tau| ~ 2.1 on ComplexMech, the same magnitude as the signal.
        # Seeding per attempt (not once per realization) also keeps realization r
        # independent of how many retries the preceding realizations burned.
        torch.manual_seed(attempt_seed)
        scm = sampler.sample(seed=attempt_seed)
        # Adjacency must come from the pre-intervention graph.
        org_scm = deepcopy(scm)

        all_nodes = list(scm.dag.nodes())
        if len(all_nodes) < 2:
            continue

        rng = torch.Generator()
        rng.manual_seed(attempt_seed)

        # --- observational context ------------------------------------------
        scm.sample_exogenous(num_samples=n_train)
        scm._fixed_endogenous_vec = None
        scm.sample_endogenous(num_samples=n_train)
        obs0_raw = scm.propagate(num_samples=n_train)
        obs0_raw = {k: v.clone().detach() for k, v in obs0_raw.items()}

        t_node = all_nodes[torch.randint(0, len(all_nodes), (1,), generator=rng).item()]

        # --- D1: binarise T using UWYK's own factory -------------------------
        try:
            bin_mech = BinarizingMechanism.from_observational_data(
                wrapped_mechanism=scm.mechanisms[t_node],
                obs_values=obs0_raw[t_node].clone().detach(),
            )
        except ValueError:
            continue  # constant T; resample the SCM (UWYK's own rejection style)
        scm.mechanisms[t_node] = bin_mech
        org_scm.mechanisms[t_node] = bin_mech
        t0_value, t1_value = float(bin_mech.t0), float(bin_mech.t1)

        # Re-sample the context so it reflects the binarised treatment.
        scm.sample_exogenous(num_samples=n_train)
        scm._fixed_endogenous_vec = None
        scm.sample_endogenous(num_samples=n_train)
        obs0_raw = scm.propagate(num_samples=n_train)
        obs0_raw = {k: v.clone().detach() for k, v in obs0_raw.items()}
        if obs0_raw[t_node].reshape(-1).unique().numel() < 2:
            continue  # binarisation collapsed to one class

        # --- outcome node + path regime (checked pre-intervention) -----------
        avail = [v for v in all_nodes if v != t_node]
        y_node = avail[torch.randint(0, len(avail), (1,), generator=rng).item()]

        if regime == "path_TY":
            ok = scm.exists_treatment_outcome_path(t_node, y_node)
        elif regime == "path_YT":
            ok = scm.exists_outcome_treatment_path(t_node, y_node)
        else:
            ok = scm.exists_no_connection_treatment_outcome(t_node, y_node)
        if not ok:
            continue
        if float(obs0_raw[y_node].reshape(-1).var()) < min_target_variance:
            continue

        # --- observational test pass (D3: source of X for both arms) ---------
        scm.sample_exogenous(num_samples=n_test)
        scm._fixed_endogenous_vec = None
        scm.sample_endogenous(num_samples=n_test)
        obs_test = scm.propagate(num_samples=n_test)
        obs_test = {k: v.clone().detach() for k, v in obs_test.items()}

        # --- D2: paired do(t0) / do(t1) sharing the observational noise ------
        intv_scm = deepcopy(scm)
        intv_scm.intervene(node=t_node)
        res0, res1 = _propagate_paired(scm, intv_scm, t_node, n_test, t0_value, t1_value)

        # --- saturation rejection (see max_pinned / max_ate_sd_ratio above) ---
        _y0 = res0[y_node].reshape(-1).detach().cpu().numpy().astype(np.float64)
        _y1 = res1[y_node].reshape(-1).detach().cpu().numpy().astype(np.float64)
        # TWO saturation measures, because the absolute one is only meaningful if
        # y is bounded to [-1, 1]:
        #
        #   _pinned_abs  frac(|y| > 1 - sat_eps).  This is what the earlier
        #       diagnosis used (99.2% of units at |Y| = 1). It is correct for a
        #       clipped/tanh outcome and MEANINGLESS otherwise -- a standard
        #       normal puts ~32% of its mass beyond |1|, so on unbounded y it
        #       reports a third of every sample as saturated.
        #
        #   _pinned_rel  mass piled within sat_eps of the sample's OWN range.
        #       Scale-free: ~0.2% for any continuous y, large only when values
        #       actually stack at a boundary. This is the one that detects
        #       clipping without assuming where the clip is.
        #
        # Rejection uses the relative measure; both are reported so the filter's
        # behaviour can be read off rather than assumed.
        def _pin_abs(v):
            return float((np.abs(v) > 1.0 - sat_eps).mean())

        def _pin_rel(v):
            lo, hi = float(v.min()), float(v.max())
            span = hi - lo
            if not np.isfinite(span) or span <= 0:
                return 1.0                      # constant outcome: fully degenerate
            tol = sat_eps * span
            return float(((v >= hi - tol) | (v <= lo + tol)).mean())

        _pinned_abs = max(_pin_abs(_y0), _pin_abs(_y1))
        _pinned = max(_pin_rel(_y0), _pin_rel(_y1))
        _sd0 = float(_y0.std())
        _tau = _y1 - _y0
        _tau_mean = float(_tau.mean())
        _tau_sd = float(_tau.std())
        _ratio = abs(_tau_mean) / max(_sd0, 1e-12)
        # HETEROGENEITY is what decides whether per-realization coverage can be
        # anything other than 0 or 1. If tau is near-constant across units, every
        # query in the realization has essentially the same truth, so one interval
        # either covers all of them or none -- which is exactly the binary 0/1
        # coverage seen on this benchmark. Reported relative to tau's own level
        # and to the outcome scale, since either alone can be misread.
        _tau_cv = _tau_sd / max(abs(_tau_mean), 1e-12)
        _tau_rel = _tau_sd / max(_sd0, 1e-12)
        # Record WHY, and record the value even when accepting: a rejection that
        # resamples successfully is otherwise invisible in n_ok / n_fail, so
        # "100 ok / 0 fail" cannot distinguish a filter that fired 42 times from
        # one that never fired at all.
        REJECT_TALLY["seen"] += 1
        REJECT_TALLY["pinned_vals"].append(_pinned)
        REJECT_TALLY["abs_vals"].append(_pinned_abs)
        REJECT_TALLY["tau_cv"].append(_tau_cv)
        REJECT_TALLY["tau_rel"].append(_tau_rel)
        REJECT_TALLY["tau_sd"].append(_tau_sd)
        REJECT_TALLY["ratio_vals"].append(_ratio)
        if _pinned > max_pinned:
            REJECT_TALLY["saturation"] += 1
            continue
        # NO-HETEROGENEITY rejection. Measured on this prior: 39% of realizations
        # at n=5 and 19% at n=20 have sd(tau)/sd(y0) < 0.01, i.e. tau is constant
        # across units. Every query then shares one truth, so a realization's
        # coverage can only be 0 or 1 -- which is the binary coverage this
        # benchmark produced. The mass sits AT zero rather than spread out (39
        # below 0.01 against 48 below 0.20), so a small threshold removes the
        # degenerate spike without trimming healthy realizations.
        # BAND, not a floor. Per-realization coverage behaves like
        #     cov_i ~= 2*Phi(h / sigma_i) - 1
        # with h the model's interval half-width, so a realization at half the
        # median spread already reads 0.9999 and one at 4x reads 0.68. Measured on
        # this prior, sigma/sd(y0) spans 1.5e13 unfiltered, and 73% of realizations
        # land as near-total-cover or near-total-miss. A floor alone leaves 43% of
        # them at >0.99, because the high tail is as damaging as the zeros.
        #
        # The band's job is NOT to make coverage stable -- only a model that adapts
        # its width per realization can do that, which is exactly what a calibrated
        # conditional density should do. It is to remove the PATHOLOGICAL cases:
        # sigma = 0, where every query shares one truth and coverage is 0 or 1 by
        # construction, and the extreme tail. What remains is attributable to the
        # models rather than to the data.
        if _tau_rel < min_tau_het:
            REJECT_TALLY["no_het"] += 1
            continue
        if _tau_rel > max_tau_het:
            REJECT_TALLY["hi_het"] += 1
            continue
        if _ratio > max_ate_sd_ratio:
            REJECT_TALLY["ate_ratio"] += 1
            continue
        REJECT_TALLY["accepted"] += 1
        REJECT_TALLY["accepted_pinned"].append(_pinned)
        REJECT_TALLY["acc_tau_rel"].append(_tau_rel)

        descendants = nx.descendants(scm.dag.g, t_node)
        n_descendant_features = sum(
            1 for v in all_nodes if v not in (t_node, y_node) and v in descendants
        )
        break
    else:
        raise RuntimeError(
            f"no usable SCM for regime={regime} in {max_resample_attempts} attempts "
            f"(seed={seed})"
        )

    # --- build the doubled test split -----------------------------------------
    # Rows [0:n_test) are do(t0), rows [n_test:2*n_test) are do(t1). X is the
    # observational pass tiled twice, so both arms share one x. Feeding both arms
    # as ONE split is what puts Y_do0 and Y_do1 on a common affine: UWYK's
    # process_from_splits concatenates train+test and fits shared statistics, so
    # two separate calls would scale the arms differently and PEHE would be
    # meaningless. It also forces shuffle_samples=False internally, so the
    # row-pairing between the halves survives.
    def _finalize(obs0_raw):
        """Train sample -> processed splits -> one realization record.

        Nested so n_train_draws can call it repeatedly with fresh training
        samples while the SCM, the queries and their potential outcomes stay
        exactly as accepted. Duplicating it in a separate script instead would
        let the two copies drift, and a silent divergence here changes what the
        benchmark IS.
        """
        def _col(x):
            return x.reshape(-1, 1).float()

        train_ds = {k: _col(v) for k, v in obs0_raw.items()}
        test_ds = {}
        for v in obs0_raw:
            if v == t_node:
                test_ds[v] = torch.cat([
                    torch.full((n_test, 1), t0_value), torch.full((n_test, 1), t1_value)
                ], dim=0)
            elif v == y_node:
                test_ds[v] = torch.cat([_col(res0[v]), _col(res1[v])], dim=0)
            else:
                tiled = _col(obs_test[v])
                test_ds[v] = torch.cat([tiled, tiled], dim=0)

        processor = BasicProcessing(
            n_features=max_features, max_n_features=max_features,
            n_train_samples=n_train, max_n_train_samples=n_train,
            n_test_samples=2 * n_test, max_n_test_samples=2 * n_test,
            dropout_prob=_val(pp, "dropout_prob", 0.0) or 0.0,
            target_feature=y_node,           # D5: pin the outcome we regime-checked
            intervened_feature=t_node,
            random_seed=seed,
            test_feature_mask_fraction=test_feature_mask_fraction,   # D4
            feature_standardize=_val(pp, "feature_standardize", True),
            feature_negative_one_one_scaling=_val(pp, "feature_negative_one_one_scaling", False),
            target_negative_one_one_scaling=_val(pp, "target_negative_one_one_scaling", True),
            yeo_johnson=_val(pp, "yeo_johnson", False),
            remove_outliers=_val(pp, "remove_outliers", True),
            outlier_quantile=_val(pp, "outlier_quantile", 0.99),
            shuffle_samples=False,           # keep do(t0)/do(t1) rows aligned
            shuffle_features=True,
            eps=1e-8,
        )
        X_tr, T_tr, Y_tr, X_te, T_te, Y_te = processor.process_from_splits(
            train_dataset=train_ds, test_dataset=test_ds, mode="fast",
        )

        X_test = X_te[:n_test]
        Y_do0 = Y_te[:n_test].reshape(-1)
        Y_do1 = Y_te[n_test:].reshape(-1)
        # X is tiled, so the two halves must be identical after processing.
        assert torch.allclose(X_te[:n_test], X_te[n_test:]), "arm X mismatch"

        # --- graph matrices, aligned to the processed column order ---------------
        kept = processor.kept_feature_indices
        ordered_nodes = [t_node, y_node] + list(kept)
        adj_raw = org_scm.get_adjacency_matrix(node_order=ordered_nodes)
        anc = 2.0 * adjacency_to_ancestor_matrix(adj_raw).float() - 1.0

        real_n = 2 + len(kept)
        if hide_fraction > 0.0:
            hide_rng = torch.Generator()
            hide_rng.manual_seed(seed + 424242)
            mask = torch.rand(real_n, real_n, generator=hide_rng) < hide_fraction
            anc[:real_n, :real_n][mask] = 0.0
        anc = propagate_ancestor_knowledge(anc)

        target_size = max_features + 2
        if anc.shape[0] < target_size:
            padded = torch.full((target_size, target_size), -1.0)
            padded[:anc.shape[0], :anc.shape[1]] = anc
            anc = padded

        _y0r = res0[y_node].reshape(-1).numpy().astype(np.float64)
        _y1r = res1[y_node].reshape(-1).numpy().astype(np.float64)
        _yp = np.concatenate([Y_do0.numpy().astype(np.float64),
                              Y_do1.numpy().astype(np.float64)])
        _yr = np.concatenate([_y0r, _y1r])
        _A = np.stack([np.ones_like(_yr), _yr], axis=1)
        _aff_a, _aff_b = np.linalg.lstsq(_A, _yp, rcond=None)[0]
        _aff_res = float(np.abs(_yp - (_aff_a + _aff_b * _yr)).max())

        return {
            "X_train": X_tr.numpy(),
            # Remapped to {0,1}; the raw levels are kept as t0_value / t1_value.
            "T_train": _to_binary(T_tr.reshape(-1).numpy(), t0_value, t1_value),
            "Y_train": Y_tr.reshape(-1).numpy(),
            "X_test": X_test.numpy(),
            "T_test_do0": _to_binary(T_te[:n_test].reshape(-1).numpy(), t0_value, t1_value),
            "T_test_do1": _to_binary(T_te[n_test:].reshape(-1).numpy(), t0_value, t1_value),
            "Y_do0": Y_do0.numpy(), "Y_do1": Y_do1.numpy(),
            "true_cate": (Y_do1 - Y_do0).numpy(),
            "anc_matrix": anc.numpy(), "adj_matrix": adj_raw.numpy(),
            "regime": regime,
            "hide_fraction": np.float32(hide_fraction),
            "n_nodes": np.int32(len(all_nodes)),
            "n_real_features": np.int32(len(kept)),
            # D3 diagnostic: 0 means this realization is exactly faithful to UWYK's
            # covariate construction (no descendant of T among the features).
            "n_descendant_features": np.int32(n_descendant_features),
            "test_feature_mask_fraction": np.float32(test_feature_mask_fraction),
            "t0_value": np.float32(t0_value), "t1_value": np.float32(t1_value),
            "seed": np.int64(seed),
            # --- unprocessed truth, for repeated-sample coverage -------------
            # true_cate above is in PROCESSED units, and the processor refits its
            # target scaling on whatever training sample it is handed, so that
            # number MOVES between draws of the same realization. These do not:
            # they are the SCM's own units, identical across every n_train_draws
            # replicate, and a fixed-estimand coverage statement can only be made
            # about a quantity that does not move.
            "Y_do0_raw": _y0r, "Y_do1_raw": _y1r,
            "true_cate_raw": (_y1r - _y0r),
            # processed = a + b * raw, recovered by least squares over both arms,
            # with the worst residual so a caller can tell when it does NOT hold:
            # outlier clipping is not affine, and silently rescaling by a bad b
            # would corrupt every interval instead of failing.
            "target_affine_a": np.float32(_aff_a),
            "target_affine_b": np.float32(_aff_b),
            "target_affine_resid": np.float32(_aff_res),
        }

    # n_train_draws > 1: the same estimand under repeated observational samples.
    # The redraw happens HERE, after the test pass and the paired do() passes have
    # already been drawn and stored, so those tensors cannot move -- a redraw before
    # them would advance the single RNG stream and change the queries too.
    recs = [_finalize(obs0_raw)]
    for _ in range(1, max(1, int(n_train_draws))):
        scm.sample_exogenous(num_samples=n_train)
        scm._fixed_endogenous_vec = None
        scm.sample_endogenous(num_samples=n_train)
        _ob = {k: v.clone().detach()
               for k, v in scm.propagate(num_samples=n_train).items()}
        recs.append(_finalize(_ob))
    return recs[0] if int(n_train_draws) <= 1 else recs


# ── Sweep driver ──────────────────────────────────────────────────────────────

def _save(rec: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **rec)


def run_sweep(args) -> dict:
    uwyk = _import_uwyk()
    SCMSampler = uwyk[0]

    manifest: dict[str, Any] = {
        "prior": args.prior,
        "nodes": args.nodes,
        "regimes": args.regimes,
        "hide_fractions": args.hide_fractions,
        "n_realizations": args.n_realizations,
        "test_feature_mask_fraction": args.test_feature_mask_fraction,
        "uwyk_root": _UWYK_ROOT,
        "cells": [],
    }

    for n_nodes in args.nodes:
        for regime in args.regimes:
            for hide in args.hide_fractions:
                try:
                    cfg, cfg_path, synth = resolve_config(
                        args.prior, n_nodes, regime, hide)
                except FileNotFoundError as exc:
                    print(f"[skip] {exc}", flush=True)
                    continue
                if synth:
                    print(f"[synth] n={n_nodes} not shipped by UWYK; cloned "
                          f"{os.path.basename(os.path.dirname(os.path.dirname(cfg_path)))}"
                          f" and set num_nodes={n_nodes}", flush=True)
                sampler = SCMSampler(cfg["scm_config"], seed=args.seed_base * 31 + 17)

                cell_dir = os.path.join(args.out_dir, args.prior, f"{n_nodes}node",
                                        regime, f"hide_{hide}")
                t0 = time.time()
                n_ok = n_fail = 0
                _reset_tally()
                descendant_free = 0
                for r in range(args.n_realizations):
                    out_path = os.path.join(cell_dir, f"r{r}.npz")
                    if os.path.exists(out_path) and not args.overwrite:
                        n_ok += 1
                        continue
                    seed = args.seed_base + 1_000_000 * r
                    try:
                        rec = generate_realization(
                            cfg, sampler, uwyk, regime,
                            seed=seed, hide_fraction=hide,
                            test_feature_mask_fraction=args.test_feature_mask_fraction,
                            max_pinned=args.max_pinned,
                            min_tau_het=args.min_tau_het,
                            max_tau_het=args.max_tau_het,
                            max_ate_sd_ratio=args.max_ate_sd_ratio,
                            n_test_override=args.n_test,
                        )
                    except Exception as exc:  # noqa: BLE001
                        n_fail += 1
                        print(f"[fail] {regime} n={n_nodes} hide={hide} r={r}: {exc}",
                              flush=True)
                        continue
                    if int(rec["n_descendant_features"]) == 0:
                        descendant_free += 1
                    _save(rec, out_path)
                    n_ok += 1

                dt = time.time() - t0
                cell = {
                    "config": cfg_path, "dir": cell_dir, "n_nodes": n_nodes,
                    "regime": regime, "hide_fraction": hide,
                    "n_ok": n_ok, "n_fail": n_fail,
                    "n_descendant_free": descendant_free,
                    "seconds": round(dt, 1),
                }
                manifest["cells"].append(cell)
                print(_tally_line(), flush=True)
                if os.environ.get("CMECH_DUMP_TALLY"):
                    # Raw per-attempt values, so a threshold or band can be chosen
                    # offline against the real prior instead of from quantiles.
                    _tp = os.path.join(
                        args.out_dir,
                        f"tally_{regime}_n{n_nodes}_h{hide}.npz")
                    os.makedirs(args.out_dir, exist_ok=True)
                    np.savez(_tp,
                             tau_rel=np.asarray(REJECT_TALLY["tau_rel"], dtype=float),
                             tau_cv=np.asarray(REJECT_TALLY["tau_cv"], dtype=float),
                             tau_sd=np.asarray(REJECT_TALLY["tau_sd"], dtype=float),
                             pinned=np.asarray(REJECT_TALLY["pinned_vals"], dtype=float),
                             ratio=np.asarray(REJECT_TALLY["ratio_vals"], dtype=float))
                    print(f"  [filter] tally -> {_tp}", flush=True)
                print(f"[done] {regime} n={n_nodes} hide={hide}: {n_ok} ok / "
                      f"{n_fail} fail, {descendant_free} descendant-free, {dt:.0f}s",
                      flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    man_path = os.path.join(args.out_dir, f"manifest_{args.prior}.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[manifest] {man_path}", flush=True)
    return manifest


def self_test(args) -> None:
    """Cheap invariant checks on a couple of realizations per regime.

    Asserts the two properties PEHE depends on:
      * regimes with no T -> Y path give an exactly-zero true CATE;
      * the path_TY regime actually moves Y (non-degenerate effect).
    """
    uwyk = _import_uwyk()
    SCMSampler = uwyk[0]
    n_nodes = args.nodes[0]
    hide = args.hide_fractions[0]

    for regime in args.regimes:
        try:
            cfg, cfg_path, _synth = resolve_config(args.prior, n_nodes, regime, hide)
        except FileNotFoundError as exc:
            print(f"[skip] {exc}")
            continue
        sampler = SCMSampler(cfg["scm_config"], seed=args.seed_base * 31 + 17)
        rec = generate_realization(
            cfg, sampler, uwyk, regime, seed=args.seed_base, hide_fraction=hide,
            test_feature_mask_fraction=args.test_feature_mask_fraction,
        )
        cate = rec["true_cate"]
        amax = float(np.abs(cate).max())
        tag = f"{args.prior} {regime} n={n_nodes}"
        if regime in NULL_EFFECT_REGIMES:
            assert amax == 0.0, f"{tag}: expected zero CATE, got max|cate|={amax:g}"
            print(f"[ok] {tag}: CATE identically zero, as required")
        else:
            assert amax > 0.0, f"{tag}: T->Y regime produced an all-zero CATE"
            print(f"[ok] {tag}: max|cate|={amax:.4g}  sd={float(cate.std()):.4g}  "
                  f"desc_feats={int(rec['n_descendant_features'])}/"
                  f"{int(rec['n_real_features'])}")
        # X must be identical across arms and T must actually differ.
        assert not np.array_equal(rec["T_test_do0"], rec["T_test_do1"]), \
            f"{tag}: both arms got the same treatment level"

        # T must reach the harnesses as {0,1}: they all slice on `T == 1`, and
        # raw quantile levels select nothing and produce nan metrics.
        for key in ("T_train", "T_test_do0", "T_test_do1"):
            vals = np.unique(rec[key])
            assert np.isin(vals, (0.0, 1.0)).all(), \
                f"{tag}: {key} not in {{0,1}} (got {vals[:5]})"
        assert set(np.unique(rec["T_train"])) == {0.0, 1.0}, \
            f"{tag}: T_train is single-arm ({np.unique(rec['T_train'])})"
        assert (rec["T_test_do0"] == 0).all() and (rec["T_test_do1"] == 1).all(), \
            f"{tag}: do-arms mislabelled"
        _f1 = float(rec["T_train"].mean())
        print(f"[ok] {tag}: T in {{0,1}}, treated fraction {_f1:.3f}")

        # Regenerating the same realization must reproduce it bitwise, or the
        # benchmark cannot be rebuilt and method comparisons across runs are
        # meaningless. Guards the per-attempt global reseed in
        # generate_realization -- SCMSampler.sample(seed=) alone does NOT pin
        # mechanism weights or noise.
        sampler2 = SCMSampler(cfg["scm_config"], seed=args.seed_base * 31 + 17)
        rec2 = generate_realization(
            cfg, sampler2, uwyk, regime, seed=args.seed_base, hide_fraction=hide,
            test_feature_mask_fraction=args.test_feature_mask_fraction,
        )
        for key in ("true_cate", "X_train", "Y_train", "X_test", "anc_matrix"):
            assert np.array_equal(rec[key], rec2[key]), \
                f"{tag}: {key} not reproducible across identical runs"
        print(f"[ok] {tag}: regenerates bitwise-identically")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prior", choices=("lingaus", "complexmech"), required=True,
                   help="lingaus = Figure 3 prior, complexmech = Figure 4 prior")
    p.add_argument("--nodes", type=int, nargs="+", default=[2, 5, 20, 35, 50])
    p.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=REGIMES)
    p.add_argument("--hide-fractions", type=float, nargs="+", default=None,
                   help="ancestor-matrix hide fractions; default 0.0 for lingaus, "
                        "the full Fig-4 sweep for complexmech")
    p.add_argument("--n-realizations", type=int, default=100)
    p.add_argument("--n-test", type=int, default=None,
                   help="queries per realization; overrides the prior config's "
                        "max_number_test_samples_per_dataset (currently 1000)")
    p.add_argument("--max-pinned", type=float, default=1.0,
                   help="reject a draw when this fraction of |Y| exceeds "
                        "1 - sat_eps in either arm (saturated outcome). "
                        "1.0 = no rejection; 0.01 was measured to flag 42%% of "
                        "5node/path_TY draws")
    p.add_argument("--min-tau-het", type=float, default=0.0,
                   help="reject a realization when sd(tau)/sd(y0) is below this. "
                        "Guards against tau being CONSTANT across units, which makes "
                        "per-realization coverage necessarily 0 or 1. Measured on "
                        "complexmech: 39%% of n=5 realizations are below 0.01.")
    p.add_argument("--max-tau-het", type=float, default=float("inf"),
                   help="reject when sd(tau)/sd(y0) EXCEEDS this. Pairs with "
                        "--min-tau-het to bound the dynamic range of tau's spread. "
                        "Unfiltered that range is 1.5e13 on complexmech, and the "
                        "extreme tail forces coverage to 0 just as the zeros force "
                        "it to 1. See benchmarks/cmech_band_analysis.py.")
    p.add_argument("--max-ate-sd-ratio", type=float, default=float("inf"),
                   help="reject when |ATE| / sd(Y_do0) exceeds this -- an effect "
                        "larger than a couple of times the outcome's own spread "
                        "is a saturation artifact, not a recoverable effect")
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--out-dir", default=os.path.join(_REPO_ROOT, "UWYK_Fig3_4", "data"))
    p.add_argument("--test-feature-mask-fraction", type=float, default=0.0,
                   help="D4. UWYK's configs use 1.0, which zeroes EVERY test "
                        "covariate and collapses CATE to a constant. Default 0.0 "
                        "keeps covariates; pass 1.0 to reproduce UWYK exactly.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--self-test", action="store_true",
                   help="run invariant checks on one realization per regime and exit")
    args = p.parse_args()

    if args.hide_fractions is None:
        args.hide_fractions = [0.0] if args.prior == "lingaus" \
            else [0.0, 0.25, 0.5, 0.75, 1.0]

    if not os.path.isdir(_UWYK_SRC):
        raise SystemExit(
            f"UWYK_SRC not found at {_UWYK_SRC}. Set UWYK_SRC (and UWYK_ROOT) to "
            f"the deployed UWYK checkout, e.g. $DEPLOY_ROOT/external/uwyk/src."
        )

    if args.test_feature_mask_fraction >= 1.0:
        print("[warn] test_feature_mask_fraction=1.0 zeroes every test covariate; "
              "tau(x) becomes constant and PEHE degenerates to an ATE error.",
              flush=True)

    if args.self_test:
        self_test(args)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
