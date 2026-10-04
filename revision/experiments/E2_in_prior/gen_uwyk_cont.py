"""UWYK 1D on its own training prior, continuous T (no binarization).
python gen_uwyk_cont.py {new,same} R0:R1 out_root          (UWYK src on PYTHONPATH, env UWYK)

As in UWYK's InterventionalDataset (best_model_config.yaml):
  - the SCM prior is its scm_config; T is the raw, untransformed node value (BasicProcessing.process);
  - an intervention value is a random draw from the observed T values (interventional_distribution_type
    = resample -> ResamplingDist).
For each query unit we draw two such values, t0 < t1, and the truth is that unit's
Delta = Y(do t1) - Y(do t0): shared noise (both arms reuse every draw) and independent noise
(arm 0 gets fresh endogenous noise for every descendant of T), as for the other priors.
World search = scm_prior.generate_paired_sample_with_raw (graph2d's _generate_one) minus binarization.
The per-query levels are saved as T_test0 / T_test1 for the harness (T_ENCODING=continuous).
"""
import os, random, sys
from copy import deepcopy
import numpy as np
import networkx as nx
import torch
import yaml
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "../../..")
sys.path.insert(0, os.path.join(REPO, "benchmarks", "context_sweep"))
import scm_prior as SP
from common import write_world, done, N_CTX, N_Q

mode, (R0, R1), out = sys.argv[1], map(int, sys.argv[2].split(":")), sys.argv[3]
SEED = 20261003
uwyk = os.environ.get("UWYK", os.path.join(REPO, "external/uwyk_reproduce"))
cfg_all = yaml.safe_load(open(os.path.join(uwyk, "experiments/checkpoints/full_conditioned_model/"
                                           "final_earlytest_full_conditioning_16773252.0/best_model_config.yaml")))
scm_config = cfg_all["scm_config"]


def world(r):
    """SCM search exactly as scm_prior.generate_paired_sample_with_raw, without binarizing T."""
    from priors.causal_prior.scm.SCMSampler import SCMSampler
    random.seed(SEED + r); np.random.seed(SEED + r)
    sampler = SCMSampler(scm_config, seed=(SEED + r * 1_000) * 31 + 17)
    for attempt in range(10_000):
        s = SEED + r * 1_000 + attempt * 997
        torch.manual_seed(s)
        scm = sampler.sample(seed=s)
        nodes = sorted(scm.dag.nodes())
        if len(nodes) < 3:
            continue
        g = torch.Generator(); g.manual_seed(s)
        for _ in range(30):
            T = nodes[torch.randint(0, len(nodes), (1,), generator=g).item()]
            rest = [v for v in nodes if v != T]
            Y = rest[torch.randint(0, len(rest), (1,), generator=g).item()]
            if scm.exists_treatment_outcome_path(T, Y):
                break
        else:
            continue
        feats = [v for v in nodes if v not in (T, Y)]
        # UWYK's own rejection rule (dataset_config): target variance >= 1e-2, >= 20% unique values
        scm.sample_exogenous(N_CTX); scm._fixed_endogenous_vec = None; scm.sample_endogenous(N_CTX)
        y = scm.propagate(N_CTX)[Y].reshape(-1)
        if not torch.isfinite(y).all() or y.var() < 1e-2 or y.unique().numel() < 0.2 * N_CTX:
            continue
        intv = deepcopy(scm); intv.intervene(T)
        return scm, intv, T, Y, feats
    raise RuntimeError("no SCM")


def propagate_pairs(scm, intv, T, Y, t0, t1, fresh_seed=None):
    """Both arms of the N_Q query units now cached in scm, at per-unit levels t0, t1 (tensors).
    As scm_prior._propagate_paired: every noise draw tiled across the arms. With fresh_seed, arm 0's
    endogenous noise is redrawn for every descendant of T (the independent truth)."""
    n, B2 = N_Q, 2 * N_Q
    exo = torch.zeros(B2, intv._total_exo_dim)
    for v in intv._exo_order:
        s, e = intv._exo_slices[v]
        if v == T:
            exo[:, s:e] = torch.cat([t0, t1]).reshape(B2, e - s)
        else:
            old = scm._fixed_exogenous[v]
            exo[:, s:e] = (old.repeat(2) if old.dim() == 1 else old.repeat(2, 1)).reshape(B2, e - s)
    view = lambda vec, sl, v, ex: vec[:, slice(*sl[v])].reshape(B2) if (ex and intv.use_exogenous_mechanisms) \
        else vec[:, slice(*sl[v])].reshape(B2, *intv._node_shape.get(v, ()))
    intv._fixed_exogenous_vec, intv._fixed_batch = exo, B2
    intv._fixed_exogenous = {v: view(exo, intv._exo_slices, v, True) for v in intv._exo_order}
    endo = torch.zeros(B2, intv._total_endo_dim)
    for v in intv._endo_order:
        s, e = intv._endo_slices[v]
        old = scm._fixed_endogenous.get(v) if scm._fixed_endogenous else None
        if old is not None:
            endo[:, s:e] = old.reshape(n, e - s).repeat(2, 1)
    if fresh_seed is not None:
        with torch.random.fork_rng():
            torch.manual_seed(fresh_seed)
            intv.sample_endogenous(B2)
        fresh = intv._fixed_endogenous_vec
        for v in nx.descendants(intv.dag.g, T):
            s, e = intv._endo_slices[v]
            endo[:n, s:e] = fresh[:n, s:e]
    intv._fixed_endogenous_vec = endo
    intv._fixed_endogenous = {v: view(endo, intv._endo_slices, v, False) for v in intv._endo_order}
    res = intv.propagate(B2)
    return res[Y][:n].reshape(-1).float(), res[Y][n:].reshape(-1).float()


def dataset(r, w):
    scm, intv, T, Y, feats = w
    torch.manual_seed(SEED + 1_000_003 + r)
    cols = lambda o, m: torch.cat([o[v].reshape(m, -1).float() for v in feats], dim=1)
    scm.sample_exogenous(N_CTX); scm._fixed_endogenous_vec = None; scm.sample_endogenous(N_CTX)
    obs = scm.propagate(N_CTX)
    T_ctx = obs[T].reshape(-1).float()
    pick = torch.stack([torch.randperm(N_CTX)[:2] for _ in range(N_Q)])   # 2 draws, no replacement
    t0, t1 = T_ctx[pick].min(1).values, T_ctx[pick].max(1).values
    scm.sample_exogenous(N_Q); scm._fixed_endogenous_vec = None; scm.sample_endogenous(N_Q)
    obs_test = scm.propagate(N_Q)
    y0, y1 = propagate_pairs(scm, intv, T, Y, t0, t1)
    y0i, y1i = propagate_pairs(scm, intv, T, Y, t0, t1, fresh_seed=SEED + 7 * r + 1)
    X_ctx, X_q = SP._standardize(cols(obs, N_CTX), cols(obs_test, N_Q))
    write_world(out, r, X_ctx, T_ctx, obs[Y].reshape(-1).float(), X_q,
                {"delta_shared": y1 - y0, "delta_indep": y1i - y0i},
                extra={"T_test0": t0, "T_test1": t1})
    print(f"r={r} F={len(feats)}", flush=True)


if mode == "new":
    for r in range(R0, R1):
        if not done(out, r):
            dataset(r, world(r))
else:
    w = world(0)                                               # one world; every shard redraws the same one
    for r in range(R0, R1):
        if not done(out, r):
            dataset(r, w)
