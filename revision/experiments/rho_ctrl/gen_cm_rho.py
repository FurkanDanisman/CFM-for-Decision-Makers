"""ComplexMech n = 5 with a controlled cross-world coupling of the outcome noise.
python gen_cm_rho.py OUT_ROOT [--per-bin 100] [--max-tries 20000]
    env UWYK_SRC, UWYK_ROOT as for UWYK_Fig3_4/generate_pehe_benchmark.py

The benchmark's own generator (UWYK_Fig3_4/generate_pehe_benchmark.generate_realization, the n = 5 path_TY
hide_0.0 config, the generation job's rejection settings), with one change inside the paired do-pass
(_propagate_paired, replicated here line for line): Y's endogenous noise in the do(t1) half becomes
    rho_knob * eps + sqrt(1 - rho_knob^2) * eps',   eps' a fresh draw from Y's own noise distribution
(drawn with the global RNG state saved and restored, so every other draw of the generator is unchanged).
rho_knob = 1 is the original generator. Mean and variance of the treated-arm noise are kept; its shape is a
mixture of two draws of Y's noise law.

Kept: realizations whose covariates are all pre-treatment (no mediator, no descendant of Y; the checks of
benchmarks/rho/count_pretreatment.py on the stored graph) and in which noise reaches Y. Their true
rho(x) is computed exactly as in revision/experiments/rho_true/cmech_true_rho.py (redraw only Y's noise,
coupled at rho_knob, over the same doubled batch); dataset rho = median over its queries.
Bins by the computed rho: [-0.5, 0), [0, 0.5), [0.5, 1]; the knob is drawn uniformly from the emptiest bin.
Layout (the harness reads it with DATASET=CMECH_n5, UWYK_FIG34_DATA=OUT_ROOT/bin{b}):
    OUT_ROOT/bin{b}/complexmech/5node/path_TY/hide_0.0/r{i}.npz
true_cate in each file is Y_do1 - Y_do0 of the processed outcomes, i.e. the realized ITE; extra keys
rho_knob, rho_true, rho_true_q.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(REPO, "UWYK_Fig3_4"))
sys.path.insert(0, os.path.join(REPO, "benchmarks", "rho"))
import generate_pehe_benchmark as G                           # noqa: E402
from count_pretreatment import reach                           # noqa: E402

MIN_HET, MAX_HET, MAX_PIN = 0.05, 2.0, 0.5                     # submit_cmech_gen.sbatch
N_TEST = 100
SEED_BASE = 20261014
BINS = ((-0.5, 0.0), (0.0, 0.5), (0.5, 1.0))
K = 500

STATE = {"knob": 1.0, "y": None, "cap": None}


def _record_path(orig):
    def f(self, t, y):
        STATE["y"] = y
        return orig(self, t, y)
    return f


def _draw_raw(intv, v, n):
    """n draws of node v's endogenous noise, (n, d), from the current global RNG state."""
    dist = intv.endogenous_noise
    s, e = intv._endo_slices[v]
    if hasattr(dist, "sample_shape") and not isinstance(dist, dict):
        return torch.as_tensor(dist.sample_shape((n, intv._total_endo_dim)), dtype=torch.float32)[:, s:e]
    return torch.as_tensor(dist[v].sample_shape((n, intv._flat_dim[v])), dtype=torch.float32)


def _draw_like(intv, v, n):
    """As _draw_raw, with the global RNG state restored afterwards (the generator's stream is untouched)."""
    st, nst = torch.get_rng_state(), np.random.get_state()
    try:
        return _draw_raw(intv, v, n)
    finally:
        torch.set_rng_state(st); np.random.set_state(nst)


def _propagate_paired_coupled(obs_scm, intv_scm, treatment_node, n_test, t0_value, t1_value):
    """training/data/PairedInterventionalDataset._propagate_paired, plus the coupling of Y's noise."""
    B2 = 2 * n_test
    total_exo = intv_scm._total_exo_dim
    fixed_exo_vec = torch.zeros(B2, total_exo, dtype=torch.float32)
    for v in intv_scm._exo_order:
        s, e = intv_scm._exo_slices[v]
        d = e - s
        if v == treatment_node:
            t_vals = torch.cat([torch.full((n_test,), t0_value), torch.full((n_test,), t1_value)])
            fixed_exo_vec[:, s:e] = t_vals.reshape(B2, d)
        else:
            old = obs_scm._fixed_exogenous[v]
            tiled = old.repeat(2) if old.dim() == 1 else old.repeat(2, 1)
            fixed_exo_vec[:, s:e] = tiled.reshape(B2, d)
    fixed_exo = {}
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
        # ── the one change: couple Y's own noise across the two arms at rho_knob ──
        y = STATE["y"]
        k = float(STATE["knob"])
        if y in getattr(intv_scm, "_endo_slices", {}) and k != 1.0:
            s, e = intv_scm._endo_slices[y]
            fresh = _draw_like(intv_scm, y, n_test)
            fixed_endo_vec[n_test:, s:e] = k * fixed_endo_vec[:n_test, s:e] + np.sqrt(max(1 - k * k, 0.0)) * fresh
    fixed_endo = {}
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
    STATE["cap"] = (intv_scm, n_test, res0, res1)
    return res0, res1


G._propagate_paired = _propagate_paired_coupled


def pretreatment(rec):
    A = np.asarray(rec["adj_matrix"], float)
    nf = int(rec["n_real_features"])
    if int(rec["n_nodes"]) != 2 + nf:
        return False
    R = reach(A[:2 + nf, :2 + nf])
    return not any((R[0, k] and R[k, 1]) or R[1, k] for k in range(2, 2 + nf))


def true_rho():
    """rho(x) per query: redraw Y's noise (coupled at the knob) over the same doubled batch."""
    intv, n, r0, r1 = STATE["cap"]
    y, k = STATE["y"], float(STATE["knob"])
    if not intv._fixed_endogenous or y not in intv._fixed_endogenous:
        return None
    B2 = 2 * n
    mech = intv.mechanisms[y]
    feat = torch.empty(B2, intv._parent_total_dim[y], dtype=intv.dtype)
    for p, s, e in intv._parent_slices[y]:
        feat[:, s:e] = torch.cat([r0[p].reshape(n, -1), r1[p].reshape(n, -1)], dim=0)
    eps_used = intv._fixed_endogenous[y]
    with torch.no_grad():
        back = mech._forward(feat, eps=eps_used).reshape(B2)
    if not (np.allclose(back[:n].numpy(), r0[y].reshape(-1).numpy(), atol=1e-5)
            and np.allclose(back[n:].numpy(), r1[y].reshape(-1).numpy(), atol=1e-5)):
        raise RuntimeError("Y's mechanism does not reproduce the generated outcomes")
    Y0 = np.empty((K, n)); Y1 = np.empty((K, n))
    st, nst = torch.get_rng_state(), np.random.get_state()      # the redraws must not move the generator's stream
    try:
        torch.manual_seed(SEED_BASE); np.random.seed(SEED_BASE % (2 ** 32))
        with torch.no_grad():
            for j in range(K):
                e0 = _draw_raw(intv, y, n); e1 = _draw_raw(intv, y, n)   # fresh, different draws
                eps = torch.cat([e0, k * e0 + np.sqrt(max(1 - k * k, 0.0)) * e1], dim=0).reshape(eps_used.shape)
                out = mech._forward(feat, eps=eps).reshape(B2).numpy()
                Y0[j], Y1[j] = out[:n], out[n:]
    finally:
        torch.set_rng_state(st); np.random.set_state(nst)
    s0, s1 = Y0.std(0), Y1.std(0)
    det = (s0 < 1e-9) | (s1 < 1e-9)
    if det.all():
        return None
    c = ((Y0 - Y0.mean(0)) * (Y1 - Y1.mean(0))).mean(0) / np.where(det, 1.0, s0 * s1)
    return np.where(det, np.nan, c)


def bin_of(r):
    if not np.isfinite(r) or r < -0.5 or r > 1.0:
        return None
    return 0 if r < 0.0 else (1 if r < 0.5 else 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--per-bin", type=int, default=100)
    ap.add_argument("--max-tries", type=int, default=20000)
    a = ap.parse_args()
    uwyk = G._import_uwyk()
    SCM = sys.modules["priors.causal_prior.scm.SCM"].SCM
    SCM.exists_treatment_outcome_path = _record_path(SCM.exists_treatment_outcome_path)
    cfg, _, _ = G.resolve_config("complexmech", 5, "path_TY", 0.0)
    sampler = uwyk[0](cfg["scm_config"], seed=SEED_BASE * 31 + 17)
    rng = np.random.default_rng(SEED_BASE)
    have = [0, 0, 0]
    stats = dict(tries=0, failed=0, not_pretreatment=0, no_noise=0, out_of_range=0, bin_full=0)
    t0 = time.time()
    for r in range(a.max_tries):
        if min(have) >= a.per_bin:
            break
        target = int(np.argmin(have))
        STATE["knob"] = float(rng.uniform(*BINS[target]))
        stats["tries"] += 1
        try:
            rec = G.generate_realization(cfg, sampler, uwyk, "path_TY", seed=SEED_BASE + 1_000_000 * r,
                                         hide_fraction=0.0, test_feature_mask_fraction=0.0, max_pinned=MAX_PIN,
                                         min_tau_het=MIN_HET, max_tau_het=MAX_HET, n_test_override=N_TEST)
        except Exception:                                      # run_sweep also skips failures
            stats["failed"] += 1
            continue
        if not pretreatment(rec):
            stats["not_pretreatment"] += 1
            continue
        rq = true_rho()
        if rq is None:
            stats["no_noise"] += 1
            continue
        rt = float(np.nanmedian(rq))
        b = bin_of(rt)
        if b is None:
            stats["out_of_range"] += 1
            continue
        if have[b] >= a.per_bin:
            stats["bin_full"] += 1
            continue
        rec.update(rho_knob=np.float32(STATE["knob"]), rho_true=np.float32(rt), rho_true_q=rq.astype(np.float32))
        d = os.path.join(a.out, f"bin{b}", "complexmech", "5node", "path_TY", "hide_0.0")
        os.makedirs(d, exist_ok=True)
        G._save(rec, os.path.join(d, f"r{have[b]}.npz"))
        have[b] += 1
        if stats["tries"] % 50 == 0 or min(have) >= a.per_bin:
            print(f"tries={stats['tries']} per bin={have} {stats} ({time.time() - t0:.0f}s)", flush=True)
    print(f"done: per bin={have} {stats} ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(a.out, "manifest_cm.json"), "w") as fh:
        json.dump(dict(per_bin=have, **stats), fh, indent=1)


if __name__ == "__main__":
    main()
