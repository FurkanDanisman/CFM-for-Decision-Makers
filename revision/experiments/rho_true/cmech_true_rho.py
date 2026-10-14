"""True cross-world correlation rho(x) on the ComplexMech realizations whose covariates are all
pre-treatment, exactly from UWYK's generator.
python cmech_true_rho.py DATA_ROOT OUT_PREFIX [--nodes 5 10 20 30 40 50] [--k 500]
    DATA_ROOT = $SCRATCH/cmech_data_v2   (complexmech/<n>node/path_TY/hide_0.0/r*.npz)
    env UWYK_SRC, UWYK_ROOT as for UWYK_Fig3_4/generate_pehe_benchmark.py

Which realizations: no covariate is a mediator (T -> X_k -> Y) or a descendant of Y
(benchmarks/rho/count_pretreatment.py). Then every parent of Y is T or a covariate, the covariates do
not depend on Y's noise, and rho(x) = corr(Y^do(1), Y^do(0) | X = x) is obtained by redrawing ONLY Y's
endogenous noise, shared between the two arms, with everything else as generated.

Per realization:
 1. Regenerate it with the benchmark's own call (generate_realization, the cell's config, a fresh
    SCMSampler(seed = 17) as run_sweep builds it, the stored seed, the sbatch's rejection settings) and
    capture the accepted paired do-pass. Check: the regenerated Y_do0_raw / Y_do1_raw equal the stored ones,
    and Y's mechanism re-evaluated on the captured inputs and noise reproduces them.
 2. K times: draw Y's endogenous noise for the n_test units from Y's own noise distribution, tile it over
    the two arms (as _propagate_paired does), and re-evaluate Y's mechanism on the SAME doubled batch of
    parent values (UWYK's batch norm, if any, then sees the same batch). rho(x_i) = corr over the K draws of
    (Y_i^do(0), Y_i^do(1)). A unit whose two outcomes do not vary (no noise reaches Y) is `deterministic`.
Raw SCM outcomes are used (the processed ones are clipped at the 1 % / 99 % quantiles).
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(REPO, "UWYK_Fig3_4"))
sys.path.insert(0, os.path.join(REPO, "benchmarks", "rho"))
import generate_pehe_benchmark as G                           # noqa: E402
from count_pretreatment import classify                        # noqa: E402

# the generation job's settings (benchmarks/cluster/submit_cmech_gen.sbatch)
MIN_HET, MAX_HET, MAX_PIN = 0.05, 2.0, 0.5
SEED_BASE = 0

_CAP = []
_orig = G._propagate_paired


def _capture(scm, intv_scm, t_node, n_test, t0, t1):
    r0, r1 = _orig(scm, intv_scm, t_node, n_test, t0, t1)
    _CAP.append((intv_scm, n_test, r0, r1))
    return r0, r1


G._propagate_paired = _capture


def _cat(r0, r1, v, B2):
    return torch.cat([r0[v].reshape(B2 // 2, -1), r1[v].reshape(B2 // 2, -1)], dim=0)


def _draw_y_noise(intv, y, n, like):
    dist = intv.endogenous_noise
    s, e = intv._endo_slices[y]
    if hasattr(dist, "sample_shape") and not isinstance(dist, dict):
        x = dist.sample_shape((n, intv._total_endo_dim))
        x = torch.as_tensor(x, dtype=intv.dtype)[:, s:e]
    else:
        x = torch.as_tensor(dist[y].sample_shape((n, intv._flat_dim[y])), dtype=intv.dtype)
    x = x.repeat(2, 1)                                          # shared by both arms
    return x.reshape(like.shape)


def regenerate(cfg, uwyk, sampler, r, n_test):
    """generate_realization exactly as run_sweep calls it for realization r (same sampler, in order)."""
    _CAP.clear()
    rec = G.generate_realization(cfg, sampler, uwyk, "path_TY", seed=SEED_BASE + 1_000_000 * r, hide_fraction=0.0,
                                 test_feature_mask_fraction=0.0, max_pinned=MAX_PIN, min_tau_het=MIN_HET,
                                 max_tau_het=MAX_HET, n_test_override=n_test)
    return rec, _CAP[-1]


def one(rec, cap, f, K):
    z = np.load(f, allow_pickle=True)
    y0s, y1s = np.asarray(z["Y_do0_raw"], float), np.asarray(z["Y_do1_raw"], float)
    regen_ok = (np.allclose(rec["Y_do0_raw"], y0s, atol=1e-5) and np.allclose(rec["Y_do1_raw"], y1s, atol=1e-5))
    intv, n, r0, r1 = cap
    B2 = 2 * n
    y = next((v for v in r0 if np.allclose(r0[v].reshape(-1).numpy(), y0s, atol=1e-5)
              and np.allclose(r1[v].reshape(-1).numpy(), y1s, atol=1e-5)), None)
    if y is None:
        return dict(regen_ok=int(regen_ok), mech_ok=0)
    mech = intv.mechanisms[y]
    D = intv._parent_total_dim[y]
    feat = torch.empty(B2, D, dtype=intv.dtype)
    for p, s, e in intv._parent_slices[y]:
        feat[:, s:e] = _cat(r0, r1, p, B2)
    eps0 = intv._fixed_endogenous.get(y) if intv._fixed_endogenous else None
    with torch.no_grad():
        back = mech._forward(feat, eps=eps0).reshape(B2)
    mech_ok = bool(np.allclose(back[:n].numpy(), y0s, atol=1e-5) and np.allclose(back[n:].numpy(), y1s, atol=1e-5))
    if eps0 is None:                                            # no noise enters Y at all
        return dict(regen_ok=int(regen_ok), mech_ok=int(mech_ok), n_units=n, n_det=n,
                    rho_median=float("nan"), share_gt099=float("nan"))
    Y0 = np.empty((K, n)); Y1 = np.empty((K, n))
    with torch.no_grad():
        for k in range(K):
            out = mech._forward(feat, eps=_draw_y_noise(intv, y, n, eps0)).reshape(B2).numpy()
            Y0[k], Y1[k] = out[:n], out[n:]
    s0, s1 = Y0.std(0), Y1.std(0)
    det = (s0 < 1e-9) | (s1 < 1e-9)
    c = ((Y0 - Y0.mean(0)) * (Y1 - Y1.mean(0))).mean(0) / np.where(det, 1.0, s0 * s1)
    rho = np.where(det, np.nan, c)
    fin = rho[np.isfinite(rho)]
    return dict(regen_ok=int(regen_ok), mech_ok=int(mech_ok), n_units=n, n_det=int(det.sum()),
                rho_median=float(np.median(fin)) if fin.size else float("nan"),
                share_gt099=float(np.mean(fin > 0.99)) if fin.size else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out")
    ap.add_argument("--nodes", nargs="+", type=int, default=[5, 10, 20, 30, 40, 50])
    ap.add_argument("--k", type=int, default=500)
    a = ap.parse_args()
    uwyk = G._import_uwyk()
    rows = []
    for nn in a.nodes:
        cfg, _, _ = G.resolve_config("complexmech", nn, "path_TY", 0.0)
        cell = os.path.join(a.root, "complexmech", f"{nn}node", "path_TY", "hide_0.0")
        files = {int(os.path.basename(p)[1:-4]): p for p in glob.glob(os.path.join(cell, "r*.npz"))}
        if not files:
            continue
        n_test = int(np.load(files[min(files)])["Y_do0_raw"].size)
        # run_sweep: ONE sampler per cell, realizations generated in order r = 0, 1, ... (the sampler carries
        # state between realizations, so every realization up to the last needed one is regenerated)
        sampler = uwyk[0](cfg["scm_config"], seed=SEED_BASE * 31 + 17)
        need = [r for r, f in sorted(files.items()) if (lambda c: c[0] and not c[1] and not c[2])(classify(f))]
        for r in range(max(need) + 1 if need else 0):
            try:
                rec, cap = regenerate(cfg, uwyk, sampler, r, n_test)
            except Exception as exc:                            # run_sweep also skipped failures
                rec, cap = None, None
                gen_err = f"{type(exc).__name__}: {exc}"[:200]
            if r not in need:
                continue
            f = files[r]
            torch.manual_seed(12345)
            try:
                res = one(rec, cap, f, a.k) if rec is not None else dict(regen_ok=0, mech_ok=0, error=gen_err)
            except Exception as exc:                            # recorded, never dropped silently
                res = dict(regen_ok=0, mech_ok=0, error=f"{type(exc).__name__}: {exc}"[:200])
            res.update(n_nodes=nn, file=os.path.basename(f))
            rows.append(res)
            print(nn, os.path.basename(f), res, flush=True)
    keys = ["n_nodes", "file", "regen_ok", "mech_ok", "n_units", "n_det", "rho_median", "share_gt099", "error"]
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in keys} for r in rows])
    md = ["# True ρ(x) on ComplexMech, pre-treatment realizations (exact, from UWYK's generator)", "",
          "ρ(x) = corr(Y(1), Y(0) | X = x) by redrawing only Y's noise (shared across arms). `verified` = "
          "regenerated outcomes equal the stored ones AND Y's mechanism reproduces them; only those are summarised. "
          "ρ per realization = median over its query units; `deterministic` = units where no noise reaches Y.", "",
          "| n | pre-treatment realizations | verified | ρ median (min, max) over realizations | "
          "share of units with ρ > 0.99 | deterministic units |", "|---|---|---|---|---|---|"]
    for nn in a.nodes:
        sel = [r for r in rows if r["n_nodes"] == nn]
        ok = [r for r in sel if r.get("regen_ok") and r.get("mech_ok")]
        v = np.array([r["rho_median"] for r in ok if np.isfinite(r.get("rho_median", np.nan))])
        sh = np.array([r["share_gt099"] for r in ok if np.isfinite(r.get("share_gt099", np.nan))])
        det = sum(r.get("n_det", 0) for r in ok); tot = sum(r.get("n_units", 0) for r in ok)
        rng = f"{np.median(v):+.3f} ({v.min():+.3f}, {v.max():+.3f})" if v.size else "—"
        md.append(f"| {nn} | {len(sel)} | {len(ok)} | {rng} | {np.mean(sh):.2f} | {det}/{tot} |"
                  if v.size else f"| {nn} | {len(sel)} | {len(ok)} | — | — | {det}/{tot} |")
    open(a.out + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
