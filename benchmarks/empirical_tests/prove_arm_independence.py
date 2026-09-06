"""Test whether the two potential outcomes are conditionally independent given X
on every dataset in the RealCause table (IHDP, ACIC, CPS, PSID).

Why this matters. Our 2D joint head's only structural advantage over UWYK's 1D
head run twice is that it can represent dependence between Y_do(0) and Y_do(1)
at a fixed x. If the benchmark's true joint factorises, that advantage is worth
exactly zero nats there, and any Tier-B/Tier-C loss the joint takes is
structural rather than a modelling failure. See `density_eval_pipeline.md` §3.

WHAT "PROOF" MEANS HERE
-----------------------
Independence cannot be proven, only bounded. A credible claim needs four parts,
and this script produces all four:

  1. An estimate of cross-arm dependence WITH a confidence interval.
  2. An OMNIBUS test, not just a correlation. Zero correlation is not
     independence, and Lalonde earnings have a ~15% point mass at y=0 where a
     purely linear test is blind.
  3. A POWER statement: the smallest dependence the design could have detected.
     "We failed to reject" is worthless without it, so every omnibus test is
     re-run on copula-coupled data at a sweep of known rho.
  4. A DECISION-RELEVANT bound: convert the dependence CI into an upper bound on
     the log-likelihood a joint head could gain. Under a Gaussian copula
     I(Y0;Y1|X) = -0.5*log(1-rho^2) nats, so a bound on rho bounds the prize.
     That number, not the p-value, settles the design question.

EVERY OMNIBUS STATISTIC IS PERMUTATION-CALIBRATED
-------------------------------------------------
The likelihood-ratio statistic G is biased upward at small per-table counts, and
we pool thousands of 100-draw tables, so even a tiny per-table bias becomes many
sigma of apparent "dependence" against the asymptotic chi-square. The first
draft of this script reported exactly that artifact. So the null for every
pooled statistic is obtained by permuting y1 within each unit (which destroys
cross-arm dependence and preserves everything else) and re-deriving the
statistic B times. Reported z-scores are against that empirical null.

TWO DESIGNS, BECAUSE THE DATASETS DIFFER
----------------------------------------
Family M -- "known means" (IHDP, ACIC). Both potential outcomes and both
  conditional means are shipped, so the noise is directly observable:
  eps_t = y_t - mu_t(x). Independence of the arms is exactly independence of
  (eps0, eps1), testable POOLED across all units and realizations.

Family R -- "replicates" (CPS, PSID). No mu is available, but the RealCause
  files are 100 resamples of the SAME units (covariates are byte-identical
  across all 100 CSVs while t/y/y0/y1 are redrawn). Each unit therefore supplies
  100 iid draws from p(y0, y1 | x_i): test per unit, pool across units.

Usage:
    python -u benchmarks/empirical_tests/prove_arm_independence.py
    DATASETS=IHDP,CPS N_PERM=200 python -u benchmarks/.../prove_arm_independence.py
    ACIC_CACHE_DIR=/path/with/zymu_csvs ...   # else downloads from causallib
    OUT_JSON=results_independence/arm_independence.json ...
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy import stats

CAUSALPFN = os.environ.get(
    'CAUSALPFN', '/home/lukez/projects/aip-rgrosse/lukez/CausalPFN')
ACIC_CACHE_DIR = os.environ.get('ACIC_CACHE_DIR', '') or None
DATASETS = [s.strip() for s in
            os.environ.get('DATASETS', 'IHDP,ACIC,CPS,PSID').split(',') if s.strip()]
OUT_JSON = os.environ.get('OUT_JSON', '')
SEED = int(os.environ.get('SEED', '0'))
N_BINS = int(os.environ.get('N_BINS', '4'))       # 100 draws / 4 = 25 per bin exactly
N_PERM = int(os.environ.get('N_PERM', '200'))     # permutation null replicates
POWER_RHO = [float(v) for v in
             os.environ.get('POWER_RHO', '0.01,0.02,0.05,0.10').split(',')]
ALPHA = 0.05

# Two-sided alpha=0.05 at 80% power costs (z_{a/2} + z_beta) = 2.80 sigma.
Z_POWER = stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(0.80)


# ══════════════════════════════════════════════════════════════════════════
# Generic statistics
# ══════════════════════════════════════════════════════════════════════════
def fisher_ci(r: float, n: int, alpha: float = ALPHA):
    """CI and z-score for a Pearson correlation via the Fisher z transform."""
    if n < 4:
        return float('nan'), float('nan'), float('nan')
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    crit = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se)), float(z / se)


def mi_bound_nats(rho_abs: float) -> float:
    """MI of a bivariate Gaussian at correlation rho, in nats.

    Under a Gaussian-copula model of the dependence this is the entire prize
    available to a joint head over independent marginals.
    """
    rho_abs = min(abs(float(rho_abs)), 0.999999)
    return float(-0.5 * np.log(max(1.0 - rho_abs ** 2, 1e-300)))


def rank_bins(x: np.ndarray, n_bins: int, rng) -> np.ndarray:
    """Equal-frequency bins along axis 0, ties broken uniformly at random.

    x: (n, U). Returns int bins in [0, n_bins). Lalonde earnings have a large
    point mass at 0, so plain quantile binning gives wildly unequal bins and a
    mis-calibrated table. Randomized tie-breaking restores exactly equal
    margins under H0. It can only destroy dependence carried inside the ties,
    never create it, which is why `zero_pattern_stat` covers the point mass
    separately.
    """
    n, u = x.shape
    idx = np.broadcast_to(np.arange(n)[:, None], (n, u))
    perm = rng.permuted(idx, axis=0)                     # random order per column
    order = np.argsort(np.take_along_axis(x, perm, 0), axis=0, kind='stable')
    orig = np.take_along_axis(perm, order, 0)            # original row per rank
    ranks = np.empty((n, u), dtype=np.int64)
    np.put_along_axis(ranks, orig, idx, 0)
    return (ranks * n_bins) // n


def pooled_g(b0: np.ndarray, b1: np.ndarray, n_bins: int) -> float:
    """Sum of per-unit likelihood-ratio statistics for independence.

    b0, b1: (n, U) bin labels. Additive across units under H0, which is what
    lets thousands of small tables be pooled into one powerful statistic. The
    scale of this number is meaningless on its own -- it is only interpretable
    against the permutation null.
    """
    n, u = b0.shape
    cells = n_bins * n_bins
    flat = (b0 * n_bins + b1) * u + np.arange(u)[None, :]
    obs = np.bincount(flat.ravel(), minlength=cells * u).reshape(n_bins, n_bins, u)
    rows = obs.sum(1)                                    # (n_bins, U)
    cols = obs.sum(0)
    exp = rows[:, None, :] * cols[None, :, :] / n
    nz = obs > 0
    ratio = np.where(nz, obs / np.maximum(exp, 1e-300), 1.0)
    return float(2.0 * np.sum(np.where(nz, obs * np.log(ratio), 0.0)))


def pooled_rank_table_g(b0: np.ndarray, b1: np.ndarray, n_bins: int) -> float:
    """G for ONE table built from the within-unit rank bins of every unit.

    Pooling thousands of separate 100-draw tables (`pooled_g`) is statistically
    wasteful: each contributes a tiny signal and its own noise. Because the bins
    are within-unit ranks, they are uniform per unit under H0, so the pairs can
    be stacked into a single table with n = n_units * n_draws. That table is the
    empirical copula, and it is the right statistic for "is there a COMMON
    dependence pattern across units" -- far more powerful, at the cost of being
    blind to unit-specific dependence that cancels out (which `pooled_g` and the
    var(z) test cover instead).
    """
    obs = np.bincount((b0.ravel() * n_bins + b1.ravel()),
                      minlength=n_bins * n_bins).reshape(n_bins, n_bins).astype(float)
    n = obs.sum()
    exp = obs.sum(1, keepdims=True) @ obs.sum(0, keepdims=True) / n
    nz = obs > 0
    return float(2.0 * np.sum(obs[nz] * np.log(obs[nz] / np.maximum(exp[nz], 1e-300))))


def zero_pattern_stat(Y0: np.ndarray, Y1: np.ndarray) -> float:
    """Pooled G for the 2x2 table 1[y0==0] x 1[y1==0], per unit.

    Randomized bin ranks cannot see structure inside the point mass, so test it
    directly: does 'earned nothing under control' predict 'earned nothing under
    treatment' for the same person?
    """
    return pooled_g(Y0 == 0.0, Y1 == 0.0, 2)


def permutation_null(stat_fn, Y1_labels, rng, n_perm: int):
    """Null distribution of a pooled statistic under within-unit permutation.

    Permuting y1 within a unit destroys cross-arm dependence while preserving
    both marginals and every small-sample quirk of the statistic -- exactly the
    null we need.
    """
    draws = np.empty(n_perm)
    for b in range(n_perm):
        draws[b] = stat_fn(rng.permuted(Y1_labels, axis=0))
    return draws


def calibrated(obs: float, null: np.ndarray):
    """(z, empirical two-sided p) of `obs` against an empirical null sample."""
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    z = (obs - mu) / sd if sd > 0 else float('nan')
    ge = int(np.sum(np.abs(null - mu) >= abs(obs - mu)))
    return z, (ge + 1) / (null.size + 1), mu, sd


def gaussian_copula_couple(y0: np.ndarray, y1: np.ndarray, rho: float, rng):
    """Reorder y1 along axis 0 to induce ~rho Gaussian-copula dependence.

    Marginals are untouched -- only the pairing changes. Used for the power
    sweep: if the pipeline cannot flag data we KNOW is dependent, its null
    result on the real data means nothing.
    """
    n = y0.shape[0]
    u0 = (stats.rankdata(y0, method='ordinal', axis=0) - 0.5) / n
    g0 = stats.norm.ppf(u0)
    target = rho * g0 + np.sqrt(max(1.0 - rho ** 2, 0.0)) * rng.standard_normal(y0.shape)
    ranks = np.argsort(np.argsort(target, axis=0), axis=0)
    return np.take_along_axis(np.sort(y1, axis=0), ranks, 0)


# ══════════════════════════════════════════════════════════════════════════
# Family M -- known conditional means (IHDP, ACIC)
# ══════════════════════════════════════════════════════════════════════════
def load_ihdp_residuals():
    d = os.path.join(CAUSALPFN, 'benchmarks', 'IHDP')
    e0, e1, tags = [], [], []
    for fname in ('ihdp_npci_1-100.train.npz', 'ihdp_npci_1-100.test.npz'):
        z = np.load(os.path.join(d, fname))
        yf, ycf, t, mu0, mu1 = (z[k].astype(np.float64)
                                for k in ('yf', 'ycf', 't', 'mu0', 'mu1'))
        # yf is the observed arm, ycf the other one; t says which is which.
        y0 = np.where(t > 0.5, ycf, yf)
        y1 = np.where(t > 0.5, yf, ycf)
        e0.append((y0 - mu0).ravel(order='F'))
        e1.append((y1 - mu1).ravel(order='F'))
        tags.append(np.repeat(np.arange(t.shape[1]), t.shape[0]))
    return np.concatenate(e0), np.concatenate(e1), np.concatenate(tags)


def load_acic_residuals():
    import pandas as pd
    url = (lambda i: f'https://raw.githubusercontent.com/BiomedSciAI/causallib/'
                     f'master/causallib/datasets/data/acic_challenge_2016/zymu_{i}.csv')
    e0, e1, tags = [], [], []
    for r in range(10):
        # ACIC is the only dataset here that is not shipped locally. Cache it on
        # first download so a rerun works on a compute node with no outbound
        # network -- otherwise this row of the proof is not reproducible there.
        frame, path = None, None
        if ACIC_CACHE_DIR:
            path = os.path.join(ACIC_CACHE_DIR, f'zymu_{r + 1}.csv')
            if os.path.isfile(path):
                frame = pd.read_csv(path)
        if frame is None:
            frame = pd.read_csv(url(r + 1))
            if path:
                os.makedirs(ACIC_CACHE_DIR, exist_ok=True)
                frame.to_csv(path, index=False)
        frame.columns = ['z', 'y0', 'y1', 'mu0', 'mu1']
        e0.append(frame['y0'].to_numpy(float) - frame['mu0'].to_numpy(float))
        e1.append(frame['y1'].to_numpy(float) - frame['mu1'].to_numpy(float))
        tags.append(np.full(len(frame), r))
    return np.concatenate(e0), np.concatenate(e1), np.concatenate(tags)


def analyse_known_means(name: str, e0, e1, realization, rng):
    # Standardize within realization: sigma differs across realizations, and
    # pooling heteroscedastic blocks would bias the pooled correlation.
    z0, z1 = np.empty_like(e0), np.empty_like(e1)
    for r in np.unique(realization):
        m = realization == r
        z0[m] = (e0[m] - e0[m].mean()) / max(e0[m].std(ddof=1), 1e-12)
        z1[m] = (e1[m] - e1[m].mean()) / max(e1[m].std(ddof=1), 1e-12)

    n = z0.size
    r_p = float(np.corrcoef(z0, z1)[0, 1])
    lo, hi, zscore = fisher_ci(r_p, n)
    r_s = float(stats.spearmanr(z0, z1).statistic)

    # One pooled table (n is large, so no small-count bias) -- still calibrated.
    col0, col1 = z0[:, None], z1[:, None]
    b0 = rank_bins(col0, N_BINS, rng)
    b1 = rank_bins(col1, N_BINS, rng)
    g_obs = pooled_g(b0, b1, N_BINS)
    g_z, g_p, g_mu, g_sd = calibrated(
        g_obs, permutation_null(lambda lab: pooled_g(b0, lab, N_BINS), b1, rng, N_PERM))

    # Power sweep on the same omnibus statistic.
    power = {}
    for rho in POWER_RHO:
        coupled = gaussian_copula_couple(col0, col1, rho, rng)
        gz = (pooled_g(b0, rank_bins(coupled, N_BINS, rng), N_BINS) - g_mu) / g_sd
        rc = float(np.corrcoef(col0.ravel(), coupled.ravel())[0, 1])
        power[f'{rho:g}'] = dict(g_z=float(gz), copula_z=float(gz),
                                 pearson_z=fisher_ci(rc, n)[2])

    rho_mde_linear = float(np.tanh(Z_POWER / np.sqrt(max(n - 3, 1))))
    rho_bound = max(abs(lo), abs(hi))
    return dict(
        dataset=name, design='known-means (pooled residuals)',
        n_pairs=int(n), n_realizations=int(np.unique(realization).size),
        pearson_r=r_p, ci_lo=lo, ci_hi=hi, pearson_z=zscore, spearman_r=r_s,
        g_obs=g_obs, g_null_mean=g_mu, g_null_sd=g_sd, g_z=g_z, g_p=g_p,
        copula_z=g_z, copula_p=g_p,
        rho_mde_linear=rho_mde_linear, rho_bound=rho_bound,
        mi_bound_nats=mi_bound_nats(rho_bound), power=power,
        independent=bool(abs(zscore) < 1.96 and g_p > ALPHA),
    )


# ══════════════════════════════════════════════════════════════════════════
# Family R -- replicate design (CPS, PSID)
# ══════════════════════════════════════════════════════════════════════════
def load_realcause_replicates(kind: str, n_samples: int = 100):
    """Y0, Y1 of shape (n_samples, n_units), plus the covariate-identity check
    that the whole replicate design rests on."""
    import pandas as pd
    root = os.path.join(CAUSALPFN, 'benchmarks', 'realcause_datasets')
    frames = [pd.read_csv(os.path.join(root, f'{kind}_sample{i}.csv'))
              for i in range(n_samples)]
    cov = [c for c in frames[0].columns if c not in ('t', 'y', 'y0', 'y1', 'ite')]
    base = frames[0][cov].to_numpy()
    same_units = all(np.array_equal(base, f[cov].to_numpy()) for f in frames)
    Y0 = np.stack([f['y0'].to_numpy(float) for f in frames])
    Y1 = np.stack([f['y1'].to_numpy(float) for f in frames])
    return Y0, Y1, same_units


def _per_unit_fisher_z(Y0, Y1):
    """Per-unit Pearson r mapped to z ~ N(0,1) under H0. Returns (z, r, keep)."""
    n = Y0.shape[0]
    a, b = Y0 - Y0.mean(0), Y1 - Y1.mean(0)
    sa, sb = Y0.std(0, ddof=1), Y1.std(0, ddof=1)
    keep = (sa > 1e-9) & (sb > 1e-9)
    r = np.zeros(Y0.shape[1])
    r[keep] = ((a * b).sum(0)[keep] / (n - 1)) / (sa[keep] * sb[keep])
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r) * np.sqrt(n - 3), r, keep


def analyse_replicates(name: str, Y0, Y1, same_units, rng):
    n_draws, _ = Y0.shape
    z, r, keep = _per_unit_fisher_z(Y0, Y1)
    Y0, Y1 = Y0[:, keep], Y1[:, keep]
    z = z[keep]
    N = z.size

    # (a) COMMON dependence -- is the average rho zero? Fisher z is additive.
    mean_z = float(z.mean() * np.sqrt(N))
    se_atanh = 1.0 / np.sqrt(N * (n_draws - 3))
    centre = z.mean() / np.sqrt(n_draws - 3)
    rho_common = float(np.tanh(centre))
    ci = (float(np.tanh(centre - 1.96 * se_atanh)),
          float(np.tanh(centre + 1.96 * se_atanh)))

    # (b) HETEROGENEOUS dependence -- unit-specific rho of either sign cancels
    # in the mean but inflates var(z) above its null value of 1.
    var_z = float(z.var(ddof=1))
    var_z_stat = float((var_z - 1.0) / np.sqrt(2.0 / N))
    rms_rho_bound = float(np.sqrt(max(var_z - 1.0 + 1.96 * np.sqrt(2.0 / N), 0.0)
                                  / (n_draws - 3)))

    # (c) OMNIBUS, permutation-calibrated. Catches non-monotone dependence.
    b0 = rank_bins(Y0, N_BINS, rng)
    b1 = rank_bins(Y1, N_BINS, rng)
    g_obs = pooled_g(b0, b1, N_BINS)
    g_z, g_p, g_mu, g_sd = calibrated(
        g_obs, permutation_null(lambda lab: pooled_g(b0, lab, N_BINS), b1, rng, N_PERM))

    # (c2) COPULA omnibus -- one big table of within-unit ranks.
    cop_obs = pooled_rank_table_g(b0, b1, N_BINS)
    cop_z, cop_p, cop_mu, cop_sd = calibrated(
        cop_obs,
        permutation_null(lambda lab: pooled_rank_table_g(b0, lab, N_BINS),
                         b1, rng, N_PERM))

    # (d) POINT MASS, permutation-calibrated.
    z0m, z1m = (Y0 == 0.0), (Y1 == 0.0)
    zg_obs = pooled_g(z0m, z1m, 2)
    zg_z, zg_p, zg_mu, zg_sd = calibrated(
        zg_obs, permutation_null(lambda lab: pooled_g(z0m, lab, 2), z1m, rng, N_PERM))

    # (e) Variance identity: var(tau) == var0 + var1 iff cov == 0.
    v_tau = (Y1 - Y0).var(0, ddof=1)
    v_sum = Y0.var(0, ddof=1) + Y1.var(0, ddof=1)
    ok = v_sum > 1e-12
    ratio = float(v_tau[ok].mean() / v_sum[ok].mean())
    idx = rng.integers(0, int(ok.sum()), (200, int(ok.sum())))
    boot = v_tau[ok][idx].mean(1) / v_sum[ok][idx].mean(1)

    # (f) POWER SWEEP -- the claim is only as strong as what we could have seen.
    power = {}
    for rho in POWER_RHO:
        coupled = gaussian_copula_couple(Y0, Y1, rho, rng)
        zc, _, _ = _per_unit_fisher_z(Y0, coupled)
        bc = rank_bins(coupled, N_BINS, rng)
        gz = (pooled_g(b0, bc, N_BINS) - g_mu) / g_sd
        cz = (pooled_rank_table_g(b0, bc, N_BINS) - cop_mu) / cop_sd
        power[f'{rho:g}'] = dict(pearson_z=float(zc.mean() * np.sqrt(zc.size)),
                                 g_z=float(gz), copula_z=float(cz))

    rho_bound = max(abs(ci[0]), abs(ci[1]))
    return dict(
        dataset=name, design='replicates (per-unit, pooled)',
        n_units=int(N), n_draws=int(n_draws), covariates_identical=bool(same_units),
        rho_common=rho_common, ci_lo=ci[0], ci_hi=ci[1], pearson_z=mean_z,
        var_z=var_z, var_z_stat=var_z_stat, rms_rho_bound=rms_rho_bound,
        g_obs=g_obs, g_null_mean=g_mu, g_null_sd=g_sd, g_z=g_z, g_p=g_p,
        copula_obs=cop_obs, copula_z=cop_z, copula_p=cop_p,
        zero_obs=zg_obs, zero_null_mean=zg_mu, zero_null_sd=zg_sd,
        zero_z=zg_z, zero_p=zg_p,
        var_ratio=ratio, var_ratio_ci=[float(np.percentile(boot, 2.5)),
                                       float(np.percentile(boot, 97.5))],
        rho_mde_linear=float(np.tanh(Z_POWER * se_atanh)),
        rho_bound=max(rho_bound, rms_rho_bound),
        mi_bound_nats=mi_bound_nats(max(rho_bound, rms_rho_bound)),
        power=power,
        independent=bool(abs(mean_z) < 1.96 and g_p > ALPHA and zg_p > ALPHA
                         and cop_p > ALPHA),
    )


# ══════════════════════════════════════════════════════════════════════════
def main():
    results = []
    for name in DATASETS:
        rng = np.random.default_rng(SEED)          # per dataset, reproducible
        print(f'\n{"=" * 74}\n{name}\n{"=" * 74}', flush=True)
        t0 = time.time()
        try:
            if name == 'IHDP':
                res = analyse_known_means(name, *load_ihdp_residuals(), rng)
            elif name == 'ACIC':
                res = analyse_known_means(name, *load_acic_residuals(), rng)
            elif name in ('CPS', 'PSID'):
                kind = 'lalonde_cps' if name == 'CPS' else 'lalonde_psid'
                res = analyse_replicates(name, *load_realcause_replicates(kind), rng)
            else:
                print(f'  unknown dataset {name}; skipping', flush=True)
                continue
        except Exception as exc:                                   # noqa: BLE001
            print(f'  FAILED: {type(exc).__name__}: {exc}', flush=True)
            continue
        res['seconds'] = round(time.time() - t0, 1)
        results.append(res)
        for k, v in res.items():
            if k == 'power':
                for rho, d in v.items():
                    print(f'  power rho={rho:<6s}     pearson_z={d["pearson_z"]:+8.2f}   '
                          f'copula_z={d["copula_z"]:+8.2f}   per_unit_z={d["g_z"]:+8.2f}')
            elif isinstance(v, float):
                print(f'  {k:22s} {v: .6g}')
            else:
                print(f'  {k:22s} {v}')

    if not results:
        return 1

    print(f'\n{"=" * 104}\nSUMMARY -- cross-arm dependence given X\n{"=" * 104}')
    head = (f'{"dataset":8s} {"pairs":>9s} {"rho":>9s} {"95% CI":>20s} '
            f'{"copula z":>9s} {"per-unit z":>11s} {"zero z":>8s} '
            f'{"max nats":>10s}  verdict')
    print(head + '\n' + '-' * len(head))
    for r in results:
        rho = r.get('pearson_r', r.get('rho_common'))
        n = r.get('n_pairs') or r.get('n_units', 0) * r.get('n_draws', 1)
        zz = r.get('zero_z', float('nan'))
        print(f'{r["dataset"]:8s} {n:9d} {rho:+9.5f} '
              f'[{r["ci_lo"]:+.5f},{r["ci_hi"]:+.5f}] {r["copula_z"]:9.2f} '
              f'{r["g_z"]:11.2f} {zz:8.2f} {r["mi_bound_nats"]:10.2e}  '
              f'{"INDEPENDENT" if r["independent"] else "DEPENDENCE FOUND"}')

    worst = max(r['mi_bound_nats'] for r in results)
    print(f'\nUpper bound on what a joint head can gain over independent marginals,')
    print(f'across all tested datasets:  {worst:.2e} nats per query.')
    print('(Gaussian-copula MI at the far end of the correlation CI.)')

    print('\nPower -- copula-omnibus z when a known Gaussian-copula rho is injected:')
    rhos = sorted({k for r in results for k in r['power']}, key=float)
    print('  ' + 'dataset '.ljust(9) + ''.join(f'rho={k:<8s}' for k in rhos))
    for r in results:
        cells = ''.join(f'{r["power"][k]["copula_z"]:>+11.1f}' if k in r['power']
                        else ' ' * 12 for k in rhos)
        print(f'  {r["dataset"]:8s}{cells}')
    print('  (|z| > 1.96 = detected. The smallest detected rho is the omnibus MDE.)')

    if OUT_JSON:
        os.makedirs(os.path.dirname(OUT_JSON) or '.', exist_ok=True)
        with open(OUT_JSON, 'w') as fh:
            json.dump(results, fh, indent=2)
        print(f'\nwrote {OUT_JSON}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
