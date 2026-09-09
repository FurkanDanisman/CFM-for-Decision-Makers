"""Measure raw and conditional Y0/Y1 dependence in IHDP, ACIC, CPS and PSID.

Raw Pearson/Spearman correlations describe the paired outcomes actually stored
in the benchmark files, pooled and separately for each realization. They can
be large just because both conditional means vary with X. Pooled raw results
mix realizations (and, for CPS/PSID, repeated units), so they are descriptive:
no iid p-value or confidence interval is attached to them.

IHDP / ACIC: subtract the supplied mu0/mu1 from those SAME sampled outcomes.
The cited generators use separate Gaussian draws with sigma=1 in raw units:
  IHDP: https://github.com/vdorie/npci/blob/master/examples/ihdp_sim/data.R
  ACIC: https://github.com/vdorie/aciccomp/blob/master/2016/R/dgp.R#L211
  noise: https://github.com/vdorie/aciccomp/blob/master/2016/R/constants.R#L9
Thus Y0 is independent of Y1 given X AND the fixed realization's parameters
by generator construction. Marginal densities alone would not imply this;
the separate noise draws specify the coupling. Residual tests audit the saved
samples against that construction. Use the known scale, not a fitted scale
that would hide a mismatch. Pooling residuals is justified by this common
additive noise law, not by knowledge of the means alone. Per-realization
tables also check for dependence patterns that cancel when pooled.

CPS / PSID: each file must contain the same units in the same order. Assuming
the files are independent draws from the same fitted RealCause generator,
test dependence across draws at each fixed unit. Covariate identity is checked
and required; it does not itself establish iid resampling. Rank-table and
zero-mass tests supplement correlation for non-Gaussian earnings.

G statistics use upper-tail permutation p-values, with permutations restricted
to a realization (IHDP/ACIC) or a unit (CPS/PSID). Dataset-level decisions use
Bonferroni correction across the reported dependence tests. Failure to reject
is NOT proof of independence. Fixed-bin tables can miss within-bin structure.
Fisher intervals and the linear 80%-power MDE are Gaussian approximations,
especially for CPS/PSID. One injected sample per rho is only a sensitivity
check, not an empirical power estimate or an omnibus MDE.

The Gaussian MI equivalent is illustrative: -log(1-rho^2)/2 is valid for a
bivariate Gaussian (or latent Gaussian copula correlation). Raw Pearson rho
does not supply a general MI bound for non-Gaussian or mixed marginals.

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
DATASETS = [s.strip().upper() for s in
            os.environ.get('DATASETS', 'IHDP,ACIC,CPS,PSID').split(',') if s.strip()]
OUT_JSON = os.environ.get('OUT_JSON', '')
SEED = int(os.environ.get('SEED', '0'))
N_BINS = int(os.environ.get('N_BINS', '4'))       # 100 draws / 4 = 25 per bin exactly
N_PERM = int(os.environ.get('N_PERM', '200'))     # permutation null replicates
POWER_RHO = [float(v) for v in
             os.environ.get('POWER_RHO', '0.01,0.02,0.05,0.10').split(',')]
ALPHA = 0.05
GENERATOR_SIGMA_RAW = {'IHDP': 1.0, 'ACIC': 1.0}

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


def gaussian_mi_equivalent_nats(rho_abs: float) -> float:
    """Gaussian MI at this rho; not a general dependence/likelihood bound."""
    rho_abs = min(abs(float(rho_abs)), 0.999999)
    return float(-0.5 * np.log(max(1.0 - rho_abs ** 2, 1e-300)))


def correlation_summary(y0, y1):
    """Descriptive correlations only; preserve the original outcome pairing."""
    a, b = np.asarray(y0).ravel(), np.asarray(y1).ravel()
    if a.shape != b.shape or a.size < 4:
        raise ValueError('Expected at least four aligned outcome pairs')
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('Outcome pairs must be finite')
    variable = np.ptp(a) > 0 and np.ptp(b) > 0
    return dict(n_pairs=int(a.size),
                pearson_r=float(np.corrcoef(a, b)[0, 1]) if variable else None,
                spearman_r=float(stats.spearmanr(a, b).statistic) if variable else None)


def dependence_decision(pvalues):
    """Control false positives across the dependence tests within a dataset."""
    cutoff = ALPHA / len(pvalues)
    rejected = [key for key, p in pvalues.items() if p <= cutoff]
    return dict(dependence_detected=bool(rejected), rejected_tests=rejected,
                dependence_pvalues=pvalues, decision_p_cutoff=cutoff,
                decision_rule='Bonferroni within dataset at alpha=0.05')


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


def permutation_null(stat_fn, Y1_labels, rng, n_perm: int, groups=None):
    """Null distribution of a pooled statistic under within-unit permutation.

    Permuting y1 within a unit destroys cross-arm dependence while preserving
    both marginals and every small-sample quirk of the statistic -- exactly the
    null we need.
    """
    blocks = ([np.flatnonzero(groups == r) for r in np.unique(groups)]
              if groups is not None else None)
    draws = np.empty(n_perm)
    for b in range(n_perm):
        if blocks is None:
            shuffled = rng.permuted(Y1_labels, axis=0)
        else:
            shuffled = np.empty_like(Y1_labels)
            for idx in blocks:
                shuffled[idx] = rng.permuted(Y1_labels[idx], axis=0)
        draws[b] = stat_fn(shuffled)
    return draws


def calibrated(obs: float, null: np.ndarray, alternative='greater'):
    """Empirical p-value; large G/variance indicates dependence (upper tail).

    A signed correlation statistic instead uses two tails about zero.
    The z-score is descriptive: G's permutation null need not be Gaussian.
    """
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    z = (obs - mu) / sd if sd > 0 else 0.0
    if alternative == 'greater':
        ge = int(np.sum(null >= obs))
    elif alternative == 'two-sided':
        ge = int(np.sum(np.abs(null) >= abs(obs)))
    else:
        raise ValueError(f'Unknown alternative: {alternative}')
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
def load_ihdp_outcomes():
    """Return paired raw y0/y1, oracle means and realization IDs, both splits."""
    d = os.path.join(CAUSALPFN, 'benchmarks', 'IHDP')
    columns = [[] for _ in range(5)]
    for fname in ('ihdp_npci_1-100.train.npz', 'ihdp_npci_1-100.test.npz'):
        with np.load(os.path.join(d, fname)) as z:
            yf, ycf, t, mu0, mu1 = (z[k].astype(np.float64)
                                    for k in ('yf', 'ycf', 't', 'mu0', 'mu1'))
        if t.ndim != 2 or any(a.shape != t.shape for a in (yf, ycf, mu0, mu1)):
            raise ValueError('IHDP columns must have matching (unit, realization) shapes')
        if not np.isin(t, [0, 1]).all():
            raise ValueError('IHDP treatment must be binary')
        # yf is the observed arm, ycf the other one; t says which is which.
        y0 = np.where(t > 0.5, ycf, yf)
        y1 = np.where(t > 0.5, yf, ycf)
        for values, array in zip(columns[:4], (y0, y1, mu0, mu1)):
            values.append(array.ravel(order='F'))
        columns[4].append(np.repeat(np.arange(t.shape[1]), t.shape[0]))
    return tuple(np.concatenate(values) for values in columns)


def load_acic_outcomes():
    import pandas as pd
    url = (lambda i: f'https://raw.githubusercontent.com/BiomedSciAI/causallib/'
                     f'master/causallib/datasets/data/acic_challenge_2016/zymu_{i}.csv')
    columns = [[] for _ in range(5)]
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
        if not {'z', 'y0', 'y1', 'mu0', 'mu1'}.issubset(frame.columns):
            raise ValueError('ACIC files must contain z,y0,y1,mu0,mu1 columns')
        for values, key in zip(columns[:4], ('y0', 'y1', 'mu0', 'mu1')):
            values.append(frame[key].to_numpy(float))
        columns[4].append(np.full(len(frame), r))
    return tuple(np.concatenate(values) for values in columns)


def grouped_g(b0, b1, group, n_groups, n_bins):
    """Sum G over realization tables, retaining dependence of opposite signs."""
    flat = (b0 * n_bins + b1) * n_groups + group
    obs = np.bincount(flat, minlength=n_bins * n_bins * n_groups)
    obs = obs.reshape(n_bins, n_bins, n_groups).astype(float)
    total = obs.sum(axis=(0, 1))
    exp = obs.sum(1)[:, None, :] * obs.sum(0)[None, :, :] / total
    nz = obs > 0
    return float(2 * np.sum(obs[nz] * np.log(obs[nz] / exp[nz])))


def normal_noise_diagnostic(e, sigma):
    """Check the specified N(0, sigma^2), with no parameters fitted to e."""
    ks = stats.kstest(e / sigma, 'norm')
    return dict(mean_raw=float(e.mean()), sd_raw=float(e.std(ddof=1)),
                expected_mean_raw=0.0, expected_sd_raw=sigma,
                normal_ks_stat=float(ks.statistic), normal_ks_p=float(ks.pvalue))


def analyse_known_means(name: str, y0, y1, mu0, mu1, realization, rng):
    y0, y1, mu0, mu1, realization = map(np.asarray, (y0, y1, mu0, mu1, realization))
    if y0.ndim != 1 or any(a.shape != y0.shape for a in (y1, mu0, mu1, realization)):
        raise ValueError('Known-mean outcomes, means and IDs must be aligned vectors')
    raw = correlation_summary(y0, y1)
    e0, e1 = y0 - mu0, y1 - mu1
    if not (np.isfinite(e0).all() and np.isfinite(e1).all()):
        raise ValueError('Oracle means and residuals must be finite')
    sigma = GENERATOR_SIGMA_RAW[name]
    z0, z1 = e0 / sigma, e1 / sigma
    ids, group = np.unique(realization, return_inverse=True)
    per_realization = []
    for r in ids:
        m = realization == r
        raw_r = correlation_summary(y0[m], y1[m])
        noise_r = correlation_summary(e0[m], e1[m])
        per_realization.append(dict(realization=int(r), raw_outcomes=raw_r,
                                    residuals=noise_r,
                                    noise0=normal_noise_diagnostic(e0[m], sigma),
                                    noise1=normal_noise_diagnostic(e1[m], sigma)))

    n = z0.size
    r_p = float(np.corrcoef(z0, z1)[0, 1])
    if not np.isfinite(r_p):
        raise ValueError('Residual correlation is undefined: an arm is constant')
    lo, hi, zscore = fisher_ci(r_p, n)
    r_s = float(stats.spearmanr(z0, z1).statistic)

    # Oracle PIT bins: edges are fixed by N(0,1), rather than fitted ranks.
    # Permute within realizations so between-realization differences survive.
    col0, col1 = z0[:, None], z1[:, None]
    edges = stats.norm.ppf(np.arange(1, N_BINS) / N_BINS)
    b0, b1 = np.searchsorted(edges, col0), np.searchsorted(edges, col1)
    g_obs = pooled_g(b0, b1, N_BINS)
    g_null = permutation_null(lambda lab: pooled_g(b0, lab, N_BINS),
                              b1, rng, N_PERM, groups=realization)
    g_z, g_p, g_mu, g_sd = calibrated(g_obs, g_null)
    by_realization = lambda lab: grouped_g(b0.ravel(), lab.ravel(), group,
                                          len(ids), N_BINS)
    rg_obs = by_realization(b1)
    rg_null = permutation_null(by_realization, b1, rng, N_PERM, groups=realization)
    rg_z, rg_p, _, _ = calibrated(rg_obs, rg_null)

    # One injected sample per rho: a sensitivity check, not estimated power.
    power = {}
    for rho in POWER_RHO:
        coupled = np.empty_like(col1)
        # Couple within each realization, preserving the permutation blocks.
        for r in ids:
            m = realization == r
            coupled[m] = gaussian_copula_couple(col0[m], col1[m], rho, rng)
        gz, gp, _, _ = calibrated(pooled_g(b0, np.searchsorted(edges, coupled), N_BINS),
                                  g_null)
        rc = float(np.corrcoef(col0.ravel(), coupled.ravel())[0, 1])
        power[f'{rho:g}'] = dict(g_z=float(gz), copula_z=float(gz),
                                 g_p=gp, copula_p=gp, pearson_z=fisher_ci(rc, n)[2])

    rho_mde_linear = float(np.tanh(Z_POWER / np.sqrt(max(n - 3, 1))))
    rho_bound = max(abs(lo), abs(hi))
    return dict(
        dataset=name, design='known-means (pooled residuals)',
        n_pairs=int(n), n_realizations=int(ids.size),
        raw_outcomes=raw, per_realization=per_realization,
        generator_truth=dict(sigma_raw=sigma, conditional_rho=0.0,
                             conditional_independence=True,
                             conditioning='X and fixed realization parameters'),
        noise0=normal_noise_diagnostic(e0, sigma),
        noise1=normal_noise_diagnostic(e1, sigma),
        residual_tau_variance=float((e1 - e0).var(ddof=1)),
        expected_residual_tau_variance=2 * sigma**2,
        pearson_r=r_p, ci_lo=lo, ci_hi=hi, pearson_z=zscore, spearman_r=r_s,
        g_obs=g_obs, g_null_mean=g_mu, g_null_sd=g_sd, g_z=g_z, g_p=g_p,
        copula_z=g_z, copula_p=g_p,
        realization_g_obs=rg_obs, realization_g_z=rg_z, realization_g_p=rg_p,
        rho_mde_linear=rho_mde_linear, rho_bound=rho_bound,
        gaussian_mi_equivalent_nats=gaussian_mi_equivalent_nats(rho_bound), power=power,
        **dependence_decision(dict(pearson=float(2 * stats.norm.sf(abs(zscore))),
                                   pooled_g=g_p, realization_g=rg_p)),
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
    if not cov:
        raise ValueError('Replicate files need covariates to verify unit alignment')
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
    keep = (sa > 0) & (sb > 0)
    r = np.zeros(Y0.shape[1])
    r[keep] = ((a * b).sum(0)[keep] / (n - 1)) / (sa[keep] * sb[keep])
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r) * np.sqrt(n - 3), r, keep


def analyse_replicates(name: str, Y0, Y1, same_units, rng):
    if not same_units:
        raise ValueError('Replicate covariates differ: cannot condition on the same unit')
    Y0, Y1 = np.asarray(Y0, dtype=float), np.asarray(Y1, dtype=float)
    if Y0.ndim != 2 or Y0.shape != Y1.shape or min(Y0.shape) < 4:
        raise ValueError('Replicates require matching (draw, unit) arrays, each axis >= 4')
    raw = correlation_summary(Y0, Y1)
    per_realization = [dict(realization=i, raw_outcomes=correlation_summary(a, b))
                       for i, (a, b) in enumerate(zip(Y0, Y1))]
    n_draws, n_units_total = Y0.shape
    z, r, keep = _per_unit_fisher_z(Y0, Y1)
    Y0, Y1 = Y0[:, keep], Y1[:, keep]
    z = z[keep]
    N = z.size
    if N < 2:
        raise ValueError('Need at least two units with nonconstant outcomes in both arms')
    r = r[keep]
    _, rank_r, _ = _per_unit_fisher_z(stats.rankdata(Y0, axis=0),
                                     stats.rankdata(Y1, axis=0))

    # (a) COMMON dependence -- is the average rho zero? Fisher z is additive.
    mean_z = float(z.mean() * np.sqrt(N))
    se_atanh = 1.0 / np.sqrt(N * (n_draws - 3))
    centre = z.mean() / np.sqrt(n_draws - 3)
    rho_common = float(np.tanh(centre))
    ci = (float(np.tanh(centre - 1.96 * se_atanh)),
          float(np.tanh(centre + 1.96 * se_atanh)))

    # (b) HETEROGENEOUS dependence -- opposing signs can cancel in the mean.
    # Calibrate both correlation statistics on the actual earnings marginals;
    # var(z)=1 is only an approximation for Gaussian data.
    var_z = float(z.var(ddof=1))
    corr_null = np.empty((N_PERM, 2))
    for i in range(N_PERM):
        zp, _, _ = _per_unit_fisher_z(Y0, rng.permuted(Y1, axis=0))
        corr_null[i] = zp.mean() * np.sqrt(N), zp.var(ddof=1)
    _, pearson_p, _, _ = calibrated(mean_z, corr_null[:, 0], alternative='two-sided')
    var_z_stat, var_z_p, _, _ = calibrated(var_z, corr_null[:, 1])

    # (c) OMNIBUS, permutation-calibrated. Catches non-monotone dependence.
    b0 = rank_bins(Y0, N_BINS, rng)
    b1 = rank_bins(Y1, N_BINS, rng)
    g_obs = pooled_g(b0, b1, N_BINS)
    g_null = permutation_null(lambda lab: pooled_g(b0, lab, N_BINS), b1, rng, N_PERM)
    g_z, g_p, g_mu, g_sd = calibrated(g_obs, g_null)

    # (c2) COPULA omnibus -- one big table of within-unit ranks.
    cop_obs = pooled_rank_table_g(b0, b1, N_BINS)
    cop_null = permutation_null(lambda lab: pooled_rank_table_g(b0, lab, N_BINS),
                                b1, rng, N_PERM)
    cop_z, cop_p, cop_mu, cop_sd = calibrated(cop_obs, cop_null)

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

    # (f) Injected-dependence sensitivity check (one draw per rho).
    power = {}
    for rho in POWER_RHO:
        coupled = gaussian_copula_couple(Y0, Y1, rho, rng)
        zc, _, _ = _per_unit_fisher_z(Y0, coupled)
        bc = rank_bins(coupled, N_BINS, rng)
        gz, gp, _, _ = calibrated(pooled_g(b0, bc, N_BINS), g_null)
        cz, cp, _, _ = calibrated(pooled_rank_table_g(b0, bc, N_BINS), cop_null)
        power[f'{rho:g}'] = dict(pearson_z=float(zc.mean() * np.sqrt(zc.size)),
                                 g_z=float(gz), copula_z=float(cz), g_p=gp, copula_p=cp)

    rho_bound = max(abs(ci[0]), abs(ci[1]))
    return dict(
        dataset=name, design='replicates (per-unit, pooled)',
        n_units=int(N), n_draws=int(n_draws), covariates_identical=bool(same_units),
        n_units_total=int(n_units_total), n_units_constant=int(n_units_total - N),
        raw_outcomes=raw, per_realization=per_realization,
        per_unit_pearson_quantiles=dict(zip(('min', 'q25', 'median', 'q75', 'max'),
                                            map(float, np.quantile(r, [0, .25, .5, .75, 1])))),
        per_unit_spearman_mean=float(rank_r.mean()),
        ci_method='Approximate common Fisher correlation; Gaussian iid approximation',
        rho_common=rho_common, ci_lo=ci[0], ci_hi=ci[1], pearson_z=mean_z,
        pearson_p=pearson_p, var_z=var_z, var_z_stat=var_z_stat, var_z_p=var_z_p,
        g_obs=g_obs, g_null_mean=g_mu, g_null_sd=g_sd, g_z=g_z, g_p=g_p,
        copula_obs=cop_obs, copula_z=cop_z, copula_p=cop_p,
        zero_obs=zg_obs, zero_null_mean=zg_mu, zero_null_sd=zg_sd,
        zero_z=zg_z, zero_p=zg_p,
        var_ratio=ratio, var_ratio_ci=[float(np.percentile(boot, 2.5)),
                                       float(np.percentile(boot, 97.5))],
        rho_mde_linear=float(np.tanh(Z_POWER * se_atanh)),
        rho_bound=rho_bound,
        gaussian_mi_equivalent_nats=gaussian_mi_equivalent_nats(rho_bound),
        power=power,
        **dependence_decision(dict(pearson=pearson_p, heterogeneous_correlation=var_z_p,
                                   per_unit_g=g_p, copula_g=cop_p, zero_g=zg_p)),
    )


# ══════════════════════════════════════════════════════════════════════════
def main():
    if N_PERM < 2 or N_BINS < 2 or any(not -1 <= rho <= 1 for rho in POWER_RHO):
        raise ValueError('Require N_PERM >= 2, N_BINS >= 2, and POWER_RHO in [-1, 1]')
    results = []
    failed = []
    for name in DATASETS:
        rng = np.random.default_rng(SEED)          # per dataset, reproducible
        print(f'\n{"=" * 74}\n{name}\n{"=" * 74}', flush=True)
        t0 = time.time()
        try:
            if name == 'IHDP':
                res = analyse_known_means(name, *load_ihdp_outcomes(), rng)
            elif name == 'ACIC':
                res = analyse_known_means(name, *load_acic_outcomes(), rng)
            elif name in ('CPS', 'PSID'):
                kind = 'lalonde_cps' if name == 'CPS' else 'lalonde_psid'
                res = analyse_replicates(name, *load_realcause_replicates(kind), rng)
            else:
                raise ValueError(f'Unknown dataset {name}')
        except Exception as exc:                                   # noqa: BLE001
            print(f'  FAILED: {type(exc).__name__}: {exc}', flush=True)
            failed.append(name)
            continue
        res['config'] = dict(seed=SEED, n_perm=N_PERM, n_bins=N_BINS,
                             alpha=ALPHA, injected_rhos=POWER_RHO)
        res['seconds'] = round(time.time() - t0, 1)
        results.append(res)
        for k, v in res.items():
            if k == 'per_realization':
                print(f'  {k:22s} {len(v)} rows (saved in OUT_JSON)')
            elif k == 'power':
                for rho, d in v.items():
                    print(f'  injected rho={rho:<6s}  pearson_z={d["pearson_z"]:+8.2f}   '
                          f'copula_z={d["copula_z"]:+8.2f}   per_unit_z={d["g_z"]:+8.2f}')
            elif isinstance(v, float):
                print(f'  {k:22s} {v: .6g}')
            else:
                print(f'  {k:22s} {v}')

    if not results:
        return 1

    print('\nRAW SAMPLED OUTCOMES -- descriptive correlation across units and realizations')
    print(f'{"dataset":8s} {"pairs":>9s} {"Pearson r":>12s} {"Spearman r":>12s}')
    for r in results:
        raw = r['raw_outcomes']
        print(f'{r["dataset"]:8s} {raw["n_pairs"]:9d} '
              f'{raw["pearson_r"]:+12.5f} {raw["spearman_r"]:+12.5f}')
    print('These raw correlations include variation in the conditional means with X.')

    print(f'\n{"=" * 104}\nCONDITIONAL DEPENDENCE AUDIT\n{"=" * 104}')
    head = (f'{"dataset":8s} {"pairs":>9s} {"rho":>9s} {"95% CI":>20s} '
            f'{"copula z":>9s} {"per-unit z":>11s} {"zero z":>8s} '
            ' verdict')
    print(head + '\n' + '-' * len(head))
    for r in results:
        rho = r.get('pearson_r', r.get('rho_common'))
        n = r.get('n_pairs') or r.get('n_units', 0) * r.get('n_draws', 1)
        zz = r.get('zero_z', float('nan'))
        print(f'{r["dataset"]:8s} {n:9d} {rho:+9.5f} '
              f'[{r["ci_lo"]:+.5f},{r["ci_hi"]:+.5f}] {r["copula_z"]:9.2f} '
              f'{r["g_z"]:11.2f} {zz:8.2f}  '
              f'{"DEPENDENCE DETECTED" if r["dependence_detected"] else "NOT REJECTED"}')
    print('IHDP/ACIC rho: oracle residual Pearson; CPS/PSID rho: common per-unit Fisher estimate.')
    print('CIs are Gaussian approximations. NOT REJECTED does not establish independence.')
    print('Decisions use empirical p-values (Gaussian Pearson for IHDP/ACIC), with')
    print('Bonferroni correction within each dataset. No correction across datasets.')
    print('Gaussian MI equivalents in the detailed output are not general MI upper bounds.')

    print('\nSensitivity -- copula-omnibus p when Gaussian-copula dependence is injected:')
    rhos = sorted({k for r in results for k in r['power']}, key=float)
    print('  ' + 'dataset '.ljust(9) + ''.join(f'rho={k:<8s}' for k in rhos))
    for r in results:
        cells = ''.join(f'{r["power"][k]["copula_p"]:>11.4f}' if k in r['power']
                        else ' ' * 12 for k in rhos)
        print(f'  {r["dataset"]:8s}{cells}')
    print('  One sample per rho; this does not estimate detection probability or omnibus MDE.')

    if OUT_JSON:
        os.makedirs(os.path.dirname(OUT_JSON) or '.', exist_ok=True)
        with open(OUT_JSON, 'w') as fh:
            json.dump(results, fh, indent=2, allow_nan=False)
        print(f'\nwrote {OUT_JSON}')
    if failed:
        print(f'\nIncomplete run; failed datasets: {", ".join(failed)}')
    return int(bool(failed))


if __name__ == '__main__':
    sys.exit(main())
