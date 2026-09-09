"""Generator truth for the Tier-C IHDP / ACIC density pipeline.

The supported files are IHDP's ihdp_npci_1-100.{train,test}.npz and
causallib's ACIC 2016 zymu_1.csv through zymu_10.csv. Their generators draw
independent Gaussian arm noise with standard deviation 1 in raw units:
  IHDP: https://github.com/vdorie/npci/blob/master/examples/ihdp_sim/data.R
        sigma.y = 1.0; separate rnorm calls for y.0 and y.1.
  ACIC: https://github.com/vdorie/aciccomp/blob/master/2016/R/constants.R
        RSP_SIGMA_Y = 1; separate rnorm calls in R/dgp.R.

Training factual residuals estimate that scale for diagnostics only. Truth
uses the affine transform returned by the model harness for its actual
context; this module never fits a separate outcome scaler.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np


GENERATOR_SIGMA_RAW = {'IHDP': 1.0, 'ACIC': 1.0}
ACIC_ZY_URL = ('https://raw.githubusercontent.com/BiomedSciAI/causallib/master/'
               'causallib/datasets/data/acic_challenge_2016/zymu_{}.csv')


def harness_y_affine(y_offset: float, y_span: float, mode: str):
    """Decode H._scale_y's return values as raw = scaled * scale + shift.

    The harness returns twice the inverse effect scale in its third slot.
    Its second slot is the raw mean for std, but the raw minimum for minmax.
    """
    scale = float(y_span) / 2.0
    if mode == 'std':
        shift = float(y_offset)
    elif mode == 'minmax':
        shift = float(y_offset) + scale
    else:
        raise ValueError(f'Unsupported density outcome scaling: {mode!r}')
    if not np.isfinite(shift) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Outcome shift must be finite and scale finite and positive')
    return shift, scale


@dataclass
class DensityTruth:
    mu0_scaled: np.ndarray
    mu1_scaled: np.ndarray
    tau_star_scaled: np.ndarray
    sigma_raw: float
    sigma_scaled: float
    sigma_residual_raw: float
    sigma_residual_scaled: float

    def noise_metadata(self):
        """Persist the scoring convention and diagnostics in both NPZ outputs."""
        return dict(truth_noise_source='generator', sigma_raw=self.sigma_raw,
                    sigma_scaled=self.sigma_scaled,
                    sigma_residual_raw=self.sigma_residual_raw,
                    sigma_residual_scaled=self.sigma_residual_scaled)


def load_density_truth(dataset: str, r: int, *, y_shift: float, y_scale: float,
                       causalpfn_dir: str, acic_cache_dir: str) -> DensityTruth:
    """Load means, paired test outcomes and full-training residual diagnostics.

    Test ordering matches CausalPFN: NPZ order for IHDP; the final 10% of
    default_rng(42 + r).permutation(n) for ACIC. All columns for the test
    targets use the same indices. Context subsampling affects y_shift and
    y_scale only; residual diagnostics always use the full training split.
    """
    if dataset == 'IHDP':
        if not 0 <= r < 100:
            raise ValueError('IHDP density truth supports realizations 0..99')
        root = Path(causalpfn_dir) / 'benchmarks' / 'IHDP'
        with np.load(root / 'ihdp_npci_1-100.train.npz') as train:
            t = train['t'][..., r].reshape(-1)
            yf = train['yf'][..., r].astype(np.float64).reshape(-1)
            m0 = train['mu0'][..., r].astype(np.float64).reshape(-1)
            m1 = train['mu1'][..., r].astype(np.float64).reshape(-1)
            residuals = yf - np.where(t > 0.5, m1, m0)
        with np.load(root / 'ihdp_npci_1-100.test.npz') as test:
            t = test['t'][..., r].reshape(-1)
            yf = test['yf'][..., r].astype(np.float64).reshape(-1)
            ycf = test['ycf'][..., r].astype(np.float64).reshape(-1)
            y0, y1 = np.where(t == 0, yf, ycf), np.where(t == 1, yf, ycf)
            mu0 = test['mu0'][..., r].astype(np.float64).reshape(-1)
            mu1 = test['mu1'][..., r].astype(np.float64).reshape(-1)
    elif dataset == 'ACIC':
        if not 0 <= r < 10:
            raise ValueError('ACIC density truth supports realizations 0..9')
        path = Path(acic_cache_dir) / f'zymu_{r + 1}.csv'
        with (path.open('rb') if path.is_file()
              else urlopen(ACIC_ZY_URL.format(r + 1))) as source:
            sim = np.loadtxt(source, delimiter=',', skiprows=1, ndmin=2)
        if sim.shape[1] != 5:
            raise ValueError('ACIC truth expects columns z,y0,y1,mu0,mu1')
        t, y0, y1, mu0, mu1 = sim.T
        perm = np.random.default_rng(42 + r).permutation(len(sim))
        split = int(len(sim) * 0.9)
        train_idx, test_idx = perm[:split], perm[split:]
        residuals = np.where(t == 1, y1 - mu1, y0 - mu0)[train_idx]
        mu0, mu1 = mu0[test_idx], mu1[test_idx]
        y0, y1 = y0[test_idx], y1[test_idx]
    else:
        raise ValueError(f'Density truth supports IHDP and ACIC, got {dataset!r}')

    sigma_raw = GENERATOR_SIGMA_RAW[dataset]
    sigma_residual = float(np.std(residuals, ddof=1))
    return DensityTruth(
        mu0_scaled=(mu0 - y_shift) / y_scale,
        mu1_scaled=(mu1 - y_shift) / y_scale,
        tau_star_scaled=(y1 - y0) / y_scale,
        sigma_raw=sigma_raw, sigma_scaled=sigma_raw / y_scale,
        sigma_residual_raw=sigma_residual,
        sigma_residual_scaled=sigma_residual / y_scale,
    )
