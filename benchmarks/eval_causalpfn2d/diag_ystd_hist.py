"""Compute histogram of standardised training-Y across many synthetic tasks.

Answers: what fraction of training y_std falls in [-3, +3] vs [-10, +10]?
Justifies (or refutes) the idea of ignoring outer bins at eval.

Uses BackdoorDGPMetaDataset — same generator training uses — and applies
pooled_std standardisation identical to what the 2D model does.

Env:
  CAUSALPFN     path to external/causalpfn (needed for the meta-dataset yaml)
  N_TASKS       tasks to sample                (default 200)
  N_SAMPLES     samples per task               (default 1024)
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch


CAUSALPFN = os.environ['CAUSALPFN']
N_TASKS   = int(os.environ.get('N_TASKS',   200))
N_SAMPLES = int(os.environ.get('N_SAMPLES', 1024))

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, CAUSALPFN)
sys.path.insert(0, CAUSALPFN + '/src')

# Set env vars train_causalpfn_2d expects to build the meta-dataset
os.environ.setdefault('CAUSALPFN_ROOT', CAUSALPFN)
os.environ.setdefault('N_SAMPLES_PER_TASK', str(N_SAMPLES))
os.environ.setdefault('MAX_N_COVARIATES', '50')
os.environ.setdefault('NUM_FEATURES', '50')

from causalpfn.training.priors import BackdoorDGPMetaDataset  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from hydra.utils import instantiate  # noqa: E402


def build_meta_dataset():
    yaml_path = os.path.join(CAUSALPFN, 'conf', 'meta_dataset', 'synthetic_backdoor.yaml')
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f'yaml not found: {yaml_path}')
    cfg = OmegaConf.load(yaml_path)
    cfg.n_samples           = N_SAMPLES
    cfg.max_n_covariates    = 50
    cfg.post_padding_n_cols = 50
    return instantiate(cfg)


def main():
    from torch.utils.data import DataLoader
    print(f'[bootstrap] instantiating BackdoorDGPMetaDataset ({N_TASKS} tasks × {N_SAMPLES} samples)', flush=True)
    ds = build_meta_dataset()
    # BackdoorDGPMetaDataset is an IterableDataset — must iterate via DataLoader
    loader = DataLoader(ds, batch_size=1, num_workers=0)
    it = iter(loader)

    ys_std_all = []
    max_per_task = []
    for t in range(N_TASKS):
        batch = next(it)
        y = np.asarray(batch['y']).reshape(-1).astype(np.float64)  # (1, N) → (N,)
        mu = y.mean(); sd = max(y.std(), 1e-6)
        y_std = (y - mu) / sd
        ys_std_all.append(y_std)
        max_per_task.append(float(np.abs(y_std).max()))
        if (t + 1) % 50 == 0:
            print(f'  sampled {t+1}/{N_TASKS} tasks', flush=True)

    y_all = np.concatenate(ys_std_all)
    N_total = len(y_all)

    print(f'\n══ standardised Y histogram ({N_TASKS} tasks × {N_SAMPLES} pts = {N_total:,d} samples) ══')
    print(f'  min={y_all.min():.2f}  max={y_all.max():.2f}  '
          f'q99={np.quantile(np.abs(y_all), 0.99):.2f}  '
          f'q999={np.quantile(np.abs(y_all), 0.999):.2f}')
    print()

    for lo, hi in [(-1, 1), (-2, 2), (-3, 3), (-4, 4), (-5, 5), (-7, 7), (-10, 10)]:
        in_range = (y_all >= lo) & (y_all <= hi)
        frac = float(np.mean(in_range))
        print(f'  |y_std| in [{lo:>+4d}, {hi:>+4d}]:  {frac*100:7.4f}%   '
              f'({int(frac * N_total):>10,d} of {N_total:,d})')

    frac_outside_10 = float(np.mean((y_all < -10) | (y_all > 10)))
    print(f'\n  |y_std| >  10           :  {frac_outside_10*100:7.4f}%   '
          f'({int(frac_outside_10 * N_total):>10,d} of {N_total:,d})')

    max_per_task = np.array(max_per_task)
    print(f'\n══ per-task max|y_std| distribution ══')
    for q in (0.5, 0.75, 0.9, 0.95, 0.99, 1.0):
        print(f'  quantile {q:.2f}:  {np.quantile(max_per_task, q):6.2f}')
    print(f'\n  # tasks with max|y_std| > 3:   {int(np.sum(max_per_task > 3))}/{N_TASKS}')
    print(f'  # tasks with max|y_std| > 5:   {int(np.sum(max_per_task > 5))}/{N_TASKS}')
    print(f'  # tasks with max|y_std| > 10:  {int(np.sum(max_per_task > 10))}/{N_TASKS}')


if __name__ == '__main__':
    main()
