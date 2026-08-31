"""Probe our training SCM sampler's distribution of n_real (real feature count).

At eval:
  IHDP    n_real = 25
  CPS     n_real = 8
  PSID    n_real = 8
  ACIC    n_real = 50  ← only dataset with n_real == F (no padding)

If the training sampler predominantly produces small n_real, then the
"no padded slots" mask geometry of ACIC is OOD for our model — training
never saw that case often. Testable directly by drawing a few hundred
training batches and printing the histogram.

Uses the same streaming loader the trainer uses (make_streaming_loader),
same env-var config as training. Prints histogram + summary stats.

Env: N_BATCHES (default 200) — how many batches to sample
"""
from __future__ import annotations
import os, sys
from collections import Counter
import numpy as np
import torch


REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
UWYK = os.environ.get('UWYK', os.path.join(os.path.dirname(REPO_SRC), 'external', 'uwyk'))
sys.path.insert(0, UWYK); sys.path.insert(0, UWYK + '/src')

# Import the same loader the trainer uses.
from training.data.PairedInterventionalDataset import make_streaming_loader  # noqa: E402


N_BATCHES = int(os.environ.get('N_BATCHES', 200))
NUM_FEATURES = int(os.environ.get('NUM_FEATURES', 50))


def infer_n_real_from_anc(anc_matrix, F, eps=1e-6):
    """anc_matrix is (max_features+2, max_features+2). Padded feature slots
    have -1 on their entire row/col. Count how many feature-slots are NOT
    fully-padded.
    """
    # Feature slots start at index 2.
    a = anc_matrix.detach().cpu().numpy() if hasattr(anc_matrix, 'detach') else anc_matrix
    n_real = 0
    for i in range(F):
        row = a[2 + i, :]
        # A fully-padded slot has all -1 (or all-negative-close-to-1) in its row.
        # Real slots have a mix of {-1, 0, +1}.
        if not np.all(np.isclose(row, -1.0, atol=eps)):
            n_real += 1
    return n_real


def main():
    print(f'[probe] sampling {N_BATCHES} training batches; NUM_FEATURES={NUM_FEATURES}',
          flush=True)

    loader, _ = make_streaming_loader(
        max_features=NUM_FEATURES,
        n_train_samples=1000,
        n_test_samples=250,
        min_train_split=0.333,
        max_train_split=0.666,
        overlap_threshold=0.01,
        n_samples_per_task=2048,
        max_n_covariates=NUM_FEATURES - 2,   # matches trainer defaults
        seed=42,
        batch_size=4,
        num_workers=2,
    )

    counts = Counter()
    for i, batch in enumerate(loader):
        if i >= N_BATCHES:
            break
        anc = batch['anc_matrix']  # (B, F+2, F+2)
        for b in range(anc.shape[0]):
            n_real = infer_n_real_from_anc(anc[b], NUM_FEATURES)
            counts[n_real] += 1

    total = sum(counts.values())
    print(f'\n[probe] total tasks sampled: {total}')
    print(f'[probe] n_real histogram:')
    for n in sorted(counts.keys()):
        pct = 100 * counts[n] / total
        bar = '#' * int(pct)
        print(f'  n_real={n:3d}  count={counts[n]:5d}  ({pct:5.1f}%)  {bar}')

    # ACIC-relevant summary
    at_max = counts.get(NUM_FEATURES, 0)
    print(f'\n[probe] tasks with n_real == NUM_FEATURES ({NUM_FEATURES}): '
          f'{at_max}/{total}  ({100*at_max/total:.1f}%)')
    print(f'[probe] tasks with n_real >= {NUM_FEATURES-5}: '
          f'{sum(c for n, c in counts.items() if n >= NUM_FEATURES-5)}/{total}')


if __name__ == '__main__':
    main()
