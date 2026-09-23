"""Compare two prediction dumps and fail if they are not the same computation.

Written for submit_density_tauC_repair.sbatch, whose whole problem is that it
back-fills realizations into a shard directory produced MONTHS EARLIER, on a
DIFFERENT CLUSTER. The resume logic in eval_density_tauC.py skips any
realization whose shard exists, which makes back-filling trivial -- and trivial
in the dangerous way, because it trusts the shard and not the config that made
it. Re-running realization 80 with a different checkpoint, a different Do-PFN
library version or a different context seed writes a shard that looks exactly
like its 90 neighbours and is not comparable to any of them.

So the repair job first RE-RUNS ONE REALIZATION THAT ALREADY EXISTS into a
scratch directory and points this script at the pair. Agreement is evidence
that the environment reproduces the original run; disagreement stops the repair
before it contaminates the shard.

WHY THIS AND NOT A FIELD-BY-FIELD CONFIG CHECK. Most of the config IS
recoverable from a dump -- checkpoint paths, n_y0, the model list, the context
seed. One thing is not: Do-PFN's `native` row is the pretrained LIBRARY model,
recorded as dopfn_sources[0] == 'library' with no version, hash or path to
compare. A metadata check cannot see that the package moved under it. Comparing
the actual predicted logits can, and catches everything the metadata check
would have caught as well.

WHY A TOLERANCE, AND WHY THIS ONE. Bit-exactness is the wrong bar: the same
model on a different GPU, or on CPU instead of CUDA, moves logits by ~1e-6
relative through non-associative reductions alone. A DIFFERENT model moves them
by O(1). Those differ by six orders of magnitude, so the threshold is not
delicate -- 1e-4 relative sits in the empty space between them. Anything that
trips it is a real difference, not numerical noise.

Deterministic fields (true_cate, mu0_scaled, mu1_scaled, tau_star_scaled) are
held to a much tighter bar, because they come from the dataset and the seeded
context subsample rather than from a forward pass. A drift there means the
context or the truth loader changed, which invalidates the comparison itself --
so they are reported separately rather than averaged in with the logits.

Usage:
    python benchmarks/eval_graph2d/diff_prediction_shards.py REF.npz NEW.npz
    RTOL=1e-4 python ... diff_prediction_shards.py REF.npz NEW.npz

Exit codes:  0 identical within tolerance   1 differs   2 cannot compare
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Forward-pass outputs: hardware-dependent at the 1e-6 level, model-dependent
# at O(1). See the module docstring for why the gap makes this threshold safe.
RTOL = float(os.environ.get('RTOL', '1e-4'))
# Dataset + seeded-context quantities. These involve no forward pass, so they
# should agree to round-off; a looser bar here would hide a context change.
RTOL_EXACT = float(os.environ.get('RTOL_EXACT', '1e-9'))
_EXACT_KEYS = ('true_cate', 'mu0_scaled', 'mu1_scaled', 'tau_star_scaled',
               'sigma_scaled', 'sigma_raw', 'y_scale', 'y_shift', 'n_context',
               'context_seed')

# Recorded provenance, not computation. These are EXPECTED to differ -- the
# repair runs from a different checkout than the original shard, so absolute
# paths move. The checkpoint BYTES are what matter, and the logits prove those.
_PATH_KEYS = ('ckpt', 'uwyk_ckpt', 'uwyk_cfg', 'dopfn_root', 'causalpfn_root',
              'dopfn_sources', 'causalpfn_sources', 'source_dump')


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    """max |a-b| / max(1, max|a|) -- relative to SCALE, not per-element.

    Per-element relative error is the wrong statistic for logits: one entry
    passing through zero sends it to infinity while the vector is unchanged for
    every practical purpose. Normalising by the array's own scale asks the
    question that matters -- did this prediction move.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = max(1.0, float(np.nanmax(np.abs(a))) if a.size else 1.0)
    return float(np.nanmax(np.abs(a - b))) / denom if a.size else 0.0


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-4], file=sys.stderr)
        print('usage: diff_prediction_shards.py REF.npz NEW.npz', file=sys.stderr)
        return 2
    ref_path, new_path = sys.argv[1], sys.argv[2]
    for p in (ref_path, new_path):
        if not os.path.isfile(p):
            print(f'FATAL: no such dump: {p}', file=sys.stderr)
            return 2

    with np.load(ref_path, allow_pickle=True) as z:
        ref = {k: z[k] for k in z.files}
    with np.load(new_path, allow_pickle=True) as z:
        new = {k: z[k] for k in z.files}

    print(f'[diff] ref = {ref_path}')
    print(f'[diff] new = {new_path}')
    print(f'[diff] rtol = {RTOL:g}  (deterministic fields {RTOL_EXACT:g})')

    only_ref = sorted(set(ref) - set(new))
    only_new = sorted(set(new) - set(ref))
    if only_ref or only_new:
        # A key set mismatch means the two runs did not score the same models,
        # which is a config difference no tolerance can absorb.
        print(f'MISMATCH: key sets differ\n  only in ref: {only_ref}'
              f'\n  only in new: {only_new}', file=sys.stderr)
        return 1

    worst: list[tuple[float, str, float]] = []
    skipped, failures = [], []
    for k in sorted(ref):
        if k in _PATH_KEYS:
            skipped.append(k)
            continue
        a, b = np.asarray(ref[k]), np.asarray(new[k])
        if a.dtype.kind in 'USO' or b.dtype.kind in 'USO':
            # Strings and object arrays: identity or nothing. Model lists and
            # kinds live here, and those must match exactly.
            if not np.array_equal(a, b):
                failures.append(f'{k}: string/object field differs '
                                f'({a.ravel()[:3]} vs {b.ravel()[:3]})')
            continue
        if a.shape != b.shape:
            failures.append(f'{k}: shape {a.shape} vs {b.shape}')
            continue
        tol = RTOL_EXACT if k in _EXACT_KEYS else RTOL
        r = _rel(a, b)
        worst.append((r, k, tol))
        if not np.isfinite(r) or r > tol:
            failures.append(f'{k}: relative deviation {r:.3e} > {tol:g}')

    worst.sort(reverse=True)
    print('\n[diff] largest deviations:')
    for r, k, tol in worst[:8]:
        flag = 'FAIL' if (not np.isfinite(r) or r > tol) else 'ok'
        print(f'   {k:38s} {r:11.3e}  (tol {tol:g})  {flag}')
    if skipped:
        print(f'\n[diff] provenance fields not compared (paths move between '
              f'checkouts): {", ".join(skipped)}')

    if failures:
        print(f'\nMISMATCH: {len(failures)} field(s) differ -- this is NOT the '
              f'same computation.', file=sys.stderr)
        for f in failures[:20]:
            print(f'  {f}', file=sys.stderr)
        print('\nDo NOT back-fill into the existing shard directory. Either fix '
              'the environment (checkpoint, Do-PFN/CausalPFN version, config) '
              'until this passes, or re-run the whole family into a fresh '
              'OUT_ROOT so every realization comes from one configuration.',
              file=sys.stderr)
        return 1

    print(f'\n[diff] OK -- {len(worst)} numeric field(s) agree within tolerance. '
          f'This environment reproduces the original run.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
