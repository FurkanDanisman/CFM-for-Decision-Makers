"""Patch UWYK's PreprocessingGraphConditionedPFN.py to fix a 2D-vs-1D
shape bug in the clustered + prediction_type='sample' path.

Bug: at the top of the clustered branch, the wrapper allocates
    preds = np.zeros(n_test_original, dtype=np.float32)
unconditionally. But when prediction_type='sample', _predict_single_cluster
returns (n_test, num_samples). Then `preds[test_mask] = cluster_preds`
raises "NumPy boolean array indexing assignment requires a 0 or
1-dimensional input, input has 2 dimensions".

Fix: allocate with shape (n_test_original, num_samples) when
prediction_type='sample'.

Usage:
    python patch_uwyk_wrapper.py $DEPLOY_ROOT/external/uwyk/src/models/PreprocessingGraphConditionedPFN.py

Idempotent: no-op if already patched. Prints [patched] / [skip] / [error].
"""
import sys

BUGGY = "preds = np.zeros(n_test_original, dtype=np.float32)"

FIXED = (
    "if prediction_type == 'sample':\n"
    "                preds = np.zeros((n_test_original, num_samples), dtype=np.float32)\n"
    "            else:\n"
    "                preds = np.zeros(n_test_original, dtype=np.float32)"
)


def main(path: str) -> int:
    with open(path) as f:
        src = f.read()

    already = ("if prediction_type == 'sample':" in src
               and "np.zeros((n_test_original, num_samples)" in src)

    # Count occurrences of the buggy 1D allocation.
    n_buggy = src.count(BUGGY)

    if already and n_buggy == 1:
        # The unclustered else-branch (line 878+ in upstream) also allocates
        # 1D; if we've already added the if/else for the clustered branch,
        # the remaining buggy occurrence is the fallback path — expected.
        print(f'[skip] {path} already patched')
        return 0

    if n_buggy == 0:
        print(f'[error] no buggy line found; inspect manually', file=sys.stderr)
        return 1

    # Replace only the first occurrence — that's the one in the clustered branch.
    src2 = src.replace(BUGGY, FIXED, 1)
    with open(path, 'w') as f:
        f.write(src2)
    print(f'[patched] {path}')
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: patch_uwyk_wrapper.py <path-to-PreprocessingGraphConditionedPFN.py>')
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
