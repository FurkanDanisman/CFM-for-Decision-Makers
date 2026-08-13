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

# Old buggy line — allocates preds as 1D regardless of prediction_type.
BUGGY = "preds = np.zeros(n_test_original, dtype=np.float32)"

# v1 patch (broken): assumed cluster_preds shape is (n_test, num_samples).
# In practice _predict_single_cluster returns (n_test, num_bars=1000) in
# sample mode, ignoring the outer num_samples. So we allocate lazily —
# preds = None initially, size it from the first cluster's output.
V1_BROKEN = (
    "if prediction_type == 'sample':\n"
    "                preds = np.zeros((n_test_original, num_samples), dtype=np.float32)\n"
    "            else:\n"
    "                preds = np.zeros(n_test_original, dtype=np.float32)"
)

V2_FIXED = (
    "# [patched] Lazy allocation: sized from the first cluster's cluster_preds.\n"
    "            # Works for both point ('mean'/'mode') and 'sample'/'point' modes\n"
    "            # regardless of what num_samples column count is returned.\n"
    "            preds = None"
)

BUGGY_ASSIGN = "preds[test_mask] = cluster_preds"

FIXED_ASSIGN = (
    "if preds is None:\n"
    "                    if getattr(cluster_preds, 'ndim', 1) == 2:\n"
    "                        preds = np.zeros((n_test_original, cluster_preds.shape[1]),\n"
    "                                          dtype=np.float32)\n"
    "                    else:\n"
    "                        preds = np.zeros(n_test_original, dtype=np.float32)\n"
    "                preds[test_mask] = cluster_preds"
)


def main(path: str) -> int:
    with open(path) as f:
        src = f.read()

    # Idempotent: if the v2 fix is already applied, no-op.
    if V2_FIXED.split('\n')[-1] in src and 'if preds is None:' in src:
        print(f'[skip] {path} already patched (v2)')
        return 0

    # Upgrade v1 → v2 if v1 was applied earlier.
    if V1_BROKEN in src:
        src = src.replace(V1_BROKEN, V2_FIXED, 1)
        src = src.replace(BUGGY_ASSIGN, FIXED_ASSIGN, 1)
        with open(path, 'w') as f:
            f.write(src)
        print(f'[patched] {path} (upgraded v1 → v2)')
        return 0

    # Fresh patch on upstream: replace the 1D allocation with lazy None,
    # then rewrite the assignment site to allocate on first cluster.
    if BUGGY in src:
        src = src.replace(BUGGY, V2_FIXED, 1)
        src = src.replace(BUGGY_ASSIGN, FIXED_ASSIGN, 1)
        with open(path, 'w') as f:
            f.write(src)
        print(f'[patched] {path} (fresh v2 patch)')
        return 0

    print(f'[error] no known buggy pattern found; inspect manually',
          file=sys.stderr)
    return 1


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: patch_uwyk_wrapper.py <path-to-PreprocessingGraphConditionedPFN.py>')
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
