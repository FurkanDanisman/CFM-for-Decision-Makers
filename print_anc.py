"""Print full ancestral matrices (v3a, v3b, v3c) for CPS and PSID.

Both CPS and PSID have n_real = 8 real features, so their matrices are
identical (only the underlying X/Y/T data differs). Padded region (indices
10..51) is -1 rows/cols.

Run:
    python print_anc.py
"""
import numpy as np

F = 50                                  # max features (matrix is F+2 = 52 x 52)
N_REAL_BY_DATASET = {'CPS': 8, 'PSID': 8}


def _padded_neg1_only(F, n_real):
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    for i in range(n_real, F):
        A[2 + i, :] = -1.0
        A[:, 2 + i] = -1.0
        A[2 + i, 2 + i] = -1.0
    return A


def build_v3a(F, n_real):
    """T→Y=+1, X→T=+1, X→Y=+1. Rest of real block 0."""
    A = _padded_neg1_only(F, n_real)
    A[0, 1] = 1.0
    for i in range(n_real):
        A[2 + i, 0] = 1.0
        A[2 + i, 1] = 1.0
    return A


def build_v3b(F, n_real):
    """v3a + all reverses = -1 (Y→T, T→X, Y→X)."""
    A = _padded_neg1_only(F, n_real)
    A[0, 1] = 1.0
    A[1, 0] = -1.0
    for i in range(n_real):
        A[2 + i, 0] = 1.0
        A[2 + i, 1] = 1.0
        A[0, 2 + i] = -1.0
        A[1, 2 + i] = -1.0
    return A


def build_v3c(F, n_real):
    """v3b + real-block diagonal -1."""
    A = build_v3b(F, n_real)
    real_n = 2 + n_real
    for i in range(real_n):
        A[i, i] = -1.0
    return A


def build_uwyk_eval(F, n_real):
    """UWYK's exact eval matrix, verbatim from benchmarks/methods/uwyk.py:22-33:

        adj = np.zeros((F + 2, F + 2))
        adj[T, Y] = 1
        for i in range(n_real):
            adj[X_i, T] = 1
            adj[X_i, Y] = 1
        for padded feature fi:
            adj[fi, :] = -1
            adj[:, fi] = -1
            adj[fi, fi] = -1

    Identical to v3a. No reverse -1s in the real block, no real-block
    diagonal -1s, no propagation. This is the reference: their model +
    this matrix reproduces UWYK Table 3 anc-wins-noanc on ACIC.
    """
    adj = np.zeros((F + 2, F + 2), dtype=np.float32)
    T_idx = 0; Y_idx = 1; feature_offset = 2
    adj[T_idx, Y_idx] = 1.0
    for i in range(n_real):
        adj[feature_offset + i, T_idx] = 1.0
        adj[feature_offset + i, Y_idx] = 1.0
    for i in range(n_real, F):
        fi = feature_offset + i
        adj[fi, :] = -1.0
        adj[:, fi] = -1.0
        adj[fi, fi] = -1.0
    return adj


def build_v4a(F, n_real):
    """v4a: v3a MINUS T→Y edge. X→T = +1, X→Y = +1, but A[T,Y] = 0.
    Probes whether the T→Y=+1 assertion is what our model can't use."""
    A = _padded_neg1_only(F, n_real)
    for i in range(n_real):
        A[2 + i, 0] = 1.0
        A[2 + i, 1] = 1.0
    return A


BUILDERS = {
    'uwyk': build_uwyk_eval,
    'v3a':  build_v3a,
    'v3b':  build_v3b,
    'v3c':  build_v3c,
    'v4a':  build_v4a,
}


if __name__ == '__main__':
    np.set_printoptions(linewidth=400, threshold=10000,
                        formatter={'int': lambda x: f'{x:+2d}'})

    for ds, n_real in N_REAL_BY_DATASET.items():
        for name, fn in BUILDERS.items():
            A = fn(F, n_real).astype(int)
            print(f'\n═══ {ds}  variant={name}  n_real={n_real}  shape={A.shape} ═══')
            print(A)

    # Sanity: UWYK's eval matrix and our v3a should be byte-identical.
    print('\n═══ SANITY: UWYK eval matrix vs our v3a ═══')
    for ds, n_real in N_REAL_BY_DATASET.items():
        u = build_uwyk_eval(F, n_real)
        v = build_v3a(F, n_real)
        print(f'  {ds:5s}  n_real={n_real:2d}  '
              f'identical={np.array_equal(u, v)}  '
              f'max_abs_diff={np.abs(u - v).max():.1f}')
