"""95% interval of tau = Y1 - Y0 read from a histogram prediction, exactly.

A histogram puts its mass uniformly inside each bin. For two arms on the same bin width h,
the difference of two uniforms is a triangle of half-width h. So the diagonal mass S[k] that the
paper's scorer places as a POINT at atom a_k is really spread as a triangle on [a_k - h, a_k + h].
The CDF of that triangle mixture is piecewise quadratic; its quantiles are solved in closed form.
"""
import numpy as np


def interval_hist(atoms, p, lo_q=0.025, hi_q=0.975):
    h = atoms[1] - atoms[0]
    a = np.r_[atoms[0] - h, atoms, atoms[-1] + h]            # zero-mass atom at each end
    S = np.r_[0.0, p / p.sum(), 0.0]
    Fm = np.cumsum(S) - S / 2                                  # CDF at each atom: all mass left + half its own
    out = []
    for q in (lo_q, hi_q):
        m = int(np.clip(np.searchsorted(Fm, q, side="right") - 1, 0, len(a) - 2))
        # between a_m and a_{m+1}, with u = (x - a_m)/h:  F = Fm[m] + S_m u + (S_{m+1} - S_m) u^2 / 2
        A, B, d = (S[m + 1] - S[m]) / 2, S[m], q - Fm[m]
        den = B + np.sqrt(max(B * B + 4 * A * d, 0.0))
        u = 0.0 if den <= 0 else np.clip(2 * d / den, 0.0, 1.0)
        out.append(a[m] + u * h)
    return tuple(out)


if __name__ == "__main__":                                     # check against sampling from the histogram
    rng = np.random.default_rng(0)
    J, h = 10, 0.2
    joint = rng.dirichlet(np.ones(J * J) * 0.3).reshape(J, J)
    joint[3, 3] += 5; joint /= joint.sum()                     # most mass in one cell: the hard case
    S = np.array([np.trace(joint, offset=k) for k in range(-(J - 1), J)])
    atoms = np.arange(-(J - 1), J) * h
    cells = rng.choice(J * J, 400_000, p=joint.ravel())
    i, j = np.divmod(cells, J)
    d = (j + rng.random(cells.size)) * h - (i + rng.random(cells.size)) * h
    print("exact     ", np.round(interval_hist(atoms, S), 4))
    print("sampled   ", np.round(np.quantile(d, [0.025, 0.975]), 4))
