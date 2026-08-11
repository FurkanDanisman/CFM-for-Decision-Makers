"""Recompute an Ours-method's CATE/ATE from its own marginals under independence.

Rationale: for a joint-head model, the diagonal integration of the smoothed
2D density gives one CATE derivation. Alternative: convolve the two arm
marginals under independence assumption (Do-PFN's recipe). On IHDP the
true DGP is independent, so the second derivation is optimal IF the
marginals are correct. Comparing the two tells us whether the model's
joint density has spurious correlation vs. its marginals.

Reads shards directly — no model re-inference required.

Usage
-----
    python R-PFN/benchmarks/l2_ihdp/recompute_fn10_from_marginals.py \\
        --shards-glob "/scratch/.../out_dopfn_bb_j10_step35000.r*.npz" \\
        --dopfn-shards-glob "/scratch/.../out.r*.npz" \\
        --key ours_dopfn_bb
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


Y_EDGES = np.linspace(-1.5, 1.5, 101)
Y_CENTERS = 0.5 * (Y_EDGES[:-1] + Y_EDGES[1:])
Y_BIN = float(Y_CENTERS[1] - Y_CENTERS[0])
TAU_EDGES = np.linspace(-3.0, 3.0, 601)
TAU_CENTERS = 0.5 * (TAU_EDGES[:-1] + TAU_EDGES[1:])
TAU_BIN = float(TAU_CENTERS[1] - TAU_CENTERS[0])


def naive_p_tau_from_marginals(p_y0, p_y1):
    """p_tau(t) = integral p1(y0 + t) p0(y0) dy0  under independence."""
    p_y0 = np.asarray(p_y0, dtype=np.float64).reshape(-1)
    p_y1 = np.asarray(p_y1, dtype=np.float64).reshape(-1)
    n = p_y0.shape[0]
    density = np.correlate(p_y1, p_y0, mode='full')
    tau_native = np.arange(-(n - 1), n) * Y_BIN
    out = np.interp(TAU_CENTERS, tau_native, density, left=0.0, right=0.0)
    s = out.sum() * TAU_BIN
    if s > 0:
        out = out / s
    return out


def _l2(f, g, grid):
    trap = getattr(np, 'trapezoid', np.trapz)
    return float(np.sqrt(trap((f - g) ** 2, grid)))


def _barycenter_1d(densities, grid, n_tau=4001):
    densities = np.atleast_2d(np.asarray(densities, dtype=float))
    N, M = densities.shape
    dx = float(grid[1] - grid[0])
    taus = np.linspace(1.0 / (2 * n_tau), 1.0 - 1.0 / (2 * n_tau), n_tau)
    quantiles = np.zeros((N, n_tau))
    for i in range(N):
        f = densities[i]
        F = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * dx)])
        if F[-1] <= 0:
            quantiles[i] = np.mean(grid)
            continue
        F = F / F[-1]
        quantiles[i] = np.interp(taus, F, grid)
    Q_bary = quantiles.mean(axis=0)
    hist, _ = np.histogram(Q_bary, bins=np.concatenate(
        [[grid[0] - 0.5 * dx], grid + 0.5 * dx]))
    bary = hist.astype(float) / hist.sum() / dx
    return bary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards-glob', required=True,
                    help='shard glob for the Ours method to analyse')
    ap.add_argument('--key', default='ours_fn10',
                    help='method key in the shard (e.g. ours_fn10, ours_dopfn_bb)')
    ap.add_argument('--dopfn-shards-glob', default='',
                    help='if given, print Do-PFN reference numbers')
    args = ap.parse_args()

    shards = sorted(glob.glob(args.shards_glob))
    if not shards:
        print(f'[fatal] no shards match {args.shards_glob}'); return 2
    print(f'[load] {len(shards)} shards for method {args.key!r}')

    key = args.key
    l2_tau_joint = []
    l2_tau_marg  = []
    l2_ate_joint = []
    l2_ate_marg  = []

    for path in shards:
        with np.load(path) as f:
            p_y0_true  = np.asarray(f['p_y0_true'])
            p_y1_true  = np.asarray(f['p_y1_true'])
            p_tau_true = np.asarray(f['p_tau_true'])
            p_ate_true = np.asarray(f['p_ate_true'])
            key_y0  = f'{key}__p_y0'
            key_y1  = f'{key}__p_y1'
            key_tau = f'{key}__p_tau'
            if key_y0 not in f.files:
                print(f'[skip] {path}: missing {key_y0}'); continue
            p_y0_hat  = np.asarray(f[key_y0])
            p_y1_hat  = np.asarray(f[key_y1])
            p_tau_hat_joint = np.asarray(f[key_tau])

        n_q = p_y0_hat.shape[0]
        p_tau_hat_marg = np.stack([
            naive_p_tau_from_marginals(p_y0_hat[q], p_y1_hat[q])
            for q in range(n_q)])

        for q in range(n_q):
            l2_tau_joint.append(_l2(p_tau_hat_joint[q], p_tau_true[q], TAU_CENTERS))
            l2_tau_marg.append(_l2(p_tau_hat_marg[q], p_tau_true[q], TAU_CENTERS))

        p_ate_joint = _barycenter_1d(p_tau_hat_joint, TAU_CENTERS)
        p_ate_marg  = _barycenter_1d(p_tau_hat_marg,  TAU_CENTERS)
        p_ate_joint /= (p_ate_joint.sum() * TAU_BIN)
        p_ate_marg  /= (p_ate_marg.sum() * TAU_BIN)
        l2_ate_joint.append(_l2(p_ate_joint, p_ate_true, TAU_CENTERS))
        l2_ate_marg.append(_l2(p_ate_marg,  p_ate_true, TAU_CENTERS))

    l2_tau_joint = np.array(l2_tau_joint)
    l2_tau_marg  = np.array(l2_tau_marg)
    l2_ate_joint = np.array(l2_ate_joint)
    l2_ate_marg  = np.array(l2_ate_marg)

    print()
    print(f'{"quantity":30s} {"n":>6s} {"mean":>8s} {"median":>8s} {"std":>8s}')
    print('-' * 68)
    for name, arr in [
        (f'{key} CATE (joint-diagonal)', l2_tau_joint),
        (f'{key} CATE (marginals-conv)', l2_tau_marg),
        (f'{key} ATE  (joint-diagonal)', l2_ate_joint),
        (f'{key} ATE  (marginals-conv)', l2_ate_marg),
    ]:
        print(f'{name:30s} {arr.size:>6d} {arr.mean():>8.4f} '
              f'{np.median(arr):>8.4f} {arr.std(ddof=1):>8.4f}')

    if args.dopfn_shards_glob:
        dp_shards = sorted(glob.glob(args.dopfn_shards_glob))
        if dp_shards:
            l2_tau_dp = []
            l2_ate_dp = []
            for path in dp_shards:
                with np.load(path) as f:
                    if 'dopfn__l2_tau' not in f.files:
                        continue
                    l2_tau_dp.append(np.asarray(f['dopfn__l2_tau']).reshape(-1))
                    l2_ate_dp.append(float(np.asarray(f['dopfn__l2_ate'])))
            l2_tau_dp = np.concatenate(l2_tau_dp) if l2_tau_dp else np.array([])
            l2_ate_dp = np.array(l2_ate_dp)
            print()
            print('Reference: Do-PFN (from --dopfn-shards-glob)')
            print(f'{"Do-PFN CATE":30s} {l2_tau_dp.size:>6d} '
                  f'{l2_tau_dp.mean():>8.4f} '
                  f'{np.median(l2_tau_dp):>8.4f} {l2_tau_dp.std(ddof=1):>8.4f}')
            print(f'{"Do-PFN ATE":30s} {l2_ate_dp.size:>6d} '
                  f'{l2_ate_dp.mean():>8.4f} '
                  f'{np.median(l2_ate_dp):>8.4f} {l2_ate_dp.std(ddof=1):>8.4f}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
