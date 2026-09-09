"""Dataset-generic density metrics for the W2-BARYCENTER ATE density.

Sibling of eval_density_metrics.py, but the compared densities are
per-realization p_ATE (a single τ-density summarizing the sample's
CATE) instead of per-query p(τ|x_q).

  Estimated p_ATE(τ)  = W2 barycenter across per-query p(τ|x_q); stored
                        in ate_w2_{tag}_r<###>.npz under keys `p_ate_raw`
                        and `tau_raw`. Produced by compute_ate_density_w2_cell.
  Truth   p_ATE(τ)    = mean over queries of the truth p(τ|x_q).
                        Same "summary of per-query CATE" object, so it has
                        the same width scale as the W2 barycenter (NOT
                        the sampling distribution N(true_ATE, σ²/N),
                        which would be much narrower).

Metrics per realization (all in raw Y units, on a shared tight τ grid):
  NLL     = -log p_est(true_ATE)                  (point-vs-density)
  L2      = sqrt(∫ (p_true − p_est)² dτ)
  KL_fwd  = ∫ p_true · log(p_true / p_est) dτ     (truth ‖ est)
  KL_rev  = ∫ p_est  · log(p_est  / p_true) dτ    (est ‖ truth)
  ATE_err = |E[p_est] − true_ATE|                 (density-mean bias)

Aggregation: per-realization values → mean ± SE across realizations.

Usage (matches CATE evaluator's env conventions):
    python realcause_eval/eval_ate_density_metrics.py \\
        --dataset IHDP \\
        --root-1d $OUT_1D --root-2d $OUT_2D \\
        --causalpfn $CAUSALPFN --repo $REPO \\
        --ate-tag malc_B500 \\
        --out-md metrics_ATE_IHDP.md
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


DEFAULT_T = 4001
_EPS = 1e-12
_ANALYTIC_GAUSSIAN = {'IHDP', 'ACIC'}
_REALCAUSE_CSV     = {'CPS', 'PSID', 'PSID_bal'}


def _gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def _renorm(p, dx):
    p = np.clip(p, 0.0, None)
    s = np.sum(p, axis=-1, keepdims=True) * dx
    return p / np.where(s > 0, s, 1.0)


def _l2(f, g, dx):
    return float(np.sqrt(np.sum((f - g) ** 2) * dx))


def _kl(f, g, dx):
    f_ = np.clip(f, _EPS, None); g_ = np.clip(g, _EPS, None)
    return float(np.sum(f_ * np.log(f_ / g_)) * dx)


def _nll_at_point(p_est, tau_grid, y_true):
    """NLL = −log p_est(y_true) via linear interp of p_est on tau_grid."""
    T = tau_grid.size
    dtau = float(tau_grid[1] - tau_grid[0])
    idx_f = (y_true - tau_grid[0]) / dtau
    idx_lo = int(np.clip(np.floor(idx_f), 0, T - 2))
    frac   = float(np.clip(idx_f - idx_lo, 0.0, 1.0))
    p_at   = (1.0 - frac) * p_est[idx_lo] + frac * p_est[idx_lo + 1]
    return -float(np.log(np.clip(p_at, _EPS, None)))


def _mean_se(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0: return float('nan'), float('nan')
    m  = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float('nan')
    return m, se


def _fmt(m, se, big=False):
    if not np.isfinite(m): return '—'
    if big:
        m_s  = f'{m:,.2f}'
        se_s = f'{se:,.2f}' if np.isfinite(se) else '—'
    else:
        m_s  = f'{m:.4f}'
        se_s = f'{se:.4f}' if np.isfinite(se) else '—'
    return f'{m_s} ± {se_s}'


# ── Method p_ATE loader ────────────────────────────────────────────────────

def _resolve_ate_npz(root, method, dataset, r, ate_tag):
    """Try inline + split filenames, both 3- and 2-digit padding."""
    ds_on_disk = dataset
    if method == 'dopfnbb' and dataset == 'PSID_bal':
        ds_on_disk = 'PSIDbal'
    cell = os.path.join(root, method, ds_on_disk)
    for pad in (3, 2):
        cand = os.path.join(cell, f'ate_w2_{ate_tag}_r{r:0{pad}d}.npz')
        if os.path.isfile(cand): return cand
    return None


def _load_p_ate_on_grid(root, method, dataset, r, ate_tag, tau_grid_raw):
    path = _resolve_ate_npz(root, method, dataset, r, ate_tag)
    if not path: return None
    with np.load(path, allow_pickle=True) as z:
        p_ate = np.asarray(z['p_ate_raw'], dtype=np.float64)   # (T_native,)
        tau_native = np.asarray(z['tau_raw'], dtype=np.float64)
    out = np.interp(tau_grid_raw, tau_native, p_ate, left=0.0, right=0.0)
    dtau = float(tau_grid_raw[1] - tau_grid_raw[0])
    return _renorm(out[None, :], dtau)[0]      # (T,)


# ── Truth p_ATE per realization ──────────────────────────────────────────

def _truth_and_grid_for_realization(r, dataset, causalpfn_dir, ds_obj,
                                     repo, T, tau_pad_sigmas=6.0):
    """Return (tau_grid_raw, p_true_ATE, true_ATE).

    Truth p_ATE = mean over queries of the truth per-query p(τ|x_q).
    Same 'summary of per-query CATE' object as the W2 barycenter.
    """
    if dataset == 'IHDP':
        from true_ihdp import load_ihdp_truth
        ds = ds_obj[r][0]
        y_train_raw = np.asarray(ds.y_train, dtype=np.float64).reshape(-1)
        truth = load_ihdp_truth(r, causalpfn_dir, y_train_raw)
        scale = truth.y_rng / 2.0
        mu_diff   = (truth.mu1_test_scaled - truth.mu0_test_scaled) * scale   # (N_q,)
        sigma_tau = np.sqrt(2.0) * float(truth.sigma_scaled) * scale          # scalar
        lo = float(mu_diff.min() - tau_pad_sigmas * sigma_tau)
        hi = float(mu_diff.max() + tau_pad_sigmas * sigma_tau)
        tau_grid = np.linspace(lo, hi, T)
        dtau = float(tau_grid[1] - tau_grid[0])
        # Truth per-query densities → average across queries.
        p_q = np.stack([_gaussian(tau_grid, mu, sigma_tau) for mu in mu_diff])
        p_true_ATE = _renorm(p_q.mean(axis=0), dtau)
        true_ATE = float(mu_diff.mean())
        return tau_grid, p_true_ATE, true_ATE

    if dataset == 'ACIC':
        from true_acic import load_acic_truth
        ds = ds_obj[r][0]
        y_train_raw = np.asarray(ds.y_train, dtype=np.float64).reshape(-1)
        truth = load_acic_truth(r, y_train_raw)
        scale = truth.y_rng / 2.0
        mu_diff   = (truth.mu1_test_scaled - truth.mu0_test_scaled) * scale
        sigma_tau = np.sqrt(2.0) * float(truth.sigma_scaled) * scale
        lo = float(mu_diff.min() - tau_pad_sigmas * sigma_tau)
        hi = float(mu_diff.max() + tau_pad_sigmas * sigma_tau)
        tau_grid = np.linspace(lo, hi, T)
        dtau = float(tau_grid[1] - tau_grid[0])
        p_q = np.stack([_gaussian(tau_grid, mu, sigma_tau) for mu in mu_diff])
        p_true_ATE = _renorm(p_q.mean(axis=0), dtau)
        true_ATE = float(mu_diff.mean())
        return tau_grid, p_true_ATE, true_ATE

    # RealCause KDE truth. Reuse the CATE helpers.
    from eval_density_metrics import _load_realcause_stack, _match_by_ite
    _, y0_all, y1_all = _load_realcause_stack(dataset, causalpfn_dir)
    tau_all = y1_all - y0_all
    ds = ds_obj[r][0]
    test_idx_full = _match_by_ite(dataset, r, causalpfn_dir, ds)
    keep = test_idx_full >= 0
    test_idx = test_idx_full[keep]
    true_cate = np.asarray(ds.true_cate, dtype=np.float64).reshape(-1)[keep]
    tau_samples = tau_all[test_idx]              # (N_q, K=100)
    N_q, K = tau_samples.shape
    s_all = tau_samples.std(ddof=1)
    lo = float(tau_samples.min() - 3.0 * s_all)
    hi = float(tau_samples.max() + 3.0 * s_all)
    tau_grid = np.linspace(lo, hi, T)
    dtau = float(tau_grid[1] - tau_grid[0])
    # Per-query KDE, then average. Silverman bandwidth.
    sigmas = tau_samples.std(axis=1, ddof=1)
    sigma_floor = 1e-3 * (tau_grid[-1] - tau_grid[0])
    h = np.maximum(1.06 * sigmas * (K ** (-1.0 / 5.0)), sigma_floor)
    p_q = np.empty((N_q, T), dtype=np.float64)
    for q in range(N_q):
        z = (tau_grid[:, None] - tau_samples[q, None, :]) / h[q]
        p_q[q] = np.exp(-0.5 * z ** 2).sum(axis=1) / (K * h[q] * np.sqrt(2.0 * np.pi))
    p_q = _renorm(p_q, dtau)
    p_true_ATE = _renorm(p_q.mean(axis=0), dtau)
    true_ATE = float(true_cate.mean())
    return tau_grid, p_true_ATE, true_ATE


# ── Per-realization pipeline ─────────────────────────────────────────────

def evaluate_realization(r, dataset, causalpfn_dir, ds_obj, root_1d, root_2d,
                          methods_1d, methods_2d, ate_tag, repo, T):
    tau_grid, p_true, true_ATE = _truth_and_grid_for_realization(
        r, dataset, causalpfn_dir, ds_obj, repo, T)
    dtau = float(tau_grid[1] - tau_grid[0])

    results = {}
    for method, kind in [(m, '1d') for m in methods_1d] + [(m, '2d') for m in methods_2d]:
        root = root_1d if kind == '1d' else root_2d
        try:
            p_est = _load_p_ate_on_grid(root, method, dataset, r, ate_tag, tau_grid)
        except Exception as e:
            print(f'  [warn] r={r:03d} {method}: {e}', file=sys.stderr)
            p_est = None
        if p_est is None:
            results[method] = None; continue
        nll = _nll_at_point(p_est, tau_grid, true_ATE)
        l2  = _l2(p_true, p_est, dtau)
        kf  = _kl(p_true, p_est, dtau)
        kr  = _kl(p_est,  p_true, dtau)
        E_est = float((tau_grid * p_est).sum() * dtau)
        ate_err = abs(E_est - true_ATE)
        results[method] = dict(nll=nll, l2=l2, kl_fwd=kf, kl_rev=kr, ate_err=ate_err)
    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True,
                    choices=['IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal'])
    ap.add_argument('--root-1d', required=True)
    ap.add_argument('--root-2d', required=True)
    ap.add_argument('--causalpfn', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--methods-1d', nargs='+', default=['cpfn1d', 'dopfn', 'uwyk1d'])
    ap.add_argument('--methods-2d', nargs='+', default=['cpfn2d', 'graph2d', 'dopfnbb'])
    ap.add_argument('--n-realizations', type=int, default=None)
    ap.add_argument('--T', type=int, default=DEFAULT_T)
    ap.add_argument('--ate-tag', required=True,
                    help='ate_w2_<tag>_r<###>.npz filename tag. '
                         'For 1D+2D MALC B=500, use "malc_B500".')
    ap.add_argument('--out-md', default=None)
    args = ap.parse_args()

    for p in (args.repo, os.path.join(args.repo, 'realcause_eval'),
              os.path.join(args.repo, 'benchmarks', 'l2_ihdp'),
              os.path.join(args.repo, 'benchmarks', 'l2_acic')):
        if p not in sys.path: sys.path.insert(0, p)
    sys.path.insert(0, args.causalpfn)
    sys.path.insert(0, os.path.join(args.causalpfn, 'src'))

    # faiss stub (see eval_density_metrics.py for context)
    try:
        import faiss  # noqa: F401
    except ImportError:
        import types as _types
        sys.modules['faiss'] = _types.ModuleType('faiss')

    if args.dataset == 'IHDP':
        from benchmarks import IHDPDataset; ds_obj = IHDPDataset(); n_default = 100
    elif args.dataset == 'ACIC':
        from benchmarks import ACIC2016Dataset; ds_obj = ACIC2016Dataset(); n_default = 10
    elif args.dataset == 'CPS':
        from benchmarks import RealCauseLalondeCPSDataset
        ds_obj = RealCauseLalondeCPSDataset(); n_default = 100
    else:
        from benchmarks import RealCauseLalondePSIDDataset
        ds_obj = RealCauseLalondePSIDDataset(); n_default = 100

    n_realizations = args.n_realizations or n_default
    methods_1d = list(args.methods_1d); methods_2d = list(args.methods_2d)
    all_methods = methods_1d + methods_2d
    print(f'[bootstrap] dataset={args.dataset}  n_realizations={n_realizations}  '
          f'T={args.T}  ate_tag={args.ate_tag}  '
          f'methods_1d={methods_1d}  methods_2d={methods_2d}', flush=True)

    per_r = []
    t0 = time.time()
    for r in range(n_realizations):
        tr = time.time()
        try:
            res = evaluate_realization(r, args.dataset, args.causalpfn, ds_obj,
                                         args.root_1d, args.root_2d,
                                         methods_1d, methods_2d,
                                         args.ate_tag, args.repo, args.T)
        except Exception as e:
            print(f'  [warn] r={r:03d}: {e}', file=sys.stderr); continue
        per_r.append(res)
        print(f'  [{time.strftime("%H:%M:%S")}] r={r:03d}  ({time.time() - tr:.1f}s)',
              flush=True)

    _truth_kind = ('analytic Gaussian mixture (mean of per-query N(μ_diff, 2σ²))'
                    if args.dataset in _ANALYTIC_GAUSSIAN
                    else 'empirical KDE mixture (mean of per-query KDE from 100 CSVs)')
    lines = [
        f'\nATE-density metrics — {args.dataset} — tight T={args.T} raw τ grid '
        f'(estimated p_ATE = W2 barycenter, tag={args.ate_tag})',
        '',
        f'(truth p_ATE = {_truth_kind}; metrics per realization, aggregated '
        f'as mean ± SE across {n_realizations} realizations. Lower = better.)',
        '',
        '| Method | NLL@true_ATE | L2 | KL_fwd (truth‖est) | KL_rev (est‖truth) | |E[est] − true_ATE| |',
        '|---|---|---|---|---|---|',
    ]
    _big = args.dataset in ('CPS', 'PSID', 'PSID_bal')
    for method in all_methods:
        vals = {k: [] for k in ('nll', 'l2', 'kl_fwd', 'kl_rev', 'ate_err')}
        for res in per_r:
            m = res.get(method) if res else None
            if not m: continue
            for k in vals: vals[k].append(m[k])
        cells = [method]
        for k in ('nll', 'l2', 'kl_fwd', 'kl_rev'):
            cells.append(_fmt(*_mean_se(vals[k])))
        cells.append(_fmt(*_mean_se(vals['ate_err']), big=_big))
        lines.append('| ' + ' | '.join(cells) + ' |')

    md = '\n'.join(lines) + '\n'
    print(md)
    if args.out_md:
        with open(args.out_md, 'w') as f: f.write(md)
        print(f'wrote {args.out_md}', flush=True)
    print(f'[done] total={time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
