"""CATE 95% CI / coverage / length from 1D density dumps.

Reads per-realization NPZs written by realcause_eval/{do_pfn,cpfn1d,uwyk1d}
runs with DENSITY_DUMP=1. Each NPZ must contain (see the emitters):

    edges          shape (nbins+1,)   bar-dist edges in scaled Y units
                                       (RAW Y for DoPFN — y_shift=0, y_scale=1)
    p_y0_scaled    shape (N_q, nbins) softmax marginal p(Y|do(0)) per query
    p_y1_scaled    shape (N_q, nbins) softmax marginal p(Y|do(1)) per query
    y_shift        scalar             raw = scaled*y_scale + y_shift
    y_scale        scalar             (drops out in τ = Y_1 - Y_0)
    true_cate_per_query shape (N_q,)  ground-truth CATE per query

Method:
    1. Assume Y|do(0) ⊥ Y|do(1). Then p(τ = Y_1 - Y_0) = p_y1 * flip(p_y0)
       via 1D discrete convolution (2·nbins-1 output bins).
    2. Convolution mean equals E[Y_1] - E[Y_0] — SAME as the point CATE the
       models already report. So point estimates are guaranteed consistent.
    3. Per-query CI: find τ_lo, τ_hi via linear-interp on the discrete CDF
       at levels 0.025 and 0.975.
    4. Un-scale τ_lo, τ_hi to raw units by multiplying by y_scale (y_shift
       cancels in the difference).
    5. Coverage = mean_q 1[τ_true[q] ∈ [τ_lo[q], τ_hi[q]]].
       Length   = mean_q (τ_hi[q] - τ_lo[q]).
    6. Aggregate across realizations: mean of coverages, mean of lengths.

Usage:
    python realcause_eval/cate_ci_from_density.py \\
        --out-root /scratch/.../results_realcause_all \\
        --methods cpfn1d dopfn uwyk1d
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


DATASETS = ('IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal')

# Per-method point-estimate keys (from the same NPZs).
# cpfn1d writes pehe_raw / err_raw (unsuffixed).
# dopfn  writes pehe_dopfn / err_dopfn.
# uwyk1d writes suffixed keys — use noanc for the point row (matches paper
# Table 3 UWYK No-Anc). Change to _v3b via --uwyk-tag if you want that row.
_POINT_KEYS = {
    'cpfn1d': ('pehe_raw',       'err_raw'),
    'dopfn':  ('pehe_dopfn',     'err_dopfn'),
    'uwyk1d': ('pehe_raw_noanc', 'err_raw_noanc'),
}


def _load_density(npz_path: str):
    """Return (edges, p_y0, p_y1, y_shift, y_scale, true_cate_pq) or None."""
    try:
        with np.load(npz_path, allow_pickle=True) as z:
            required = {'edges', 'p_y0_scaled', 'p_y1_scaled', 'y_shift',
                        'y_scale', 'true_cate_per_query'}
            if not required.issubset(set(z.files)):
                return None
            edges = np.asarray(z['edges'], dtype=np.float64)
            p_y0  = np.asarray(z['p_y0_scaled'], dtype=np.float64)
            p_y1  = np.asarray(z['p_y1_scaled'], dtype=np.float64)
            y_shift = float(z['y_shift'])
            y_scale = float(z['y_scale'])
            true_cate_pq = np.asarray(z['true_cate_per_query'], dtype=np.float64)
        # Ensure per-query densities normalized.
        p_y0 /= p_y0.sum(axis=-1, keepdims=True).clip(min=1e-12)
        p_y1 /= p_y1.sum(axis=-1, keepdims=True).clip(min=1e-12)
        return edges, p_y0, p_y1, y_shift, y_scale, true_cate_pq
    except Exception as e:
        print(f'  [warn] {npz_path}: {e}', file=sys.stderr)
        return None


def _is_uniform(centers: np.ndarray, rtol: float = 1e-4) -> bool:
    """True iff consecutive gaps are equal within rtol × mean gap."""
    if centers.size < 3:
        return True
    d = np.diff(centers)
    return bool(np.max(np.abs(d - d.mean())) < rtol * abs(d.mean()) + 1e-12)


def _ci_from_atoms_uniform(centers: np.ndarray, p_y0: np.ndarray, p_y1: np.ndarray,
                            lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """FAST path: bins are uniform, so N² pairs bucket onto 2N-1 τ values.
    p_τ[q, k] = Σ_{i-j=k+(N-1)} p_y1[q, i] · p_y0[q, j]  via FFT conv.
    Then CDF → interp at lo/hi. Returns (τ_lo, τ_hi) shape (N_q,)."""
    from numpy.fft import rfft, irfft
    N = centers.size
    n_out = 2 * N - 1
    n_fft = 1 << (n_out - 1).bit_length()
    F1 = rfft(p_y1, n=n_fft, axis=-1)
    F0 = rfft(p_y0[:, ::-1], n=n_fft, axis=-1)
    p_tau = irfft(F1 * F0, n=n_fft, axis=-1)[:, :n_out]
    p_tau = np.clip(p_tau, 0.0, None)
    p_tau /= p_tau.sum(axis=-1, keepdims=True).clip(min=1e-12)
    width = float(centers[1] - centers[0])
    tau = (np.arange(n_out) - (N - 1)).astype(np.float64) * width
    # tau is already sorted → cumulative interpolate directly.
    cdf = np.cumsum(p_tau, axis=-1)
    return _quantile_from_sorted(tau[None, :], cdf, lo), _quantile_from_sorted(tau[None, :], cdf, hi)


def _ci_from_atoms_general(centers: np.ndarray, p_y0: np.ndarray, p_y1: np.ndarray,
                            lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """GENERAL path: bins may be non-uniform. Enumerate all N² atoms per query,
    sort by τ, cumulate probabilities, interpolate CDF quantiles.
    Cost: O(N_q · N² · log N²)  — fine for N ≤ ~200. For large N use the
    uniform path via _is_uniform dispatch."""
    N_q = p_y0.shape[0]
    N = centers.size
    tau_mat = centers[:, None] - centers[None, :]   # (N, N) shared across queries
    tau_flat = tau_mat.ravel()                       # (N²,)
    order = np.argsort(tau_flat, kind='stable')
    tau_sorted = tau_flat[order]                     # (N²,) — same for every query

    tau_lo_arr = np.empty(N_q, dtype=np.float64)
    tau_hi_arr = np.empty(N_q, dtype=np.float64)
    for q in range(N_q):
        p_mat = p_y1[q, :, None] * p_y0[q, None, :]  # (N, N)
        p_sorted = p_mat.ravel()[order]
        cdf = np.cumsum(p_sorted)
        cdf /= max(cdf[-1], 1e-12)
        tau_lo_arr[q] = _quantile_from_sorted(tau_sorted[None, :], cdf[None, :], lo)[0]
        tau_hi_arr[q] = _quantile_from_sorted(tau_sorted[None, :], cdf[None, :], hi)[0]
    return tau_lo_arr, tau_hi_arr


def _quantile_from_sorted(tau: np.ndarray, cdf: np.ndarray, level: float) -> np.ndarray:
    """Linear-interp inverse-CDF at `level`. tau and cdf are sorted along axis=-1.
    Broadcasts if tau shape (1, K) and cdf shape (N_q, K), returns (N_q,)."""
    # For each row of cdf, find first index where cdf >= level.
    below = cdf < level
    idx = np.argmax(~below, axis=-1)
    idx = np.where(cdf[..., -1] < level, cdf.shape[-1] - 1, idx)
    row = np.arange(cdf.shape[0])
    c_hi = cdf[row, idx]
    c_lo = np.where(idx > 0, cdf[row, np.maximum(idx - 1, 0)], 0.0)
    t_row = np.broadcast_to(tau, cdf.shape)
    t_hi = t_row[row, idx]
    t_lo = np.where(idx > 0, t_row[row, np.maximum(idx - 1, 0)], t_row[row, 0])
    w = np.where(c_hi > c_lo, (level - c_lo) / (c_hi - c_lo), 0.0)
    return t_lo + w * (t_hi - t_lo)




def process_npz(npz_path: str, pehe_key: str, err_key: str):
    """→ dict with density-derived PEHE / ε_ATE / coverage / length.

    Point CATE (direct):
        cate_hat[q] = (Σ_i c_i · p_y1[q, i] - Σ_i c_i · p_y0[q, i]) · y_scale
    Works for any bin geometry. y_shift cancels between arms.

    95% CI (exact, N²-atom PMF):
        p(τ = c_i - c_j | q) = p_y1[q, i] · p_y0[q, j]   under independence.
        Enumerate atoms, sort by τ, cumulate → CDF → interp at 0.025 / 0.975.
    If bin centers are UNIFORM (cpfn1d, uwyk1d likely) the N² atoms collapse
    onto 2N-1 τ bins via discrete convolution — same result, way faster.
    """
    loaded = _load_density(npz_path)
    if loaded is None:
        return None
    edges, p_y0, p_y1, y_shift, y_scale, true_cate_pq = loaded
    nbins = p_y0.shape[-1]

    # Bin centers on the density axis (handles non-uniform edges).
    if edges.size >= nbins + 1:
        centers = 0.5 * (edges[:nbins] + edges[1:nbins + 1])
    else:
        width = float(edges[1] - edges[0])
        centers = np.arange(nbins) * width + float(edges[0])

    # ── point CATE via subtract-then-mean (works for ANY bin geometry).
    e_y0 = (p_y0 * centers[None, :]).sum(axis=-1)
    e_y1 = (p_y1 * centers[None, :]).sum(axis=-1)
    cate_hat = (e_y1 - e_y0) * y_scale                # y_shift cancels
    resid = cate_hat - true_cate_pq
    pehe_den = float(np.sqrt(np.mean(resid * resid)))
    ate_hat  = float(cate_hat.mean())
    true_ate = float(true_cate_pq.mean())
    err_ate_den = float(abs(ate_hat - true_ate) / max(abs(true_ate), 0.1))

    # ── EXACT CI: enumerate atoms of p(τ) under independence, sort, quantile.
    # If centers are uniform → convolution collapses N² atoms into 2N-1 bins.
    # Else → enumerate all N² atoms per query and sort.
    # Both give τ CI on the density axis; multiply by y_scale for raw units.
    if _is_uniform(centers):
        tau_lo_axis, tau_hi_axis = _ci_from_atoms_uniform(centers, p_y0, p_y1, 0.025, 0.975)
    else:
        tau_lo_axis, tau_hi_axis = _ci_from_atoms_general(centers, p_y0, p_y1, 0.025, 0.975)
    tau_lo = tau_lo_axis * y_scale
    tau_hi = tau_hi_axis * y_scale
    coverage = float(np.mean((true_cate_pq >= tau_lo) & (true_cate_pq <= tau_hi)))
    length   = float(np.mean(tau_hi - tau_lo))

    # ── stored point estimates + cross-check.
    try:
        with np.load(npz_path, allow_pickle=True) as z:
            pehe_stored = float(z[pehe_key]) if pehe_key in z.files else float('nan')
            err_stored  = float(z[err_key])  if err_key  in z.files else float('nan')
            cate_pred_stored = np.asarray(z['cate_pred']).astype(np.float64) \
                if 'cate_pred' in z.files else None
    except Exception:
        pehe_stored, err_stored, cate_pred_stored = float('nan'), float('nan'), None

    cate_max_diff = float('nan')
    if cate_pred_stored is not None and cate_pred_stored.shape == cate_hat.shape:
        cate_max_diff = float(np.max(np.abs(cate_hat - cate_pred_stored)))

    return {
        'pehe_den':    pehe_den,
        'err_den':     err_ate_den,
        'pehe_stored': pehe_stored,
        'err_stored':  err_stored,
        'cate_max_diff': cate_max_diff,
        'coverage':    coverage,
        'length':      length,
        'n_queries':   int(true_cate_pq.size),
    }


def _mean_se(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float('nan'), float('nan')
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float('nan')
    return m, se


def summarize_method_dataset(method_dir: str, dataset: str,
                              pehe_key: str, err_key: str):
    """Walk NPZs; return per-cell aggregates + a max |derived - stored| diff
    diagnostic so we can eyeball whether density-derived point estimates
    reproduce the stored ones (should be ~0 under independence + normalization)."""
    paths = sorted(glob.glob(os.path.join(method_dir, dataset, f'{dataset}_r*.npz')))
    if not paths:
        return None
    pehes_d, errs_d, covs, lens = [], [], [], []
    pehe_diffs, err_diffs, cate_diffs = [], [], []
    for p in paths:
        got = process_npz(p, pehe_key, err_key)
        if got is None:
            continue
        pehes_d.append(got['pehe_den']); errs_d.append(got['err_den'])
        covs.append(got['coverage']);    lens.append(got['length'])
        if np.isfinite(got['pehe_stored']):
            pehe_diffs.append(got['pehe_den'] - got['pehe_stored'])
        if np.isfinite(got['err_stored']):
            err_diffs.append(got['err_den']  - got['err_stored'])
        if np.isfinite(got['cate_max_diff']):
            cate_diffs.append(got['cate_max_diff'])
    if not covs:
        return None
    return {
        'pehe':          _mean_se(pehes_d),
        'err':           _mean_se(errs_d),
        'cov':           _mean_se(covs),
        'len':           _mean_se(lens),
        'n':             len(covs),
        'pehe_max_diff': float(np.max(np.abs(pehe_diffs))) if pehe_diffs else float('nan'),
        'err_max_diff':  float(np.max(np.abs(err_diffs)))  if err_diffs  else float('nan'),
        'cate_max_diff': float(np.max(cate_diffs)) if cate_diffs else float('nan'),
    }


def _fmt(m, se, big=False):
    if not np.isfinite(m):
        return '—'
    if big:
        m_s = f'{m:,.2f}'
        se_s = f'{se:,.2f}' if np.isfinite(se) else '—'
    else:
        m_s = f'{m:.3f}'
        se_s = f'{se:.3f}' if np.isfinite(se) else '—'
    return f'{m_s} ± {se_s}'


def _debug_single(npz_path: str, pehe_key: str, err_key: str) -> None:
    """Print a fingerprint of one realization: edges range, density means,
    density-derived cate summary, and every stored scalar we can grab for
    cross-check (pehe_raw, ate_raw, true_ate, cate_pred per-query if present).
    If mean(density-derived cate) differs from stored ate_raw, the density
    saved in the NPZ is NOT the density used to compute stored PEHE."""
    with np.load(npz_path, allow_pickle=True) as z:
        edges = np.asarray(z['edges'], dtype=np.float64)
        p_y0  = np.asarray(z['p_y0_scaled'], dtype=np.float64)
        p_y1  = np.asarray(z['p_y1_scaled'], dtype=np.float64)
        y_shift = float(z['y_shift']); y_scale = float(z['y_scale'])
        true_cate_pq = np.asarray(z['true_cate_per_query'], dtype=np.float64)
        cate_pred = np.asarray(z['cate_pred']).astype(np.float64) if 'cate_pred' in z.files else None
        pehe_stored = float(z[pehe_key]) if pehe_key in z.files else float('nan')
        # Pull any stored ate scalar for cross-check.
        ate_stored = float('nan')
        for k in ('ate_raw', 'ate_em', 'ate_full'):
            if k in z.files:
                ate_stored = float(z[k]); ate_stored_key = k; break
        else:
            ate_stored_key = None
        true_ate_stored = float(z['true_ate']) if 'true_ate' in z.files else float('nan')
        stored_keys = list(z.files)

    p_y0 /= p_y0.sum(axis=-1, keepdims=True).clip(min=1e-12)
    p_y1 /= p_y1.sum(axis=-1, keepdims=True).clip(min=1e-12)
    nbins = p_y0.shape[-1]
    centers = 0.5 * (edges[:nbins] + edges[1:nbins + 1]) if edges.size >= nbins + 1 \
              else (np.arange(nbins) * float(edges[1] - edges[0]) + edges[0])
    e_y0 = (p_y0 * centers[None, :]).sum(axis=-1)
    e_y1 = (p_y1 * centers[None, :]).sum(axis=-1)
    cate_from_diff = (e_y1 - e_y0) * y_scale

    print(f'\n[{npz_path}]')
    print(f'  NPZ keys: {sorted(stored_keys)}')
    print(f'  edges: shape={edges.shape}  range=[{edges.min():.4g}, {edges.max():.4g}]')
    print(f'  p_y0/p_y1: shape={p_y0.shape}')
    print(f'  centers[0:3]={centers[:3]}   centers[-3:]={centers[-3:]}')
    print(f'  y_shift={y_shift:.4g}  y_scale={y_scale:.4g}')
    print(f'  E[Y_0] (density-axis) mean q: {float(e_y0.mean()):.4g}')
    print(f'  E[Y_1] (density-axis) mean q: {float(e_y1.mean()):.4g}')
    print(f'  cate_from_diff (density → raw) mean q: {float(cate_from_diff.mean()):.4g}')
    if cate_pred is not None:
        print(f'  cate_pred (stored per-query)  mean q: {float(cate_pred.mean()):.4g}')
        print(f'  max |cate_from_diff - cate_pred|: {float(np.max(np.abs(cate_from_diff - cate_pred))):.4g}')
    if ate_stored_key is not None:
        gap = float(cate_from_diff.mean()) - ate_stored
        print(f'  stored {ate_stored_key} (scalar ate_hat) = {ate_stored:.4g}    '
              f'Δ vs density-mean = {gap:+.4g}')
    print(f'  true_cate_per_query mean q = {float(true_cate_pq.mean()):.4g}   '
          f'stored true_ate = {true_ate_stored:.4g}')
    print(f'  stored PEHE ({pehe_key}) = {pehe_stored:.4g}')
    if ate_stored_key is not None and np.isfinite(pehe_stored) and np.isfinite(true_ate_stored):
        bias = abs(ate_stored - true_ate_stored)
        print(f'  sanity: |ate_stored - true_ate| = {bias:.4g}  must be ≤ PEHE = {pehe_stored:.4g}  '
              f'→ {"OK" if bias <= pehe_stored + 1e-6 else "VIOLATED"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', required=True,
                    help='Parent dir; expects <out-root>/<method>/<DATASET>/<D>_r<###>.npz.')
    ap.add_argument('--methods', nargs='+', default=['cpfn1d', 'dopfn', 'uwyk1d'],
                    help='Methods to include (one row each).')
    ap.add_argument('--uwyk-tag', default='noanc', choices=['noanc', 'v3b'],
                    help='Which anc-tag row of uwyk1d to use for PEHE/ε_ATE. '
                         'Default: noanc (matches paper Table 3 UWYK No-Anc).')
    ap.add_argument('--out-md', default=None,
                    help='Also write the markdown table to this path.')
    ap.add_argument('--debug-one', nargs=2, metavar=('METHOD', 'DATASET'),
                    help='Print one-realization fingerprint (edges range, density '
                         'means, cate_from_diff vs cate_pred) for the first NPZ of '
                         'the given (method, dataset). Skips the aggregate table.')
    args = ap.parse_args()

    if args.debug_one is not None:
        method, dataset = args.debug_one
        point_keys = dict(_POINT_KEYS)
        point_keys['uwyk1d'] = (f'pehe_raw_{args.uwyk_tag}', f'err_raw_{args.uwyk_tag}')
        pehe_key, err_key = point_keys.get(method, ('pehe_raw', 'err_raw'))
        paths = sorted(glob.glob(os.path.join(args.out_root, method, dataset,
                                              f'{dataset}_r*.npz')))
        if not paths:
            sys.exit(f'FATAL: no NPZs found for {method}/{dataset}')
        _debug_single(paths[0], pehe_key, err_key)
        return

    if not os.path.isdir(args.out_root):
        sys.exit(f'FATAL: --out-root not found: {args.out_root}')

    # uwyk1d's point keys are suffixed by anc-tag — resolve via --uwyk-tag.
    point_keys = dict(_POINT_KEYS)
    point_keys['uwyk1d'] = (f'pehe_raw_{args.uwyk_tag}', f'err_raw_{args.uwyk_tag}')

    big_pehe = {'CPS', 'PSID', 'PSID_bal'}
    big_len  = big_pehe

    header = '| Method | ' + ' | '.join(DATASETS) + ' |'
    sep    = '|' + '|'.join(['---'] * (1 + len(DATASETS))) + '|'
    lines = [
        f'\nRealCause density-CI — {args.out_root}',
        '',
        '(each cell, top → bottom: √PEHE, ε_ATE, Coverage, Length; '
        'ALL derived from the CATE density via convolution p_y1 * flip(p_y0). '
        'CI = 95%, assume Y|do(0) ⊥ Y|do(1); n = realizations)',
        '',
        header, sep,
    ]

    verify_lines = ['', '## Sanity check: max |PEHE_density - PEHE_stored| per cell',
                    '(should be ≈ 0 — density mean equals point CATE by construction)',
                    '', header, sep]

    for method in args.methods:
        method_dir = os.path.join(args.out_root, method)
        pehe_key, err_key = point_keys.get(method, ('pehe_raw', 'err_raw'))
        cells = [method]
        verify_cells = [method]
        for d in DATASETS:
            got = summarize_method_dataset(method_dir, d, pehe_key, err_key)
            if got is None:
                cells.append('—')
                verify_cells.append('—')
                continue
            pehe_str = _fmt(*got['pehe'], big=d in big_pehe)
            err_str  = _fmt(*got['err'],  big=False)
            cov_str  = _fmt(*got['cov'],  big=False)
            len_str  = _fmt(*got['len'],  big=d in big_len)
            n = got['n']
            cells.append(
                f'PEHE {pehe_str}<br>'
                f'ε_ATE {err_str}<br>'
                f'Cov {cov_str}<br>'
                f'Len {len_str} (n={n})'
            )
            pd_max = got['pehe_max_diff']
            ed_max = got['err_max_diff']
            if np.isfinite(pd_max) or np.isfinite(ed_max):
                _pfmt = 'nan' if not np.isfinite(pd_max) else (f'{pd_max:.2e}' if abs(pd_max) < 1 else f'{pd_max:.4f}')
                _efmt = 'nan' if not np.isfinite(ed_max) else f'{ed_max:.2e}'
                verify_cells.append(f'ΔPEHE {_pfmt}<br>Δε_ATE {_efmt}')
            else:
                verify_cells.append('(no stored keys)')
        lines.append('| ' + ' | '.join(cells) + ' |')
        verify_lines.append('| ' + ' | '.join(verify_cells) + ' |')

    lines.extend(verify_lines)

    md = '\n'.join(lines) + '\n'
    print(md)
    if args.out_md:
        with open(args.out_md, 'w') as f:
            f.write(md)
        print(f'wrote {args.out_md}')


if __name__ == '__main__':
    main()
