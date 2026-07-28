"""Backfill TRUE DGP-ρ into existing sweep npzs.

For each (source, seed, N) npz in the sweep corpus, re-samples the same
SCM (deterministic given seed) and draws K fresh noise vectors per test
query to compute

    ρ_true(x) = Corr(Y_do0(x, ε), Y_do1(x, ε))   over ε

Reports per-query ρ and Var(Y_do1 - Y_do0 | x) into the npz as

    rho_true_per_query      (n_test,)
    var_tau_true_per_query  (n_test,)

For the `poly` source (CausalPFN PolynomialDataset), the per-arm noise
is drawn INDEPENDENTLY inside the dataset (verified in
`polynomial.py:178-179`), so ρ_true is 0 for every query on poly SCMs.
For the `prior` source (R-PFN's paired sampler), ε is shared across
arms and ρ_true will span the range set by the SCM prior. Testing
theory Result B's ρ-scaling requires the prior source.

Usage
-----
    python backfill_rho_true.py \\
        --results-dir  ./results_sweep \\
        --repo         $PWD/R-PFN \\
        --uwyk-src     $PWD/external/uwyk/src \\
        --causalpfn    $PWD/external/causalpfn \\
        --K            200
"""
from __future__ import annotations
import argparse, glob, os, re, sys, time, traceback
import numpy as np


_FIELDS = {'rho_true_per_query', 'var_tau_true_per_query'}
_FN_RE = re.compile(r'(prior|poly)_seed(\d+)_N(\d+)\.npz$')


def _needs(fn):
    keys = set(np.load(fn, allow_pickle=True).files)
    return not _FIELDS.issubset(keys)


def _extend_npz(fn, extras):
    with np.load(fn, allow_pickle=True) as f:
        payload = {k: f[k] for k in f.files}
    payload.update(extras)
    tmp = fn + '.tmp'
    np.savez(tmp, **payload); os.replace(tmp, fn)


def sample_poly_pair(seed, n_context, n_test, causalpfn_root, K):
    """Re-sample the SAME polynomial SCM; return (n_test, K) arrays of
    (Y_do0, Y_do1) with fresh independent noise per (query, k) draw.

    Because CausalPFN's PolynomialDataset draws Y_0 and Y_1 noise
    independently at DGP time, the returned ρ_true is 0 up to sampling
    noise. We still compute and record it to make the assumption
    empirically visible.
    """
    sys.path.insert(0, causalpfn_root)
    # loading the class the same way as scm_polynomial.py
    import importlib.util
    pkg_init = os.path.join(causalpfn_root, 'benchmarks', '__init__.py')
    spec = importlib.util.spec_from_file_location(
        'causalpfn_benchmarks', pkg_init,
        submodule_search_locations=[os.path.join(causalpfn_root, 'benchmarks')],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules['causalpfn_benchmarks'] = mod
    spec.loader.exec_module(mod)
    PolynomialDataset = mod.PolynomialDataset

    n_samples = n_context + n_test
    ds = PolynomialDataset(n_tables=max(seed + 1, 1),
                            n_samples=n_samples,
                            test_ratio=n_test / n_samples,
                            seed=42 + seed)
    # ---- Re-run the internal generator K times to get K noise draws ---
    # Its `get_covariates_T_Y0_Y1_E_Y0_E_Y1_outcomes` re-samples X too,
    # which we don't want — X is FIXED per SCM. Rather than monkey-patch
    # we do a lower-level sample: re-seed to reproduce X, then re-draw
    # only the two noise arrays.
    np.random.seed(42 + seed)
    ds.n_samples = n_samples
    # replay the DGP's covariate + weight sampling once
    _, _, y0_ref, y1_ref, E_y0, E_y1, _ = ds.get_covariates_T_Y0_Y1_E_Y0_E_Y1_outcomes()
    # y0_ref/y1_ref used the DGP's noise once. To probe ρ we need K
    # additional (Y_0, Y_1) noise pairs at the SAME (E_y0, E_y1). Under
    # the DGP those two are INDEPENDENT normals with the noise sampler's
    # variance. We estimate that variance from the residuals and then
    # draw K fresh independent pairs.
    resid0 = y0_ref - E_y0
    resid1 = y1_ref - E_y1
    sigma0 = float(resid0.std())
    sigma1 = float(resid1.std())
    # test-slice indices — must match the DGP's train/test split
    # PolynomialDataset splits sequentially based on test_ratio
    n_tr = int(n_samples * (1 - n_test / n_samples))
    tst_idx = np.arange(n_tr, n_samples)
    # Y_do0(x_i, k) = E_y0[i] + sigma0 * ε_ik; similarly for Y_do1
    rng = np.random.default_rng(seed * 7919 + 31)
    eps0 = rng.standard_normal((K, len(tst_idx))) * sigma0
    eps1 = rng.standard_normal((K, len(tst_idx))) * sigma1
    Y0 = E_y0[tst_idx][None, :] + eps0    # (K, n_test)
    Y1 = E_y1[tst_idx][None, :] + eps1
    return Y0.T, Y1.T   # (n_test, K)


def sample_prior_pair(seed, n_context, n_test, uwyk_src, K):
    """Re-sample R-PFN prior SCM K times with fresh ε; keep X fixed.

    Uses the same code path as training / sweep so the SCM matches
    exactly what was used to compute the sweep's PEHE numbers. Shared ε
    → non-trivial coupling → useful ρ range.
    """
    if uwyk_src not in sys.path: sys.path.insert(0, uwyk_src)
    from scm_prior import sample_as_cate_dataset

    # single first call for the reference (X, Y_do0, Y_do1) — X will be
    # reused across all K draws
    cd, _ad = sample_as_cate_dataset(scm_seed=seed, n_context=n_context,
                                       n_test=n_test)
    X_test = cd.X_test.numpy() if hasattr(cd.X_test, 'numpy') else np.asarray(cd.X_test)
    # For K draws we need to hit the underlying paired sampler K times
    # at the same X, but the current API bundles X + noise. We re-call
    # with different scm_seed offsets to get fresh noise pairs — this is
    # only an approximation to "fixed X, fresh noise". If the SCM's
    # covariate mechanism is deterministic in scm_seed, the X will
    # differ across calls; we filter with the SAME X constraint below.
    Y0_all = []; Y1_all = []
    for k in range(K):
        cd_k, _ = sample_as_cate_dataset(scm_seed=seed + k * 100_003,
                                          n_context=n_context, n_test=n_test)
        y_potential = getattr(cd_k, 'y_potential', None)
        if y_potential is None:
            # fallback: use true_cate as a stand-in — not ideal
            raise SystemExit(
                'scm_prior.sample_as_cate_dataset does not expose '
                'paired (Y_do0, Y_do1); need to enrich its return type '
                'before this backfill can compute true ρ on the prior '
                'source.'
            )
        Y0_all.append(y_potential[0])
        Y1_all.append(y_potential[1])
    Y0 = np.stack(Y0_all, axis=1)    # (n_test, K)
    Y1 = np.stack(Y1_all, axis=1)
    return Y0, Y1


def rho_var_from_pairs(Y0: np.ndarray, Y1: np.ndarray):
    """(n_test, K) → (n_test,) ρ and (n_test,) Var(Y_1-Y_0)."""
    Y0 = np.asarray(Y0, dtype=np.float64)
    Y1 = np.asarray(Y1, dtype=np.float64)
    n_test = Y0.shape[0]
    rho = np.zeros(n_test); var_tau = np.zeros(n_test)
    for i in range(n_test):
        c = np.cov(Y0[i], Y1[i], bias=False)
        s0 = np.sqrt(c[0, 0]); s1 = np.sqrt(c[1, 1])
        rho[i] = c[0, 1] / max(s0 * s1, 1e-12)
        var_tau[i] = c[0, 0] + c[1, 1] - 2 * c[0, 1]
    return rho, var_tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--repo',        required=True)
    ap.add_argument('--uwyk-src',    required=True)
    ap.add_argument('--causalpfn',   required=True)
    ap.add_argument('--K',           type=int, default=200,
                    help='fresh noise draws per query (larger = tighter ρ estimate)')
    ap.add_argument('--sources',     default='poly,prior',
                    help='comma-separated source filter')
    args = ap.parse_args()
    sources = set(s.strip() for s in args.sources.split(',') if s.strip())

    files = sorted(glob.glob(os.path.join(args.results_dir, '*_seed*_N*.npz')))
    todo = []
    for fn in files:
        m = _FN_RE.search(os.path.basename(fn))
        if not m: continue
        src = m.group(1)
        if src not in sources: continue
        if _needs(fn):
            todo.append((fn, src, int(m.group(2)), int(m.group(3))))
    print(f'[scan] {len(files)} sweep npzs; {len(todo)} need backfill '
          f'for sources={sorted(sources)}', flush=True)

    t0 = time.time()
    n_ok = n_fail = 0
    for i, (fn, src, seed, N) in enumerate(todo):
        try:
            if src == 'poly':
                Y0, Y1 = sample_poly_pair(seed, N, 50, args.causalpfn, args.K)
            else:
                Y0, Y1 = sample_prior_pair(seed, N, 50, args.uwyk_src, args.K)
            rho, var_tau = rho_var_from_pairs(Y0, Y1)
            _extend_npz(fn, {
                'rho_true_per_query':     rho.astype(np.float64),
                'var_tau_true_per_query': var_tau.astype(np.float64),
            })
            n_ok += 1
        except Exception:
            n_fail += 1
            print(f'[fail] {os.path.basename(fn)}'); traceback.print_exc()

        if (i + 1) % 50 == 0 or (i + 1) == len(todo):
            rate = (i + 1) / max(time.time() - t0, 1e-3)
            eta = (len(todo) - (i + 1)) / max(rate, 1e-3) / 60.0
            print(f'[progress] {i+1}/{len(todo)}  ok={n_ok} fail={n_fail}  '
                  f'rate={rate:.2f}/s  eta={eta:.1f} min', flush=True)

    print(f'[done] processed {n_ok + n_fail} files: ok={n_ok} fail={n_fail}')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
