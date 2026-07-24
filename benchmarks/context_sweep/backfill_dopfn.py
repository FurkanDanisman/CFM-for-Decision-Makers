"""Add Do-PFN predictions to existing sweep npz files in place.

For each existing `<source>_seed<seed>_N<N>.npz` that lacks the dopfn
fields, this script:

  1. Re-samples the SAME SCM (deterministic given `source`, `seed`,
     `N_context`, `n_test=50`).
  2. Loads Do-PFN and runs its `predict_cate` pipeline.
  3. Rewrites the npz to include `pehe_dopfn`, `err_dopfn`, `ate_dopfn`
     alongside every other existing field.

Existing fields are preserved verbatim. Files that already contain the
Do-PFN fields are skipped, so it's safe to interrupt and resume.

Usage
-----
    python benchmarks/context_sweep/backfill_dopfn.py \\
        --results-dir  ./results_sweep \\
        --repo         $PWD/R-PFN \\
        --dopfn        $PWD/external/dopfn \\
        --causalpfn    $PWD/external/causalpfn \\
        --uwyk-src     $PWD/external/uwyk/src \\
        --source       poly
"""
from __future__ import annotations
import argparse, glob, os, re, sys, time, traceback, types
import numpy as np
import torch
from sklearn.metrics import mean_squared_error

_HERE  = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
sys.path.insert(0, _BENCH)
sys.path.insert(0, _HERE)
from methods.dopfn import dopfn_pipeline


NAME_RE = re.compile(r'^(?P<src>prior|poly)_seed(?P<seed>\d+)_N(?P<N>\d+)\.npz$')


def _pehe(true_cate, pred_cate):
    return float(np.sqrt(mean_squared_error(true_cate, pred_cate)))


def _ate_relerr(true_cate, pred_cate):
    ta = float(np.mean(true_cate)); pa = float(np.mean(pred_cate))
    if abs(ta) < 1e-12:
        return 0.0 if abs(pa) < 1e-12 else float('inf')
    return abs(ta - pa) / abs(ta)


def _load_dopfn(dopfn_root):
    """Return the DoPFNRegressor class. Do-PFN's package needs to be importable
    from its own root — inject it plus a synthetic `datasets` module so
    scripts.transformer_prediction_interface.base imports cleanly."""
    sys.path.insert(0, dopfn_root)
    ds_mod = types.ModuleType('datasets')
    with open(os.path.join(dopfn_root, 'datasets', '__init__.py')) as fp:
        _src = fp.read().split('def load_semi_real')[0]
    exec(_src, ds_mod.__dict__)
    sys.modules['datasets'] = ds_mod
    from scripts.transformer_prediction_interface.base import DoPFNRegressor
    return DoPFNRegressor


def _sample_scm(source, seed, N, n_test, uwyk_src, causalpfn_root):
    if source == 'prior':
        if uwyk_src not in sys.path: sys.path.insert(0, uwyk_src)
        from scm_prior import sample_as_cate_dataset
        cd, _ = sample_as_cate_dataset(scm_seed=seed, n_context=N, n_test=n_test)
        return cd
    elif source == 'poly':
        from scm_polynomial import sample_as_cate_dataset
        return sample_as_cate_dataset(scm_seed=seed, n_context=N, n_test=n_test,
                                       causalpfn_root=causalpfn_root)
    raise ValueError(source)


def _has_dopfn_fields(npz_path):
    with np.load(npz_path, allow_pickle=True) as f:
        return {'pehe_dopfn', 'err_dopfn', 'ate_dopfn'} <= set(f.files)


def _extend_npz(npz_path, extras):
    """Rewrite the npz with all existing fields + `extras` merged in."""
    with np.load(npz_path, allow_pickle=True) as f:
        payload = {k: f[k] for k in f.files}
    payload.update(extras)
    tmp_path = npz_path + '.tmp.npz'
    np.savez(tmp_path, **payload)
    os.replace(tmp_path, npz_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir',   required=True)
    ap.add_argument('--repo',          required=True)
    ap.add_argument('--dopfn',         required=True, help='path to Do-PFN repo root')
    ap.add_argument('--uwyk-src',      required=True, help='needed for scm_prior sampler')
    ap.add_argument('--causalpfn',     required=True, help='needed for scm_polynomial')
    ap.add_argument('--n-test',        type=int, default=50)
    ap.add_argument('--source',        choices=['prior', 'poly'], default=None)
    ap.add_argument('--limit',         type=int, default=None)
    ap.add_argument('--dry-run',       action='store_true')
    ap.add_argument('--shard-idx',     type=int, default=0)
    ap.add_argument('--n-shards',      type=int, default=1)
    args = ap.parse_args()
    assert 0 <= args.shard_idx < args.n_shards, "shard-idx must be in [0, n-shards)"

    files = sorted(glob.glob(os.path.join(args.results_dir, '*.npz')))
    print(f"[scan] found {len(files)} npz files", flush=True)

    todo, skipped_have, skipped_badname, skipped_src = [], 0, 0, 0
    for fn in files:
        base = os.path.basename(fn)
        m = NAME_RE.match(base)
        if not m:
            skipped_badname += 1; continue
        if args.source and m.group('src') != args.source:
            skipped_src += 1; continue
        if _has_dopfn_fields(fn):
            skipped_have += 1; continue
        todo.append((fn, m.group('src'), int(m.group('seed')), int(m.group('N'))))

    if args.source:
        print(f"[scan] skipped (other source):     {skipped_src}", flush=True)
    print(f"[scan] already have dopfn fields:  {skipped_have}", flush=True)
    print(f"[scan] non-matching filename:      {skipped_badname}", flush=True)
    print(f"[scan] to process:                 {len(todo)}", flush=True)

    if args.n_shards > 1:
        todo = todo[args.shard_idx::args.n_shards]
        print(f"[shard] {args.shard_idx}/{args.n_shards}: {len(todo)} files", flush=True)

    if args.dry_run:
        return
    if args.limit:
        todo = todo[:args.limit]
        print(f"[scan] limited to first {args.limit}", flush=True)

    print(f"[load] Do-PFN…", flush=True)
    # Do-PFN expects to be imported from its own root; the current working dir
    # is often the caller's, which may not include Do-PFN's packages.
    os.chdir(args.dopfn)
    DoPFNRegressor = _load_dopfn(args.dopfn)
    # Now go back so relative paths in scm samplers still work
    os.chdir(_HERE)

    t0 = time.time()
    n_ok = 0; n_fail = 0
    for i, (fn, src, seed, N) in enumerate(todo):
        try:
            cd = _sample_scm(src, seed, N, args.n_test, args.uwyk_src, args.causalpfn)
            true_cate = cd.true_cate.numpy().reshape(-1) if hasattr(cd.true_cate, 'numpy') \
                        else np.asarray(cd.true_cate).reshape(-1)
            dopfn_cate = dopfn_pipeline(cd, DoPFNRegressor)
            extras = {
                'pehe_dopfn': np.array(_pehe(true_cate, dopfn_cate), dtype=np.float64),
                'err_dopfn':  np.array(_ate_relerr(true_cate, dopfn_cate), dtype=np.float64),
                'ate_dopfn':  np.array(float(np.mean(dopfn_cate)), dtype=np.float64),
            }
            _extend_npz(fn, extras)
            n_ok += 1
        except Exception:
            n_fail += 1
            print(f"[fail] {os.path.basename(fn)}", flush=True)
            traceback.print_exc()

        if (i + 1) % 25 == 0 or (i + 1) == len(todo):
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta_s = (len(todo) - (i + 1)) / max(rate, 1e-6)
            print(f"[progress] {i+1}/{len(todo)}  ok={n_ok} fail={n_fail}  "
                  f"rate={rate:.2f}/s  eta={eta_s/60:.1f} min", flush=True)

    print(f"[done] processed {len(todo)} files: ok={n_ok} fail={n_fail}", flush=True)


if __name__ == '__main__':
    main()
