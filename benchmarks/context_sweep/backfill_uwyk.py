"""Add UWYK No-Ancestral predictions to existing sweep npz files in place.

For each existing `<source>_seed<seed>_N<N>.npz` that lacks the uwyk_noanc
fields, this script:

  1. Re-samples the SAME SCM (deterministic given `source`, `seed`, `N_context`,
     `n_test=50`).
  2. Loads the UWYK Ancestral checkpoint (same as the benchmarks), passes it a
     zero adjacency (paper's "no info" convention — 0 = edge unknown, −1 = no
     edge for padded feats), and runs `model.predict` twice.
  3. Rewrites the npz to include `pehe_uwyk_noanc`, `err_uwyk_noanc`,
     `ate_uwyk_noanc` alongside the existing OURS fields.

Existing OURS fields are preserved verbatim. Files that already contain the
UWYK fields are skipped. Safe to interrupt and resume.

Usage
-----
    python benchmarks/context_sweep/backfill_uwyk.py \\
        --results-dir  ./results_sweep \\
        --repo         $PWD/R-PFN \\
        --uwyk-src     $PWD/external/uwyk/src \\
        --uwyk-ckpt-dir $PWD/external/uwyk/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0 \\
        --causalpfn    $PWD/external/causalpfn

The result: same 5728 files, each now with UWYK No-Ancestral columns.
"""
from __future__ import annotations
import argparse, glob, importlib, os, re, sys, time, traceback
import numpy as np
import torch
from sklearn.metrics import mean_squared_error

_HERE  = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
sys.path.insert(0, _BENCH)
sys.path.insert(0, _HERE)
from methods.uwyk import uwyk_no_ancestral_pipeline


NAME_RE = re.compile(r'^(?P<src>prior|poly)_seed(?P<seed>\d+)_N(?P<N>\d+)\.npz$')


def _pehe(true_cate, pred_cate):
    return float(np.sqrt(mean_squared_error(true_cate, pred_cate)))


def _ate_relerr(true_cate, pred_cate):
    ta = float(np.mean(true_cate)); pa = float(np.mean(pred_cate))
    if abs(ta) < 1e-12:
        return 0.0 if abs(pa) < 1e-12 else float('inf')
    return abs(ta - pa) / abs(ta)


def _load_uwyk_model(uwyk_src, uwyk_ckpt_dir):
    # Isolate the model namespace so it doesn't leak our own `models/`.
    _saved = {}
    for name in list(sys.modules):
        if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
            _saved[name] = sys.modules.pop(name)
    sys.path.insert(0, uwyk_src)
    UWYK_pre_mod = importlib.import_module('models.PreprocessingGraphConditionedPFN')
    sys.path.remove(uwyk_src)
    for name in list(sys.modules):
        if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
            del sys.modules[name]
    sys.modules.update(_saved)

    orig_load = torch.load
    def _p_load(*a, **kw):
        kw.setdefault('weights_only', False); return orig_load(*a, **kw)
    torch.load = _p_load

    return UWYK_pre_mod.PreprocessingGraphConditionedPFN(
        config_path=os.path.join(uwyk_ckpt_dir, 'best_model_config.yaml'),
        checkpoint_path=os.path.join(uwyk_ckpt_dir, 'best_model.pt'),
        device='cpu', verbose=False,
    ).load()


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


def _has_uwyk_fields(npz_path):
    with np.load(npz_path, allow_pickle=True) as f:
        return {'pehe_uwyk_noanc', 'err_uwyk_noanc', 'ate_uwyk_noanc'} <= set(f.files)


def _extend_npz(npz_path, extras):
    """Rewrite the npz with all existing fields + `extras` merged in."""
    with np.load(npz_path, allow_pickle=True) as f:
        payload = {k: f[k] for k in f.files}
    payload.update(extras)
    tmp_path = npz_path + '.tmp'
    np.savez(tmp_path, **payload)
    os.replace(tmp_path, npz_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir',   required=True)
    ap.add_argument('--repo',          required=True)
    ap.add_argument('--uwyk-src',      required=True)
    ap.add_argument('--uwyk-ckpt-dir', required=True)
    ap.add_argument('--causalpfn',     required=True)
    ap.add_argument('--n-test',        type=int, default=50,
                    help='must match what run_one.py used (default 50)')
    ap.add_argument('--limit',         type=int, default=None,
                    help='process at most this many files (for smoke tests)')
    ap.add_argument('--dry-run',       action='store_true',
                    help='count what would be processed; do not touch npz files')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, '*.npz')))
    print(f"[scan] found {len(files)} npz files", flush=True)

    todo, skipped_have, skipped_badname = [], 0, 0
    for fn in files:
        base = os.path.basename(fn)
        m = NAME_RE.match(base)
        if not m:
            skipped_badname += 1; continue
        if _has_uwyk_fields(fn):
            skipped_have += 1; continue
        todo.append((fn, m.group('src'), int(m.group('seed')), int(m.group('N'))))

    print(f"[scan] already have UWYK fields: {skipped_have}", flush=True)
    print(f"[scan] non-matching filename:     {skipped_badname}", flush=True)
    print(f"[scan] to process:                {len(todo)}", flush=True)
    if args.dry_run:
        return

    if args.limit:
        todo = todo[:args.limit]
        print(f"[scan] limited to first {args.limit}", flush=True)

    print(f"[load] UWYK model…", flush=True)
    uwyk_model = _load_uwyk_model(args.uwyk_src, args.uwyk_ckpt_dir)

    t0 = time.time()
    n_ok = 0; n_fail = 0
    for i, (fn, src, seed, N) in enumerate(todo):
        try:
            cd = _sample_scm(src, seed, N, args.n_test, args.uwyk_src, args.causalpfn)
            true_cate = cd.true_cate.numpy().reshape(-1) if hasattr(cd.true_cate, 'numpy') \
                        else np.asarray(cd.true_cate).reshape(-1)
            uwyk_cate = uwyk_no_ancestral_pipeline(uwyk_model, cd)
            extras = {
                'pehe_uwyk_noanc': np.array(_pehe(true_cate, uwyk_cate), dtype=np.float64),
                'err_uwyk_noanc':  np.array(_ate_relerr(true_cate, uwyk_cate), dtype=np.float64),
                'ate_uwyk_noanc':  np.array(float(np.mean(uwyk_cate)), dtype=np.float64),
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
