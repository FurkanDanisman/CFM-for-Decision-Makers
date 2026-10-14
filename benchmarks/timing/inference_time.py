"""Inference time of the six RealCause models on ONE dataset: IHDP, realization 0.

Three steps, timed separately (forward: 3 warm-up + 20 timed runs; density steps: 1 + 5; median and
IQR in seconds; time.perf_counter, torch.cuda.synchronize() before each clock read
of a GPU step):

  forward      the model call(s) that produce the predictive output for all query units,
               exactly the call the RealCause harness makes when DENSITY_DUMP=1
  ite_density  per-query ITE (tau) densities for all queries on TAU_CENTERS, through the
               NLL scorer's own routines (nll_from_dumps.build_objects ->
               eval_density_ate.tau_densities: independence convolution for 1D arms,
               joint_tau_density for 2D heads)
  ate_density  W2 barycenter of those per-query densities, normalised
               (ot_barycenter.wasserstein_barycenter_1d -> eval_density_ate._normalise),
               as nll_from_dumps.score_objects does

Nothing is reimplemented. Each model runs in its own subprocess with the environment the
paper's RealCause run gave its harness (benchmarks/cluster/submit_realcause_density_unified.sbatch
+ the realcause_eval/Table1 wrapper). The child imports the harness module, wraps the
harness's own forward function with a recorder, and calls the harness's own main() with
MAX_REAL=1, so realization 0 is loaded, preprocessed and dumped by the harness itself.
The recorded call (same function, same arguments) is then re-run for the forward timing,
and the dump the harness just wrote is read back once (outside the timing) as the input of
the density steps, exactly as nll_from_dumps reads it.

The two UWYK models are timed in the no-ancestral mode only (one forward per pass); the
harness still runs its v3a/v3b passes during the untimed setup call (ANC_MODE=v3ab_only,
as in the paper run).

    python benchmarks/timing/inference_time.py --out-dir revision/results/d262t1
    python benchmarks/timing/inference_time.py --model cpfn2d --out-dir /tmp/it
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_DEFAULT = os.path.dirname(os.path.dirname(_HERE))
DATASET, REALIZATION = 'IHDP', 0
STEPS = ('forward', 'ite_density', 'ate_density')
CSV_COLS = ('model', 'step', 'median_s', 'q1_s', 'q3_s', 'n_context', 'n_query', 'device')

# key -> (row label as in nll_from_dumps.MODELS, nll_from_dumps dump format, adjacency mode)
MODELS = {
    'dopfn1d': ('Do-PFN',         'dopfn1d', None),
    'dopfn2d': ('Do-PFN 2D',      'dopfn2d', None),
    'uwyk1d':  ('UWYK No-Anc',    'uwyk1d',  'noanc'),
    'uwyk2d':  ('UWYK No-Anc 2D', 'uwyk2d',  'noanc'),
    'cpfn1d':  ('CausalPFN-C',    'cpfn1d',  None),
    'cpfn2d':  ('CausalPFN-C 2D', 'cpfn2d',  None),
}

# Every variable any of the six harnesses reads. Cleared before each child so nothing
# leaks from the calling shell (e.g. common.sh exports DOPFN_SOFTMAX_TEMP=0) into a
# model that must not see it.
_HARNESS_ENV = ('DOPFN_CKPT', 'DOPFN_SOFTMAX_TEMP', 'DOPFN_FP32', 'DOPFN_QUERY_T',
                'EVAL_MAX_CONTEXT', 'EVAL_CONTEXT_SEED', 'CKPT', 'CONFIG', 'ANC_MODE',
                'T_ENCODING', 'X_CLIP_QUANTILE', 'DENSITY_ANC_TAG', 'DENSITY_PRIMARY_MODE',
                'Y_SCALING', 'STD_TARGET', 'STD_MODE', 'Y_STD_MODE_EVAL', 'COMPILE', 'SKIP_EM',
                'DENSITY_DUMP', 'MAX_REAL', 'OUT', 'DATASET', 'REPO', 'QUERY_CHUNK',
                'MATCH_UWYK_PADDING', 'T_INTV_OVERRIDE', 'EVAL_DIAG', 'BIAS_EDGE_SCALE',
                'PROPAGATE_ANC', 'INDEP_FEATURES', 'PSID_BAL_SEED', 'CPFN_V0_LOCAL',
                'ALLOW_SCALING_MISMATCH', 'ANC_VARIANT')


# ---------------------------------------------------------------------------
# Parent: one subprocess per model, environment as in the paper's RealCause run
# ---------------------------------------------------------------------------
def model_env(key, a, out_dir):
    """Harness script + environment, mirroring submit_realcause_density_unified.sbatch
    and the realcause_eval/Table1 wrapper of each model."""
    env = {k: v for k, v in os.environ.items() if k not in _HARNESS_ENV}
    # unified sbatch: exported for every model
    env.update(DATASET=DATASET, OUT=out_dir, DENSITY_DUMP='1', SKIP_EM='1', COMPILE='0',
               CAUSALPFN=a.causalpfn, DOPFN_ROOT=a.dopfn_root, UWYK=a.uwyk,
               PYTHONUNBUFFERED='1', MAX_REAL=str(REALIZATION + 1))
    shim = os.path.join(a.repo, 'benchmarks', 'uwyk_table1', 'shims')
    if shim not in env.get('PYTHONPATH', '').split(':'):
        env['PYTHONPATH'] = shim + (f':{env["PYTHONPATH"]}' if env.get('PYTHONPATH') else '')
    b = os.path.join(a.repo, 'benchmarks')
    if key == 'dopfn1d':          # case dopfn_native, no DOPFN_CKPT: the released Do-PFN
        script, ckpts = os.path.join(b, 'eval_scm_case_studies', 'eval_native_dopfn.py'), []
    elif key == 'dopfn2d':        # same harness, our 262k joint_2d weights, temperature off
        script, ckpts = os.path.join(b, 'eval_scm_case_studies', 'eval_native_dopfn.py'), [a.dopfn2d_ckpt]
        env.update(DOPFN_CKPT=a.dopfn2d_ckpt, DOPFN_SOFTMAX_TEMP='0')
    elif key == 'uwyk1d':         # realcause_eval/Table1/uwyk1d.py env.update, sbatch flags
        ckpt = os.path.join(a.uwyk_ckpt_dir, 'best_model.pt')
        cfg = os.path.join(a.uwyk_ckpt_dir, 'best_model_config.yaml')
        script, ckpts = os.path.join(b, 'eval_graph2d', 'eval_uwyk1d_realcause.py'), [ckpt, cfg]
        env.update(CKPT=ckpt, CONFIG=cfg, ANC_MODE='v3ab_only', T_ENCODING='target',
                   EVAL_MAX_CONTEXT='1000', EVAL_CONTEXT_SEED='43', X_CLIP_QUANTILE='0.99',
                   DENSITY_ANC_TAG='noanc')
    elif key == 'uwyk2d':         # realcause_eval/Table1/graph2d.py env.update, sbatch flags
        script, ckpts = os.path.join(b, 'eval_graph2d', 'eval_graph2d_realcause.py'), [a.graph2d_ckpt]
        env.update(CKPT=a.graph2d_ckpt, REPO=a.repo, ANC_MODE='v3ab_only',
                   DENSITY_PRIMARY_MODE='noanc', EVAL_MAX_CONTEXT='1000',
                   EVAL_CONTEXT_SEED='43', X_CLIP_QUANTILE='0.99', Y_SCALING='minmax',
                   STD_TARGET='0.3')
    elif key == 'cpfn1d':         # realcause_eval/Table1/cpfn1d.py env.update, --std-mode pooled
        script, ckpts = os.path.join(b, 'eval_causalpfn2d', 'eval_causalpfn_v0_realcause.py'), [a.cpfn1d_ckpt]
        env.update(CKPT=a.cpfn1d_ckpt, EVAL_MAX_CONTEXT='', EVAL_CONTEXT_SEED='1',
                   STD_MODE='pooled')
    elif key == 'cpfn2d':         # realcause_eval/Table1/cpfn2d.py env.update, --std-mode pooled
        script, ckpts = os.path.join(b, 'eval_causalpfn2d', 'eval_cpfn2d_realcause.py'), [a.cpfn2d_ckpt]
        env.update(CKPT=a.cpfn2d_ckpt, EVAL_MAX_CONTEXT='', EVAL_CONTEXT_SEED='1',
                   STD_MODE='pooled', STD_TARGET='3.0')
    else:
        raise ValueError(key)
    return script, ckpts, env


def run_parent(a):
    keys = list(MODELS) if a.model == 'all' else [a.model]
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.work_dir, exist_ok=True)
    rows, failed = [], []
    for key in keys:
        dump_dir = os.path.join(a.work_dir, key, 'dump')
        rows_csv = os.path.join(a.work_dir, key, 'rows.csv')
        shutil.rmtree(os.path.join(a.work_dir, key), ignore_errors=True)   # never read a stale dump
        os.makedirs(dump_dir)
        script, ckpts, env = model_env(key, a, dump_dir)
        missing = [p for p in [script] + ckpts if not os.path.isfile(p)]
        if missing:
            print(f'[{key}] FAILED: missing {missing}', flush=True)
            failed.append(key)
            continue
        cmd = [sys.executable, '-u', os.path.abspath(__file__), '--child', key,
               '--harness', script, '--rows-csv', rows_csv, '--repo', a.repo,
               '--causalpfn', a.causalpfn, '--n-warmup', str(a.n_warmup), '--n-rep', str(a.n_rep),
               '--n-warmup-density', str(a.n_warmup_density), '--n-rep-density', str(a.n_rep_density)]
        print(f'\n===== [{time.strftime("%F %T")}] {key}: {MODELS[key][0]}  harness={script}', flush=True)
        for k in ('DOPFN_CKPT', 'DOPFN_SOFTMAX_TEMP', 'CKPT', 'CONFIG', 'ANC_MODE', 'STD_MODE',
                  'EVAL_MAX_CONTEXT', 'EVAL_CONTEXT_SEED', 'X_CLIP_QUANTILE', 'T_ENCODING'):
            if k in env:
                print(f'  {k}={env[k]!r}', flush=True)
        st = subprocess.run(cmd, env=env, cwd=a.work_dir).returncode
        if st != 0 or not os.path.isfile(rows_csv):
            print(f'[{key}] FAILED (exit {st})', flush=True)
            failed.append(key)
            continue
        with open(rows_csv) as fh:
            rows += list(csv.DictReader(fh))

    out_csv = os.path.join(a.out_dir, 'inference_time.csv')
    out_md = os.path.join(a.out_dir, 'inference_time.md')
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(rows)
    with open(out_md, 'w') as fh:
        fh.write(markdown(rows, a, failed))
    print(f'\nwrote {out_csv}\nwrote {out_md}', flush=True)
    print(open(out_md).read(), flush=True)
    if failed:
        print(f'FAILED models: {failed}', flush=True)
        return 1
    return 0


def markdown(rows, a, failed):
    by = {(r['model'], r['step']): r for r in rows}
    gpu = next((r['device'] for r in rows if r['step'] == 'forward'), 'n/a')
    cell = lambda r: (f'{float(r["median_s"]):.4g} [{float(r["q1_s"]):.4g}, {float(r["q3_s"]):.4g}]'
                      if r else 'n/a')
    out = [f'### Inference time, RealCause {DATASET} realization {REALIZATION}',
           '',
           f'Median [Q1, Q3] in seconds: forward over {a.n_rep} timed repetitions after {a.n_warmup} '
           f'warm-up runs, density steps over {a.n_rep_density} after {a.n_warmup_density}. Forward on {gpu}; ITE and ATE density steps are the NLL scorer\'s '
           f'numpy routines (CPU). UWYK models: no-ancestral mode only.',
           '',
           '| Model | n_context | n_query | Forward (s) | ITE density (s) | ATE density (s) |',
           '|---|---|---|---|---|---|']
    for key, (label, _, _) in MODELS.items():
        rs = [by.get((label, s)) for s in STEPS]
        if not any(rs):
            continue
        r0 = next(r for r in rs if r)
        out.append(f'| {label} | {r0["n_context"]} | {r0["n_query"]} | '
                   + ' | '.join(cell(r) for r in rs) + ' |')
    if failed:
        out += ['', f'Not timed (failed): {", ".join(failed)}']
    return '\n'.join(out) + '\n'


# ---------------------------------------------------------------------------
# Child: one model, inside the environment the parent built
# ---------------------------------------------------------------------------
def _import_file(path, name):
    """Import a harness script as a module, with its own directory first on sys.path
    (as when it runs as a script). sys.argv is emptied for the import because the
    harnesses parse_known_args() at import time."""
    d = os.path.dirname(os.path.abspath(path))
    if d not in sys.path:
        sys.path.insert(0, d)
    argv, sys.argv = sys.argv, [path]
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


def _recorder(store, fn):
    def wrapper(*args, **kwargs):
        store.append((args, kwargs))
        return fn(*args, **kwargs)
    return wrapper


class _Dump(dict):
    """The dump fully in memory, with the .files attribute build_objects reads."""
    @property
    def files(self):
        return list(self.keys())


def _maxdiff(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.shape != y.shape:
        return f'shape {x.shape} vs {y.shape}'
    return f'{float(np.nanmax(np.abs(x - y))):.3g}'


def _softmax(l):
    l = np.asarray(l, np.float64)
    p = np.exp(l - l.max(axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def setup_child(key, harness):
    """Run the harness's own main() on realization 0 with its forward recorded.
    Returns (forward_fn, n_context, check_fn) where check_fn(forward_output, dump) -> str."""
    import torch  # noqa: F401  (harnesses import it too)
    if key in ('dopfn1d', 'dopfn2d'):
        D = _import_file(harness, '_it_eval_native_dopfn')
        reg = D.model
        dopfn_root, cwd0 = os.environ['DOPFN_ROOT'], os.getcwd()
        if key == 'dopfn1d':
            if D._IS_2D or D._DOPFN_CKPT:
                raise SystemExit('dopfn1d must run the released Do-PFN (no DOPFN_CKPT)')
            calls = []
            orig = reg.predict_full                       # bound method
            reg.predict_full = _recorder(calls, orig)     # instance attribute shadows it
            D.main()
            del reg.predict_full
            # evaluate(): predict_cate (2 predict_full calls) then the density-dump pair
            # full0 = predict_full(X0_t), full1 = predict_full(X1_t)
            # (eval_native_dopfn.py:451, :455-460). The last two calls are that pair.
            if len(calls) != 4:
                raise SystemExit(f'expected 4 predict_full calls in evaluate(), got {len(calls)}')
            X0, X1 = calls[2][0][0].clone(), calls[3][0][0].clone()
            if not (bool((X0[:, 0] == 0).all()) and bool((X1[:, 0] == 1).all())):
                raise SystemExit('recorded predict_full pair is not do(0), do(1)')

            def forward():
                os.chdir(dopfn_root)                      # evaluate() runs it from DOPFN_ROOT
                try:
                    return reg.predict_full(X0), reg.predict_full(X1)
                finally:
                    os.chdir(cwd0)

            def check(out, z):
                return (f'p_y0 {_maxdiff(_softmax(out[0]["logits"]), z["p_y0_scaled"])}  '
                        f'p_y1 {_maxdiff(_softmax(out[1]["logits"]), z["p_y1_scaled"])}')
            return forward, int(reg.X_.shape[0]), check
        if not D._IS_2D:
            raise SystemExit('dopfn2d: DOPFN_CKPT is not a joint_2d checkpoint')
        if os.environ.get('DOPFN_SOFTMAX_TEMP') != '0' or float(reg.softmax_temperature) != 0.0:
            raise SystemExit('dopfn2d must run with DOPFN_SOFTMAX_TEMP=0')
        calls = []
        orig = D._predict_joint2d
        D._predict_joint2d = _recorder(calls, orig)       # evaluate() resolves it as a global
        D.main()
        D._predict_joint2d = orig
        if len(calls) != 1:
            raise SystemExit(f'expected 1 _predict_joint2d call, got {len(calls)}')
        args, kwargs = calls[0]

        def forward():
            os.chdir(dopfn_root)
            try:
                return orig(*args, **kwargs)
            finally:
                os.chdir(cwd0)

        def check(out, z):
            return f'logits_2d {_maxdiff(out[1]["logits_2d"], z["logits_2d"])}'
        return forward, int(reg.X_.shape[0]), check

    if key in ('uwyk1d', 'uwyk2d'):
        if key == 'uwyk1d':
            M = _import_file(harness, '_it_eval_uwyk1d_realcause')
            H, fwd_name = M.H, 'cate_from_uwyk'           # M.evaluate calls H.build_mode_list
        else:
            M = _import_file(harness, '_it_eval_graph2d_realcause')
            H, fwd_name = M, 'marginals_from_forward'
        modes, calls = [], []
        orig_bml, orig_fwd = H.build_mode_list, getattr(M, fwd_name)

        def rec_bml(*a, **k):
            r = orig_bml(*a, **k)
            modes.append(r)
            return r
        H.build_mode_list = rec_bml
        setattr(M, fwd_name, _recorder(calls, orig_fwd))
        M.main()
        H.build_mode_list = orig_bml
        setattr(M, fwd_name, orig_fwd)
        if len(modes) != 1 or len(calls) != len(modes[0]):
            raise SystemExit(f'{key}: {len(modes)} mode lists, {len(calls)} forward calls')
        tags = [m for m, _ in modes[0]]
        i = tags.index('noanc')
        args, kwargs = calls[i]
        if args[5] is not modes[0][i][1]:                 # (.., X_te, adj, ..): adj is arg 5
            raise SystemExit(f'{key}: forward call {i} did not receive the noanc adjacency')
        print(f'[child] {key}: modes {tags}; timing call {i} (noanc)', flush=True)

        def forward():
            return orig_fwd(*args, **kwargs)

        if key == 'uwyk1d':
            def check(out, z):
                K = int(z['num_bars'])
                return (f'p_y0_noanc {_maxdiff(_softmax(out[1][..., :K + 2]), z["p_y0_scaled_noanc"])}  '
                        f'p_y1_noanc {_maxdiff(_softmax(out[2][..., :K + 2]), z["p_y1_scaled_noanc"])}')
        else:
            def check(out, z):
                return f'logits_2d_noanc {_maxdiff(out[2], z["logits_2d_noanc"])}'
        return forward, int(args[1].shape[0]), check

    if key in ('cpfn1d', 'cpfn2d'):
        if key == 'cpfn1d':
            M, fwd_name = _import_file(harness, '_it_eval_causalpfn_v0_realcause'), 'cate_raw_and_em'
        else:
            M, fwd_name = _import_file(harness, '_it_eval_cpfn2d_realcause'), 'forward_pmats'
        calls, orig = [], getattr(M, fwd_name)
        setattr(M, fwd_name, _recorder(calls, orig))
        M.main()
        setattr(M, fwd_name, orig)
        if len(calls) != 1:
            raise SystemExit(f'{key}: expected 1 {fwd_name} call, got {len(calls)}')
        args, kwargs = calls[0]

        def forward():
            return orig(*args, **kwargs)

        if key == 'cpfn1d':
            def check(out, z):
                return (f'p_y0 {_maxdiff(out[2]["p_y0_scaled"], z["p_y0_scaled"])}  '
                        f'p_y1 {_maxdiff(out[2]["p_y1_scaled"], z["p_y1_scaled"])}')
        else:
            def check(out, z):
                return f'logits_2d {_maxdiff(out[1], z["logits_2d"])}'
        return forward, int(args[1].shape[0]), check
    raise ValueError(key)


def time_step(fn, gpu, n_warmup, n_rep):
    import torch
    sync = torch.cuda.synchronize if gpu and torch.cuda.is_available() else (lambda: None)
    for _ in range(n_warmup):
        fn()
    ts = []
    for _ in range(n_rep):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return np.asarray(ts)


def run_child(a):
    key = a.child
    label, fmt, mode = MODELS[key]
    forward, n_context, check = setup_child(key, a.harness)

    # The scorer, imported only now so the harness set up sys.path exactly as in its own run.
    N = _import_file(os.path.join(a.repo, 'benchmarks', 'eval_graph2d', 'nll_from_dumps.py'),
                     '_it_nll_from_dumps')
    found = N._realization_files(os.environ['OUT'])
    if set(found) != {REALIZATION}:
        raise SystemExit(f'{key}: expected one dump for r={REALIZATION} in {os.environ["OUT"]}, '
                         f'found {sorted(found)}')
    with np.load(found[REALIZATION], allow_pickle=True) as zf:   # read once, outside the timing
        z = _Dump({k: zf[k] for k in zf.files})

    # Truth and working axis exactly as nll_from_dumps._work / load_truth_raw.
    acic_cache = os.environ.get('ACIC_CACHE_DIR', os.path.join(a.repo, 'data', 'acic_cache'))
    truth = N.load_truth_raw(DATASET, REALIZATION, a.causalpfn, acic_cache)
    tc = np.asarray(z['true_cate_per_query'], np.float64).reshape(-1)
    cate = truth['mu1'] - truth['mu0']
    if tc.shape != cate.shape or float(np.abs(tc - cate).max()) > 1e-3 * max(np.abs(cate).max(), 1e-9):
        raise SystemExit(f'{key}: dump queries misaligned with the density truth')
    shift_w, scale_w = truth['axis']
    n_query = int(tc.size)

    import torch
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    out = forward()
    print(f'[child] {key}: re-run forward vs dumped arrays, max |diff|: {check(out, z)}', flush=True)

    def ite():
        family, kind, per_query = N.build_objects(fmt, mode, z, shift_w, scale_w)
        return N.tau_densities(family, kind, per_query, N.N_Y0)

    pt = ite()
    if pt.shape != (n_query, N.TAU_CENTERS.size):
        raise SystemExit(f'{key}: ITE densities have shape {pt.shape}')

    def ate():
        return N._normalise(N.wasserstein_barycenter_1d(pt, N.TAU_CENTERS))

    # Same L_ATE line as nll_from_dumps.score_objects, so the log can be checked
    # against revision/results/d262t1/nll_IHDP.csv (realization 0).
    p_ate = ate()
    at = float(np.interp(float(cate.mean()) / scale_w, N.TAU_CENTERS, p_ate))
    l_ate = float('inf') if at <= 0 else float(N.work_to_raw_nll(-np.log(at), scale_w, 1))
    print(f'[child] {key}: n_context={n_context} n_query={n_query}  L_ATE(r0)={l_ate:.4f} '
          f'(tau mass min {float(np.trapezoid(pt, N.TAU_CENTERS, axis=1).min()):.4f})', flush=True)

    rows = []
    for step, fn, gpu in (('forward', forward, True), ('ite_density', ite, False),
                          ('ate_density', ate, False)):
        t0 = time.time()
        # GPU forward: n_warmup/n_rep; the deterministic CPU density steps: n_warmup_density/n_rep_density
        nw, nr = (a.n_warmup, a.n_rep) if gpu else (a.n_warmup_density, a.n_rep_density)
        ts = time_step(fn, gpu, nw, nr)
        q1, med, q3 = np.percentile(ts, [25, 50, 75])
        rows.append(dict(model=label, step=step, median_s=f'{med:.6g}', q1_s=f'{q1:.6g}',
                         q3_s=f'{q3:.6g}', n_context=n_context, n_query=n_query,
                         device=device if gpu else 'cpu'))
        print(f'[child] {key} {step:12s} median={med:.4g}s IQR=[{q1:.4g}, {q3:.4g}]  '
              f'({time.time() - t0:.0f}s wall)', flush=True)
    with open(a.rows_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(rows)
    return 0


def main():
    kit = os.environ.get('KIT', '/scratch/furkanbd/rpfn_bench_kit')
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', default='all', choices=['all'] + list(MODELS))
    ap.add_argument('--out-dir', default='')
    ap.add_argument('--work-dir', default='',
                    help='harness dumps and per-model rows (default $SCRATCH/inference_time_work)')
    ap.add_argument('--n-warmup', type=int, default=3)
    ap.add_argument('--n-rep', type=int, default=20)
    ap.add_argument('--n-warmup-density', type=int, default=1)
    ap.add_argument('--n-rep-density', type=int, default=5)
    ap.add_argument('--kit', default=kit)
    ap.add_argument('--repo', default=_REPO_DEFAULT)
    ap.add_argument('--causalpfn', default='')
    ap.add_argument('--dopfn-root', default='')
    ap.add_argument('--uwyk', default='')
    ap.add_argument('--uwyk-ckpt-dir', default='')
    ap.add_argument('--dopfn2d-ckpt', default='')
    ap.add_argument('--graph2d-ckpt', default='')
    ap.add_argument('--cpfn1d-ckpt', default='')
    ap.add_argument('--cpfn2d-ckpt', default='')
    # child mode (internal)
    ap.add_argument('--child', default='', choices=[''] + list(MODELS), help=argparse.SUPPRESS)
    ap.add_argument('--harness', default='', help=argparse.SUPPRESS)
    ap.add_argument('--rows-csv', default='', help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        return run_child(a)

    if not a.out_dir:
        ap.error('--out-dir is required')
    # Defaults: the unified sbatch's ($DEPLOY_ROOT = kit, $REPO = kit/R-PFN), the D262k
    # common.sh 262k checkpoint, and 15_submit_t1.sh's eta0 CKPT_CPFN2D.
    a.causalpfn = a.causalpfn or os.environ.get('CAUSALPFN') or f'{a.kit}/external/causalpfn'
    a.dopfn_root = a.dopfn_root or os.environ.get('DOPFN_ROOT') or f'{a.kit}/external/dopfn'
    a.uwyk = a.uwyk or os.environ.get('UWYK') or f'{a.kit}/external/uwyk_reproduce'
    a.uwyk_ckpt_dir = a.uwyk_ckpt_dir or (f'{a.uwyk}/experiments/checkpoints/full_conditioned_model/'
                                          'final_earlytest_full_conditioning_16773252.0')
    a.dopfn2d_ckpt = a.dopfn2d_ckpt or f'{a.repo}/Required_checkpoints/dopfn_repro_joint2d_step262144.pt'
    a.graph2d_ckpt = a.graph2d_ckpt or f'{a.repo}/Required_checkpoints/graph2d_step_50000.pt'
    # Required_checkpoints copy of the unified sbatch's CKPT_CPFN1D default
    # (causalpfn_j1024_headrand_A1_h100/step_checkpoints/run/step_0050000.pt).
    a.cpfn1d_ckpt = a.cpfn1d_ckpt or f'{a.repo}/Required_checkpoints/cpfn1d_j1024_headrand_step_50000.pt'
    a.cpfn2d_ckpt = a.cpfn2d_ckpt or f'{a.repo}/Required_checkpoints/cpfn2d_j32_eta0_y01_step50000.pt'
    a.work_dir = a.work_dir or os.path.join(os.environ.get('SCRATCH', '/tmp'), 'inference_time_work')
    a.out_dir, a.work_dir = os.path.abspath(a.out_dir), os.path.abspath(a.work_dir)
    print('[parent] ' + json.dumps({k: v for k, v in vars(a).items()
                                    if k not in ('child', 'harness', 'rows_csv')}, indent=1), flush=True)
    return run_parent(a)


if __name__ == '__main__':
    sys.exit(main())
