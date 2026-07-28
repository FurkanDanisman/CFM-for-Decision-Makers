"""Backfill per-query ρ and Var(Y_1 - Y_0 | X) into existing Table-3 npzs.

Rationale
---------
Testing the theory predictions from theory_joint_advantage.tex requires
per-dataset averages of

    ρ(x)  = Corr(Y_do0, Y_do1 | X = x)    (Pearson, on Ours' joint)
    V(x)  = Var(Y_do1 - Y_do0 | X = x)    (from Ours' joint)

computed over the SAME 100 realisations × N_test queries used to produce
Table 3, so that the Δ_marg/Δ_dopfn error bars line up with the ρ/V
error bars. Storing the raw (n_test, J, J) joints in each npz would add
~150 MB per realisation on CPS. Instead we cache only two per-query
scalars — ρ and V — a payload of ~4·n_test bytes per npz.

Reads the same OURS checkpoint that produced the results, re-runs the
model forward pass on each realisation's test queries, and merges

    rho_ours   : (n_test,)  Pearson ρ from p_mat per query
    var_tau_ours: (n_test,)  Var(Y_1 - Y_0 | X = x) from p_mat per query

into the existing npz. Files that already contain both fields are
skipped, so it's safe to interrupt and resume.

Usage (killarney)
-----------------
    sbatch --account=aip-rgrosse \\
        --export=ALL,CHECKPOINT=$ROOT/R-PFN/checkpoints/step_50000_final.pt,\\
                    RESULTS_DIR=$ROOT/results \\
        R-PFN/benchmarks/submit_backfill_rho_v.sbatch

or directly:
    python R-PFN/benchmarks/backfill_rho_v.py \\
        --results-dir  ./results \\
        --repo         $PWD/R-PFN \\
        --checkpoint   $PWD/R-PFN/checkpoints/step_50000_final.pt \\
        --dopfn        $PWD/external/dopfn \\
        --causalpfn    $PWD/external/causalpfn
"""
from __future__ import annotations
import argparse, glob, os, re, sys, time, traceback, types
import numpy as np
import torch

DEVICE = torch.device('cpu')
_FIELDS = {'rho_ours', 'var_tau_ours'}

_FN_RE = re.compile(r'([A-Za-z0-9]+)_r(\d+)\.npz$')


def _needs_backfill(fn):
    existing = set(np.load(fn, allow_pickle=True).files)
    return not _FIELDS.issubset(existing)


def _rho_var_from_pmat(p_mat_np: np.ndarray, edges: np.ndarray):
    p = p_mat_np.astype(np.float64)
    p /= max(p.sum(), 1e-12)
    centers = 0.5 * (edges[:-1] + edges[1:])
    m0 = centers[:, None]; m1 = centers[None, :]
    ey0 = float((p * m0).sum())
    ey1 = float((p * m1).sum())
    v0 = float((p * (m0 - ey0) ** 2).sum())
    v1 = float((p * (m1 - ey1) ** 2).sum())
    cov = float((p * (m0 - ey0) * (m1 - ey1)).sum())
    rho = cov / (np.sqrt(v0 * v1) + 1e-12)
    var_tau = v0 + v1 - 2 * cov
    return float(rho), float(var_tau)


def _extend_npz(fn, extras: dict):
    """Rewrite npz preserving every existing field + adding new ones."""
    with np.load(fn, allow_pickle=True) as f:
        payload = {k: f[k] for k in f.files}
    payload.update(extras)
    tmp = fn + '.tmp'
    np.savez(tmp, **payload)
    os.replace(tmp, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', required=True,
                    help='Directory of Table-3 npzs to backfill in place.')
    ap.add_argument('--repo',        required=True)
    ap.add_argument('--checkpoint',  required=True,
                    help='OURS checkpoint (must match the one that produced '
                         'the CATE fields in the npzs).')
    ap.add_argument('--dopfn',       required=True)
    ap.add_argument('--causalpfn',   required=True)
    ap.add_argument('--datasets',    default='IHDP,ACIC,CPS,PSID,PSIDbal',
                    help='Comma-separated dataset names to process.')
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]

    # ── Do-PFN's datasets shim + CausalPFN benchmarks path ──────────────
    _orig_torch_load = torch.load
    def _p_load(*a, **kw):
        kw.setdefault('weights_only', False); return _orig_torch_load(*a, **kw)
    torch.load = _p_load

    sys.path.insert(0, args.dopfn)
    ds_mod = types.ModuleType('datasets')
    _src = open(os.path.join(args.dopfn, 'datasets/__init__.py')).read().split('def load_semi_real')[0]
    exec(_src, ds_mod.__dict__)
    sys.modules['datasets'] = ds_mod
    sys.path.insert(0, args.causalpfn)

    # ── Ours model ────────────────────────────────────────────────────────
    sys.path.insert(0, args.repo); sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    from models.InterventionalPFN import InterventionalPFN
    from losses.BarDistribution2D import unpack_pred
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    cfg = ckpt['config']; J = cfg['J']
    edges = ckpt['edges'].cpu().numpy()
    bin_width = float(edges[1] - edges[0])
    NUM_FEATURES = cfg['num_features']
    model = InterventionalPFN(
        num_features=NUM_FEATURES, d_model=cfg['d_model'], depth=cfg['depth'],
        heads_feat=cfg['heads'], heads_samp=cfg['heads'], dropout=0.0,
        output_dim=J*J + 9 + 4, hidden_mult=cfg['hidden_mult'],
        normalize_features=True, normalize_treatment=False,
        use_treatment_in_query=False, use_checkpoint=False,
    ).to(DEVICE).eval()
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'[load] OURS J={J}', flush=True)

    # ── Loaders per dataset ──────────────────────────────────────────────
    from benchmarks import (IHDPDataset, ACIC2016Dataset,
                              RealCauseLalondeCPSDataset, RealCauseLalondePSIDDataset)
    LOADERS = {
        'IHDP':    IHDPDataset,
        'ACIC':    ACIC2016Dataset,
        'CPS':     RealCauseLalondeCPSDataset,
        'PSID':    RealCauseLalondePSIDDataset,
        'PSIDbal': RealCauseLalondePSIDDataset,
    }

    def _pad(X, n_feat):
        d = X.shape[1]
        if d < n_feat:
            pad = np.full((X.shape[0], n_feat - d), np.nan, dtype=np.float32)
            return np.concatenate([X.astype(np.float32), pad], axis=1)
        return X.astype(np.float32)[:, :n_feat]

    # ── Scan work list ───────────────────────────────────────────────────
    all_files = []
    for ds in datasets:
        all_files += sorted(glob.glob(os.path.join(args.results_dir, f'{ds}_r*.npz')))
    todo = [fn for fn in all_files if _needs_backfill(fn)]
    print(f'[scan] {len(all_files)} candidates; {len(todo)} need backfill',
          flush=True)

    t0 = time.time()
    n_ok = n_fail = 0
    ds_cache = {}   # cache loader outputs per (ds, realisation)
    for i, fn in enumerate(todo):
        base = os.path.basename(fn); m = _FN_RE.match(base)
        if not m:
            print(f'[skip] weird filename {base}'); n_fail += 1; continue
        ds_name, r = m.group(1), int(m.group(2))
        try:
            key = (ds_name, r)
            if key not in ds_cache:
                if ds_name not in LOADERS:
                    print(f'[skip] unknown dataset {ds_name}'); n_fail += 1; continue
                cd, _ = LOADERS[ds_name]()[r]
                ds_cache[key] = cd
            cd = ds_cache[key]

            X_ctx = _pad(np.asarray(cd.X_train), NUM_FEATURES)
            T_ctx = np.asarray(cd.t_train).astype(np.float32).reshape(-1, 1)
            Y_ctx_raw = np.asarray(cd.y_train).astype(np.float32).reshape(-1, 1)
            X_qry = _pad(np.asarray(cd.X_test), NUM_FEATURES)

            y_min = float(Y_ctx_raw.min()); y_max = float(Y_ctx_raw.max())
            y_rng = max(y_max - y_min, 1e-6)
            Y_ctx = ((Y_ctx_raw - y_min) / y_rng * 2.0 - 1.0).astype(np.float32)

            with torch.no_grad():
                pred = model(torch.from_numpy(X_ctx).unsqueeze(0),
                              torch.from_numpy(T_ctx).unsqueeze(0),
                              torch.from_numpy(Y_ctx).unsqueeze(0),
                              torch.from_numpy(X_qry).unsqueeze(0))['predictions'][0]

            n_test = X_qry.shape[0]
            rho_arr = np.zeros(n_test, dtype=np.float64)
            v_arr   = np.zeros(n_test, dtype=np.float64)
            for q in range(n_test):
                p_mat, *_ = unpack_pred(pred[q], J, bin_width)
                rho_arr[q], v_arr[q] = _rho_var_from_pmat(
                    p_mat.detach().cpu().numpy(), edges,
                )
            # scale variance back to raw y units (undo the [-1, 1] rescaling)
            scale2 = (y_rng / 2.0) ** 2
            v_arr *= scale2

            _extend_npz(fn, {
                'rho_ours':     rho_arr.astype(np.float64),
                'var_tau_ours': v_arr.astype(np.float64),
            })
            n_ok += 1
        except Exception:
            n_fail += 1
            print(f'[fail] {base}'); traceback.print_exc()

        if (i + 1) % 25 == 0 or (i + 1) == len(todo):
            rate = (i + 1) / max(time.time() - t0, 1e-3)
            eta_min = (len(todo) - (i + 1)) / max(rate, 1e-3) / 60.0
            print(f'[progress] {i+1}/{len(todo)}  ok={n_ok} fail={n_fail} '
                  f'rate={rate:.2f}/s  eta={eta_min:.1f} min', flush=True)

    print(f'[done] processed {n_ok + n_fail} files: ok={n_ok} fail={n_fail}',
          flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
