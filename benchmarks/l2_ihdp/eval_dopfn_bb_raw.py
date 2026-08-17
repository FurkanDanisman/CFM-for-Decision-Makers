"""Fast IHDP eval for a DoPFN-backbone Ours checkpoint — RAW-MEAN CATE only.

Skips MALC (1D and 2D), skips L2/KL density metrics. Just:
    load ckpt → forward on each IHDP realization → raw marginal means →
    CATE_pred_raw = E[Y1] - E[Y0] → PEHE, eps_ATE against truth.

Use to quickly compare training checkpoints (e.g. steps 30k/40k/50k/60k of
the J=10 DoPFN-backbone training) without paying for MALC.

Usage:
    python eval_dopfn_bb_raw.py \
        --repo      $DEPLOY_ROOT/R-PFN \
        --dopfn     $DEPLOY_ROOT/external/dopfn \
        --causalpfn $DEPLOY_ROOT/external/causalpfn \
        --checkpoint $DEPLOY_ROOT/checkpoints_dopfn_backbone_realj10/step_50000.pt \
        --n-realizations 100
"""
from __future__ import annotations
import argparse, os, sys, time, types, traceback
import numpy as np
import torch


def _install_dopfn_datasets_shim(dopfn_dir):
    if 'datasets' in sys.modules:
        return
    sys.path.insert(0, dopfn_dir)
    ds_mod = types.ModuleType('datasets')
    with open(os.path.join(dopfn_dir, 'datasets/__init__.py')) as fp:
        src = fp.read().split('def load_semi_real')[0]
    exec(src, ds_mod.__dict__)
    sys.modules['datasets'] = ds_mod


def _np(a):
    if isinstance(a, torch.Tensor): return a.numpy()
    return np.asarray(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo',            required=True)
    ap.add_argument('--dopfn',           required=True)
    ap.add_argument('--causalpfn',       required=True)
    ap.add_argument('--checkpoint',      required=True,
                    help='Path to DoPFN-backbone checkpoint (.pt with model_state_dict + config + edges).')
    ap.add_argument('--n-realizations',  type=int, default=100,
                    help='How many IHDP realizations to score (0..N-1). IHDP has 100 total.')
    ap.add_argument('--start-realization', type=int, default=0)
    ap.add_argument('--out', default='',
                    help='Optional .npz with per-realization PEHE / eps_ATE arrays.')
    ap.add_argument('--n-context', type=int, default=0,
                    help='If > 0, subsample this many context rows per realization.')
    args = ap.parse_args()

    # ── Paths ─────────────────────────────────────────────────────────────
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, args.repo)
    sys.path.insert(0, os.path.join(args.repo, 'training_dopfn_base'))
    sys.path.insert(0, _here)

    from true_ihdp import load_ihdp_truth
    from dopfn_backbone_head import DoPFNBackboneWith2DHead
    from losses.BarDistribution2D import unpack_pred

    _install_dopfn_datasets_shim(args.dopfn)
    sys.path.insert(0, args.causalpfn)
    from benchmarks import IHDPDataset

    # ── Load checkpoint ───────────────────────────────────────────────────
    print(f'[load] {args.checkpoint}', flush=True)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt['config']; J = int(cfg['J'])
    edges_np = ckpt['edges'].cpu().numpy()
    bin_width = float(edges_np[1] - edges_np[0])
    centers_scaled = 0.5 * (edges_np[:-1] + edges_np[1:])   # (J,) in scaled [-1, 1]
    step = int(ckpt.get('step', -1))
    print(f'[ckpt] J={J}  edges=[{edges_np[0]:+.2f}, {edges_np[-1]:+.2f}]  '
          f'bw={bin_width:.4f}  step={step}', flush=True)

    model = DoPFNBackboneWith2DHead(dopfn_root=args.dopfn, K=J).eval()
    model.load_state_dict(ckpt['model_state_dict'])

    # ── Iterate over realizations ─────────────────────────────────────────
    pehe_list, eps_ate_list = [], []
    t_all = time.time()
    end = min(args.start_realization + args.n_realizations, 100)
    for r in range(args.start_realization, end):
        t_r = time.time()
        cd, _ = IHDPDataset()[r]
        y_train_full = _np(cd.y_train)
        truth = load_ihdp_truth(r, args.causalpfn, y_train_full)
        y_min = float(truth.y_min); y_rng = float(truth.y_rng)

        # Subsample context if requested
        if args.n_context > 0 and args.n_context < cd.X_train.shape[0]:
            rng = np.random.default_rng(r)
            idx = rng.choice(cd.X_train.shape[0], args.n_context, replace=False)
            X_ctx = _np(cd.X_train)[idx]
            T_ctx = _np(cd.t_train)[idx]
            Y_ctx = _np(cd.y_train)[idx]
        else:
            X_ctx = _np(cd.X_train)
            T_ctx = _np(cd.t_train)
            Y_ctx = _np(cd.y_train)
        X_qry = _np(cd.X_test)

        # Rescale Y to [-1, 1] using truth's y_min / y_rng
        Y_ctx_s = ((Y_ctx.reshape(-1) - y_min) / y_rng * 2.0 - 1.0).astype(np.float32)

        X_ctx_t = torch.from_numpy(X_ctx.astype(np.float32)).unsqueeze(0)
        T_ctx_t = torch.from_numpy(T_ctx.astype(np.float32).reshape(-1, 1)).unsqueeze(0)
        Y_ctx_t = torch.from_numpy(Y_ctx_s.reshape(-1, 1)).unsqueeze(0)
        X_qry_t = torch.from_numpy(X_qry.astype(np.float32)).unsqueeze(0)

        with torch.no_grad():
            pred = model(X_ctx_t, T_ctx_t, Y_ctx_t, X_qry_t)['predictions'][0]

        n_test = X_qry.shape[0]
        cate_pred_scaled = np.zeros(n_test, dtype=np.float64)
        for q in range(n_test):
            p_mat, *_ = unpack_pred(pred[q], J, bin_width)
            pm = p_mat.detach().cpu().numpy().astype(np.float64)
            s = pm.sum()
            if s > 0: pm /= s
            p_marg0 = pm.sum(axis=1)  # marg over Y1 → gives Y0 marginal
            p_marg1 = pm.sum(axis=0)  # marg over Y0 → gives Y1 marginal
            mean0_s = float((centers_scaled * p_marg0).sum())
            mean1_s = float((centers_scaled * p_marg1).sum())
            cate_pred_scaled[q] = mean1_s - mean0_s

        cate_pred_raw = cate_pred_scaled * (y_rng / 2.0)
        true_cate_raw = _np(cd.true_cate).reshape(-1).astype(np.float64)
        pehe = float(np.sqrt(np.mean((cate_pred_raw - true_cate_raw) ** 2)))
        eps_ate = float(abs(cate_pred_raw.mean() - true_cate_raw.mean()))
        pehe_list.append(pehe); eps_ate_list.append(eps_ate)
        print(f'  r={r:3d}  PEHE={pehe:.4f}  eps_ATE={eps_ate:.4f}  '
              f'({time.time()-t_r:.1f}s)', flush=True)

    pehe_arr = np.array(pehe_list)
    eps_arr  = np.array(eps_ate_list)
    print('')
    print(f'== step={step}  n={len(pehe_arr)}  total={time.time()-t_all:.1f}s ==')
    print(f'PEHE     mean={pehe_arr.mean():.4f}  '
          f'std={pehe_arr.std(ddof=1) if len(pehe_arr)>1 else 0:.4f}  '
          f'median={np.median(pehe_arr):.4f}  '
          f'sem={pehe_arr.std(ddof=1)/np.sqrt(len(pehe_arr)) if len(pehe_arr)>1 else 0:.4f}')
    print(f'eps_ATE  mean={eps_arr.mean():.4f}  '
          f'std={eps_arr.std(ddof=1) if len(eps_arr)>1 else 0:.4f}  '
          f'median={np.median(eps_arr):.4f}  '
          f'sem={eps_arr.std(ddof=1)/np.sqrt(len(eps_arr)) if len(eps_arr)>1 else 0:.4f}')

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        np.savez(args.out, pehe=pehe_arr, eps_ate=eps_arr,
                  step=step, checkpoint=args.checkpoint)
        print(f'[save] {args.out}')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
