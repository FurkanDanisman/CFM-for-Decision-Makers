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
    pehe_em_k1_list, eps_em_k1_list = [], []
    std_y0_list, std_y1_list = [], []
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
        cate_raw_scaled     = np.zeros(n_test, dtype=np.float64)   # E[Y1] - E[Y0], raw
        cate_em_k1_scaled   = np.zeros(n_test, dtype=np.float64)   # K=1 Gaussian MLE mean
        std_em_k1_y0_scaled = np.zeros(n_test, dtype=np.float64)   # per-arm fitted σ
        std_em_k1_y1_scaled = np.zeros(n_test, dtype=np.float64)

        for q in range(n_test):
            p_mat, *_ = unpack_pred(pred[q], J, bin_width)
            pm = p_mat.detach().cpu().numpy().astype(np.float64)
            s = pm.sum()
            if s > 0: pm /= s
            p_marg0 = pm.sum(axis=1)  # marg over Y1 → gives Y0 marginal
            p_marg1 = pm.sum(axis=0)  # marg over Y0 → gives Y1 marginal

            # ── raw-mean (weighted expectation) ─────────────────────────
            mean0_s = float((centers_scaled * p_marg0).sum())
            mean1_s = float((centers_scaled * p_marg1).sum())
            cate_raw_scaled[q] = mean1_s - mean0_s

            # ── EM K=1 Gaussian MLE ─────────────────────────────────────
            # For K=1 the closed-form MLE for (μ, σ²) given discrete probs
            # p_i on bin centers c_i is exactly:
            #     μ_k1 = Σ p_i · c_i       (identical to the raw mean)
            #     σ_k1 = √( Σ p_i · (c_i − μ)² )
            # Reported separately for paper-table completeness; σ is new
            # info (raw-mean has no width companion).
            cate_em_k1_scaled[q] = mean1_s - mean0_s   # ≡ raw
            std_em_k1_y0_scaled[q] = float(np.sqrt(((centers_scaled - mean0_s) ** 2 * p_marg0).sum()))
            std_em_k1_y1_scaled[q] = float(np.sqrt(((centers_scaled - mean1_s) ** 2 * p_marg1).sum()))

        cate_pred_raw    = cate_raw_scaled   * (y_rng / 2.0)
        cate_pred_em_k1  = cate_em_k1_scaled * (y_rng / 2.0)
        std_em_k1_y0_raw = std_em_k1_y0_scaled * (y_rng / 2.0)
        std_em_k1_y1_raw = std_em_k1_y1_scaled * (y_rng / 2.0)

        true_cate_raw = _np(cd.true_cate).reshape(-1).astype(np.float64)
        pehe        = float(np.sqrt(np.mean((cate_pred_raw   - true_cate_raw) ** 2)))
        pehe_em_k1  = float(np.sqrt(np.mean((cate_pred_em_k1 - true_cate_raw) ** 2)))
        eps_ate       = float(abs(cate_pred_raw.mean()   - true_cate_raw.mean()))
        eps_ate_em_k1 = float(abs(cate_pred_em_k1.mean() - true_cate_raw.mean()))
        pehe_list.append(pehe); eps_ate_list.append(eps_ate)
        # Also collect em_k1 arrays for the summary
        try:
            pehe_em_k1_list.append(pehe_em_k1); eps_em_k1_list.append(eps_ate_em_k1)
            std_y0_list.append(float(std_em_k1_y0_raw.mean()))
            std_y1_list.append(float(std_em_k1_y1_raw.mean()))
        except NameError:
            pass
        print(f'  r={r:3d}  PEHE(raw)={pehe:.4f}  PEHE(em_k1)={pehe_em_k1:.4f}  '
              f'eps_ATE(raw)={eps_ate:.4f}  eps_ATE(em_k1)={eps_ate_em_k1:.4f}  '
              f'<σ_Y0>={std_em_k1_y0_raw.mean():.3f}  <σ_Y1>={std_em_k1_y1_raw.mean():.3f}  '
              f'({time.time()-t_r:.1f}s)', flush=True)

    def _summary(arr, label):
        arr = np.asarray(arr)
        n = len(arr)
        m = arr.mean() if n else float('nan')
        s = arr.std(ddof=1) if n > 1 else 0.0
        sem = s / np.sqrt(n) if n > 1 else 0.0
        med = float(np.median(arr)) if n else float('nan')
        print(f'{label:<20s}  mean={m:.4f}  std={s:.4f}  median={med:.4f}  sem={sem:.4f}')

    pehe_arr        = np.array(pehe_list)
    eps_arr         = np.array(eps_ate_list)
    pehe_em_k1_arr  = np.array(pehe_em_k1_list)
    eps_em_k1_arr   = np.array(eps_em_k1_list)
    std_y0_arr      = np.array(std_y0_list)
    std_y1_arr      = np.array(std_y1_list)
    print('')
    print(f'== step={step}  n={len(pehe_arr)}  total={time.time()-t_all:.1f}s ==')
    _summary(pehe_arr,        'PEHE (raw)')
    _summary(pehe_em_k1_arr,  'PEHE (em_k1)')
    _summary(eps_arr,         'eps_ATE (raw)')
    _summary(eps_em_k1_arr,   'eps_ATE (em_k1)')
    _summary(std_y0_arr,      'fitted σ Y_do(0)')
    _summary(std_y1_arr,      'fitted σ Y_do(1)')

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        np.savez(args.out,
                  pehe=pehe_arr, eps_ate=eps_arr,
                  pehe_em_k1=pehe_em_k1_arr, eps_ate_em_k1=eps_em_k1_arr,
                  std_em_k1_y0=std_y0_arr, std_em_k1_y1=std_y1_arr,
                  step=step, checkpoint=args.checkpoint)
        print(f'[save] {args.out}')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
