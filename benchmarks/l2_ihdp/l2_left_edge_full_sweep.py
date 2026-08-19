"""Compute LEFT-edge L2 at J=10 and J=100 resolutions across a full IHDP L2
shard sweep. Reads stored p_y0/p_y1 densities from each shard, regenerates
truth per realization, then computes the LEFT-edge L2 for each variant.

Reports mean±SEM across all queries in all shards.

Usage:
    python l2_left_edge_full_sweep.py \\
        --shards-glob "$DEPLOY_ROOT/ihdp_l2_dopfnbb_j10_s150k_B100K1_full.r*.npz" \\
        --repo $DEPLOY_ROOT/R-PFN --causalpfn $DEPLOY_ROOT/external/causalpfn \\
        --checkpoint-dopfn-bb $DEPLOY_ROOT/checkpoints_dopfn_backbone_realj10/step_150000.pt
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards-glob', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--causalpfn', required=True)
    ap.add_argument('--checkpoint-dopfn-bb', required=True,
                     help='Only needed to read J from checkpoint config; not run.')
    ap.add_argument('--dataset', choices=['ihdp', 'acic'], default='ihdp')
    ap.add_argument('--acic-cache-dir', default='')
    ap.add_argument('--dopfn', default='',
                     help='dopfn root — needed for ACIC dataset shim.')
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'l2_ihdp'))
    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'l2_acic'))
    sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    sys.path.insert(0, os.path.join(args.repo, 'MALC', 'Optimal_Transport'))
    sys.path.insert(0, args.repo)
    sys.path.insert(0, args.causalpfn)
    import torch
    from ot_barycenter import wasserstein_barycenter_1d
    if args.dataset == 'ihdp':
        from true_ihdp import (load_ihdp_truth, true_marginals_per_query,
                                true_cate_per_query, true_ate_barycenter,
                                Y_CENTERS, TAU_CENTERS)
        from benchmarks import IHDPDataset
    else:
        from true_acic import (load_acic_truth, true_marginals_per_query,
                                true_cate_per_query, true_ate_barycenter,
                                Y_CENTERS, TAU_CENTERS)
        from benchmarks import ACIC2016Dataset

    # J=10 edges from checkpoint
    ckpt = torch.load(args.checkpoint_dopfn_bb, map_location='cpu', weights_only=False)
    edges_J10 = ckpt['edges'].cpu().numpy()
    J = int(ckpt['config']['J'])
    bin_w_J10 = float(edges_J10[1] - edges_J10[0])
    Y_BIN = float(Y_CENTERS[1] - Y_CENTERS[0])
    TAU_BIN = float(TAU_CENTERS[1] - TAU_CENTERS[0])
    shift_left = -bin_w_J10 / 2.0
    # For J=10 τ resolution: 10 bins covering TAU_CENTERS range.
    tau_edges_J10 = np.linspace(TAU_CENTERS[0] - TAU_BIN/2,
                                 TAU_CENTERS[-1] + TAU_BIN/2, J + 1)
    tau_bin_w_J10 = tau_edges_J10[1] - tau_edges_J10[0]
    # LEFT-edge for τ / ATE: apply the SAME shift magnitude as marginals
    # (bin_w_J10 / 2) — consistent with the physical scale of Y1 and Y0.
    # Previously tried tau_bin_w_J10/2 (=0.30), which was 3× too large and
    # made τ/ATE L2 worse by 30-40% at J=100. This 0.10-shift version treats
    # τ as living on the same physical scale as Y.
    shift_left_tau = shift_left

    def shift_y(d_on_Y):
        p = np.interp(Y_CENTERS, Y_CENTERS - shift_left, d_on_Y, left=0.0, right=0.0)
        s = p.sum() * Y_BIN
        return p / s if s > 0 else p

    def shift_tau(d_on_tau):
        p = np.interp(TAU_CENTERS, TAU_CENTERS - shift_left_tau, d_on_tau, left=0.0, right=0.0)
        s = p.sum() * TAU_BIN
        return p / s if s > 0 else p

    def to_j10_y(density_100):
        p_bin = np.zeros(J)
        for j in range(J):
            mask = (Y_CENTERS >= edges_J10[j]) & (Y_CENTERS < edges_J10[j+1])
            p_bin[j] = np.array(density_100)[mask].sum() * Y_BIN
        total = p_bin.sum()
        return p_bin / total if total > 0 else p_bin

    def to_j10_tau(density_on_tau):
        p_bin = np.zeros(J)
        for j in range(J):
            mask = (TAU_CENTERS >= tau_edges_J10[j]) & (TAU_CENTERS < tau_edges_J10[j+1])
            p_bin[j] = np.array(density_on_tau)[mask].sum() * TAU_BIN
        total = p_bin.sum()
        return p_bin / total if total > 0 else p_bin

    def l2(p, q, dx):
        return float(np.sqrt(np.sum((np.asarray(p) - np.asarray(q))**2) * dx))

    VARIANTS = [
        ('ours_dopfn_bb',         'BB LOGLIN'),
        ('ours_dopfn_bb_old',     'BB OLD'),
        ('ours_dopfn_bb_rawmarg', 'BB RAW'),
        ('ours_dopfn_bb_indep',   'BB INDEP-τ'),
        ('ours_fn50',             'fn=50'),
        ('uwyk_noanc',            'UWYK-NoAnc'),
        ('uwyk_anc',              'UWYK-FullAnc'),
        ('dopfn',                 'Do-PFN'),
    ]

    # Accumulators — J=10 (coarse), J=100 (Y_CENTERS native), J=1000 (upsampled fine)
    METRICS = ['y0_j100', 'y1_j100', 'tau_j100', 'ate_j100',
               'y0_j10',  'y1_j10',  'tau_j10',  'ate_j10',
               'y0_j1000','y1_j1000','tau_j1000','ate_j1000']
    acc = {label: {m: [] for m in METRICS} for _, label in VARIANTS}

    # J=1000 grids — 10× upsampled from Y_CENTERS / TAU_CENTERS
    Y_1000 = np.linspace(Y_CENTERS[0], Y_CENTERS[-1], 1000)
    Y_BIN_1000 = float(Y_1000[1] - Y_1000[0])
    TAU_1000 = np.linspace(TAU_CENTERS[0], TAU_CENTERS[-1], 1000)
    TAU_BIN_1000 = float(TAU_1000[1] - TAU_1000[0])

    def to_1000_y(density_on_Y):
        p = np.interp(Y_1000, Y_CENTERS, density_on_Y, left=0.0, right=0.0)
        s = p.sum() * Y_BIN_1000
        return p / s if s > 0 else p

    def to_1000_tau(density_on_tau):
        p = np.interp(TAU_1000, TAU_CENTERS, density_on_tau, left=0.0, right=0.0)
        s = p.sum() * TAU_BIN_1000
        return p / s if s > 0 else p

    shards = sorted(glob.glob(args.shards_glob))
    if not shards:
        sys.exit(f'no shards match {args.shards_glob}')
    print(f'[load] {len(shards)} shards', flush=True)

    for si, shard_path in enumerate(shards):
        r = int(shard_path.split('.r')[-1].split('.')[0])
        # Load truth for this realization
        if args.dataset == 'ihdp':
            cd, _ = IHDPDataset()[r]
            y_train_full = np.asarray(cd.y_train.detach().cpu()
                                      if hasattr(cd.y_train, 'detach') else cd.y_train)
            truth = load_ihdp_truth(r, args.causalpfn, y_train_full)
        else:
            # ACIC: mirror l2_acic/eval_realization.py
            if args.dopfn:
                from eval_realization import _install_dopfn_datasets_shim  # noqa
                try: _install_dopfn_datasets_shim(args.dopfn)
                except Exception: pass
            cd, _ = ACIC2016Dataset()[r]
            y_train_full = np.asarray(cd.y_train.detach().cpu()
                                      if hasattr(cd.y_train, 'detach') else cd.y_train)
            truth = load_acic_truth(r, y_train_full, cache_dir=(args.acic_cache_dir or None))
        p_y0_true, p_y1_true = true_marginals_per_query(truth)
        p_tau_true = true_cate_per_query(truth)
        p_ate_true = true_ate_barycenter(p_tau_true, wasserstein_barycenter_1d)

        with np.load(shard_path) as z:
            n_q = p_y0_true.shape[0]
            for key, label in VARIANTS:
                py0_key = f'{key}__p_y0'; py1_key = f'{key}__p_y1'
                ptau_key = f'{key}__p_tau'; pate_key = f'{key}__p_ate'
                if py0_key not in z.files: continue
                # ATE (single value, not per-query) — LEFT-edge shift on τ grid
                if pate_key in z.files:
                    ate_L = shift_tau(z[pate_key])
                    acc[label]['ate_j100'].append(l2(ate_L, p_ate_true, TAU_BIN))
                    ate_j10   = to_j10_tau(ate_L)
                    t_ate_j10 = to_j10_tau(p_ate_true)
                    acc[label]['ate_j10'].append(l2(ate_j10/tau_bin_w_J10, t_ate_j10/tau_bin_w_J10, tau_bin_w_J10))
                    acc[label]['ate_j1000'].append(l2(to_1000_tau(ate_L), to_1000_tau(p_ate_true), TAU_BIN_1000))
                for q in range(n_q):
                    # y0 / y1 — LEFT-edge convention
                    p0_L = shift_y(z[py0_key][q])
                    p1_L = shift_y(z[py1_key][q])
                    acc[label]['y0_j100'].append(l2(p0_L, p_y0_true[q], Y_BIN))
                    acc[label]['y1_j100'].append(l2(p1_L, p_y1_true[q], Y_BIN))
                    acc[label]['y0_j10'].append(l2(to_j10_y(p0_L)/bin_w_J10, to_j10_y(p_y0_true[q])/bin_w_J10, bin_w_J10))
                    acc[label]['y1_j10'].append(l2(to_j10_y(p1_L)/bin_w_J10, to_j10_y(p_y1_true[q])/bin_w_J10, bin_w_J10))
                    acc[label]['y0_j1000'].append(l2(to_1000_y(p0_L), to_1000_y(p_y0_true[q]), Y_BIN_1000))
                    acc[label]['y1_j1000'].append(l2(to_1000_y(p1_L), to_1000_y(p_y1_true[q]), Y_BIN_1000))
                    # τ — LEFT-edge shift (per user request 2026-08-19)
                    if ptau_key in z.files:
                        tau_L = shift_tau(z[ptau_key][q])
                        acc[label]['tau_j100'].append(l2(tau_L, p_tau_true[q], TAU_BIN))
                        acc[label]['tau_j10'].append(l2(to_j10_tau(tau_L)/tau_bin_w_J10,
                                                        to_j10_tau(p_tau_true[q])/tau_bin_w_J10,
                                                        tau_bin_w_J10))
                        acc[label]['tau_j1000'].append(l2(to_1000_tau(tau_L), to_1000_tau(p_tau_true[q]), TAU_BIN_1000))
        if (si + 1) % 10 == 0:
            print(f'  processed {si+1}/{len(shards)}', flush=True)

    def _agg(vs):
        arr = np.asarray(vs, dtype=np.float64); arr = arr[np.isfinite(arr)]
        if arr.size == 0: return 'na'
        m = arr.mean(); sem = arr.std(ddof=1)/np.sqrt(arr.size) if arr.size > 1 else 0.0
        return f'{m:.4f}±{sem:.4f}'

    print()
    print(f'══ {args.dataset.upper()} — LEFT-edge y0/y1, center-conv τ/ATE  '
          f'(n_queries pooled across {len(shards)} realizations) ══')
    print(f'{"variant":18s}  '
          f'{"y0 J=10":>16s}  {"y0 J=100":>16s}  {"y0 J=1000":>16s}  '
          f'{"y1 J=10":>16s}  {"y1 J=100":>16s}  {"y1 J=1000":>16s}  '
          f'{"τ J=10":>16s}  {"τ J=100":>16s}  {"τ J=1000":>16s}  '
          f'{"ATE J=10":>16s}  {"ATE J=100":>16s}  {"ATE J=1000":>16s}')
    for _, label in VARIANTS:
        row = f'{label:18s}  '
        for metric in ['y0_j10','y0_j100','y0_j1000',
                        'y1_j10','y1_j100','y1_j1000',
                        'tau_j10','tau_j100','tau_j1000',
                        'ate_j10','ate_j100','ate_j1000']:
            row += f'  {_agg(acc[label][metric]):>16s}'
        print(row)


if __name__ == '__main__':
    main()
