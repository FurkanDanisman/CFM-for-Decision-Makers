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
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'l2_ihdp'))
    sys.path.insert(0, args.repo)
    sys.path.insert(0, args.causalpfn)
    import torch
    from true_ihdp import load_ihdp_truth, true_marginals_per_query, Y_CENTERS
    from benchmarks import IHDPDataset

    # J=10 edges from checkpoint
    ckpt = torch.load(args.checkpoint_dopfn_bb, map_location='cpu', weights_only=False)
    edges_J10 = ckpt['edges'].cpu().numpy()
    J = int(ckpt['config']['J'])
    bin_w_J10 = float(edges_J10[1] - edges_J10[0])
    Y_BIN = float(Y_CENTERS[1] - Y_CENTERS[0])
    shift_left = -bin_w_J10 / 2.0

    def shift(d_on_Y):
        p = np.interp(Y_CENTERS, Y_CENTERS - shift_left, d_on_Y, left=0.0, right=0.0)
        s = p.sum() * Y_BIN
        return p / s if s > 0 else p

    def to_j10(density_100):
        p_bin = np.zeros(J)
        for j in range(J):
            mask = (Y_CENTERS >= edges_J10[j]) & (Y_CENTERS < edges_J10[j+1])
            p_bin[j] = np.array(density_100)[mask].sum() * Y_BIN
        total = p_bin.sum()
        return p_bin / total if total > 0 else p_bin

    def l2(p, q, dx):
        return float(np.sqrt(np.sum((np.asarray(p) - np.asarray(q))**2) * dx))

    VARIANTS = [
        ('ours_dopfn_bb',         'BB LOGLIN'),
        ('ours_dopfn_bb_old',     'BB OLD'),
        ('ours_dopfn_bb_rawmarg', 'BB RAW'),
        ('dopfn',                 'Do-PFN'),
    ]

    # Accumulators
    acc = {label: {'y0_j100': [], 'y1_j100': [], 'y0_j10': [], 'y1_j10': []}
           for _, label in VARIANTS}

    shards = sorted(glob.glob(args.shards_glob))
    if not shards:
        sys.exit(f'no shards match {args.shards_glob}')
    print(f'[load] {len(shards)} shards', flush=True)

    for si, shard_path in enumerate(shards):
        r = int(shard_path.split('.r')[-1].split('.')[0])
        # Load truth for this realization
        cd, _ = IHDPDataset()[r]
        y_train_full = np.asarray(cd.y_train.detach().cpu()
                                  if hasattr(cd.y_train, 'detach') else cd.y_train)
        truth = load_ihdp_truth(r, args.causalpfn, y_train_full)
        p_y0_true, p_y1_true = true_marginals_per_query(truth)

        with np.load(shard_path) as z:
            n_q = p_y0_true.shape[0]
            for key, label in VARIANTS:
                py0_key = f'{key}__p_y0'; py1_key = f'{key}__p_y1'
                if py0_key not in z.files: continue
                for q in range(n_q):
                    # J=100 LEFT-edge L2
                    p0_L = shift(z[py0_key][q])
                    p1_L = shift(z[py1_key][q])
                    acc[label]['y0_j100'].append(l2(p0_L, p_y0_true[q], Y_BIN))
                    acc[label]['y1_j100'].append(l2(p1_L, p_y1_true[q], Y_BIN))
                    # J=10 L2 (downsample both to J=10 probs, compare density)
                    p0_j10 = to_j10(p0_L)
                    p1_j10 = to_j10(p1_L)
                    t0_j10 = to_j10(p_y0_true[q])
                    t1_j10 = to_j10(p_y1_true[q])
                    acc[label]['y0_j10'].append(l2(p0_j10/bin_w_J10, t0_j10/bin_w_J10, bin_w_J10))
                    acc[label]['y1_j10'].append(l2(p1_j10/bin_w_J10, t1_j10/bin_w_J10, bin_w_J10))
        if (si + 1) % 10 == 0:
            print(f'  processed {si+1}/{len(shards)}', flush=True)

    def _agg(vs):
        arr = np.asarray(vs, dtype=np.float64); arr = arr[np.isfinite(arr)]
        if arr.size == 0: return 'na'
        m = arr.mean(); sem = arr.std(ddof=1)/np.sqrt(arr.size) if arr.size > 1 else 0.0
        return f'{m:.4f}±{sem:.4f}'

    print()
    print(f'══ IHDP — LEFT-edge L2 (n_queries pooled across {len(shards)} realizations) ══')
    print(f'{"variant":18s}  {"y0 J=10":>16s}  {"y0 J=100":>16s}  '
          f'{"y1 J=10":>16s}  {"y1 J=100":>16s}')
    for _, label in VARIANTS:
        row = f'{label:18s}  '
        for metric in ['y0_j10', 'y0_j100', 'y1_j10', 'y1_j100']:
            row += f'  {_agg(acc[label][metric]):>16s}'
        print(row)


if __name__ == '__main__':
    main()
