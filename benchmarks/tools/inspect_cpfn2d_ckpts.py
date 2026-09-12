"""Enumerate and VERIFY cpfn2d (and cpfn1d) checkpoints.

Folder names lie. This derives the architecture from the weights themselves and
then cross-checks whatever the config claims, reporting any disagreement.

How J is recovered. The 2D joint head is Linear(ninp, n_out + J**2 + 9 + 4), so
from the head's out_features:

    J = sqrt(out_features - n_out - 13)

If that is a whole number the checkpoint is a 2D joint head. If instead
out_features - n_out is itself a plausible bin count, it is a 1D head with
nbins = out_features - n_out. Either way the number comes from the tensor, not
from the directory name or the config.

Usage:
    python benchmarks/tools/inspect_cpfn2d_ckpts.py                 # scan $SCRATCH
    python benchmarks/tools/inspect_cpfn2d_ckpts.py <dir-or-file>...
    python benchmarks/tools/inspect_cpfn2d_ckpts.py --all-steps     # every step
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

N_OUT_DEFAULT = 10
EXTRA_2D = 9 + 4


def _fmt_size(n):
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024:
            return f'{n:.0f}{u}'
        n /= 1024
    return f'{n:.1f}TB'


def _unwrap(ck):
    for k in ('model_state_dict', 'state_dict', 'model'):
        if isinstance(ck, dict) and k in ck and isinstance(ck[k], dict):
            return ck[k]
    return ck if isinstance(ck, dict) else {}


def _head_out_features(sd):
    """Largest 2-D '...head...weight' tensor: the final projection."""
    best = None
    for k, v in sd.items():
        if not hasattr(v, 'shape') or len(getattr(v, 'shape', ())) != 2:
            continue
        if 'head' not in k or not k.endswith('weight'):
            continue
        if best is None or v.shape[0] > best[1].shape[0]:
            best = (k, v)
    return best


def _derive(out_features, n_out=N_OUT_DEFAULT):
    """Return (kind, J_or_nbins). Prefers an exact 2D factorisation."""
    rem = out_features - n_out - EXTRA_2D
    if rem > 0:
        r = int(round(math.isqrt(rem)))
        for cand in (r - 1, r, r + 1):
            if cand > 0 and cand * cand == rem:
                return '2D-joint', cand
    return '1D-bars', out_features - n_out


def inspect(path, all_steps=False):
    import torch
    try:
        ck = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        return {'path': path, 'error': f'{type(e).__name__}: {e}'}

    sd = _unwrap(ck)
    hit = _head_out_features(sd)
    row = {'path': path, 'size': os.path.getsize(path)}

    if hit is None:
        row['error'] = 'no head weight found'
        row['keys'] = list(sd)[:6]
        return row
    key, W = hit
    out_f, in_f = int(W.shape[0]), int(W.shape[1])
    kind, val = _derive(out_f)
    row.update(head_key=key, out_features=out_f, ninp=in_f, kind=kind)
    row['J' if kind == '2D-joint' else 'nbins'] = val

    cfg = {}
    for ck_key in ('config', 'model_config', 'hyper_parameters', 'cfg'):
        if isinstance(ck, dict) and isinstance(ck.get(ck_key), dict):
            cfg = ck[ck_key]
            break
    flat = {}

    def _walk(d, pre=''):
        for k, v in d.items():
            if isinstance(v, dict):
                _walk(v, pre + k + '.')
            else:
                flat[pre + k] = v
    if cfg:
        _walk(cfg)
    for want in ('J', 'nbins', 'num_buckets', 'loss_type', 'hlgauss_sigma',
                 'y_scaling_mode', 'num_features', 'edge_lo', 'edge_hi',
                 'ninp', 'nlayers', 'nhead'):
        for k, v in flat.items():
            if k.split('.')[-1] == want:
                row[f'cfg.{want}'] = v
                break
    for k in ('step', 'actual_step', 'global_step', 'epoch'):
        if isinstance(ck, dict) and k in ck and not isinstance(ck[k], dict):
            row[k] = ck[k]

    # cross-check: does the config agree with the weights?
    claimed = row.get('cfg.J')
    if kind == '2D-joint' and claimed is not None and int(claimed) != val:
        row['MISMATCH'] = f'config J={claimed} but weights say J={val}'
    if 'edges' in (ck if isinstance(ck, dict) else {}):
        e = ck['edges']
        try:
            row['edges'] = f'{len(e)} pts [{float(e[0]):.3g}, {float(e[-1]):.3g}]'
        except Exception:
            pass
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('targets', nargs='*')
    ap.add_argument('--all-steps', action='store_true',
                    help='inspect every step_*.pt, not just the newest 2')
    a = ap.parse_args()

    targets = a.targets
    if not targets:
        root = os.environ.get('SCRATCH', '') + '/rpfn_bench_kit'
        targets = sorted(glob.glob(os.path.join(root, '*cpfn*')))
        if not targets:
            print(f'no *cpfn* dirs under {root}; pass paths explicitly')
            return

    files = []
    for t in targets:
        if os.path.isfile(t):
            files.append(t)
            continue
        found = sorted(glob.glob(os.path.join(t, '**', '*.pt'), recursive=True))
        found = [f for f in found if 'beforeheadslice' not in f]
        if not a.all_steps and len(found) > 2:
            steps = [f for f in found if 'step_' in os.path.basename(f)]
            others = [f for f in found if f not in steps]
            found = others + steps[-2:]
        files.extend(found)

    if not files:
        print('no .pt files found'); return
    print(f'inspecting {len(files)} checkpoint(s)\n')
    for f in files:
        r = inspect(f, a.all_steps)
        print(f)
        if 'error' in r:
            print(f'   ERROR: {r["error"]}')
            if 'keys' in r:
                print(f'   keys: {r["keys"]}')
            print()
            continue
        j = r.get('J')
        head = (f'2D joint  J={j}  (head {r["out_features"]:,} = 10 + {j}^2 + 13)'
                if j is not None else
                f'1D bars   nbins={r.get("nbins")}  (head {r["out_features"]:,})')
        print(f'   {head}   ninp={r["ninp"]}   {_fmt_size(r["size"])}')
        extras = [f'{k.split(".")[-1]}={v}' for k, v in r.items()
                  if k.startswith('cfg.')]
        if extras:
            print('   config: ' + '  '.join(extras))
        st = [f'{k}={r[k]}' for k in ('step', 'actual_step', 'global_step', 'epoch')
              if k in r]
        if st:
            print('   ' + '  '.join(st))
        if 'edges' in r:
            print(f'   edges: {r["edges"]}')
        if 'MISMATCH' in r:
            print(f'   *** MISMATCH: {r["MISMATCH"]} ***')
        print()


if __name__ == '__main__':
    main()
