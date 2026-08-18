"""Count the fraction of total parameters in the head vs backbone for a
DoPFN-backbone-with-2D-head checkpoint. Also loads DoPFN's original
regressor and reports its head fraction for direct comparison — so you
can back the "<1%" or whatever number in the ICLR paragraph.

Usage:
    python count_head_params.py \
        --repo $DEPLOY_ROOT/R-PFN \
        --dopfn $DEPLOY_ROOT/external/dopfn \
        --checkpoint $DEPLOY_ROOT/checkpoints_dopfn_backbone_realj10/step_150000.pt \
        --also-dopfn
"""
from __future__ import annotations
import argparse, os, sys


def _fmt_M(n: int) -> str:
    return f'{n / 1e6:.3f} M'


def count_ours(args):
    import torch
    sys.path.insert(0, args.repo)
    sys.path.insert(0, os.path.join(args.repo, 'training_dopfn_base'))
    from dopfn_backbone_head import DoPFNBackboneWith2DHead

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt['config']; J = int(cfg['J'])
    model = DoPFNBackboneWith2DHead(dopfn_root=args.dopfn, K=J).eval()
    model.load_state_dict(ckpt['model_state_dict'])

    total = sum(p.numel() for p in model.parameters())
    # Head = anything under the module path 'head' (our added 2D BarDist head)
    head_names = [n for n, _ in model.named_parameters() if n.startswith('head.')]
    head = sum(p.numel() for n, p in model.named_parameters() if n.startswith('head.'))

    print(f'\n══ Ours (DoPFN-bb + 2D head, J={J})  |  {os.path.basename(args.checkpoint)} ══')
    print(f'  total params:  {_fmt_M(total):>10s}   ({total:,})')
    print(f'  head params:   {_fmt_M(head):>10s}   ({head:,})')
    print(f'  head fraction: {head / max(total, 1) * 100:.3f}%')
    if head_names:
        print(f'  head modules:  {sorted({n.split(".")[1] for n in head_names})}')
    return total, head


def count_dopfn(args):
    """Load DoPFN's own DoPFNRegressor and count params on its inner model."""
    import torch
    # Reuse the loader shim used by Fig 2 / eval_dopfn_bb_raw
    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'empirical_tests'))
    from dopfn_helpers import load_dopfn
    DoPFNRegressor = load_dopfn(args)
    # Instantiate and reach into the transformer
    reg = DoPFNRegressor()
    # DoPFN's regressor lazily loads its transformer via `.load_model()` or
    # by fitting. We touch a small tensor to force initialisation.
    import numpy as np
    x = np.zeros((4, 3), dtype=np.float32)
    y = np.zeros(4, dtype=np.float32)
    try:
        reg.fit(torch.tensor(x), torch.tensor(y))
    except Exception as e:
        print(f'[warn] DoPFN fit failed (small dummy input): {type(e).__name__}: {e}')

    # Try common attribute paths for the loaded model
    for attr in ('model_', 'model', 'clf', 'regressor', 'net'):
        m = getattr(reg, attr, None)
        if m is not None and hasattr(m, 'parameters'):
            break
    else:
        print('[error] could not find DoPFN transformer object; report failed')
        return None

    total = sum(p.numel() for p in m.parameters())
    # Do-PFN's head goes by different names across TabPFN versions
    HEAD_HINTS = ('criterion', 'decoder', 'head', 'y_encoder', 'output')
    head_names = [n for n, _ in m.named_parameters()
                   if any(h in n.lower() for h in HEAD_HINTS)]
    head = sum(p.numel() for n, p in m.named_parameters()
                if any(h in n.lower() for h in HEAD_HINTS))
    print(f'\n══ Original Do-PFN ══')
    print(f'  total params:  {_fmt_M(total):>10s}   ({total:,})')
    print(f'  head-ish:      {_fmt_M(head):>10s}   ({head:,})')
    print(f'  head fraction: {head / max(total, 1) * 100:.3f}%')
    if head_names:
        preview = sorted({n.split(".")[0] for n in head_names})
        print(f'  matched under: {preview}')
    return total, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--dopfn', required=True)
    ap.add_argument('--causalpfn', default='',
                    help='Not strictly needed for param counting, but load_dopfn '
                         'may want it on some builds.')
    ap.add_argument('--checkpoint', required=True,
                    help='Path to our DoPFN-backbone + 2D head checkpoint.')
    ap.add_argument('--also-dopfn', action='store_true',
                    help='Also load DoPFN\'s original DoPFNRegressor and report its '
                         'total / head params for direct comparison.')
    args = ap.parse_args()

    ours_total, ours_head = count_ours(args)
    if args.also_dopfn:
        try:
            res = count_dopfn(args)
            if res is not None:
                dpf_total, dpf_head = res
                print(f'\n══ Comparison ══')
                print(f'  DoPFN total:  {_fmt_M(dpf_total)}')
                print(f'  Ours total:   {_fmt_M(ours_total)}')
                diff = (ours_total - dpf_total) / max(dpf_total, 1) * 100
                print(f'  delta total:  {diff:+.3f}%  '
                      f'({"+" if diff>=0 else ""}{ours_total - dpf_total:,} params)')
        except Exception as e:
            print(f'[error] DoPFN reference counting failed: {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
