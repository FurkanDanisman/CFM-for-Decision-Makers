"""Standalone data-pipeline diagnostic — no GPU, no model, no training.

Purpose: figure out whether the `none_dealloc` worker crash and/or the
numerical instability come from the data generator, by exercising
PairedInterventionalDataset on its own.

Two modes:

  num_workers=0 (default)  — generate samples *in this process*, sequentially.
      A C-level abort therefore crashes THIS process and faulthandler prints the
      real stack at the offending line, and because we iterate in order we know
      the exact idx/seed. This is the clean, cheap (CPU-only) reproducer.

  num_workers>0            — go through a real DataLoader to reproduce the
      multiprocessing-specific crash seen in training. Here a worker abort is
      reported via the breadcrumb files in logs/ (see below), not a clean trace.

Either way each sample is validated (NaN/Inf, target |max| ≫ 1) and a summary
is printed so we can quantify how dirty the stream is.

Examples:
  python diagnose_data.py                          # 2000 samples, workers=0
  python diagnose_data.py --n 5000 --workers 8     # mimic training's loader
  python diagnose_data.py --start 500 --n 50 -v    # zoom in, log every sample

After a workers>0 crash, find the culprit sample(s) with:
  grep -L DONE logs/databreadcrumb_*.txt   # files whose last line is still ENTER
  cat logs/databreadcrumb_*.txt
"""
from __future__ import annotations

# faulthandler first, so an abort in-process dumps a real C-level stack.
import faulthandler
faulthandler.enable()

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from data.PairedInterventionalDataset import (  # noqa: E402
    PairedInterventionalDataset,
    make_streaming_loader,
    sample_metrics,
)


def _strip_batch_dim(batch: dict) -> dict:
    """DataLoader stacks a leading batch dim of 1; drop it for metrics."""
    return {k: (v[0] if torch.is_tensor(v) and v.dim() > 0 else v) for k, v in batch.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2000, help="number of samples to generate")
    ap.add_argument("--start", type=int, default=0, help="starting idx (workers=0 only)")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers; 0 = in-process")
    ap.add_argument("--seed-base", type=int, default=int(os.environ.get("STREAM_SEED", 42)))
    ap.add_argument("--every", type=int, default=100, help="progress print interval")
    ap.add_argument("-v", "--verbose", action="store_true", help="print metrics for every sample")
    args = ap.parse_args()

    print(f"diagnose_data: n={args.n} start={args.start} workers={args.workers} "
          f"seed_base={args.seed_base}")
    print(f"breadcrumbs → {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')}")
    print("─" * 72)

    n_seen = 0
    n_nonfinite = 0
    n_blowup = 0
    worst_absmax = 0.0
    worst_idx = -1
    t0 = time.time()

    def inspect(idx, out):
        nonlocal n_seen, n_nonfinite, n_blowup, worst_absmax, worst_idx
        m = sample_metrics(out)
        n_seen += 1
        if m["n_nonfinite"]:
            n_nonfinite += 1
        if m["target_absmax"] > 10.0:
            n_blowup += 1
        if m["target_absmax"] > worst_absmax:
            worst_absmax = m["target_absmax"]
            worst_idx = idx
        if args.verbose:
            print(f"  idx={idx:>6}  target_absmax={m['target_absmax']:.3g}  "
                  f"nonfinite={m['n_nonfinite']}")
        if n_seen % args.every == 0:
            rate = n_seen / (time.time() - t0)
            print(f"  [{n_seen}/{args.n}] {rate:.1f} samp/s  "
                  f"nonfinite={n_nonfinite} blowups={n_blowup} "
                  f"worst_absmax={worst_absmax:.3g}@idx{worst_idx}")
            sys.stdout.flush()

    if args.workers == 0:
        ds = PairedInterventionalDataset(seed_base=args.seed_base)
        for idx in range(args.start, args.start + args.n):
            inspect(idx, ds[idx])
    else:
        loader = make_streaming_loader(
            batch_size=1, num_workers=args.workers, seed_base=args.seed_base,
        )
        it = iter(loader)
        for idx in range(args.n):
            inspect(idx, _strip_batch_dim(next(it)))

    dt = time.time() - t0
    print("─" * 72)
    print(f"DONE: {n_seen} samples in {dt:.1f}s ({n_seen / dt:.1f} samp/s)")
    print(f"  samples with non-finite values : {n_nonfinite}")
    print(f"  samples with |target| > 10      : {n_blowup}")
    print(f"  worst |target|                  : {worst_absmax:.3g} (idx {worst_idx})")
    if n_nonfinite or n_blowup:
        print("  → data IS dirty; this is a plausible source of the loss spikes/NaNs.")
    else:
        print("  → stream looks clean over this range.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
