"""
Generate a corpus of paired potential-outcome tasks to disk.

Why this exists:
  - PairedInterventionalDataset.__getitem__ has a memory-corruption bug
    somewhere in the SCM mutation code that hits a fatal `none_dealloc`
    C-extension error after roughly a few hundred tasks.
  - On-the-fly generation also caps throughput at ~31 s/step (single
    process) — far slower than 50 K steps can absorb.
  - Caching N tasks to disk once amortizes both problems:
      * crashes during corpus generation are recoverable (workers die,
        others continue)
      * training reads from disk = ~ms per task, no SCM cost on the
        critical path

Usage (env-overridable):
    N_SAMPLES=10000           how many tasks to generate
    N_WORKERS=16              how many CPU worker processes
    OUT_DIR=outputs_corpus    where to save sample_NNNNN.pt files
    SEED_BASE=42              base seed; each task uses SEED_BASE+idx
    UWYK_SRC=/path/to/g4cfm/src

Each task is saved as outputs_corpus/sample_NNNNN.pt containing the
same 7-tuple dict that `generate_paired_samples.py` produced:
    X_obs, T_obs, Y_obs, X_intv, Y_do0, Y_do1, anc_matrix

Resumes automatically: if outputs_corpus/sample_00042.pt exists,
that index is skipped.
"""
from __future__ import annotations

import os
import sys
import time
import glob
import signal
import traceback
import multiprocessing as mp

import torch

REPO_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_SRC)

# ── config ────────────────────────────────────────────────────────────────────
N_SAMPLES = int(os.environ.get('N_SAMPLES', 10000))
N_WORKERS = int(os.environ.get('N_WORKERS', 16))
OUT_DIR   = os.environ.get('OUT_DIR', 'outputs_corpus')
SEED_BASE = int(os.environ.get('SEED_BASE', 42))
CHUNK     = int(os.environ.get('CHUNK', 50))  # tasks per worker call

# ── worker ────────────────────────────────────────────────────────────────────

_DATASET = None


def _init_worker():
    """One PairedInterventionalDataset per worker. Done lazily so the import
    is paid once per worker rather than once per task."""
    global _DATASET
    from data.PairedInterventionalDataset import PairedInterventionalDataset
    _DATASET = PairedInterventionalDataset(seed_base=SEED_BASE)


def _generate_chunk(args):
    """
    Generate a contiguous chunk of indices [start, end). Save each task
    to disk individually. Returns a list of (idx, ok, err_str) tuples so
    main can track progress.

    If the C-extension crash hits mid-chunk, the worker dies and the
    multiprocessing Pool replaces it; the remaining indices in the chunk
    will be re-dispatched.
    """
    start, end = args
    results = []
    for idx in range(start, end):
        path = os.path.join(OUT_DIR, f'sample_{idx:05d}.pt')
        if os.path.exists(path):
            results.append((idx, True, 'cached'))
            continue
        try:
            out = _DATASET._generate_one(idx)
            # Save atomically — write to .tmp then rename
            tmp = path + '.tmp'
            torch.save(out, tmp)
            os.replace(tmp, path)
            results.append((idx, True, ''))
        except Exception as e:
            results.append((idx, False, str(e)[:200]))
    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Corpus generation:", flush=True)
    print(f"  N_SAMPLES = {N_SAMPLES}", flush=True)
    print(f"  N_WORKERS = {N_WORKERS}", flush=True)
    print(f"  OUT_DIR   = {OUT_DIR}", flush=True)
    print(f"  CHUNK     = {CHUNK}", flush=True)

    # Count existing
    existing = set()
    for p in glob.glob(os.path.join(OUT_DIR, 'sample_*.pt')):
        try:
            idx = int(os.path.basename(p).split('_')[1].split('.')[0])
            existing.add(idx)
        except (ValueError, IndexError):
            pass
    print(f"  found {len(existing)} cached samples; will skip", flush=True)

    # Build chunk ranges over missing indices
    missing = [i for i in range(N_SAMPLES) if i not in existing]
    if not missing:
        print("All tasks already generated. Nothing to do.", flush=True)
        return

    print(f"  need to generate {len(missing)} new samples", flush=True)

    # Chunk the missing indices. We keep chunks contiguous in the index space
    # so SEED_BASE+idx remains deterministic per task.
    chunks = []
    i = 0
    while i < len(missing):
        start = missing[i]
        end = start
        while i < len(missing) and missing[i] == end:
            end += 1
            i += 1
        # Now [start, end) is a contiguous run of missing indices
        # Split into sub-chunks of size CHUNK
        for s in range(start, end, CHUNK):
            chunks.append((s, min(s + CHUNK, end)))

    print(f"  scheduled {len(chunks)} chunks of up to {CHUNK} tasks each\n", flush=True)

    t0 = time.time()
    n_ok = len(existing)
    n_err = 0
    last_report = t0

    with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
        for results in pool.imap_unordered(_generate_chunk, chunks):
            for idx, ok, err in results:
                if ok:
                    n_ok += 1
                else:
                    n_err += 1
                    print(f"  [err] sample {idx}: {err}", flush=True)
            now = time.time()
            if now - last_report > 10:
                rate = (n_ok - len(existing)) / max(1, now - t0)
                eta = (N_SAMPLES - n_ok) / rate if rate > 0 else float('inf')
                print(f"  progress: {n_ok}/{N_SAMPLES} ok, {n_err} errors, "
                      f"{rate:.1f}/s, ETA {eta/60:.1f} min", flush=True)
                last_report = now

    print(f"\nDone. {n_ok}/{N_SAMPLES} tasks in {OUT_DIR}/  "
          f"({n_err} failed)  total wall = {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    # Ensure 'fork' start method (faster than spawn, lower memory)
    mp.set_start_method('fork', force=True)
    main()
