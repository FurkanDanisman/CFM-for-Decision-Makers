"""Both true outcomes of every query unit of the multi-world Do-PFN data (multi.sh), which gen_dopfn.py did not save.
python multi_truth_y.py ROOT W0:W1          (PYTHONHASHSEED=0 and DOPFN_SRC as in gen_e2.sbatch)

World w is redrawn exactly as gen_dopfn.py same mode draws it (sample_batch(SEED + 1000 w, seq_len 1000 * 1500,
batch 1, resample)); dataset r's query units are rows r*1500+1000 .. r*1500+1499. The redraw must reproduce the
saved truth delta_shared = y1 - y0 of every dataset, else the script stops. Writes
ROOT/data/w<w>/dopfn/truths_y/r###.npz with y0, y1 (the outcomes under do(0) and do(1)).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
import training_dopfn_repro.prior as P
from common import N_CTX, N_Q

assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0 (as gen_e2.sbatch)"
root, (W0, W1) = sys.argv[1], map(int, sys.argv[2].split(":"))
SEED, ROWS, R = 20261003, N_CTX + N_Q, 1000
for w in range(W0, W1):
    out = os.path.join(root, "data", f"w{w}", "dopfn", "truths_y")
    if os.path.exists(os.path.join(out, f"r{R - 1:03d}.npz")):
        continue
    os.makedirs(out, exist_ok=True)
    rec = P.sample_batch(SEED + w * 1_000, P.PriorConfig(seq_len=R * ROWS, batch_size=1, on_nonfinite="resample"))
    y0, y1 = rec["y_do0"][:, 0].numpy(), rec["y_do1"][:, 0].numpy()
    for r in range(R):
        q = slice(r * ROWS + N_CTX, (r + 1) * ROWS)
        saved = np.load(os.path.join(root, "data", f"w{w}", "dopfn", "truths", f"r{r:03d}.npz"))["delta_shared"]
        if not np.allclose((y1[q] - y0[q]).astype(np.float32), saved, rtol=1e-5, atol=1e-6):
            raise SystemExit(f"world {w} dataset {r}: redraw does not reproduce the saved ITE -- stopping")
        np.savez(os.path.join(out, f"r{r:03d}.npz"), y0=y0[q].astype(np.float32), y1=y1[q].astype(np.float32))
    print(f"world {w}: {R} datasets, redraw verified against the saved ITE", flush=True)
