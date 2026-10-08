#!/bin/bash
# Stage 12: why Do-PFN 2D (262k) returns NaN intervals in some E2 worlds. Reruns the harness on
# the first 5 worlds of shard 6 (world 604 = index 4) with the dumps KEPT, then prints which
# queries are NaN, which dump fields are NaN, and that query's inputs against the context.
#   cd $KIT && sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --time=0:30:00 --cpus-per-task=4 --mem=32G \
#        -o logs_e2/nan_diag_%j.out R-PFN/revision/experiments/D262k/12_nan_diag.sh
set -uo pipefail
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"
source "$KIT/venv/bin/activate"; export PYTHONUNBUFFERED=1
export CAUSALPFN="$KIT/external/causalpfn" DOPFN_ROOT="$KIT/external/dopfn"
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
export DATASET=CMECH_n5 DENSITY_DUMP=1 MAX_REAL=5
export UWYK_FIG34_DATA="$SCRATCH/e2/new/data/dopfn/shard6"
export OUT="$SCRATCH/e2/nan_diag/dumps_shard6"; rm -rf "$OUT"; mkdir -p "$OUT"
DOPFN_CKPT="$REPO/Required_checkpoints/dopfn_repro_joint2d_step262144.pt" \
  python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset "$DATASET" 2>&1 | grep -v "^\s*$" | tail -40
python - <<PY
import numpy as np, glob, os
out = "$OUT"
for f in sorted(glob.glob(out + "/r*.npz")):
    z = np.load(f); bad = {}
    for k in z.files:
        a = z[k]
        if a.dtype.kind == "f" and a.ndim >= 1 and a.shape[0] > 0:
            m = ~np.isfinite(a.reshape(a.shape[0], -1)).all(axis=1) if a.ndim > 1 else ~np.isfinite(a)
            if m.any():
                bad[k] = (a.shape, np.flatnonzero(m)[:10].tolist())
    print(os.path.basename(f), "non-finite fields:", bad or "none")
f = out + "/r004.npz"; z = np.load(f)
print("dump keys:", {k: z[k].shape for k in z.files})
w = np.load("$SCRATCH/e2/new/data/dopfn/shard6/complexmech/5node/path_TY/hide_0.0/r604.npz")
print("data keys:", {k: w[k].shape for k in w.files})
for k in w.files:
    a = w[k]
    if a.dtype.kind == "f":
        print(f"  {k:12s} finite={np.isfinite(a).all()} min={np.nanmin(a):.4g} max={np.nanmax(a):.4g} sd={np.nanstd(a):.4g}")
PY
