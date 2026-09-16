#!/bin/bash
# PEHE and L1-ATE per (model, dataset), read from the same dumps the
# calibration step scores. Cheap enough to run interactively.
#
#   OUT_ROOT=$SCRATCH/rc_dens_uni bash summarize.sh
set -euo pipefail
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"
DEPLOY_ROOT="${DEPLOY_ROOT:-$SCRATCH/rpfn_bench_kit}"
source "$DEPLOY_ROOT/venv/bin/activate"
python - "$OUT_ROOT" <<'PY'
import glob, os, sys
import numpy as np
root = sys.argv[1]
SETS = ["IHDP", "ACIC", "CPS", "PSID", "PSID_bal"]
MODELS = ["dopfn_native", "dopfn_bb", "uwyk1d", "graph2d",
          "cpfn1d", "cpfn2d_pooled"]
print(f"{'model':16s}" + "".join(f"{d:>22s}" for d in SETS))
print(f"{'':16s}" + "".join(f"{'PEHE / L1-ATE':>22s}" for _ in SETS))
print("-" * (16 + 22 * len(SETS)))
for m in MODELS:
    row = f"{m:16s}"
    for ds in SETS:
        pe, l1 = [], []
        for f in sorted(glob.glob(f"{root}/{m}/{ds}/*.npz")):
            if os.path.basename(f) == "summary.npz":
                continue
            try:
                with np.load(f) as z:
                    pk = sorted(k for k in z.files if k.startswith("pehe_raw"))
                    ak = sorted(k for k in z.files if k.startswith("ate_raw"))
                    if not pk:
                        continue
                    pe.append(float(np.asarray(z[pk[0]]).reshape(-1)[0]))
                    if ak and "true_ate" in z.files:
                        l1.append(abs(float(np.asarray(z[ak[0]]).reshape(-1)[0])
                                      - float(np.asarray(z["true_ate"]).reshape(-1)[0])))
            except Exception:
                pass
        cell = (f"{np.mean(pe):.3f} / {np.mean(l1):.3f}" if pe and l1
                else (f"{np.mean(pe):.3f} / -" if pe else "-"))
        row += f"{cell:>22s}"
    print(row)
PY
