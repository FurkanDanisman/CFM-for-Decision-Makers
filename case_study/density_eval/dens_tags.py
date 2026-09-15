"""What is actually INSIDE each model directory's prediction dumps.

The submit-script directory names do not map one-to-one onto methods: one
tauC run emits several (density_tauc.DIR_METHOD), and the adjacency it
conditioned on is recorded per file as `anc_tag`, not in the path. So
"where is graph2d v3a" is answered by reading anc_tag + which *_logits keys
are present, not by looking for a directory called graph2d_v3a.

    python dens_tags.py $RES [shift0] [d2] [ctx1000]
"""
import os, sys, glob
from collections import Counter
import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith("--")]
root  = args[0]
shift = args[1] if len(args) > 1 else "shift0"
d     = args[2] if len(args) > 2 else "d2"
ctx   = args[3] if len(args) > 3 else "ctx1000"

base = os.path.join(root, shift, d, ctx)
print(f"{base}\n")
print(f"{'model dir':15s} {'n_pred':>7s}  {'anc_tag':18s} {'model_family':14s} logits present")
print("-" * 104)
for m in sorted(os.listdir(base)):
    mdir = os.path.join(base, m)
    if not os.path.isdir(mdir):
        continue
    tags, fams, keys = Counter(), Counter(), Counter()
    n = 0
    for f in glob.glob(os.path.join(mdir, "*", "predictions", "*.npz")):
        try:
            with np.load(f, allow_pickle=True) as z:
                ks = set(z.files)
                if "tau_grid" not in ks:
                    continue
                n += 1
                tags[str(np.asarray(z["anc_tag"]).reshape(-1)[0])
                     if "anc_tag" in ks else "?"] += 1
                fams[str(np.asarray(z["model_family"]).reshape(-1)[0])
                     if "model_family" in ks else "?"] += 1
                keys[",".join(sorted(k for k in ks if k.endswith("logits")))] += 1
        except Exception:
            continue
    if not n:
        print(f"{m:15s} {0:7d}  (no tauC prediction dumps)")
        continue
    print(f"{m:15s} {n:7d}  "
          f"{','.join(f'{k}x{v}' for k, v in tags.most_common(3)):18s} "
          f"{','.join(f'{k}x{v}' for k, v in fams.most_common(2)):14s} "
          f"{keys.most_common(1)[0][0]}")
