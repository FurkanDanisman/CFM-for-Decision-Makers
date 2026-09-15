"""Are the uwyk anc variants actually different predictions, or the same bytes?

uwyk / uwyk_v3a / uwyk_noanc scored identically to 4 decimals, which is either
(a) the adjacency never reaches the UWYK head, (b) the three runs were handed
the same adjacency, or (c) they genuinely coincide. Those need different
fixes, so compare the dumps directly: the recorded anc_tag, the adjacency
matrix each run used, and the raw head outputs.

    python cmp_anc_dumps.py $RES shift0 d5 ctx1000 Observed_Confounder
"""
import os, sys
import numpy as np

root, shift, d, ctx, case = sys.argv[1:6]
DIRS = ["uwyk", "uwyk_v3a", "uwyk_noanc", "graph2d"]
R = int(sys.argv[6]) if len(sys.argv) > 6 else 0

ref = {}
for m in DIRS:
    p = os.path.join(root, shift, d, ctx, m, case, "predictions",
                     f"{case}_r{R:03d}.npz")
    if not os.path.isfile(p):
        print(f"{m:12s} MISSING {p}")
        continue
    with np.load(p, allow_pickle=True) as z:
        tag = str(np.asarray(z["anc_tag"]).reshape(-1)[0]) if "anc_tag" in z.files else "?"
        fam = str(np.asarray(z["model_family"]).reshape(-1)[0]) if "model_family" in z.files else "?"
        rec = {"anc_tag": tag, "family": fam}
        for k in ("adj_uwyk", "adj_joint", "uwyk_pred0", "joint_logits"):
            rec[k] = np.asarray(z[k], dtype=np.float64) if k in z.files else None
        ref[m] = rec
        shapes = {k: (v.shape if v is not None else None)
                  for k, v in rec.items() if k not in ("anc_tag", "family")}
        print(f"{m:12s} anc_tag={tag:12s} family={fam:8s} {shapes}")

print("\npairwise: are the arrays bitwise identical?")
keys = ["adj_uwyk", "adj_joint", "uwyk_pred0", "joint_logits"]
names = [m for m in DIRS if m in ref]
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = ref[names[i]], ref[names[j]]
        out = []
        for k in keys:
            if a[k] is None or b[k] is None:
                out.append(f"{k}=n/a")
            elif a[k].shape != b[k].shape:
                out.append(f"{k}=SHAPE")
            else:
                same = np.array_equal(a[k], b[k])
                md = float(np.abs(a[k] - b[k]).max())
                out.append(f"{k}={'SAME' if same else f'diff({md:.3g})'}")
        print(f"  {names[i]:12s} vs {names[j]:12s}  " + "  ".join(out))
