import glob, sys, numpy as np
cell = sys.argv[1] if len(sys.argv) > 1 else \
    "/scratch/furkanbd/cmech_data_rho99/complexmech/5node/path_TY/hide_0.0"
fs = sorted(glob.glob(cell + "/r*.npz"), key=lambda p: int(p.split("/r")[-1][:-4]))
print(f"realizations: {len(fs)}")
nq, nnz, nz, sd = [], [], [], []
for f in fs:
    z = np.load(f)
    y0 = np.asarray(z["Y_do0"]).ravel()
    t = np.asarray(z["true_cate"]).ravel()
    nq.append(y0.size); nnz.append(int((t != 0).sum())); nz.append(int((t == 0).sum()))
    sd.append(float(t.std()))
for nm, v in (("queries/realization", nq), ("nonzero-tau queries", nnz),
              ("zero-tau queries", nz)):
    v = np.asarray(v)
    print(f"{nm:22s} min={v.min():5d} median={int(np.median(v)):5d} max={v.max():5d}")
print(f"\nrealizations with >=1 zero-tau query: {int((np.asarray(nz) > 0).sum())} / {len(fs)}")
print(f"  -> scoring --subset total concatenates the nonzero cell and the zero cell,")
print(f"     so the scored realization count will be {len(fs)} + "
      f"{int((np.asarray(nz) > 0).sum())} = "
      f"{len(fs) + int((np.asarray(nz) > 0).sum())}, not {len(fs)}.")
s = np.asarray(sd)
print(f"\nsd(true_cate): min={s.min():.4f} median={np.median(s):.4f}  "
      f"(constant-tau realizations: {int((s == 0).sum())})")
