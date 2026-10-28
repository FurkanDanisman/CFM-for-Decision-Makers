"""In-prior calibration, new SCM per dataset: coverage of y0, y1, the joint region (2D) and the ITE, per world.
python newworld_y_table.py DATA RES NW OUT_CSV
Per world (one dataset of 500 queries): coverage = share of queries whose truth lies in the 95% interval.
  y0, y1     posterior: the predictive interval; formula: mu +- 1.96 sd. Truth: the prior's own outcomes
             (CausalPFN-C 1D: its training target m_t).
  joint      2D only: the joint's 95% region; truth pair with shared noise (sh) and with independent noise (in).
  ITE        the paper's interval (1D: independence, rho = 0; 2D: learned joint) and, for 2D, rho = 0;
             truth: delta_shared (sh) and delta_indep (in).
Printed per model and target: mean coverage over worlds (SE) and the number of worlds within [0.93, 0.97]."""
import glob, os, re, sys
import numpy as np

DATA, RES, NW, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
Z = 1.959963984540054
MODELS = [("dopfn_native", "Do-PFN", "dopfn", False), ("dopfn_joint2d_262k", "Do-PFN 2D", "dopfn", True),
          ("uwyk1d", "UWYK (binary T)", "uwyk_1d", False), ("uwyk1d_cont", "UWYK (continuous T)", "uwyk_1dc", False),
          ("graph2d", "UWYK 2D", "uwyk_2d", True), ("cpfn1d", "CausalPFN-C", "cpfn_1d", False),
          ("cpfn2d", "CausalPFN-C 2D", "cpfn_2d", True)]


def load_shards(model):
    """global world index -> (outcome readouts dict, (ite lo, ite hi))"""
    out = {}
    for f in glob.glob(os.path.join(RES, model, "outall_shard*.npz")):
        k = int(re.search(r"shard(\d+)\.npz$", f).group(1))
        z = np.load(f); zi = np.load(os.path.join(RES, model, f"shard{k}.npz"))
        ite = {int(i): (lo, hi) for i, lo, hi in zip(zi["idx"], zi["lo"], zi["hi"])}
        for j, i in enumerate(z["idx"]):
            r = k * 100 + int(i)
            if r < NW and int(i) in ite:
                out[r] = ({c: z[c][j] for c in z.files if c != "idx"}, ite[int(i)])
    return out


rows = []
for model, name, prior, is2d in MODELS:
    got = load_shards(model)
    if not got:
        print(f"{name}: no results"); continue
    cov = {}
    for r, (o, (ilo, ihi)) in sorted(got.items()):
        ty = np.load(os.path.join(DATA, prior, "truths_y", f"r{r:03d}.npz"))
        td = np.load(os.path.join(DATA, prior, "truths", f"r{r:03d}.npz"))
        y0, y1 = (ty["m0"], ty["m1"]) if prior == "cpfn_1d" else (ty["y0"], ty["y1"])
        add = lambda key, hit: cov.setdefault(key, []).append((r, float(np.mean(hit))))
        if not is2d:
            for t, y in ((0, y0), (1, y1)):
                add(f"y{t} posterior", (o[f"lo{t}"] <= y) & (y <= o[f"hi{t}"]))
                add(f"y{t} formula", np.abs(y - o[f"mu{t}"]) <= Z * o[f"sd{t}"])
        else:
            for v in ("sh", "in"):
                if f"joint_hit_{v}" in o: add(f"joint ({v})", o[f"joint_hit_{v}"])
        for v, key in (("sh", "delta_shared"), ("in", "delta_indep")):
            d = td[key]
            add(f"ITE {'learned' if is2d else 'rho=0'} ({v})", (ilo <= d) & (d <= ihi))
            if is2d:
                add(f"ITE rho=0 ({v})", (o["lo_ite_rho0"] <= d) & (d <= o["hi_ite_rho0"]))
    print(f"\n{name}  ({len(got)} worlds)")
    for key, vals in cov.items():
        c = np.array([x for _, x in vals]); n = c.size
        print(f"  {key:22s} {c.mean():.3f} ({c.std(ddof=1) / np.sqrt(n):.3f})   within [0.93,0.97]: {int(((c >= .93) & (c <= .97)).sum()):4d}/{n}")
        rows += [(name, key, r, x) for r, x in vals]
with open(OUT, "w") as fh:
    fh.write("model,target,world,coverage\n"); fh.writelines(f"{m},{k},{r},{x:.6f}\n" for m, k, r, x in rows)
print("\nper-world values:", OUT)
