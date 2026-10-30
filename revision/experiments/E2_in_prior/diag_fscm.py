"""Is the fixed SCM the setups trained on the same distribution as the same-world data we score them on?
python diag_fscm.py SAME_DOPFN [N_VISITS]      (env DOPFN_SRC, PYTHONHASHSEED=0; SAME_DOPFN = $SCRATCH/e2/same/data/dopfn)

TRAIN: FixedSCM.sample exactly as train.py calls it (training_dopfn_repro/fixed_scm.py), N_VISITS visits of the
prior's training shape (batch 4 x 2200 rows), pooled. EVAL: the first 50 observational sets of the same-world data
(their context rows and query units, and the query units' true outcomes). Printed side by side: treatment rate,
each covariate, the observational outcome, both potential outcomes and the ITE (mean, sd, 5/50/95% quantiles), and
the least-squares fit of the observational outcome on (T, covariates) in each, so a difference in the mechanism
shows even when the marginals agree.
"""
import os, sys
import numpy as np

E2 = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [E2, os.path.join(E2, "../../..")]
import training_dopfn_repro.prior as P                           # noqa: E402
from training_dopfn_repro.fixed_scm import FixedSCM              # noqa: E402

same = sys.argv[1]; n_vis = int(sys.argv[2]) if len(sys.argv) > 2 else 50
inp = lambda r: os.path.join(same, f"shard{r // 100}", "complexmech", "5node", "path_TY", "hide_0.0", f"r{r:03d}.npz")
fs = FixedSCM(inp(0))
print(fs.describe(), flush=True)

# TRAIN side
cfg = P.PriorConfig()
X, T, Y, Y0, Y1 = [], [], [], [], []
for v in range(n_vis):
    rec = fs.sample(9_000_000 + v, cfg)
    if rec["meta"]["nonfinite"]:
        continue
    X.append(rec["x_obs"][:, :, 1:].permute(1, 0, 2).reshape(-1, rec["x_obs"].shape[-1] - 1).numpy())
    T.append(rec["x_obs"][:, :, 0].T.reshape(-1).numpy())
    Y.append(rec["y_obs"].T.reshape(-1).numpy()); Y0.append(rec["y_do0"].T.reshape(-1).numpy()); Y1.append(rec["y_do1"].T.reshape(-1).numpy())
tr = dict(X=np.concatenate(X), T=np.concatenate(T), Y=np.concatenate(Y), Y0=np.concatenate(Y0), Y1=np.concatenate(Y1))

# EVAL side
Xc, Tc, Yc, Xq, y0, y1 = [], [], [], [], [], []
for r in range(50):
    z = np.load(inp(r)); n = int(z["n_real_features"])
    Xc.append(z["X_train"][:, :n]); Tc.append(z["T_train"]); Yc.append(z["Y_train"]); Xq.append(z["X_test"][:, :n])
    ty = os.path.join(same, "truths_y", f"r{r:03d}.npz")
    if os.path.exists(ty):
        t = np.load(ty); y0.append(t["y0"]); y1.append(t["y1"])
ev = dict(X=np.concatenate(Xc), T=np.concatenate(Tc), Y=np.concatenate(Yc), Xq=np.concatenate(Xq))
if y0:
    ev["Y0"], ev["Y1"] = np.concatenate(y0), np.concatenate(y1)

def row(name, a, b):
    q = lambda v: np.percentile(v, [5, 50, 95])
    fa = f"{a.mean():9.4f} {a.std():9.4f} " + " ".join(f"{x:9.4f}" for x in q(a))
    fb = f"{b.mean():9.4f} {b.std():9.4f} " + " ".join(f"{x:9.4f}" for x in q(b)) if b is not None else "(none)"
    print(f"{name:16s} TRAIN {fa}\n{'':16s} EVAL  {fb}")

print(f"\nrows: train {tr['T'].size}, eval context {ev['T'].size}, eval queries {ev['Xq'].shape[0]}")
print(f"{'':16s}       {'mean':>9s} {'sd':>9s} {'q05':>9s} {'q50':>9s} {'q95':>9s}")
row("T (obs)", tr["T"], ev["T"])
for j in range(tr["X"].shape[1]):
    row(f"x{j} context", tr["X"][:, j], ev["X"][:, j])
    row(f"x{j} query", tr["X"][:, j], ev["Xq"][:, j])
row("Y obs", tr["Y"], ev["Y"])
row("Y do(0)", tr["Y0"], ev.get("Y0"))
row("Y do(1)", tr["Y1"], ev.get("Y1"))
row("ITE (shared)", tr["Y1"] - tr["Y0"], ev["Y1"] - ev["Y0"] if "Y0" in ev else None)

def ls(Xm, t, y):
    A = np.column_stack([np.ones_like(t), t, Xm]); b, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ b
    return b, res.std()
bt, st = ls(tr["X"], tr["T"], tr["Y"]); be, se = ls(ev["X"], ev["T"], ev["Y"])
print("\nleast squares  Y_obs ~ 1 + T + covariates   (coefficients, residual sd)")
print("TRAIN", np.round(bt, 4), f"resid sd {st:.4f}")
print("EVAL ", np.round(be, 4), f"resid sd {se:.4f}")
