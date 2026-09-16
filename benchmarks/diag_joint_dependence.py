"""Why is a 2D head's interval wider than a 1D head's?

Three things differ between e.g. cpfn1d and cpfn2d -- checkpoint, bin count,
and 1D-vs-2D -- so their interval widths are not attributable to the head.
This separates them per model:

  J, n_atoms      the tau support. A 95% interval on a coarse discrete law
                  can only end on an atom boundary, so it rounds OUTWARD;
                  fewer atoms means wider intervals and over-coverage
                  independently of anything the model learned.
  rho_implied     correlation between the arms under the learned joint.
                  Var(tau) = s0^2 + s1^2 - 2 rho s0 s1, so rho > 0 NARROWS
                  tau relative to independence and rho < 0 widens it.
  sd_learned      sd of tau from the anti-diagonal projection
  sd_forced       sd of tau from the SAME joint's marginals, re-coupled
                  independently
  ratio           sd_forced / sd_learned -- what the dependence is worth,
                  with model and resolution held fixed

    python diag_joint_dependence.py <cell-dir> [<cell-dir> ...]
"""
import glob, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "UWYK_Fig3_4"))
from cate_density_metrics import (tau_atoms, tau_pmf_joint, tau_pmf_indep,
                                  _bin_width)                      # noqa: E402


def stats(path, max_q=200):
    with np.load(path, allow_pickle=True) as z:
        ys = float(np.asarray(z["y_scale"]).reshape(-1)[0]) if "y_scale" in z.files else 1.0
        jk = sorted(k for k in z.files if k.startswith("p_joint_scaled"))
        if jk:
            Jt = np.asarray(z[jk[0]], dtype=np.float64)[:max_q]
            J = Jt.shape[-1]
            e = np.asarray(z["edges"], dtype=np.float64).reshape(-1) if "edges" in z.files else None
            c = 0.5*(e[:-1]+e[1:]) if e is not None and e.size == J+1 else np.linspace(-1, 1, J)
            atoms = tau_atoms(J, _bin_width(z, J)) * ys
            rho, sl, sf = [], [], []
            for P in Jt:
                m0, m1 = P.sum(axis=1), P.sum(axis=0)
                E0, E1 = m0 @ c, m1 @ c
                v0 = max(m0 @ c**2 - E0**2, 1e-12)
                v1 = max(m1 @ c**2 - E1**2, 1e-12)
                cov = (P * np.outer(c, c)).sum() - E0*E1
                rho.append(cov / np.sqrt(v0*v1))
                pl, pf = tau_pmf_joint(P), tau_pmf_indep(m0, m1)
                sd = lambda p: np.sqrt(max((p*atoms**2).sum() - (p*atoms).sum()**2, 0))
                sl.append(sd(pl)); sf.append(sd(pf))
            return dict(kind="2D", J=J, n_atoms=len(atoms),
                        rho=float(np.mean(rho)), sd_l=float(np.mean(sl)),
                        sd_f=float(np.mean(sf)))
        z0 = sorted(k for k in z.files if k.startswith("p_y0_scaled"))
        if not z0:
            return None
        J = np.asarray(z[z0[0]]).shape[-1]
        return dict(kind="1D", J=J, n_atoms=2*J-1, rho=float("nan"),
                    sd_l=float("nan"), sd_f=float("nan"))


print(f"{'model':16s} {'kind':5s} {'J':>6s} {'atoms':>7s} {'rho':>8s} "
      f"{'sd_learn':>9s} {'sd_indep':>9s} {'ratio':>7s}")
print("-" * 76)
for cell in sys.argv[1:]:
    fs = [f for f in sorted(glob.glob(os.path.join(cell, "*", "*.npz")))
          if os.path.basename(f) != "summary.npz"]
    if not fs:
        fs = [f for f in sorted(glob.glob(os.path.join(cell, "*.npz")))
              if os.path.basename(f) != "summary.npz"]
    if not fs:
        print(f"{os.path.basename(cell.rstrip('/')):16s} (no npz)"); continue
    try:
        st = stats(fs[0])
    except Exception as exc:
        print(f"{os.path.basename(cell.rstrip('/')):16s} ERROR {exc}"); continue
    if st is None:
        continue
    r = st["sd_f"]/st["sd_l"] if st["sd_l"] > 0 else float("nan")
    print(f"{os.path.basename(cell.rstrip('/')):16s} {st['kind']:5s} {st['J']:6d} "
          f"{st['n_atoms']:7d} {st['rho']:8.3f} {st['sd_l']:9.4f} "
          f"{st['sd_f']:9.4f} {r:7.3f}")
