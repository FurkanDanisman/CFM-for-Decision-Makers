"""Is the arm noise shared or independent, per benchmark?

The 1D heads turn two marginals into a law for tau by ASSUMING a coupling,
and Var(tau | x) = s0^2 + s1^2 - 2 rho s0 s1 makes that assumption's cost
explicit. This reports rho per benchmark so the assumption can be checked
rather than asserted.

Three regimes, and each needs a different check:

  measured      RealCause stores paired potential outcomes, so rho is
                estimated from residuals (see
                realcause_eval/plot_arm_noise_independence.py).

  exact         ComplexMech fixes the exogenous draw across the do(0)/do(1)
                passes, so Y_do1 - Y_do0 is true_cate with NO residual:
                rho = 1 and Var(tau | x) = 0 identically. Nothing to
                estimate -- the check is an identity, verified to float32
                epsilon.

  unavailable   the case-study npz stores the FACTUAL Y only, so the two
                arms are never seen together and rho is not recoverable from
                the written files; it has to be read off the generator.

    python check_arm_noise_coupling.py --deploy-root $DEPLOY_ROOT
"""
import argparse, glob, os
import numpy as np


def complexmech(deploy_root, max_files=50):
    pat = os.path.join(deploy_root, "R-PFN", "UWYK_Fig3_4", "data",
                       "complexmech", "*", "path_TY", "hide_0.0", "r*.npz")
    fs = sorted(glob.glob(pat))[:max_files]
    if not fs:
        return None
    gaps, spread = [], []
    for f in fs:
        with np.load(f) as z:
            y0 = np.asarray(z["Y_do0"], float)
            y1 = np.asarray(z["Y_do1"], float)
            tc = np.asarray(z["true_cate"], float)
        gaps.append(float(np.max(np.abs((y1 - y0) - tc))))
        spread.append(float(np.std(y0)))
    return dict(n=len(fs), max_gap=max(gaps), mean_gap=float(np.mean(gaps)),
                y_sd=float(np.mean(spread)))


def case_study(deploy_root):
    pat = os.path.join(deploy_root, "case_study_data", "d_variation",
                       "shift0", "d5", "*", "N1000", "*.npz")
    fs = sorted(glob.glob(pat))[:1]
    if not fs:
        return None
    with np.load(fs[0], allow_pickle=True) as z:
        keys = set(z.files)
        has_pair = ({"y0", "y1"} <= keys) or ({"Y_do0", "Y_do1"} <= keys)
        return dict(keys=sorted(keys), has_pair=has_pair,
                    exo=float(np.asarray(z["exo_std"]).reshape(-1)[0])
                    if "exo_std" in keys else None,
                    noise=float(np.asarray(z["noise_std"]).reshape(-1)[0])
                    if "noise_std" in keys else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy-root", default=os.environ.get("DEPLOY_ROOT", ""))
    a = ap.parse_args()

    print("=== ComplexMech ===")
    cm = complexmech(a.deploy_root)
    if cm is None:
        print("  no data found")
    else:
        rel = cm["max_gap"] / max(cm["y_sd"], 1e-12)
        print(f"  files={cm['n']}  max |(Y1-Y0) - true_cate| = {cm['max_gap']:.3e}"
              f"  (sd(Y) = {cm['y_sd']:.3g}, ratio {rel:.1e})")
        if rel < 1e-5:
            print("  -> the exogenous draw is SHARED exactly: rho = 1,")
            print("     Var(tau | x) = 0. A 1D head assuming independence")
            print("     predicts spread sqrt(s0^2 + s1^2) where the truth is 0.")
        else:
            print("  -> residual is non-trivial; the arms do NOT share noise exactly")

    print("\n=== case studies ===")
    cs = case_study(a.deploy_root)
    if cs is None:
        print("  no data found")
    else:
        print(f"  exo_std={cs['exo']}  noise_std={cs['noise']}")
        if cs["has_pair"]:
            print("  -> both arms stored; rho is estimable from the files")
        else:
            print("  -> FACTUAL Y only; the two arms are never observed for the")
            print("     same unit, so rho is not recoverable from these files.")
            print("     Read it off the generator, or regenerate storing both arms.")


if __name__ == "__main__":
    main()
