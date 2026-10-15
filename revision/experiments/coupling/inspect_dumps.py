"""E3 prep: record what one density dump per (model, benchmark) contains, so the per-arm marginal scorer can
be written against the real format (raw-unit grid of p(Y^do(0)), p(Y^do(1)); per-query truth alignment).
Reads only. Writes a text report.

  python inspect_dumps.py OUT_TXT
"""
import glob
import os
import sys

import numpy as np

S = os.environ["SCRATCH"]
T1 = "dopfn_repro_joint2d_262k_t1"
# label | glob of ONE dump (first match is used)
SPECS = [
    ("RC IHDP orig", f"{S}/rc_dens_uni/**/IHDP*r000.npz"),
    ("RC IHDP t1", f"{S}/dumps_all/{T1}/rc/**/IHDP*r000.npz"),
    ("RC IHDP eta0", f"{S}/rc_dens_eta0/**/IHDP*r000.npz"),
    ("CS orig", f"{S}/cs_dvar_dens/shift0/d5/ctx1000/**/*Observed_Confounder*r*.npz"),
    ("CS t1", f"{S}/dumps_all/{T1}/cs/shift0/d5/ctx1000/**/*Observed_Confounder*r*.npz"),
    ("CS eta0", f"{S}/cs_dvar_eta0/shift0/d5/ctx1000/**/*Observed_Confounder*r*.npz"),
    ("CM dopfn_native", f"{S}/cmech_dumps/dopfn_native/N1000/**/*.npz"),
    ("CM t1", f"{S}/cmech_dumps/{T1}/N1000/**/*.npz"),
    ("CM uwyk1d", f"{S}/cmech_dumps/uwyk1d/N1000/**/*.npz"),
    ("CM graph2d", f"{S}/cmech_dumps/graph2d/N1000/**/*.npz"),
    ("CM cpfn1d", f"{S}/cmech_dumps/cpfn1d_j1024/N1000/**/*.npz"),
    ("CM cpfn2d", f"{S}/cmech_dumps/cpfn2d_eta0/N1000/**/*.npz"),
]


def desc(v):
    a = np.asarray(v)
    if a.dtype.kind in "OUS":
        return f"{a.dtype} {a.shape} {str(a.reshape(-1)[:3])[:80]}"
    if a.size <= 3:
        return f"{a.dtype} {a.shape} = {a.reshape(-1).tolist()}"
    f = a.astype(float).reshape(-1)
    return f"{a.dtype} {a.shape} min={np.nanmin(f):.4g} max={np.nanmax(f):.4g} first={f[:3].round(4).tolist()}"


def main(out):
    lines = []
    for label, pat in SPECS:
        fs = sorted(glob.glob(pat, recursive=True))
        lines.append(f"## {label}   ({len(fs)} matches for {pat})")
        if not fs:
            continue
        # one file per method directory, at most 4
        seen = {}
        for f in fs:
            seen.setdefault(os.path.dirname(f), f)
        for f in list(seen.values())[:4]:
            lines.append(f"### {f}")
            try:
                with np.load(f, allow_pickle=True) as z:
                    for k in z.files:
                        lines.append(f"  {k}: {desc(z[k])}")
            except Exception as e:
                lines.append(f"  ERROR {e}")
    # data files with realized outcomes
    for pat in (f"{S}/hist/truth/IHDP_r000.npz", f"{S}/cmech_data_v2/**/5node/**/r0.npz",
                os.environ.get("CS_DATA", "") + "/shift0/d5/Observed_Confounder/N1000/Observed_Confounder_0.npz"):
        fs = sorted(glob.glob(pat, recursive=True))[:1]
        lines.append(f"## DATA {pat}  ({len(fs)})")
        for f in fs:
            with np.load(f, allow_pickle=True) as z:
                lines.append(f"### {f}")
                for k in z.files:
                    lines.append(f"  {k}: {desc(z[k])}")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main(sys.argv[1])
