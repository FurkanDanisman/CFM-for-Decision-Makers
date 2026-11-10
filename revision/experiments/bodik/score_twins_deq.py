"""Twins with the context outcome dequantized (y + U(-w, w)), and the no-dependence control, scored like Table 2.
python score_twins_deq.py DATA_ROOT OUT_PREFIX

Per variant (twins: w = 0.1, 0.25, 0.5; twinsind: w = 0, 0.1, 0.25, 0.5), covariate set (d = 10, 1) and model:
  mse          score_twins_bodik.per_query (the Table 2 read-out: the known twin's binary outcome conditions the 2D
               joint; 1D uses the other arm's marginal); 2D also with the dependence off (rho = 0)
  rho_pred     2D only: correlation of the predicted joint (raw bin centres), median over queries
  rho_true     correlation of the two twins' binary outcomes in the variant's truths
Aggregation as score_twins_bodik (all families: SE over families; UWYK: per context draw, SE over draws).
"""
import csv
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_twins_bodik as S                                   # noqa: E402
from bodik_cf import _map                                       # noqa: E402

MODELS = [("Do-PFN", "dopfn1d", None, "full"), ("Do-PFN 2D", "dopfn2d", None, "full"),
          ("CausalPFN-C", "cpfn1d", None, "full"), ("CausalPFN-C 2D", "cpfn2d", None, "full"),
          ("UWYK No-Anc", "uwyk1d", "noanc", "sub"), ("UWYK Anc", "uwyk1d", "v3a", "sub"),
          ("UWYK 2D No-Anc", "graph2d", "noanc", "sub"), ("UWYK 2D Anc", "graph2d", "v3a", "sub")]
VARIANTS = [("twins", w) for w in ("0.1", "0.25", "0.5")] + [("twinsind", w) for w in ("0", "0.1", "0.25", "0.5")]


def rho_pred(f, tag):
    with np.load(f, allow_pickle=True) as z:
        k = f"p_joint_scaled_{tag}" if tag and f"p_joint_scaled_{tag}" in z.files else "p_joint_scaled"
        J = np.asarray(z[k], np.float64)
        J = J / J.sum((1, 2), keepdims=True)
        es = np.asarray(z["edges"], np.float64).reshape(-1)
        (s0, m0), (s1, m1) = _map(z, 0), _map(z, 1)
    e0, e1 = es * s0 + m0, es * s1 + m1
    c0, c1 = 0.5 * (e0[:-1] + e0[1:]), 0.5 * (e1[:-1] + e1[1:])
    p0, p1 = J.sum(2), J.sum(1)
    mu0, mu1 = p0 @ c0, p1 @ c1
    v0, v1 = p0 @ c0 ** 2 - mu0 ** 2, p1 @ c1 ** 2 - mu1 ** 2
    cov = np.einsum("qij,i,j->q", J, c0, c1) - mu0 * mu1
    return cov / np.sqrt(np.maximum(v0 * v1, 1e-12))


def agg(got, sub):
    if not sub:
        V = np.concatenate([v for _, v in got])
        return V.mean(), V.std(ddof=1) / np.sqrt(V.size)
    draws = {}
    for k, v in got:
        draws.setdefault(k // S.NB, []).append(v)
    M = np.array([np.concatenate(v).mean() for v in draws.values()])
    return M.mean(), M.std(ddof=1) / np.sqrt(M.size)


def main(D, out):
    recs = []
    for kind, w in VARIANTS:
        for d in (10, 1):
            for label, mdir, tag, part in MODELS:
                sub = part == "sub"
                pname = f"{kind}_{part}_deq{w}"
                files = sorted(glob.glob(os.path.join(D, "dumps", mdir, pname, f"d{d}", "*r[0-9]*.npz")))
                if not files:
                    print("missing", pname, label, d, flush=True); continue
                is2d = "2D" in label
                rows = [(label, False)] + ([(label + " (ρ = 0)", True)] if is2d else [])
                for lab, ld_off in rows:
                    got = []
                    for f in files:
                        k, V = S.per_query(D, pname, f, tag, is2d, d, ld_off=ld_off)
                        got.append((k, V[:, 0]))
                    m, se = agg(got, sub)
                    r = dict(variant=kind, w=w, setting=f"d={d}", model=lab, mse=float(m), mse_se=float(se))
                    if is2d and not ld_off:
                        r["rho_pred"] = float(np.median(np.concatenate([rho_pred(f, tag) for f in files])))
                    t = np.load(os.path.join(D, pname, f"d{d}", "truths", "r000.npz"))
                    r["rho_true"] = float(np.corrcoef(t["y0"], t["y1"])[0, 1])
                    recs.append(r)
                    print(f"{kind:8s} w={w:4s} d={d:2d} {lab:26s} MSE {m:.4f} ± {se:.4f}"
                          + (f"  rho_pred {r['rho_pred']:.3f}" if "rho_pred" in r else "")
                          + f"  rho_true {r['rho_true']:.3f}", flush=True)
    keys = ["variant", "w", "setting", "model", "mse", "mse_se", "rho_pred", "rho_true"]
    with open(out + ".csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(recs)
    print(f"wrote {out}.csv", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
