"""Outcome-level readouts of one shard of dumps, for every model format (1D and 2D), run before the dumps are deleted.
python score_outcomes_all.py DUMP_DIR OUT_NPZ [TRUTH_Y_DIR SHARD]

Per query and outcome t (raw units): mean, sd, and the 95% predictive interval (lo, hi) of the predicted distribution
of y^{do(t)}: the 2.5% / 97.5% quantiles on the bin edges (half-normal quantiles inside a tail), as the paper's read-out.
  1D dumps: the model's own outcome distributions (Do-PFN: K bins; UWYK: [left tail, K bins, right tail]; CausalPFN-C).
  2D dumps: the two marginals of the joint.
2D dumps also get: the ITE interval with the learned dependence switched off (rho = 0), and, with TRUTH_Y_DIR, whether the
true pair (y0, y1) lies in the joint's 95% region (fewest cells reaching 0.95) for each truth version in the file.
Saves idx and, per dataset, arrays of shape (n_queries,)."""
import glob, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../bodik")); sys.path.insert(0, os.path.join(HERE, "../../../UWYK_Fig3_4"))
from bodik_cf import _map                                      # noqa: E402
import bodik_cf_readouts as R                                  # noqa: E402

K2 = np.sqrt(2 / np.pi)
dump_dir, out = sys.argv[1], sys.argv[2]
truth_dir = sys.argv[3] if len(sys.argv) > 3 else None
shard = int(sys.argv[4]) if len(sys.argv) > 4 else 0


def arms(z):
    """per arm t: E (K+1 raw edges), W (q, K), T (q, 4) = (pL, sL, pR, sR) raw; and the joint (q, K, K) if 2D"""
    es = np.asarray(z["edges"], np.float64).reshape(-1)
    J = np.asarray(z["p_joint_scaled"], np.float64) if "p_joint_scaled" in z.files else None
    res = {}
    for t in (0, 1):
        s, m = _map(z, t); E = es * s + m
        if J is not None and J.ndim == 3:
            Jn = J / J.sum((1, 2), keepdims=True)
            W = Jn.sum(2) if t == 0 else Jn.sum(1)
            T = np.zeros((W.shape[0], 4)) + [0, 1, 0, 1]
        else:
            P = np.asarray(z[f"p_y{t}_scaled"], np.float64); P = P / P.sum(1, keepdims=True)
            if P.shape[1] == es.size + 1:                       # UWYK: [left tail, K bins, right tail]
                soft = lambda x: np.logaddexp(0.0, x); fl = float(z["scale_floor"])
                sL = float(z["base_s_left"]) * (soft(np.asarray(z[f"sL_raw_{t}"], np.float64)) + fl) * s
                sR = float(z["base_s_right"]) * (soft(np.asarray(z[f"sR_raw_{t}"], np.float64)) + fl) * s
                W, T = P[:, 1:-1], np.stack([P[:, 0], sL, P[:, -1], sR], 1)
            else:
                W, T = P, np.zeros((P.shape[0], 4)) + [0, 1, 0, 1]
        res[t] = (E, W, T)
    return res, (J / J.sum((1, 2), keepdims=True) if J is not None and J.ndim == 3 else None), es


def moments(E, W, T, bm=None):
    c = 0.5 * (E[:-1] + E[1:]) if bm is None else bm
    w2 = np.diff(E) ** 2 / 12.0
    mu = (W * c).sum(1) + T[:, 0] * (E[0] - K2 * T[:, 1]) + T[:, 2] * (E[-1] + K2 * T[:, 3])
    m2 = (W * (c ** 2 + w2)).sum(1) \
        + T[:, 0] * (E[0] ** 2 - 2 * E[0] * K2 * T[:, 1] + T[:, 1] ** 2) \
        + T[:, 2] * (E[-1] ** 2 + 2 * E[-1] * K2 * T[:, 3] + T[:, 3] ** 2)
    return mu, np.sqrt(np.maximum(m2 - mu ** 2, 0.0))


def region_hit(Jq, E0, E1, y0, y1):
    K = Jq.shape[0]
    i0 = np.searchsorted(E0, y0, side="right") - 1; i1 = np.searchsorted(E1, y1, side="right") - 1
    if not (0 <= i0 < K and 0 <= i1 < K):
        return 0.0
    flat = Jq.ravel(); order = np.argsort(-flat)
    k = int(np.searchsorted(np.cumsum(flat[order]), 0.95, side="left")) + 1
    return float(i0 * K + i1 in set(order[:k].tolist()))


rows, idx = {}, []
for f in sorted(glob.glob(os.path.join(dump_dir, "*r[0-9][0-9][0-9].npz"))):
    i = int(re.search(r"r(\d+)\.npz$", f).group(1))
    with np.load(f, allow_pickle=True) as z:
        if "edges" not in z.files:
            continue
        A, J, es = arms(z)
        bm = np.asarray(z["bucket_means"], np.float64) if "bucket_means" in z.files else None
        rec = {}
        for t in (0, 1):
            E, W, T = A[t]
            mu, sd = moments(E, W, T, bm)
            lo, hi = R.hist_interval(np.tile(E, (W.shape[0], 1)), W, T)
            rec.update({f"mu{t}": mu, f"sd{t}": sd, f"lo{t}": lo, f"hi{t}": hi})
        if J is not None:
            from cate_density_metrics import _query_pmfs, interval_95
            from hist_interval import interval_hist
            atoms, pmfs, _ = _query_pmfs(f, joint_coupling="indep")
            uni = np.allclose(np.diff(atoms), atoms[1] - atoms[0], rtol=1e-3)
            iv = np.array([interval_hist(atoms, p) if uni else interval_95(atoms, p) for p in pmfs])
            rec["lo_ite_rho0"], rec["hi_ite_rho0"] = iv[:, 0], iv[:, 1]
            if truth_dir:
                tz = np.load(os.path.join(truth_dir, f"r{shard * 100 + i:03d}.npz"))
                for v, (k0, k1) in {"sh": ("y0", "y1"), "in": ("y0_ind", "y1_ind")}.items():
                    if k0 in tz.files and k1 in tz.files:
                        rec[f"joint_hit_{v}"] = np.array([region_hit(J[q], A[0][0], A[1][0], tz[k0][q], tz[k1][q])
                                                          for q in range(J.shape[0])])
    idx.append(i)
    for k, a in rec.items():
        rows.setdefault(k, []).append(np.asarray(a, np.float32))
np.savez(out, idx=np.array(idx), **{k: np.array(v) for k, v in rows.items()})
print(f"{out}: {len(idx)} datasets; keys {sorted(rows)}")
