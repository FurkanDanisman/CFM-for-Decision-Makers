"""Reduce one shard of harness dumps to the OT (2-Wasserstein barycenter) ATE density of each dataset.
python reduce_bary.py DUMP_DIR OUT_NPZ [TAG]                             (one variant, as before)
python reduce_bary.py DUMP_DIR --base BASE --shard K [--tags noanc,v3a] [--couplings indep,gauss:-0.5,...]

Per dump (one dataset): the per-query ITE densities (cate_density_metrics._query_pmfs, as score_shard.py),
each turned into a quantile function on the grid U (CDF linear between atoms, mass of an atom centred on
it); the 1D barycenter is the average of these quantile functions (Section 3.2, Eq. ate-quantile).
Single-outcome models get their ITE density from the two predicted outcome densities under an assumed
coupling: indep (rho = 0) or gauss:<rho>, the paper's Gaussian copula (gauss:1 = comonotone); a 2D model's
dump carries its learned joint, which is used whatever the coupling. Saved per variant: idx, theta (mean of
the per-query ITE means = the barycenter's centre), truth (mean of the dump's true_cate_per_query), U, Q
(n_datasets, len(U)) the barycenter quantiles. Multi-variant output: BASE[-TAG][-rho<r>]/shardK.npz, r
written as m0.5 / 0 / 0.5 / 1 (no rho suffix when --couplings is not given, i.e. a 2D model).
"""
import argparse, glob, os, re, sys
import multiprocessing as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../../UWYK_Fig3_4"))
from cate_density_metrics import _query_pmfs

U = np.linspace(0.0025, 0.9975, 399)


def quantiles(atoms, p):
    p = np.clip(p, 0, None); p = p / p.sum()
    cdf = np.cumsum(p) - p / 2                                  # mass of each atom centred on it
    keep = np.r_[True, np.diff(cdf) > 0]                        # strictly increasing for interp
    return np.interp(U, cdf[keep], atoms[keep])


def rho_label(coupling):
    if coupling == "indep":
        return "0"
    r = float(coupling.split(":", 1)[1])
    return ("m" if r < 0 else "") + f"{abs(r):g}"


def one(args):
    """one dump -> {(tag, coupling): (theta, truth, Q)} or None per variant"""
    f, variants = args
    out = {}
    for tag, coup in variants:
        got = _query_pmfs(f, tag, coup or "indep")
        if got is None:
            out[(tag, coup)] = None; continue
        atoms, pmfs, y = got
        ok = np.isfinite(pmfs).all(1) & (pmfs.sum(1) > 0)
        if not ok.any() or not np.isfinite(y).all():
            out[(tag, coup)] = None; continue
        pm = pmfs[ok]
        tau_hat = (pm @ atoms) / pm.sum(1)
        out[(tag, coup)] = (tau_hat.mean(), y.mean(), np.mean([quantiles(atoms, p) for p in pm], axis=0))
    return int(re.search(r"r(\d+)\.npz$", f).group(1)), out


if __name__ == "__main__":
    if len(sys.argv) >= 3 and not sys.argv[2].startswith("--"):          # legacy single-variant form
        dump_dir, outs = sys.argv[1], None
        variants = [(sys.argv[3] if len(sys.argv) > 3 else None, None)]
        single = sys.argv[2]
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("dump_dir"); ap.add_argument("--base", required=True); ap.add_argument("--shard", required=True)
        ap.add_argument("--tags", default=""); ap.add_argument("--couplings", default="")
        ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
        a = ap.parse_args()
        dump_dir, single = a.dump_dir, None
        tags = [t for t in a.tags.split(",") if t] or [None]
        coups = [c for c in a.couplings.split(",") if c] or [None]
        variants = [(t, c) for t in tags for c in coups]
    files = sorted(glob.glob(os.path.join(dump_dir, "*r[0-9][0-9][0-9].npz")))
    workers = 1 if single else a.workers
    jobs = [(f, variants) for f in files]
    res = list(mp.get_context("fork").Pool(workers).imap(one, jobs, chunksize=4)) if workers > 1 else [one(j) for j in jobs]
    for tag, coup in variants:
        rows = [(i, o[(tag, coup)]) for i, o in res if o[(tag, coup)] is not None]
        if not rows:
            sys.exit(f"{dump_dir}: no usable dumps for tag={tag} coupling={coup}")
        if single:
            out = single
        else:
            d = a.base + (f"-{tag}" if tag else "") + (f"-rho{rho_label(coup)}" if coup else "")
            os.makedirs(d, exist_ok=True); out = os.path.join(d, f"shard{a.shard}.npz")
        np.savez(out, idx=np.array([i for i, _ in rows]), theta=np.array([r[0] for _, r in rows]),
                 truth=np.array([r[1] for _, r in rows]), U=U, Q=np.array([r[2] for _, r in rows]))
        print(f"{out}: {len(rows)} datasets")
