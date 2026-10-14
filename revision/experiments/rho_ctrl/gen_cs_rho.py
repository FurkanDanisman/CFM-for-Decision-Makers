"""Case studies with a controlled cross-world coupling of the outcome noise.
python gen_cs_rho.py OUT_ROOT [--per-cell 10] [--ds 5 10 20 30 40 50] [--shifts 0 +2 -2] [--n 1000]

The paper's d-variation generator (case_study/d_variation/generation_d.py, _SampledSCM) unchanged, except
that the outcome's own noise is no longer the same draw in both arms:
    eps_Y(0) = eps,   eps_Y(1) = rho_knob * eps + sqrt(1 - rho_knob^2) * eps',   eps' ~ N(0, noise_std) fresh.
Every other draw (roots, hidden U, mediator noise, treatment noise) stays shared, as in the paper. A unit's
factual outcome uses the noise of its own arm. rho_knob = 1 reproduces the paper's generator exactly.

The true rho(x) = corr(Y(1), Y(0) | X = x) is computed exactly per query (fix x, redraw everything else,
K draws; the four scenarios have pre-treatment covariates), and a dataset's rho is the median over its
first Q = 100 rows (the harness's queries). Bins (by the COMPUTED rho): [-0.5, 0), [0, 0.5), [0.5, 1].
Adaptive filling per (scenario, d, shift): the knob is drawn uniformly from the range of the emptiest bin
still short of `per-cell`, the dataset goes to the bin its computed rho falls in (dropped if that bin is
full or rho < -0.5), until every bin has `per-cell` datasets.

Layout (the paper's npz backend reads it unchanged, CASE_STUDY_DATA_ROOT=OUT_ROOT/bin{b}; extra keys are
ignored by the harness): all d and shifts of one (bin, scenario) in one folder, so one harness call covers them:
    OUT_ROOT/bin{0,1,2}/{Case}/N{N}/{Case}_{k}.npz
    extra keys: d, shift, delta (N,) realized ITE Y(1) - Y(0), rho_knob, rho_true (dataset median), rho_true_q (Q,)
Run it once into an empty OUT_ROOT (no resume).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(REPO, "case_study"))
sys.path.insert(0, os.path.join(REPO, "case_study", "d_variation"))
from generation import ACTIVATIONS, _SampledSCM              # noqa: E402
from generation_d import _cell_seed_d, build_dag_d          # noqa: E402

CASES = ("Observed_Confounder", "Backdoor_Criterion", "Frontdoor_Criterion", "Unobserved_Confounder")
BINS = ((-0.5, 0.0), (0.0, 0.5), (0.5, 1.0))
SEED_BASE = 20261014
Q, K = 100, 2000


def bin_of(r):
    if not np.isfinite(r) or r < -0.5 or r > 1.0:
        return None
    return 0 if r < 0.0 else (1 if r < 0.5 else 2)


class CoupledSCM(_SampledSCM):
    """_SampledSCM with the outcome noise coupled at rho_knob across the two arms."""

    def __init__(self, nodes, N, rng, cate_shift, rho_knob, rng_extra):
        super().__init__(nodes, N=N, rng=rng, cate_shift=cate_shift)
        self.rho_knob = float(rho_knob)
        e0 = self._noise[self.y_name]
        e_new = rng_extra.normal(0.0, self.noise_std, size=N)
        self._noise_y1 = self.rho_knob * e0 + np.sqrt(max(1.0 - self.rho_knob ** 2, 0.0)) * e_new

    def forward(self, do_T=None, y_noiseless=False):
        if y_noiseless:
            return super().forward(do_T=do_T, y_noiseless=True)
        e0 = self._noise[self.y_name]
        if do_T is not None:
            self._noise[self.y_name] = self._noise_y1 if float(do_T) == 1.0 else e0
            try:
                return super().forward(do_T=do_T)
            finally:
                self._noise[self.y_name] = e0
        # observational: run once to get T, then the outcome noise of each unit's own arm
        vals = super().forward()
        t = vals[self.t_name]
        self._noise[self.y_name] = np.where(t == 1.0, self._noise_y1, e0)
        try:
            return super().forward()
        finally:
            self._noise[self.y_name] = e0


def true_rho(scm, X, names, rng):
    """rho(x) per query, exact: x fixed, every other draw redrawn (K), coupled exactly as generated."""
    out = np.empty(min(Q, X.shape[0]))
    for i in range(out.size):
        x = dict(zip(names, X[i]))
        draws = {}
        for n in scm.nodes:
            if n.observed or n.is_treatment:
                continue
            sd = scm.exo_std if n.kind == "root_normal" else scm.noise_std
            draws[n.name] = rng.normal(0.0, sd, size=K)
        e_new = rng.normal(0.0, scm.noise_std, size=K)
        ys = []
        for t in (0.0, 1.0):
            vals = {}
            for n in scm.nodes:
                if n.observed:
                    vals[n.name] = np.full(K, x[n.name])
                elif n.is_treatment:
                    vals[n.name] = np.full(K, t)
                elif n.kind == "root_normal":
                    vals[n.name] = draws[n.name]
                else:
                    lin = np.stack([vals[p] for p in n.parents], axis=-1) @ scm._weights[n.name]
                    eps = draws[n.name]
                    if n.is_outcome and t == 1.0:
                        eps = scm.rho_knob * eps + np.sqrt(max(1.0 - scm.rho_knob ** 2, 0.0)) * e_new
                    v = ACTIVATIONS[scm._activation[n.name]](lin) + eps
                    if n.is_outcome and scm._shift_outcome:
                        v = v + scm._shift * t
                    vals[n.name] = v
            ys.append(vals[scm.y_name])
        s0, s1 = ys[0].std(), ys[1].std()
        out[i] = np.nan if s0 < 1e-12 or s1 < 1e-12 else np.corrcoef(ys[0], ys[1])[0, 1]
    return out


def make(case, d, N, seed, shift, rho_knob):
    nodes = build_dag_d(case, d)
    scm = CoupledSCM(nodes, N, np.random.default_rng(seed), shift, rho_knob,
                     np.random.default_rng([seed, 7]))
    obs = scm.forward()
    y0 = scm.forward(do_T=0.0)[scm.y_name]
    y1 = scm.forward(do_T=1.0)[scm.y_name]
    mu0 = scm.forward(do_T=0.0, y_noiseless=True)[scm.y_name]
    mu1 = scm.forward(do_T=1.0, y_noiseless=True)[scm.y_name]
    names = [n.name for n in nodes if n.observed]
    X = np.stack([obs[m] for m in names], axis=-1)
    T, Y = obs[scm.t_name], obs[scm.y_name]
    for a in (X, T, Y, y0, y1, mu0, mu1):
        if not np.all(np.isfinite(a)):
            raise ValueError("non-finite values; resample")
    assert np.allclose(np.where(T == 1.0, y1, y0), Y), "factual outcome must equal its arm's potential outcome"
    rq = true_rho(scm, X, names, np.random.default_rng([seed, 11]))
    rec = dict(X=X.astype(np.float32), T=T.astype(np.float32), Y=Y.astype(np.float32),
               cate=(mu1 - mu0).astype(np.float32), mu_0=mu0.astype(np.float32), mu_1=mu1.astype(np.float32),
               feature_names=np.asarray(names, dtype=object), graph_edges=np.asarray(scm.graph_edges(), dtype=object),
               case_study=case, n_context=int(N), seed=int(seed), exo_std=scm.exo_std, noise_std=scm.noise_std,
               cate_shift=scm._shift if scm._shift_outcome else 0.0,
               delta=(y1 - y0).astype(np.float32), rho_knob=np.float32(rho_knob),
               rho_true=np.float32(np.nanmedian(rq)), rho_true_q=rq.astype(np.float32))
    return rec


def fill_cell(out, case, case_idx, d, shift_s, N, per_cell, max_tries=2000):
    shift = float(shift_s)
    have = [0, 0, 0]
    rng = np.random.default_rng([SEED_BASE, case_idx, d, int(shift * 10) + 100, N])
    tries = dropped = 0
    while min(have) < per_cell and tries < max_tries:
        target = int(np.argmin(have))
        knob = float(rng.uniform(*BINS[target]))
        seed = _cell_seed_d(SEED_BASE + int(shift * 10) + 100, d, case_idx, N, tries, 0)
        tries += 1
        try:
            rec = make(case, d, N, seed, shift, knob)
        except ValueError:
            continue
        b = bin_of(float(rec["rho_true"]))
        if b is None or have[b] >= per_cell:
            dropped += 1
            continue
        cell = os.path.join(out, f"bin{b}", case, f"N{N}")      # one folder per (bin, scenario): every d, shift
        os.makedirs(cell, exist_ok=True)
        k = len([f for f in os.listdir(cell) if f.endswith(".npz")])
        rec.update(d=np.int32(d), shift=np.float32(shift))
        np.savez_compressed(os.path.join(cell, f"{case}_{k}.npz"), **rec)
        have[b] += 1
    return have, tries, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--ds", nargs="+", type=int, default=[5, 10, 20, 30, 40, 50])
    ap.add_argument("--shifts", nargs="+", default=["0", "+2", "-2"])
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    t0 = time.time(); log = []
    for ci, case in enumerate(CASES):
        for d in a.ds:
            for s in a.shifts:
                have, tries, dropped = fill_cell(a.out, case, ci, d, s, a.n, a.per_cell)
                log.append(dict(case=case, d=d, shift=s, per_bin=have, tries=tries, dropped=dropped))
                print(f"{case:24s} d={d:2d} shift={s:>2s}  per bin={have}  tries={tries}  dropped={dropped}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(a.out, "manifest_cs.json"), "w") as fh:
        json.dump(log, fh, indent=1)


if __name__ == "__main__":
    main()
