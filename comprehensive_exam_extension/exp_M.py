"""Model comparison on the Observed Confounder case (main-text figure): the paper's CATE
baselines (Do-PFN paper, App. D.3.2) on the exact data sets of Experiments B, C and F.

    python exp_M.py --group st --n-real 100 --workers 8 --out results/exp_M

Conditions (76; data, seeds and 50/50 split identical to exp_B_shapes.py, exp_C.py, exp_F.py):
  in-prior; B scaled and unscaled, noise on all nodes, 5 families x 5 levels; C: 5 functions at
  lambda = 1 on the outcome equation; F: treated fraction 10, 20, 40, 80, 90%, categorical
  covariate k = 20, 10, 5, 3, 2, irrelevant covariates m = 1, 2, 5, 10, 20, categorical
  outcome k = 20, 10, 5, 3, 2.
Model groups (one cluster job each):
  st   S-learner (TabPFN), T-learner (TabPFN)
  x    X-learner (TabPFN)          mu_0, mu_1, imputed-effect models and propensity all TabPFN
  dml  DML (TabPFN)                R-learner (Nie and Wager, 2021): 2-fold cross-fitted TabPFN
                                   outcome and propensity models; final TabPFN on the R-loss
                                   pseudo-outcome, weights via a weighted resample
  cf   Causal forest (DML)         econml CausalForestDML, hyperparameters tuned (.tune)
  nn   TARNet, DragonNet           PyTorch, CATENets default architecture and training
                                   (3 x 200 representation, 2 x 100 heads, ELU, Adam 1e-4,
                                   L2 1e-4, batch 100, early stopping on a 30% split);
                                   DragonNet adds a propensity head and targeted regularization
                                   (Shi et al., 2019)
Do-PFN's errors on the same data sets are in the Exp B/C/F per_dataset.csv files (join on
condition and r). Scored on the test half: MSE_CATE and Rel. ATE error, as for Do-PFN.
Resumable: (condition, model) pairs already in the CSV are skipped.
A model that cannot be fit on a data set (e.g. TabPFN when every feature it sees is constant,
or a causal-forest cross-fit fold with no treated unit) is recorded as failed: NaN errors and
the exception in the `error` column; failures are counted per condition in the log.
"""
import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from generation import _cell_seed, generate_realization  # noqa: E402

CASE, CASE_IDX = "Observed_Confounder", 0
FAMILIES = {"gamma": (16, 8, 4, 2, 1), "bimodal": (1, 1.5, 2, 2.5, 3), "contam": (2, 3, 5, 7, 10),
            "lognorm": (0.25, 0.5, 0.75, 1.0, 1.25), "gennorm": (1.5, 1.25, 1.0, 0.75, 0.6)}
GROUPS = {"st": ("S-learner (TabPFN)", "T-learner (TabPFN)"), "x": ("X-learner (TabPFN)",),
          "dml": ("DML (TabPFN)",), "cf": ("Causal forest (DML)",), "nn": ("TARNet", "DragonNet")}

p = argparse.ArgumentParser()
p.add_argument("--group", required=True, choices=list(GROUPS))
p.add_argument("--n", type=int, default=200)
p.add_argument("--n-real", type=int, default=100)
p.add_argument("--workers", type=int, default=1)
p.add_argument("--seed-base", type=int, default=0)
p.add_argument("--out", default="results/exp_M")
a = p.parse_args()


def conditions():
    """(row, dial, level, generator kwargs); row/dial/level match the Do-PFN CSVs."""
    yield "in_prior", "in_prior", 0.0, {}
    for scale in ("scaled", "unscaled"):
        for fam, levels in FAMILIES.items():
            for lv in levels:
                yield f"B_{scale}", fam, float(lv), dict(noise_dist=fam, noise_param=lv, noise_target="all",
                                                         noise_standardize=(scale == "scaled"))
    for g in ("sin", "sign", "cubic", "expsq", "sin3"):
        yield "C", g, 1.0, dict(mix_g=g, mix_lambda=1.0, mix_target="outcome")
    for dial, name, levels in (("treat", "treat_frac", (0.1, 0.2, 0.4, 0.8, 0.9)),
                               ("covk", "cov_k", (20, 10, 5, 3, 2)),
                               ("irrel", "extra_cols", (1, 2, 5, 10, 20)),
                               ("outk", "out_k", (20, 10, 5, 3, 2))):
        for lv in levels:
            yield f"F_{dial}", dial, float(lv), {name: lv}


def realization(r, kw):
    for attempt in range(50):
        seed = _cell_seed(a.seed_base, CASE_IDX, a.n, r, attempt)
        try:
            return seed, generate_realization(CASE, a.n, seed, **kw)
        except ValueError:
            continue
    raise RuntimeError(f"r{r}: 50 resamples all non-finite")


# ── models: fit on (X, T, Y) of the train half, return tau_hat on the test covariates ──────────
def _tabpfn_reg(seed):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", random_state=seed % 2**31)


def _tabpfn_clf(seed):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    return TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=seed % 2**31)


def s_learner(X, T, Y, Xte, seed):
    m = _tabpfn_reg(seed).fit(np.c_[T, X], Y)
    return m.predict(np.c_[np.ones(len(Xte)), Xte]) - m.predict(np.c_[np.zeros(len(Xte)), Xte])


def _arm_models(X, T, Y, seed):
    return [_tabpfn_reg(seed).fit(X[T == t], Y[T == t]) for t in (0, 1)]


def t_learner(X, T, Y, Xte, seed):
    m0, m1 = _arm_models(X, T, Y, seed)
    return m1.predict(Xte) - m0.predict(Xte)


def x_learner(X, T, Y, Xte, seed):
    m0, m1 = _arm_models(X, T, Y, seed)
    d1 = Y[T == 1] - m0.predict(X[T == 1])            # imputed effects of the treated
    d0 = m1.predict(X[T == 0]) - Y[T == 0]            # ... and of the controls
    tau1 = _tabpfn_reg(seed).fit(X[T == 1], d1)
    tau0 = _tabpfn_reg(seed).fit(X[T == 0], d0)
    e = _tabpfn_clf(seed).fit(X, T).predict_proba(Xte)[:, 1]
    return e * tau0.predict(Xte) + (1 - e) * tau1.predict(Xte)


def dml(X, T, Y, Xte, seed):
    """R-learner: 2-fold cross-fitted m(x) = E[Y | x] and e(x) = P(T = 1 | x), then the final
    TabPFN fit of (Y - m) / (T - e) on x with weights (T - e)^2, applied as a weighted resample
    (TabPFN takes no sample weights)."""
    rng = np.random.default_rng(seed)
    folds = rng.permutation(len(Y)) % 2
    m_hat, e_hat = np.zeros(len(Y)), np.zeros(len(Y))
    for k in (0, 1):
        fit, ev = folds != k, folds == k
        m_hat[ev] = _tabpfn_reg(seed).fit(X[fit], Y[fit]).predict(X[ev])
        if len(np.unique(T[fit])) == 2:
            e_hat[ev] = _tabpfn_clf(seed).fit(X[fit], T[fit]).predict_proba(X[ev])[:, 1]
        else:
            e_hat[ev] = T[fit].mean()
    res_t, res_y = T - e_hat, Y - m_hat
    w = res_t ** 2
    keep = w > 1e-12
    pick = rng.choice(np.flatnonzero(keep), size=len(Y), replace=True, p=w[keep] / w[keep].sum())
    return _tabpfn_reg(seed).fit(X[pick], res_y[pick] / res_t[pick]).predict(Xte)


def causal_forest(X, T, Y, Xte, seed):
    from econml.dml import CausalForestDML
    cf = CausalForestDML(discrete_treatment=True, random_state=seed % 2**31)
    cf.tune(Y, T, X=X)
    cf.fit(Y, T, X=X)
    return cf.effect(Xte)


def _net(X, T, Y, Xte, seed, dragon):
    """TARNet / DragonNet with the CATENets defaults (catenets/models/constants.py)."""
    import torch
    from torch import nn
    torch.manual_seed(seed % 2**31)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Y))
    n_val = int(round(0.3 * len(Y)))
    va, tr = idx[:n_val], idx[n_val:]
    f = lambda v: torch.tensor(v, dtype=torch.float32)   # noqa: E731
    Xt, Tt, Yt = f(X), f(T), f(Y)

    def mlp(d_in, units, layers):
        mods = []
        for _ in range(layers):
            mods += [nn.Linear(d_in, units), nn.ELU()]
            d_in = units
        return nn.Sequential(*mods)

    rep = mlp(X.shape[1], 200, 3)
    heads = nn.ModuleList([nn.Sequential(mlp(200, 100, 2), nn.Linear(100, 1)) for _ in range(2)])
    prop = nn.Linear(200, 1)                      # DragonNet propensity head
    eps = nn.Parameter(torch.zeros(1))            # DragonNet targeted-regularization epsilon
    params = list(rep.parameters()) + list(heads.parameters()) + (list(prop.parameters()) + [eps] if dragon else [])
    opt = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-4)

    def loss(ix):
        z = rep(Xt[ix])
        mu0, mu1 = heads[0](z).squeeze(1), heads[1](z).squeeze(1)
        t, y = Tt[ix], Yt[ix]
        mu = t * mu1 + (1 - t) * mu0
        out = ((y - mu) ** 2).mean()
        if dragon:
            e = torch.sigmoid(prop(z).squeeze(1)).clamp(0.01, 0.99)
            out = out + nn.functional.binary_cross_entropy(e, t)
            q = mu + eps * (t / e - (1 - t) / (1 - e))
            out = out + ((y - q) ** 2).mean()
        return out

    best, best_state, bad = np.inf, None, 0
    for it in range(10000):
        batch = tr[rng.permutation(len(tr))[:100]]
        opt.zero_grad()
        loss(batch).backward()
        opt.step()
        with torch.no_grad():
            v = float(loss(va))
        if v < best - 1e-8:
            best, bad = v, 0
            best_state = [p_.detach().clone() for p_ in params]
        elif it >= 200:
            bad += 1
            if bad >= 10:
                break
    with torch.no_grad():
        for p_, s_ in zip(params, best_state):
            p_.copy_(s_)
        z = rep(f(Xte))
        return (heads[1](z) - heads[0](z)).squeeze(1).numpy().astype(np.float64)


def tarnet(X, T, Y, Xte, seed):
    return _net(X, T, Y, Xte, seed, dragon=False)


def dragonnet(X, T, Y, Xte, seed):
    return _net(X, T, Y, Xte, seed, dragon=True)


MODELS = {"S-learner (TabPFN)": s_learner, "T-learner (TabPFN)": t_learner, "X-learner (TabPFN)": x_learner,
          "DML (TabPFN)": dml, "Causal forest (DML)": causal_forest, "TARNet": tarnet, "DragonNet": dragonnet}


def run(task):
    """One (condition, r): every model of the group on the same data set."""
    warnings.filterwarnings("ignore")
    row, dial, level, kw, r = task
    seed, real = realization(r, kw)
    X, T, Y, tau = real.X.astype(np.float64), real.T.astype(int), real.Y.astype(np.float64), \
        real.cate.astype(np.float64)
    perm = np.random.default_rng(seed).permutation(len(Y))
    tr, te = perm[: len(Y) // 2], perm[len(Y) // 2:]
    out = []
    ate = tau[te].mean()
    for model in GROUPS[a.group]:
        try:
            tau_hat = np.asarray(MODELS[model](X[tr], T[tr], Y[tr], X[te], seed), dtype=np.float64).ravel()
        except Exception as err:   # the model cannot be fit on this data set: recorded as failed (NaN)
            out.append(dict(case=CASE, row=row, dial=dial, level=level, r=r, seed=seed, model=model,
                            mse_cate=np.nan, rel_ate=np.nan, ate=float(ate), ate_hat=np.nan,
                            error=f"{type(err).__name__}: {err}"[:200]))
            continue
        out.append(dict(case=CASE, row=row, dial=dial, level=level, r=r, seed=seed, model=model,
                        mse_cate=float(np.mean((tau_hat - tau[te]) ** 2)),
                        rel_ate=float(abs(tau_hat.mean() - ate) / abs(ate)) if ate != 0 else np.nan,
                        ate=float(ate), ate_hat=float(tau_hat.mean()), error=""))
    return out


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    out = os.path.join(os.path.abspath(a.out), a.group)
    os.makedirs(out, exist_ok=True)
    csv = os.path.join(out, "per_dataset.csv")
    rows = pd.read_csv(csv).to_dict("records") if os.path.exists(csv) else []
    done = {(d["row"], d["dial"], float(d["level"])) for d in rows}
    pool = ProcessPoolExecutor(a.workers) if a.workers > 1 else None
    for row, dial, level, kw in conditions():
        if (row, dial, level) in done:
            continue
        tasks = [(row, dial, level, kw, r) for r in range(a.n_real)]
        res = pool.map(run, tasks) if pool else map(run, tasks)
        new = [x for chunk in res for x in chunk]
        rows += new
        pd.DataFrame(rows).to_csv(csv, index=False)
        g = pd.DataFrame(new).groupby("model").mse_cate
        d, nfail = g.median(), g.apply(lambda s: int(s.isna().sum()))
        print(f"{row:10s} {dial:8s} {level:<6g} " + "  ".join(f"{m}={v:.4g} (failed {nfail[m]})"
                                                             for m, v in d.items()), flush=True)
