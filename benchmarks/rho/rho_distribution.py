#!/usr/bin/env python
"""Distribution of the cross-world (arm-noise) correlation rho per dataset, ONE estimator for every source.

ESTIMATOR (identical for every source, CausalPFN included): Table 9's `ols-on-X`
(benchmarks/arm_noise_table.py: _ols_resid + _rho). For one dataset with query covariates X and
both potential outcomes Y(0), Y(1) of the same query units,
    e_t = Y(t) - OLS fit of Y(t) on [1, X]        rho = corr(e0, e1)
No stored mean is used for any source, so the numbers are comparable across sources.

SOURCES
  cmech   : $SCRATCH/cmech_data_v2/complexmech/<n>node/path_TY/hide_<h>/r*.npz, keys X_test, Y_do0, Y_do1
            (1000 query units). Summarised for hide_0.0 (the paper's main tables) pooled and per node
            count, and for every file Table 9's glob matches (*node/*/*/r*.npz), to reconcile with Table 9.
  dopfn   : Do-PFN joint prior (training_dopfn_repro/prior.py sample_batch; the joint_2d variant uses the
            same prior, only the batching differs). Draw code and seeds of
            revision/experiments/E2_in_prior/gen_dopfn.py (new-world run): seq_len 2200, query rows
            1000..1499 (500 units), X = x_obs[:, 1:] (column 0 is T), Y(t) = y_do{t}.
  uwyk2d  : graph2d's paired prior (benchmarks/context_sweep/scm_prior.py generate_paired_sample_with_raw
            with E2's graph2d_scm_config.DEFAULT_SCM_CONFIG), draw code and seeds of
            E2_in_prior/gen_uwyk.py 2d new: 500 query units, X = X_intv[:, :F], Y(t) = Y_do{t}_raw.
  cpfn2d  : CausalPFN prior with eta1 := eta0, draw code and seeds of E2_in_prior/gen_cpfn.py 2d new:
            rows 1000..1499 of the 2048-row table (500 units), X = X, Y(0) = y0, Y(1) = E_y1 + (y0 - E_y0)
            (trainer.py's eta1 := eta0, exactly as gen_cpfn.py builds it).
  The independent-noise hooks of the E2 generators are NOT installed: only the prior's own paired arms
  are needed, and the draws here are the unmodified prior.

    python rho_distribution.py sample  --source {dopfn,uwyk2d,cpfn2d} --n 1000 --work W
    python rho_distribution.py cmech   --data $SCRATCH/cmech_data_v2 --work W
    python rho_distribution.py summarize --work W --out-dir revision/results/d262t1
Each sampler appends one row per dataset to W/<source>.csv and skips datasets already there,
so a timed-out job resumes where it stopped.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
E2 = os.path.join(REPO, "revision", "experiments", "E2_in_prior")
sys.path.append(os.path.join(REPO, "benchmarks"))
from arm_noise_table import _ols_resid, _rho, _combine  # noqa: E402  Table 9's estimator, verbatim

SEED = 20261003                       # E2_in_prior generators' SEED
N_CTX, N_Q = 1000, 500                # E2_in_prior/common.py
FIELDS = ["source", "dataset", "n_nodes", "hide", "n_units", "n_features", "rho", "error"]


def rho_ols(X, y0, y1):
    """Table 9 `ols-on-X`: residualise each arm on X (with intercept), correlate the residuals."""
    X = np.asarray(X, dtype=float)
    y0 = np.asarray(y0, dtype=float).ravel()
    y1 = np.asarray(y1, dtype=float).ravel()
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[1] == 0:
        X = np.zeros((y0.size, 0))
    return _rho(_ols_resid(X, y0), _ols_resid(X, y1))


def _np(a):
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().double().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(a, dtype=float)


# ── prior draws (E2_in_prior new-world runs, independent-truth hooks removed) ───────────────────
def drawer(source):
    """Return draw(r) -> (X_q, y0_q, y1_q) for world r of the given prior."""
    import torch
    if source == "dopfn":
        sys.path.insert(0, REPO)
        import training_dopfn_repro.prior as P                     # E2 gen_dopfn.py:13

        def draw(r):                                              # E2 gen_dopfn.py:45-52, 56-57
            rec = P.sample_batch(SEED + r * 1_000,
                                 P.PriorConfig(seq_len=2200, batch_size=1, on_nonfinite="resample"))
            x = rec["x_obs"][:, 0]
            q = slice(N_CTX, N_CTX + N_Q)
            return _np(x[q, 1:]), _np(rec["y_do0"][:, 0][q]), _np(rec["y_do1"][:, 0][q])
        return draw

    if source == "uwyk2d":
        sys.path.insert(0, os.path.join(REPO, "benchmarks", "context_sweep"))
        sys.path.insert(0, E2)
        import scm_prior as SP                                     # E2 gen_uwyk.py:22
        from graph2d_scm_config import DEFAULT_SCM_CONFIG as scm_config   # E2 gen_uwyk.py:28

        def draw(r):                                              # E2 gen_uwyk.py:67-72, 96-99
            random.seed(SEED + r); np.random.seed(SEED + r)
            s = SP.generate_paired_sample_with_raw(scm_seed=SEED + r * 1_000, n_train=N_CTX,
                                                   n_test=N_Q, scm_config=scm_config)
            F = len(s["feature_nodes"])
            return (_np(s["X_intv"][:, :F]), _np(s["Y_do0_raw"]).reshape(-1),
                    _np(s["Y_do1_raw"]).reshape(-1))
        return draw

    if source == "cpfn2d":
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        with initialize_config_dir(config_dir=os.path.join(os.environ["CAUSALPFN"], "conf"),
                                   version_base=None):             # E2 gen_cpfn.py:21-23
            cfg = compose(config_name="train", overrides=["+experiment=simple_configuration"])
        prior = instantiate(cfg.train_meta_dataset)

        def draw(r):                                              # E2 gen_cpfn.py:54-71, 79-80
            torch.manual_seed(SEED + r); random.seed(SEED + r); np.random.seed(SEED + r)
            s = prior.get_sample()
            X, y0, E0, E1 = (s[k].double() for k in ("X", "y0", "E_y0", "E_y1"))
            y1 = E1 + (y0 - E0)                                    # eta1 := eta0
            q = slice(N_CTX, N_CTX + N_Q)
            return _np(X[q]), _np(y0[q]).reshape(-1), _np(y1[q]).reshape(-1)
        return draw

    raise SystemExit(f"unknown source {source!r}")


def _done(path):
    if not os.path.isfile(path):
        return set()
    with open(path, newline="") as fh:
        return {int(r["dataset"]) for r in csv.DictReader(fh)}


def _append(path, row):
    new = not os.path.isfile(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
        fh.flush()


def cmd_sample(a):
    import torch
    torch.set_num_threads(a.threads)
    os.makedirs(a.work, exist_ok=True)
    path = os.path.join(a.work, f"{a.source}.csv")
    done = _done(path)
    draw = drawer(a.source)
    for r in range(a.n):
        if r in done:
            continue
        row = dict(source=a.source, dataset=r)
        try:
            X, y0, y1 = draw(r)
            rho, n = rho_ols(X, y0, y1)
            row.update(n_units=n, n_features=X.shape[1], rho=rho)
        except Exception as e:                                     # recorded, never silently dropped
            row.update(rho=float("nan"), error=f"{type(e).__name__}: {e}"[:200])
        _append(path, row)
        if r % 50 == 0:
            print(f"[{a.source}] r={r} rho={row['rho']}", flush=True)
    print(f"[{a.source}] done: {a.n} datasets in {path}", flush=True)


def cmd_cmech(a):
    """Table 9's ComplexMech loop (arm_noise_table._npz_rows, ols-on-X branch) per file."""
    os.makedirs(a.work, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.data, "complexmech", "*node", "*", "*", "r*.npz")))
    if not files:
        raise SystemExit(f"no ComplexMech npz under {a.data}/complexmech")
    path = os.path.join(a.work, "cmech.csv")
    if os.path.isfile(path):
        os.remove(path)
    for f in files:
        parts = f.split(os.sep)
        n_nodes = parts[-4].replace("node", "")
        hide = parts[-2].replace("hide_", "")
        rid = os.path.splitext(parts[-1])[0]
        row = dict(source="cmech", dataset=f"{parts[-4]}/{parts[-3]}/{parts[-2]}/{rid}",
                   n_nodes=n_nodes, hide=hide)
        try:
            z = np.load(f, allow_pickle=True)
            if {"mu0", "mu1"} <= set(z.files):
                raise ValueError("file stores mu0/mu1: Table 9 would use them, not ols-on-X")
            X = np.asarray(z["X_test"], float)
            rho, n = rho_ols(X, z["Y_do0"], z["Y_do1"])
            row.update(n_units=n, n_features=X.shape[1], rho=rho)
        except Exception as e:
            row.update(rho=float("nan"), error=f"{type(e).__name__}: {e}"[:200])
        _append(path, row)
    print(f"[cmech] {len(files)} files -> {path}", flush=True)


# ── summary ────────────────────────────────────────────────────────────────────────────────────
QS = (0.05, 0.25, 0.50, 0.75, 0.95)
SUMMARY_FIELDS = (["source", "n_datasets", "n_nan", "min"] + [f"q{int(q * 100):02d}" for q in QS]
                  + ["max", "mean", "sd", "share_gt_0.99", "share_lt_0.5", "share_lt_0",
                     "rho_fisher_combined"])


def summarise(name, rows):
    rs = np.array([float(r["rho"]) for r in rows], dtype=float)
    ns = [int(float(r["n_units"])) if r["n_units"] not in ("", None) else 0 for r in rows]
    fin = rs[np.isfinite(rs)]
    out = dict(source=name, n_datasets=len(rs), n_nan=int((~np.isfinite(rs)).sum()))
    if fin.size == 0:
        return out
    qv = np.quantile(fin, QS)
    out.update({"min": fin.min(), "max": fin.max(), "mean": fin.mean(),
                "sd": fin.std(ddof=1) if fin.size > 1 else 0.0,
                "share_gt_0.99": float((fin > 0.99).mean()),
                "share_lt_0.5": float((fin < 0.5).mean()),
                "share_lt_0": float((fin < 0).mean()),
                "rho_fisher_combined": _combine(list(rs), ns)[0]})
    out.update({f"q{int(q * 100):02d}": v for q, v in zip(QS, qv)})
    return out


def _read(path):
    if not os.path.isfile(path):
        print(f"note: missing {path}", file=sys.stderr)
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def cmd_summarize(a):
    os.makedirs(a.out_dir, exist_ok=True)
    cm = _read(os.path.join(a.work, "cmech.csv"))
    pri = {s: _read(os.path.join(a.work, f"{s}.csv")) for s in ("dopfn", "uwyk2d", "cpfn2d")}
    for s, rows in pri.items():                    # one row per dataset id, first n only
        seen, keep = set(), []
        for r in rows:
            d = int(r["dataset"])
            if d < a.n and d not in seen:
                seen.add(d); keep.append(r)
        pri[s] = sorted(keep, key=lambda r: int(r["dataset"]))

    def _hkey(h):
        try:
            return float(h)
        except ValueError:
            return float("nan")

    main = [r for r in cm if _hkey(r["hide"]) == 0.0]
    summ = [summarise("ComplexMech hide_0.0 (main tables), pooled", main)]
    for n in sorted({int(r["n_nodes"]) for r in main}):
        summ.append(summarise(f"ComplexMech hide_0.0, n={n}",
                              [r for r in main if int(r["n_nodes"]) == n]))
    hides = sorted({r["hide"] for r in cm}, key=_hkey)
    summ.append(summarise(f"ComplexMech, Table 9 glob (hides {', '.join(hides)}), pooled", cm))
    label = {"dopfn": "Do-PFN joint prior (2D)", "uwyk2d": "UWYK paired prior (graph2d)",
             "cpfn2d": "CausalPFN prior, eta1:=eta0 (cpfn2d)"}
    for s, rows in pri.items():
        summ.append(summarise(label[s], rows))

    with open(os.path.join(a.out_dir, "rho_per_dataset.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in cm + [r for rows in pri.values() for r in rows]:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    def fmt(v):
        return f"{v:+.4f}" if isinstance(v, (float, np.floating)) and math.isfinite(v) else str(v)

    with open(os.path.join(a.out_dir, "rho_distribution.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for s in summ:
            w.writerow({k: (f"{s[k]:.6g}" if isinstance(s.get(k), float) else s.get(k, ""))
                        for k in SUMMARY_FIELDS})

    hdr = ["source", "n", "nan", "min", "5%", "25%", "50%", "75%", "95%", "max", "mean", "sd",
           "> 0.99", "< 0.5", "< 0", "Fisher-combined"]
    L = ["## Cross-world correlation rho per dataset (Table 9 `ols-on-X`, every source)", "",
         "rho = corr(Y(0) - OLS_X[Y(0)], Y(1) - OLS_X[Y(1)]) over one dataset's query units "
         "(benchmarks/arm_noise_table.py `_ols_resid` + `_rho`). Shares are over finite rho; "
         "Fisher-combined is Table 9's inverse-variance column, for cross-checking.", "",
         "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for s in summ:
        vals = [s["source"], s["n_datasets"], s["n_nan"]] + [
            fmt(s.get(k, "—")) for k in SUMMARY_FIELDS[3:12]] + [
            f"{s[k]:.3f}" if k in s else "—" for k in ("share_gt_0.99", "share_lt_0.5", "share_lt_0")] + [
            fmt(s.get("rho_fisher_combined", "—"))]
        L.append("| " + " | ".join(str(v) for v in vals) + " |")
    L += ["", f"Query units per dataset: ComplexMech 1000 (X_test); priors {N_Q} "
          f"(E2_in_prior new-world draws, worlds 0..{a.n - 1}, seed base {SEED}).",
          "OLS-on-X misses nonlinearity in E[Y|X], which stays in the residual; rows are proxies "
          "(upper bounds when the shared noise dominates), measured identically for every source."]
    errs = [r for r in cm + [r for rows in pri.values() for r in rows] if r.get("error")]
    if errs:
        L += ["", f"{len(errs)} datasets failed (rho = nan); first: {errs[0]['source']} "
              f"{errs[0]['dataset']}: {errs[0]['error']}"]
    txt = "\n".join(L) + "\n"
    open(os.path.join(a.out_dir, "rho_distribution.md"), "w").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--source", required=True, choices=["dopfn", "uwyk2d", "cpfn2d"])
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--work", required=True)
    p.add_argument("--threads", type=int, default=2)
    p = sub.add_parser("cmech")
    p.add_argument("--data", default=os.path.join(os.environ.get("SCRATCH", ""), "cmech_data_v2"))
    p.add_argument("--work", required=True)
    p = sub.add_parser("summarize")
    p.add_argument("--work", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    {"sample": cmd_sample, "cmech": cmd_cmech, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
