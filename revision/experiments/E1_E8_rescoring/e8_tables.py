"""E8 tables: |c - 0.95| with its Monte Carlo SE, and length, for the published models.

c is the mean over contexts (realizations) of each context's coverage over its
queries; SE = sd / sqrt(n_contexts). Length is the mean central-95% length, same
aggregation. Reads per-realization arrays only, nothing is rescored:

  RealCause IHDP, ACIC   <e1-perreal>/<label>/<stage>__<DS>__-.npz    (truth = realized delta, E1)
  RealCause CPS, PSID*   <perreal>/<label>/<stage>__<DS>__-.npz        (truth = RealCause ite)
  Case study             <perreal>/<label>/<stage>__d<D>_<Case>__shift<S>.npz
                         pooled, per shift (d pooled), per d (shifts pooled)
  ComplexMech (all rho)  <perreal>/<cmech label>/<stage>__cmech_n<N>__-.npz

stage = raw | malc. Writes e8_long.csv (every slice) and e8_tables.md.

  python revision/experiments/E1_E8_rescoring/e8_tables.py \
      --perreal $SCRATCH/perreal --e1-perreal $SCRATCH/revision_e1/perreal \
      --out-dir $SCRATCH/revision_e1/tables
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

# display | RealCause/case-study perreal label | method key | ComplexMech perreal label
MODELS = [
    ("Do-PFN 1D",          "orig",    "dopfn_native",        "dopfn_native"),
    ("Do-PFN 2D",          "joint2d", "dopfn_repro_joint2d", "dopfn_repro_joint2d"),
    ("UWYK 1D (noanc)",    "orig",    "uwyk1d-noanc",        "uwyk1d"),
    ("UWYK 1D (v3a)",      "orig",    "uwyk1d-v3a",          "uwyk1d"),
    ("UWYK 2D (noanc)",    "orig",    "graph2d-noanc",       "graph2d"),
    ("UWYK 2D (v3a)",      "orig",    "graph2d-v3a",         "graph2d"),
    ("CausalPFN 1D",       "orig",    "cpfn1d",              "cpfn1d_j1024"),
    ("CausalPFN 2D",       "eta0",    "cpfn2d",              "cpfn2d_eta0"),
]
RC_E1 = ["IHDP", "ACIC"]
RC_OLD = ["CPS", "PSID", "PSID_bal"]
CASES = ["Observed_Confounder", "Observed_Mediator", "Observed_Mediator_and_Confounder",
         "Unobserved_Confounder", "Frontdoor_Criterion", "Backdoor_Criterion"]
SHIFTS = ["shift-2", "shift0", "shift+2"]
DS = ["5", "10", "20", "30", "40", "50"]
NODES = [5, 10, 20, 30, 40, 50]
STAGES = ["raw", "malc"]
MISSING = []


def load(files, method):
    """Per-realization cover and length for one method, concatenated over files."""
    cov, ln = [], []
    for f in files:
        if not os.path.exists(f):
            MISSING.append(f)
            continue
        with np.load(f) as z:
            if f"{method}__cover" not in z.files:
                MISSING.append(f"{f} [no key {method}; has "
                               f"{sorted({k.rsplit('__', 1)[0] for k in z.files})}]")
                continue
            cov.append(np.asarray(z[f"{method}__cover"], float).ravel())
            ln.append(np.asarray(z[f"{method}__length"], float).ravel())
    if not cov:
        return None
    c, l = np.concatenate(cov), np.concatenate(ln)
    n = c.size
    se = lambda v: float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return dict(n=n, cov=float(c.mean()), dev=abs(float(c.mean()) - 0.95), se=se(c),
                len=float(l.mean()), len_se=se(l))


def cell(s):
    return "—" if s is None else f"{s['dev']:.3f} ({s['se']:.3f}) · {s['len']:.3g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perreal", required=True)
    ap.add_argument("--e1-perreal", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rows, md = [], []

    def add(bench, case, sl, name, stage, s):
        if s is not None:
            rows.append(dict(benchmark=bench, case=case, slice=sl, model=name, stage=stage,
                             n_real=s["n"], coverage=s["cov"], abs_dev=s["dev"], se=s["se"],
                             length=s["len"], length_se=s["len_se"]))

    def table(title, cols, getter):
        for stage in STAGES:
            md.append(f"\n### {title} — {stage}\n")
            md.append("| model | " + " | ".join(c for c, _ in cols) + " |")
            md.append("|---|" + "---|" * len(cols))
            for name, *_ in MODELS:
                md.append(f"| {name} | " + " | ".join(
                    cell(getter(name, stage, key)) for _, key in cols) + " |")

    md.append("# E8 — |ĉ − 0.95| (MC SE over contexts) · mean 95% length\n")
    md.append("IHDP/ACIC scored against realized Δ = y1 − y0 (E1). "
              "ComplexMech is all-ρ. Case study d ∈ {5,…,50}.")

    # RealCause
    cache = {}
    for name, lbl, meth, _ in MODELS:
        for stage in STAGES:
            for ds in RC_E1 + RC_OLD:
                root = a.e1_perreal if ds in RC_E1 else a.perreal
                s = load([os.path.join(root, lbl, f"{stage}__{ds}__-.npz")], meth)
                cache[(name, stage, ds)] = s
                add("RealCause", ds, "all", name, stage, s)
    md.append("\n## RealCause")
    table("RealCause", [(d, d) for d in RC_E1 + RC_OLD],
          lambda n, st, k: cache[(n, st, k)])

    # Case study
    md.append("\n## Case study")
    for case in CASES:
        cs = {}
        for name, lbl, meth, _ in MODELS:
            f = lambda d, sh, st: os.path.join(a.perreal, lbl, f"{st}__d{d}_{case}__{sh}.npz")
            for stage in STAGES:
                slices = [("pooled", [f(d, sh, stage) for d in DS for sh in SHIFTS])]
                slices += [(sh, [f(d, sh, stage) for d in DS]) for sh in SHIFTS]
                slices += [(f"d{d}", [f(d, sh, stage) for sh in SHIFTS]) for d in DS]
                for sl, files in slices:
                    s = load(files, meth)
                    cs[(name, stage, sl)] = s
                    add("CaseStudy", case, sl, name, stage, s)
        cols = [("pooled", "pooled")] + [(sh, sh) for sh in SHIFTS] + [(f"d={d}", f"d{d}") for d in DS]
        table(f"Case study: {case}", cols, lambda n, st, k: cs[(n, st, k)])

    # ComplexMech
    cm = {}
    for name, _, meth, clbl in MODELS:
        for stage in STAGES:
            for n in NODES:
                s = load([os.path.join(a.perreal, clbl, f"{stage}__cmech_n{n}__-.npz")], meth)
                cm[(name, stage, n)] = s
                add("ComplexMech", "all-rho", f"n{n}", name, stage, s)
    md.append("\n## ComplexMech (all ρ, N=1000, subset=total)")
    table("ComplexMech", [(f"n={n}", n) for n in NODES], lambda nm, st, k: cm[(nm, st, k)])

    with open(os.path.join(a.out_dir, "e8_long.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["benchmark"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out_dir, "e8_tables.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"{len(rows)} slices -> {a.out_dir}")
    if MISSING:
        print(f"\n{len(MISSING)} missing inputs:", file=sys.stderr)
        for m in sorted(set(MISSING)):
            print("  " + m, file=sys.stderr)


if __name__ == "__main__":
    main()
