#!/usr/bin/env python
"""Percent-complete table for the a3 coverage scoring.

    python benchmarks/a3_cov_progress.py --out $SCRATCH/a3_cov

One row per (model, smoother), one column per benchmark. A selector counts as
done when its .npz exists and is non-empty -- the same condition
submit_a3_cov.sbatch uses to skip it on resume, so this table and the job's own
behaviour can never disagree.

Model names contain underscores, so the filename is parsed from the END against
the known selector lists rather than by splitting on '_'.
"""
from __future__ import annotations
import argparse, glob, os, re

RC = ["IHDP", "ACIC", "CPS", "PSID", "PSID_bal"]
CM = [f"n{n}" for n in (5, 10, 20, 30, 40, 50)]
CASES = ["Observed_Confounder", "Backdoor_Criterion", "Observed_Mediator",
         "Observed_Mediator_and_Confounder", "Unobserved_Confounder",
         "Frontdoor_Criterion"]
CS_D = ["2", "3", "5", "10", "20", "30", "40", "50"]
CS_SHIFT = ["0", "+2", "-2"]
# A case-study "selector" is one CELL: 6 cases x 8 d x 3 shifts = 144.
CS = [(d, c, s) for c in CASES for d in CS_D for s in CS_SHIFT]
SEL = {"rc": RC, "cm": CM, "cs": CS}


def parse(stem):
    """'<bench>_<model>_<selector>' -> (bench, model, selector), or None.

    Matched from the right: PSID_bal and every case name contain underscores, so
    splitting left-to-right would attribute part of the selector to the model.
    """
    for bench, sels in (("rc", RC), ("cm", CM)):
        if not stem.startswith(bench + "_"):
            continue
        rest = stem[len(bench) + 1:]
        for s in sorted(sels, key=len, reverse=True):
            if rest.endswith("_" + s):
                return bench, rest[: -len(s) - 1], s
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the OUT dir given to submit_a3_cov")
    ap.add_argument("--smoothers", nargs="+", default=["none", "malc"])
    a = ap.parse_args()

    found = {}
    for sm in a.smoothers:
        # rc / cm: one flat file per selector
        for f in glob.glob(os.path.join(a.out, sm, "*.npz")):
            if os.path.getsize(f) == 0:
                continue
            p = parse(os.path.basename(f)[:-4])
            if p:
                found.setdefault((p[1], sm), {}).setdefault(p[0], set()).add(p[2])
        # cs: one file PER CELL, perreal/<model>/<stage>__d<D>_<Case>__shift<S>.npz
        # -- the layout final_table.py reads. Counted as cells, not cases, since
        # a case is only complete when all its (shift, d) cells are.
        for d in glob.glob(os.path.join(a.out, sm, "perreal", "*")):
            if not os.path.isdir(d):
                continue
            model = os.path.basename(d)
            for f in glob.glob(os.path.join(d, "*.npz")):
                if os.path.getsize(f) == 0:
                    continue
                m = re.match(r"^[a-z]+__d(\d+)_(.+)__shift(\S+)\.npz$",
                             os.path.basename(f))
                if m:
                    found.setdefault((model, sm), {}).setdefault(
                        "cs", set()).add((m.group(1), m.group(2), m.group(3)))

    if not found:
        print(f"nothing under {a.out}/{{{','.join(a.smoothers)}}}/ yet")
        return

    w = max(len(m) for m, _ in found) + 2
    print(f"\n{'model':{w}s} {'smoother':10s} "
          + "".join(f"{b:>18s}" for b in SEL) + f"{'TOTAL':>10s}")
    for (model, sm) in sorted(found):
        got, exp, row = 0, 0, f"{model:{w}s} {sm:10s}"
        for b, sels in SEL.items():
            g = len(found[(model, sm)].get(b, ()))
            got += g; exp += len(sels)
            row += f"{f'{g}/{len(sels)}':>18s}"
        print(row + f"{f'{100*got//exp}%':>10s}")

    # Name what is missing: a percentage does not say which selector to rerun.
    print()
    for (model, sm) in sorted(found):
        for b, sels in SEL.items():
            have = found[(model, sm)].get(b, ())
            miss = [s for s in sels if s not in have]
            if not miss:
                continue
            if b == "cs":
                by_case = {}
                for d, c, sh in miss:
                    by_case.setdefault(c, 0)
                    by_case[c] += 1
                txt = " ".join(f"{c}({n})" for c, n in sorted(by_case.items()))
                print(f"  missing {model} [{sm}] cs cells: {txt}")
            else:
                print(f"  missing {model} [{sm}] {b}: {' '.join(miss)}")
    print()


if __name__ == "__main__":
    main()
