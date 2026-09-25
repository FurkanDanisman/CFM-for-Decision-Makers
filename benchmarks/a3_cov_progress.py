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
import argparse, glob, os

RC = ["IHDP", "ACIC", "CPS", "PSID", "PSID_bal"]
CM = [f"n{n}" for n in (5, 10, 20, 30, 40, 50)]
CS = ["Observed_Confounder", "Backdoor_Criterion", "Observed_Mediator",
      "Observed_Mediator_and_Confounder", "Unobserved_Confounder",
      "Frontdoor_Criterion"]
SEL = {"rc": RC, "cm": CM, "cs": CS}


def parse(stem):
    """'<bench>_<model>_<selector>' -> (bench, model, selector), or None.

    Matched from the right: PSID_bal and every case name contain underscores, so
    splitting left-to-right would attribute part of the selector to the model.
    """
    for bench, sels in SEL.items():
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
        for f in glob.glob(os.path.join(a.out, sm, "*.npz")):
            if os.path.getsize(f) == 0:
                continue
            p = parse(os.path.basename(f)[:-4])
            if p:
                found.setdefault((p[1], sm), {}).setdefault(p[0], set()).add(p[2])

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
            miss = [s for s in sels if s not in found[(model, sm)].get(b, ())]
            if miss:
                print(f"  missing {model} [{sm}] {b}: {' '.join(miss)}")
    print()


if __name__ == "__main__":
    main()
