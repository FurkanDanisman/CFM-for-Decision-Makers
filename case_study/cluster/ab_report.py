"""Report PEHE (+ L1-ATE) per variant × case for the A/B scaling test.

Walks <AB>/<variant>/<case>/ and handles both output shapes:
  * uniform (cpfn2d): per-realization <case>_r###.npz with pehe_raw + ate_pred/
    ate_raw + true_ate.
  * dopfn_bb: one summary.npz per case with arrays pehe[], ate_pred[], true_ate[].

Usage:
    python case_study/cluster/ab_report.py --ab $DEPLOY_ROOT/results_case_study/ab
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def _first(z, keys):
    for k in keys:
        if k in z.files:
            return np.asarray(z[k], dtype=np.float64).reshape(-1)
    return None


def _cell(cell_dir):
    """Return (pehe_mean, l1_mean, n) for one variant/case dir, or None."""
    summ = os.path.join(cell_dir, "summary.npz")
    if os.path.isfile(summ):
        with np.load(summ, allow_pickle=True) as z:
            pehe = _first(z, ["pehe", "pehe_raw"])
            ate_pred = _first(z, ["ate_pred", "ate_raw"])
            true_ate = _first(z, ["true_ate"])
    else:
        files = sorted(glob.glob(os.path.join(cell_dir, "*_r*.npz")))
        if not files:
            return None
        pehe, ate_pred, true_ate = [], [], []
        for f in files:
            with np.load(f, allow_pickle=True) as z:
                p = _first(z, ["pehe_raw", "pehe_dopfn", "pehe"])
                ap = _first(z, ["ate_pred", "ate_raw"])
                ta = _first(z, ["true_ate"])
                if p is not None: pehe.append(float(p[0]))
                if ap is not None and ta is not None:
                    ate_pred.append(float(ap[0])); true_ate.append(float(ta[0]))
        pehe = np.asarray(pehe)
        ate_pred = np.asarray(ate_pred); true_ate = np.asarray(true_ate)
    if pehe is None or not len(pehe):
        return None
    l1 = (np.mean(np.abs(ate_pred - true_ate))
          if ate_pred is not None and len(ate_pred) else float("nan"))
    return float(np.mean(pehe)), float(l1), int(len(pehe))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", required=True, help="A/B output root (holds <variant>/<case>/).")
    a = ap.parse_args()

    variants = sorted(d for d in os.listdir(a.ab)
                      if os.path.isdir(os.path.join(a.ab, d)))
    cases = sorted({c for v in variants
                    for c in os.listdir(os.path.join(a.ab, v))
                    if os.path.isdir(os.path.join(a.ab, v, c))})

    for metric, i in (("PEHE", 0), ("L1-ATE", 1)):
        print(f"\n══ {metric} ══")
        hdr = "variant".ljust(20) + "".join(c[:16].rjust(18) for c in cases)
        print(hdr); print("-" * len(hdr))
        for v in variants:
            line = v.ljust(20)
            for c in cases:
                r = _cell(os.path.join(a.ab, v, c))
                line += ("—".rjust(18) if r is None else f"{r[i]:18.3f}")
            print(line)
    print()


if __name__ == "__main__":
    main()
