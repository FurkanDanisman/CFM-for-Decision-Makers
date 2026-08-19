"""Compute density L2 tables under LEFT-edge convention at two resolutions:
J=10 (model's native) and J=100 (Y_CENTERS comparison grid).

For each case (JSON from dump_for_plot.py), reports L2 for:
  - Do-PFN
  - DoPFN-bb OLD    (MALC-1D + linear-in-prob resample)
  - DoPFN-bb LOGLIN (MALC-1D + log-linear evaluate)
  - DoPFN-bb RAW    (raw marginal, no MALC)

LEFT-edge convention: bin j's probability mass is placed at edges[j] (not center).
For a shared grid comparison, the model densities are SHIFTED LEFT by delta/2
relative to their native center-convention values; truth stays where it is.

Usage:
    python l2_table_left_edge.py --json <path>
"""
from __future__ import annotations
import argparse, json
import numpy as np


def _l2(p, q, dx):
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    return float(np.sqrt(np.sum((p - q) ** 2) * dx))


def _to_j10_probs(density_on_Y, Y, Y_BIN, edges_J10):
    """Integrate a 100-pt density over each J=10 bin → J=10 probability vector."""
    p_bin = np.zeros(len(edges_J10) - 1)
    for j in range(len(p_bin)):
        mask = (Y >= edges_J10[j]) & (Y < edges_J10[j+1])
        p_bin[j] = np.array(density_on_Y)[mask].sum() * Y_BIN
    total = p_bin.sum()
    return p_bin / total if total > 0 else p_bin


def _shift_left(d_on_Y, Y, shift):
    """Shift a density defined on Y to Y+shift (i.e., 'left-edge convention')."""
    p = np.interp(Y, Y - shift, d_on_Y, left=0.0, right=0.0)
    Y_BIN = Y[1] - Y[0]
    s = p.sum() * Y_BIN
    return p / s if s > 0 else p


def process(json_path):
    with open(json_path) as f:
        b = json.load(f)
    Y = np.array(b['Y_CENTERS']); Y_BIN = Y[1] - Y[0]
    edges_J10 = np.array(b['edges_scaled'])
    bin_w_J10 = edges_J10[1] - edges_J10[0]
    centers_J10 = 0.5 * (edges_J10[:-1] + edges_J10[1:])

    print(f'\n══════ case: {b.get("case","?")}  r={b.get("realization","?")}  '
          f'q={b.get("query","?")}  seed={b.get("seed","-")} ══════')

    for arm_tag, key_true, key_dopfn, key_bb_lgl, key_bb_old, key_raw in [
        ('Y0', 'p_y0_true', 'dopfn_p_y0', 'py_bb_p_y0', 'py_bb_old_p_y0', 'p_marg_y0_raw'),
        ('Y1', 'p_y1_true', 'dopfn_p_y1', 'py_bb_p_y1', 'py_bb_old_p_y1', 'p_marg_y1_raw'),
    ]:
        p_true = np.array(b[key_true])
        p_dopfn = np.array(b[key_dopfn])
        p_bb_lgl = np.array(b[key_bb_lgl])
        p_bb_old = np.array(b[key_bb_old]) if b.get(key_bb_old) is not None else None
        p_marg_raw = np.array(b[key_raw])
        # RAW density on Y grid = linear-in-prob interp of J=10 probs
        d_raw_native = p_marg_raw / p_marg_raw.sum() / bin_w_J10
        p_raw_100 = np.interp(Y, centers_J10, d_raw_native, left=0.0, right=0.0)
        s = p_raw_100.sum() * Y_BIN
        if s > 0: p_raw_100 /= s

        # LEFT-edge convention: shift model densities LEFT by half bin (J=10 delta / 2 = 0.10)
        # Truth stays; model shifted so its density is aligned with left-edge assignment.
        shift = -bin_w_J10 / 2  # -0.10
        p_dopfn_L    = _shift_left(p_dopfn,   Y, shift)
        p_bb_lgl_L   = _shift_left(p_bb_lgl,  Y, shift)
        p_bb_old_L   = _shift_left(p_bb_old,  Y, shift) if p_bb_old is not None else None
        p_raw_L      = _shift_left(p_raw_100, Y, shift)

        # J=100 L2 (LEFT convention)
        L2_100 = {
            'Do-PFN': _l2(p_dopfn_L, p_true, Y_BIN),
            'BB OLD (MALC + linP)': _l2(p_bb_old_L, p_true, Y_BIN) if p_bb_old_L is not None else float('nan'),
            'BB LOGLIN':            _l2(p_bb_lgl_L, p_true, Y_BIN),
            'BB RAW':               _l2(p_raw_L,   p_true, Y_BIN),
        }

        # J=10 L2 (LEFT convention): downsample all to J=10 probabilities, compare on that grid
        p_true_j10 = _to_j10_probs(p_true,      Y, Y_BIN, edges_J10)
        p_dopfn_j10 = _to_j10_probs(p_dopfn_L, Y, Y_BIN, edges_J10)
        p_bb_lgl_j10 = _to_j10_probs(p_bb_lgl_L, Y, Y_BIN, edges_J10)
        p_bb_old_j10 = _to_j10_probs(p_bb_old_L, Y, Y_BIN, edges_J10) if p_bb_old_L is not None else None
        # RAW is native at J=10 — under LEFT edge, prob is at edges_J10[:-1] which corresponds
        # to bin j. But we've already integrated. Just take p_marg_raw directly:
        p_bb_raw_j10 = p_marg_raw / p_marg_raw.sum()

        # Convert probs → density on J=10 grid for L2 comparability
        def to_dens(p): return np.asarray(p) / bin_w_J10
        L2_10 = {
            'Do-PFN':               _l2(to_dens(p_dopfn_j10), to_dens(p_true_j10), bin_w_J10),
            'BB OLD (MALC + linP)': _l2(to_dens(p_bb_old_j10), to_dens(p_true_j10), bin_w_J10) if p_bb_old_j10 is not None else float('nan'),
            'BB LOGLIN':            _l2(to_dens(p_bb_lgl_j10), to_dens(p_true_j10), bin_w_J10),
            'BB RAW':               _l2(to_dens(p_bb_raw_j10), to_dens(p_true_j10), bin_w_J10),
        }

        print(f'\n── {arm_tag}  (LEFT-edge convention) ──')
        print(f'{"method":<26s}  {"L2 J=10":>10s}  {"L2 J=100":>10s}')
        for m in ['Do-PFN', 'BB OLD (MALC + linP)', 'BB LOGLIN', 'BB RAW']:
            print(f'{m:<26s}  {L2_10[m]:>10.4f}  {L2_100[m]:>10.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True, nargs='+',
                    help='One or more JSONs from dump_for_plot.py')
    args = ap.parse_args()
    for p in args.json:
        process(p)


if __name__ == '__main__':
    main()
