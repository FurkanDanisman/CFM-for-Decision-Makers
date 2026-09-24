#!/usr/bin/env python3
"""Recover inference/MALC/OT wall times from logs that already exist.

Nothing is recomputed: every number here was printed by a job that already
ran. Point it at the Slurm log directories and it emits the app:compute
inference table.

    python R-PFN/benchmarks/mine_inference_timings.py \
        --logs logs_cs_dvar logs_fq logs_dens logs_ihdp_dens \
        --latex

Three stages are mined, each from a different print site:

  forward pass   submit_cs_dvar_density.sbatch:121 gives 'model=<harness>'
                 and ':122' gives 'N=<ctx> N_QUERY=<nq>', so the reference
                 setting is read off the log rather than assumed. The
                 per-realization line is 'r=NNN ... (NNNs)'.
                 CAREFUL: eval_density_tauC.py:489 sets t0 OUTSIDE the loop,
                 so that number is CUMULATIVE. We detect monotone runs and
                 difference them; per-r timers are used as-is.

  MALC           compute_malc_ci_cell.py:296 'wrote <...malc_ci_...npz>
                 N_q=<n> ... (<t>s)'. elapsed_s is never persisted to the npz,
                 so the log is the only record. t is WALL time with
                 n_workers=16, so s/query is wall-per-query, not CPU-seconds.

  OT             compute_ate_density_w2_cell.py:341 'rNNN ate_mean=... (<t>s)'
                 -- t0 is reset per realization, so this one is already per-r.
"""
import argparse, os, re, sys
from collections import defaultdict

# --- print sites -------------------------------------------------------------
RE_MODEL = re.compile(r'\bmodel=(\S+)')
# submit_cs_dvar_density.sbatch:121 echoes the HARNESS, not the model: joint2d
# runs through the dopfn_native harness, so 'model=' pools two checkpoints into
# one row. fq4dump_<model>_cells.log names the model in the FILENAME, which is
# the only reliable attribution we have. Harness is kept as a '~'-marked
# fallback so it is never silently mistaken for a model.
RE_FNMODEL = re.compile(r'fq4dump_(.+?)_cells\.log$')
RE_OUTROOT = re.compile(r'OUT(?:_ROOT)?=\S*?/([^/\s]+)/shift\d')
RE_CTX   = re.compile(r'\bN=(\d+)\s+N_QUERY=(\d+)')
RE_FWD   = re.compile(r'^\s*r=(\d+)\b.*?\((\d+(?:\.\d+)?)s\)\s*$')
RE_MALC  = re.compile(r'wrote\s+(\S*malc_ci_\S+\.npz)\s+N_q=(\d+).*?\((\d+(?:\.\d+)?)s\)')
RE_OT    = re.compile(r'\br(\d+)\s+ate_mean=.*?\((\d+(?:\.\d+)?)s\)')
RE_MALCB = re.compile(r'malc(\d+)|B=(\d+)')

def _undo_cumulative(times):
    """eval_density_tauC prints cumulative elapsed; per-r timers do not.

    A cumulative series is non-decreasing over its whole length. Diff it and
    keep the first value (elapsed from loop start to the first r, which IS
    that realization's time). A per-r series is returned untouched.
    """
    if len(times) < 3:
        return times, 'as-is (too short to classify)'
    if all(b >= a for a, b in zip(times, times[1:])):
        return [times[0]] + [b - a for a, b in zip(times, times[1:])], 'differenced (cumulative)'
    return times, 'as-is (per-realization timer)'

def _stats(xs):
    if not xs:
        return None
    n = len(xs); m = sum(xs) / n
    ss = sorted(xs)
    med = ss[n // 2] if n % 2 else 0.5 * (ss[n // 2 - 1] + ss[n // 2])
    sd = 0.0 if n < 2 else (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    return m, sd, n, med

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', nargs='+', required=True,
                    help='Slurm log directories (searched recursively)')
    ap.add_argument('--latex', action='store_true', help='also emit the LaTeX table body')
    ap.add_argument('--malc-b', type=int, default=None,
                    help='keep only MALC lines at this B (tag malc<B>)')
    ap.add_argument('--min-r', type=int, default=0,
                    help='drop the first K realizations per log as warm-up')
    ap.add_argument('--n', type=int, default=None, help='keep only logs with this context N')
    ap.add_argument('--q', type=int, default=None, help='keep only logs with this N_QUERY')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()

    files = []
    for root in a.logs:
        if os.path.isfile(root):
            files.append(root); continue
        for dp, _, fns in os.walk(root):
            files += [os.path.join(dp, f) for f in fns
                      if f.endswith(('.out', '.log', '.err'))]
    if not files:
        sys.exit(f'no log files under {a.logs}')

    fwd  = defaultdict(list)   # harness -> [s per dataset]
    malc = defaultdict(list)   # harness -> [(s per cell, N_q)]
    ot   = defaultdict(list)   # harness -> [s per realization]
    settings = defaultdict(set)
    unattributed = defaultdict(int)
    nonlocal_neg = [0]

    for path in files:
        try:
            txt = open(path, errors='replace').read()
        except OSError:
            continue
        harness = None
        fn = RE_FNMODEL.search(os.path.basename(path))
        if fn:
            harness = fn.group(1)                      # model, from filename
        else:
            o = RE_OUTROOT.search(txt)
            if o:
                harness = o.group(1)                   # model, from OUT_ROOT
            else:
                m = RE_MODEL.search(txt)
                if m:
                    harness = '~' + m.group(1)         # harness only
        c = RE_CTX.search(txt)
        NQ = (int(c.group(1)), int(c.group(2))) if c else (None, None)
        if c and harness:
            settings[harness].add(NQ)
        if a.n is not None and NQ[0] != a.n: continue
        if a.q is not None and NQ[1] != a.q: continue
        key = (harness, NQ)

        raw_fwd = []
        for line in txt.splitlines():
            mf = RE_FWD.match(line)
            if mf:
                raw_fwd.append((int(mf.group(1)), float(mf.group(2))))
            mm = RE_MALC.search(line)
            if mm:
                npz, nq, t = mm.group(1), int(mm.group(2)), float(mm.group(3))
                if a.malc_b is not None:
                    tag = RE_MALCB.search(os.path.basename(npz))
                    b = tag and (tag.group(1) or tag.group(2))
                    if str(b) != str(a.malc_b):
                        continue
                malc[key if harness else ('<unknown>', NQ)].append((t, nq))
            mo = RE_OT.search(line)
            if mo and 'ate_mean' in line:
                ot[key if harness else ('<unknown>', NQ)].append(float(mo.group(2)))

        if raw_fwd:
            raw_fwd.sort()
            times, how = _undo_cumulative([t for _, t in raw_fwd])
            times = times[a.min_r:]
            neg = [t for t in times if t < 0]
            if neg:
                nonlocal_neg[0] += len(neg)
                times = [t for t in times if t >= 0]
            if harness:
                fwd[key] += times
            else:
                unattributed[os.path.basename(path)] += len(times)
            if a.verbose:
                print(f'  {os.path.basename(path):34s} harness={harness or "?":16s} '
                      f'{len(times):4d} fwd  [{how}]', file=sys.stderr)

    if unattributed:
        print(f'[warn] {sum(unattributed.values())} forward-pass lines in '
              f'{len(unattributed)} log(s) had no "model=" header and were dropped',
              file=sys.stderr)

    if nonlocal_neg[0]:
        print(f'[warn] dropped {nonlocal_neg[0]} NEGATIVE per-dataset diffs '
              f'(a cumulative timer that went backwards = restarted/concatenated job)',
              file=sys.stderr)
    harnesses = sorted(set(fwd) | set(malc) | set(ot), key=lambda k: (str(k[0]), k[1]))
    print(f'\nlogs scanned: {len(files)}')
    print('  (rows are (model, (N, Q)); a ~prefix means harness-level only)')

    print(f'\n{"model / (N,Q)":34s} {"fwd s/dataset":>22s} {"MALC s/query":>18s} '
          f'{"MALC s/cell":>18s} {"OT s":>18s}')
    rows = {}
    for h in harnesses:
        f = _stats(fwd.get(h, []))
        mc = malc.get(h, [])
        mq = _stats([t / nq for t, nq in mc if nq]) if mc else None
        mt = _stats([t for t, _ in mc]) if mc else None
        o = _stats(ot.get(h, []))
        fmt = lambda s: '---' if s is None else f'{s[0]:.3g}±{s[1]:.3g} m={s[3]:.3g} n={s[2]}'
        print(f'{str(h):34s} {fmt(f):>26s} {fmt(mq):>18s} {fmt(mt):>18s} {fmt(o):>18s}')
        rows[h] = (f, mq, mt, o)

    if a.latex:
        print('\n% --- tab:inference body, mined from existing logs ---')
        for h in harnesses:
            f, mq, mt, o = rows[h]
            cell = lambda s: '---' if s is None else f'${s[0]:.3g} \\pm {s[1]:.3g}$'
            tot = None
            parts = [x[0] for x in (f, mt, o) if x]
            if parts:
                tot = sum(parts)
            print(f'{str(h):34s} & {cell(f)} & {cell(mq)} & {cell(mt)} & {cell(o)} & '
                  f'{"---" if tot is None else f"${tot:.3g}$"} \\\\')

if __name__ == '__main__':
    main()
