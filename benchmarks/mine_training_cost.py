#!/usr/bin/env python3
"""Recover throughput, GPU-hours and peak memory for tab:train_cost from
training logs and sacct records that already exist.

    # throughput, from the logs
    python R-PFN/benchmarks/mine_training_cost.py --logs logs_dopfn logs_causalpfn2d_l40s ...

    # GPU-hours + peak memory, from Slurm's own accounting
    sacct -S 2026-06-01 -X --units=G -P -o \
      JobID,JobName%40,State,Elapsed,AllocTRES%80,MaxRSS,TresUsageInTot%90 > sacct.psv
    python R-PFN/benchmarks/mine_training_cost.py --sacct sacct.psv

Three throughput sources, which do NOT report the same unit:

  Do-PFN    training_dopfn_repro/train.py:417-422 prints
            "<step> <loss> <lr> <dt> <skipped>" where dt = (t - t0)/log_every,
            i.e. ALREADY seconds per optimizer step. Used as-is.

  CausalPFN rpfn_patches/causalpfn_step_ckpt/trainer.py:338 builds a tqdm bar
            over range(epochs * updates * num_agg * world_size), so the bar
            counts MICRO-batches. tqdm's rate is therefore micro-batches/s and
            one optimizer step costs num_agg*world_size of them. Pass --accum
            (default 16, from causalpfn2d_50k.yaml) to convert. Reporting the
            raw tqdm rate as s/step would understate cost by that factor.

  fallback  Elapsed / steps from sacct, which is the only uniform measure and
            the only one available for UWYK (its trainer is external code).

GPU-hours = Elapsed x (GPUs in AllocTRES). Peak memory is NOT instrumented
anywhere in this codebase -- no max_memory_allocated call exists -- so it can
only come from Slurm if the cluster records gres/gpumem in TresUsageInTot.
MaxRSS is HOST memory and is reported separately; it is not the GPU figure the
table asks for.
"""
import argparse, os, re, sys
from collections import defaultdict

# Do-PFN: step, loss(4dp), lr(sci), s/step(3dp), skipped
RE_DOPFN = re.compile(
    r'^\s*(\d+)\s+(-?\d+\.\d{4})\s+(\d+\.\d{3}e[+-]\d+)\s+(\d+\.\d{3})\s+(\d+)\s*$')
# tqdm: "Train Batches:  42%|####  | 1234/5000 [10:11<20:22,  2.02it/s]"  or "s/it"
RE_TQDM = re.compile(r'(\d+(?:\.\d+)?)\s*(it/s|s/it)\]')
RE_JOBID = re.compile(r'(\d{6,})')

def _stats(xs):
    if not xs: return None
    n=len(xs); m=sum(xs)/n; ss=sorted(xs)
    med = ss[n//2] if n%2 else 0.5*(ss[n//2-1]+ss[n//2])
    sd = 0.0 if n<2 else (sum((x-m)**2 for x in xs)/(n-1))**0.5
    return m, sd, n, med

def _hms(s):
    """Slurm Elapsed -> hours. Accepts D-HH:MM:SS, HH:MM:SS, MM:SS."""
    d = 0
    if '-' in s:
        ds, s = s.split('-', 1); d = int(ds)
    parts = [int(x) for x in s.split(':')]
    while len(parts) < 3: parts.insert(0, 0)
    h, m, sec = parts
    return d*24 + h + m/60 + sec/3600

def mine_logs(dirs, accum, verbose):
    files = []
    for root in dirs:
        if os.path.isfile(root): files.append(root); continue
        for dp,_,fns in os.walk(root):
            files += [os.path.join(dp,f) for f in fns if f.endswith(('.out','.err','.log'))]
    dopfn, tqdm_rate = defaultdict(list), defaultdict(list)
    for path in files:
        tag = os.path.basename(os.path.dirname(path)) or path
        try: txt = open(path, errors='replace').read()
        except OSError: continue
        n_d = n_t = 0
        for line in txt.splitlines():
            m = RE_DOPFN.match(line)
            if m:
                dopfn[tag].append(float(m.group(4))); n_d += 1; continue
            t = RE_TQDM.search(line)
            if t:
                v = float(t.group(1))
                # normalise to seconds per MICRO-batch, then to per step
                spb = v if t.group(2) == 's/it' else 1.0/v
                tqdm_rate[tag].append(spb * accum); n_t += 1
        if verbose and (n_d or n_t):
            print(f'  {path:60s} dopfn_rows={n_d:5d} tqdm_rows={n_t:6d}', file=sys.stderr)
    print('\n=== throughput from logs ===')
    print(f'{"log dir":30s} {"s/step (Do-PFN column)":>30s} {"s/step (tqdm x accum)":>30s}')
    for tag in sorted(set(dopfn)|set(tqdm_rate)):
        f = lambda s: '---' if s is None else f'{s[0]:.4g}±{s[1]:.3g} med={s[3]:.4g} n={s[2]}'
        print(f'{tag:30s} {f(_stats(dopfn.get(tag))):>30s} {f(_stats(tqdm_rate.get(tag))):>30s}')
    if tqdm_rate:
        print(f'  (tqdm converted with accum x world = {accum}; pass --accum to change)')

def mine_sacct(path, steps_map):
    rows = [l.rstrip('\n') for l in open(path) if l.strip()]
    if not rows: sys.exit('empty sacct file')
    hdr = rows[0].split('|')
    idx = {h.strip(): i for i, h in enumerate(hdr)}
    need = ['JobID','JobName','State','Elapsed','AllocTRES']
    miss = [n for n in need if n not in idx]
    if miss: sys.exit(f'sacct file missing columns {miss}; header was {hdr}')
    print('\n=== GPU-hours / memory from sacct ===')
    print(f'{"JobID":>12s} {"JobName":34s} {"State":11s} {"Elapsed":>11s} '
          f'{"nGPU":>4s} {"GPU-h":>8s} {"MaxRSS":>9s} {"gpumem (if recorded)":>22s}')
    tot = defaultdict(float)
    for line in rows[1:]:
        c = line.split('|')
        if len(c) < len(hdr): continue
        jid, name, st, el = (c[idx['JobID']], c[idx['JobName']],
                             c[idx['State']], c[idx['Elapsed']])
        tres = c[idx['AllocTRES']]
        g = re.search(r'gres/gpu(?::\w+)?=(\d+)', tres)
        ngpu = int(g.group(1)) if g else 0
        try: hrs = _hms(el)
        except Exception: continue
        rss = c[idx['MaxRSS']] if 'MaxRSS' in idx else ''
        usage = c[idx['TresUsageInTot']] if 'TresUsageInTot' in idx else ''
        gmem = ''
        mm = re.search(r'gres/gpumem[^=]*=(\S+?)(?:,|$)', usage)
        if mm: gmem = mm.group(1)
        print(f'{jid:>12s} {name[:34]:34s} {st[:11]:11s} {el:>11s} {ngpu:>4d} '
              f'{hrs*max(ngpu,1):>8.2f} {rss:>9s} {gmem or "not recorded":>22s}')
        tot[name] += hrs * max(ngpu, 1)
    print('\n  GPU-hours summed per job name:')
    for k in sorted(tot, key=lambda k: -tot[k]):
        line = f'    {k[:46]:46s} {tot[k]:8.2f} GPU-h'
        if k in steps_map and steps_map[k]:
            line += f'   -> {tot[k]*3600/steps_map[k]:.4g} s/step at {steps_map[k]:,} steps'
        print(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', nargs='*', default=[])
    ap.add_argument('--sacct', default=None, help='sacct -P output (pipe-separated)')
    ap.add_argument('--accum', type=int, default=16,
                    help='num_agg x world_size, to turn tqdm micro-batches into steps')
    ap.add_argument('--steps', default='',
                    help='name=steps,name=steps to derive s/step from Elapsed')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    if not a.logs and not a.sacct:
        sys.exit('give --logs and/or --sacct')
    steps_map = {}
    for kv in filter(None, a.steps.split(',')):
        k, v = kv.split('='); steps_map[k] = int(v)
    if a.logs: mine_logs(a.logs, a.accum, a.verbose)
    if a.sacct: mine_sacct(a.sacct, steps_map)

if __name__ == '__main__':
    main()
