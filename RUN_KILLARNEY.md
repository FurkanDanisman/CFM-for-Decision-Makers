# FOR_LUKE.md, adapted for Killarney / lukez

Furkan's `FOR_LUKE.md` assumes his Trillium deploy payload (`/scratch/furkanbd/rpfn_bench_kit`,
built by `benchmarks/cluster/deploy_local.sh`). This is the same four tasks, set up from
public sources on Killarney under `lukez`.

## 0a. EVERY new shell (and every new node you land on)

`DEPLOY_ROOT` is not persistent. Without it, every `$DEPLOY_ROOT/R-PFN/...` below
expands to `/R-PFN/...` and you get `sbatch: error: Unable to open file`. Your shell
expands that path *before* sbatch runs, so this cannot be fixed from inside the
`.sbatch` files — and they must not hardcode it, since Furkan's deploy root differs.

```bash
export DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit          # /scratch/lukez/rpfn_bench_kit
cd $DEPLOY_ROOT                                     # logs_*/ are relative to the submit dir
```

Make it permanent: `echo 'export DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit' >> ~/.bashrc`.
Sanity check: `ls $DEPLOY_ROOT/R-PFN/benchmarks/cluster/submit_ihdp_l2.sbatch`.

Inside a job it resolves two ways regardless — sbatch's default `--export=ALL`
propagates it, and each submit script falls back to `${DEPLOY_ROOT:-$PWD}`.

## 0b. One-time setup — ALREADY DONE (2026-08-12), skip it

```bash
bash /project/6105522/lukez/CFM-for-Decision-Makers/benchmarks/cluster/setup_killarney.sh
DEPLOY_ROOT=$DEPLOY_ROOT bash /project/6105522/lukez/CFM-for-Decision-Makers/benchmarks/cluster/smoke_test.sh
```

`$DEPLOY_ROOT` is built and the smoke test passes all four tasks. `R-PFN` is a symlink
**you own**, in your own scratch, pointing at your own checkout — the name is just a
leftover from the old repo name, not a path of Furkan's.

This builds:

```
$SCRATCH/rpfn_bench_kit/
├── R-PFN                          -> /project/6105522/lukez/CFM-for-Decision-Makers
├── external/dopfn                    github.com/jr2021/Do-PFN     (+ 2 patches, below)
├── external/causalpfn                github.com/vdblm/CausalPFN   (ships the IHDP NPZs)
├── external/uwyk                  -> <repo>/g4cfm  (+ `git lfs pull` for its checkpoint)
├── checkpoints_dopfn_backbone_j10/step_200000.pt
│                                  -> <repo>/checkpoints_shared/dopfn_bb_step_200000.pt
└── venv/
```

`external/` really is just the two public clones. Two mechanical patches are needed and
`setup_killarney.sh` applies them (this is what "patched Do-PFN" in `deploy_local.sh` meant):

- `model/layer.py` imports `Optional` from `torch.nn.modules.transformer`, which only ever
  worked because torch leaked its own `typing` import there. Gone in modern torch.
- `base.py` calls `check_array(..., force_all_finite=)`; sklearn renamed that to
  `ensure_all_finite` in 1.6 and dropped the old name in 1.8.

**Always `cd $DEPLOY_ROOT` before submitting** — `#SBATCH --output=logs_*/…` and the
scripts' `mkdir -p` are relative to the submit directory, and `/project` is the wrong
place to write job output.

## Killarney specifics

| | |
|---|---|
| Account | `--account=aip-rgrosse` (your only Slurm association; `def-rgrosse` is not on this cluster) |
| Partition | **don't set one** — Slurm routes by `--time` (`≤3h`→`l40s_b1`, `≤12h`→`l40s_b2`, …) |
| GPU | **don't request one.** Verified with `sbatch --test-only`: CPU-only jobs are accepted even though every partition is `gpubase_*` |

Everything here runs on CPU — UWYK's loader hardcodes `device='cpu'`, `DoPFNRegressor`
defaults to `cpu`, and our checkpoints load with `map_location='cpu'`. Nothing calls
`.cuda()`. So an H100 buys **zero** speedup and costs more: `TRESBillingWeights` on
`gpubase_h100_*` is `cpu=1016.67, gres/gpu=12200` vs `cpu=321.88, gres/gpu=10300` on
`l40s`, and h100 has 8–10 nodes against l40s's 168 — in a `--test-only` probe the same job
was scheduled ~1.5 h later on h100. If you want the GPU tier anyway, add
`--partition=gpubase_h100_b2 --gres=gpu:h100:1`.

Every job is **resubmit-safe**: each script skips a shard whose `.npz` already exists, so
if an array task hits the wall clock just submit it again and it resumes.

---

## TASK 0 — IHDP baseline shards (do this first)

Not in Furkan's file, but Tasks 2 and 3 both pass
`--dopfn-shards-glob "$DEPLOY_ROOT/l2_ihdp/out.r*.npz"` to `summary_ihdp.py`, and that
argument is `required`. Those shards are from an earlier run you don't have. Either get
them from Furkan, or generate them:

```bash
cd $DEPLOY_ROOT
sbatch --account=aip-rgrosse \
    $DEPLOY_ROOT/R-PFN/benchmarks/cluster/submit_ihdp_l2.sbatch
# defaults: METHODS=ours_fn50,ours_fn10,uwyk_noanc,dopfn   OUT=$DEPLOY_ROOT/l2_ihdp/out
ls $DEPLOY_ROOT/l2_ihdp/out.r*.npz | wc -l   # target 100
```

## TASK 1 — context simulation sweep, 4 outputs for both models

The in-file `#SBATCH` block is Trillium-shaped (`--cpus-per-task=32`, no `--mem`). The
script's own header says to use 8 CPUs + 64 G on Killarney; overriding on the command line
does that without editing the file. `--time` is raised to just under the 12 h `l40s_b2` cap
because 8 workers is a quarter of Trillium's 32.

```bash
cd $DEPLOY_ROOT

# fn=50 sweep
sbatch --account=aip-rgrosse --cpus-per-task=8 --mem=64G --time=11:59:00 \
    $DEPLOY_ROOT/R-PFN/benchmarks/context_sweep/submit_sweep.sbatch

# DoPFN-bb 200K sweep
BACKBONE=dopfn_bb \
CHECKPOINT=$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt \
OUTDIR=$DEPLOY_ROOT/results_sweep_dopfn_bb_200K \
sbatch --account=aip-rgrosse --cpus-per-task=8 --mem=64G --time=11:59:00 \
    $DEPLOY_ROOT/R-PFN/benchmarks/context_sweep/submit_sweep.sbatch
```

```bash
# ── WHEN DONE (both sweep jobs) ── target ~5000 shards per dir
ls $DEPLOY_ROOT/results_sweep/*.npz 2>/dev/null | wc -l
ls $DEPLOY_ROOT/results_sweep_dopfn_bb_200K/*.npz 2>/dev/null | wc -l

source $DEPLOY_ROOT/venv/bin/activate
python $DEPLOY_ROOT/R-PFN/benchmarks/context_sweep/aggregate.py \
    --results $DEPLOY_ROOT/results_sweep
python $DEPLOY_ROOT/R-PFN/benchmarks/context_sweep/aggregate.py \
    --results $DEPLOY_ROOT/results_sweep_dopfn_bb_200K
```

## TASK 2 — UWYK variant for the density-estimation table

```bash
cd $DEPLOY_ROOT
MALC_B=500 METHODS=uwyk_anc \
OUT=$DEPLOY_ROOT/l2_ihdp/out_uwyk_anc_B500 \
sbatch --account=aip-rgrosse \
    $DEPLOY_ROOT/R-PFN/benchmarks/cluster/submit_ihdp_l2.sbatch
```

```bash
# ── WHEN DONE ──
ls $DEPLOY_ROOT/l2_ihdp/out_uwyk_anc_B500.r*.npz | wc -l   # target 100
source $DEPLOY_ROOT/venv/bin/activate
python $DEPLOY_ROOT/R-PFN/benchmarks/l2_ihdp/summary_ihdp.py \
    --ours-shards-glob  "$DEPLOY_ROOT/l2_ihdp/out_uwyk_anc_B500.r*.npz" \
    --dopfn-shards-glob "$DEPLOY_ROOT/l2_ihdp/out.r*.npz" \
    --ours-key   uwyk_anc \
    --ours-label "UWYK Full-Anc"
```

## TASK 3 — IHDP density L2 for the Do-PFN-2DMALC checkpoint

```bash
cd $DEPLOY_ROOT
MALC_B=500 METHODS=ours_dopfn_bb \
CHECKPOINT_DOPFN_BB=$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt \
OUT=$DEPLOY_ROOT/l2_ihdp/out_dopfn_bb_j10_B500_step200000 \
sbatch --account=aip-rgrosse \
    $DEPLOY_ROOT/R-PFN/benchmarks/cluster/submit_ihdp_l2.sbatch
```

```bash
# ── WHEN DONE ──
ls $DEPLOY_ROOT/l2_ihdp/out_dopfn_bb_j10_B500_step200000.r*.npz | wc -l   # target 100
source $DEPLOY_ROOT/venv/bin/activate
python $DEPLOY_ROOT/R-PFN/benchmarks/l2_ihdp/summary_ihdp.py \
    --ours-shards-glob  "$DEPLOY_ROOT/l2_ihdp/out_dopfn_bb_j10_B500_step200000.r*.npz" \
    --dopfn-shards-glob "$DEPLOY_ROOT/l2_ihdp/out.r*.npz" \
    --ours-key   ours_dopfn_bb \
    --ours-label "Ours(DoPFN-bb 200K)"
```

## TASK 4 — Linear-Gaussian synthetic (task 5), 100 seeds

`--time=03:00:00` in the file is exactly the `l40s_b1` cap; bumped so a slow seed doesn't
lose the task.

```bash
cd $DEPLOY_ROOT
MALC_B=500 METHODS=dopfn,ours_dopfn_bb \
CHECKPOINT_DOPFN_BB=$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt \
OUT=$DEPLOY_ROOT/l2_syn/out_N500_d5_B500 \
sbatch --account=aip-rgrosse --time=05:00:00 \
    $DEPLOY_ROOT/R-PFN/benchmarks/cluster/submit_syn_l2.sbatch
```

```bash
# ── WHEN DONE ──
ls $DEPLOY_ROOT/l2_syn/out_N500_d5_B500.s*.npz | wc -l   # target 100
source $DEPLOY_ROOT/venv/bin/activate
python $DEPLOY_ROOT/R-PFN/benchmarks/l2_syn/summary_syn.py \
    --shards-glob "$DEPLOY_ROOT/l2_syn/out_N500_d5_B500.s*.npz"
```

---

## Progress check

```bash
squeue --me -o "%.10i %.24j %.8T %.10M %.6D %R" | head -40

for d in $DEPLOY_ROOT/results_sweep \
         $DEPLOY_ROOT/results_sweep_dopfn_bb_200K \
         $DEPLOY_ROOT/l2_ihdp \
         $DEPLOY_ROOT/l2_acic \
         $DEPLOY_ROOT/l2_syn; do
    [ -d "$d" ] && echo "$(basename $d): $(ls $d/*.npz 2>/dev/null | wc -l) shards"
done
```

## Repo fixes these tasks required

Three bugs on code paths that had evidently never been executed. All are committed:

- `benchmarks/l2_syn/l2.py`, `benchmarks/l2_acic/l2.py` — `from l2 import l2_distance`
  resolved to the module itself (it is on `sys.path` ahead of `l2_ihdp/`, and is already in
  `sys.modules` while executing), so it raised a circular-import `ImportError` every time.
  Now loaded by file path.
- `benchmarks/l2_syn/eval_realization.py`, `benchmarks/l2_acic/eval_realization.py` —
  inserted their own directory ahead of `l2_ihdp/`, so `import eval_realization as ihdp_ev`
  re-imported *themselves* and every `ihdp_ev._run_*` raised `AttributeError`. Order swapped.
- `training_dopfn_base/dopfn_backbone_head.py` — Do-PFN and UWYK both import a top-level
  `utils`. `l2_ihdp` gets away with it by running "ours" before UWYK, but
  `context_sweep/run_one.py` loads UWYK first, so `--backbone dopfn_bb` (the Task 1 200K
  sweep) died on `cannot import name 'print_once' from 'utils'`. The loader now shadows and
  restores `utils`/`model`/`models` around the unpickle.


