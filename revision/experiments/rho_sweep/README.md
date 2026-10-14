# rho sweep

The eight main-table models, unchanged checkpoints, on one simple SCM where only the cross-world noise
correlation rho = corr(eps0, eps1) varies: rho in {-0.5, 0, 0.5, 0.9, 1.0}, 100 datasets per rho,
N = 1000 context units and Q = 100 query units each (`gen_rho.py` documents the SCM and the format).

| file | what |
|---|---|
| `gen_rho.py` | datasets, written with `E2_in_prior/common.write_world` (ComplexMech npz format + truths) |
| `1_gen.sh` | step 1, CPU job: runs `gen_rho.py` into `$SCRATCH/rho_sweep/data` |
| `2_eval.sh <model>` | step 2, one GPU job per model: the paper's harness, same command and settings as `E2_in_prior/eval_e2.sbatch`, `DENSITY_DUMP=1`, the five rho cells in turn; dumps kept in `$SCRATCH/rho_sweep/dumps/<model>/rho<+r.rr>` |
| `rho_score.py` | per (model, rho): hist 95% interval exactly as `E2_in_prior/score_shard.py`, coverage against the realized delta, length, IS_0.05, sqrt(PEHE) of the raw point estimate (`point_raw_em.cate_for_file`) against mu1 - mu0 |
| `3_collect.sh` | step 3, CPU job: `rho_score.py` -> `revision/results/d262t1/rho_sweep.{csv,md}`, git add/commit/pull --rebase/push |

Models (eight rows from six jobs; the UWYK harnesses run `--anc-mode v3ab_only`, so No-Anc = tag `noanc`
and Anc = tag `v3a` come from the same pass, as in ComplexMech):

| row | job | harness | checkpoint / env |
|---|---|---|---|
| Do-PFN 1D | `dopfn1d` | `benchmarks/eval_scm_case_studies/eval_native_dopfn.py` | released Do-PFN (`$KIT/external/dopfn`), its own inference config |
| Do-PFN 2D | `dopfn2d` | same | `DOPFN_CKPT=Required_checkpoints/dopfn_repro_joint2d_step262144.pt DOPFN_SOFTMAX_TEMP=0` |
| UWYK 1D No-Anc / Anc | `uwyk1d` | `realcause_eval/Table1/uwyk1d.py` | UWYK `best_model.pt` + `best_model_config.yaml`, `--t-encoding target --eval-max-context 1000 --eval-context-seed 43` |
| UWYK 2D No-Anc / Anc | `graph2d` | `realcause_eval/Table1/graph2d.py` | `graph2d_step_50000.pt`, `--density-primary-mode noanc --eval-max-context 1000 --eval-context-seed 43` |
| CausalPFN-C 1D | `cpfn1d` | `realcause_eval/Table1/cpfn1d.py` | `cpfn1d_j1024_headrand_step_50000.pt`, `--std-mode pooled` (as the paper's ComplexMech runs) |
| CausalPFN-C 2D | `cpfn2d` | `realcause_eval/Table1/cpfn2d.py` | `cpfn2d_j32_eta0_y01_step50000.pt`, `--std-mode pooled` |

The Anc matrix v3a (T->Y, every X->T, every X->Y = +1, rest 0) is the true ancestral relation of this SCM
(X1..X3 -> Y directly, X4 and X5 through T), so no graph needs to be stored with the data.

## Submit, in order (on killarney, after `git push` here)

```bash
cd /scratch/furkanbd/rpfn_bench_kit && git -C R-PFN pull --rebase
bash R-PFN/revision/experiments/rho_sweep/1_gen.sh                  # wait for it to finish (< 1 min of work)
for m in dopfn1d dopfn2d uwyk1d graph2d cpfn1d cpfn2d; do
  bash R-PFN/revision/experiments/rho_sweep/2_eval.sh $m             # 6 GPU jobs, one per model
done
# after all six have finished:
bash R-PFN/revision/experiments/rho_sweep/3_collect.sh               # scores, commits and pushes
```

Logs: `$KIT/logs_rho_sweep/`. A finished rho cell (100 dumps) is skipped, so resubmitting a model resumes it.
