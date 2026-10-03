# E1 + E8 — rescoring from existing dumps

No model is run. CPU only, on killarney.

## E1: IHDP and ACIC against realized Δ
The dumps' truth on IHDP/ACIC is τ = mu1 − mu0. E1 scores the same densities
against realized Δ = y1 − y0 at the same query units.

- `build_delta_truth.py` writes `<DS>_r###.npz` with `delta` and `tau`. The query units
  are the CausalPFN loaders' test units: IHDP uses the test file in order, and ACIC uses
  `default_rng(42+idx).permutation`, split at 0.9. Before writing, the script checks
  that the rebuilt τ equals the loader's `true_cate` exactly.
- `benchmarks/coverage_by_realization.py --truth-dir` scores against `delta`. It first
  requires the dump's own truth to equal the file's `tau` on every query, and aborts
  on any mismatch.
- Both datasets have independent arm noise: corr(y0 − mu0, y1 − mu1) is −0.01 (IHDP) and
  0.01 (ACIC zymu_1), and sd(Δ − τ) = 1.42 ≈ √2 on both.

Models: the published ones (`plan.md`, Resolved decisions 1), raw and MALC
(B=1000, K=1, n_tau=4001, matching `submit_full_table.sbatch`).

| label | root | methods |
|---|---|---|
| orig | `rc_dens_uni` | dopfn_native, uwyk1d-noanc/v3a, graph2d-noanc/v3a, cpfn1d |
| joint2d | `dumps_all/dopfn_repro_joint2d/rc` | dopfn_repro_joint2d |
| eta0 | `rc_dens_eta0` | cpfn2d (= cpfn2d_j32_eta0_y01) |

## E8: tables
`e8_tables.py` reads per-realization arrays only:
- IHDP and ACIC come from E1.
- CPS, PSID and PSID_bal, the case study, and ComplexMech come from the existing
  `$SCRATCH/perreal` (`submit_full_table.sbatch`, `submit_cmech_score_one.sbatch`).

Each cell is |ĉ − 0.95| (SE) · length, where:
- ĉ is the mean over contexts of the per-context coverage;
- SE = sd / √n_contexts;
- length is the mean central-95% length.

`e8_long.csv` also carries ĉ and the length SE.

- RealCause: per dataset.
- Case study: per case, pooled, per shift (−2, 0, +2; d pooled), and per d (5…50;
  shifts pooled).
- ComplexMech: all-ρ (`cmech_dumps`, i.e. `cmech_data_v2`), per node count.

## Run
```bash
cd $KIT && git -C R-PFN pull
sbatch --account=aip-rgrosse R-PFN/revision/experiments/E1_E8_rescoring/submit_e1.sbatch
# after it finishes (login node is fine: reads small npz only)
python R-PFN/revision/experiments/E1_E8_rescoring/e8_tables.py \
    --perreal $SCRATCH/perreal --e1-perreal $SCRATCH/revision_e1/perreal \
    --out-dir $SCRATCH/revision_e1/tables
```
`e8_tables.py` lists every missing perreal file or method key on stderr. A `—` cell
means the input does not exist; it is not a zero.
