### Tasks

1) Re-do all the results of Do-PFN-bb-j=10 for table 3
    - DONE
2) d=2,3,4,5,6 with linear N calculations needs to be updated with Do-PFN-bb-j=10
    - DONE
<!-- 3) rho=0,0.2,0.4,0.6,0.8,1 calculations for Marginal, CATE, and ATE density and CATE PEHE.  -->
4) d=5,.. with N=200 needs to be updated with Do-PFN-bb-j=10
    - DONE
5) N simulation needs to be updated with Do-PFN-bb-j=10
    - DONE
<!-- 6) Job + Sales needs to be updated with Do-PFN-bb-j=10
     - Done -->
7) Density Calc for Marginal, CATE, and ATE for IHDP - Also force independence for our methods
    - DONE
8) Density Calc for Marginal, CATE, and ATE for ACIC - - Also force independence for our methods
    - DONE

### Results

export DEPLOY_ROOT=/scratch/furkanbd/rpfn_bench_kit
cd $DEPLOY_ROOT/R-PFN && git pull origin main && cd $DEPLOY_ROOT
source $DEPLOY_ROOT/venv/bin/activate

echo "=== fn=50 (with log-Y row) ==="
python $DEPLOY_ROOT/R-PFN/benchmarks/summary_table3_two_row.py \
    --results       $DEPLOY_ROOT/results_ours_only_B500_v2 \
    --logy-results  $DEPLOY_ROOT/results_ours_only_fn50_logy

echo "=== DoPFN-bb 200K (with log-Y row) ==="
python $DEPLOY_ROOT/R-PFN/benchmarks/summary_table3_two_row.py \
    --results       $DEPLOY_ROOT/results_ours_only_dopfn_bb_200K_B500_v2 \
    --logy-results  $DEPLOY_ROOT/results_ours_only_dopfn_bb_200K_logy


python $DEPLOY_ROOT/R-PFN/benchmarks/context_sweep/aggregate.py \
    --results $DEPLOY_ROOT/results_sweep_dopfn_bb_200K_nomalc \
    --metric both \
    --show-n

Confirmed via aggregate_results.py:73: stderr = stats.sem(values) — UWYK reports SEM (Standard Error of the Mean) = SD/√n. We report SD directly.

Conversion: paper's ± × √n ≈ our ±

 <!-- ▎ We reproduce UWYK's Table 3 numbers for IHDP, ACIC, and PSID (balanced) using the released PreprocessingGraphConditionedPFN wrapper and their dofm_full_conditioning.py pipeline. For CPS and PSID (unbalanced), our reproduction differs from the reported values — running the paper's own pipeline verbatim on the same released benchmarks produces the numbers we report (∼22K PSID, ∼13K CPS). The PSID (balanced) match rules out data drift as the explanation, since balanced PSID is a fixed-seed subsample of the same underlying data. The discrepancy is confined to configurations that trigger hierarchical clustering (use_clustering=True); we were unable to identify the exact source from the released artifacts. -->


