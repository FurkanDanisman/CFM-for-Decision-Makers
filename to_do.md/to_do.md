1) Re-do all the results of UWYK-2DMALC with B=500 for table 3 if the results do not change then just go with it for the context sweep as well. 

2) Re-do the density calculations with B=500 for ACIC and IHDP

3) Check the results with J=10, and if there is no improvement over Do-PFN for density calculations, it is most likely due to dimensional problems, so we try on a controlled setting with lower dimension. (Most likely I need to re-do some of the dimensional sweep results either way) 

4) We will need to update the result for context sweep + Table 3 + dimesion + rho testing for Do-PFN-2DMALC

5) Complete the UWYK-NO ANC PSID blanced results 

Tasks

1) Re-do all the results of UWYK-2DMALC with B=500 for table 3

Submitted.

2) Do all the results of Table 3 with B=500 for the 200k checkpoint.

3) Do the IHDP density calc with B=500 for the 200k checkpoint and do with UYWK-Full ANC.

4) Do the density calculation for ACIC for UWYK No Anc., UWYK-2DMALC, UWYK - FULL ANC, Do-PFN, 200k checkpoint.

5) For the dataset used in context sweep, fix N=500 and d=5, and do density calc evaulation for Do-PFN and 200k checkpoint.

6) Do the context sweep results for our 200k checkpoint.



Ready-to-submit list

Prereq every time:
export DEPLOY_ROOT=/scratch/furkanbd/rpfn_bench_kit
cd $DEPLOY_ROOT/R-PFN && git pull origin main && cd $DEPLOY_ROOT

1) IHDP density L2 for 200K checkpoint (task 3a) — ~2h, 100 tasks

MALC_B=500 METHODS=ours_dopfn_bb \
CHECKPOINT_DOPFN_BB=$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt \
OUT=$DEPLOY_ROOT/l2_ihdp/out_dopfn_bb_j10_B500_step200000 \
sbatch $DEPLOY_ROOT/R-PFN/benchmarks/cluster/submit_ihdp_l2.sbatch

2) IHDP density L2 with UWYK-Full-Anc added (task 3b) — ~2h

MALC_B=500 METHODS=uwyk_anc \
OUT=$DEPLOY_ROOT/l2_ihdp/out_uwyk_anc_B500 \
sbatch $DEPLOY_ROOT/R-PFN/benchmarks/cluster/submit_ihdp_l2.sbatch
Later combined with existing shards for the 5-way IHDP summary.

3) Linear-Gaussian synthetic (task 5) — ~30min, 100 seeds

MALC_B=500 METHODS=dopfn,ours_dopfn_bb \
CHECKPOINT_DOPFN_BB=$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt \
OUT=$DEPLOY_ROOT/l2_syn/out_N500_d5_B500 \
sbatch $DEPLOY_ROOT/R-PFN/benchmarks/cluster/submit_syn_l2.sbatch

4) ACIC density L2 (task 4) — ~4-8h, 10 tasks

MALC_B=500 METHODS=ours_fn50,uwyk_noanc,uwyk_anc,dopfn,ours_dopfn_bb \
CHECKPOINT_DOPFN_BB=$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt \
OUT=$DEPLOY_ROOT/l2_acic/out_all_B500 \
sbatch $DEPLOY_ROOT/R-PFN/benchmarks/cluster/submit_acic_l2.sbatch

5) Table 3 fn=50 rerun at B=500 (task 1, already running as results_ours_only_B500)

MALC_B=500 BACKBONE=ipfn \
OUTDIR=$FN50_OUT \
CHECKPOINT=$DEPLOY_ROOT/R-PFN/checkpoints/step_50000_final.pt \
sbatch $DEPLOY_ROOT/R-PFN/benchmarks/cluster/submit_ours_only.sbatch

Submitted 

6) Table 3 for 200K DoPFN-bb (task 2) — ~4-8h, 500 tasks

MALC_B=500 BACKBONE=dopfn_bb \
OUTDIR=$DOPFNBB_OUT \
CHECKPOINT=$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt \
sbatch $DEPLOY_ROOT/R-PFN/benchmarks/cluster/submit_ours_only.sbatch

Submitted

7) Context sweep for 200K (task 6) — depends on sweep size (We will not do the final N)

export DEPLOY_ROOT=/scratch/furkanbd/rpfn_bench_kit
cd $DEPLOY_ROOT/R-PFN && git pull origin main && cd $DEPLOY_ROOT

# fn=50 sweep
sbatch $DEPLOY_ROOT/R-PFN/benchmarks/context_sweep/submit_sweep.sbatch

# DoPFN-bb 200K sweep
BACKBONE=dopfn_bb \
CHECKPOINT=$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt \
OUTDIR=$DEPLOY_ROOT/results_sweep_dopfn_bb_200K \
sbatch $DEPLOY_ROOT/R-PFN/benchmarks/context_sweep/submit_sweep.sbatch


Both arrays running. Progress script

export DEPLOY_ROOT=/scratch/furkanbd/rpfn_bench_kit
FN50_OUT=$DEPLOY_ROOT/results_ours_only_B500_v2
DOPFNBB_OUT=$DEPLOY_ROOT/results_ours_only_dopfn_bb_200K_B500_v2

# Per-dataset shard counts (target: IHDP=100, ACIC=10, CPS=100, PSID=100, PSIDbal=100 → 410 total)
for label in "fn=50 ($FN50_OUT)" "DoPFN-bb 200K ($DOPFNBB_OUT)"; do
    dir=$(echo "$label" | sed 's/.*(\(.*\)).*/\1/')
    echo
    echo "=== $label ==="
    if [ ! -d "$dir" ]; then echo "  (dir does not exist yet)"; continue; fi
    total=0
    for ds in IHDP:100 ACIC:10 CPS:100 PSID:100 PSIDbal:100; do
        name="${ds%:*}"; expect="${ds#*:}"
        n=$(ls $dir/${name}_r*.npz 2>/dev/null | wc -l)
        total=$((total + n))
        printf "  %-8s %3d / %3d\n" "$name" "$n" "$expect"
    done
    printf "  %-8s %3d / %3d\n" "TOTAL" "$total" "410"
done


Results

export DEPLOY_ROOT=/scratch/furkanbd/rpfn_bench_kit
source $DEPLOY_ROOT/venv/bin/activate

echo "=== fn=50 ==="
python $DEPLOY_ROOT/R-PFN/benchmarks/summary_table3_two_row.py \
    --results $DEPLOY_ROOT/results_ours_only_B500_v2

echo "=== DoPFN-bb 200K ==="
python $DEPLOY_ROOT/R-PFN/benchmarks/summary_table3_two_row.py \
    --results $DEPLOY_ROOT/results_ours_only_dopfn_bb_200K_B500_v2
