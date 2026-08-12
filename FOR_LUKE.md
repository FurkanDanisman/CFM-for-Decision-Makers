### TASK NUMBER 1 

# This one does context simulation study with 4 outputs for our two models. 

export DEPLOY_ROOT=/scratch/furkanbd/rpfn_bench_kit
cd $DEPLOY_ROOT/R-PFN && git pull origin main && cd $DEPLOY_ROOT

# fn=50 sweep
sbatch $DEPLOY_ROOT/R-PFN/benchmarks/context_sweep/submit_sweep.sbatch

# DoPFN-bb 200K sweep
BACKBONE=dopfn_bb \
CHECKPOINT=$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt \
OUTDIR=$DEPLOY_ROOT/results_sweep_dopfn_bb_200K \
sbatch $DEPLOY_ROOT/R-PFN/benchmarks/context_sweep/submit_sweep.sbatch

 # ── WHEN DONE (both sweep jobs) ──
  # Completion check (should hit ~5000 shards per dir at the end)
  ls $DEPLOY_ROOT/results_sweep/*.npz 2>/dev/null | wc -l
  ls $DEPLOY_ROOT/results_sweep_dopfn_bb_200K/*.npz 2>/dev/null | wc -l
  # Summary — one per model
  source $DEPLOY_ROOT/venv/bin/activate
  python $DEPLOY_ROOT/R-PFN/benchmarks/context_sweep/aggregate.py \
      --results $DEPLOY_ROOT/results_sweep
  python $DEPLOY_ROOT/R-PFN/benchmarks/context_sweep/aggregate.py \
      --results $DEPLOY_ROOT/results_sweep_dopfn_bb_200K

### TASK NUMBER 2 

# This one adds a variation of UWYK to density estimation table 

MALC_B=500 METHODS=uwyk_anc \
OUT=$DEPLOY_ROOT/l2_ihdp/out_uwyk_anc_B500 \
sbatch $DEPLOY_ROOT/R-PFN/benchmarks/cluster/submit_ihdp_l2.sbatch

# ── WHEN DONE ──
ls $DEPLOY_ROOT/l2_ihdp/out_uwyk_anc_B500.r*.npz | wc -l   # target 100
source $DEPLOY_ROOT/venv/bin/activate
python $DEPLOY_ROOT/R-PFN/benchmarks/l2_ihdp/summary_ihdp.py \
    --ours-shards-glob  "$DEPLOY_ROOT/l2_ihdp/out_uwyk_anc_B500.r*.npz" \
    --dopfn-shards-glob "$DEPLOY_ROOT/l2_ihdp/out.r*.npz" \
    --ours-key   uwyk_anc \
    --ours-label "UWYK Full-Anc"

### TASK NUMBER 3

# 1) IHDP density L2 for Do-PFN-2DMALC checkpoint

MALC_B=500 METHODS=ours_dopfn_bb \
CHECKPOINT_DOPFN_BB=$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt \
OUT=$DEPLOY_ROOT/l2_ihdp/out_dopfn_bb_j10_B500_step200000 \
sbatch $DEPLOY_ROOT/R-PFN/benchmarks/cluster/submit_ihdp_l2.sbatch

# ── WHEN DONE ──
ls $DEPLOY_ROOT/l2_ihdp/out_dopfn_bb_j10_B500_step200000.r*.npz | wc -l   # target 100
source $DEPLOY_ROOT/venv/bin/activate
python $DEPLOY_ROOT/R-PFN/benchmarks/l2_ihdp/summary_ihdp.py \
    --ours-shards-glob  "$DEPLOY_ROOT/l2_ihdp/out_dopfn_bb_j10_B500_step200000.r*.npz" \
    --dopfn-shards-glob "$DEPLOY_ROOT/l2_ihdp/out.r*.npz" \
    --ours-key   ours_dopfn_bb \
    --ours-label "Ours(DoPFN-bb 200K)"

### TASK NUMBER 4

# 3) Linear-Gaussian synthetic (task 5) — ~30min, 100 seeds

MALC_B=500 METHODS=dopfn,ours_dopfn_bb \
CHECKPOINT_DOPFN_BB=$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt \
OUT=$DEPLOY_ROOT/l2_syn/out_N500_d5_B500 \
sbatch $DEPLOY_ROOT/R-PFN/benchmarks/cluster/submit_syn_l2.sbatch


# ── WHEN DONE ──
ls $DEPLOY_ROOT/l2_syn/out_N500_d5_B500.s*.npz | wc -l   # target 100
source $DEPLOY_ROOT/venv/bin/activate
python $DEPLOY_ROOT/R-PFN/benchmarks/l2_syn/summary_syn.py \
    --shards-glob "$DEPLOY_ROOT/l2_syn/out_N500_d5_B500.s*.npz"



### PROGRESS CHECK (all jobs)
for jid_line in $(squeue --me -h -o "%.10i %.20j" | awk '{print $1":"$2}' | sort -u); do
    jid="${jid_line%:*}"; name="${jid_line#*:}"
    r=$(squeue --me -h -j "$jid" -t R 2>/dev/null | wc -l)
    p=$(squeue --me -h -j "$jid" -t PD 2>/dev/null | wc -l)
    printf "  %-15s %-25s  running=%3d  pending=%3d\n" "$jid" "$name" "$r" "$p"
done

# Shard counts for all output dirs
for d in $DEPLOY_ROOT/results_ours_only_B500_v2 \
         $DEPLOY_ROOT/results_ours_only_dopfn_bb_200K_B500_v2 \
         $DEPLOY_ROOT/results_sweep \
         $DEPLOY_ROOT/results_sweep_dopfn_bb_200K \
         $DEPLOY_ROOT/l2_ihdp \
         $DEPLOY_ROOT/l2_acic \
         $DEPLOY_ROOT/l2_syn; do
    [ -d "$d" ] && echo "$(basename $d): $(ls $d/*.npz 2>/dev/null | wc -l) shards"
done