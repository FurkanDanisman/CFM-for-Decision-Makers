#!/bin/bash
# In-prior PREDICTED rho of the three 2D heads: each head on 1000 datasets of its own training prior
# (E2 new-world data, shards 0..9), the paper's harness and settings (as E2_in_prior/eval_e2.sbatch);
# per shard, every query's predicted rho is computed from the dumped joint (pred_rho_shard.py) and the
# dumps are deleted. ONE GPU job, the three models in turn; a shard with a result file is skipped.
#   cd $KIT && git -C R-PFN pull --rebase && bash R-PFN/revision/experiments/rho_pred/1_inprior_pred_rho.sh
# The same file is the job body: sbatch --wrap runs "bash <this file> run" (never `source`).
set -uo pipefail
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
SELF="$KIT/R-PFN/revision/experiments/rho_pred/1_inprior_pred_rho.sh"

if [ "${1:-}" != run ]; then
  cd "$KIT" && mkdir -p logs_rho_pred
  sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --time="${RP_TIME:-3:00:00}" --cpus-per-task=4 --mem=32G \
    --job-name=rho-pred-prior -o logs_rho_pred/inprior_%j.out --wrap "KIT=$KIT bash $SELF run"
  exit $?
fi

# ---- job body ----
set -e
REPO="$KIT/R-PFN"; CK="$REPO/Required_checkpoints"
DATA="${E2_DATA:-$SCRATCH/e2/new/data}"; RES="${RP_RESULTS:-$SCRATCH/rho_pred/inprior}"
export CAUSALPFN="$KIT/external/causalpfn" DOPFN_ROOT="$KIT/external/dopfn"
UWYK="$KIT/external/uwyk_reproduce"
cd "$KIT"; source venv/bin/activate
export DATASET=CMECH_n5 DENSITY_DUMP=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi --query-gpu=name --format=csv,noheader || true

#        model                prior dir
for pair in "dopfn2d:dopfn" "graph2d:uwyk_2d" "cpfn2d:cpfn_2d"; do
  MODEL=${pair%%:*}; PRIOR=${pair##*:}
  mkdir -p "$RES/$MODEL"
  for SHARD in $(seq 0 9); do
    [ -f "$RES/$MODEL/shard$SHARD.npz" ] && { echo "skip $MODEL shard $SHARD"; continue; }
    export UWYK_FIG34_DATA="$DATA/$PRIOR/shard$SHARD"
    export OUT="$RES/$MODEL/dumps_shard$SHARD"
    rm -rf "$OUT"; mkdir -p "$OUT"
    echo "[$(date '+%F %T')] model=$MODEL data=$UWYK_FIG34_DATA"
    case "$MODEL" in
      dopfn2d)                                       # the paper's Do-PFN 2D: 262k steps, temperature off
        DOPFN_CKPT="$CK/dopfn_repro_joint2d_step262144.pt" DOPFN_SOFTMAX_TEMP=0 \
          python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset "$DATASET" ;;
      graph2d)
        python -u "$REPO/realcause_eval/Table1/graph2d.py" --dataset "$DATASET" --outdir "$OUT" \
          --density-dump 1 --ckpt "$CK/graph2d_step_50000.pt" \
          --repo "$REPO" --causalpfn "$CAUSALPFN" --uwyk "$UWYK" \
          --anc-mode v3ab_only --density-primary-mode noanc --eval-max-context 1000 --eval-context-seed 43 ;;
      cpfn2d)
        python -u "$REPO/realcause_eval/Table1/cpfn2d.py" --dataset "$DATASET" --outdir "$OUT" \
          --density-dump 1 --ckpt "$CK/cpfn2d_j32_eta0_y01_step50000.pt" \
          --repo "$REPO" --causalpfn "$CAUSALPFN" --std-mode pooled ;;
    esac
    python "$REPO/revision/experiments/rho_pred/pred_rho_shard.py" "$OUT" "$RES/$MODEL/shard$SHARD.npz"
    rm -rf "$OUT"
  done
done
python "$REPO/revision/experiments/rho_pred/summarize_inprior.py" "$RES" "$REPO/revision/results/d262t1/rho_pred_inprior"
cd "$REPO" && git add revision/results/d262t1/rho_pred_inprior.* \
  && git commit -q -m "d262t1: in-prior predicted rho of the three 2D heads" \
  && git pull -q --rebase && git push -q && echo pushed
