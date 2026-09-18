── UWYK NoAnc vs Ours (fn=50)   at N=200, ρ=0 ──
   d    n        UWYK NoAnc (mean±SEM)      Ours (fn=50) (mean±SEM)  √PEHE ratio    stability
--------------------------------------------------------------------------------------------------
   5   15          1.488 ±    0.156            1.200 ±    0.117          1.197        0.752
  10   15          1.431 ±    0.073            1.326 ±    0.076          1.120        1.051
  20   15          1.347 ±    0.059            1.331 ±    0.062          1.008        1.046
  30   15          1.418 ±    0.043            1.393 ±    0.050          1.026        1.167
  50   15          1.450 ±    0.034            1.463 ±    0.034          0.990        0.999
[save] /scratch/furkanbd/rpfn_bench_kit/d_scaling_linear/out_all.png

── Do-PFN vs Ours-DoPFN-bb (150k)   at N=200, ρ=0 ──
   d    n            Do-PFN (mean±SEM) Ours-DoPFN-bb (200K) (mean±SEM)  √PEHE ratio    stability
--------------------------------------------------------------------------------------------------
   5   15          1.375 ±    0.153            0.620 ±    0.056          2.608        0.367
  10   15          1.435 ±    0.075            1.147 ±    0.067          1.272        0.892
  20   15          1.331 ±    0.059            1.227 ±    0.052          1.052        0.870
  30   15          1.418 ±    0.042            1.370 ±    0.044          1.031        1.034
  50   15          1.450 ±    0.034            1.442 ±    0.035          1.004        1.054
[save] /scratch/furkanbd/rpfn_bench_kit/d_scaling_linear_dopfnbb_j10s150k.png_dopfn.png


── UWYK vs Ours(fn=50)  (N ≈ 1250·d) ──
  d      N    n            UWYK (mean±SEM)     Ours(fn=50) (mean±SEM)  √PEHE ratio  MSE ratio
------------------------------------------------------------------------------------------------
  2   2500   15        1.103 ±    0.097          0.643 ±    0.073          1.715      2.942
  3   3750   15        0.898 ±    0.076          0.572 ±    0.046          1.571      2.467
  4   5000   15        1.135 ±    0.089          0.719 ±    0.066          1.578      2.490
  5   6250   15        1.007 ±    0.084          0.703 ±    0.051          1.432      2.051
  6   7500   15        1.175 ±    0.093          0.820 ±    0.061          1.432      2.051
  8  10000   15        1.215 ±    0.076          0.910 ±    0.058          1.336      1.784
[save] /scratch/furkanbd/rpfn_bench_kit/d_n_grid/out_all.png


── Do-PFN vs Ours(DoPFN-bb)  (N ≈ 1250·d) ──
  d      N    n         Do-PFN (mean±SEM)  Ours(DoPFN-bb) (mean±SEM)  √PEHE ratio  MSE ratio
------------------------------------------------------------------------------------------------
  2   2500   15        1.273 ±    0.123          0.591 ±    0.059          2.154      4.639
  3   3750   15        0.866 ±    0.102          0.452 ±    0.049          1.916      3.671
  4   5000   15        1.067 ±    0.120          0.550 ±    0.091          1.940      3.764
  5   6250   15        0.842 ±    0.081          0.483 ±    0.046          1.743      3.038
  6   7500   15        1.025 ±    0.092          0.600 ±    0.058          1.708      2.917
  8  10000   15        1.160 ±    0.084          0.744 ±    0.049          1.559      2.430

── Do-PFN vs Ours-DoPFN-bb(200K)  (N ≈ 1250·d) ──
  d      N    n          Do-PFN (mean±SEM) Ours-DoPFN-bb(200K) (mean±SEM)  √PEHE ratio  MSE ratio
------------------------------------------------------------------------------------------------
  2   2500   15        1.273 ±    0.123          0.630 ±    0.061          2.021      4.084
  3   3750   15        0.866 ±    0.102          0.516 ±    0.036          1.680      2.821
  4   5000   15        1.067 ±    0.120          0.531 ±    0.039          2.009      4.037
  5   6250   15        0.842 ±    0.081          0.499 ±    0.030          1.687      2.846
  6   7500   15        1.025 ±    0.092          0.562 ±    0.030          1.823      3.325
  8  10000   15        1.160 ±    0.084          0.584 ±    0.032          1.987      3.948
[save] /scratch/furkanbd/rpfn_bench_kit/d_n_grid/out_all_dopfn.png


══ ACIC — per-bin probability L2 (J=10 bins for y0/y1; 20 τ bins width 0.20) ══

method        y0                y1          τ (CATE)               ATE

J = 10

B=1000 (2D-τ) 2D-marg

Do-PFN         1.3298±0.0074   1.3900±0.0076     1.0414±0.0081     1.2947±0.062
Do-PFN-bb MALC 1.4278±0.0061   1.4554±0.0062     1.3021±0.0053     1.1861±0.1034


J = 100
B=1000 (2D-τ) 2D-marg 

fn=50        2.2090±0.0109  2.3540±0.0100  1.7459±0.0089     1.5707±0.1564
UWYK-NoAnc   2.5180±0.0115  2.7026±0.0104  1.9851±0.0079     1.7835±0.1409
UWYK-FullAnc 2.5401±0.0119  2.6384±0.0113  1.8754±0.0083     1.6054±0.1327

<!-- J = 10
Ours(fn=50)  0.9870±0.0079     1.1557±0.0079     1.0062±0.0065     0.8816±0.0972
UWYK-NoAnc   0.9238±0.0080     1.2021±0.0085     0.9652±0.0092     1.0917±0.0842
UWYK-FullAnc 0.8798±0.0085     1.0798±0.0089     0.8837±0.0098     0.8815±0.0855 -->

══ IHDP — per-bin probability L2 (J=10 bins for y0/y1; 20 τ bins width 0.20) ══

method            y0                y1          τ(CATE)           ATE

J = 10

B=1000 (2D-τ) 2D-marg

Do-PFN        1.0027±0.0066     0.9714±0.0050     1.1373±0.0063     0.9013±0.0414
Do-PFN-bb MALC 0.9773±0.0063     1.0427±0.0070     1.0693±0.0066     0.8989±0.0507

J=100
(2D-τ) 2D-marg B=1000
fn=50  1.5265±0.0145     1.5736±0.0134     1.3315±0.0122     1.2336±0.0984 UWYK-NoAnc 1.8256±0.0136    1.8503±0.0129  1.5543±0.0120     1.4156±0.0905
UWYK-FullAnc 1.8233±0.0136  1.8080±0.0133  1.4459±0.0128     1.2518±0.0996


<!-- J=10
(2D-τ) 2D-marg B=1000 
method            y0                y1          τ(CATE)           ATE
Ours(fn=50)   0.9113±0.0056     0.9604±0.0038     0.9278±0.0050     0.8550±0.0388
UWYK-NoAnc    0.9315±0.0057     1.0637±0.0051     1.2444±0.0064     1.0340±0.0337
UWYK-FullAnc  0.8150±0.0063     0.9587±0.0061     1.1268±0.0076     0.8670±0.0435 -->


python benchmarks/l2_ihdp/l2_per_bin_prob.py \
  --shards-glob                    "$DEPLOY_ROOT/ihdp_l2_dopfnbb_j10_s150k_B100K1_all5.r*.npz" \
  --bb-b500-shards-glob            "$DEPLOY_ROOT/ihdp_l2_dopfnbb_j10_s150k_B500K1_all5.r*.npz" \
  --bb-b1000-shards-glob           "$DEPLOY_ROOT/ihdp_l2_dopfnbb_j10_s150k_B1000K1_all5.r*.npz" \
  --bb-2dmarg-shards-glob          "$DEPLOY_ROOT/ihdp_l2_dopfnbb_j10_s150k_2dmarg_B100K1.r*.npz" \
  --bb-2dmarg-b500-shards-glob     "$DEPLOY_ROOT/ihdp_l2_dopfnbb_j10_s150k_2dmarg_B500K1.r*.npz" \
  --bb-2dmarg-b1000-shards-glob    "$DEPLOY_ROOT/ihdp_l2_dopfnbb_j10_s150k_2dmarg_B1000K1.r*.npz" \
  --dopfn-shards-glob              "$DEPLOY_ROOT/ihdp_l2_3methods_B100K1_loglin.r*.npz" \
  --fn50-shards-glob               "$DEPLOY_ROOT/ihdp_l2_3methods_B100K1_loglin.r*.npz" \
  --fn50-b500-shards-glob          "$DEPLOY_ROOT/ihdp_l2_fn50_B500K1.r*.npz" \
  --fn50-b1000-shards-glob         "$DEPLOY_ROOT/ihdp_l2_fn50_B1000K1.r*.npz" \
  --fn50-2dmarg-shards-glob        "$DEPLOY_ROOT/ihdp_l2_fn50_2dmarg_B100K1.r*.npz" \
  --fn50-2dmarg-b500-shards-glob   "$DEPLOY_ROOT/ihdp_l2_fn50_2dmarg_B500K1.r*.npz" \
  --fn50-2dmarg-b1000-shards-glob  "$DEPLOY_ROOT/ihdp_l2_fn50_2dmarg_B1000K1.r*.npz" \
  --uwyk-shards-glob               "$DEPLOY_ROOT/ihdp_l2_uwyk_B100K1_loglin.r*.npz" \
  --repo $DEPLOY_ROOT/R-PFN --causalpfn $DEPLOY_ROOT/external/causalpfn \
  --checkpoint-dopfn-bb $CKPT_BB --dataset ihdp


 python benchmarks/l2_ihdp/l2_per_bin_prob.py \
    --shards-glob                    "$DEPLOY_ROOT/acic_l2_dopfnbb_j10_s150k_B100K1_all5.r*.npz" \
    --bb-b500-shards-glob            "$DEPLOY_ROOT/acic_l2_dopfnbb_j10_s150k_B500K1_all5.r*.npz" \
    --bb-b1000-shards-glob           "$DEPLOY_ROOT/acic_l2_dopfnbb_j10_s150k_B1000K1_all5.r*.npz" \
    --bb-2dmarg-shards-glob          "$DEPLOY_ROOT/acic_l2_dopfnbb_j10_s150k_2dmarg_B100K1.r*.npz" \
    --bb-2dmarg-b500-shards-glob     "$DEPLOY_ROOT/acic_l2_dopfnbb_j10_s150k_2dmarg_B500K1.r*.npz" \
    --bb-2dmarg-b1000-shards-glob    "$DEPLOY_ROOT/acic_l2_dopfnbb_j10_s150k_2dmarg_B1000K1.r*.npz" \
    --dopfn-shards-glob              "$DEPLOY_ROOT/acic_l2_3methods_B100K1_loglin.r*.npz" \
    --fn50-shards-glob               "$DEPLOY_ROOT/acic_l2_3methods_B100K1_loglin.r*.npz" \
    --fn50-b500-shards-glob          "$DEPLOY_ROOT/acic_l2_fn50_B500K1.r*.npz" \
    --fn50-b1000-shards-glob         "$DEPLOY_ROOT/acic_l2_fn50_B1000K1.r*.npz" \
    --fn50-2dmarg-shards-glob        "$DEPLOY_ROOT/acic_l2_fn50_2dmarg_B100K1.r*.npz" \
    --fn50-2dmarg-b500-shards-glob   "$DEPLOY_ROOT/acic_l2_fn50_2dmarg_B500K1.r*.npz" \
    --fn50-2dmarg-b1000-shards-glob  "$DEPLOY_ROOT/acic_l2_fn50_2dmarg_B1000K1.r*.npz" \
    --uwyk-shards-glob               "$DEPLOY_ROOT/acic_l2_uwyk_B100K1_loglin.r*.npz" \
    --repo $DEPLOY_ROOT/R-PFN \
    --causalpfn $DEPLOY_ROOT/external/causalpfn \
    --dopfn $DEPLOY_ROOT/external/dopfn \
    --checkpoint-dopfn-bb $DEPLOY_ROOT/checkpoints_dopfn_backbone_realj10/step_150000.pt \
    --dataset acic



│ step │ chunk │               elapsed_in_chunk               │ cumulative H100-hours │
├──────┼───────┼──────────────────────────────────────────────┼───────────────────────┤
│  25k │ 1     │                                     17,192 s │                4.78 h │
├──────┼───────┼──────────────────────────────────────────────┼───────────────────────┤
│  50k │ 1     │                                     34,036 s │                9.45 h │
├──────┼───────┼──────────────────────────────────────────────┼───────────────────────┤
│ 100k │ 1     │                                     67,951 s │               18.88 h │
├──────┼───────┼──────────────────────────────────────────────┼───────────────────────┤
│ 150k │ 2     │ 33,470 s (chunk 2) + full chunk 1 (71,948 s) │               29.28 h │

<!-- ══ d=6 SYNTHETIC — per-bin probability L2 (J=10 y-bins, 20 τ-bins @0.20) ══

method          y0                 y1          τ (CATE)               ATE
Do-PFN     0.4853±0.0173     0.5359±0.0187     0.4200±0.0144     0.3080±0.0325
Do-PFN-bb  0.5639±0.0215     0.6524±0.0216     0.6444±0.0240     0.5243±0.1235


for one query with one seed.

0.9768±0.0000     1.8124±0.0000     0.9753±0.0000     0.9745±0.0000
0.8821±0.0000     1.2912±0.0000     0.5438±0.0000     0.5431±0.0000 -->


══ ACIC  (n=10) ══
  err_dopfn                           mean=        0.6645  sem=    0.0411  n=10
  err_ours_mean                       mean=        0.1542  sem=    0.0383  n=10
  err_uwyk_anc                        mean=        0.1447  sem=    0.0364  n=10
  err_uwyk_noanc                      mean=        0.3639  sem=    0.0586  n=10
  pehe_dopfn                          mean=        4.1144  sem=    0.5480  n=10
  pehe_ours_mean                      mean=        2.8236  sem=    0.4619  n=10
  pehe_uwyk_anc                       mean=        2.7345  sem=    0.4390  n=10
  pehe_uwyk_noanc                     mean=        3.3179  sem=    0.4497  n=10

══ CPS  (n=100) ══
  err_dopfn                           mean=        0.8797  sem=    0.0064  n=100
  err_ours_mean                       mean=        1.0164  sem=    0.0016  n=100
  err_uwyk_anc                        mean=        1.0716  sem=    0.0031  n=100
  err_uwyk_noanc                      mean=        1.0777  sem=    0.0034  n=100
  pehe_dopfn                          mean=    12014.6412  sem=   32.0163  n=100
  pehe_ours_mean                      mean=    12826.0381  sem=   17.7787  n=100
  pehe_uwyk_anc                       mean=    12938.6962  sem=   22.7391  n=100
  pehe_uwyk_noanc                     mean=    13057.1503  sem=   23.4424  n=100

══ PSID  (n=100) ══
  err_dopfn                           mean=        0.9270  sem=    0.0069  n=100
  err_ours_mean                       mean=        0.9089  sem=    0.0042  n=100
  err_uwyk_anc                        mean=        0.9188  sem=    0.0070  n=100
  err_uwyk_noanc                      mean=        0.9636  sem=    0.0013  n=100
  pehe_dopfn                          mean=    20907.1986  sem=  138.2358  n=100
  pehe_ours_mean                      mean=    21992.3657  sem=  131.4558  n=100
  pehe_uwyk_anc                       mean=    22234.6924  sem=  155.4589  n=100
  pehe_uwyk_noanc                     mean=    22401.3362  sem=  131.8464  n=100


grep -E "^\s+[0-9]" /scratch/furkanbd/rpfn_bench_kit/logs_graph2d/train_5028410.out | tail -20

grep -E "^\s+[0-9]" /scratch/furkanbd/rpfn_bench_kit/logs_causalpfn2d/train_5032910.out | tail -20


method                             PEHE ± SEM      eps_ATE ± SEM   n
UWYK Predictive                 3.136 ± 0.475      0.376 ± 0.057   10
UWYK No-Anc                     3.409 ± 0.521      0.372 ± 0.075   10
UWYK Anc                        2.695 ± 0.418      0.177 ± 0.062   10
ours fn=50 (null-t)             2.858 ± 0.500      0.207 ± 0.047   10



cd /scratch/furkanbd/rpfn_bench_kit

JOB_ID_MAIN=5032883 \
JOB_ID_PREDSTYLE=5032912 \
JOB_ID_ACIC_UWYK=5032845 \
JOB_ID_ACIC_FN50=5032869 \
JOB_ID_ACIC_FN50P=5032873 \
UWYK_REPRO=/scratch/furkanbd/rpfn_bench_kit/external/uwyk_reproduce \
TABLE1_OUT_ROOT=/scratch/furkanbd/rpfn_bench_kit/results_table1_all \
python3 /scratch/furkanbd/rpfn_bench_kit/R-PFN/benchmarks/uwyk_table1/aggregate_all_datasets.py



cd /scratch/furkanbd/rpfn_bench_kit

JOB_ID_MAIN=5032883 \
JOB_ID_ACIC_FN50=5032869 \
UWYK_REPRO=/scratch/furkanbd/rpfn_bench_kit/external/uwyk_reproduce \
TABLE1_OUT_ROOT=/scratch/furkanbd/rpfn_bench_kit/results_table1_all \
python3 /scratch/furkanbd/rpfn_bench_kit/R-PFN/benchmarks/uwyk_table1/aggregate_all_datasets.py



JOB=5032938
python3 -c "
import glob, pickle, numpy as np
files = sorted(glob.glob(f'/scratch/furkanbd/rpfn_bench_kit/results_table1_all/table1_all_CPS_fn50c_{$JOB}/*'))
pehes = [pickle.load(open(f,'rb'))['pehe'] for f in files]
ates  = [pickle.load(open(f,'rb'))['ate_rel_err'] for f in files]
pehes, ates = np.array(pehes), np.array(ates)
print(f'CPS fn=50 clustered: n={len(pehes)}')
print(f'  PEHE     = {pehes.mean():.0f} ± {pehes.std(ddof=1)/np.sqrt(len(pehes)):.0f}')
print(f'  eps_ATE  = {ates.mean():.3f} ± {ates.std(ddof=1)/np.sqrt(len(ates)):.3f}')
print(f'  vs UWYK Predictive = 11856  |  vs fn=50 single-pass = 12703  |  vs paper Anc = 11213')
"

- ACIC / CPS drift because n_train > 1000 always triggers an unseeded 1000-row subsample, and the paper's subsample ≠ our subsample

source /scratch/furkanbd/rpfn_bench_kit/venv/bin/activate

# cpfn2d (H100 job 5214864, ~7h in)
echo "=== cpfn2d-j32-rand (H100) ==="
ls -1 /scratch/furkanbd/rpfn_bench_kit/cpfn2d_j32_random_A1_h100/step_checkpoints/run/step_*.pt | sort -V | tail -3
LATEST_LOG=$(ls -t logs_cpfn2d_j32_random/train_5214864*.out 2>/dev/null | head -1)
[ -n "$LATEST_LOG" ] && grep -E 'step-ckpt.*written|resuming from|actual_step' "$LATEST_LOG" | tail -5

# cpfn1d (L40s job 5214936, ~8h in)
echo ""
echo "=== cpfn-orig-headrand (L40s) ==="
ls -1 /scratch/furkanbd/rpfn_bench_kit/causalpfn_j1024_headrand_A1_h100/step_checkpoints/run/step_*.pt | sort -V | tail -3
LATEST_LOG=$(ls -t logs_causalpfn_j1024_headrand/train_5214936*.out 2>/dev/null | head -1)
[ -n "$LATEST_LOG" ] && grep -E 'step-ckpt.*written|resuming from|actual_step' "$LATEST_LOG" | tail -5


tail -f logs_scm_case_studies_regen/regen_5222711.out



for f in logs_cpfn2d_rc_cs/eval_*.out; do
    echo "===== $f ====="
    grep -A5 '══.*summary' "$f"
done

Or scoped to a single job:
grep -A5 '══.*summary' logs_cpfn2d_rc_cs/eval_5283446__*.out




OUT=/scratch/furkanbd/rpfn_bench_kit/results_ci95_ptau
for m in uwyk dopfn_native dopfn_bb graph2d; do
    echo "=== $m ==="
    for d in IHDP ACIC CPS PSID PSID_bal PSIDbal; do
        n=$(ls $OUT/$m/$d/*.npz 2>/dev/null | wc -l)
        [ "$n" -gt 0 ] && printf "  %-10s %s npzs\n" "$d" "$n"
    done
done



source /scratch/furkanbd/rpfn_bench_kit/venv/bin/activate
OUT=/scratch/furkanbd/rpfn_bench_kit/results_ci95_ptau

python benchmarks/scripts/compute_ci95_coverage.py \
    --results-root $OUT \
    --pattern '{model}/{dataset}/*.npz' \
    --models uwyk dopfn_native dopfn_bb graph2d \
    --datasets IHDP ACIC CPS PSID PSID_bal PSIDbal \
    --out $OUT/ci95_ptau_summary.csv

cat $OUT/ci95_ptau_summary.csv

python R-PFN/realcause_eval/aggregate_all_methods.py     --out-root /scratch/furkanbd/rpfn_bench_kit/results_realcause_all


tail -f logs_rc_1d_ci/agg_5292698.out               # see per-cell progress

cat logs_rc_1d_ci/agg_5293284.out | tail -30

| Method | IHDP | ACIC | CPS | PSID | PSID_bal |
|---|---|---|---|---|---|
| cpfn1d | PEHE 0.825 ± 0.146<br>ε_ATE 0.052 ± 0.022<br>Cov 0.998 ± 0.001<br>Len 4.712 ± 0.128 (n=100) | PEHE 1.412 ± 0.232<br>ε_ATE 0.090 ± 0.021<br>Cov 0.997 ± 0.001<br>Len 8.761 ± 0.819 (n=10) | PEHE 10,042.87 ± 181.46<br>ε_ATE 0.352 ± 0.042<br>Cov 0.980 ± 0.002<br>Len 47,788.48 ± 263.18 (n=100) | PEHE 18,281.14 ± 298.92<br>ε_ATE 0.684 ± 0.026<br>Cov 0.951 ± 0.002<br>Len 76,383.48 ± 1,242.61 (n=100) | PEHE 18,645.49 ± 315.53<br>ε_ATE 0.707 ± 0.028<br>Cov 0.935 ± 0.002<br>Len 71,298.36 ± 1,217.27 (n=100) |
| dopfn | PEHE 6.065 ± 0.899<br>ε_ATE 0.932 ± 0.400<br>Cov 0.988 ± 0.002<br>Len 25.912 ± 3.190 (n=100) | PEHE 4.114 ± 0.548<br>ε_ATE 0.664 ± 0.041<br>Cov 0.986 ± 0.007<br>Len 26.359 ± 0.858 (n=10) | PEHE 12,014.15 ± 32.00<br>ε_ATE 0.880 ± 0.006<br>Cov 0.943 ± 0.002<br>Len 48,873.92 ± 115.79 (n=100) | PEHE 20,906.86 ± 138.26<br>ε_ATE 0.927 ± 0.007<br>Cov 0.978 ± 0.001<br>Len 86,170.80 ± 298.37 (n=100) | PEHE 22,892.15 ± 161.33<br>ε_ATE 1.090 ± 0.012<br>Cov 0.972 ± 0.001<br>Len 86,369.13 ± 546.38 (n=100) |
| uwyk1d | PEHE 6.284 ± 0.792<br>ε_ATE 0.668 ± 0.049<br>Cov 0.994 ± 0.001<br>Len 25.500 ± 2.662 (n=100) | PEHE 3.347 ± 0.531<br>ε_ATE 0.444 ± 0.088<br>Cov 0.993 ± 0.004<br>Len 23.161 ± 1.522 (n=10) | PEHE 12,564.98 ± 36.99<br>ε_ATE 0.958 ± 0.005<br>Cov 0.950 ± 0.002<br>Len 44,851.51 ± 94.42 (n=100) | PEHE 22,442.89 ± 132.43<br>ε_ATE 0.976 ± 0.002<br>Cov 0.883 ± 0.003<br>Len 60,142.62 ± 246.75 (n=100) | PEHE 21,896.20 ± 136.84<br>ε_ATE 0.936 ± 0.004<br>Cov 0.899 ± 0.003<br>Len 60,267.88 ± 318.46 (n=100) |


| Method | IHDP | ACIC | CPS | PSID | PSID_bal |
|---|---|---|---|---|---|
| cpfn2d | PEHE 1.218 ± 0.123<br>ε_ATE 0.055 ± 0.006<br>Cov 0.999 ± 0.000<br>Len 14.027 ± 1.544 (n=100) | PEHE 1.449 ± 0.237<br>ε_ATE 0.146 ± 0.029<br>Cov 0.998 ± 0.002<br>Len 13.149 ± 0.485 (n=10) | PEHE 10,770.40 ± 266.74<br>ε_ATE 0.441 ± 0.059<br>Cov 0.949 ± 0.003<br>Len 41,581.00 ± 194.00 (n=100) | PEHE 14,797.71 ± 182.26<br>ε_ATE 0.238 ± 0.011<br>Cov 0.968 ± 0.001<br>Len 54,739.93 ± 610.67 (n=100) | PEHE 16,364.75 ± 376.47<br>ε_ATE 0.322 ± 0.030<br>Cov 0.960 ± 0.002<br>Len 59,323.72 ± 949.50 (n=100) |
| graph2d | PEHE 4.507 ± 0.630<br>ε_ATE 0.375 ± 0.088<br>Cov 0.997 ± 0.001<br>Len 26.732 ± 2.880 (n=100) | PEHE 2.743 ± 0.500<br>ε_ATE 0.134 ± 0.044<br>Cov 0.979 ± 0.012<br>Len 18.431 ± 2.568 (n=10) | PEHE 12,299.18 ± 30.84<br>ε_ATE 0.876 ± 0.007<br>Cov 0.908 ± 0.003<br>Len 41,210.41 ± 393.52 (n=100) | PEHE 21,426.07 ± 146.15<br>ε_ATE 0.849 ± 0.012<br>Cov 0.769 ± 0.015<br>Len 41,558.74 ± 1,071.65 (n=100) | PEHE 20,039.15 ± 141.96<br>ε_ATE 0.676 ± 0.009<br>Cov 0.944 ± 0.002<br>Len 58,465.90 ± 454.55 (n=100) |
| dopfnbb | PEHE 5.057 ± 0.707<br>ε_ATE 0.530 ± 0.183<br>Cov 0.774 ± 0.020<br>Len 15.304 ± 2.010 (n=100) | PEHE 3.148 ± 0.640<br>ε_ATE 0.289 ± 0.082<br>Cov 0.819 ± 0.063<br>Len 10.308 ± 1.296 (n=10) | PEHE 10,715.28 ± 16.42<br>ε_ATE 0.117 ± 0.008<br>Cov 0.565 ± 0.001<br>Len 18,834.20 ± 35.47 (n=100) | PEHE 18,617.25 ± 146.10<br>ε_ATE 0.437 ± 0.009<br>Cov 0.736 ± 0.005<br>Len 37,228.86 ± 266.47 (n=100) | PEHE 18,711.03 ± 158.33<br>ε_ATE 0.445 ± 0.015<br>Cov 0.780 ± 0.006<br>Len 40,183.43 ± 390.85 (n=100) |


PEHE + ATE_err + Coverage + Length + l2 density 

cat logs_rc_2d_ci/agg_5293805.out
cat $OUT/cate_ci_2d.md

squeue --me
tail -F logs_rc_2d_malc_ci/dump_5293950_{0,5,10}.out

tail -F logs_rc_2d_malc_ci/dump_5293950_10.out


for f in $OUT_1D/cate_ci.md $OUT_2D/cate_ci_2d_malc_B500.md \
         $OUT_1D/ate_w2_marginals.md $OUT_2D/ate_w2_malc_B500.md; do
  echo ""; echo "=========== $f ==========="
  cat $f
done


(venv) furkanbd@klogin02:/scratch/furkanbd/rpfn_bench_kit/R-PFN$ for f in $OUT_1D/cate_ci.md $OUT_2D/cate_ci_2d_malc_B500.md \
         $OUT_1D/ate_w2_marginals.md $OUT_2D/ate_w2_malc_B500.md; do
  echo ""; echo "=========== $f ==========="
  cat $f
done

=========== /scratch/furkanbd/rpfn_bench_kit/results_1d_density/cate_ci.md ===========

RealCause density-CI — /scratch/furkanbd/rpfn_bench_kit/results_1d_density

(each cell, top → bottom: √PEHE / ε_ATE (from stored point CATE — matches realcause_eval mega-sbatch), Coverage / Length (95% CI from the density, assuming Y|do(0) ⊥ Y|do(1)); n = realizations)

| Method | IHDP | ACIC | CPS | PSID | PSID_bal |
|---|---|---|---|---|---|
| cpfn1d | PEHE 0.825 ± 0.146<br>ε_ATE 0.052 ± 0.022<br>Cov 0.998 ± 0.001<br>Len 4.712 ± 0.128 (n=100) | PEHE 1.412 ± 0.232<br>ε_ATE 0.090 ± 0.021<br>Cov 0.997 ± 0.001<br>Len 8.761 ± 0.819 (n=10) | PEHE 10,042.87 ± 181.46<br>ε_ATE 0.352 ± 0.042<br>Cov 0.980 ± 0.002<br>Len 47,788.48 ± 263.18 (n=100) | PEHE 18,281.14 ± 298.92<br>ε_ATE 0.684 ± 0.026<br>Cov 0.951 ± 0.002<br>Len 76,383.48 ± 1,242.61 (n=100) | PEHE 18,645.49 ± 315.53<br>ε_ATE 0.707 ± 0.028<br>Cov 0.935 ± 0.002<br>Len 71,298.36 ± 1,217.27 (n=100) |
| dopfn | PEHE 6.065 ± 0.899<br>ε_ATE 0.932 ± 0.400<br>Cov 0.988 ± 0.002<br>Len 25.912 ± 3.190 (n=100) | PEHE 4.114 ± 0.548<br>ε_ATE 0.664 ± 0.041<br>Cov 0.986 ± 0.007<br>Len 26.359 ± 0.858 (n=10) | PEHE 12,014.15 ± 32.00<br>ε_ATE 0.880 ± 0.006<br>Cov 0.943 ± 0.002<br>Len 48,873.92 ± 115.79 (n=100) | PEHE 20,906.86 ± 138.26<br>ε_ATE 0.927 ± 0.007<br>Cov 0.978 ± 0.001<br>Len 86,170.80 ± 298.37 (n=100) | PEHE 22,892.15 ± 161.33<br>ε_ATE 1.090 ± 0.012<br>Cov 0.972 ± 0.001<br>Len 86,369.13 ± 546.38 (n=100) |
| uwyk1d | PEHE 6.284 ± 0.792<br>ε_ATE 0.668 ± 0.049<br>Cov 0.994 ± 0.001<br>Len 25.500 ± 2.662 (n=100) | PEHE 3.347 ± 0.531<br>ε_ATE 0.444 ± 0.088<br>Cov 0.993 ± 0.004<br>Len 23.161 ± 1.522 (n=10) | PEHE 12,564.98 ± 36.99<br>ε_ATE 0.958 ± 0.005<br>Cov 0.950 ± 0.002<br>Len 44,851.51 ± 94.42 (n=100) | PEHE 22,442.89 ± 132.43<br>ε_ATE 0.976 ± 0.002<br>Cov 0.883 ± 0.003<br>Len 60,142.62 ± 246.75 (n=100) | PEHE 21,896.20 ± 136.84<br>ε_ATE 0.936 ± 0.004<br>Cov 0.899 ± 0.003<br>Len 60,267.88 ± 318.46 (n=100) |

## Sanity: max |mean(density) − stored ate| across realizations
(should be ≈ 0 — the density used for CI IS the density that produced the reported point CATE; if not, CI is on a different distribution than the stored PEHE row)

| Method | IHDP | ACIC | CPS | PSID | PSID_bal |
|---|---|---|---|---|---|
| cpfn1d | |Δate density-vs-stored| max = 1.75e-06 | |Δate density-vs-stored| max = 3.75e-07 | |Δate density-vs-stored| max = 1.15e-03 | |Δate density-vs-stored| max = 7.63e-04 | |Δate density-vs-stored| max = 8.79e-04 |
| dopfn | |Δate density-vs-stored| max = 2.90e-06 | |Δate density-vs-stored| max = 3.35e-07 | |Δate density-vs-stored| max = 2.04e-04 | |Δate density-vs-stored| max = 4.66e-04 | |Δate density-vs-stored| max = 4.33e-04 |
| uwyk1d | |Δate density-vs-stored| max = 1.78e-06 | |Δate density-vs-stored| max = 1.85e-07 | |Δate density-vs-stored| max = 8.65e-05 | |Δate density-vs-stored| max = 6.71e-04 | |Δate density-vs-stored| max = 7.42e-04 |

=========== /scratch/furkanbd/rpfn_bench_kit/results_rc_2d_density/cate_ci_2d_malc_B500.md ===========

RealCause density-CI, 2D-MALC-smoothed p(τ) — /scratch/furkanbd/rpfn_bench_kit/results_rc_2d_density

(each cell, top → bottom: √PEHE / ε_ATE (from stored point CATE — matches realcause_eval mega-sbatch), Coverage / Length (95% CI from the 2D-MALC-smoothed p(τ), NOT the raw joint); n = realizations)

| Method | IHDP | ACIC | CPS | PSID | PSID_bal |
|---|---|---|---|---|---|
| cpfn2d | PEHE 1.218 ± 0.123<br>ε_ATE 0.055 ± 0.006<br>Cov 0.999 ± 0.000<br>Len 14.073 ± 1.539 (n=100) | PEHE 1.449 ± 0.237<br>ε_ATE 0.146 ± 0.029<br>Cov 0.997 ± 0.002<br>Len 13.330 ± 0.480 (n=10) | PEHE 10,770.40 ± 266.74<br>ε_ATE 0.441 ± 0.059<br>Cov 0.930 ± 0.007<br>Len 39,956.02 ± 143.91 (n=100) | PEHE 14,797.71 ± 182.26<br>ε_ATE 0.238 ± 0.011<br>Cov 0.966 ± 0.001<br>Len 52,914.53 ± 381.35 (n=100) | PEHE 16,364.75 ± 376.47<br>ε_ATE 0.322 ± 0.030<br>Cov 0.957 ± 0.003<br>Len 55,380.01 ± 694.69 (n=100) |
| graph2d | PEHE 4.507 ± 0.630<br>ε_ATE 0.375 ± 0.088<br>Cov 0.996 ± 0.001<br>Len 26.282 ± 3.034 (n=100) | PEHE 2.743 ± 0.500<br>ε_ATE 0.134 ± 0.044<br>Cov 0.981 ± 0.011<br>Len 18.850 ± 2.272 (n=10) | PEHE 12,299.18 ± 30.84<br>ε_ATE 0.876 ± 0.007<br>Cov 0.878 ± 0.002<br>Len 38,401.11 ± 190.34 (n=100) | PEHE 21,426.07 ± 146.15<br>ε_ATE 0.849 ± 0.012<br>Cov 0.799 ± 0.008<br>Len 51,722.99 ± 696.72 (n=100) | PEHE 20,039.15 ± 141.96<br>ε_ATE 0.676 ± 0.009<br>Cov 0.917 ± 0.003<br>Len 63,558.72 ± 486.20 (n=100) |
| dopfnbb | PEHE 5.057 ± 0.707<br>ε_ATE 0.530 ± 0.183<br>Cov 0.919 ± 0.011<br>Len 16.879 ± 2.182 (n=100) | PEHE 3.148 ± 0.640<br>ε_ATE 0.289 ± 0.082<br>Cov 0.899 ± 0.045<br>Len 13.250 ± 1.333 (n=10) | PEHE 10,715.28 ± 16.42<br>ε_ATE 0.117 ± 0.008<br>Cov 0.743 ± 0.001<br>Len 26,411.47 ± 48.96 (n=100) | PEHE 18,617.25 ± 146.10<br>ε_ATE 0.437 ± 0.009<br>Cov 0.853 ± 0.003<br>Len 45,488.19 ± 211.86 (n=100) | PEHE 18,711.03 ± 158.33<br>ε_ATE 0.445 ± 0.015<br>Cov 0.863 ± 0.004<br>Len 47,208.44 ± 370.57 (n=100) |

## Sanity — |mean(MALC p(τ)) − stored point ate| max, MALC fit-fail fraction max
(expect small non-zero bias from MALC smoothing; fit-fail should be ~0)

| Method | IHDP | ACIC | CPS | PSID | PSID_bal |
|---|---|---|---|---|---|
| cpfn2d | |Δ| max = 1.08e-01  fails≤0.0% | |Δ| max = 1.80e-02  fails≤0.0% | |Δ| max = 70.3291  fails≤0.1% | |Δ| max = 288.5103  fails≤0.4% | |Δ| max = 423.5565  fails≤0.4% |
| graph2d | |Δ| max = —  fails≤1.3% | |Δ| max = —  fails≤0.6% | |Δ| max = —  fails≤0.6% | |Δ| max = —  fails≤6.0% | |Δ| max = —  fails≤1.5% |
| dopfnbb | |Δ| max = —  fails≤5.3% | |Δ| max = —  fails≤1.2% | |Δ| max = —  fails≤0.4% | |Δ| max = —  fails≤2.6% | |Δ| max = —  fails≤1.1% |

=========== /scratch/furkanbd/rpfn_bench_kit/results_1d_density/ate_w2_marginals.md ===========

ATE density (1D W2 barycenter over queries) — tag=marginals — /scratch/furkanbd/rpfn_bench_kit/results_1d_density

(each cell, top → bottom: ATE_mean ± SE, ATE_bias (mean − true_ATE), Coverage / Length of 95% CI on p_ATE; n = realizations)

| Method | IHDP | ACIC | CPS | PSID | PSID_bal |
|---|---|---|---|---|---|
| cpfn1d | ATE 4.320 ± 0.163<br>bias +0.050<br>Cov 1.000 ± 0.000<br>Len 4.746 ± 0.134 (n=100) | ATE 3.891 ± 0.416<br>bias +0.310<br>Cov 1.000 ± 0.000<br>Len 8.770 ± 0.818 (n=10) | ATE -4,651.42 ± 302.14<br>bias +2,371.74<br>Cov 1.000 ± 0.000<br>Len 47,793.16 ± 263.42 (n=100) | ATE -4,192.53 ± 348.71<br>bias +9,016.51<br>Cov 1.000 ± 0.000<br>Len 76,395.03 ± 1,242.55 (n=100) | ATE -3,881.05 ± 367.16<br>bias +9,327.99<br>Cov 1.000 ± 0.000<br>Len 71,310.57 ± 1,217.20 (n=100) |
| dopfn | ATE 4.718 ± 0.427<br>bias +0.448<br>Cov 1.000 ± 0.000<br>Len 26.067 ± 3.206 (n=100) | ATE 1.170 ± 0.181<br>bias -2.411<br>Cov 1.000 ± 0.000<br>Len 26.456 ± 0.858 (n=10) | ATE -842.89 ± 45.18<br>bias +6,180.28<br>Cov 1.000 ± 0.000<br>Len 48,973.34 ± 119.75 (n=100) | ATE -968.77 ± 89.49<br>bias +12,240.26<br>Cov 1.000 ± 0.000<br>Len 86,426.92 ± 298.90 (n=100) | ATE 1,163.40 ± 159.65<br>bias +14,372.44<br>Cov 1.000 ± 0.000<br>Len 86,652.90 ± 546.95 (n=100) |
| uwyk1d | ATE 1.674 ± 0.063<br>bias -2.596<br>Cov 1.000 ± 0.000<br>Len 25.224 ± 2.618 (n=100) | ATE 2.261 ± 0.426<br>bias -1.321<br>Cov 1.000 ± 0.000<br>Len 22.999 ± 1.520 (n=10) | ATE -292.78 ± 31.99<br>bias +6,730.38<br>Cov 1.000 ± 0.000<br>Len 44,817.19 ± 94.26 (n=100) | ATE -317.60 ± 24.27<br>bias +12,891.44<br>Cov 1.000 ± 0.000<br>Len 59,931.29 ± 247.34 (n=100) | ATE -849.53 ± 55.48<br>bias +12,359.51<br>Cov 1.000 ± 0.000<br>Len 59,966.74 ± 317.43 (n=100) |

=========== /scratch/furkanbd/rpfn_bench_kit/results_rc_2d_density/ate_w2_malc_B500.md ===========

ATE density (1D W2 barycenter over queries) — tag=malc_B500 — /scratch/furkanbd/rpfn_bench_kit/results_rc_2d_density

(each cell, top → bottom: ATE_mean ± SE, ATE_bias (mean − true_ATE), Coverage / Length of 95% CI on p_ATE; n = realizations)

| Method | IHDP | ACIC | CPS | PSID | PSID_bal |
|---|---|---|---|---|---|
| cpfn2d | ATE 4.413 ± 0.160<br>bias +0.143<br>Cov 1.000 ± 0.000<br>Len 14.253 ± 1.561 (n=100) | ATE 4.028 ± 0.412<br>bias +0.447<br>Cov 1.000 ± 0.000<br>Len 13.478 ± 0.486 (n=10) | ATE -4,175.95 ± 433.44<br>bias +2,847.21<br>Cov 1.000 ± 0.000<br>Len 40,119.32 ± 143.02 (n=100) | ATE -10,069.00 ± 153.65<br>bias +3,140.03<br>Cov 1.000 ± 0.000<br>Len 53,229.15 ± 379.87 (n=100) | ATE -9,037.69 ± 393.28<br>bias +4,171.34<br>Cov 1.000 ± 0.000<br>Len 55,688.48 ± 692.84 (n=100) |
| graph2d | ATE 3.218 ± 0.140<br>bias -1.052<br>Cov 1.000 ± 0.000<br>Len 26.296 ± 3.036 (n=100) | ATE 3.943 ± 0.386<br>bias +0.362<br>Cov 1.000 ± 0.000<br>Len 18.870 ± 2.270 (n=10) | ATE -833.43 ± 45.24<br>bias +6,189.73<br>Cov 1.000 ± 0.000<br>Len 38,404.07 ± 190.34 (n=100) | ATE -1,930.89 ± 154.68<br>bias +11,278.15<br>Cov 1.000 ± 0.000<br>Len 51,756.37 ± 696.42 (n=100) | ATE -4,176.76 ± 108.56<br>bias +9,032.28<br>Cov 1.000 ± 0.000<br>Len 63,583.85 ± 486.25 (n=100) |
| dopfnbb | ATE 3.662 ± 0.267<br>bias -0.608<br>Cov 1.000 ± 0.000<br>Len 16.896 ± 2.183 (n=100) | ATE 3.685 ± 0.461<br>bias +0.104<br>Cov 1.000 ± 0.000<br>Len 13.271 ± 1.332 (n=10) | ATE -6,270.44 ± 60.76<br>bias +752.72<br>Cov 1.000 ± 0.000<br>Len 26,441.87 ± 48.89 (n=100) | ATE -7,382.06 ± 119.57<br>bias +5,826.97<br>Cov 1.000 ± 0.000<br>Len 45,531.23 ± 211.73 (n=100) | ATE -7,290.70 ± 192.98<br>bias +5,918.33<br>Cov 1.000 ± 0.000<br>Len 47,249.66 ± 370.50 (n=100) |
(venv) furkanbd@klogin02:/scratch/furkanbd/rpfn_bench_kit/R-PFN$ 






Pushed. Here's the full submit-and-go sequence for the cluster. Uses new v3b OUT_ROOTs so the noanc tables you already have stay intact. Only re-runs the two methods that have anc modes (uwyk1d for 1D, graph2d for 2D); other methods don't have anc so they don't need v3b variants.

cd /scratch/furkanbd/rpfn_bench_kit/R-PFN && git pull
OUT_1D=/scratch/furkanbd/rpfn_bench_kit/results_1d_density                 # existing (noanc)
OUT_2D=/scratch/furkanbd/rpfn_bench_kit/results_rc_2d_density               # existing (noanc)
OUT_1D_V3B=/scratch/furkanbd/rpfn_bench_kit/results_1d_density_v3b          # new
OUT_2D_V3B=/scratch/furkanbd/rpfn_bench_kit/results_rc_2d_density_v3b       # new
CAUSALPFN=/scratch/furkanbd/rpfn_bench_kit/external/causalpfn               # adjust if diff

# ── 1. Density dumps at v3b ──────────────────────────────────────────────────
# 1D: only uwyk1d (task 10-14 in the 15-task array). Skips cpfn1d/dopfn.
sbatch --array=10-14 \
       --export=ALL,OUT_ROOT=$OUT_1D_V3B,DENSITY_ANC_TAG=v3b \
       benchmarks/cluster/submit_realcause_1d_density_dump.sbatch
# 2D: only graph2d (tasks 5-9). Skips cpfn2d/dopfnbb.
sbatch --array=5-9 \
       --export=ALL,OUT_ROOT=$OUT_2D_V3B,DENSITY_PRIMARY_MODE=v3b \
       benchmarks/cluster/submit_realcause_2d_density_dump.sbatch

# ── 2. Wait for step 1 to finish (density dumps ~30-90 min each cell). Then:

# MALC CI for graph2d only (tasks 5-9), B=500
sbatch --array=5-9 \
       --export=ALL,OUT_ROOT=$OUT_2D_V3B,MALC_B=500,OUT_TAG=B500 \
       benchmarks/cluster/submit_realcause_2d_malc_ci_dump.sbatch

# ATE W2 dumps
sbatch --array=10-14 \
       --export=ALL,OUT_ROOT=$OUT_1D_V3B,METHOD_SET=1d,SOURCE=marginals \
       benchmarks/cluster/submit_realcause_ate_density_w2_dump.sbatch
sbatch --array=5-9 \
       --export=ALL,OUT_ROOT=$OUT_2D_V3B,METHOD_SET=2d,SOURCE=malc,MALC_TAG=B500 \
       benchmarks/cluster/submit_realcause_ate_density_w2_dump.sbatch

# ── 3. Wait for step 2. Then final aggregators + density metrics:

# CATE tables at v3b (uwyk1d + graph2d rows only)
sbatch --export=ALL,OUT_ROOT=$OUT_1D_V3B,METHODS=uwyk1d,UWYK_TAG=v3b,SKIP_CRPS=1 \
       benchmarks/cluster/submit_realcause_1d_ci_summary.sbatch
sbatch --export=ALL,OUT_ROOT=$OUT_2D_V3B,METHODS=graph2d,IN_TAG=B500,GRAPH2D_TAG=v3b,SKIP_CRPS=1 \
       benchmarks/cluster/submit_realcause_2d_malc_ci_summary.sbatch

# ATE tables at v3b
sbatch --export=ALL,OUT_ROOT=$OUT_1D_V3B,TAGS=marginals,SKIP_CRPS=1 \
       benchmarks/cluster/submit_realcause_ate_density_w2_summary.sbatch
sbatch --export=ALL,OUT_ROOT=$OUT_2D_V3B,TAGS=malc_B500,SKIP_CRPS=1 \
       benchmarks/cluster/submit_realcause_ate_density_w2_summary.sbatch

# Density metrics at v3b — 5-task array (all datasets)
sbatch --export=ALL,OUT_1D=$OUT_1D_V3B,OUT_2D=$OUT_2D_V3B,CAUSALPFN=$CAUSALPFN \
       benchmarks/cluster/submit_realcause_density_metrics.sbatch

Runtime estimates — total wall-clock ~2-3 hours if step 1 and step 2 run sequentially. If cluster has slots for parallel runs step 1 alone dominates (~1 hour for CPS density dump).

When you come back, read the four v3b tables + density metrics:
OUT_1D_V3B=/scratch/furkanbd/rpfn_bench_kit/results_1d_density_v3b
OUT_2D_V3B=/scratch/furkanbd/rpfn_bench_kit/results_rc_2d_density_v3b
cat $OUT_1D_V3B/cate_ci.md                # uwyk1d v3b CATE
cat $OUT_2D_V3B/cate_ci_2d_malc_B500.md   # graph2d v3b CATE
cat $OUT_1D_V3B/ate_w2_marginals.md       # uwyk1d v3b ATE
cat $OUT_2D_V3B/ate_w2_malc_B500.md       # graph2d v3b ATE
for d in IHDP ACIC CPS PSID PSID_bal; do
  cat $OUT_2D_V3B/density_metrics_${d}.md
done



Also fix the density-metrics files issue in your current shell first — you need OUT_2D set:
OUT_2D=/scratch/furkanbd/rpfn_bench_kit/results_rc_2d_density
ls -l $OUT_2D/density_metrics_*.md   # check what's there for the ORIGINAL (noanc) run


furkanbd@klogin01:/scratch/furkanbd/rpfn_bench_kit/R-PFN$ for d in IHDP ACIC CPS PSID PSID_bal; do   cat $OUT_2D/density_metrics_${d}.md; done

Density metrics — IHDP — tight T=4001 raw τ grid

(truth = analytic Gaussian N(μ_1(x)−μ_0(x), 2σ²); metrics per query averaged, then averaged across 100 realizations. Lower = better.)

| Method | NLL | L2 | KL_fwd (truth‖est) | KL_rev (est‖truth) |
|---|---|---|---|---|
| cpfn1d | 21.4047 ± 0.2594 | 1.4854 ± 0.0181 | 22.7814 ± 0.0535 | 2.4964 ± 0.0518 |
| dopfn | 5.9479 ± 0.2176 | 0.6645 ± 0.0070 | 8.6852 ± 0.2182 | 6.6500 ± 0.5652 |
| uwyk1d | 9.9698 ± 0.5702 | 0.5144 ± 0.0060 | 16.6669 ± 0.2948 | 7.7232 ± 0.5554 |

| cpfn2d | 1.9848 ± 0.0650 | 0.2175 ± 0.0090 | 0.4971 ± 0.0493 | 2.3117 ± 0.4075 |
| dopfnbb | 2.7056 ± 0.1178 | 0.3131 ± 0.0098 | 2.0661 ± 0.0894 | 4.8153 ± 0.5451 |
| graph2d | 2.5015 ± 0.0814 | 0.2944 ± 0.0078 | 0.8998 ± 0.0706 | 5.2102 ± 0.5375 |

Density metrics — ACIC — tight T=4001 raw τ grid

(truth = analytic Gaussian N(μ_1(x)−μ_0(x), 2σ²); metrics per query averaged, then averaged across 10 realizations. Lower = better.)

| Method | NLL | L2 | KL_fwd (truth‖est) | KL_rev (est‖truth) |
|---|---|---|---|---|
| cpfn1d | 22.3898 ± 0.9737 | 1.3571 ± 0.1469 | 23.1929 ± 0.3732 | 3.5078 ± 0.2845 |
| dopfn | 9.7485 ± 2.5955 | 1.2462 ± 0.3861 | 9.7174 ± 1.1874 | 9.4779 ± 0.5601 |
| uwyk1d | 16.5535 ± 2.3400 | 0.8985 ± 0.2442 | 19.4712 ± 1.0398 | 7.7010 ± 0.6194 |

| cpfn2d | 2.2272 ± 0.0463 | 0.2762 ± 0.0084 | 0.5584 ± 0.0393 | 2.0673 ± 0.2704 |
| dopfnbb | 3.9890 ± 0.7423 | 0.3426 ± 0.0125 | 2.4494 ± 0.7791 | 4.3570 ± 0.8217 |
| graph2d | 2.5757 ± 0.1301 | 0.3079 ± 0.0218 | 0.9095 ± 0.1086 | 4.4882 ± 0.8081 |

Density metrics — CPS — tight T=4001 raw τ grid

(truth = empirical KDE from 100 RealCause CSVs (K=100 MC pairs per unit); metrics per query averaged, then averaged across 100 realizations. Lower = better.)

| Method | NLL | L2 | KL_fwd (truth‖est) | KL_rev (est‖truth) |
|---|---|---|---|---|
| cpfn1d | 13.7162 ± 0.1128 | 0.0156 ± 0.0004 | 14.5082 ± 0.0341 | 2.8089 ± 0.0377 |
| dopfn | 9.4196 ± 0.0626 | 0.0118 ± 0.0001 | 5.5525 ± 0.0842 | 1.9726 ± 0.0306 |
| uwyk1d | 8.5002 ± 0.0329 | 0.0160 ± 0.0003 | 1.2123 ± 0.0734 | 1.0183 ± 0.0225 |

| cpfn2d | 10.4545 ± 0.0112 | 0.0036 ± 0.0001 | 0.5163 ± 0.0198 | 0.6664 ± 0.0323 |
| dopfnbb | 10.8263 ± 0.0449 | 0.0046 ± 0.0001 | 1.9601 ± 0.0671 | 0.7886 ± 0.0353 |
| graph2d | 10.3579 ± 0.0119 | 0.0035 ± 0.0001 | 0.7844 ± 0.0250 | 0.5097 ± 0.0191 |

Density metrics — PSID — tight T=4001 raw τ grid

(truth = empirical KDE from 100 RealCause CSVs (K=100 MC pairs per unit); metrics per query averaged, then averaged across 100 realizations. Lower = better.)

| Method | NLL | L2 | KL_fwd (truth‖est) | KL_rev (est‖truth) |
|---|---|---|---|---|
| cpfn1d | 11.6050 ± 0.1555 | 0.0103 ± 0.0001 | 12.7674 ± 0.0547 | 2.0286 ± 0.0283 |
| dopfn | 8.9532 ± 0.0590 | 0.0076 ± 0.0001 | 3.2334 ± 0.0500 | 1.7742 ± 0.0339 |
| uwyk1d | 8.4939 ± 0.0729 | 0.0116 ± 0.0003 | 5.5216 ± 0.1465 | 1.2719 ± 0.0348 |

| cpfn2d | 10.6825 ± 0.0416 | 0.0020 ± 0.0001 | 0.3957 ± 0.0120 | 0.4052 ± 0.0222 |
| dopfnbb | 11.1019 ± 0.0678 | 0.0031 ± 0.0000 | 1.2824 ± 0.0537 | 0.8271 ± 0.0315 |
| graph2d | 11.0750 ± 0.0781 | 0.0025 ± 0.0000 | 0.9897 ± 0.0684 | 0.6570 ± 0.0370 |


Density metrics — PSID_bal — tight T=4001 raw τ grid

(truth = empirical KDE from 100 RealCause CSVs (K=100 MC pairs per unit); metrics per query averaged, then averaged across 100 realizations. Lower = better.)

| Method | NLL | L2 | KL_fwd (truth‖est) | KL_rev (est‖truth) |
|---|---|---|---|---|
| cpfn1d | 11.4682 ± 0.1555 | 0.0105 ± 0.0001 | 12.7736 ± 0.0558 | 2.0996 ± 0.0302 |
| dopfn | 8.9555 ± 0.0687 | 0.0090 ± 0.0001 | 3.9585 ± 0.0638 | 1.9623 ± 0.0363 |
| uwyk1d | 8.6702 ± 0.0686 | 0.0092 ± 0.0002 | 5.1508 ± 0.1611 | 1.1445 ± 0.0333 |

| cpfn2d | 10.7104 ± 0.0392 | 0.0022 ± 0.0001 | 0.4362 ± 0.0117 | 0.5383 ± 0.0280 |
| dopfnbb | 11.0083 ± 0.0634 | 0.0031 ± 0.0000 | 1.1124 ± 0.0509 | 0.8477 ± 0.0349 |
| graph2d | 10.7827 ± 0.0580 | 0.0025 ± 0.0000 | 0.6299 ± 0.0351 | 0.8321 ± 0.0366 |



============================================
=== IHDP — MALC K=1 B=500 ===
============================================

Density metrics — IHDP — tight T=4001 raw τ grid (1D source: MALC-smoothed (B500); 2D source: MALC-smoothed (B500))

(truth = analytic Gaussian N(μ_1(x)−μ_0(x), 2σ²); metrics per query averaged, then averaged across 100 realizations. Lower = better.)

| Method | NLL | L2 | KL_fwd (truth‖est) | KL_rev (est‖truth) | √PEHE (from density mean) |
|---|---|---|---|---|---|
| cpfn1d | 4.3206 ± 0.7125 | 0.1981 ± 0.0096 | 3.4281 ± 0.6834 | 0.3038 ± 0.0616 | 3.1555 ± 0.7154 |
| dopfn | 3.4938 ± 0.1150 | 0.3856 ± 0.0021 | 1.7533 ± 0.1143 | 7.8861 ± 0.3858 | 6.0986 ± 0.7327 |
| uwyk1d | 2.8593 ± 0.0759 | 0.3553 ± 0.0045 | 1.1951 ± 0.0701 | 6.5244 ± 0.5289 | 6.1429 ± 0.7417 |
| cpfn2d | 1.9848 ± 0.0650 | 0.2175 ± 0.0090 | 0.4971 ± 0.0493 | 2.3117 ± 0.4075 | 1.3127 ± 0.1455 |
| graph2d | 2.5015 ± 0.0814 | 0.2944 ± 0.0078 | 0.8998 ± 0.0706 | 5.2102 ± 0.5375 | 4.5347 ± 0.6271 |
| dopfnbb | 2.7056 ± 0.1178 | 0.3131 ± 0.0098 | 2.0661 ± 0.0894 | 4.8153 ± 0.5451 | 5.0967 ± 0.7061 |

============================================
=== ACIC — MALC K=1 B=500 ===
============================================

Density metrics — ACIC — tight T=4001 raw τ grid (1D source: MALC-smoothed (B500); 2D source: MALC-smoothed (B500))

(truth = analytic Gaussian N(μ_1(x)−μ_0(x), 2σ²); metrics per query averaged, then averaged across 10 realizations. Lower = better.)

| Method | NLL | L2 | KL_fwd (truth‖est) | KL_rev (est‖truth) | √PEHE (from density mean) |
|---|---|---|---|---|---|
| cpfn1d | 1.8178 ± 0.0872 | 0.1942 ± 0.0155 | 0.3337 ± 0.0545 | 1.0371 ± 0.2716 | 1.4149 ± 0.2337 |
| dopfn | 3.6789 ± 0.1707 | 0.3825 ± 0.0036 | 1.9281 ± 0.1711 | 8.3086 ± 0.8744 | 3.5764 ± 0.6890 |
| uwyk1d | 2.7462 ± 0.0984 | 0.3460 ± 0.0079 | 1.0374 ± 0.0929 | 5.5189 ± 0.7053 | 3.2497 ± 0.5746 |
| cpfn2d | 2.2272 ± 0.0463 | 0.2762 ± 0.0084 | 0.5584 ± 0.0393 | 2.0673 ± 0.2704 | 1.4436 ± 0.2404 |
| graph2d | 2.5757 ± 0.1301 | 0.3079 ± 0.0218 | 0.9095 ± 0.1086 | 4.4882 ± 0.8081 | 2.7876 ± 0.4988 |
| dopfnbb | 3.9890 ± 0.7423 | 0.3426 ± 0.0125 | 2.4494 ± 0.7791 | 4.3570 ± 0.8217 | 3.1876 ± 0.6253 |

============================================
=== CPS — MALC K=1 B=500 ===
============================================

Density metrics — CPS — tight T=4001 raw τ grid (1D source: MALC-smoothed (B500); 2D source: MALC-smoothed (B500))

(truth = empirical KDE from 100 RealCause CSVs (K=100 MC pairs per unit); metrics per query averaged, then averaged across 100 realizations. Lower = better.)

| Method | NLL | L2 | KL_fwd (truth‖est) | KL_rev (est‖truth) | √PEHE (from density mean) |
|---|---|---|---|---|---|
| cpfn1d | 10.4822 ± 0.0206 | 0.0036 ± 0.0001 | 0.5683 ± 0.0291 | 0.7247 ± 0.0357 | 7,707.84 ± 86.36 |
| dopfn | 10.9304 ± 0.1046 | 0.0039 ± 0.0001 | 0.9639 ± 0.1003 | 0.6036 ± 0.0214 | 7,333.51 ± 94.60 |
| uwyk1d | 10.3565 ± 0.0261 | 0.0038 ± 0.0001 | 1.0203 ± 0.0356 | 0.4918 ± 0.0137 | 7,580.95 ± 97.31 |
| cpfn2d | 10.4545 ± 0.0112 | 0.0036 ± 0.0001 | 0.5163 ± 0.0198 | 0.6664 ± 0.0323 | 7,849.96 ± 111.25 |
| graph2d | 10.3579 ± 0.0119 | 0.0035 ± 0.0001 | 0.7844 ± 0.0250 | 0.5097 ± 0.0191 | 7,415.24 ± 95.53 |
| dopfnbb | 10.8263 ± 0.0449 | 0.0046 ± 0.0001 | 1.9601 ± 0.0671 | 0.7886 ± 0.0353 | 7,948.49 ± 76.48 |

============================================
=== PSID — MALC K=1 B=500 ===
============================================

Density metrics — PSID — tight T=4001 raw τ grid (1D source: MALC-smoothed (B500); 2D source: MALC-smoothed (B500))

(truth = empirical KDE from 100 RealCause CSVs (K=100 MC pairs per unit); metrics per query averaged, then averaged across 100 realizations. Lower = better.)

| Method | NLL | L2 | KL_fwd (truth‖est) | KL_rev (est‖truth) | √PEHE (from density mean) |
|---|---|---|---|---|---|
| cpfn1d | 11.0618 ± 0.0856 | 0.0026 ± 0.0000 | 0.9355 ± 0.0777 | 0.5071 ± 0.0251 | 11,101.40 ± 547.12 |
| dopfn | 15.1491 ± 0.1840 | 0.0033 ± 0.0001 | 4.6814 ± 0.1782 | 0.5903 ± 0.0246 | 13,198.97 ± 617.05 |
| uwyk1d | 10.7127 ± 0.0740 | 0.0027 ± 0.0000 | 0.7807 ± 0.0521 | 0.6370 ± 0.0362 | 13,392.70 ± 654.35 |
| cpfn2d | 10.6825 ± 0.0416 | 0.0020 ± 0.0001 | 0.3957 ± 0.0120 | 0.4052 ± 0.0222 | 9,737.21 ± 510.10 |
| graph2d | 11.0750 ± 0.0781 | 0.0025 ± 0.0000 | 0.9897 ± 0.0684 | 0.6570 ± 0.0370 | 12,853.96 ± 641.67 |
| dopfnbb | 11.1019 ± 0.0678 | 0.0031 ± 0.0000 | 1.2824 ± 0.0537 | 0.8271 ± 0.0315 | 12,950.60 ± 524.44 |

============================================
=== PSID_bal — MALC K=1 B=500 ===
============================================

Density metrics — PSID_bal — tight T=4001 raw τ grid (1D source: MALC-smoothed (B500); 2D source: MALC-smoothed (B500))

(truth = empirical KDE from 100 RealCause CSVs (K=100 MC pairs per unit); metrics per query averaged, then averaged across 100 realizations. Lower = better.)

| Method | NLL | L2 | KL_fwd (truth‖est) | KL_rev (est‖truth) | √PEHE (from density mean) |
|---|---|---|---|---|---|
| cpfn1d | 10.8945 ± 0.0782 | 0.0023 ± 0.0000 | 0.8742 ± 0.0640 | 0.5109 ± 0.0251 | 11,448.98 ± 531.62 |
| dopfn | 13.9115 ± 0.1513 | 0.0030 ± 0.0000 | 3.5309 ± 0.1481 | 0.6742 ± 0.0288 | 13,908.51 ± 631.97 |
| uwyk1d | 10.7413 ± 0.0697 | 0.0026 ± 0.0000 | 0.7988 ± 0.0562 | 0.6057 ± 0.0346 | 13,034.73 ± 648.00 |
| cpfn2d | 10.7104 ± 0.0392 | 0.0022 ± 0.0001 | 0.4362 ± 0.0117 | 0.5383 ± 0.0280 | 11,410.50 ± 517.07 |
| graph2d | 10.7827 ± 0.0580 | 0.0025 ± 0.0000 | 0.6299 ± 0.0351 | 0.8321 ± 0.0366 | 12,360.78 ± 618.84 |
| dopfnbb | 11.0083 ± 0.0634 | 0.0031 ± 0.0000 | 1.1124 ± 0.0509 | 0.8477 ± 0.0349 | 13,019.51 ± 530.65 |

============================================
=== IHDP — MALC max_K=3 B=500 ===
============================================

Density metrics — IHDP — tight T=4001 raw τ grid (1D source: MALC-smoothed (B500_maxK3); 2D source: MALC-smoothed (B500_maxK3))

(truth = analytic Gaussian N(μ_1(x)−μ_0(x), 2σ²); metrics per query averaged, then averaged across 100 realizations. Lower = better.)

| Method | NLL | L2 | KL_fwd (truth‖est) | KL_rev (est‖truth) | √PEHE (from density mean) |
|---|---|---|---|---|---|
| cpfn1d | 3.6902 ± 0.7358 | 0.2049 ± 0.0110 | 2.7481 ± 0.7078 | 0.6622 ± 0.1152 | 3.1658 ± 0.9456 |
| dopfn | 3.1738 ± 0.0576 | 0.3863 ± 0.0022 | 1.4343 ± 0.0569 | 8.1575 ± 0.4253 | 6.2355 ± 0.7417 |
| uwyk1d | — | — | — | — | — |
| cpfn2d | 1.9888 ± 0.0650 | 0.2184 ± 0.0089 | 0.5013 ± 0.0493 | 2.3530 ± 0.4062 | 1.4921 ± 0.1913 |
| graph2d | 2.5906 ± 0.0820 | 0.3102 ± 0.0073 | 0.9800 ± 0.0714 | 5.9909 ± 0.5546 | 5.4359 ± 0.7137 |
| dopfnbb | 2.5484 ± 0.1089 | 0.3141 ± 0.0096 | 1.7925 ± 0.0740 | 5.0457 ± 0.5410 | 5.3404 ± 0.7542 |

============================================
=== ACIC — MALC max_K=3 B=500 ===
============================================

Density metrics — ACIC — tight T=4001 raw τ grid (1D source: MALC-smoothed (B500_maxK3); 2D source: MALC-smoothed (B500_maxK3))

(truth = analytic Gaussian N(μ_1(x)−μ_0(x), 2σ²); metrics per query averaged, then averaged across 10 realizations. Lower = better.)

| Method | NLL | L2 | KL_fwd (truth‖est) | KL_rev (est‖truth) | √PEHE (from density mean) |
|---|---|---|---|---|---|
| cpfn1d | 1.7466 ± 0.0721 | 0.1872 ± 0.0141 | 0.3002 ± 0.0473 | 0.9099 ± 0.2223 | 1.2838 ± 0.1840 |
| dopfn | 3.1506 ± 0.1163 | 0.3811 ± 0.0045 | 1.4069 ± 0.1147 | 8.2253 ± 0.8959 | 3.7820 ± 0.6620 |
| uwyk1d | — | — | — | — | — |
| cpfn2d | 2.2372 ± 0.0461 | 0.2772 ± 0.0083 | 0.5686 ± 0.0394 | 2.2145 ± 0.2740 | 1.5652 ± 0.2574 |
| graph2d | 2.5826 ± 0.1423 | 0.3140 ± 0.0222 | 0.9103 ± 0.1195 | 5.0506 ± 0.9085 | 2.9450 ± 0.5233 |
| dopfnbb | 3.6550 ± 0.6448 | 0.3384 ± 0.0128 | 2.1093 ± 0.6885 | 4.3696 ± 0.8444 | 3.2095 ± 0.6480 |

============================================
=== CPS — MALC max_K=3 B=500 ===
============================================

Density metrics — CPS — tight T=4001 raw τ grid (1D source: MALC-smoothed (B500_maxK3); 2D source: MALC-smoothed (B500_maxK3))

(truth = empirical KDE from 100 RealCause CSVs (K=100 MC pairs per unit); metrics per query averaged, then averaged across 100 realizations. Lower = better.)

| Method | NLL | L2 | KL_fwd (truth‖est) | KL_rev (est‖truth) | √PEHE (from density mean) |
|---|---|---|---|---|---|
| cpfn1d | 10.5665 ± — | 0.0033 ± — | 0.3663 ± — | 0.5942 ± — | 8,864.78 ± — |
| dopfn | 10.9136 ± 0.0673 | 0.0039 ± 0.0002 | 1.0528 ± 0.0955 | 0.7134 ± 0.0545 | 8,168.87 ± 255.38 |
| uwyk1d | — | — | — | — | — |
| cpfn2d | 10.4899 ± 0.0213 | 0.0036 ± 0.0002 | 0.4899 ± 0.0336 | 0.7195 ± 0.0601 | 8,076.52 ± 188.81 |
| graph2d | 10.3098 ± 0.0283 | 0.0034 ± 0.0001 | 0.7347 ± 0.0469 | 0.4571 ± 0.0346 | 7,335.33 ± 222.74 |
| dopfnbb | 10.7028 ± 0.0777 | 0.0046 ± 0.0002 | 1.7029 ± 0.1470 | 1.1031 ± 0.0859 | 7,914.15 ± 233.68 |

============================================
=== PSID — MALC max_K=3 B=500 ===
============================================

Density metrics — PSID — tight T=4001 raw τ grid (1D source: MALC-smoothed (B500_maxK3); 2D source: MALC-smoothed (B500_maxK3))

(truth = empirical KDE from 100 RealCause CSVs (K=100 MC pairs per unit); metrics per query averaged, then averaged across 100 realizations. Lower = better.)

| Method | NLL | L2 | KL_fwd (truth‖est) | KL_rev (est‖truth) | √PEHE (from density mean) |
|---|---|---|---|---|---|
| cpfn1d | 10.7579 ± — | 0.0023 ± — | 0.2326 ± — | 0.6290 ± — | 8,510.46 ± — |
| dopfn | 11.1665 ± 0.0597 | 0.0028 ± 0.0000 | 0.7735 ± 0.0519 | 1.0605 ± 0.0407 | 14,164.74 ± 581.54 |
| uwyk1d | — | — | — | — | — |
| cpfn2d | 10.7252 ± 0.0371 | 0.0022 ± 0.0000 | 0.4193 ± 0.0111 | 0.6644 ± 0.0324 | 11,494.87 ± 514.77 |
| graph2d | 10.8093 ± 0.0511 | 0.0025 ± 0.0000 | 0.6140 ± 0.0293 | 0.7277 ± 0.0373 | 12,916.34 ± 637.91 |
| dopfnbb | 10.9721 ± 0.0607 | 0.0032 ± 0.0000 | 1.0361 ± 0.0416 | 1.1602 ± 0.0404 | 13,976.78 ± 632.36 |

============================================
=== PSID_bal — MALC max_K=3 B=500 ===
============================================

Density metrics — PSID_bal — tight T=4001 raw τ grid (1D source: MALC-smoothed (B500_maxK3); 2D source: MALC-smoothed (B500_maxK3))

(truth = empirical KDE from 100 RealCause CSVs (K=100 MC pairs per unit); metrics per query averaged, then averaged across 100 realizations. Lower = better.)

| Method | NLL | L2 | KL_fwd (truth‖est) | KL_rev (est‖truth) | √PEHE (from density mean) |
|---|---|---|---|---|---|
| cpfn1d | 10.3083 ± 0.0892 | 0.0026 ± 0.0002 | 0.7195 ± 0.1890 | 0.4678 ± 0.1444 | 10,060.27 ± 1,511.62 |
| dopfn | 10.9042 ± 0.0627 | 0.0030 ± 0.0000 | 0.8222 ± 0.0484 | 1.0704 ± 0.0381 | 15,124.89 ± 664.26 |
| uwyk1d | — | — | — | — | — |
| cpfn2d | 10.8206 ± 0.0447 | 0.0024 ± 0.0001 | 0.4932 ± 0.0125 | 1.0277 ± 0.0572 | 15,129.65 ± 577.65 |
| graph2d | 10.7847 ± 0.0394 | 0.0028 ± 0.0000 | 0.5081 ± 0.0145 | 0.9977 ± 0.0404 | 12,661.79 ± 605.24 |
| dopfnbb | 10.9045 ± 0.0509 | 0.0032 ± 0.0000 | 0.9333 ± 0.0358 | 1.1714 ± 0.0403 | 13,724.12 ± 577.11 |
(venv) furkanbd@klogin01:/scratch/furkanbd/rpfn_bench_kit/R-PFN$ 




# 1. rebuild the CSV with SEM columns (same npz pass, no re-eval)
python $REPO/case_study/cluster/dsweep_report.py --root $RES --out $RES/dsweep.csv

# 2. your 8 tables at N=1000 — each d gets a PEHE and an L1-ATE block (mean±SEM)
python $REPO/case_study/cluster/dsweep_pivot.py --csv $RES/dsweep.csv \
    --shift shift-2 --n 500 --panels d --readout raw

# 2. your 8 tables at N=1000 — each d gets a PEHE and an L1-ATE block (mean±SEM)
python $REPO/case_study/cluster/dsweep_pivot.py --csv $RES/dsweep.csv \
    --shift shift+2 --n 500 --panels d --readout raw

# 2. your 8 tables at N=1000 — each d gets a PEHE and an L1-ATE block (mean±SEM)
python $REPO/case_study/cluster/dsweep_pivot.py --csv $RES/dsweep.csv \
    --shift shift-5 --n 500 --panels d --readout raw

# 2. your 8 tables at N=1000 — each d gets a PEHE and an L1-ATE block (mean±SEM)
python $REPO/case_study/cluster/dsweep_pivot.py --csv $RES/dsweep.csv \
    --shift shift+2 --n 500 --panels d --readout raw



  scp -r furkanbd@killarney.alliancecan.ca:/scratch/furkanbd/cmech_v3/figures ~/Downloads/cmech_figures
open ~/Downloads/cmech_figures


scp -r furkanbd@killarney.alliancecan.ca:/scratch/furkanbd/cmech_v3/figures ~/Downloads/cmech_figures
scp 'furkanbd@killarney.alliancecan.ca:/scratch/furkanbd/rpfn_bench_kit/results_case_study/dvar/fig3_cen3_*.png' ~/Downloads/



# ATE density calibration — d=5, N=1000, total, 1D coupling = indep

Scored object: ATE density per dataset (Wasserstein barycenter of its per-query CATE densities), scored against that dataset's realized ATE.

2D heads: diagonal projection of the joint. 1D heads: independence
convolution of the two arm marginals. Raw densities, no MALC.

coverage95 should be ~0.95; below means over-confident, above means
the intervals are wider than they need to be. Read it WITH length —
a wide interval buys coverage for free. CRPS and WIS are proper, so
lower is better on both and they penalise that trade-off.

`n_files` counts npz files scored, not distinct datasets: under
`--subset total` each dataset contributes up to two (its zero and its
non-zero queries), so n_files exceeds the realization count. `n_query`
is the honest total and is exact — the subsets are disjoint and
together complete.

`sd_ratio` = mean predictive sd / sd of the true tau. A UNITS CHECK:
a density dumped on the wrong scale still scores finitely, it just
looks like a bad model. Values near 1 are well-calibrated in spread;
a ratio in the tens means the density is on the wrong axis, not that
the model is that much worse. `bias` is mean(E[tau]) - tau_true.

| method | n_files | n_query | coverage95 | length | crps | wis | pred_sd | true_sd | sd_ratio | bias |
|---|---|---|---|---|---|---|---|---|---|---|
| dopfn_native | 100 | 100 | 0.830 | 21.8086 ± 2.2462 | 7.1885 ± 1.3682 | 6.7409 ± 1.3201 | 5.5074 | 0.5815 | 9.5 | +1.1835 |
| dopfn_bb | 100 | 100 | 0.840 | 0.8918 ± 0.1132 | 0.1572 ± 0.0304 | 0.1468 ± 0.0292 | 0.2672 | 0.5815 | 0.5 | -0.0751 |
| uwyk1d-noanc | 100 | 100 | 0.880 | 0.9694 ± 0.0796 | 0.2155 ± 0.0383 | 0.1987 ± 0.0363 | 0.2594 | 0.5815 | 0.4 | -0.0448 |
| uwyk1d-v3a | 100 | 100 | 0.920 | 0.9340 ± 0.0614 | 0.1373 ± 0.0353 | 0.1273 ± 0.0331 | 0.2454 | 0.5815 | 0.4 | -0.0261 |
| graph2d-noanc | 100 | 100 | 0.950 | 0.8525 ± 0.0813 | 0.1092 ± 0.0288 | 0.1005 ± 0.0264 | 0.2339 | 0.5815 | 0.4 | -0.0252 |
| graph2d-v3a | 100 | 100 | 0.940 | 0.7231 ± 0.0708 | 0.1103 ± 0.0296 | 0.1019 ± 0.0276 | 0.2041 | 0.5815 | 0.4 | -0.0273 |
| cpfn1d | 100 | 100 | 0.920 | 0.7687 ± 0.0842 | 0.1322 ± 0.0346 | 0.1247 ± 0.0337 | 0.1984 | 0.5815 | 0.3 | -0.0165 |
| cpfn2d | 100 | 100 | 0.940 | 0.9709 ± 0.0750 | 0.1397 ± 0.0321 | 0.1292 ± 0.0305 | 0.2560 | 0.5815 | 0.4 | -0.0027 |


# ATE density calibration — d=10, N=1000, total, 1D coupling = indep

Scored object: ATE density per dataset (Wasserstein barycenter of its per-query CATE densities), scored against that dataset's realized ATE.

2D heads: diagonal projection of the joint. 1D heads: independence
convolution of the two arm marginals. Raw densities, no MALC.

coverage95 should be ~0.95; below means over-confident, above means
the intervals are wider than they need to be. Read it WITH length —
a wide interval buys coverage for free. CRPS and WIS are proper, so
lower is better on both and they penalise that trade-off.

`n_files` counts npz files scored, not distinct datasets: under
`--subset total` each dataset contributes up to two (its zero and its
non-zero queries), so n_files exceeds the realization count. `n_query`
is the honest total and is exact — the subsets are disjoint and
together complete.

`sd_ratio` = mean predictive sd / sd of the true tau. A UNITS CHECK:
a density dumped on the wrong scale still scores finitely, it just
looks like a bad model. Values near 1 are well-calibrated in spread;
a ratio in the tens means the density is on the wrong axis, not that
the model is that much worse. `bias` is mean(E[tau]) - tau_true.

| method | n_files | n_query | coverage95 | length | crps | wis | pred_sd | true_sd | sd_ratio | bias |
|---|---|---|---|---|---|---|---|---|---|---|
| dopfn_native | 100 | 100 | 0.940 | 25.0160 ± 2.7082 | 3.6473 ± 0.7411 | 3.3335 ± 0.6867 | 6.2256 | 0.2915 | 21.4 | -1.5941 |
| dopfn_bb | 100 | 100 | 0.950 | 0.9544 ± 0.1063 | 0.0997 ± 0.0139 | 0.0898 ± 0.0123 | 0.2618 | 0.2915 | 0.9 | -0.0215 |
| uwyk1d-noanc | 100 | 100 | 0.910 | 0.7296 ± 0.0586 | 0.1001 ± 0.0194 | 0.0907 ± 0.0175 | 0.1995 | 0.2915 | 0.7 | +0.0265 |
| uwyk1d-v3a | 100 | 100 | 0.990 | 0.8438 ± 0.0615 | 0.0667 ± 0.0122 | 0.0605 ± 0.0107 | 0.2201 | 0.2915 | 0.8 | +0.0113 |
| graph2d-noanc | 100 | 100 | 1.000 | 0.7894 ± 0.0734 | 0.0485 ± 0.0049 | 0.0447 ± 0.0044 | 0.2091 | 0.2915 | 0.7 | -0.0026 |
| graph2d-v3a | 100 | 100 | 0.990 | 0.5900 ± 0.0567 | 0.0440 ± 0.0052 | 0.0402 ± 0.0046 | 0.1598 | 0.2915 | 0.5 | +0.0042 |
| cpfn1d | 100 | 100 | 0.960 | 0.6707 ± 0.0705 | 0.0689 ± 0.0117 | 0.0624 ± 0.0104 | 0.1707 | 0.2915 | 0.6 | +0.0154 |
| cpfn2d | 100 | 100 | 0.990 | 0.9489 ± 0.0643 | 0.0760 ± 0.0079 | 0.0690 ± 0.0071 | 0.2496 | 0.2915 | 0.9 | -0.0024 |


# ATE density calibration — d=20, N=1000, total, 1D coupling = indep

Scored object: ATE density per dataset (Wasserstein barycenter of its per-query CATE densities), scored against that dataset's realized ATE.

2D heads: diagonal projection of the joint. 1D heads: independence
convolution of the two arm marginals. Raw densities, no MALC.

coverage95 should be ~0.95; below means over-confident, above means
the intervals are wider than they need to be. Read it WITH length —
a wide interval buys coverage for free. CRPS and WIS are proper, so
lower is better on both and they penalise that trade-off.

`n_files` counts npz files scored, not distinct datasets: under
`--subset total` each dataset contributes up to two (its zero and its
non-zero queries), so n_files exceeds the realization count. `n_query`
is the honest total and is exact — the subsets are disjoint and
together complete.

`sd_ratio` = mean predictive sd / sd of the true tau. A UNITS CHECK:
a density dumped on the wrong scale still scores finitely, it just
looks like a bad model. Values near 1 are well-calibrated in spread;
a ratio in the tens means the density is on the wrong axis, not that
the model is that much worse. `bias` is mean(E[tau]) - tau_true.

| method | n_files | n_query | coverage95 | length | crps | wis | pred_sd | true_sd | sd_ratio | bias |
|---|---|---|---|---|---|---|---|---|---|---|
| dopfn_native | 100 | 100 | 0.990 | 27.2257 ± 2.6044 | 1.8366 ± 0.2283 | 1.6875 ± 0.2136 | 6.8115 | 0.2124 | 32.1 | -0.1019 |
| dopfn_bb | 100 | 100 | 0.980 | 1.1242 ± 0.1096 | 0.1035 ± 0.0144 | 0.0932 ± 0.0128 | 0.3123 | 0.2124 | 1.5 | -0.0087 |
| uwyk1d-noanc | 100 | 100 | 0.880 | 0.6822 ± 0.0644 | 0.0963 ± 0.0145 | 0.0878 ± 0.0132 | 0.1851 | 0.2124 | 0.9 | +0.0270 |
| uwyk1d-v3a | 100 | 100 | 0.960 | 0.7829 ± 0.0608 | 0.0618 ± 0.0075 | 0.0561 ± 0.0066 | 0.2084 | 0.2124 | 1.0 | +0.0180 |
| graph2d-noanc | 100 | 100 | 0.990 | 0.8250 ± 0.0719 | 0.0555 ± 0.0074 | 0.0507 ± 0.0066 | 0.2126 | 0.2124 | 1.0 | +0.0032 |
| graph2d-v3a | 100 | 100 | 0.980 | 0.5931 ± 0.0552 | 0.0481 ± 0.0063 | 0.0440 ± 0.0057 | 0.1626 | 0.2124 | 0.8 | +0.0127 |
| cpfn1d | 100 | 100 | 0.910 | 0.6017 ± 0.0659 | 0.0627 ± 0.0091 | 0.0570 ± 0.0081 | 0.1511 | 0.2124 | 0.7 | +0.0227 |
| cpfn2d | 100 | 100 | 0.990 | 0.9734 ± 0.0752 | 0.0766 ± 0.0072 | 0.0694 ± 0.0064 | 0.2556 | 0.2124 | 1.2 | +0.0185 |


# ATE density calibration — d=30, N=1000, total, 1D coupling = indep

Scored object: ATE density per dataset (Wasserstein barycenter of its per-query CATE densities), scored against that dataset's realized ATE.

2D heads: diagonal projection of the joint. 1D heads: independence
convolution of the two arm marginals. Raw densities, no MALC.

coverage95 should be ~0.95; below means over-confident, above means
the intervals are wider than they need to be. Read it WITH length —
a wide interval buys coverage for free. CRPS and WIS are proper, so
lower is better on both and they penalise that trade-off.

`n_files` counts npz files scored, not distinct datasets: under
`--subset total` each dataset contributes up to two (its zero and its
non-zero queries), so n_files exceeds the realization count. `n_query`
is the honest total and is exact — the subsets are disjoint and
together complete.

`sd_ratio` = mean predictive sd / sd of the true tau. A UNITS CHECK:
a density dumped on the wrong scale still scores finitely, it just
looks like a bad model. Values near 1 are well-calibrated in spread;
a ratio in the tens means the density is on the wrong axis, not that
the model is that much worse. `bias` is mean(E[tau]) - tau_true.

| method | n_files | n_query | coverage95 | length | crps | wis | pred_sd | true_sd | sd_ratio | bias |
|---|---|---|---|---|---|---|---|---|---|---|
| dopfn_native | 100 | 100 | 1.000 | 34.3646 ± 3.6642 | 1.8460 ± 0.2230 | 1.7027 ± 0.2034 | 8.4279 | 0.1878 | 44.9 | +0.4516 |
| dopfn_bb | 100 | 100 | 0.960 | 1.0908 ± 0.1170 | 0.1145 ± 0.0173 | 0.1028 ± 0.0152 | 0.3128 | 0.1878 | 1.7 | -0.0038 |
| uwyk1d-noanc | 100 | 100 | 0.940 | 0.7198 ± 0.0714 | 0.0697 ± 0.0115 | 0.0636 ± 0.0104 | 0.1981 | 0.1878 | 1.1 | -0.0119 |
| uwyk1d-v3a | 100 | 100 | 0.950 | 0.7351 ± 0.0685 | 0.0570 ± 0.0085 | 0.0522 ± 0.0076 | 0.1992 | 0.1878 | 1.1 | -0.0027 |
| graph2d-noanc | 100 | 100 | 1.000 | 0.7281 ± 0.0753 | 0.0508 ± 0.0071 | 0.0463 ± 0.0063 | 0.1934 | 0.1878 | 1.0 | +0.0017 |
| graph2d-v3a | 100 | 100 | 0.980 | 0.5069 ± 0.0544 | 0.0479 ± 0.0079 | 0.0438 ± 0.0070 | 0.1368 | 0.1878 | 0.7 | -0.0016 |
| cpfn1d | 100 | 100 | 0.960 | 0.5044 ± 0.0499 | 0.0506 ± 0.0096 | 0.0462 ± 0.0087 | 0.1291 | 0.1878 | 0.7 | +0.0076 |
| cpfn2d | 100 | 100 | 1.000 | 0.9389 ± 0.0676 | 0.0675 ± 0.0064 | 0.0615 ± 0.0058 | 0.2467 | 0.1878 | 1.3 | +0.0178 |


# ATE density calibration — d=40, N=1000, total, 1D coupling = indep

Scored object: ATE density per dataset (Wasserstein barycenter of its per-query CATE densities), scored against that dataset's realized ATE.

2D heads: diagonal projection of the joint. 1D heads: independence
convolution of the two arm marginals. Raw densities, no MALC.

coverage95 should be ~0.95; below means over-confident, above means
the intervals are wider than they need to be. Read it WITH length —
a wide interval buys coverage for free. CRPS and WIS are proper, so
lower is better on both and they penalise that trade-off.

`n_files` counts npz files scored, not distinct datasets: under
`--subset total` each dataset contributes up to two (its zero and its
non-zero queries), so n_files exceeds the realization count. `n_query`
is the honest total and is exact — the subsets are disjoint and
together complete.

`sd_ratio` = mean predictive sd / sd of the true tau. A UNITS CHECK:
a density dumped on the wrong scale still scores finitely, it just
looks like a bad model. Values near 1 are well-calibrated in spread;
a ratio in the tens means the density is on the wrong axis, not that
the model is that much worse. `bias` is mean(E[tau]) - tau_true.

| method | n_files | n_query | coverage95 | length | crps | wis | pred_sd | true_sd | sd_ratio | bias |
|---|---|---|---|---|---|---|---|---|---|---|
| dopfn_native | 100 | 100 | 1.000 | 28.0108 ± 2.5813 | 1.4204 ± 0.1337 | 1.3120 ± 0.1228 | 6.8687 | 0.1233 | 55.7 | -0.1585 |
| dopfn_bb | 100 | 100 | 0.990 | 0.9058 ± 0.1082 | 0.0772 ± 0.0120 | 0.0698 ± 0.0107 | 0.2582 | 0.1233 | 2.1 | +0.0068 |
| uwyk1d-noanc | 100 | 100 | 0.980 | 0.6526 ± 0.0559 | 0.0537 ± 0.0094 | 0.0498 ± 0.0090 | 0.1823 | 0.1233 | 1.5 | +0.0184 |
| uwyk1d-v3a | 100 | 100 | 0.980 | 0.6807 ± 0.0568 | 0.0478 ± 0.0085 | 0.0441 ± 0.0079 | 0.1855 | 0.1233 | 1.5 | +0.0229 |
| graph2d-noanc | 100 | 100 | 1.000 | 0.6675 ± 0.0603 | 0.0370 ± 0.0046 | 0.0344 ± 0.0042 | 0.1776 | 0.1233 | 1.4 | +0.0003 |
| graph2d-v3a | 100 | 100 | 1.000 | 0.5162 ± 0.0512 | 0.0319 ± 0.0040 | 0.0295 ± 0.0036 | 0.1378 | 0.1233 | 1.1 | +0.0123 |
| cpfn1d | 100 | 100 | 0.940 | 0.5036 ± 0.0491 | 0.0451 ± 0.0079 | 0.0413 ± 0.0072 | 0.1282 | 0.1233 | 1.0 | +0.0170 |
| cpfn2d | 100 | 100 | 1.000 | 0.8336 ± 0.0560 | 0.0576 ± 0.0055 | 0.0524 ± 0.0050 | 0.2188 | 0.1233 | 1.8 | +0.0084 |


# ATE density calibration — d=50, N=1000, total, 1D coupling = indep

Scored object: ATE density per dataset (Wasserstein barycenter of its per-query CATE densities), scored against that dataset's realized ATE.

2D heads: diagonal projection of the joint. 1D heads: independence
convolution of the two arm marginals. Raw densities, no MALC.

coverage95 should be ~0.95; below means over-confident, above means
the intervals are wider than they need to be. Read it WITH length —
a wide interval buys coverage for free. CRPS and WIS are proper, so
lower is better on both and they penalise that trade-off.

`n_files` counts npz files scored, not distinct datasets: under
`--subset total` each dataset contributes up to two (its zero and its
non-zero queries), so n_files exceeds the realization count. `n_query`
is the honest total and is exact — the subsets are disjoint and
together complete.

`sd_ratio` = mean predictive sd / sd of the true tau. A UNITS CHECK:
a density dumped on the wrong scale still scores finitely, it just
looks like a bad model. Values near 1 are well-calibrated in spread;
a ratio in the tens means the density is on the wrong axis, not that
the model is that much worse. `bias` is mean(E[tau]) - tau_true.

| method | n_files | n_query | coverage95 | length | crps | wis | pred_sd | true_sd | sd_ratio | bias |
|---|---|---|---|---|---|---|---|---|---|---|
| dopfn_native | 100 | 100 | 1.000 | 32.0361 ± 3.1107 | 1.6097 ± 0.1698 | 1.4849 ± 0.1552 | 7.9493 | 0.1494 | 53.2 | -0.0859 |
| dopfn_bb | 100 | 100 | 1.000 | 0.9552 ± 0.1127 | 0.0848 ± 0.0137 | 0.0764 ± 0.0122 | 0.2841 | 0.1494 | 1.9 | -0.0137 |
| uwyk1d-noanc | 100 | 100 | 0.980 | 0.7680 ± 0.0735 | 0.0562 ± 0.0121 | 0.0520 ± 0.0114 | 0.2044 | 0.1494 | 1.4 | +0.0034 |
| uwyk1d-v3a | 100 | 100 | 0.980 | 0.7358 ± 0.0739 | 0.0522 ± 0.0115 | 0.0482 ± 0.0107 | 0.1933 | 0.1494 | 1.3 | +0.0035 |
| graph2d-noanc | 100 | 100 | 1.000 | 0.6713 ± 0.0804 | 0.0353 ± 0.0055 | 0.0330 ± 0.0050 | 0.1759 | 0.1494 | 1.2 | +0.0033 |
| graph2d-v3a | 100 | 100 | 1.000 | 0.5537 ± 0.0689 | 0.0329 ± 0.0057 | 0.0303 ± 0.0051 | 0.1453 | 0.1494 | 1.0 | +0.0027 |
| cpfn1d | 100 | 100 | 0.970 | 0.5731 ± 0.0760 | 0.0385 ± 0.0080 | 0.0354 ± 0.0073 | 0.1392 | 0.1494 | 0.9 | +0.0023 |
| cpfn2d | 100 | 100 | 1.000 | 0.9394 ± 0.0745 | 0.0616 ± 0.0069 | 0.0560 ± 0.0062 | 0.2447 | 0.1494 | 1.6 | +0.0086 |
# CATE density calibration — d=5, N=1000, total, 1D coupling = indep

Scored object: CATE density per query, scored against that query's true tau.

2D heads: diagonal projection of the joint. 1D heads: independence
convolution of the two arm marginals. Raw densities, no MALC.

coverage95 should be ~0.95; below means over-confident, above means
the intervals are wider than they need to be. Read it WITH length —
a wide interval buys coverage for free. CRPS and WIS are proper, so
lower is better on both and they penalise that trade-off.

`n_files` counts npz files scored, not distinct datasets: under
`--subset total` each dataset contributes up to two (its zero and its
non-zero queries), so n_files exceeds the realization count. `n_query`
is the honest total and is exact — the subsets are disjoint and
together complete.

`sd_ratio` = mean predictive sd / sd of the true tau. A UNITS CHECK:
a density dumped on the wrong scale still scores finitely, it just
looks like a bad model. Values near 1 are well-calibrated in spread;
a ratio in the tens means the density is on the wrong axis, not that
the model is that much worse. `bias` is mean(E[tau]) - tau_true.

| method | n_files | n_query | coverage95 | length | crps | wis | pred_sd | true_sd | sd_ratio | bias |
|---|---|---|---|---|---|---|---|---|---|---|
| dopfn_native | 119 | 100000 | 0.799 | 21.3247 ± 0.0768 | 7.3959 ± 0.0439 | 6.9805 ± 0.0426 | 5.5003 | 0.6144 | 9.0 | +1.1833 |
| dopfn_bb | 119 | 100000 | 0.729 | 0.8798 ± 0.0036 | 0.1936 ± 0.0010 | 0.1819 ± 0.0010 | 0.2692 | 0.6144 | 0.4 | -0.0751 |
| uwyk1d-noanc | 119 | 100000 | 0.830 | 0.9653 ± 0.0028 | 0.2406 ± 0.0013 | 0.2221 ± 0.0012 | 0.2688 | 0.6144 | 0.4 | -0.0452 |
| uwyk1d-v3a | 119 | 100000 | 0.887 | 0.9344 ± 0.0023 | 0.1625 ± 0.0012 | 0.1499 ± 0.0011 | 0.2582 | 0.6144 | 0.4 | -0.0260 |
| graph2d-noanc | 119 | 100000 | 0.769 | 0.8014 ± 0.0027 | 0.1299 ± 0.0010 | 0.1188 ± 0.0009 | 0.2291 | 0.6144 | 0.4 | -0.0241 |
| graph2d-v3a | 119 | 100000 | 0.739 | 0.6657 ± 0.0024 | 0.1315 ± 0.0010 | 0.1206 ± 0.0010 | 0.1975 | 0.6144 | 0.3 | -0.0262 |
| cpfn1d | 119 | 100000 | 0.818 | 0.7678 ± 0.0029 | 0.1629 ± 0.0011 | 0.1526 ± 0.0011 | 0.2042 | 0.6144 | 0.3 | -0.0165 |
| cpfn2d | 119 | 100000 | 0.890 | 0.8365 ± 0.0022 | 0.1651 ± 0.0011 | 0.1525 ± 0.0011 | 0.2174 | 0.6144 | 0.4 | -0.0029 |


# CATE density calibration — d=10, N=1000, total, 1D coupling = indep

Scored object: CATE density per query, scored against that query's true tau.

2D heads: diagonal projection of the joint. 1D heads: independence
convolution of the two arm marginals. Raw densities, no MALC.

coverage95 should be ~0.95; below means over-confident, above means
the intervals are wider than they need to be. Read it WITH length —
a wide interval buys coverage for free. CRPS and WIS are proper, so
lower is better on both and they penalise that trade-off.

`n_files` counts npz files scored, not distinct datasets: under
`--subset total` each dataset contributes up to two (its zero and its
non-zero queries), so n_files exceeds the realization count. `n_query`
is the honest total and is exact — the subsets are disjoint and
together complete.

`sd_ratio` = mean predictive sd / sd of the true tau. A UNITS CHECK:
a density dumped on the wrong scale still scores finitely, it just
looks like a bad model. Values near 1 are well-calibrated in spread;
a ratio in the tens means the density is on the wrong axis, not that
the model is that much worse. `bias` is mean(E[tau]) - tau_true.

| method | n_files | n_query | coverage95 | length | crps | wis | pred_sd | true_sd | sd_ratio | bias |
|---|---|---|---|---|---|---|---|---|---|---|
| dopfn_native | 132 | 100000 | 0.873 | 24.6022 ± 0.0913 | 3.8523 ± 0.0260 | 3.5342 ± 0.0241 | 6.3648 | 0.3768 | 16.9 | -1.5938 |
| dopfn_bb | 132 | 100000 | 0.814 | 0.9459 ± 0.0034 | 0.1504 ± 0.0007 | 0.1379 ± 0.0006 | 0.2645 | 0.3768 | 0.7 | -0.0215 |
| uwyk1d-noanc | 132 | 100000 | 0.844 | 0.7240 ± 0.0023 | 0.1369 ± 0.0008 | 0.1249 ± 0.0008 | 0.2082 | 0.3768 | 0.6 | +0.0265 |
| uwyk1d-v3a | 132 | 100000 | 0.940 | 0.8403 ± 0.0024 | 0.1032 ± 0.0006 | 0.0931 ± 0.0006 | 0.2316 | 0.3768 | 0.6 | +0.0114 |
| graph2d-noanc | 132 | 100000 | 0.885 | 0.7466 ± 0.0026 | 0.0804 ± 0.0005 | 0.0727 ± 0.0004 | 0.2065 | 0.3768 | 0.5 | -0.0028 |
| graph2d-v3a | 132 | 100000 | 0.832 | 0.5306 ± 0.0020 | 0.0796 ± 0.0005 | 0.0720 ± 0.0005 | 0.1533 | 0.3768 | 0.4 | +0.0042 |
| cpfn1d | 132 | 100000 | 0.870 | 0.6702 ± 0.0026 | 0.1095 ± 0.0007 | 0.0997 ± 0.0006 | 0.1804 | 0.3768 | 0.5 | +0.0154 |
| cpfn2d | 132 | 100000 | 0.935 | 0.7989 ± 0.0021 | 0.1086 ± 0.0006 | 0.0982 ± 0.0006 | 0.2078 | 0.3768 | 0.6 | -0.0024 |


# CATE density calibration — d=20, N=1000, total, 1D coupling = indep

Scored object: CATE density per query, scored against that query's true tau.

2D heads: diagonal projection of the joint. 1D heads: independence
convolution of the two arm marginals. Raw densities, no MALC.

coverage95 should be ~0.95; below means over-confident, above means
the intervals are wider than they need to be. Read it WITH length —
a wide interval buys coverage for free. CRPS and WIS are proper, so
lower is better on both and they penalise that trade-off.

`n_files` counts npz files scored, not distinct datasets: under
`--subset total` each dataset contributes up to two (its zero and its
non-zero queries), so n_files exceeds the realization count. `n_query`
is the honest total and is exact — the subsets are disjoint and
together complete.

`sd_ratio` = mean predictive sd / sd of the true tau. A UNITS CHECK:
a density dumped on the wrong scale still scores finitely, it just
looks like a bad model. Values near 1 are well-calibrated in spread;
a ratio in the tens means the density is on the wrong axis, not that
the model is that much worse. `bias` is mean(E[tau]) - tau_true.

| method | n_files | n_query | coverage95 | length | crps | wis | pred_sd | true_sd | sd_ratio | bias |
|---|---|---|---|---|---|---|---|---|---|---|
| dopfn_native | 137 | 100000 | 0.977 | 26.8718 ± 0.0899 | 1.9732 ± 0.0098 | 1.8082 ± 0.0089 | 6.9214 | 0.3325 | 20.8 | -0.1021 |
| dopfn_bb | 137 | 100000 | 0.811 | 1.1162 ± 0.0035 | 0.1695 ± 0.0007 | 0.1550 ± 0.0007 | 0.3148 | 0.3325 | 0.9 | -0.0087 |
| uwyk1d-noanc | 137 | 100000 | 0.805 | 0.6802 ± 0.0025 | 0.1447 ± 0.0008 | 0.1342 ± 0.0007 | 0.1947 | 0.3325 | 0.6 | +0.0271 |
| uwyk1d-v3a | 137 | 100000 | 0.880 | 0.7814 ± 0.0024 | 0.1105 ± 0.0006 | 0.1012 ± 0.0006 | 0.2209 | 0.3325 | 0.7 | +0.0181 |
| graph2d-noanc | 137 | 100000 | 0.848 | 0.7910 ± 0.0026 | 0.1008 ± 0.0006 | 0.0911 ± 0.0006 | 0.2141 | 0.3325 | 0.6 | +0.0028 |
| graph2d-v3a | 137 | 100000 | 0.794 | 0.5520 ± 0.0020 | 0.0964 ± 0.0006 | 0.0879 ± 0.0006 | 0.1596 | 0.3325 | 0.5 | +0.0127 |
| cpfn1d | 137 | 100000 | 0.815 | 0.6006 ± 0.0024 | 0.1204 ± 0.0007 | 0.1117 ± 0.0006 | 0.1609 | 0.3325 | 0.5 | +0.0227 |
| cpfn2d | 137 | 100000 | 0.906 | 0.8032 ± 0.0023 | 0.1213 ± 0.0006 | 0.1100 ± 0.0006 | 0.2101 | 0.3325 | 0.6 | +0.0184 |


# CATE density calibration — d=30, N=1000, total, 1D coupling = indep

Scored object: CATE density per query, scored against that query's true tau.

2D heads: diagonal projection of the joint. 1D heads: independence
convolution of the two arm marginals. Raw densities, no MALC.

coverage95 should be ~0.95; below means over-confident, above means
the intervals are wider than they need to be. Read it WITH length —
a wide interval buys coverage for free. CRPS and WIS are proper, so
lower is better on both and they penalise that trade-off.

`n_files` counts npz files scored, not distinct datasets: under
`--subset total` each dataset contributes up to two (its zero and its
non-zero queries), so n_files exceeds the realization count. `n_query`
is the honest total and is exact — the subsets are disjoint and
together complete.

`sd_ratio` = mean predictive sd / sd of the true tau. A UNITS CHECK:
a density dumped on the wrong scale still scores finitely, it just
looks like a bad model. Values near 1 are well-calibrated in spread;
a ratio in the tens means the density is on the wrong axis, not that
the model is that much worse. `bias` is mean(E[tau]) - tau_true.

| method | n_files | n_query | coverage95 | length | crps | wis | pred_sd | true_sd | sd_ratio | bias |
|---|---|---|---|---|---|---|---|---|---|---|
| dopfn_native | 146 | 100000 | 0.989 | 34.1167 ± 0.1241 | 1.8799 ± 0.0081 | 1.7288 ± 0.0073 | 8.5709 | 0.3295 | 26.0 | +0.4515 |
| dopfn_bb | 146 | 100000 | 0.832 | 1.0826 ± 0.0037 | 0.1693 ± 0.0008 | 0.1548 ± 0.0008 | 0.3155 | 0.3295 | 1.0 | -0.0038 |
| uwyk1d-noanc | 146 | 100000 | 0.887 | 0.7190 ± 0.0027 | 0.1089 ± 0.0008 | 0.1006 ± 0.0007 | 0.2085 | 0.3295 | 0.6 | -0.0118 |
| uwyk1d-v3a | 146 | 100000 | 0.910 | 0.7345 ± 0.0026 | 0.0959 ± 0.0007 | 0.0884 ± 0.0007 | 0.2105 | 0.3295 | 0.6 | -0.0027 |
| graph2d-noanc | 146 | 100000 | 0.794 | 0.6830 ± 0.0027 | 0.0805 ± 0.0006 | 0.0725 ± 0.0005 | 0.1922 | 0.3295 | 0.6 | +0.0024 |
| graph2d-v3a | 146 | 100000 | 0.708 | 0.4346 ± 0.0020 | 0.0844 ± 0.0007 | 0.0765 ± 0.0006 | 0.1280 | 0.3295 | 0.4 | -0.0010 |
| cpfn1d | 146 | 100000 | 0.874 | 0.5030 ± 0.0022 | 0.0991 ± 0.0007 | 0.0923 ± 0.0007 | 0.1399 | 0.3295 | 0.4 | +0.0076 |
| cpfn2d | 146 | 100000 | 0.932 | 0.7865 ± 0.0021 | 0.1060 ± 0.0007 | 0.0969 ± 0.0006 | 0.2019 | 0.3295 | 0.6 | +0.0177 |


# CATE density calibration — d=40, N=1000, total, 1D coupling = indep

Scored object: CATE density per query, scored against that query's true tau.

2D heads: diagonal projection of the joint. 1D heads: independence
convolution of the two arm marginals. Raw densities, no MALC.

coverage95 should be ~0.95; below means over-confident, above means
the intervals are wider than they need to be. Read it WITH length —
a wide interval buys coverage for free. CRPS and WIS are proper, so
lower is better on both and they penalise that trade-off.

`n_files` counts npz files scored, not distinct datasets: under
`--subset total` each dataset contributes up to two (its zero and its
non-zero queries), so n_files exceeds the realization count. `n_query`
is the honest total and is exact — the subsets are disjoint and
together complete.

`sd_ratio` = mean predictive sd / sd of the true tau. A UNITS CHECK:
a density dumped on the wrong scale still scores finitely, it just
looks like a bad model. Values near 1 are well-calibrated in spread;
a ratio in the tens means the density is on the wrong axis, not that
the model is that much worse. `bias` is mean(E[tau]) - tau_true.

| method | n_files | n_query | coverage95 | length | crps | wis | pred_sd | true_sd | sd_ratio | bias |
|---|---|---|---|---|---|---|---|---|---|---|
| dopfn_native | 135 | 100000 | 0.995 | 27.7580 ± 0.0910 | 1.4283 ± 0.0049 | 1.3190 ± 0.0045 | 6.9430 | 0.2517 | 27.6 | -0.1580 |
| dopfn_bb | 135 | 100000 | 0.817 | 0.8968 ± 0.0034 | 0.1286 ± 0.0006 | 0.1184 ± 0.0006 | 0.2602 | 0.2517 | 1.0 | +0.0068 |
| uwyk1d-noanc | 135 | 100000 | 0.924 | 0.6522 ± 0.0022 | 0.0853 ± 0.0006 | 0.0785 ± 0.0006 | 0.1892 | 0.2517 | 0.8 | +0.0184 |
| uwyk1d-v3a | 135 | 100000 | 0.943 | 0.6804 ± 0.0022 | 0.0787 ± 0.0006 | 0.0721 ± 0.0005 | 0.1931 | 0.2517 | 0.8 | +0.0229 |
| graph2d-noanc | 135 | 100000 | 0.807 | 0.6101 ± 0.0022 | 0.0681 ± 0.0005 | 0.0616 ± 0.0004 | 0.1739 | 0.2517 | 0.7 | -0.0016 |
| graph2d-v3a | 135 | 100000 | 0.742 | 0.4511 ± 0.0019 | 0.0664 ± 0.0005 | 0.0601 ± 0.0004 | 0.1282 | 0.2517 | 0.5 | +0.0107 |
| cpfn1d | 135 | 100000 | 0.863 | 0.5028 ± 0.0019 | 0.0848 ± 0.0006 | 0.0783 ± 0.0006 | 0.1345 | 0.2517 | 0.5 | +0.0170 |
| cpfn2d | 135 | 100000 | 0.939 | 0.6894 ± 0.0016 | 0.0909 ± 0.0006 | 0.0827 ± 0.0005 | 0.1791 | 0.2517 | 0.7 | +0.0084 |


# CATE density calibration — d=50, N=1000, total, 1D coupling = indep

Scored object: CATE density per query, scored against that query's true tau.

2D heads: diagonal projection of the joint. 1D heads: independence
convolution of the two arm marginals. Raw densities, no MALC.

coverage95 should be ~0.95; below means over-confident, above means
the intervals are wider than they need to be. Read it WITH length —
a wide interval buys coverage for free. CRPS and WIS are proper, so
lower is better on both and they penalise that trade-off.

`n_files` counts npz files scored, not distinct datasets: under
`--subset total` each dataset contributes up to two (its zero and its
non-zero queries), so n_files exceeds the realization count. `n_query`
is the honest total and is exact — the subsets are disjoint and
together complete.

`sd_ratio` = mean predictive sd / sd of the true tau. A UNITS CHECK:
a density dumped on the wrong scale still scores finitely, it just
looks like a bad model. Values near 1 are well-calibrated in spread;
a ratio in the tens means the density is on the wrong axis, not that
the model is that much worse. `bias` is mean(E[tau]) - tau_true.

| method | n_files | n_query | coverage95 | length | crps | wis | pred_sd | true_sd | sd_ratio | bias |
|---|---|---|---|---|---|---|---|---|---|---|
| dopfn_native | 146 | 100000 | 0.996 | 31.7186 ± 0.1077 | 1.6088 ± 0.0064 | 1.4867 ± 0.0059 | 8.0635 | 0.3052 | 26.4 | -0.0857 |
| dopfn_bb | 146 | 100000 | 0.872 | 0.9458 ± 0.0036 | 0.1420 ± 0.0008 | 0.1297 ± 0.0007 | 0.2862 | 0.3052 | 0.9 | -0.0137 |
| uwyk1d-noanc | 146 | 100000 | 0.943 | 0.7675 ± 0.0028 | 0.0938 ± 0.0007 | 0.0863 ± 0.0007 | 0.2130 | 0.3052 | 0.7 | +0.0034 |
| uwyk1d-v3a | 146 | 100000 | 0.945 | 0.7354 ± 0.0028 | 0.0894 ± 0.0007 | 0.0820 ± 0.0007 | 0.2014 | 0.3052 | 0.7 | +0.0035 |
| graph2d-noanc | 146 | 100000 | 0.791 | 0.6172 ± 0.0028 | 0.0670 ± 0.0006 | 0.0605 ± 0.0005 | 0.1721 | 0.3052 | 0.6 | +0.0027 |
| graph2d-v3a | 146 | 100000 | 0.720 | 0.4918 ± 0.0025 | 0.0671 ± 0.0006 | 0.0605 ± 0.0005 | 0.1365 | 0.3052 | 0.4 | +0.0018 |
| cpfn1d | 146 | 100000 | 0.898 | 0.5721 ± 0.0027 | 0.0851 ± 0.0007 | 0.0787 ± 0.0007 | 0.1472 | 0.3052 | 0.5 | +0.0023 |
| cpfn2d | 146 | 100000 | 0.951 | 0.7674 ± 0.0022 | 0.0944 ± 0.0007 | 0.0861 ± 0.0006 | 0.1983 | 0.3052 | 0.6 | +0.0085 |
(venv) furkanbd@klogin03:/scratch/furkanbd/rpfn_bench_kit/R-PFN$ 



tail -f $SCRATCH/rpfn_bench_kit/R-PFN/logs_density/report_5433401.out


sleep 300; grep -c "cen3 d" $SCRATCH/rpfn_bench_kit/R-PFN/logs_density/report_5449976.out

head -2 $SCRATCH/rpfn_bench_kit/R-PFN/logs_density/report_5449976.out
tail -3 $SCRATCH/rpfn_bench_kit/R-PFN/logs_density/report_5449976.out

export RES=$DEPLOY_ROOT/results_case_study/dvar
for s in shift0 shift+2 shift-2; do
  for m in graph2d uwyk uwyk_v3a uwyk_noanc dopfn_native dopfn_bb; do
    tot=$(ls $RES/$s/d*/ctx1000/$m/*/*_r*.npz \
             $RES/$s/d*/ctx1000/$m/*/predictions/*_r*.npz 2>/dev/null | wc -l)
    printf "%-8s %-14s %6s / 4800\n" "$s" "$m" "$tot"
  done
done

            
for m in dopfn_native dopfn_bb uwyk1d graph2d cpfn1d cpfn2d; do
  printf "%-14s" $m
  for d in IHDP ACIC CPS PSID PSID_bal; do
    printf "  %s:%s" $d $(ls $SCRATCH/rc_dens/$m/$d/*.npz 2>/dev/null | wc -l)
  done; echo
done




L=$SCRATCH/rpfn_bench_kit/R-PFN/logs_cpfn2d_j32_armcentered

# 1. is arm_centered actually active?
grep -iE "y_scale|arm_centered|J=|sigma" $L/train_5464652.out | head -5

# 2. checkpoint retention
grep "step-ckpt" $L/train_5464652.out | head -10

# 3. is the loss finite and moving?  ← the one that matters
grep -iE "loss|Effective batch" $L/train_5464652.out | head -8

# 4. anything broken
tail -25 $L/train_5464652.err



sbatch --time=24:00:00 --gres=gpu:h100:1 \
  --export=ALL,COMPILE=0,STEP_CKPT_EVERY=2000,STEP_CKPT_KEEP=5 \
  benchmarks/cluster/submit_train_cpfn2d_j32_armcentered.sbatch



source $DEPLOY_ROOT/venv/bin/activate
python - <<'PY'
import glob, os
base = os.path.expandvars("$SCRATCH/cs_dvar_dens")
MODELS = ["dopfn_native","dopfn_bb","uwyk1d","graph2d","cpfn1d","cpfn2d_pooled"]
DS = [2,3,5,10,20,30,40,50]
print("CASE STUDIES  (600 npz per cell = 6 cases x 100)\n")
for s in ("shift0","shift+2","shift-2"):
    print(f"=== {s} ===")
    print(f"{'model':16s}" + "".join(f"{'d'+str(d):>9s}" for d in DS))
    for m in MODELS:
        row = f"{m:16s}"
        for d in DS:
            n = len(glob.glob(f"{base}/{s}/d{d}/ctx1000/{m}/*/*.npz"))
            row += f"{n:9d}"
        print(row)
    print()
PY

python - <<'PY'
import glob, os
base = os.path.expandvars("$SCRATCH/cmech_1d2d")
print("COMPLEXMECH dopfn_native  (100 npz per cell)\n")
print(f"{'ctx':>6s}" + "".join(f"{'n'+str(n):>8s}" for n in (5,10,20,30,40,50)))
for ctx in (50,100,250,500,1000):
    row = f"{ctx:6d}"
    for n in (5,10,20,30,40,50):
        c = len(glob.glob(f"{base}/n{n}_nonzero/*.npz"))
        row += f"{c:8d}"
    print(row)
PY



grep -A4 "^| method" logs_cmech_score/score_5470751.out
ls $SCRATCH/cmech_1d2d/ate_density_*.md


# after it finishes
grep -hE "bootstrap|scaling" logs_cpfn2d_realcause/eval_5470758_*.out
grep -hc "^r=" logs_cpfn2d_realcause/eval_5470758_*.out


grep -E "step-ckpt|step-limit|nbins|both-arms" logs_cpfn1d_j32_headrand/train_5472267.out
grep -E "step-ckpt|step-limit|both-arms" logs_cpfn1d_j1024_botharms/train_5472268.out


grep -hE "bootstrap|scaling" logs_cpfn2d_realcause/eval_5485932_*.out
python realcause_eval/summarize_realcause.py --out-root $DEPLOY_ROOT/realcause_armc_step30000



grep -hE "TYPE=|scaling|STD_MODE" logs_cpfn2d_rc_cs/eval_5485959_*.out
grep -hA7 "summary" logs_cpfn2d_rc_cs/eval_5485959_*.out



tail -40 logs_rc_malc_variants/run_5500443.out


D=$DEPLOY_ROOT/cpfn2d_j32_eta0_y01_A1_h100/step_checkpoints/run
E=$DEPLOY_ROOT/logs_cpfn2d_j32_sharednoise_y01/train_467042.err

ls -t $D | head -3
python3 - <<EOF
import re
raw = open("$E","rb").read()[-600:].decode("utf-8","replace").replace("\r","\n")
m = [l for l in raw.split("\n") if "it/s" in l][-1]
b = int(re.search(r"\| (\d+)/", m).group(1)); r = float(re.search(r"([\d.]+)it/s", m).group(1))
st, sps = b//8, r/8
print(f"step {st} @ {sps:.3f} steps/s")
for t in (40000, 50000):
    print(f"  -> {t}: {(t-st)/sps/3600:5.1f} h from now")
EOF



| method | n_files | n_query | coverage95 | length | is05 | pred_sd | true_sd | sd_ratio | bias |
|---|---|---|---|---|---|---|---|---|---|
| dopfn_native | 3 | 3 | 1.000 | 9.1500 ± 0.1058 | 9.1500 ± 0.1058 | 2.3271 | 0.0727 | 32.0 | -1.5699 |
| dopfn_bb | 3 | 3 | 1.000 | 3.2755 ± 0.9155 | 3.2755 ± 0.9155 | 0.8192 | 0.0727 | 11.3 | -0.4239 |
| uwyk1d-noanc | 3 | 3 | 1.000 | 10.9368 ± 0.1165 | 10.9368 ± 0.1165 | 2.8342 | 0.0727 | 39.0 | -2.2517 |
| uwyk1d-v3a | 3 | 3 | 1.000 | 9.5495 ± 0.2918 | 9.5495 ± 0.2918 | 2.4573 | 0.0727 | 33.8 | -1.3280 |
| graph2d-noanc | 3 | 3 | 1.000 | 9.1421 ± 0.3209 | 9.1421 ± 0.3209 | 2.2243 | 0.0727 | 30.6 | -1.1665 |
| graph2d-v3a | 3 | 3 | 1.000 | 8.2464 ± 0.4174 | 8.2464 ± 0.4174 | 2.0065 | 0.0727 | 27.6 | -0.9616 |
| cpfn1d | 3 | 3 | 1.000 | 4.3716 ± 0.0688 | 4.3716 ± 0.0688 | 1.0913 | 0.0727 | 15.0 | -0.0184 |
| cpfn2d | 3 | 3 | 1.000 | 5.5836 ± 0.0460 | 5.5836 ± 0.0460 | 1.4246 | 0.0727 | 19.6 | +0.1361 |

cd /scratch/furkanbd/rpfn_bench_kit
sbatch --array=0 --time=00:45:00 --export=ALL,MAX_REAL=3,MALC_K=3,STAGES=t \
  R-PFN/benchmarks/cluster/submit_malc_case_studies.sbatch



cd $DEPLOY_ROOT
python3 - <<'EOF'
import re, glob, os
R = os.environ['DEPLOY_ROOT']
jobs = [
    ("cpfn1d j32_headrand",   f"{R}/R-PFN/logs_cpfn1d_j32_headrand/train_466773.err",
                              f"{R}/cpfn1d_j32_headrand_output/step_checkpoints/run"),
    ("cpfn1d j1024_botharms", f"{R}/R-PFN/logs_cpfn1d_j1024_botharms/train_466774.err",
                              f"{R}/cpfn1d_j1024_botharms_output/step_checkpoints/run"),
    ("cpfn2d j32_eta0_y01",   f"{R}/logs_cpfn2d_j32_sharednoise_y01/train_467042.err",
                              f"{R}/cpfn2d_j32_eta0_y01_A1_h100/step_checkpoints/run"),
]
print(f"{'run':24}{'step':>8}{'steps/s':>9}{'loss':>8}   ckpts on disk        ->50k")
for name, err, d in jobs:
    try:
        raw = open(err, "rb").read()[-800:].decode("utf-8", "replace").replace("\r", "\n")
        line = [l for l in raw.split("\n") if "it/s" in l][-1]
        b = int(re.search(r"\| (\d+)/", line).group(1))
        r = float(re.search(r"([\d.]+)it/s", line).group(1))
        loss = re.search(r"loss=([\d.]+)", line)
        st, sps = b // 8, r / 8
        eta = (50000 - st) / sps / 3600
        cks = sorted(os.path.basename(p) for p in glob.glob(f"{d}/step_*.pt"))
        tail = ",".join(c[5:-3].lstrip("0") for c in cks[-3:]) or "-"
        print(f"{name:24}{st:>8}{sps:>9.3f}{float(loss.group(1)) if loss else 0:>8.2f}   {tail:20} {eta:5.1f}h")
    except Exception as e:
        print(f"{name:24}  -- {type(e).__name__}: {e}")
EOF



cpfn_step logs_cpfn1d_j1024_botharms/train_60247982.err

grep "resuming from actual_step" logs_cpfn2d_j32_sharednoise_y01/train_60275347.out

echo "───── dopfn (steps printed directly)"
for j in 60251112:1d 60251113:2d; do
  id=${j%%:*}; v=${j##*:}
  echo -n "  $v ($id): "; tail -2 logs_dopfn_repro_$v/train_$id.out | tr '\n' ' '; echo
done
ls -t $DEPLOY_ROOT/checkpoints_dopfn_repro/*/ 2>/dev/null | head -6




cpfn1d_j32_step50000.pt
dopfn_repro_1d_step150000.pt
dopfn_repro_joint2d_step150000.pt

(venv) [furkanbd@login2 R-PFN]$ # on Fir — stage them under one name each
mkdir -p $DEPLOY_ROOT/final_checkpoints
cp $DEPLOY_ROOT/checkpoints_dopfn_repro/dopfn_1d/step_150000_final.pt \
   $DEPLOY_ROOT/final_checkpoints/dopfn_repro_1d_step150000.pt
cp $DEPLOY_ROOT/checkpoints_dopfn_repro/joint_2d/step_150000_final.pt \
   $DEPLOY_ROOT/final_checkpoints/dopfn_repro_joint2d_step150000.pt
ls -lh $DEPLOY_ROOT/final_checkpoints/
total 306M
-rw-r----- 1 furkanbd furkanbd 208M Sep 17 16:08 cpfn1d_j32_step50000.pt
-rw-r----- 1 furkanbd furkanbd  85M Sep 17 19:17 dopfn_repro_1d_step150000.pt
-rw-r----- 1 furkanbd furkanbd  85M Sep 17 19:17 dopfn_repro_joint2d_step150000.pt


python ~/steps.py



IHDP   √PEHE 6.005 ± 0.801     ε_ATE 0.531
ACIC         4.113 ± 0.545           0.716
PSID    17,272.6 ± 151.8            0.109
PSIDbal 17,205.8 ± 155.6            0.114
