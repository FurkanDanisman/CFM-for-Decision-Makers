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

