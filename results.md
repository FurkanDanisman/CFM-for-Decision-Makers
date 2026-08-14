## TASK 2 — UWYK variant for the density-estimation table
── Table 3 addition — IHDP ────────────────────────────────────────
Method                                   sqrt(PEHE)            eps_ATE
--------------------------------------------------------------------------
Do-PFN                               6.00 ± 8.95        0.93 ± 4.00      (n=100)
UWYK Full-Anc (MALC-mean, density)     7.77 ± 8.21        0.94 ± 0.07      (n=100)

── Density L2 — IHDP ──────────────────────────────────────────────
Method                      p(Y_do0)      p(Y_do1)       p(CATE)        p(ATE)
------------------------------------------------------------------------------
Do-PFN                 1.633±1.422 1.662±1.455 1.418±1.132 1.285±1.024   (ATE n=100)
UWYK Full-Anc          1.866±1.371 1.825±1.309 1.653±1.046 1.574±0.912   (ATE n=100)

## TASK 3 — IHDP density L2 for the Do-PFN-2DMALC checkpoint
── Table 3 addition — IHDP ────────────────────────────────────────
Method                                   sqrt(PEHE)            eps_ATE
--------------------------------------------------------------------------
Do-PFN                               6.00 ± 8.95        0.93 ± 4.00      (n=100)
Ours(DoPFN-bb 200K) (MALC-mean, density)     5.17 ± 7.31        0.72 ± 1.52      (n=100)
Ours(DoPFN-bb 200K) (raw-mean)       5.00 ± 7.05        0.61 ± 1.09      (n=100)
Ours(DoPFN-bb 200K) (EM mixture)     5.17 ± 7.34        0.73 ± 1.56      (n=100)
Ours(DoPFN-bb 200K) (EM K=1)         5.01 ± 7.27        0.70 ± 1.26      (n=100)

── Density L2 — IHDP ──────────────────────────────────────────────
Method                      p(Y_do0)      p(Y_do1)       p(CATE)        p(ATE)
------------------------------------------------------------------------------
Do-PFN                 1.633±1.422 1.662±1.455 1.418±1.132 1.285±1.024   (ATE n=100)
Ours(DoPFN-bb 200K)    1.588±1.462 1.666±1.673 1.561±1.166 1.347±1.071   (ATE n=100)

## TASK 4 — Linear-Gaussian synthetic (task 5), 100 seeds

── Linear-Gaussian SCM (N=500, d=5, seeds=100) ────────
Method                               sqrt(PEHE)          eps_ATE
------------------------------------------------------------------------------
Do-PFN                             0.98 ± 0.22      0.06 ± 0.06    (n=100)
Ours(DoPFN-bb 200K) (MALC-CATE-mean)   1.02 ± 0.15      0.14 ± 0.08    (n=100)
Ours(DoPFN-bb 200K) (Raw-mean)     1.01 ± 0.15      0.13 ± 0.08    (n=100)
Ours(DoPFN-bb 200K) (EM-mean-K1)   1.01 ± 0.15      0.14 ± 0.08    (n=100)
Ours(DoPFN-bb 200K) (EM-mean-Kselection)   1.01 ± 0.15      0.14 ± 0.08    (n=100)

── Density L2 — Linear-Gaussian SCM ──────────────────────────────
Method                              p(Y_do0)      p(Y_do1)       p(CATE)        p(ATE)
----------------------------------------------------------------------------------------
Do-PFN                         0.460±0.263 0.393±0.244 0.416±0.231 0.228±0.072   (ATE n=100)
Ours(DoPFN-bb 200K)            0.549±0.316 0.420±0.266 0.723±0.368 0.470±0.188   (ATE n=100)