## Cross-world correlation rho per dataset (Table 9 `ols-on-X`, every source)

rho = corr(Y(0) - OLS_X[Y(0)], Y(1) - OLS_X[Y(1)]) over one dataset's query units (benchmarks/arm_noise_table.py `_ols_resid` + `_rho`). Shares are over finite rho; Fisher-combined is Table 9's inverse-variance column, for cross-checking.

| source | n | nan | min | 5% | 25% | 50% | 75% | 95% | max | mean | sd | > 0.99 | < 0.5 | < 0 | Fisher-combined |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ComplexMech hide_0.0 (main tables), pooled | 600 | 0 | -0.9881 | -0.3671 | +0.1559 | +0.6540 | +0.9335 | +0.9977 | +1.0000 | +0.5062 | +0.4603 | 0.137 | 0.440 | 0.150 | +0.7887 |
| ComplexMech hide_0.0, n=5 | 100 | 0 | -0.9881 | -0.5599 | +0.3778 | +0.7732 | +0.9850 | +0.9986 | +1.0000 | +0.5782 | +0.4898 | 0.200 | 0.330 | 0.120 | +0.8726 |
| ComplexMech hide_0.0, n=10 | 100 | 0 | -0.9062 | -0.4854 | +0.0791 | +0.5512 | +0.9574 | +0.9996 | +1.0000 | +0.4488 | +0.5057 | 0.180 | 0.490 | 0.200 | +0.7977 |
| ComplexMech hide_0.0, n=20 | 100 | 0 | -0.7421 | -0.4267 | +0.1299 | +0.4841 | +0.8793 | +0.9982 | +0.9999 | +0.4533 | +0.4647 | 0.100 | 0.510 | 0.160 | +0.7270 |
| ComplexMech hide_0.0, n=30 | 100 | 0 | -0.8453 | -0.3924 | +0.0731 | +0.3939 | +0.8812 | +0.9960 | +0.9974 | +0.4216 | +0.4620 | 0.070 | 0.560 | 0.180 | +0.6580 |
| ComplexMech hide_0.0, n=40 | 100 | 0 | -0.4492 | -0.1529 | +0.2809 | +0.6993 | +0.9105 | +0.9956 | +0.9998 | +0.5774 | +0.3871 | 0.130 | 0.360 | 0.110 | +0.7946 |
| ComplexMech hide_0.0, n=50 | 100 | 0 | -0.4748 | -0.1560 | +0.1700 | +0.6870 | +0.9550 | +0.9963 | +1.0000 | +0.5580 | +0.4251 | 0.140 | 0.390 | 0.130 | +0.8262 |
| ComplexMech, Table 9 glob (hides 0.0, 0.25, 0.5, 0.75, 1.0), pooled | 3000 | 0 | -0.9881 | -0.3671 | +0.1559 | +0.6540 | +0.9335 | +0.9977 | +1.0000 | +0.5062 | +0.4600 | 0.137 | 0.440 | 0.150 | +0.7885 |
| Do-PFN joint prior (2D) | 1000 | 0 | -0.9022 | +0.4375 | +0.9745 | +0.9998 | +1.0000 | +1.0000 | +1.0000 | +0.9152 | +0.2404 | 0.694 | 0.056 | 0.016 | +0.9998 |
| UWYK paired prior (graph2d) | 1000 | 0 | -0.8611 | -0.2030 | +0.1506 | +0.7677 | +0.9983 | +1.0000 | +1.0000 | +0.5712 | +0.4573 | 0.304 | 0.406 | 0.142 | +0.9737 |
| CausalPFN prior, eta1:=eta0 (cpfn2d) | 1000 | 0 | -0.6373 | +0.1116 | +0.3483 | +0.5532 | +0.7362 | +0.8972 | +1.0000 | +0.5322 | +0.2515 | 0.002 | 0.417 | 0.012 | +0.5983 |

Query units per dataset: ComplexMech 1000 (X_test); priors 500 (E2_in_prior new-world draws, worlds 0..999, seed base 20261003).
OLS-on-X misses nonlinearity in E[Y|X], which stays in the residual; rows are proxies (upper bounds when the shared noise dominates), measured identically for every source.
