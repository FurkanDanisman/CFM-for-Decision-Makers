# Learnable CATE heterogeneity (regime path_TY)

`pehe_constant` ignores X; `pehe_oracle` is a random forest fitted on the
true tau, i.e. an optimistic ceiling for any CATE method. `learnable` is
the fraction of within-dataset CATE variance X can explain. Near 0 means
PEHE on that cell cannot separate a CATE method from an ATE estimator.

| prior | n_nodes | n_features | pehe_constant | pehe_oracle | r2_tau | r2_tau_median | learnable | learnable_median | frac_learnable_gt_0.1 |
|---|---|---|---|---|---|---|---|---|---|
| lingaus | 2 | 0 | 3.47e-08 | 3.47e-08 | 0 | 0 | 0 | 0 | 0 |
| lingaus | 5 | 3 | 4.737e-08 | 4.92e-08 | -0.08549 | -0.07711 | 0 | 0 | 0 |
| lingaus | 20 | 18 | 5.23e-08 | 5.413e-08 | -0.07761 | -0.06199 | 0.0001289 | 0 | 0 |
| lingaus | 35 | 33 | 5.92e-08 | 6.1e-08 | -0.06643 | -0.06667 | 0 | 0 | 0 |
| lingaus | 50 | 48 | 6.062e-08 | 6.255e-08 | -0.07002 | -0.07196 | 0 | 0 | 0 |
| complexmech | 2 | 0 | 0.0522 | 0.0522 | 0 | 0 | 0 | 0 | 0 |
| complexmech | 5 | 2.933 | 0.132 | 0.0812 | 0.5476 | 0.7807 | 0.5764 | 0.7815 | 0.7 |
| complexmech | 20 | 17.67 | 0.1523 | 0.1163 | -0.2269 | 0.4013 | 0.4153 | 0.4041 | 0.8 |
| complexmech | 35 | 32.7 | 0.1212 | 0.09076 | 0.3165 | 0.2569 | 0.3491 | 0.2603 | 0.6 |
| complexmech | 50 | 47.13 | 0.1924 | 0.1573 | 0.2765 | 0.1424 | 0.2884 | 0.1427 | 0.5667 |
