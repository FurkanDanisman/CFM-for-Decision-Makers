# CATE / ATE distributions — UWYK Fig-3 / Fig-4 priors

`oracle_pehe_floor_mean` is the mean within-dataset sd of the true ITE:
the best sqrt(PEHE) any model can reach. `heterogeneity_ratio` is that
floor over the mean |ATE| — near 0 means the prior generates a constant
treatment effect, so PEHE there measures only ATE accuracy.

| prior | n_nodes | regime | n_realizations | cate_mean | cate_sd | cate_q05 | cate_q50 | cate_q95 | ate_mean | ate_sd | ate_q05 | ate_q95 | oracle_pehe_floor_mean | heterogeneity_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lingaus | 2 | path_TY | 100 | -0.01151 | 0.3892 | -0.6104 | 0.0008978 | 0.6199 | -0.01151 | 0.3892 | -0.6104 | 0.6199 | 3.953e-08 | 1.21e-07 |
| lingaus | 2 | path_YT | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lingaus | 2 | path_independent_TY | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lingaus | 5 | path_TY | 100 | 0.03597 | 0.3595 | -0.4654 | 0.02471 | 0.5414 | 0.03597 | 0.3595 | -0.4654 | 0.5414 | 4.963e-08 | 1.86e-07 |
| lingaus | 5 | path_YT | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lingaus | 5 | path_independent_TY | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lingaus | 20 | path_TY | 100 | 0.005906 | 0.2109 | -0.2533 | -0.007696 | 0.3282 | 0.005906 | 0.2109 | -0.2533 | 0.3282 | 5.427e-08 | 4.03e-07 |
| lingaus | 20 | path_YT | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lingaus | 20 | path_independent_TY | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lingaus | 35 | path_TY | 100 | 0.02085 | 0.1844 | -0.1583 | 0.003882 | 0.3496 | 0.02085 | 0.1844 | -0.1583 | 0.3496 | 5.795e-08 | 5.32e-07 |
| lingaus | 35 | path_YT | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lingaus | 35 | path_independent_TY | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lingaus | 50 | path_TY | 100 | -0.01243 | 0.1331 | -0.2283 | -0.004807 | 0.154 | -0.01243 | 0.1331 | -0.2283 | 0.154 | 5.809e-08 | 7.18e-07 |
| lingaus | 50 | path_YT | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lingaus | 50 | path_independent_TY | 100 | -5.96e-13 | 1.885e-10 | 0 | 0 | 0 | -5.96e-13 | 5.931e-12 | 0 | 0 | 1.884e-11 | 31.6 |
| complexmech | 2 | path_TY | 100 | -0.1135 | 0.6969 | -1.555 | -0.001885 | 0.8697 | -0.1135 | 0.6772 | -1.527 | 0.8272 | 0.07403 | 0.176 |
| complexmech | 2 | path_YT | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| complexmech | 2 | path_independent_TY | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| complexmech | 5 | path_TY | 100 | 0.05094 | 0.6146 | -0.8815 | 0 | 1.194 | 0.05094 | 0.5815 | -0.8673 | 0.8203 | 0.1069 | 0.362 |
| complexmech | 5 | path_YT | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| complexmech | 5 | path_independent_TY | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| complexmech | 20 | path_TY | 100 | -0.02792 | 0.3309 | -0.6343 | 0 | 0.5038 | -0.02792 | 0.2123 | -0.4049 | 0.3733 | 0.1573 | 1.43 |
| complexmech | 20 | path_YT | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| complexmech | 20 | path_independent_TY | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| complexmech | 35 | path_TY | 100 | 0.006254 | 0.2865 | -0.2315 | 0 | 0.3716 | 0.006254 | 0.09578 | -0.1109 | 0.1086 | 0.1329 | 3.38 |
| complexmech | 35 | path_YT | 100 | 1.192e-12 | 3.77e-10 | 0 | 0 | 0 | 1.192e-12 | 1.186e-11 | 0 | 0 | 3.768e-11 | 31.6 |
| complexmech | 35 | path_independent_TY | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| complexmech | 50 | path_TY | 100 | -0.003373 | 0.3058 | -0.2741 | 0 | 0.3041 | -0.003373 | 0.1497 | -0.1033 | 0.1142 | 0.1308 | 2.6 |
| complexmech | 50 | path_YT | 100 | -5.96e-13 | 2.144e-08 | 0 | 0 | 0 | -5.96e-13 | 5.931e-12 | 0 | 0 | 2.144e-09 | 3.6e+03 |
| complexmech | 50 | path_independent_TY | 100 | 5.96e-13 | 1.885e-10 | 0 | 0 | 0 | 5.96e-13 | 5.931e-12 | 0 | 0 | 1.884e-11 | 31.6 |
