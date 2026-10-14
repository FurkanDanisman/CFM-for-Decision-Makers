# True ρ(x) on the case studies (exact, from the generator)

Per dataset: median over its first Q units of ρ(x) = corr(Y(1), Y(0) | X = x), K redraws of all noise and hidden roots with x fixed (shared across arms). `rebuilt` = SCMs that reproduce the stored X, Y, μ0, μ1 from their seed.

| scenario | d | datasets | rebuilt | ρ median (5%, 95%) | share ρ > 0.99 | Var(ITE|x)/Var(Y(0)|x) median |
|---|---|---|---|---|---|---|
| Observed_Confounder | 5 | 300 | 300 | +1.000 (+1.000, +1.000) | 1.00 | 1.92e-29 |
| Observed_Confounder | 10 | 300 | 300 | +1.000 (+1.000, +1.000) | 1.00 | 3.36e-29 |
| Observed_Confounder | 20 | 300 | 300 | +1.000 (+1.000, +1.000) | 1.00 | 2.23e-29 |
| Observed_Confounder | 30 | 300 | 300 | +1.000 (+1.000, +1.000) | 1.00 | 1.97e-29 |
| Observed_Confounder | 40 | 300 | 300 | +1.000 (+1.000, +1.000) | 1.00 | 1.21e-29 |
| Observed_Confounder | 50 | 300 | 300 | +1.000 (+1.000, +1.000) | 1.00 | 1.77e-29 |
| Backdoor_Criterion | 5 | 300 | 300 | +1.000 (+0.999, +1.000) | 1.00 | 2.21e-29 |
| Backdoor_Criterion | 10 | 300 | 300 | +1.000 (+1.000, +1.000) | 1.00 | 2.06e-29 |
| Backdoor_Criterion | 20 | 300 | 300 | +1.000 (+1.000, +1.000) | 1.00 | 7.78e-27 |
| Backdoor_Criterion | 30 | 300 | 300 | +1.000 (+1.000, +1.000) | 1.00 | 8.9e-29 |
| Backdoor_Criterion | 40 | 300 | 300 | +1.000 (+0.999, +1.000) | 1.00 | 1.45e-06 |
| Backdoor_Criterion | 50 | 300 | 300 | +1.000 (+1.000, +1.000) | 1.00 | 8.52e-10 |
| Frontdoor_Criterion | 5 | 300 | 300 | +1.000 (+0.998, +1.000) | 0.99 | 5.15e-10 |
| Frontdoor_Criterion | 10 | 300 | 300 | +1.000 (+1.000, +1.000) | 1.00 | 4.64e-29 |
| Frontdoor_Criterion | 20 | 300 | 300 | +1.000 (+0.999, +1.000) | 1.00 | 3.5e-08 |
| Frontdoor_Criterion | 30 | 300 | 300 | +1.000 (+1.000, +1.000) | 1.00 | 5.85e-08 |
| Frontdoor_Criterion | 40 | 300 | 300 | +1.000 (+0.999, +1.000) | 1.00 | 1.32e-27 |
| Frontdoor_Criterion | 50 | 300 | 300 | +1.000 (+1.000, +1.000) | 1.00 | 3.57e-28 |
| Unobserved_Confounder | 5 | 300 | 300 | +0.999 (+0.949, +1.000) | 0.80 | 0.00881 |
| Unobserved_Confounder | 10 | 300 | 300 | +0.999 (+0.956, +1.000) | 0.86 | 0.0059 |
| Unobserved_Confounder | 20 | 300 | 300 | +1.000 (+0.990, +1.000) | 0.94 | 0.00235 |
| Unobserved_Confounder | 30 | 300 | 300 | +1.000 (+0.986, +1.000) | 0.94 | 0.00278 |
| Unobserved_Confounder | 40 | 300 | 300 | +1.000 (+0.996, +1.000) | 0.99 | 0.0026 |
| Unobserved_Confounder | 50 | 300 | 300 | +1.000 (+0.998, +1.000) | 0.98 | 0.00216 |
