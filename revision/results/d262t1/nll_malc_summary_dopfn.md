# Table 22 NLL (MALC-T) from our dumps (raw outcome units)

Scorer: benchmarks/eval_graph2d/nll_from_dumps.py, 2026-10-08

### IHDP: NLL in raw outcome units, mean ± SE over 100 realizations (MALC-T on the hist input, B=100, K=1)

| Model | n | L_{Y0+Y1} | L_tau | L_ATE | non-finite (y / tau / ate) |
|---|---|---|---|---|---|
| Do-PFN | 100 | 4.5380 ± 0.1470 | 2.7603 ± 0.0708 | 2.4410 ± 0.0726 | 0 / 0 / 0 (MALC fallback to the hist density: 189 of 7500 queries) |

### ACIC: NLL in raw outcome units, mean ± SE over 10 realizations (MALC-T on the hist input, B=100, K=1)

| Model | n | L_{Y0+Y1} | L_tau | L_ATE | non-finite (y / tau / ate) |
|---|---|---|---|---|---|
| Do-PFN | 10 | 5.7900 ± 0.0722 | 3.0238 ± 0.0671 | 2.8504 ± 0.0453 | 0 / 0 / 0 (MALC fallback to the hist density: 41 of 4810 queries) |

