# Table 22 NLL (MALC-T) from our dumps (raw outcome units)

Scorer: benchmarks/eval_graph2d/nll_from_dumps.py, 2026-10-08

### IHDP: NLL in raw outcome units, mean ± SE over 100 realizations (MALC-T on the hist input, B=100, K=1)

| Model | n | L_{Y0+Y1} | L_tau | L_ATE | non-finite (y / tau / ate) |
|---|---|---|---|---|---|
| Do-PFN 2D | 100 | 4.7679 ± 0.1557 | 2.8547 ± 0.0752 | 2.1971 ± 0.0804 | 0 / 0 / 0 (MALC fallback to the hist density: 396 of 7500 queries) |
| UWYK No-Anc | 100 | 4.3596 ± 0.1301 | 2.9180 ± 0.0675 | 2.6478 ± 0.0537 | 0 / 0 / 0 (MALC fallback to the hist density: 2159 of 7500 queries) |
| UWYK No-Anc 2D | 100 | 4.2172 ± 0.1293 | 2.6042 ± 0.0664 | 2.2479 ± 0.0759 | 0 / 0 / 0 (MALC fallback to the hist density: 36 of 7500 queries) |
| UWYK Anc | 100 | 4.0570 ± 0.1465 | 2.7390 ± 0.0761 | 2.3906 ± 0.0668 | 0 / 0 / 0 (MALC fallback to the hist density: 3607 of 7500 queries) |
| UWYK Anc 2D | 100 | 4.0587 ± 0.1307 | 2.5350 ± 0.0664 | 2.1085 ± 0.0763 | 0 / 0 / 0 (MALC fallback to the hist density: 62 of 7500 queries) |
| CausalPFN-C | 100 | 1.3092 ± 0.0414 | 1.9912 ± 0.0190 | 1.0289 ± 0.0230 | 0 / 0 / 0 (MALC fallback to the hist density: 1354 of 7500 queries) |
| CausalPFN-C 2D | 100 | 3.4568 ± 0.1248 | 2.2543 ± 0.0542 | 1.9378 ± 0.0706 | 0 / 0 / 0 (MALC fallback to the hist density: 74 of 7500 queries) |

### ACIC: NLL in raw outcome units, mean ± SE over 10 realizations (MALC-T on the hist input, B=100, K=1)

| Model | n | L_{Y0+Y1} | L_tau | L_ATE | non-finite (y / tau / ate) |
|---|---|---|---|---|---|
| Do-PFN 2D | 10 | 5.8846 ± 0.0680 | 2.9155 ± 0.1912 | 2.2601 ± 0.1408 | 0 / 0 / 0 (MALC fallback to the hist density: 343 of 4810 queries) |
| UWYK No-Anc | 10 | 4.8806 ± 0.1644 | 2.8167 ± 0.0813 | 2.5630 ± 0.0740 | 0 / 0 / 0 (MALC fallback to the hist density: 2168 of 4810 queries) |
| UWYK No-Anc 2D | 10 | 4.8393 ± 0.1992 | 2.5783 ± 0.1225 | 2.1154 ± 0.1747 | 0 / 0 / 0 (MALC fallback to the hist density: 65 of 4810 queries) |
| UWYK Anc | 10 | 4.6384 ± 0.1866 | 2.6418 ± 0.0948 | 2.3804 ± 0.0778 | 0 / 0 / 0 (MALC fallback to the hist density: 2857 of 4810 queries) |
| UWYK Anc 2D | 10 | 4.7483 ± 0.2003 | 2.5405 ± 0.1173 | 2.0604 ± 0.1631 | 0 / 0 / 0 (MALC fallback to the hist density: 61 of 4810 queries) |
| CausalPFN-C | 10 | 2.7879 ± 0.1561 | 2.0549 ± 0.0614 | 1.6377 ± 0.0964 | 0 / 0 / 0 (MALC fallback to the hist density: 1091 of 4810 queries) |
| CausalPFN-C 2D | 10 | 4.0121 ± 0.0781 | 2.3076 ± 0.0369 | 2.1604 ± 0.0417 | 0 / 0 / 0 (MALC fallback to the hist density: 4 of 4810 queries) |

