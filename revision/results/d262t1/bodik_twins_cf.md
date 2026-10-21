# Twins: Bodik's counterfactual prediction (predict the other twin given x and the known twin)

2D: learned joint conditioned on the known twin's outcome; 1D: the other arm's marginal (independence). 95% interval. Lower is better.

| setting | model | n | MSE | |ĉ − 0.95| | ĉ | length | IS₀.₀₅ |
|---|---|---|---|---|---|---|---|
| d=10 | Do-PFN | 10984 | 0.1692 | 0.041 | 0.909 | 0.711 | 3.730 |
| d=10 | Do-PFN 2D | 10984 | 0.1412 | 0.011 | 0.961 | 0.989 | 1.704 |
| d=10 | UWYK No-Anc | 10984 | 0.1573 | 0.830 | 0.120 | 0.999 | 1.007 |
| d=10 | UWYK Anc | 10984 | 0.3656 | 0.770 | 0.180 | 1.000 | 1.003 |
| d=10 | UWYK 2D No-Anc | 10984 | 0.0917 | 0.950 | 0.000 | 0.866 | 1.254 |
| d=10 | UWYK 2D Anc | 10984 | 0.0919 | 0.950 | 0.000 | 0.716 | 1.493 |
| d=10 | CausalPFN-C | 10984 | 0.1503 | 0.050 | 1.000 | 1.027 | 1.027 |
| d=10 | CausalPFN-C 2D | 10984 | 0.1772 | 0.049 | 0.999 | 1.495 | 1.496 |
| d=10 | Do-PFN (matched resolution) | 10984 | 0.1828 | 0.030 | 0.980 | 2.862 | 3.404 |
| d=10 | Do-PFN (matched label) | 10984 | 0.1576 | 0.037 | 0.987 | 1.054 | 1.308 |
| d=10 | UWYK No-Anc (matched resolution) | 10984 | 0.1473 | 0.768 | 0.182 | 1.028 | 1.065 |
| d=10 | UWYK Anc (matched resolution) | 10984 | 0.1496 | 0.768 | 0.182 | 1.026 | 1.061 |
| d=10 | UWYK No-Anc (matched label) | 10984 | 0.1535 | 0.768 | 0.182 | 1.000 | 1.001 |
| d=10 | UWYK Anc (matched label) | 10984 | 0.1463 | 0.768 | 0.182 | 1.000 | 1.001 |
| d=10 | CausalPFN-C (matched resolution) | 10984 | 0.1489 | 0.050 | 1.000 | 1.456 | 1.456 |
| d=10 | CausalPFN-C (matched label) | 10984 | 0.1528 | 0.050 | 1.000 | 1.028 | 1.028 |
| d=10 | CausalPFN (published two-stage) | 10984 | 0.1482 | 0.050 | 1.000 | 1.539 | 1.539 |
| d=10 | UWYK-B No-Anc | 10984 | 0.1633 | 0.050 | 1.000 | 1.001 | 1.001 |
| d=10 | UWYK-B Anc | 10984 | 0.2014 | 0.042 | 0.992 | 1.008 | 1.008 |
| d=1 | Do-PFN | 10984 | 0.1688 | 0.008 | 0.942 | 0.869 | 3.092 |
| d=1 | Do-PFN 2D | 10984 | 0.1612 | 0.119 | 0.831 | 0.877 | 4.891 |
| d=1 | UWYK No-Anc | 10984 | 0.1508 | 0.768 | 0.182 | 1.000 | 1.001 |
| d=1 | UWYK Anc | 10984 | 0.1834 | 0.826 | 0.124 | 1.000 | 1.002 |
| d=1 | UWYK 2D No-Anc | 10984 | 0.0964 | 0.950 | 0.000 | 0.286 | 3.211 |
| d=1 | UWYK 2D Anc | 10984 | 0.0962 | 0.950 | 0.000 | 0.219 | 2.983 |
| d=1 | CausalPFN-C | 10984 | 0.1517 | 0.050 | 1.000 | 1.028 | 1.028 |
| d=1 | CausalPFN-C 2D | 10984 | 0.1434 | 0.050 | 1.000 | 1.497 | 1.502 |
| d=1 | Do-PFN (matched resolution) | 10984 | 0.1731 | 0.008 | 0.942 | 2.549 | 4.782 |
| d=1 | Do-PFN (matched label) | 10984 | 0.1557 | 0.050 | 1.000 | 1.074 | 1.075 |
| d=1 | UWYK No-Anc (matched resolution) | 10984 | 0.1490 | 0.768 | 0.182 | 1.027 | 1.064 |
| d=1 | UWYK Anc (matched resolution) | 10984 | 0.1553 | 0.768 | 0.182 | 1.028 | 1.068 |
| d=1 | UWYK No-Anc (matched label) | 10984 | 0.1629 | 0.768 | 0.182 | 1.000 | 1.001 |
| d=1 | UWYK Anc (matched label) | 10984 | 0.1636 | 0.768 | 0.182 | 1.000 | 1.001 |
| d=1 | CausalPFN-C (matched resolution) | 10984 | 0.1528 | 0.050 | 1.000 | 1.440 | 1.444 |
| d=1 | CausalPFN-C (matched label) | 10984 | 0.1590 | 0.050 | 1.000 | 1.025 | 1.032 |
| d=1 | CausalPFN (published two-stage) | 10984 | 0.1505 | 0.050 | 1.000 | 1.549 | 1.549 |
| d=1 | UWYK-B No-Anc | 10984 | 0.1488 | 0.050 | 1.000 | 1.001 | 1.001 |
| d=1 | UWYK-B Anc | 10984 | 0.1551 | 0.050 | 1.000 | 1.009 | 1.009 |
