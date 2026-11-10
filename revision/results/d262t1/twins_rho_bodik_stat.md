# Twins: Bodik et al. (2025) cross-world correlation statistic, binary vs jittered

Justification of the Twins jittering: the cross-world correlation statistic of Bodik, Huang, Yu (2025, App. A.3.2)
-- Pearson correlation of the twins' outcomes (Y(0), Y(1)) within the most frequent value combinations of two
covariates (the three most frequent per pair of X1, X2, X3), plus the whole sample -- on the binary Twins outcomes and
on the jittered ones (make_worlds._corr_jitter: the jitter of covariate set d is calibrated on the first d
covariates; d = 10 and 1, w = 0.1, 0.25, 0.5), on the same families. 95% bootstrap interval of the binary value (2000
resamples of the group's families). Following Bodik et al. fully: every group is kept; the summary rows are over the
groups in which the binary correlation is defined (a group without deaths has none).

| Group | Families | Deaths | Binary (95% CI) | d=10 w=0.1 | d=10 w=0.25 | d=10 w=0.5 | d=1 w=0.1 | d=1 w=0.25 | d=1 w=0.5 | All jittered inside CI |
|---|---|---|---|---|---|---|---|---|---|---|
| whole sample | 11984 | 4238 | 0.649 (0.631–0.666) | 0.649 | 0.647 | 0.644 | 0.649 | 0.649 | 0.649 | True |
| X1=0, X2=0 | 7912 | 2754 | 0.653 (0.630–0.674) | 0.652 | 0.654 | 0.644 | 0.653 | 0.648 | 0.651 | True |
| X1=0, X2=1 | 3931 | 1469 | 0.646 (0.615–0.676) | 0.646 | 0.636 | 0.647 | 0.644 | 0.654 | 0.648 | True |
| X1=1, X2=0 | 101 | 9 | 0.241 (-0.044–0.623) | 0.277 | 0.368 | 0.481 | 0.291 | 0.277 | 0.385 | True |
| X1=0, X3=0 | 6937 | 1799 | 0.587 (0.558–0.616) | 0.588 | 0.594 | 0.611 | 0.586 | 0.592 | 0.613 | True |
| X1=0, X3=1 | 4906 | 2424 | 0.690 (0.666–0.712) | 0.689 | 0.683 | 0.671 | 0.691 | 0.691 | 0.677 | True |
| X1=1, X3=0 | 95 | 7 | -- (--–--) | 0.015 | 0.462 | 0.364 | 0.135 | 0.289 | 0.350 |  |
| X2=0, X3=0 | 4734 | 1170 | 0.573 (0.538–0.608) | 0.575 | 0.586 | 0.600 | 0.575 | 0.573 | 0.598 | True |
| X2=0, X3=1 | 3279 | 1593 | 0.703 (0.676–0.731) | 0.701 | 0.700 | 0.678 | 0.704 | 0.701 | 0.691 | True |
| X2=1, X3=0 | 2298 | 636 | 0.607 (0.559–0.653) | 0.607 | 0.605 | 0.625 | 0.602 | 0.622 | 0.638 | True |
| **Range** (groups with a defined correlation) | | | 0.241–0.703 | 0.277–0.701 | 0.368–0.700 | 0.481–0.678 | 0.291–0.704 | 0.277–0.701 | 0.385–0.691 | |
