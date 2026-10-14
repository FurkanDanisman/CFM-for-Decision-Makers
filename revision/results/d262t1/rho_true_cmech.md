# True ρ(x) on ComplexMech, pre-treatment realizations (exact, from UWYK's generator)

ρ(x) = corr(Y(1), Y(0) | X = x) by redrawing only Y's noise (shared across arms). `verified` = regenerated outcomes equal the stored ones (processed arrays when the file has no raw ones) AND Y's mechanism reproduces them; only those are summarised. ρ per realization = median over its query units; `deterministic` = units where no noise reaches Y.

| n | pre-treatment realizations | verified | ρ median (min, max) over realizations | share of units with ρ > 0.99 | deterministic units |
|---|---|---|---|---|---|
| 5 | 37 | 37 | +1.000 (-0.643, +1.000) | 0.61 | 901/3700 |
| 10 | 15 | 15 | +1.000 (+0.178, +1.000) | 0.66 | 600/1500 |
| 20 | 1 | 1 | +0.215 (+0.215, +0.215) | 0.00 | 0/100 |
| 30 | 3 | 3 | +1.000 (+0.875, +1.000) | 0.73 | 0/300 |
| 40 | 2 | 2 | +1.000 (+1.000, +1.000) | 1.00 | 100/200 |
| 50 | 1 | 1 | +0.946 (+0.946, +0.946) | 0.08 | 0/100 |
