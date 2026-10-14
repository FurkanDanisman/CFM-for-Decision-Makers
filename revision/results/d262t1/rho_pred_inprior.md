# In-prior predicted ρ of the 2D heads

Each 2D head on 1000 datasets of its own training prior (E2 new-world data, shards 0–9), the paper's harness. Predicted ρ per query = correlation of the head's predicted joint on the interior grid (piecewise-uniform cells). It is the model's prediction, not the prior's true ρ, and includes the model's uncertainty about the means.

## Per dataset (median over its queries)

| model | n | 5% | 25% | 50% | 75% | 95% | mean | > 0.99 | > 0.9 | < 0.5 | < 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Do-PFN 2D | 1000 | -0.001 | +0.004 | +0.283 | +0.690 | +0.928 | +0.357 | 0.000 | 0.090 | 0.627 | 0.087 |
| UWYK 2D | 1000 | +0.001 | +0.147 | +0.331 | +0.631 | +0.942 | +0.394 | 0.002 | 0.090 | 0.655 | 0.049 |
| CausalPFN-C 2D | 1000 | +0.010 | +0.043 | +0.083 | +0.141 | +0.295 | +0.108 | 0.000 | 0.000 | 0.993 | 0.014 |

## All queries pooled

| model | n | 5% | 25% | 50% | 75% | 95% | mean | > 0.99 | > 0.9 | < 0.5 | < 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Do-PFN 2D | 500000 | -0.002 | +0.007 | +0.292 | +0.695 | +0.933 | +0.363 | 0.000 | 0.088 | 0.622 | 0.103 |
| UWYK 2D | 500000 | -0.011 | +0.147 | +0.340 | +0.650 | +0.948 | +0.402 | 0.002 | 0.092 | 0.639 | 0.057 |
| CausalPFN-C 2D | 500000 | +0.005 | +0.040 | +0.085 | +0.151 | +0.337 | +0.116 | 0.000 | 0.000 | 0.981 | 0.026 |
