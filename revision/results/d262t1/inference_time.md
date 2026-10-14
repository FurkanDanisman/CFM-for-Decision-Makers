### Inference time, RealCause IHDP realization 0

Median [Q1, Q3] in seconds: forward over 20 timed repetitions after 3 warm-up runs, density steps over 5 after 1. Forward on NVIDIA L40S; ITE and ATE density steps are the NLL scorer's numpy routines (CPU). UWYK models: no-ancestral mode only.

| Model | n_context | n_query | Forward (s) | ITE density (s) | ATE density (s) |
|---|---|---|---|---|---|
| Do-PFN | 672 | 75 | 0.2745 [0.2737, 0.2756] | 16.61 [16.6, 16.64] | 0.009787 [0.00978, 0.009861] |
| Do-PFN 2D | 672 | 75 | 0.1358 [0.1356, 0.1364] | 15.43 [15.36, 15.45] | 0.009614 [0.009581, 0.009819] |
| UWYK No-Anc | 672 | 75 | 0.1263 [0.1261, 0.1266] | 15.6 [15.55, 15.61] | 0.009531 [0.009522, 0.00956] |
| UWYK No-Anc 2D | 672 | 75 | 0.07268 [0.07253, 0.07276] | 15.99 [15.99, 16.12] | 0.01012 [0.01012, 0.01014] |
| CausalPFN-C | 672 | 75 | 0.03242 [0.03207, 0.03247] | 0.04684 [0.04664, 0.0476] | 0.008982 [0.008975, 0.009087] |
| CausalPFN-C 2D | 672 | 75 | 0.01578 [0.01575, 0.01597] | 16.4 [16.18, 16.45] | 0.00932 [0.0093, 0.009329] |
