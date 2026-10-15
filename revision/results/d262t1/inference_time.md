### Inference time, RealCause IHDP realization 0

Median [Q1, Q3] in seconds: forward over 20 timed repetitions after 3 warm-up runs, density steps over 5 after 1. Forward on NVIDIA L40S; ITE and ATE density steps are the NLL scorer's numpy routines (CPU). UWYK models: no-ancestral mode only.

| Model | n_context | n_query | Forward (s) | ITE density (s) | ATE density (s) |
|---|---|---|---|---|---|
| Do-PFN | 672 | 75 | 0.2739 [0.2737, 0.274] | 15.93 [15.92, 15.94] | 0.009551 [0.009542, 0.009577] |
| Do-PFN 2D | 672 | 75 | 0.1345 [0.1342, 0.1348] | 14.65 [14.65, 14.66] | 0.009604 [0.009538, 0.009605] |
| UWYK No-Anc | 672 | 75 | 0.1267 [0.1266, 0.1269] | 14.89 [14.89, 14.9] | 0.009847 [0.009845, 0.009877] |
| UWYK No-Anc 2D | 672 | 75 | 0.0727 [0.07263, 0.07281] | 15.53 [15.53, 15.53] | 0.009552 [0.009534, 0.009583] |
| CausalPFN-C | 672 | 75 | 0.0317 [0.03169, 0.03175] | 0.04303 [0.04301, 0.04336] | 0.008888 [0.008877, 0.008954] |
| CausalPFN-C 2D | 672 | 75 | 0.01588 [0.01586, 0.01594] | 15.21 [15.21, 15.21] | 0.009019 [0.009005, 0.009104] |
| UWYK 1D binary | 672 | 75 | 0.1268 [0.1267, 0.1269] | 13.71 [13.71, 13.71] | 0.009707 [0.009672, 0.009756] |
| UWYK, matched resolution | 672 | 75 | 0.1262 [0.1261, 0.1263] | 12.65 [12.61, 12.65] | 0.009605 [0.009603, 0.009661] |
| UWYK, matched labels | 672 | 75 | 0.1268 [0.1267, 0.1269] | 15.25 [15.24, 15.25] | 0.009727 [0.009724, 0.009746] |
| CausalPFN two-stage (published) | 672 | 75 | 0.03179 [0.03175, 0.03181] | 0.04393 [0.04392, 0.04399] | 0.009087 [0.009059, 0.009123] |
| CausalPFN-C, matched resolution | 672 | 75 | 0.03133 [0.0313, 0.03137] | 0.02429 [0.02427, 0.02432] | 0.00903 [0.008981, 0.009093] |
| CausalPFN-C, matched labels | 672 | 75 | 0.03191 [0.03189, 0.03195] | 0.04297 [0.04294, 0.04327] | 0.008989 [0.008987, 0.00903] |

Not timed (failed): dopfn1d_mr, dopfn1d_ml
