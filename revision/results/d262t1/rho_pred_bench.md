# Implied (predicted) ρ of the 2D heads on the benchmarks

Per dataset: median over its queries of the correlation of the head's predicted joint. Mean ± sd over datasets (n). For the controlled-ρ benchmarks, the true ρ of the same datasets is shown next to it.

| model | benchmark | setting | n | predicted ρ (mean ± sd) | true ρ (mean ± sd) |
|---|---|---|---|---|---|
| CausalPFN-C 2D | CaseStudy | Backdoor_Criterion | 2400 | +0.042 ± 0.050 | — |
| CausalPFN-C 2D | CaseStudy | Frontdoor_Criterion | 2400 | +0.042 ± 0.047 | — |
| CausalPFN-C 2D | CaseStudy | Observed_Confounder | 2400 | +0.026 ± 0.029 | — |
| CausalPFN-C 2D | CaseStudy | Observed_Mediator | 2400 | +0.044 ± 0.096 | — |
| CausalPFN-C 2D | CaseStudy | Observed_Mediator_and_Confounder | 2400 | +0.034 ± 0.050 | — |
| CausalPFN-C 2D | CaseStudy | Unobserved_Confounder | 2400 | +0.080 ± 0.074 | — |
| CausalPFN-C 2D | ComplexMech | n=10 | 116 | +0.181 ± 0.192 | — |
| CausalPFN-C 2D | ComplexMech | n=20 | 127 | +0.130 ± 0.166 | — |
| CausalPFN-C 2D | ComplexMech | n=30 | 127 | +0.112 ± 0.124 | — |
| CausalPFN-C 2D | ComplexMech | n=40 | 115 | +0.127 ± 0.115 | — |
| CausalPFN-C 2D | ComplexMech | n=5 | 115 | +0.219 ± 0.203 | — |
| CausalPFN-C 2D | ComplexMech | n=50 | 129 | +0.115 ± 0.109 | — |
| CausalPFN-C 2D | RealCause | ACIC | 10 | +0.207 ± 0.086 | — |
| CausalPFN-C 2D | RealCause | CPS | 100 | +0.085 ± 0.031 | — |
| CausalPFN-C 2D | RealCause | IHDP | 100 | +0.024 ± 0.021 | — |
| CausalPFN-C 2D | RealCause | PSID | 100 | +0.313 ± 0.066 | — |
| CausalPFN-C 2D | RealCause | PSID_bal | 100 | +0.178 ± 0.049 | — |
| CausalPFN-C 2D | rho_ctrl_cm | ComplexMech bin0 | 100 | +0.368 ± 0.228 | -0.201 ± 0.147 |
| CausalPFN-C 2D | rho_ctrl_cm | ComplexMech bin1 | 100 | +0.365 ± 0.215 | +0.197 ± 0.144 |
| CausalPFN-C 2D | rho_ctrl_cm | ComplexMech bin2 | 100 | +0.405 ± 0.208 | +0.745 ± 0.147 |
| CausalPFN-C 2D | rho_ctrl_cs | Backdoor_Criterion bin0 | 180 | +0.037 ± 0.029 | -0.226 ± 0.131 |
| CausalPFN-C 2D | rho_ctrl_cs | Backdoor_Criterion bin1 | 180 | +0.036 ± 0.029 | +0.245 ± 0.151 |
| CausalPFN-C 2D | rho_ctrl_cs | Backdoor_Criterion bin2 | 180 | +0.039 ± 0.038 | +0.755 ± 0.151 |
| CausalPFN-C 2D | rho_ctrl_cs | Frontdoor_Criterion bin0 | 180 | +0.039 ± 0.035 | -0.222 ± 0.132 |
| CausalPFN-C 2D | rho_ctrl_cs | Frontdoor_Criterion bin1 | 180 | +0.035 ± 0.029 | +0.242 ± 0.143 |
| CausalPFN-C 2D | rho_ctrl_cs | Frontdoor_Criterion bin2 | 180 | +0.041 ± 0.031 | +0.744 ± 0.143 |
| CausalPFN-C 2D | rho_ctrl_cs | Observed_Confounder bin0 | 180 | +0.029 ± 0.035 | -0.239 ± 0.138 |
| CausalPFN-C 2D | rho_ctrl_cs | Observed_Confounder bin1 | 180 | +0.025 ± 0.022 | +0.255 ± 0.142 |
| CausalPFN-C 2D | rho_ctrl_cs | Observed_Confounder bin2 | 180 | +0.027 ± 0.030 | +0.741 ± 0.140 |
| CausalPFN-C 2D | rho_ctrl_cs | Unobserved_Confounder bin0 | 162 | +0.041 ± 0.043 | -0.144 ± 0.117 |
| CausalPFN-C 2D | rho_ctrl_cs | Unobserved_Confounder bin1 | 180 | +0.043 ± 0.040 | +0.304 ± 0.150 |
| CausalPFN-C 2D | rho_ctrl_cs | Unobserved_Confounder bin2 | 180 | +0.061 ± 0.051 | +0.914 ± 0.116 |
| Do-PFN 2D | CaseStudy | Backdoor_Criterion | 1800 | +0.751 ± 0.300 | — |
| Do-PFN 2D | CaseStudy | Frontdoor_Criterion | 1800 | +0.753 ± 0.296 | — |
| Do-PFN 2D | CaseStudy | Observed_Confounder | 1800 | +0.601 ± 0.299 | — |
| Do-PFN 2D | CaseStudy | Observed_Mediator | 1800 | +0.141 ± 0.456 | — |
| Do-PFN 2D | CaseStudy | Observed_Mediator_and_Confounder | 1800 | +0.480 ± 0.305 | — |
| Do-PFN 2D | CaseStudy | Unobserved_Confounder | 1800 | +0.649 ± 0.236 | — |
| Do-PFN 2D | ComplexMech | n=10 | 116 | +0.394 ± 0.320 | — |
| Do-PFN 2D | ComplexMech | n=20 | 127 | +0.521 ± 0.280 | — |
| Do-PFN 2D | ComplexMech | n=30 | 127 | +0.599 ± 0.279 | — |
| Do-PFN 2D | ComplexMech | n=40 | 115 | +0.642 ± 0.214 | — |
| Do-PFN 2D | ComplexMech | n=5 | 115 | +0.390 ± 0.306 | — |
| Do-PFN 2D | ComplexMech | n=50 | 129 | +0.630 ± 0.313 | — |
| Do-PFN 2D | RealCause | ACIC | 10 | +0.830 ± 0.059 | — |
| Do-PFN 2D | RealCause | CPS | 100 | +0.748 ± 0.027 | — |
| Do-PFN 2D | RealCause | IHDP | 100 | +0.602 ± 0.124 | — |
| Do-PFN 2D | RealCause | PSID | 100 | +0.496 ± 0.020 | — |
| Do-PFN 2D | RealCause | PSID_bal | 100 | +0.442 ± 0.033 | — |
| Do-PFN 2D | rho_ctrl_cm | ComplexMech bin0 | 100 | +0.467 ± 0.289 | -0.201 ± 0.147 |
| Do-PFN 2D | rho_ctrl_cm | ComplexMech bin1 | 100 | +0.447 ± 0.287 | +0.197 ± 0.144 |
| Do-PFN 2D | rho_ctrl_cm | ComplexMech bin2 | 100 | +0.451 ± 0.276 | +0.745 ± 0.147 |
| Do-PFN 2D | rho_ctrl_cs | Backdoor_Criterion bin0 | 180 | +0.750 ± 0.299 | -0.226 ± 0.131 |
| Do-PFN 2D | rho_ctrl_cs | Backdoor_Criterion bin1 | 180 | +0.746 ± 0.317 | +0.245 ± 0.151 |
| Do-PFN 2D | rho_ctrl_cs | Backdoor_Criterion bin2 | 180 | +0.765 ± 0.290 | +0.755 ± 0.151 |
| Do-PFN 2D | rho_ctrl_cs | Frontdoor_Criterion bin0 | 180 | +0.761 ± 0.279 | -0.222 ± 0.132 |
| Do-PFN 2D | rho_ctrl_cs | Frontdoor_Criterion bin1 | 180 | +0.757 ± 0.305 | +0.242 ± 0.143 |
| Do-PFN 2D | rho_ctrl_cs | Frontdoor_Criterion bin2 | 180 | +0.755 ± 0.290 | +0.744 ± 0.143 |
| Do-PFN 2D | rho_ctrl_cs | Observed_Confounder bin0 | 180 | +0.598 ± 0.304 | -0.239 ± 0.138 |
| Do-PFN 2D | rho_ctrl_cs | Observed_Confounder bin1 | 180 | +0.619 ± 0.291 | +0.255 ± 0.142 |
| Do-PFN 2D | rho_ctrl_cs | Observed_Confounder bin2 | 180 | +0.607 ± 0.297 | +0.741 ± 0.140 |
| Do-PFN 2D | rho_ctrl_cs | Unobserved_Confounder bin0 | 162 | +0.601 ± 0.253 | -0.144 ± 0.117 |
| Do-PFN 2D | rho_ctrl_cs | Unobserved_Confounder bin1 | 180 | +0.618 ± 0.262 | +0.304 ± 0.150 |
| Do-PFN 2D | rho_ctrl_cs | Unobserved_Confounder bin2 | 180 | +0.645 ± 0.248 | +0.914 ± 0.116 |
| UWYK 2D Anc | rho_ctrl_cm | ComplexMech bin0 | 100 | +0.487 ± 0.398 | -0.201 ± 0.147 |
| UWYK 2D Anc | rho_ctrl_cm | ComplexMech bin1 | 100 | +0.500 ± 0.361 | +0.197 ± 0.144 |
| UWYK 2D Anc | rho_ctrl_cm | ComplexMech bin2 | 100 | +0.582 ± 0.352 | +0.745 ± 0.147 |
| UWYK 2D Anc | rho_ctrl_cs | Backdoor_Criterion bin0 | 180 | +0.662 ± 0.187 | -0.226 ± 0.131 |
| UWYK 2D Anc | rho_ctrl_cs | Backdoor_Criterion bin1 | 180 | +0.673 ± 0.167 | +0.245 ± 0.151 |
| UWYK 2D Anc | rho_ctrl_cs | Backdoor_Criterion bin2 | 180 | +0.677 ± 0.180 | +0.755 ± 0.151 |
| UWYK 2D Anc | rho_ctrl_cs | Frontdoor_Criterion bin0 | 180 | +0.484 ± 0.259 | -0.222 ± 0.132 |
| UWYK 2D Anc | rho_ctrl_cs | Frontdoor_Criterion bin1 | 180 | +0.464 ± 0.256 | +0.242 ± 0.143 |
| UWYK 2D Anc | rho_ctrl_cs | Frontdoor_Criterion bin2 | 180 | +0.472 ± 0.277 | +0.744 ± 0.143 |
| UWYK 2D Anc | rho_ctrl_cs | Observed_Confounder bin0 | 180 | +0.450 ± 0.255 | -0.239 ± 0.138 |
| UWYK 2D Anc | rho_ctrl_cs | Observed_Confounder bin1 | 180 | +0.462 ± 0.260 | +0.255 ± 0.142 |
| UWYK 2D Anc | rho_ctrl_cs | Observed_Confounder bin2 | 180 | +0.456 ± 0.265 | +0.741 ± 0.140 |
| UWYK 2D Anc | rho_ctrl_cs | Unobserved_Confounder bin0 | 162 | +0.276 ± 0.301 | -0.144 ± 0.117 |
| UWYK 2D Anc | rho_ctrl_cs | Unobserved_Confounder bin1 | 180 | +0.257 ± 0.266 | +0.304 ± 0.150 |
| UWYK 2D Anc | rho_ctrl_cs | Unobserved_Confounder bin2 | 180 | +0.350 ± 0.284 | +0.914 ± 0.116 |
| UWYK 2D No-Anc | CaseStudy | Backdoor_Criterion | 2400 | +0.485 ± 0.268 | — |
| UWYK 2D No-Anc | CaseStudy | Frontdoor_Criterion | 2400 | +0.485 ± 0.275 | — |
| UWYK 2D No-Anc | CaseStudy | Observed_Confounder | 2400 | +0.289 ± 0.284 | — |
| UWYK 2D No-Anc | CaseStudy | Observed_Mediator | 2400 | -0.103 ± 0.150 | — |
| UWYK 2D No-Anc | CaseStudy | Observed_Mediator_and_Confounder | 2400 | +0.117 ± 0.210 | — |
| UWYK 2D No-Anc | CaseStudy | Unobserved_Confounder | 2400 | +0.324 ± 0.297 | — |
| UWYK 2D No-Anc | ComplexMech | n=10 | 116 | +0.350 ± 0.302 | — |
| UWYK 2D No-Anc | ComplexMech | n=20 | 127 | +0.411 ± 0.283 | — |
| UWYK 2D No-Anc | ComplexMech | n=30 | 127 | +0.372 ± 0.274 | — |
| UWYK 2D No-Anc | ComplexMech | n=40 | 115 | +0.452 ± 0.266 | — |
| UWYK 2D No-Anc | ComplexMech | n=5 | 115 | +0.393 ± 0.325 | — |
| UWYK 2D No-Anc | ComplexMech | n=50 | 129 | +0.484 ± 0.275 | — |
| UWYK 2D No-Anc | RealCause | ACIC | 10 | +0.451 ± 0.266 | — |
| UWYK 2D No-Anc | RealCause | CPS | 100 | +0.546 ± 0.135 | — |
| UWYK 2D No-Anc | RealCause | IHDP | 100 | +0.139 ± 0.155 | — |
| UWYK 2D No-Anc | RealCause | PSID | 100 | +0.714 ± 0.118 | — |
| UWYK 2D No-Anc | RealCause | PSID_bal | 100 | +0.563 ± 0.050 | — |
| UWYK 2D No-Anc | rho_ctrl_cm | ComplexMech bin0 | 100 | +0.487 ± 0.403 | -0.201 ± 0.147 |
| UWYK 2D No-Anc | rho_ctrl_cm | ComplexMech bin1 | 100 | +0.496 ± 0.373 | +0.197 ± 0.144 |
| UWYK 2D No-Anc | rho_ctrl_cm | ComplexMech bin2 | 100 | +0.576 ± 0.351 | +0.745 ± 0.147 |
| UWYK 2D No-Anc | rho_ctrl_cs | Backdoor_Criterion bin0 | 180 | +0.550 ± 0.240 | -0.226 ± 0.131 |
| UWYK 2D No-Anc | rho_ctrl_cs | Backdoor_Criterion bin1 | 180 | +0.576 ± 0.234 | +0.245 ± 0.151 |
| UWYK 2D No-Anc | rho_ctrl_cs | Backdoor_Criterion bin2 | 180 | +0.584 ± 0.234 | +0.755 ± 0.151 |
| UWYK 2D No-Anc | rho_ctrl_cs | Frontdoor_Criterion bin0 | 180 | +0.593 ± 0.228 | -0.222 ± 0.132 |
| UWYK 2D No-Anc | rho_ctrl_cs | Frontdoor_Criterion bin1 | 180 | +0.576 ± 0.228 | +0.242 ± 0.143 |
| UWYK 2D No-Anc | rho_ctrl_cs | Frontdoor_Criterion bin2 | 180 | +0.574 ± 0.251 | +0.744 ± 0.143 |
| UWYK 2D No-Anc | rho_ctrl_cs | Observed_Confounder bin0 | 180 | +0.376 ± 0.272 | -0.239 ± 0.138 |
| UWYK 2D No-Anc | rho_ctrl_cs | Observed_Confounder bin1 | 180 | +0.384 ± 0.277 | +0.255 ± 0.142 |
| UWYK 2D No-Anc | rho_ctrl_cs | Observed_Confounder bin2 | 180 | +0.372 ± 0.284 | +0.741 ± 0.140 |
| UWYK 2D No-Anc | rho_ctrl_cs | Unobserved_Confounder bin0 | 162 | +0.277 ± 0.303 | -0.144 ± 0.117 |
| UWYK 2D No-Anc | rho_ctrl_cs | Unobserved_Confounder bin1 | 180 | +0.258 ± 0.268 | +0.304 ± 0.150 |
| UWYK 2D No-Anc | rho_ctrl_cs | Unobserved_Confounder bin2 | 180 | +0.355 ± 0.286 | +0.914 ± 0.116 |
