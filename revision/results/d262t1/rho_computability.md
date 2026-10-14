# Where the true ρ(x) is computable

Definition (paper Eq. `eq:rho`, from Bodik, Huang & Yu 2026, arXiv 2507.12581):
ρ(x) = corr(Y^do(1), Y^do(0) | X = x). Their setting (§2.2) assumes Y(T) = f_Y(T, X, ε) with ε ⟂ T, X, i.e.
pre-treatment covariates. Under ε ⟂ X, ρ(x) is exact by fixing x and redrawing the remaining noise
(shared between the two arms, as in every benchmark here). A dataset is computable when none of its
covariates is a consequence of the treatment or the outcome (no mediator T → X_k → Y, no descendant of Y).

| benchmark | datasets | computable | reason |
|---|---|---|---|
| RealCause IHDP | 100 | all | outcomes drawn independently given real pre-treatment covariates; ρ = 0 by construction |
| RealCause ACIC | all settings | all | as IHDP |
| RealCause CPS | all draws | all | as IHDP |
| RealCause PSID | all draws | all | as IHDP |
| RealCause PSID_bal | all draws | all | as IHDP |
| Case study: Observed Confounder | all | all | covariate C is a root |
| Case study: Backdoor | all | all | C is a root; mediator M hidden (redrawn) |
| Case study: Frontdoor | all | all | C is a root; mediator M hidden (redrawn) |
| Case study: Unobserved Confounder | all | all | C is a root; hidden U is independent of C (breaks only ε ⟂ T) |
| Case study: Observed Mediator | all | none | covariate M is caused by T |
| Case study: Mediator + Confounder | all | none | covariate M is caused by T |
| ComplexMech n = 5 | 100 | 37 | 32 mediator only, 16 Y-descendant only, 15 both |
| ComplexMech n = 10 | 100 | 15 | 30 / 17 / 38 |
| ComplexMech n = 20 | 100 | 1 | 21 / 11 / 67 |
| ComplexMech n = 30 | 100 | 3 | 11 / 6 / 80 |
| ComplexMech n = 40 | 100 | 2 | 10 / 10 / 78 |
| ComplexMech n = 50 | 100 | 1 | 13 / 10 / 76 |

Case-study equations: case_study/generation.py (_observed_confounder, _backdoor_criterion,
_frontdoor_criterion, _unobserved_confounder, _observed_mediator, _observed_mediator_and_confounder).
ComplexMech counts: benchmarks/rho/count_pretreatment.py on $SCRATCH/cmech_data_v2, hide_0.0 (the main
tables), from each realization's stored graph (adj_matrix, order [T, Y, covariates]); no incomplete graphs.
RealCause: Table 8 (benchmarks/arm_noise_table.py) uses stored μ (IHDP, ACIC) or K repeated draws
(CPS, PSID) and gives ρ̂ ≈ 0.
