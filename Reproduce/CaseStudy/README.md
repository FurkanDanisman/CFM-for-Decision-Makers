# Case Study

Six DoPFN SCM case studies x 8 covariate counts (d = 2,3,5,10,20,30,40,50)
x 3 ATE shifts (0, +2, -2), at context N=1000.

```bash
export DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit
export OUT_ROOT=$SCRATCH/cs_dvar_dens

sbatch Reproduce/CaseStudy/DataGeneration/generate.sbatch
for S in 0 +2 -2; do
  OUT_ROOT=$OUT_ROOT SHIFT=$S \
    sbatch benchmarks/cluster/submit_cs_dvar_density.sbatch   # 48 tasks each
done
bash   Reproduce/CaseStudy/PEHE_ATE/summarize.sh
MODELS="" CPF=$OUT_ROOT sbatch case_study/density_eval/submit_score_cpfn_schema.sbatch
bash   Reproduce/CaseStudy/Calibration/tables.sh
bash   Reproduce/CaseStudy/PlotGeneration/plots.sh
```

## The six cases

`Observed_Confounder`, `Backdoor_Criterion`, `Observed_Mediator`,
`Observed_Mediator_and_Confounder`, `Unobserved_Confounder`,
`Frontdoor_Criterion`.

## Adjacency: use `case_family`, never `v3ab_only`

`build_anc_v3a`/`v3b` hardcode a confounder layout (X->T, X->Y) for every
dataset. That is the wrong graph for four of the six: the two mediator cases
are T->X->Y, `Unobserved_Confounder`'s X is spurious, and
`Frontdoor_Criterion` has a front-door structure. Conditioning on a
confounder graph there feeds the model WRONG ancestor information, so "anc"
loses to "noanc" for reasons unrelated to the method.

`case_family` builds the per-case DAG via `build_case_adj` and emits the same
`noanc` / `v3a` / `v3b` tags. It is also what the point sweep used
(`case_study/cluster/dsweep_eval.sh`), so PEHE and calibration agree.

## Data layout

The npz tree is `<shift>/d<K>/<Case>/N<ctx>/`, NOT DoPFN's pkls at
`<root>/<Case>/*.pkl`. `benchmarks/scm_case_study_dataset.py` swaps in the
npz backend only when `CASE_STUDY_DATA_ROOT` is set, and that backend needs
`CASE_STUDY_N` to pick the `N<ctx>` subdirectory. Setting `DOPFN_DATA_ROOT`
instead leaves the pkl loader active: it finds no `*.pkl`, reports
`summary (n=0)`, writes nothing, and exits 0.
