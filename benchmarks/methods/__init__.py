"""Method pipelines for the RealCause / IHDP / ACIC benchmarks.

Each module exposes a single function that consumes a `CATE_Dataset` and
returns a length-N array of per-query CATE predictions.

- dopfn.dopfn_pipeline           — Do-PFN baseline (their DoPFNRegressor)
- uwyk.uwyk_ancestral_pipeline   — UWYK with full graph (Ancestral Info)
- ours.ours_pipeline             — Our InterventionalPFN + MALC variants

Metrics (PEHE, ε_ATE) live in `benchmarks/run_one.py` following UWYK's
`RealCauseEval/run_baselines/eval.py` definitions.
"""
