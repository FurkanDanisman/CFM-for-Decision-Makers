# UWYK (not trained here)

`uwyk_reproduce_best_model.pt` is the Use-What-You-Know authors' own released
checkpoint. We do not retrain it; doing so would confound a head comparison
with a training-recipe difference.

```
$UWYK/experiments/checkpoints/full_conditioned_model/
    final_earlytest_full_conditioning_*/best_model.pt
    final_earlytest_full_conditioning_*/best_model_config.yaml
```

Copied to `Required_checkpoints/uwyk_reproduce_best_model.pt`. The config
must travel with it — `GraphConditionedInterventionalPFNSklearn` needs both.

## Structure, for the 1D/2D comparison

UWYK is the 1D counterpart to Graph2D: both condition on a partial ancestor
matrix, both are evaluated through the same harness with the same adjacency
modes. The head is a bar distribution with $K$ bars plus two half-normal tail
components, output dimension $K+4$ (the last two entries are raw tail
scales, and must be excluded before the softmax — including them mixes tail
parameters into the density).

Its PAM validator rejects matrices with $T[i,i] = 1$, which is why the
evaluation uses `v3ab_only` rather than `v3_family`.

## If you did want to retrain it

The training code is in the UWYK repository, not here. Reproducing our
numbers does not require it: the checkpoint is in `Required_checkpoints/`.
