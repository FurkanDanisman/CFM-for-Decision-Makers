"""Turn a prior record into the tensors the transformer consumes.

The context/query split is the one assumption the whole reproduction rests on,
and the training loop that would settle it no longer exists. What the repo does
record is the evaluation contract for ``task_type == 'do_regression'``
(``Do-PFN/datasets/__init__.py:222-227``)::

    train_ds.x = train_ds.x_obs
    train_ds.y = train_ds.y_obs
    test_ds.y  = test_ds.y_int
    test_ds.x  = deepcopy(test_ds.x_obs)
    test_ds.x[:, 0] = test_ds.x_int[:, 0]      # simulate intervention

So only column 0 of the interventional pass is used: the assigned treatment.
Every other query covariate stays at its pre-intervention value, which agrees
with the paper's Algorithm 1 ("pre-treatment values of covariates") and with
``predict_cate``, which holds covariates fixed and flips column 0.

``SPLICES`` below implements that reading plus the corruptions of it, so the
parity harness can test the assumption against the released weights instead of
taking it on faith.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

# ---------------------------------------------------------------------------
# where to cut the sequence
# ---------------------------------------------------------------------------


@dataclass
class BatchConfig:
    # The checkpoint's config contradicts itself here: bptt_extra_samples=128
    # implies a fixed 128-row query block, while min_eval_pos=8 and
    # eval_positions=[2090] imply something else. They cannot all be live.
    # 'paper' follows Algorithm 1: M_ob ~ U{10..2200}, M_in = 2200 - M_ob.
    eval_pos_strategy: str = "paper"      # 'paper' | 'fixed_query'
    min_eval_pos: int = 10                # paper's M_min
    min_query_rows: int = 1
    bptt_extra_samples: int = 128         # used by 'fixed_query'

    # Joint head only: the query treatment column carries no information once
    # the head predicts both arms. 'nan' keeps column 0 present so
    # ColumnMarkerEncoderStep still marks it, and lets NanHandlingEncoderStep
    # flag the value as unknown. 'zero' writes a constant placeholder instead.
    query_treatment: str = "nan"          # 'nan' | 'zero'

    # SETTLED BY THE PARITY HARNESS. The released borders span [-56.1, 256.1],
    # which looks like raw prior y -- but feeding context-standardised y scores
    # 2.06 +/- 0.26 nats BETTER in raw-y units over 64 paired batches (~8 sigma).
    # So the model expects a standardised target, matching the config's
    # transform_target=True, and the wide borders are simply what standardising
    # a heavy-tailed prior leaves behind: fine buckets (0.049) near zero and
    # very long tails.
    #
    # Consequence for downstream code: predictions come back in standardised
    # units. Multiply by the context sigma before reporting CATE.
    y_space: str = "zscore_ctx"           # 'raw' | 'zscore_ctx'


def sample_single_eval_pos(seed: int, seq_len: int, cfg: BatchConfig) -> int:
    """Number of context rows. Pure function of ``seed``."""
    rng = np.random.default_rng(seed)
    if cfg.eval_pos_strategy == "fixed_query":
        return seq_len - cfg.bptt_extra_samples
    if cfg.eval_pos_strategy != "paper":
        raise ValueError(f"unknown eval_pos_strategy {cfg.eval_pos_strategy!r}")
    hi = seq_len - cfg.min_query_rows
    return int(rng.integers(cfg.min_eval_pos, hi + 1))


# ---------------------------------------------------------------------------
# y space
# ---------------------------------------------------------------------------


def _apply_y_space(
    y_ctx: torch.Tensor, others: list[torch.Tensor], space: str
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Transform y using context-only statistics (config: normalize_on_train_only).

    Also returns the log-Jacobian needed to express the model's density back in
    raw-y nats. Without it, NLLs computed in different y spaces are in different
    units and cannot be compared: for y' = (y - mu)/sigma,

        -log p_raw(y) = -log p'(y') + log sigma
    """
    if space == "raw":
        return y_ctx, others, torch.zeros(1, y_ctx.shape[1])
    if space == "zscore_ctx":
        mean = y_ctx.mean(dim=0, keepdim=True)
        std = y_ctx.std(dim=0, keepdim=True).clamp_min(1e-6)
        return (
            (y_ctx - mean) / std,
            [(o - mean) / std for o in others],
            torch.log(std),
        )
    raise ValueError(f"unknown y_space {space!r}")


# ---------------------------------------------------------------------------
# splice variants
# ---------------------------------------------------------------------------


def _splice_spliced(rec, sep):
    """The inferred contract: observational covariates, intervened column 0."""
    x = rec["x_obs"].clone()
    x[sep:, :, 0] = rec["t_int01"][sep:]
    return x


def _splice_x_int_full(rec, sep):
    """Query rows taken wholesale from the interventional pass.

    This is what a naive reading of the prior's return value suggests. It leaks
    the intervention into every mediator, so covariates downstream of t reflect
    the assigned arm -- inconsistent with how predict_cate is used at inference.
    """
    x = rec["x_obs"].clone()
    x[sep:] = rec["x_int"][sep:]
    return x


def _splice_obs_only(rec, sep):
    """No intervention signal at all: column 0 stays observational.

    A lower bound -- the model cannot know which arm it is being asked about.
    """
    return rec["x_obs"].clone()


def _splice_flipped(rec, sep):
    """Correct splice, inverted treatment polarity.

    Do-PFN maps the LOWER raw level to model-facing 1. If that were backwards,
    this variant would be the faithful one.
    """
    x = rec["x_obs"].clone()
    x[sep:, :, 0] = 1.0 - rec["t_int01"][sep:]
    return x


def _splice_flipped_both(rec, sep):
    """Polarity inverted on context rows as well as query rows."""
    x = rec["x_obs"].clone()
    x[:, :, 0] = 1.0 - x[:, :, 0]
    x[sep:, :, 0] = 1.0 - rec["t_int01"][sep:]
    return x


SPLICES: dict[str, Callable[[dict, int], torch.Tensor]] = {
    "spliced": _splice_spliced,
    "x_int_full": _splice_x_int_full,
    "obs_only": _splice_obs_only,
    "flipped": _splice_flipped,
    "flipped_both": _splice_flipped_both,
}


# ---------------------------------------------------------------------------
# batch builders
# ---------------------------------------------------------------------------


def make_1d_batch(
    rec: dict[str, Any],
    sep: int,
    cfg: BatchConfig | None = None,
    splice: str = "spliced",
) -> dict[str, torch.Tensor]:
    """Do-PFN's own task: predict y under do(t) given an observational context."""
    cfg = cfg or BatchConfig()
    x = SPLICES[splice](rec, sep)

    y_ctx, (target,), logjac = _apply_y_space(
        rec["y_obs"][:sep], [rec["y_int"][sep:]], cfg.y_space
    )
    return {
        "train_x": x[:sep],       # (sep, B, F+1)
        "train_y": y_ctx,         # (sep, B)   -- model NaN-pads query rows
        "test_x": x[sep:],        # (S-sep, B, F+1)
        "target": target,         # (S-sep, B)
        "logjac": logjac,         # (1, B) add to NLL for raw-y nats
        "raw_y_ctx": rec["y_obs"][:sep],
        "raw_target": rec["y_int"][sep:],
        "single_eval_pos": sep,
    }


def make_joint_batch(
    rec: dict[str, Any],
    sep: int,
    cfg: BatchConfig | None = None,
) -> dict[str, torch.Tensor]:
    """The joint task: predict both arms from covariates alone.

    Context is byte-identical to ``make_1d_batch``. The only difference is that
    the query carries no treatment and the target is the pair.
    """
    cfg = cfg or BatchConfig()
    x = rec["x_obs"].clone()

    if cfg.query_treatment == "nan":
        x[sep:, :, 0] = float("nan")
    elif cfg.query_treatment == "zero":
        x[sep:, :, 0] = 0.0
    else:
        raise ValueError(f"unknown query_treatment {cfg.query_treatment!r}")

    y_ctx, (y0, y1), logjac = _apply_y_space(
        rec["y_obs"][:sep], [rec["y_do0"][sep:], rec["y_do1"][sep:]], cfg.y_space
    )
    return {
        "train_x": x[:sep],
        "train_y": y_ctx,
        "test_x": x[sep:],
        "y_do0": y0,              # (S-sep, B)
        "y_do1": y1,              # (S-sep, B)
        "logjac": logjac,         # (1, B); doubles for a joint over both arms
        "single_eval_pos": sep,
    }
