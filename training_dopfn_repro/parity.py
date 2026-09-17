"""Parity harness: validate the reconstructed pipeline without training anything.

Do-PFN's training script was never published, so several pieces of the data
pipeline are reconstructed rather than observed -- the context/query splice, the
treatment polarity, the binarization strategy, the space y is fed in. Normally
those would be untestable.

But we have the trained weights *and* the fitted bar-distribution borders. So we
can run the released model over batches from our reconstructed prior and measure
held-out NLL on the query rows. A faithful reconstruction scores well; a wrong
assumption shows up immediately as a worse score. Each assumption is exposed as
a variant so the comparison is head-to-head on identical SCM draws.

This costs about a GPU-minute and gates everything downstream. Run it before
committing a single GPU-day.

Usage
-----
    uv run --python 3.10 --with "torch==2.1.*" --with "numpy<2" --with networkx \\
        --with scipy --with dill --with tqdm --with scikit-learn --with einops \\
        --with "pandas<2.2" python training_dopfn_repro/parity.py --n-batches 24

Note the torch pin: Do-PFN's ``model/layer.py`` does
``from torch.nn.modules.transformer import Optional``, which only resolves on
torch 2.1.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from dataclasses import replace

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from training_dopfn_repro.batch import (  # noqa: E402
    SPLICES,
    BatchConfig,
    make_1d_batch,
    sample_single_eval_pos,
)
from training_dopfn_repro.prior import (  # noqa: E402
    DOPFN_SRC,
    PriorConfig,
    sample_batch,
)


@contextlib.contextmanager
def _in_dopfn_root():
    """model_builder.load_model ignores its path argument and hardcodes
    'artifacts/...', so the process CWD has to be the Do-PFN root."""
    prev = os.getcwd()
    if DOPFN_SRC not in sys.path:
        sys.path.insert(0, DOPFN_SRC)
    os.chdir(DOPFN_SRC)
    try:
        yield
    finally:
        os.chdir(prev)


def load_released_model(device: str = "cpu"):
    """Load the released Do-PFN checkpoint: architecture pickle + trained weights."""
    import pickle

    with _in_dopfn_root():
        with open("artifacts/dopfn_model.pkl", "rb") as fh:
            model = pickle.load(fh)
        from scripts.transformer_prediction_interface.model_builder import Checkpoint

        ckpt = Checkpoint.load(
            "artifacts/model_submitit_0ccc_id_171b69db_epoch_-1.cpkt"
        )
    missing, unexpected = model.load_state_dict(ckpt.state_dict, strict=True)
    model.eval().to(device)
    return model, ckpt.config


@torch.no_grad()
def query_nll(model, batch: dict, device: str = "cpu") -> torch.Tensor:
    """Negative log density of the true query outcomes, in **raw-y nats**.

    The log-Jacobian puts every y_space variant in the same units, so their
    NLLs are directly comparable.
    """
    logits = model(
        batch["train_x"].to(device),
        batch["train_y"].to(device),
        batch["test_x"].to(device),
        only_return_standard_out=True,
    )
    loss = model.criterion(logits, batch["target"].to(device))  # (M, B)
    loss = loss + batch["logjac"].to(device)                    # -> raw-y nats
    return loss.reshape(-1)


def gaussian_reference_nll(batch: dict) -> torch.Tensor:
    """NLL of a context-fitted Gaussian on the query outcomes, in raw-y nats.

    Variants that alter the data-generating process (binary_strategy) produce a
    different task distribution, so their absolute NLLs are not comparable --
    an easier distribution scores better regardless of faithfulness. Subtracting
    a reference fitted inside each distribution cancels that difficulty term,
    leaving how much structure the model actually extracts.
    """
    y_ctx = batch["raw_y_ctx"]
    y_q = batch["raw_target"]
    mean = y_ctx.mean(dim=0, keepdim=True)
    std = y_ctx.std(dim=0, keepdim=True).clamp_min(1e-6)
    z = (y_q - mean) / std
    nll = 0.5 * z**2 + torch.log(std) + 0.5 * float(np.log(2 * np.pi))
    return nll.reshape(-1)


def run_variant(
    model,
    *,
    splice: str,
    prior_cfg: PriorConfig,
    batch_cfg: BatchConfig,
    n_batches: int,
    seed_base: int,
    device: str,
) -> dict[str, float]:
    per_batch: list[float] = []
    per_batch_excess: list[float] = []
    skipped = 0
    for i in range(n_batches):
        rec = sample_batch(seed_base + i, prior_cfg)
        if rec["meta"]["nonfinite"]:
            skipped += 1
            continue
        sep = sample_single_eval_pos(seed_base + i, prior_cfg.seq_len, batch_cfg)
        batch = make_1d_batch(rec, sep, batch_cfg, splice=splice)
        losses = query_nll(model, batch, device)
        ref = gaussian_reference_nll(batch).to(losses.device)
        keep = torch.isfinite(losses) & torch.isfinite(ref)
        if keep.any():
            per_batch.append(float(losses[keep].mean()))
            per_batch_excess.append(float((losses[keep] - ref[keep]).mean()))

    arr = np.asarray(per_batch, dtype=np.float64)
    exc = np.asarray(per_batch_excess, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "median": float(np.median(arr)) if arr.size else float("nan"),
        "se": float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0,
        "excess": float(exc.mean()) if exc.size else float("nan"),
        "n": int(arr.size),
        "skipped": skipped,
        "_per_batch": arr,
        "_excess": exc,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-batches", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=2200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=777_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"loading released Do-PFN from {DOPFN_SRC}")
    model, config = load_released_model(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"  {type(model).__name__}  params={n_params:,}  "
        f"num_bars={model.criterion.num_bars}  device={args.device}"
    )
    borders = model.criterion.borders
    print(f"  borders: [{borders.min():.3f}, {borders.max():.3f}]\n")

    prior_cfg = PriorConfig(seq_len=args.seq_len, batch_size=args.batch_size)
    base_cfg = BatchConfig()
    common = dict(
        model=model,
        prior_cfg=prior_cfg,
        n_batches=args.n_batches,
        seed_base=args.seed_base,
        device=args.device,
    )

    # Every variant sees the identical SCM draws, so the comparison is paired.
    cases: list[tuple[str, str, BatchConfig]] = []
    for name in SPLICES:
        cases.append((f"splice={name}", name, base_cfg))
    # base_cfg now defaults to zscore_ctx (settled below); keep raw as the foil.
    cases.append(("y_space=raw", "spliced", replace(base_cfg, y_space="raw")))
    cases.append(
        (
            "eval_pos=fixed_query",
            "spliced",
            replace(base_cfg, eval_pos_strategy="fixed_query"),
        )
    )

    results: dict[str, dict] = {}
    print("NLL in raw-y nats (lower is better); 'excess' = model - context-Gaussian")
    print(f"{'variant':<26} {'mean NLL':>10} {'±se':>7} {'excess':>9} {'n':>4}")
    print("-" * 62)
    for label, splice, bcfg in cases:
        res = run_variant(splice=splice, batch_cfg=bcfg, **common)
        results[label] = res
        print(
            f"{label:<26} {res['mean']:>10.4f} {res['se']:>7.4f} "
            f"{res['excess']:>9.4f} {res['n']:>4d}"
        )

    ref_key = "splice=spliced"
    ref = results[ref_key]
    print("\nPaired deltas vs the inferred contract "
          f"({ref_key}); positive = worse than inferred")
    print("-" * 62)
    for label, res in results.items():
        if label == ref_key:
            continue
        n = min(len(ref["_per_batch"]), len(res["_per_batch"]))
        if n < 2:
            continue
        diff = res["_per_batch"][:n] - ref["_per_batch"][:n]
        se = diff.std(ddof=1) / np.sqrt(n)
        verdict = "worse" if diff.mean() > 2 * se else (
            "BETTER" if diff.mean() < -2 * se else "tie"
        )
        print(f"{label:<26} {diff.mean():>+10.4f} ±{se:<8.4f} {verdict}")

    print(
        "\nExpectation: every corruption should be clearly worse. If a corruption "
        "\nties or wins, the assumption it encodes is the one to adopt."
        "\n"
        "\nTwo readings this harness CANNOT support:"
        "\n  eval_pos=fixed_query  -- it uses a larger context (seq_len - 128) than"
        "\n     the paper sampler's average, and more context trivially lowers NLL."
        "\n     The difference measures context size, not faithfulness."
        "\n  splice=x_int_full     -- ties, because the two splices only differ on"
        "\n     covariates that are descendants of t. To give this real power,"
        "\n     restrict the prior to SCMs where t has covariate descendants."
    )

    # binary_strategy is an unrecorded free choice. It changes the data, not
    # just its presentation, so compare on excess-over-reference rather than on
    # raw NLL -- otherwise we would just be measuring which distribution is
    # intrinsically easier.
    print("\nbinary_strategy (unrecorded in the checkpoint) -- judge on 'excess':")
    print(f"{'variant':<26} {'mean NLL':>10} {'±se':>7} {'excess':>9} {'n':>4}")
    print("-" * 62)
    for strategy in ("extreme", "mean"):
        res = run_variant(
            splice="spliced",
            batch_cfg=base_cfg,
            **{**common, "prior_cfg": replace(prior_cfg, binary_strategy=strategy)},
        )
        print(
            f"{'binary_strategy=' + strategy:<26} {res['mean']:>10.4f} "
            f"{res['se']:>7.4f} {res['excess']:>9.4f} {res['n']:>4d}"
        )


if __name__ == "__main__":
    main()
