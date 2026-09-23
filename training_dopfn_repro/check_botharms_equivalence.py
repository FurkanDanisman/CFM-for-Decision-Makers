"""Assert that a doubled query block equals two separate forward passes.

``make_1d_botharms_batch`` puts both arms of every query unit into ONE query
block. That is the same computation as the Tier-C eval's two arm-flipped
forwards only if query rows cannot see each other and cannot move each other's
normalisation. Both hold by construction in this checkout:

  * ``PerFeatureEncoderLayer.attn_between_items`` and
    ``TransformerEncoderLayer`` key and value the query rows against
    ``src_[:single_eval_pos]`` only -- query rows never attend to query rows;
  * every encoder step normalises at ``single_eval_pos`` because the released
    config sets ``normalize_on_train_only=True``;
  * the row axis carries no positional embedding.

That is a read of the source, and the entire variant rests on it, so check it
against the real backbone before spending an allocation on 150k steps.

    python training_dopfn_repro/check_botharms_equivalence.py

Needs the same environment ``train.py`` does (torch 2.1 / python 3.10, with
``DOPFN_SRC`` pointing at a Do-PFN checkout that has ``artifacts/``).

A non-zero max|diff| at the 1e-6 level is float noise from differently shaped
matmuls, not leakage; leakage shows up orders of magnitude larger, because arm
0 and arm 1 rows carry different column-0 values. Raise ``--tol`` only if you
have looked at the number and understood it.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from training_dopfn_repro.batch import (  # noqa: E402
    BatchConfig,
    make_1d_botharms_batch,
    sample_single_eval_pos,
)
from training_dopfn_repro.model import (  # noqa: E402
    ModelSpec,
    build_backbone_init,
    build_model,
)
from training_dopfn_repro.prior import PriorConfig, sample_batch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # Short sequences on purpose: this is a structural property, not a
    # capacity one, and it has to run on a login node.
    ap.add_argument("--seq-len", type=int, default=240)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=700_000)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    prior_cfg = PriorConfig(seq_len=args.seq_len, batch_size=args.batch_size)
    batch_cfg = BatchConfig()
    spec = ModelSpec(variant="dopfn_1d_botharms")

    with tempfile.TemporaryDirectory() as tmp:
        init_path = os.path.join(tmp, "backbone_init.pt")
        build_backbone_init(0, init_path)
        # eval(): no dropout, so the two calls are comparable at all.
        model = build_model(spec, init_path).eval()

        worst, checked = 0.0, 0
        for i in range(args.batches):
            seed = args.seed_base + i
            rec = sample_batch(seed, prior_cfg)
            if rec["meta"]["nonfinite"]:
                print(f"  seed {seed}: non-finite prior draw, skipped")
                continue
            sep = sample_single_eval_pos(seed, prior_cfg.seq_len, batch_cfg)
            b = make_1d_botharms_batch(rec, sep, batch_cfg)
            n = int(b["n_query"])

            with torch.no_grad():
                # Exactly how train.py::compute_loss calls the model.
                both = model(
                    b["train_x"], b["train_y"], b["test_x"],
                    only_return_standard_out=True,
                )
                # Exactly how the Tier-C eval queries a 1-D head: one forward
                # per arm, same context both times.
                apart = torch.cat(
                    [
                        model(
                            b["train_x"], b["train_y"],
                            b["test_x"][k * n : (k + 1) * n],
                            only_return_standard_out=True,
                        )
                        for k in range(2)
                    ],
                    dim=0,
                )

            d = float((both - apart).abs().max())
            worst = max(worst, d)
            checked += 1
            print(
                f"  seed {seed}: sep={sep:>5} n_query={n:>5} "
                f"logits={tuple(both.shape)} max|diff|={d:.3e}"
            )

    if not checked:
        raise SystemExit("every prior draw was non-finite; nothing was checked")

    print(f"\nworst max|diff| over {checked} batches: {worst:.3e} (tol {args.tol:g})")
    if worst > args.tol:
        raise SystemExit(
            "FAIL: the doubled query block is NOT two independent forwards. "
            "Something in this checkout lets query rows influence each other, "
            "so dopfn_1d_botharms would train on a task the eval cannot "
            "reproduce. Do not train it until this is understood."
        )
    print(
        "OK: query rows are independent, so both-arms training matches the "
        "eval's two arm-flipped forwards."
    )


if __name__ == "__main__":
    main()
