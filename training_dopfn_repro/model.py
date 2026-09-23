"""Backbone initialisation and the two heads.

Both variants share one backbone, one prior, one optimiser and one initialisation
so that the head is the only moving part.

Initialisation
--------------
"Same seed" is not enough to give two variants the same backbone: RNG draws are
consumed in construction order, and the heads have different output widths, so a
head built before or between backbone modules shifts every subsequent draw.
``build_backbone_init`` therefore derives the init **once**, hashes it, and every
variant loads that exact tensor set.

Re-initialising the backbone correctly is also less obvious than it looks. The
architecture can only be obtained by unpickling ``dopfn_model.pkl`` (the repo has
no from-config builder -- ``model_builder.load_model`` only unpickles and loads a
state dict), and a naive

    for m in model.modules():
        if hasattr(m, 'reset_parameters'): m.reset_parameters()

silently misses **24 of 112 param-bearing modules**: every ``MultiheadAttention``
spells it ``_reset_parameters`` with a leading underscore. Those 24 attention
``in_proj_weight`` tensors then keep whatever the pickle shipped, identical on
every run and unaffected by the seed. Measured, not hypothetical.

(``dopfn_model.pkl`` ships *untrained* weights -- only the three criterion
buffers match the trained checkpoint -- so this is a silent loss of
randomisation, not a leak of trained weights.)

``reinit_backbone_`` handles both spellings and raises on any param-bearing
module it cannot initialise, rather than passing silently.

Heads
-----
``dopfn_1d``   100-bucket FullSupportBarDistribution. Do-PFN's own head, and
               task 1: the faithful reproduction.
``joint_2d``   J x J grid over both arms (J^2 + 9 region weights + 4 tail
               scales), J=10 to match the 1-D head's output width. Task 2: the
               joint version of task 1. This is the head that buys
               per-individual effects.

Both report loss in **nats per outcome** so one learning rate transfers
unchanged -- the 2-D joint is halved, since a density over two outcomes is
otherwise about twice the magnitude of a 1-D one.

Resolution is allocated differently by necessity, not by choice. Do-PFN's 1-D
borders are quantile-allocated (released widths run 0.049 to 245.9), whereas
``neg_log_prob_2d`` carries a single scalar ``bin_width`` and so *requires* a
uniform grid. The 2-D edges therefore span a robust quantile range of the same
target space, leaving the 9-region tail structure to absorb what falls outside.
Documented in the README; the oracle floor quantifies what it costs.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from training_dopfn_repro.prior import DOPFN_SRC  # noqa: E402

VARIANTS = ("dopfn_1d", "dopfn_1d_botharms", "joint_2d")

#: Variants driving Do-PFN's own 1-D bar head. They share the head, the loss and
#: the fitted borders exactly; they differ only in which arms of each query unit
#: batch.py emits, so anything keyed on "is this a 1-D model" must test
#: membership here rather than equality with "dopfn_1d".
VARIANTS_1D = ("dopfn_1d", "dopfn_1d_botharms")

# BarDistribution2D emits J^2 inner logits + 9 region weights + 4 tail scales.
N_REGIONS_2D = 9
N_TAIL_PARAMS_2D = 4


@contextlib.contextmanager
def _in_dopfn_root():
    prev = os.getcwd()
    if DOPFN_SRC not in sys.path:
        sys.path.insert(0, DOPFN_SRC)
    os.chdir(DOPFN_SRC)
    try:
        yield
    finally:
        os.chdir(prev)


def load_architecture():
    """Unpickle the PerFeatureTransformer. Weights are whatever the pickle holds."""
    with _in_dopfn_root():
        with open("artifacts/dopfn_model.pkl", "rb") as fh:
            return pickle.load(fh)


def load_released_config() -> dict[str, Any]:
    with _in_dopfn_root():
        from scripts.transformer_prediction_interface.model_builder import Checkpoint

        return Checkpoint.load(
            "artifacts/model_submitit_0ccc_id_171b69db_epoch_-1.cpkt"
        ).config


# ---------------------------------------------------------------------------
# initialisation
# ---------------------------------------------------------------------------


def reinit_backbone_(model: nn.Module, seed: int, zero_init: bool = True) -> dict:
    """Re-initialise every parameter in place, deterministically and completely.

    Raises if any parameter-bearing module exposes neither ``reset_parameters``
    nor ``_reset_parameters`` -- a silent skip there is the bug this exists to
    prevent.
    """
    torch.manual_seed(seed)

    done, uncovered = [], []
    for name, mod in model.named_modules():
        own_params = list(mod.parameters(recurse=False))
        if not own_params:
            continue
        fn = getattr(mod, "reset_parameters", None) or getattr(
            mod, "_reset_parameters", None
        )
        if callable(fn):
            fn()
            done.append(name)
        else:
            uncovered.append(f"{name} ({type(mod).__name__})")

    if uncovered:
        raise RuntimeError(
            "cannot initialise these parameter-bearing modules: "
            + ", ".join(uncovered[:10])
            + (f" (+{len(uncovered) - 10} more)" if len(uncovered) > 10 else "")
        )

    # Do-PFN's own zero-init: neutral residual streams at step 0.
    if zero_init and hasattr(model, "init_weights"):
        model.init_weights(zero_init=True)

    return {"initialised_modules": len(done), "seed": seed}


def state_dict_hash(sd: dict[str, torch.Tensor]) -> str:
    """Order-independent content hash, for asserting two runs share an init."""
    h = hashlib.sha256()
    for key in sorted(sd):
        t = sd[key]
        h.update(key.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().to(torch.float64).cpu().numpy().tobytes())
    return h.hexdigest()


def build_backbone_init(seed: int, out_path: str) -> str:
    """Derive the canonical backbone init once and save it. Returns its hash."""
    model = load_architecture()
    report = reinit_backbone_(model, seed)
    sd = model.state_dict()
    digest = state_dict_hash(sd)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    torch.save({"state_dict": sd, "seed": seed, "sha256": digest, **report}, out_path)
    return digest


# ---------------------------------------------------------------------------
# heads
# ---------------------------------------------------------------------------


def is_1d(variant: str) -> bool:
    """True for the variants wearing the 1-D bar head."""
    return variant in VARIANTS_1D


def head_output_dim(variant: str, num_buckets: int = 100, j_2d: int = 10) -> int:
    if is_1d(variant):
        return num_buckets
    if variant == "joint_2d":
        return j_2d * j_2d + N_REGIONS_2D + N_TAIL_PARAMS_2D
    raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")


#: Initial value for the 2-D head's four tail-scale logits. BarDistribution2D
#: computes ``scale = bin_width * (softplus(raw) + 1e-3)``, so raw=0 gives
#: ``0.69 * bin_width`` -- around 0.45 in standardised units. The prior's
#: heaviest draws reach |y| ~ 50, which then sits ~108 sigma outside the grid
#: and costs thousands of nats on a single query point. Measured: batch loss
#: correlates with max|y| at +0.96, and spikes to 63 nats/outcome against a
#: median of 3.0. Starting the tails wide removes that from initialisation;
#: the scales are learned thereafter.
#:
#: Swept over a fixed set of 40 batches with an otherwise-uniform head, showing
#: worst-case batch loss in nats per outcome against an unchanged median of
#: 2.95 -- so the wide init costs nothing in the bulk:
#:
#:     raw = 0 -> max 19.6     raw = 4 -> max 3.37
#:     raw = 2 -> max  4.58    raw = 6 -> max 3.14
#:
#: Only the tail *shape* is affected; mass allocation is the 9-way region
#: softmax, which this does not touch.
TAIL_SCALE_INIT_LOGIT = 6.0


def make_decoder(
    d_model: int,
    n_out: int,
    nhid_factor: int = 4,
    tail_slice: slice | None = None,
    tail_init: float = TAIL_SCALE_INIT_LOGIT,
) -> nn.Module:
    """Same shape as Do-PFN's decoder_dict['standard']: Linear -> GELU -> Linear.

    ``tail_slice`` biases those output units at initialisation, used by the 2-D
    head to start its Gaussian tails wide enough for a heavy-tailed prior.
    """
    nhid = nhid_factor * d_model
    final = nn.Linear(nhid, n_out)
    if tail_slice is not None:
        with torch.no_grad():
            final.bias[tail_slice] = tail_init
    return nn.Sequential(nn.Linear(d_model, nhid), nn.GELU(), final)


@dataclass
class ModelSpec:
    variant: str
    num_buckets: int = 100
    j_2d: int = 10
    head_seed: int = 0

    @property
    def n_out(self) -> int:
        return head_output_dim(self.variant, self.num_buckets, self.j_2d)


def build_model(spec: ModelSpec, backbone_init_path: str) -> nn.Module:
    """Load the canonical backbone init, then install this variant's head.

    The head is seeded separately so that swapping heads cannot perturb the
    backbone -- the whole point of deriving the init from a file rather than
    from a shared RNG stream.
    """
    blob = torch.load(backbone_init_path, map_location="cpu")
    model = load_architecture()
    model.load_state_dict(blob["state_dict"], strict=True)

    got = state_dict_hash(model.state_dict())
    if got != blob["sha256"]:
        raise RuntimeError(
            f"backbone init hash mismatch: expected {blob['sha256'][:16]}, "
            f"got {got[:16]}"
        )

    d_model = int(getattr(model, "ninp", 192))
    torch.manual_seed(spec.head_seed)
    tail_slice = None
    if spec.variant == "joint_2d":
        jj = spec.j_2d * spec.j_2d
        tail_slice = slice(jj + N_REGIONS_2D, jj + N_REGIONS_2D + N_TAIL_PARAMS_2D)
    model.decoder_dict["standard"] = make_decoder(
        d_model, spec.n_out, tail_slice=tail_slice
    )

    model.dopfn_repro_spec = {
        "variant": spec.variant,
        "n_out": spec.n_out,
        "backbone_sha256": blob["sha256"],
        "backbone_seed": blob["seed"],
        "head_seed": spec.head_seed,
    }
    return model


# ---------------------------------------------------------------------------
# losses -- all in nats per outcome
# ---------------------------------------------------------------------------


def loss_1d(criterion, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Do-PFN's own objective: mean NLL over query rows."""
    return criterion(logits, target).mean()




def loss_joint_2d(
    logits: torch.Tensor,
    y0: torch.Tensor,
    y1: torch.Tensor,
    j_2d: int,
    edges: torch.Tensor,
) -> torch.Tensor:
    """Joint density over both arms, halved to put it in nats per outcome."""
    from losses.BarDistribution2D import neg_log_prob_2d

    # BarDistribution2D is batch-first: (B, M, ...) against the model's (M, B, ...)
    pred = logits.transpose(0, 1).contiguous()
    t0 = y0.transpose(0, 1).contiguous()
    t1 = y1.transpose(0, 1).contiguous()
    return 0.5 * neg_log_prob_2d(pred, t0, t1, j_2d, edges, reduce="mean")
