"""Warmstart loader that supports a nbins override.

CausalPFN's `simple_configuration` uses `load_pretrained_in_context_model`
which loads the entire InContextModel from a HuggingFace checkpoint at
whatever nbins the checkpoint was trained with (=1024 for
tabdpt_long_context.ckpt). There's no hydra-native way to override nbins
because the model architecture is determined at load-time from the ckpt.

This patched loader:
  1. Calls the standard loader to get the model at the ckpt's native nbins
  2. If `nbins_override` is set, surgically replaces the final head layer
     (Sequential index 2 → the K-bin classifier) with a randomly-initialised
     Linear at the new nbins
  3. Re-initialises InContextModel's bin_edges/width/centers buffers to
     reflect the new nbins on the same [vmin, vmax] range

Backbone weights (all 148 of the transformer_encoder + norms + input
encoder) load cleanly. Only the head layer at ~10k params is random-init
at the new nbins. Model retrains from there.

Use via hydra override:
    model.obj._target_=rpfn_patches.loader_with_nbins.load_pretrained_in_context_model_with_nbins
    +model.obj.nbins_override=100
"""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


def load_pretrained_in_context_model_with_nbins(
    ckpt_path: Optional[str] = None,
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    revision: Optional[str] = None,
    sigma: float = 0.5,
    nbins_override: Optional[int] = None,
    head_reinit: bool = False,
):
    """Same signature as causalpfn.models.load_pretrained_in_context_model,
    plus optional `nbins_override` and `head_reinit`.

    - nbins_override: replaces the final head with a new-nbins Linear (see below).
    - head_reinit: re-initialises the final head IN PLACE at the ckpt's native
      nbins (backbone stays warmstarted; head becomes random). Used to isolate
      the effect of head warmstart while keeping architecture identical to the
      ckpt. Ignored if nbins_override is set (that already re-inits the head).
    """
    from causalpfn.models import load_pretrained_in_context_model

    # Base call — always warmstart from the ckpt at its native nbins.
    model = load_pretrained_in_context_model(
        ckpt_path=ckpt_path,
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        sigma=sigma,
    )

    if nbins_override is None:
        if head_reinit:
            backbone = model.model
            old_final = backbone.head[2]
            has_bias = old_final.bias is not None
            device = old_final.weight.device
            dtype = old_final.weight.dtype
            new_final = nn.Linear(old_final.in_features, old_final.out_features, bias=has_bias)
            new_final.to(device=device, dtype=dtype)
            nn.init.kaiming_uniform_(new_final.weight, a=(5 ** 0.5))
            if has_bias:
                nn.init.zeros_(new_final.bias)
            backbone.head[2] = new_final
            print(f'[head-reinit] head[2] re-initialised at native shape '
                  f'({old_final.in_features}, {old_final.out_features}); '
                  f'backbone kept warmstarted.')
        return model

    # Surgery: replace the final classification head Linear at new nbins.
    backbone = model.model                     # TabDPTLongContextModel
    old_final = backbone.head[2]               # final Linear
    ninp = old_final.in_features               # 768 for TabDPT
    old_out = old_final.out_features           # n_out + old_nbins
    orig_nbins = backbone.nbins if hasattr(backbone, 'nbins') else (old_out - 10)
    n_out = old_out - orig_nbins               # extra class heads (10 for TabDPT)
    new_out = n_out + nbins_override

    print(f'[nbins-override] replacing head[2]: '
          f'Linear({ninp}, {old_out}) → Linear({ninp}, {new_out}); '
          f'nbins: {orig_nbins} → {nbins_override}')

    # Match the original head's bias config and dtype/device.
    has_bias = old_final.bias is not None
    device = old_final.weight.device
    dtype = old_final.weight.dtype
    new_final = nn.Linear(ninp, new_out, bias=has_bias)
    new_final.to(device=device, dtype=dtype)
    # Small init — same std convention as PyTorch default Linear.
    nn.init.kaiming_uniform_(new_final.weight, a=(5 ** 0.5))
    if has_bias:
        nn.init.zeros_(new_final.bias)

    backbone.head[2] = new_final
    if hasattr(backbone, 'nbins'):
        backbone.nbins = nbins_override

    # Update InContextModel-level nbins + bin_edges/width/centers buffers.
    # InContextModel.__init__ does:
    #   self.nbins = model_config["model"]["nbins"]
    #   bin_edges = torch.linspace(vmin, vmax, nbins + 1)
    #   bin_width = bin_edges[1] - bin_edges[0]
    #   bin_centers = bin_edges[:-1] + 0.5 * bin_width
    #   self.register_buffer("bin_edges", bin_edges)
    #   self.register_buffer("bin_width", bin_width)
    #   self.register_buffer("bin_centers", bin_centers)
    model.nbins = nbins_override
    if hasattr(model, 'model_config') and isinstance(model.model_config, dict):
        model.model_config.setdefault('model', {})
        model.model_config['model']['nbins'] = nbins_override

    vmin = float(model.vmin) if hasattr(model, 'vmin') else -10.0
    vmax = float(model.vmax) if hasattr(model, 'vmax') else +10.0
    edges = torch.linspace(vmin, vmax, nbins_override + 1, dtype=dtype, device=device)
    width = edges[1] - edges[0]
    centers = edges[:-1] + 0.5 * width

    # Buffers need to be REPLACED (register_buffer doesn't accept size change).
    # Delete existing then register new.
    for name in ('bin_edges', 'bin_width', 'bin_centers'):
        if name in dict(model.named_buffers()):
            delattr(model, name)
    model.register_buffer('bin_edges', edges)
    model.register_buffer('bin_width', width)
    model.register_buffer('bin_centers', centers)

    print(f'[nbins-override] new bin_edges range: [{edges[0].item():.2f}, {edges[-1].item():.2f}]  '
          f'bin_width={width.item():.4f}  n_bins={nbins_override}')
    return model
