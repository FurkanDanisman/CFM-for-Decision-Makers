"""Hydra factory to instantiate CausalPFN2DHead with a TabDPT-warmstarted
backbone, so it drops into CausalPFN's own train.py via:

    model.obj._target_=rpfn_patches.cpfn2d_loader.load_cpfn2d_from_tabdpt
    +model.obj.J=32
    +model.obj.hlgauss_sigma=0.32
    +model.obj.loss_type=hlgauss
    +model.obj.y_scaling_mode=pooled_std
    +model.obj.num_features=100

The backbone is loaded from the same TabDPT checkpoint CausalPFN uses
(vdblm/causalpfn @ tabdpt_long_context.ckpt). We copy over every backbone
tensor whose shape matches — everything except the final head Linear,
which has a different output dim for the 2D-joint head (J**2 + 9 + 4
instead of the TabDPT native nbins). The head is randomly initialised.

Usage from a sbatch:

    python train.py +experiment=simple_configuration \\
        model.obj._target_=rpfn_patches.cpfn2d_loader.load_cpfn2d_from_tabdpt \\
        +model.obj.J=32 \\
        +model.obj.hlgauss_sigma=0.32 \\
        +model.obj.loss_type=hlgauss \\
        +model.obj.y_scaling_mode=pooled_std \\
        +model.obj.num_features=100 \\
        model.obj.ckpt_path=/path/to/tabdpt_long_context.ckpt \\
        ~callbacks.eval_cate

The rest — trainer, DataLoader, optimizer, DDP, checkpointing — is
CausalPFN's stock stack. Only the model is ours.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn


def load_cpfn2d_from_tabdpt(
    ckpt_path: Optional[str] = None,
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    revision: Optional[str] = None,
    *,
    J: int = 32,
    # X-feature count; backbone will be built at (num_features + 1) so TabDPT
    # ckpt's native num_features=100 matches when this is 99. That matches
    # CausalPFN's synthetic_backdoor prior (post_padding_n_cols=99).
    num_features: int = 99,
    ninp: int = 384,
    nhid: int = 768,
    nhead: int = 6,
    nlayers: int = 20,
    dropout: float = 0.0,
    n_out: int = 10,
    y_scaling_mode: str = 'pooled_std',
    loss_type: str = 'hlgauss',
    hlgauss_sigma: float = 0.32,
    # Optional inner-region edge overrides. If left None, uses the mode's
    # default ([-10, +10] for pooled_std, [-1, +1] for uwyk_minmax). Passing
    # e.g. edge_lo=-3, edge_hi=3 tightens the inner region under pooled_std
    # so the 9-region tail head sees the ~0.8% of training samples with
    # |y_std| > 3 and actually learns those params.
    edge_lo: Optional[float] = None,
    edge_hi: Optional[float] = None,
    # `sigma` accepted for hydra-side interface parity with
    # load_pretrained_in_context_model (which is called with `sigma=...`
    # from the same experiment config). Ignored — we use hlgauss_sigma.
    sigma: Optional[float] = None,
):
    """Instantiate CausalPFN2DHead and warmstart its backbone from a TabDPT ckpt.

    Args mirror `load_pretrained_in_context_model` for the source/loading
    of the pretrained ckpt, plus 2D-model constructor kwargs. Returns a
    module that exposes:
      - forward(X_context, t_context, y_context, X_query, E_y0_query, E_y1_query)
        → per-task loss (B,)      [same signature as InContextModel]
      - model_config (dict, saved by Checkpoint callback)
      - get_param_groups()        [used by schedulefree AdamW]
      - state_dict()              [standard nn.Module]
    """
    # Wire sys.path so `training_causalpfn2d` and `losses` import from our repo.
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.abspath(os.path.join(_here, '..'))
    if _repo not in sys.path:
        sys.path.insert(0, _repo)

    from training_causalpfn2d.model_causalpfn_2d import CausalPFN2DHead

    # Build our 2D model at random init.
    model = CausalPFN2DHead(
        J=J,
        num_features=num_features,
        ninp=ninp,
        nhid=nhid,
        nhead=nhead,
        nlayers=nlayers,
        dropout=dropout,
        n_out=n_out,
        y_scaling_mode=y_scaling_mode,
        loss_type=loss_type,
        hlgauss_sigma=hlgauss_sigma,
        edge_lo=edge_lo,
        edge_hi=edge_hi,
    )

    # Load the TabDPT backbone from the same ckpt CausalPFN uses.
    from causalpfn.models import load_pretrained_in_context_model
    tabdpt_model = load_pretrained_in_context_model(
        ckpt_path=ckpt_path,
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        sigma=sigma if sigma is not None else 0.5,
    )
    # InContextModel wraps a TabDPTLongContextModel at .model
    src_backbone = tabdpt_model.model

    # Copy every backbone tensor whose shape matches ours. Skip mismatches
    # (the final head Linear differs because our nbins is J^2+9+4 vs
    # TabDPT's native 1024). Report what got copied vs skipped.
    src_sd = src_backbone.state_dict()
    dst_sd = model.backbone.state_dict()
    copied, skipped = [], []
    for k, v_src in src_sd.items():
        if k in dst_sd and dst_sd[k].shape == v_src.shape:
            dst_sd[k] = v_src
            copied.append(k)
        else:
            reason = 'shape mismatch' if k in dst_sd else 'not in dst'
            skipped.append((k, reason))
    missing_in_src = [k for k in dst_sd if k not in src_sd]
    model.backbone.load_state_dict(dst_sd, strict=False)

    print(f'[cpfn2d-warmstart] copied {len(copied)} / {len(src_sd)} backbone tensors')
    print(f'[cpfn2d-warmstart] skipped {len(skipped)} (head reshape expected):')
    for k, why in skipped[:5]:
        print(f'    - {k}  [{why}]')
    if len(skipped) > 5:
        print(f'    ... ({len(skipped) - 5} more)')
    if missing_in_src:
        print(f'[cpfn2d-warmstart] {len(missing_in_src)} dst tensors NOT in src (random-init):')
        for k in missing_in_src[:5]:
            print(f'    - {k}')

    return model
