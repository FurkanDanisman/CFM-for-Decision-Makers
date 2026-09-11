"""Initialise the 2D joint head from a trained 1D BarDistribution head.

WHAT THIS DOES. The 2D head emits J**2 + 9 + 4 logits per query, laid out as

    [ 0 : J*J   ]  joint bin logits, softmaxed over the J x J grid
    [ J*J : +9  ]  region mixture weights (index 0 = inner x inner)
    [ +9  : +4  ]  4 half-Gaussian tail scale raws

Given a 1D head of the SAME J whose logits are l_i, we set

    joint_logit(i, j) = l_i + l_j

so after the softmax

    p(i, j) = exp(l_i + l_j) / sum_kl exp(l_k + l_l) = p_1d(i) * p_1d(j)

i.e. the joint starts as the INDEPENDENT PRODUCT of two copies of the 1D
marginal. Both marginals are exactly the 1D model's distribution, and the
correlation is exactly 0. Training then only has to learn the coupling, which
is the part a 1D model cannot represent -- instead of re-learning the marginals
from a random head.

Concretely, for a Linear(ninp, n_out + J*J + 13) head, row (i*J + j) of the
joint block is W1d[i] + W1d[j] and its bias is b1d[i] + b1d[j]. This holds for
every input x, so the factorisation is exact at init, not approximate.

REGIONS. Region 0 is inner x inner; regions 1-8 are the tails. We want region 0
to carry ~all the mass at init, so its logit is pinned high and the other eight
low, with ZERO weight rows -- the region mixture is then input-independent at
init and softmax gives region 0 about 1 - 8*exp(-2*REGION_LOGIT). The 4 tail
scale raws are zeroed (softplus(0) = 0.693, so s = 0.693 * bin_width: finite,
harmless, and unused while the tails carry no mass). All of this is only an
INITIALISATION -- gradients are free to move the regions later.

Usage (hydra):
    model.obj._target_=rpfn_patches.cpfn2d_head_from_1d.load_cpfn2d_from_1d \\
    model.obj.ckpt_path=<tabdpt warmstart ckpt> \\
    +model.obj.ckpt_1d=<trained 1D J=1024 checkpoint .pt> \\
    +model.obj.J=1024
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

REGION_LOGIT = 10.0          # inner-region bias; 1 - 8*exp(-20) ~ 1 - 1.6e-8


def _find_head_linear(obj):
    """Locate the final head Linear in a TabDPT-style backbone or state dict."""
    if isinstance(obj, dict):
        # raw state_dict: find the '...head.2.weight' style key
        cands = [k for k in obj if k.endswith('head.2.weight')]
        if not cands:
            cands = [k for k in obj if k.endswith('.weight') and 'head' in k]
        if not cands:
            raise KeyError('no head Linear weight found in state dict; keys like '
                           f'{[k for k in list(obj)[:8]]}')
        wk = sorted(cands, key=len)[-1]
        bk = wk[:-len('weight')] + 'bias'
        return obj[wk], obj.get(bk)
    backbone = getattr(obj, 'model', obj)
    head = getattr(backbone, 'head', None)
    if head is None:
        raise AttributeError('backbone has no .head')
    lin = head[2] if not isinstance(head, nn.Linear) else head
    return lin.weight.data, (None if lin.bias is None else lin.bias.data)


def _unwrap_state(ckpt):
    for k in ('model_state_dict', 'state_dict', 'model'):
        if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    return ckpt


def build_joint_head_from_1d(W1d, b1d, J, n_out, ninp, device, dtype,
                             region_logit=REGION_LOGIT):
    """Return (weight, bias) for Linear(ninp, n_out + J*J + 13).

    W1d: (n_out + J, ninp) or (J, ninp) -- the 1D head. Only its last J rows
    (the bin logits) are used.
    """
    W1d = W1d.detach().to(device=device, dtype=torch.float64)
    if W1d.shape[0] < J:
        raise ValueError(f'1D head has {W1d.shape[0]} rows, need at least J={J}')
    if W1d.shape[1] != ninp:
        raise ValueError(f'1D head ninp={W1d.shape[1]} != 2D head ninp={ninp}; '
                         'the two models must share a backbone width for the '
                         'row-sum construction to be meaningful')
    Wb = W1d[-J:]                                   # (J, ninp) bin-logit rows
    if b1d is None:
        bb = torch.zeros(J, dtype=torch.float64, device=device)
    else:
        bb = b1d.detach().to(device=device, dtype=torch.float64)[-J:]

    nbins_2d = J * J + 9 + 4
    out = n_out + nbins_2d
    W = torch.zeros(out, ninp, dtype=torch.float64, device=device)
    b = torch.zeros(out, dtype=torch.float64, device=device)

    # joint block: row (i*J + j) = W1d[i] + W1d[j]   -> p(i,j) = p_i * p_j
    base = n_out
    W[base:base + J * J] = (Wb[:, None, :] + Wb[None, :, :]).reshape(J * J, ninp)
    b[base:base + J * J] = (bb[:, None] + bb[None, :]).reshape(J * J)

    # region mixture: all mass to region 0, input-independent (zero weights)
    r0 = base + J * J
    b[r0] = region_logit
    b[r0 + 1:r0 + 9] = -region_logit
    # 4 tail scale raws: zero -> softplus(0)=0.693 -> s = 0.693 * bin_width
    return W.to(dtype), b.to(dtype)


def load_cpfn2d_from_1d(
    ckpt_path: Optional[str] = None,
    ckpt_1d: Optional[str] = None,
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    revision: Optional[str] = None,
    J: int = 1024,
    region_logit: float = REGION_LOGIT,
    **kwargs,
):
    """Backbone warmstart exactly as load_cpfn2d_from_tabdpt, then overwrite the
    head with the 1D-derived product initialisation."""
    from rpfn_patches.cpfn2d_loader import load_cpfn2d_from_tabdpt
    model = load_cpfn2d_from_tabdpt(
        ckpt_path=ckpt_path, repo_id=repo_id, filename=filename,
        revision=revision, J=J, **kwargs)

    if not ckpt_1d:
        raise ValueError('ckpt_1d is required: path to the trained 1D J=%d head' % J)
    if not os.path.isfile(ckpt_1d):
        raise FileNotFoundError(ckpt_1d)

    src = _unwrap_state(torch.load(ckpt_1d, map_location='cpu', weights_only=False))
    W1d, b1d = _find_head_linear(src)

    lin = model.backbone.head[2]
    n_out = model.n_out
    ninp = lin.in_features
    expect = n_out + J * J + 9 + 4
    if lin.out_features != expect:
        raise ValueError(f'2D head out_features={lin.out_features}, expected {expect}')

    W, b = build_joint_head_from_1d(W1d, b1d, J, n_out, ninp,
                                    lin.weight.device, lin.weight.dtype,
                                    region_logit=region_logit)
    with torch.no_grad():
        lin.weight.copy_(W)
        if lin.bias is not None:
            lin.bias.copy_(b)

    print(f'[cpfn2d-from1d] head initialised from {os.path.basename(ckpt_1d)}')
    print(f'[cpfn2d-from1d]   joint block {J}x{J} = W1d[i] + W1d[j]  '
          f'(p(i,j) = p_1d(i) * p_1d(j) at init, rho = 0)')
    print(f'[cpfn2d-from1d]   region 0 logit {region_logit:+.1f}, regions 1-8 '
          f'{-region_logit:+.1f} -> inner mass '
          f'{1 - 8 * torch.exp(torch.tensor(-2 * region_logit)).item():.9f}')
    return model


# ── self-test (no checkpoint needed) ─────────────────────────────────────────
def _self_test(J=16, ninp=8, n_out=10, seed=0):
    torch.manual_seed(seed)
    W1d = torch.randn(n_out + J, ninp, dtype=torch.float64)
    b1d = torch.randn(n_out + J, dtype=torch.float64)
    W, b = build_joint_head_from_1d(W1d, b1d, J, n_out, ninp,
                                    torch.device('cpu'), torch.float64)
    x = torch.randn(5, ninp, dtype=torch.float64)
    out = x @ W.T + b

    l1 = x @ W1d[-J:].T + b1d[-J:]                       # (5, J) 1D logits
    p1 = torch.softmax(l1, dim=-1)
    pj = torch.softmax(out[:, n_out:n_out + J * J], dim=-1).reshape(-1, J, J)

    err = (pj - p1[:, :, None] * p1[:, None, :]).abs().max().item()
    m0 = (pj.sum(dim=2) - p1).abs().max().item()
    m1 = (pj.sum(dim=1) - p1).abs().max().item()
    # correlation of the joint must be 0
    c = torch.arange(J, dtype=torch.float64)
    E0 = (pj.sum(2) * c).sum(1); E1 = (pj.sum(1) * c).sum(1)
    E01 = (pj * c[None, :, None] * c[None, None, :]).sum((1, 2))
    cov = (E01 - E0 * E1).abs().max().item()

    reg = torch.softmax(out[:, n_out + J * J:n_out + J * J + 9], dim=-1)
    print(f'  joint == product of marginals : max|err| = {err:.3e}')
    print(f'  marginal_0 == p_1d            : max|err| = {m0:.3e}')
    print(f'  marginal_1 == p_1d            : max|err| = {m1:.3e}')
    print(f'  covariance                    : max|cov| = {cov:.3e}')
    print(f'  region-0 mass                 : {reg[:, 0].min().item():.9f}')
    assert err < 1e-12 and m0 < 1e-12 and m1 < 1e-12 and cov < 1e-9
    assert reg[:, 0].min().item() > 1 - 1e-7
    print('  SELF-TEST PASS')


if __name__ == '__main__':
    for J in (8, 16, 64):
        print(f'J={J}:'); _self_test(J=J)
