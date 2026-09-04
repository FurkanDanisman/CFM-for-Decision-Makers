"""Bivariate-noise extension of DoPFN's MakeStructuralEquations.

Background
==========
DoPFN's `MakeStructuralEquations` (upstream file
`priors/playground_scm/MakeStructuralEquations.py`, line ~195) samples
additive noise ONCE at __init__ and reuses the same tensor on every
`.forward()` call. When `doscm.get_batch()` samples an observational
batch and then an interventional batch reusing `exogenous_vars=exo_obs`,
the Y-node's noise is IDENTICAL on both draws, so
    tau_i = Y_int_i - Y_obs_i  =  f_Y(pa_int) - f_Y(pa_obs)
is a deterministic function of the intervention (no residual variance)
and the CATE density `p(tau | x)` collapses to a delta.

For density-metric evaluation we want a proper Gaussian CATE density.
This module implements two independently useful primitives:

1. `MakeStructuralEquationsBivariate` — a drop-in subclass that stores
   TWO noise tensors, `additive_noise_obs` and `additive_noise_int`,
   drawn as a bivariate Gaussian with correlation `y_noise_correlation`.
   Its `forward(is_int_call=False, **kwargs)` picks the appropriate
   tensor. All non-Y nodes should keep using the original MakeStructural-
   Equations (so their noise stays comonotone — required for the
   `exogenous_vars=exo_obs` reuse in scm.get_next_sample to actually
   reproduce ancestor values on the interventional pass).

2. `install_bivariate_y_noise(scm, y_noise_correlation, noise_std)` —
   post-hoc patch. Given an already-instantiated StructuralCausalModel
   with `scm.y_key` set, resamples the Y-node's noise as a bivariate
   Gaussian and stores both draws on the SAME existing
   MakeStructuralEquations instance (attributes `additive_noise_obs`,
   `additive_noise_int`, `y_noise_correlation`, `_bivariate_noise_installed`).
   This preserves the fitted `nn.Linear` + activation on the Y-node.
   Companion helper `set_y_call_mode(scm, mode)` swaps the active
   noise tensor between obs and int between the two `get_next_sample`
   invocations.

Both paths give
    (eps_obs, eps_int) ~ N(0, sigma^2 * [[1, rho], [rho, 1]])
so
    p(tau | x) = N( mu_1(x) - mu_0(x),  2 * sigma^2 * (1 - rho) ).
"""
from __future__ import annotations

import math
from typing import Optional

import torch

# Try to import DoPFN's upstream MakeStructuralEquations. The regen script
# also has to have DoPFN on sys.path; this file just imports what it needs.
try:
    from priors.playground_scm.MakeStructuralEquations import (
        MakeStructuralEquations,
        make_additive_noise_gaussian,
    )
except Exception:  # pragma: no cover — env may not have DoPFN on sys.path at import
    MakeStructuralEquations = None  # type: ignore
    make_additive_noise_gaussian = None  # type: ignore


def _sample_bivariate_gaussian(
    shape: tuple,
    std: float,
    rho: float,
    eps_obs: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (eps_obs, eps_int) ~ N(0, std^2 * [[1, rho], [rho, 1]]).

    If `eps_obs` is provided, only `eps_int` is drawn (correlated to the
    given obs draw). This is how `install_bivariate_y_noise` preserves the
    marginal distribution of the original DoPFN noise sample.

    Parameters
    ----------
    shape : tuple
        Sample tensor shape, e.g. (batch_size, seq_len).
    std : float
        Marginal standard deviation.
    rho : float
        Correlation coefficient in [-1, 1].
    eps_obs : optional torch.Tensor
        If given, reused as the obs draw.
    """
    if not (-1.0 <= rho <= 1.0):
        raise ValueError(f"y_noise_correlation must lie in [-1, 1]; got {rho}")
    if eps_obs is None:
        eps_obs = torch.normal(0.0, std, shape)
    else:
        assert tuple(eps_obs.shape) == tuple(shape), (
            f"eps_obs shape {tuple(eps_obs.shape)} != requested shape {tuple(shape)}"
        )
    # Independent partner draw at the same marginal std.
    eps_indep = torch.normal(0.0, std, shape)
    # eps_int = rho * eps_obs + sqrt(1 - rho^2) * eps_indep
    # gives Cov(eps_obs, eps_int) = rho * std^2 with Var(eps_int) = std^2.
    coeff_partner = math.sqrt(max(1.0 - rho * rho, 0.0))
    eps_int = rho * eps_obs + coeff_partner * eps_indep
    return eps_obs, eps_int


if MakeStructuralEquations is not None:

    class MakeStructuralEquationsBivariate(MakeStructuralEquations):  # type: ignore[misc]
        """Y-node structural equation with correlated obs/int noise draws.

        Parameters
        ----------
        parents, samples_shape, noise_std, noise_dist, nonlins, max_hidden_layers
            Forwarded to the base MakeStructuralEquations.
        y_noise_correlation : float, default 0.0
            Correlation rho between the obs and int noise draws.

        Notes
        -----
        - The base class already samples one noise tensor into
          `self.additive_noise` at __init__. We adopt that as the obs
          draw (`self.additive_noise_obs`) and generate a partner draw
          `self.additive_noise_int` with the requested correlation.
        - `.forward(is_int_call=False, **kwargs)` temporarily swaps
          `self.additive_noise` to the chosen draw, calls the base
          `forward()`, then restores. This preserves the base's linear +
          activation math verbatim.
        - Non-Y nodes should keep using the plain `MakeStructuralEquations`
          so their `additive_noise` remains identical on obs and int calls
          (which, combined with `exogenous_vars=exo_obs` reuse, keeps the
          non-Y ancestors reproducible across the two passes).
        """

        def __init__(
            self,
            parents,
            samples_shape,
            noise_std: float,
            noise_dist: str,
            nonlins: str,
            max_hidden_layers: int,
            y_noise_correlation: float = 0.0,
        ) -> None:
            super().__init__(
                parents=parents,
                samples_shape=samples_shape,
                noise_std=noise_std,
                noise_dist=noise_dist,
                nonlins=nonlins,
                max_hidden_layers=max_hidden_layers,
            )
            if noise_dist != "gaussian":
                # We only implement the Gaussian bivariate here — DoPFN uses
                # noise_dist='gaussian' for the case studies. Extending to
                # laplace/student/gumbel would need a Gaussian-copula draw.
                raise NotImplementedError(
                    "MakeStructuralEquationsBivariate currently supports "
                    f"noise_dist='gaussian' only; got {noise_dist!r}"
                )
            self.y_noise_correlation = float(y_noise_correlation)
            self._noise_std = float(noise_std)
            # `self.additive_noise` was set by the base __init__ as eps_obs.
            eps_obs = self.additive_noise
            eps_obs, eps_int = _sample_bivariate_gaussian(
                shape=tuple(samples_shape),
                std=float(noise_std),
                rho=self.y_noise_correlation,
                eps_obs=eps_obs,
            )
            self.additive_noise_obs = eps_obs
            self.additive_noise_int = eps_int
            # Default active tensor = obs (first call is observational).
            self.additive_noise = self.additive_noise_obs
            self._bivariate_noise_installed = True

        def forward(self, is_int_call: bool = False, **kwargs) -> torch.Tensor:  # type: ignore[override]
            """Route to the appropriate noise draw, then call base forward.

            The base MakeStructuralEquations.forward does not accept an
            `is_int_call` kwarg; we consume it here and toggle
            `self.additive_noise` accordingly. Non-treatment/non-Y calls
            can safely omit `is_int_call` — default is `False` (obs).
            """
            noise = self.additive_noise_int if is_int_call else self.additive_noise_obs
            prev = self.additive_noise
            self.additive_noise = noise
            try:
                out = super().forward(**kwargs)
            finally:
                self.additive_noise = prev
            return out

        def get_noiseless_mean(self, **kwargs) -> torch.Tensor:
            """Compute the structural mean at the Y-node WITHOUT additive noise.

            Used to derive mu_0(x), mu_1(x) per-query. Mirrors the base
            `.forward()` math but skips the `+= self.additive_noise` step.
            For the 'post' nonlinearity convention (where the activation is
            applied AFTER noise addition), we also skip the noise so the
            returned tensor is `activation(linear(parents))` — the
            conditional mean of Y | parents, holding noise = 0.
            """
            if len(self.parents) == 0:
                return torch.zeros_like(self.additive_noise_obs)
            parent_values = [kwargs[parent] for parent in self.parents]
            parent_tensor = torch.stack(parent_values, dim=-1)
            with torch.no_grad():
                out = self.layers(parent_tensor).squeeze(-1)
                # Both nonlins conventions apply the activation to the
                # linear combination; only 'post' shifts noise-addition
                # ordering. For the mean we drop noise in both cases.
                out = self.activation(out)
            return out


def install_bivariate_y_noise(scm, y_noise_correlation: float, noise_std: float):
    """Post-hoc: attach correlated obs/int noise tensors to the Y-node.

    Preserves the EXISTING MakeStructuralEquations instance (with its
    fitted `nn.Linear` and sampled activation) — we just add two extra
    tensors and a flag so `set_y_call_mode` can swap between them.

    Parameters
    ----------
    scm : StructuralCausalModel
        Must have `y_key` set (DoSCM does this after graph creation).
    y_noise_correlation : float
        rho in [-1, 1] for the bivariate Gaussian.
    noise_std : float
        Marginal std of the Y-node noise; should match the value used
        when the SCM was built.

    Returns
    -------
    The Y-node MakeStructuralEquations instance, now with
    `additive_noise_obs`, `additive_noise_int`, `y_noise_correlation`,
    and `_bivariate_noise_installed = True` set. Its `additive_noise`
    attribute is initialised to the obs draw so the first
    `scm.get_next_sample(...)` call (observational) uses it.
    """
    y_key = getattr(scm, "y_key", None)
    if y_key is None:
        raise ValueError("scm.y_key must be set before installing bivariate Y noise")
    if y_key not in scm.functions:
        raise KeyError(f"Y-node {y_key!r} not found in scm.functions")
    y_fn, _y_map = scm.functions[y_key]
    if not hasattr(y_fn, "additive_noise"):
        raise TypeError(
            f"Y-node function {type(y_fn).__name__} has no `additive_noise` attribute; "
            "cannot install bivariate noise"
        )
    eps_obs = y_fn.additive_noise
    shape = tuple(eps_obs.shape)
    eps_obs, eps_int = _sample_bivariate_gaussian(
        shape=shape,
        std=float(noise_std),
        rho=float(y_noise_correlation),
        eps_obs=eps_obs,
    )
    y_fn.additive_noise_obs = eps_obs
    y_fn.additive_noise_int = eps_int
    y_fn.y_noise_correlation = float(y_noise_correlation)
    y_fn._noise_std = float(noise_std)
    y_fn._bivariate_noise_installed = True
    # Ensure obs is the active tensor for the first (observational) call.
    y_fn.additive_noise = y_fn.additive_noise_obs
    return y_fn


def set_y_call_mode(scm, mode: str) -> None:
    """Swap the Y-node's active noise tensor between the obs and int draws.

    No-op if the Y-node hasn't had `install_bivariate_y_noise` called on
    it yet (so callers can invoke this unconditionally).

    Parameters
    ----------
    scm : StructuralCausalModel
    mode : {'obs', 'int'}
    """
    y_fn, _ = scm.functions[scm.y_key]
    if not getattr(y_fn, "_bivariate_noise_installed", False):
        return
    if mode == "obs":
        y_fn.additive_noise = y_fn.additive_noise_obs
    elif mode == "int":
        y_fn.additive_noise = y_fn.additive_noise_int
    else:
        raise ValueError(f"mode must be 'obs' or 'int'; got {mode!r}")


def get_y_noise_std_empirical(scm) -> float:
    """Return the empirical std of the Y-node's noise tensor.

    Uses the obs draw if bivariate noise is installed, else falls back to
    the single stored `additive_noise` tensor. This matches the user's
    spec ('Y-node's stored noise tensor std across the whole realization')
    and gives sigma_eps for closed-form Gaussian truth densities.
    """
    y_fn, _ = scm.functions[scm.y_key]
    if getattr(y_fn, "_bivariate_noise_installed", False):
        noise = y_fn.additive_noise_obs
    else:
        noise = y_fn.additive_noise
    return float(noise.detach().std().item())
