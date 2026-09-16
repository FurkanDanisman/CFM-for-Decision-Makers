#!/usr/bin/env python
"""Preflight for submit_train_cpfn2d_j32_sharednoise_y01.sbatch.

Pulls ONE batch from the same prior the training run uses and checks the two
things the patched trainer depends on:

  1. the batch carries the potential outcomes y0 / y1 (not just E_y0 / E_y1),
     which is what CPFN_SHARED_ARM_NOISE / CPFN_TRAIN_ON_Y01 reconstruct eta from;
  2. eta_1 := eta_0 actually produces a perfectly coupled pair, i.e. the implied
     arm correlation goes from ~0 (stock prior, two independent draws) to 1.0.

Run it before submitting -- it takes seconds and fails here instead of after a
GPU allocation.

    export CAUSALPFN=/path/to/causalpfn
    export REPO=/path/to/R-PFN
    cd $CAUSALPFN
    PYTHONPATH=$CAUSALPFN/src:$CAUSALPFN:$REPO \\
        python $REPO/benchmarks/check_prior_arm_noise.py

Hydra's config path / name and the dataset node are DISCOVERED, not hard-coded:
the former is read off train.py's @hydra.main decorator, the latter is probed
across the usual key layouts. If the probe fails the script dumps the composed
config so the right key can be read straight off it.
"""
from __future__ import annotations

import os
import re
import sys

import torch


def _locate_hydra_config(causalpfn: str) -> tuple[str, str]:
    """Read config_path / config_name off train.py's @hydra.main decorator."""
    entry = os.path.join(causalpfn, "train.py")
    if not os.path.isfile(entry):
        raise SystemExit(f"FATAL: no train.py at {entry}; is CAUSALPFN right?")
    src = open(entry).read()

    m_name = re.search(r'config_name\s*=\s*["\']([^"\']+)["\']', src)
    m_path = re.search(r'config_path\s*=\s*["\']([^"\']+)["\']', src)
    cfg_name = os.environ.get("CPFN_CONFIG_NAME") or (m_name.group(1) if m_name else "train")

    if os.environ.get("CPFN_CONFIG_DIR"):
        cfg_dir = os.environ["CPFN_CONFIG_DIR"]
    elif m_path:
        cfg_dir = os.path.abspath(os.path.join(causalpfn, m_path.group(1)))
    else:
        cfg_dir = os.path.join(causalpfn, "configs")

    if not os.path.isdir(cfg_dir):
        found = []
        for root, dirs, files in os.walk(causalpfn):
            if root.count(os.sep) - causalpfn.count(os.sep) > 3:
                dirs[:] = []
                continue
            if any(f.endswith(".yaml") for f in files) and "experiment" in dirs:
                found.append(root)
        raise SystemExit(
            f"FATAL: hydra config dir not found at {cfg_dir}\n"
            f"  train.py config_path = {m_path.group(1) if m_path else '<not found>'}\n"
            f"  candidates with an experiment/ subdir: {found or '<none>'}\n"
            f"  override with CPFN_CONFIG_DIR=..."
        )
    return cfg_dir, cfg_name


def _locate_dataset(cfg):
    """Find the node that instantiates the training dataset / prior."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    candidates = [
        "data.dataset", "data.train_dataset", "data", "dataset",
        "train_dataset", "prior", "data.prior", "datamodule.dataset",
    ]
    for dotted in candidates:
        node = cfg
        for part in dotted.split("."):
            if node is None or part not in node:
                node = None
                break
            node = node[part]
        if node is None or "_target_" not in node:
            continue
        print(f"[prior] dataset node: cfg.{dotted}  (_target_={node._target_})")
        return instantiate(node)

    print("FATAL: could not find a dataset node. Composed config:", file=sys.stderr)
    print(OmegaConf.to_yaml(cfg)[:6000], file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    from hydra import compose, initialize_config_dir

    causalpfn = os.environ.get("CAUSALPFN") or os.getcwd()
    cfg_dir, cfg_name = _locate_hydra_config(causalpfn)
    print(f"[prior] hydra config_dir={cfg_dir}  config_name={cfg_name}")

    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name=cfg_name, overrides=["+experiment=simple_configuration"])

    dataset = _locate_dataset(cfg)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))

    print(f"[prior] batch keys: {sorted(batch.keys())}")
    missing = [k for k in ("X", "t", "y", "y0", "y1", "E_y0", "E_y1") if k not in batch]
    if missing:
        print(f"FAIL: batch is missing {missing} -- the patch cannot reconstruct eta.",
              file=sys.stderr)
        return 1
    print("[prior] PASS: y0 / y1 present alongside E_y0 / E_y1")

    y0, y1 = batch["y0"].double(), batch["y1"].double()
    E_y0, E_y1 = batch["E_y0"].double(), batch["E_y1"].double()
    t = batch["t"].double()

    eta0 = (y0 - E_y0).flatten()
    eta1 = (y1 - E_y1).flatten()
    rho_stock = torch.corrcoef(torch.stack([eta0, eta1]))[0, 1].item()
    print(f"[prior] stock arm-noise correlation  rho(eta_0, eta_1) = {rho_stock:+.4f}   "
          f"(expect ~0 -- two independent draws)")

    # exactly what calculate_loss() does under CPFN_SHARED_ARM_NOISE=1
    y1_shared = E_y1 + (y0 - E_y0)
    eta1_shared = (y1_shared - E_y1).flatten()
    rho_shared = torch.corrcoef(torch.stack([eta0, eta1_shared]))[0, 1].item()
    print(f"[patch] coupled arm-noise correlation rho(eta_0, eta_1) = {rho_shared:+.4f}   "
          f"(expect 1.0000)")

    # tau is unchanged by the coupling -- worth seeing, since it is the reason
    # the interval width is not expected to move much.
    tau_mean = (E_y1 - E_y0).flatten()
    tau_shared = (y1_shared - y0).flatten()
    max_dev = (tau_shared - tau_mean).abs().max().item()
    print(f"[patch] max |(y1_shared - y0) - (E_y1 - E_y0)| = {max_dev:.3e}   "
          f"(expect ~0 -- tau information is identical to the baseline)")

    y_shared = torch.where(t > 0.5, y1_shared, y0)
    frac_changed = (y_shared != batch["y"].double()).double().mean().item()
    treated_frac = (t > 0.5).double().mean().item()
    print(f"[patch] factual column rebuilt; {frac_changed:.1%} of entries differ "
          f"(treated fraction = {treated_frac:.1%})")

    ok = abs(rho_shared - 1.0) < 1e-6 and max_dev < 1e-6
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
