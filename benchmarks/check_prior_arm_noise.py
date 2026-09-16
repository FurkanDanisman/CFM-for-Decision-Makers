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

    cd $CAUSALPFN
    PYTHONPATH=$CAUSALPFN/src:$CAUSALPFN:$REPO \\
        python $REPO/benchmarks/check_prior_arm_noise.py
"""
from __future__ import annotations

import sys

import torch


def main() -> int:
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    import os

    causalpfn = os.environ.get("CAUSALPFN") or os.getcwd()
    cfg_dir = os.path.join(causalpfn, "configs")
    if not os.path.isdir(cfg_dir):
        print(f"FATAL: no configs/ under {causalpfn}; set CAUSALPFN", file=sys.stderr)
        return 1

    # CausalPFN's root config name is read off train.py's @hydra.main rather
    # than hard-coded, so this keeps working if upstream renames it.
    cfg_name = os.environ.get("CPFN_CONFIG_NAME")
    if not cfg_name:
        import re
        src = open(os.path.join(causalpfn, "train.py")).read()
        m = re.search(r'config_name\s*=\s*["\']([^"\']+)["\']', src)
        cfg_name = m.group(1) if m else "train"
    print(f"[prior] hydra config_name={cfg_name}")

    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name=cfg_name, overrides=["+experiment=simple_configuration"])
    dataset = instantiate(cfg.data.dataset if "dataset" in cfg.data else cfg.data)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))

    keys = sorted(batch.keys())
    print(f"[prior] batch keys: {keys}")

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
    print(f"[patch] factual column rebuilt; {frac_changed:.1%} of entries differ "
          f"(expect ~ the treated fraction)")

    ok = abs(rho_shared - 1.0) < 1e-6 and max_dev < 1e-6
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
