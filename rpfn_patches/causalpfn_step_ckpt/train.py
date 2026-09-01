"""Modified CausalPFN train.py — model-agnostic resume path.

CHANGE vs upstream (marked with `# ── A1 PATCH ──`):

  Original code (lines 37-45 of upstream train.py):
      if resume:
          model = InContextModel.load(
              model_state=checkpoint["model_state_dict"],
              model_config=checkpoint["model_config"],
          )
      else:
          model = instantiate(conf.model.obj)

  Problem: on resume, this HARDCODES `InContextModel.load(...)`, which
  reconstructs an `InContextModel` regardless of what `model.obj._target_`
  actually points to. That breaks any custom model class (e.g. our 2D
  head via `rpfn_patches.cpfn2d_loader.load_cpfn2d_from_tabdpt`).

  Fix: use hydra to instantiate the same _target_ on resume as on first
  run, then restore the weights via load_state_dict. This is what
  Checkpoint saves (`model_state_dict = model.state_dict()`), so the
  round-trip is exact for any nn.Module.

Everything else — wandb setup, callback wiring, train() call — is
identical to upstream.
"""

from typing import Any

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from causalpfn.models import InContextModel
from causalpfn.training.callbacks import Checkpoint
from causalpfn.training.distributed import cleanup, init_dist, install_signal_handler, seed_everything
from causalpfn.training.trainer import train

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(conf: DictConfig) -> Any:
    install_signal_handler()
    using_dist, rank, device, world_size = init_dist(conf.default_device)
    seed_everything(conf.seed, using_dist, rank)

    checkpoint = None
    resume = False
    run_name = None
    wandb_run_id = None
    if conf.resume_training.enabled and conf.resume_training.checkpoint_path is not None:
        checkpoint = torch.load(conf.resume_training.checkpoint_path, weights_only=False, map_location=device)
        run_name = checkpoint["run_name"]
        if "wandb" in run_name:
            wandb_run_id = run_name.split("-")[-1]
        resume = True

    # ── A1 PATCH ── model-agnostic init + resume. Same _target_ (hydra) both
    # for first-run and resume. Weights are restored via load_state_dict, so
    # this works for any nn.Module — including CausalPFN2DHead.
    model = instantiate(conf.model.obj)
    if resume:
        missing, unexpected = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False,
        )
        if rank == 0:
            print(f"[resume] loaded {len(checkpoint['model_state_dict'])} tensors "
                  f"from {conf.resume_training.checkpoint_path}")
            if missing:
                print(f"[resume] WARNING: {len(missing)} tensors missing in ckpt "
                      f"(kept as random init): first 5 = {missing[:5]}")
            if unexpected:
                print(f"[resume] WARNING: {len(unexpected)} tensors in ckpt not in "
                      f"model (ignored): first 5 = {unexpected[:5]}")
    train_meta_dataset = instantiate(conf.train_meta_dataset)

    if conf.wandb.enabled and rank == 0:
        wandb_run_name = str(conf.wandb.run_name) if conf.wandb.run_name is not None else None
        tags = [f"{key}:{value}" for key, value in conf.wandb.tags.items()] if "tags" in conf.wandb else []
        tags.append(f"num_gpus:{world_size}")
        wandb.init(
            project=conf.wandb.project,
            entity=conf.wandb.entity,
            config=OmegaConf.to_container(conf, resolve=True),
            name=None if resume and wandb_run_id is not None else wandb_run_name,
            tags=tags,
            settings=wandb.Settings(start_method="thread"),
            id=wandb_run_id,
            resume="must" if resume and wandb_run_id is not None else "never",
        )

    callbacks = [instantiate(callback) for callback in conf.get("callbacks", {}).values()]
    for callback in callbacks:
        if isinstance(callback, Checkpoint) and resume:
            callback.load_state(checkpoint)

    try:
        train(
            model=model,
            optimizer_partial=instantiate(conf.optimizer),
            train_meta_dataset=train_meta_dataset,
            callbacks=callbacks,
            lr_scheduler_partial=instantiate(conf.lr_scheduler) if conf.get("lr_scheduler") else None,
            compile=conf.compile,
            num_workers=conf.num_workers,
            prefetch_factor=conf.prefetch_factor,
            checkpoint=checkpoint,
            wandb_enabled=conf.wandb.enabled and rank == 0,
            device=device,
            rank=rank,
            using_dist=using_dist,
            world_size=world_size,
            **OmegaConf.to_container(conf.trainer, resolve=True),
        )
    finally:
        if conf.wandb.enabled and rank == 0:
            wandb.finish()
        if using_dist:
            cleanup()


if __name__ == "__main__":
    main()
