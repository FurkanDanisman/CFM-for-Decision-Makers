"""Training loop for both tasks.

    task 1   --variant dopfn_1d           the faithful Do-PFN reproduction
    task 1b  --variant dopfn_1d_botharms  task 1, both arms of every query unit
    task 2   --variant joint_2d           the joint version of task 1

Everything outside the head is held identical between them: prior, SCM draw
order, optimiser, schedule, backbone initialisation, target space, and the
context rows. That is the whole point -- the head is the only thing that
moves, so any difference is attributable to it.

``dopfn_1d_botharms`` is the one deliberate exception. It shares the 1-D head,
the loss and the fitted borders with ``dopfn_1d`` exactly, and differs only in
the query block: both arms of every query unit instead of the prior's
coin-flipped one. Same estimand -- column 0 is part of the query, so both fit
p(y | do(t), x, context) -- but it puts the same outcome values in front of the
1-D head that the joint head already sees. Without it, a 1-D/joint gap is
confounded with the joint having had twice the outcomes per SCM draw.

The optimiser is not tuned. It is read off the released checkpoint's
``optimizer_state.param_groups``: Adam, lr 8.4853e-5, betas (0.9, 0.999),
eps 1e-8, weight_decay 0.0, single param group. The schedule is not recorded
anywhere, but the checkpoint's live lr is 4.7e-11 x the base rate, which is
where a single cosine cycle lands in its final epoch -- so cosine to zero, with
the config's 256/2048 warmup fraction.

Loss is reported in nats per outcome for both heads (the 2-D joint is halved),
so the same learning rate transfers and the curves are readable against each
other.

Usage
-----
    python training_dopfn_repro/train.py --variant dopfn_1d          --steps 150000
    python training_dopfn_repro/train.py --variant dopfn_1d_botharms --steps 150000
    python training_dopfn_repro/train.py --variant joint_2d          --steps 150000

Needs torch 2.1 / python 3.10 -- Do-PFN's model/layer.py does
``from torch.nn.modules.transformer import Optional``, which no longer resolves.
Cluster submission is deliberately out of scope here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from training_dopfn_repro.batch import (  # noqa: E402
    BatchConfig,
    make_1d_batch,
    make_1d_botharms_batch,
    make_joint_batch,
    sample_single_eval_pos,
)
from training_dopfn_repro.borders import (  # noqa: E402
    collect_targets,
    fit_bar_borders_1d,
    fit_grid_edges_2d,
)
from training_dopfn_repro.model import (  # noqa: E402
    VARIANTS,
    ModelSpec,
    _in_dopfn_root,
    build_backbone_init,
    build_model,
    is_1d,
    loss_1d,
    loss_joint_2d,
)
from training_dopfn_repro.prior import PriorConfig, sample_batch  # noqa: E402

# Recorded in the released checkpoint. Do not tune these.
ADAM_LR = 8.485281374238571e-05
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-08
ADAM_WEIGHT_DECAY = 0.0
WARMUP_FRACTION = 256 / 2048          # config: warmup_epochs / epochs
BAR_DIST_INIT_BATCHES = 100           # config: bar_dist_init_batches

#: Which builder each 1-D variant uses. Both feed the identical head and loss;
#: only the query block differs. See batch.make_1d_botharms_batch.
_1D_BUILDERS = {
    "dopfn_1d": make_1d_batch,
    "dopfn_1d_botharms": make_1d_botharms_batch,
}


# ---------------------------------------------------------------------------
# prior stream
# ---------------------------------------------------------------------------


class PriorStream(torch.utils.data.Dataset):
    """Batches as a pure function of step index.

    Being seed-addressed rather than RNG-order-dependent buys two things: the
    stream is identical regardless of worker count, and both variants see the
    same SCMs in the same order, which makes the comparison paired.
    """

    def __init__(self, variant, seed_base, prior_cfg, batch_cfg, length=10**9):
        self.variant = variant
        self.seed_base = seed_base
        self.prior_cfg = prior_cfg
        self.batch_cfg = batch_cfg
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        seed = self.seed_base + int(index)
        rec = sample_batch(seed, self.prior_cfg)
        sep = sample_single_eval_pos(seed, self.prior_cfg.seq_len, self.batch_cfg)
        if is_1d(self.variant):
            out = _1D_BUILDERS[self.variant](rec, sep, self.batch_cfg)
            keep = ("train_x", "train_y", "test_x", "target")
        else:
            out = make_joint_batch(rec, sep, self.batch_cfg)
            keep = ("train_x", "train_y", "test_x", "y_do0", "y_do1")
        batch = {k: out[k] for k in keep}
        batch["nonfinite"] = torch.tensor(bool(rec["meta"]["nonfinite"]))
        return batch


def make_loader(variant, seed_base, prior_cfg, batch_cfg, workers):
    stream = PriorStream(variant, seed_base, prior_cfg, batch_cfg)
    return torch.utils.data.DataLoader(
        stream,
        batch_size=None,          # the prior already emits a full batch
        num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=3 if workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------


def cosine_to_zero(optimizer, total_steps, warmup_fraction=WARMUP_FRACTION):
    """Linear warmup, then a single cosine cycle decaying to ~0.

    Not a floor at some fraction of base: the released optimizer state sits at
    4.7e-11 x base, which only a full cycle produces.
    """
    warmup = max(1, int(warmup_fraction * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


def install_criterion(model, variant, spec, prior_cfg, batch_cfg, n_batches, device):
    """Fit the head's support from the prior before step one.

    Both heads are fitted from the same draws and the same target space; they
    differ only in how resolution is allocated, which the 2-D loss forces.

    ``collect_targets`` pools BOTH arms for every variant, so the 1-D borders do
    not move between ``dopfn_1d`` and ``dopfn_1d_botharms``: at a fixed seed the
    two runs start from a byte-identical head over a byte-identical support.
    """
    print(f"fitting target support from {n_batches} prior batches ...")
    targets = collect_targets(
        n_batches=n_batches, prior_cfg=prior_cfg, batch_cfg=batch_cfg
    )
    pooled = torch.cat([targets["context"], targets["query"]])

    if is_1d(variant):
        borders = fit_bar_borders_1d(pooled, spec.num_buckets)
        with _in_dopfn_root():
            from model.bar_distribution import FullSupportBarDistribution

        model.criterion = FullSupportBarDistribution(borders).to(device)
        widths = model.criterion.bucket_widths
        print(
            f"  1-D borders: {spec.num_buckets} quantile buckets over "
            f"[{borders.min():.3f}, {borders.max():.3f}], "
            f"widths {widths.min():.4f}-{widths.max():.4f}"
        )
        return None

    # Same pooled draws as the 1-D borders, from the same fixed seed, so the two
    # heads are fitted on identical data and differ only in how they allocate
    # resolution -- which the 2-D loss forces, not us.
    edges = fit_grid_edges_2d(pooled, spec.j_2d).to(device)
    bw = float(edges[1] - edges[0])
    outside = ((pooled < edges[0].cpu()) | (pooled > edges[-1].cpu())).float().mean()
    print(
        f"  2-D grid: {spec.j_2d}x{spec.j_2d} uniform over "
        f"[{edges[0]:.3f}, {edges[-1]:.3f}], bin width {bw:.4f}, "
        f"{outside:.2%} of mass in the tail regions"
    )
    return edges


def compute_loss(model, batch, variant, spec, edges, device):
    train_x = batch["train_x"].to(device, non_blocking=True)
    train_y = batch["train_y"].to(device, non_blocking=True)
    test_x = batch["test_x"].to(device, non_blocking=True)

    logits = model(train_x, train_y, test_x, only_return_standard_out=True)

    if is_1d(variant):
        # botharms stacks arm 0 then arm 1 along the row axis, so this mean is
        # still nats per outcome -- the same units as the halved joint loss.
        return loss_1d(model.criterion, logits, batch["target"].to(device))
    return loss_joint_2d(
        logits,
        batch["y_do0"].to(device),
        batch["y_do1"].to(device),
        spec.j_2d,
        edges,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--seq-len", type=int, default=2200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--j-2d", type=int, default=10)
    ap.add_argument("--num-buckets", type=int, default=100)
    ap.add_argument("--lr", type=float, default=ADAM_LR)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--backbone-seed", type=int, default=0)
    ap.add_argument("--head-seed", type=int, default=0)
    ap.add_argument("--stream-seed", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=6)   # config: 6
    ap.add_argument("--amp", choices=("fp16", "bf16", "off"), default="fp16")
    ap.add_argument("--out", default="checkpoints_dopfn_repro")
    ap.add_argument("--backbone-init", default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=5_000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.out, args.variant)
    os.makedirs(out_dir, exist_ok=True)

    prior_cfg = PriorConfig(seq_len=args.seq_len, batch_size=args.batch_size)
    batch_cfg = BatchConfig()
    spec = ModelSpec(
        variant=args.variant,
        num_buckets=args.num_buckets,
        j_2d=args.j_2d,
        head_seed=args.head_seed,
    )

    # One canonical backbone init, shared by every variant. Derived once and
    # hashed; see model.py for why a shared seed is not sufficient.
    init_path = args.backbone_init or os.path.join(
        args.out, f"backbone_init_s{args.backbone_seed}.pt"
    )
    if not os.path.exists(init_path):
        print(f"deriving backbone init (seed {args.backbone_seed}) -> {init_path}")
        digest = build_backbone_init(args.backbone_seed, init_path)
        print(f"  sha256 {digest[:16]}")

    model = build_model(spec, init_path).to(device)
    print(
        f"variant={args.variant}  n_out={spec.n_out}  "
        f"params={sum(p.numel() for p in model.parameters()):,}  device={device}\n"
        f"  backbone sha256 {model.dopfn_repro_spec['backbone_sha256'][:16]}"
    )

    edges = install_criterion(
        model, args.variant, spec, prior_cfg, batch_cfg,
        BAR_DIST_INIT_BATCHES, device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=ADAM_WEIGHT_DECAY,
    )
    scheduler = cosine_to_zero(optimizer, args.steps)
    use_scaler = args.amp == "fp16" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.amp)

    start_step = 0
    latest = os.path.join(out_dir, "latest.pt")
    if args.resume and os.path.exists(latest):
        blob = torch.load(latest, map_location=device)
        model.load_state_dict(blob["model"])
        optimizer.load_state_dict(blob["optimizer"])
        scheduler.load_state_dict(blob["scheduler"])
        if blob.get("scaler") is not None:
            scaler.load_state_dict(blob["scaler"])
        if blob.get("edges") is not None:
            edges = blob["edges"].to(device)
        start_step = int(blob["step"])
        print(f"resumed from {latest} at step {start_step}")

    provenance = {
        "variant": args.variant,
        "spec": asdict(spec),
        "prior_cfg": asdict(prior_cfg),
        "batch_cfg": asdict(batch_cfg),
        "optimizer": {
            "name": "Adam",
            "lr": args.lr,
            "betas": list(ADAM_BETAS),
            "eps": ADAM_EPS,
            "weight_decay": ADAM_WEIGHT_DECAY,
            "schedule": "cosine_to_zero",
            "warmup_fraction": WARMUP_FRACTION,
        },
        "steps": args.steps,
        "stream_seed": args.stream_seed,
        "backbone": model.dopfn_repro_spec,
        "amp": args.amp,
    }
    with open(os.path.join(out_dir, "provenance.json"), "w") as fh:
        json.dump(provenance, fh, indent=2, default=str)

    loader = make_loader(
        args.variant, args.stream_seed + start_step, prior_cfg, batch_cfg,
        args.workers,
    )

    def save(path, step):
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if use_scaler else None,
                "edges": edges.cpu() if edges is not None else None,
                "provenance": provenance,
            },
            path,
        )

    print(f"\n{'step':>8} {'loss':>11} {'lr':>10} {'s/step':>8} {'skipped':>8}")
    print("-" * 50)

    model.train()
    running, n_running, skipped = 0.0, 0, 0
    t0 = time.time()
    step = start_step

    for batch in loader:
        if step >= args.steps:
            break

        # doscm marks a degenerate SCM by filling the batch with -100; training
        # on that is meaningless, so step over it rather than let it move the
        # weights. Counted and reported, never silent.
        if bool(batch["nonfinite"]):
            skipped += 1
            step += 1
            continue

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None and device.type == "cuda",
        ):
            loss = compute_loss(model, batch, args.variant, spec, edges, device)

        if not torch.isfinite(loss):
            skipped += 1
            step += 1
            continue

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        scheduler.step()

        running += float(loss)
        n_running += 1
        step += 1

        if step % args.log_every == 0 and n_running:
            dt = (time.time() - t0) / args.log_every
            print(
                f"{step:>8} {running / n_running:>11.4f} "
                f"{scheduler.get_last_lr()[0]:>10.3e} {dt:>8.3f} {skipped:>8}"
            )
            running, n_running, t0 = 0.0, 0, time.time()

        if step % args.ckpt_every == 0:
            save(os.path.join(out_dir, f"step_{step}.pt"), step)
            save(latest, step)

    save(os.path.join(out_dir, f"step_{step}_final.pt"), step)
    save(latest, step)
    print(f"\ndone at step {step}; {skipped} batches skipped. -> {out_dir}")


if __name__ == "__main__":
    main()
