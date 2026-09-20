#!/usr/bin/env python
"""Publish the benchmark checkpoints to a Hugging Face repo, with model cards
generated from the checkpoint bytes rather than hand-written.

Every fact in a card -- J, training step, variant, head width, grid range, sha256
-- is read out of the file being uploaded. Hand-written cards drift from the
weights, and this project has already had one checkpoint's J misremembered.

    python benchmarks/hf_publish.py --plan                     # what would happen
    python benchmarks/hf_publish.py --cards-only --out /tmp/cards
    python benchmarks/hf_publish.py --upload --repo furkanbd/r-pfn-checkpoints

Defaults to a PRIVATE repo; pass --public to change that, deliberately.

Uploads the .pt files exactly as the evaluation pipeline loads them, so a
downloaded checkpoint behaves identically to the one used for the results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Model -> a one-line description of what distinguishes it. The rest of each card
# is read from the file.
NOTES = {
    "dopfn_bb_j10_step_150000":      "DoPFN backbone with a 2D joint head over (Y0, Y1).",
    "dopfn_repro_1d_J10":            "From-scratch DoPFN 1D reproduction at J=10 bins.",
    "dopfn_repro_1d_J100":           "From-scratch DoPFN 1D reproduction at J=100 bins.",
    "dopfn_repro_joint2d":           "From-scratch DoPFN reproduction with a 2D joint head.",
    "graph2d_step_50000":            "Graph-conditioned 2D joint head.",
    "cpfn1d_j1024_headrand_step_50000": "CausalPFN 1D head, J=1024, randomised head init.",
    "cpfn1d_j32_step50000":          "CausalPFN 1D head at J=32.",
    "cpfn1d_botharms_step50000":     "CausalPFN 1D head, J=1024, supervised on both arms.",
    "cpfn2d_j32_random_step_50000":  "CausalPFN 2D joint head at J=32.",
    "cpfn2d_j32_eta0_y01_step50000": "CausalPFN 2D joint head, eta_1 := eta_0 ablation, "
                                     "supervised on y0/y1.",
    "cpfn_v0_original":              "Original CausalPFN v0 weights.",
    "uwyk_reproduce_best_model":     "UWYK graph-conditioned model.",
    "uwyk_bin_step50000":            "UWYK variant with binarised treatment encoding.",
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def describe(path):
    from inspect_ckpt import describe as _d
    try:
        return _d(path)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def card(name, path, info, digest):
    def g(k, default="not recorded"):
        v = info.get(k, default)
        return default if v is None else v

    # Resolved by architecture in inspect_ckpt.resolve_j, with the evidence named.
    # Taking the first matching formula instead reported head widths as J -- 1034
    # for a J=1024 model, 113 for a J=10 one.
    J = info.get("J", "undetermined")
    J_src = info.get("J_source", "")
    lines = [
        "---", "library_name: pytorch", "tags:", "  - causal-inference",
        "  - tabular", "  - prior-fitted-network", "---", "",
        f"# {name}", "",
        NOTES.get(name, "Research checkpoint."), "",
        "## What this file is", "",
        "Every value below was read from the checkpoint itself, not recorded by hand.",
        "", "| field | value |", "|---|---|",
        f"| variant | `{g('variant')}` |",
        f"| training step | `{g('step')}` |",
        f"| bins (J) | `{J}` |" + (f" <!-- {J_src} -->" if J_src else ""),
        f"| head output width | `{g('head_width')}` |",
        f"| J determined from | {J_src or 'n/a'} |",
        f"| tensors in state dict | `{g('n_tensors')}` |",
        f"| size | `{os.path.getsize(path) / 1e6:.1f} MB` |",
        f"| sha256 | `{digest}` |",
    ]
    if "edges_range" in info:
        lines += [f"| 2D grid range | `{info['edges_range']}` |",
                  f"| grid space | `{info.get('edges_space', '?')}` |"]
    lines += [
        "", "## Loading", "",
        "```python", "import torch",
        f'blob = torch.load("{name}.pt", map_location="cpu", weights_only=False)',
        'sd = blob.get("model_state_dict") or blob.get("model")',
        "```", "",
        "`weights_only=False` is required: the checkpoint carries its config, grid",
        "edges and provenance alongside the tensors.", "",
        "## Caveats", "",
        "- Research checkpoints from a benchmark study, not a packaged library.",
        "- This is the exact file used to produce the reported results.",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=os.path.join(
        os.path.dirname(_HERE), "Required_checkpoints"))
    ap.add_argument("--repo", default="furkanbd/r-pfn-checkpoints")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--cards-only", action="store_true")
    ap.add_argument("--out", default=None, help="where to write cards")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--update-cards", action="store_true",
                    help="upload ONLY the cards, leaving the .pt files "
                         "already in the repo untouched.")
    ap.add_argument("--public", action="store_true")
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(a.ckpt_dir) if f.endswith(".pt"))
    if not files:
        print(f"no .pt under {a.ckpt_dir}"); return 1

    print(f"{len(files)} checkpoints under {a.ckpt_dir}\n")
    plan = []
    for f in files:
        p = os.path.join(a.ckpt_dir, f)
        name = f[:-3]
        real = os.path.realpath(p)
        size = os.path.getsize(real) / 1e6
        plan.append((name, p, real, size))
        link = "  (symlink -> %s)" % os.path.basename(real) if os.path.islink(p) else ""
        print(f"  {name:<40} {size:8.1f} MB{link}")
    total = sum(s for *_, s in plan)
    print(f"\n  total {total:.0f} MB")

    if a.plan:
        print(f"\nwould upload to {a.repo} ({'public' if a.public else 'PRIVATE'})")
        print("symlinks are resolved, so real bytes are uploaded, not links.")
        return 0

    out = a.out or "/tmp/hf_cards"
    os.makedirs(out, exist_ok=True)
    print()
    for name, p, real, size in plan:
        info = describe(real)
        digest = sha256(real)
        txt = card(name, real, info, digest)
        with open(os.path.join(out, f"{name}.md"), "w") as fh:
            fh.write(txt)
        print(f"  card: {name}  J={info.get('cfg.J', info.get('head_width', '?'))} "
              f"step={info.get('step')} sha={digest[:12]}")
    print(f"\ncards written to {out}")

    if a.cards_only:
        return 0
    if not (a.upload or a.update_cards):
        print("\n(no --upload given; nothing was sent)")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("\npip install huggingface_hub", file=sys.stderr); return 1
    api = HfApi()
    api.create_repo(a.repo, repo_type="model", private=not a.public, exist_ok=True)
    print(f"\nuploading to {a.repo} ({'public' if a.public else 'private'})")
    for name, p, real, size in plan:
        if not a.update_cards:
            api.upload_file(path_or_fileobj=real, path_in_repo=f"{name}.pt",
                            repo_id=a.repo)
        api.upload_file(path_or_fileobj=os.path.join(out, f"{name}.md"),
                        path_in_repo=f"cards/{name}.md", repo_id=a.repo)
        print(f"  sent {name}" + ("" if a.update_cards else f" ({size:.0f} MB)"))
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
