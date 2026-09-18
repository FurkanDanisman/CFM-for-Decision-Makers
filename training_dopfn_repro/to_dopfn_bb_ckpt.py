"""Convert a training_dopfn_repro joint_2d checkpoint into the layout
benchmarks/l2_ihdp/eval_dopfn_bb_raw.py expects, so the existing DoPFN-bb
RealCause eval can score it unchanged.

The two models are the same: same PerFeatureTransformer backbone from
artifacts/dopfn_model.pkl, same head (Linear(d,4d) -> GELU -> Linear(4d, J^2+13)),
both installed at backbone.decoder_dict['standard'], and the same input
convention (treatment in column 0 of X, zeroed for queries). Only the module
nesting differs, so only the state-dict key names differ:

    repro   :  <transformer keys>           decoder_dict.standard.{0,2}.{weight,bias}
    bb      :  backbone.<transformer keys>  backbone.decoder_dict.standard.*
                                            head_2d.*   (same tensors, second path)

and the checkpoint envelope:

    repro   :  {"step","model","optimizer","scheduler","scaler","edges","provenance"}
    bb eval :  ckpt['model_state_dict'], ckpt['config']['J'], ckpt['edges']

    python training_dopfn_repro/to_dopfn_bb_ckpt.py \
        --in  checkpoints_dopfn_repro/joint_2d/step_150000_final.pt \
        --out Required_checkpoints/dopfn_repro_joint2d_step150000_bb.pt
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def infer_j(sd, provenance):
    """J from provenance if recorded, else from the head's output width.

    Not defaulted: the eval reshapes the logits to (J, J), so a wrong J still
    "works" and silently scores a different grid.
    """
    for key in ("j_2d", "J"):
        if isinstance(provenance, dict) and key in provenance:
            return int(provenance[key])
        if isinstance(provenance, dict):
            for sub in provenance.values():
                if isinstance(sub, dict) and key in sub:
                    return int(sub[key])
    out = [v.shape[0] for k, v in sd.items()
           if k.endswith("decoder_dict.standard.2.bias")]
    if len(out) != 1:
        raise SystemExit(f"cannot locate head bias to infer J (found {len(out)})")
    n_out = int(out[0])
    j = int(round(math.sqrt(n_out - 13)))
    if j * j + 13 != n_out:
        raise SystemExit(f"head width {n_out} is not J^2+13 for integer J")
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--j", type=int, default=None, help="override inferred J")
    a = ap.parse_args()

    blob = torch.load(a.src, map_location="cpu", weights_only=False)
    if "model" not in blob:
        raise SystemExit(f"{a.src} has no 'model' key — is this a dopfn_repro checkpoint?")
    sd = blob["model"]
    prov = blob.get("provenance", {})

    variant = prov.get("variant") if isinstance(prov, dict) else None
    if variant not in (None, "joint_2d"):
        raise SystemExit(f"variant is {variant!r}; this converter is for joint_2d")

    J = a.j or infer_j(sd, prov)
    n_out_expected = J * J + 13

    # backbone.* for every key, plus head_2d.* for the decoder (bb registers the
    # same module twice, so both paths must be present and identical).
    out_sd = {}
    head_keys = 0
    for k, v in sd.items():
        out_sd["backbone." + k] = v
        if ".decoder_dict.standard." in "." + k:
            out_sd["head_2d." + k.split("decoder_dict.standard.", 1)[1]] = v
            head_keys += 1
    if head_keys != 4:                       # Linear,GELU,Linear -> 2 w + 2 b
        raise SystemExit(f"expected 4 head tensors, remapped {head_keys}")

    bias = out_sd.get("head_2d.2.bias")
    if bias is None or int(bias.shape[0]) != n_out_expected:
        raise SystemExit(
            f"head output width {None if bias is None else bias.shape[0]} "
            f"!= J^2+13 = {n_out_expected} for J={J}")

    edges = blob.get("edges")
    if edges is None:
        raise SystemExit("checkpoint has no 'edges'; the eval needs the 2D grid")

    torch.save(
        {
            "model_state_dict": out_sd,
            "config": {"J": J},
            "edges": edges,
            "step": int(blob.get("step", -1)),
            "converted_from": os.path.abspath(a.src),
            "provenance": prov,
        },
        a.dst,
    )
    print(f"J={J}  head_out={n_out_expected}  step={blob.get('step')}  "
          f"edges=[{float(edges[0]):+.2f}, {float(edges[-1]):+.2f}]")
    print(f"wrote {a.dst}")


if __name__ == "__main__":
    main()
