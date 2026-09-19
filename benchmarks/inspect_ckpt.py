#!/usr/bin/env python
"""Report what a checkpoint actually IS, from its own bytes.

Filenames and directory names are not evidence. This prints the provenance
block, the training step, the bar-distribution bin count, and the head output
width, all read off the state dict, so a model can be identified without
trusting the path it was found at.

    python benchmarks/inspect_ckpt.py final_checkpoints/*.pt

Exit status is 1 if any file failed to load, so it can gate a smoke test.
"""
import argparse, os, sys


def describe(path):
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        return {"error": f"top level is {type(blob).__name__}, not a dict"}
    sd = blob.get("model_state_dict") or blob.get("model") or blob.get("state_dict")
    if sd is None:
        cand = [k for k, v in blob.items() if isinstance(v, dict) and any(
            hasattr(t, "shape") for t in v.values())]
        sd = blob[cand[0]] if cand else None
    info = {"step": blob.get("step"), "epoch": blob.get("epoch")}
    prov = blob.get("provenance") or {}
    if isinstance(prov, dict):
        for k in ("variant", "git_commit", "config_name", "run_name"):
            if k in prov:
                info[k] = prov[k]
    cfg = blob.get("config") or blob.get("cfg") or {}
    if isinstance(cfg, dict):
        # num_buckets / n_out is the J that matters: it is the only field that
        # tells 1D J=10 apart from J=100 without running the model.
        for k in ("num_buckets", "n_buckets", "max_num_classes", "num_classes"):
            if k in cfg:
                info[f"cfg.{k}"] = cfg[k]
    if sd is None:
        info["error"] = "no state dict found"
        return info
    info["n_tensors"] = len(sd)
    # Bar-distribution borders pin J exactly; the head's out_features pins what
    # the model emits (2J for two marginals, J*J for a joint, J+3 for DoPFN).
    for k, v in sd.items():
        lk = k.lower()
        if "borders" in lk and hasattr(v, "numel"):
            info.setdefault("borders_numel", []).append((k, int(v.numel())))
        if lk.endswith("weight") and hasattr(v, "shape") and len(v.shape) == 2:
            info["last_2d_weight"] = (k, tuple(v.shape))
    heads = [(k, tuple(v.shape)) for k, v in sd.items()
             if hasattr(v, "shape") and ("decoder" in k.lower() or "head" in k.lower())
             and k.lower().endswith("weight") and len(v.shape) == 2]
    if heads:
        info["head_out"] = heads[-1]
    # Derived J. `borders` gives J = numel - 1 for the 1D criterion; note that a
    # 2D head can carry an inherited 1D criterion it never uses, so a mismatch
    # between borders and the head width is not by itself a bug -- compare the
    # head width against a reference checkpoint of known J instead.
    if "borders_numel" in info:
        info["J_from_borders"] = [n - 1 for _, n in info["borders_numel"]]
    if heads:
        W = heads[-1][1][0]
        cand = []
        for J in range(2, 257):
            if J * J == W:          cand.append(f"{J} (JxJ joint)")
            if 2 * J == W:          cand.append(f"{J} (two marginals)")
            if J * J + 2 * J == W:  cand.append(f"{J} (joint+marginals)")
            if J + 3 == W:          cand.append(f"{J} (J+3, DoPFN)")
        info["head_width"] = W
        info["J_candidates"] = cand or "no simple J decomposition"
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    bad = 0
    for p in a.paths:
        print(f"\n=== {p}")
        if not os.path.exists(p):
            print("  MISSING"); bad += 1; continue
        print(f"  size {os.path.getsize(p)/1e6:.1f} MB")
        try:
            info = describe(p)
        except Exception as e:
            print(f"  LOAD FAILED: {type(e).__name__}: {e}"); bad += 1; continue
        if "error" in info:
            bad += 1
        for k, v in info.items():
            print(f"  {k}: {v}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
