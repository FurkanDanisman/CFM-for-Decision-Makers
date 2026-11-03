#!/bin/bash
# Upload the paper's checkpoints ($KIT/final_checkpoints_paper, made by gather_checkpoints.sh) to the public
# Hugging Face model repo, with a model card listing every file, the paper model it belongs to, and its sha256.
#   source venv/bin/activate && pip install -U huggingface_hub && hf auth login     (once; a token with write access)
#   bash R-PFN/revision/cleanup/upload_hf.sh
# Default HF_REPO: furkanbd/r-pfn-checkpoints (public). Re-running uploads only what changed.
set -euo pipefail
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"; SRC="$KIT/final_checkpoints_paper"
HF_REPO="${HF_REPO:-furkanbd/r-pfn-checkpoints}"
[ -f "$SRC/MANIFEST.txt" ] || { echo "FATAL: run gather_checkpoints.sh first"; exit 1; }
cd "$KIT"; [ -f venv/bin/activate ] && source venv/bin/activate
python -c "import huggingface_hub" 2>/dev/null || pip install -q huggingface_hub
python - "$SRC" "$HF_REPO" <<'PY'
import os, sys
from huggingface_hub import HfApi
src, repo = sys.argv[1], sys.argv[2]
rows = [l.rstrip("\n") for l in open(os.path.join(src, "MANIFEST.txt")) if l.strip()]
table = ["| File | Paper model | Size (MB) | sha256 |", "|---|---|---|---|"]
for r in rows:
    left, _, _ = r.partition(" | ")
    sha, size, name, *model = left.split()
    table.append(f"| `{name}` | {' '.join(model)} | {int(size) / 1e6:.0f} | `{sha[:16]}…` |")
card = f"""---
license: other
tags: [causal-inference, prior-data-fitted-networks, tabular]
---
# Checkpoints: Joint Learning of Factual and Counter-factual Worlds for Causal PFNs

Every checkpoint used for the results of the paper. The training and evaluation code, and how each model was
trained and evaluated, are in the GitHub repository. Published models we evaluate but did not train (Do-PFN,
UWYK, the two-stage CausalPFN) are included only where the paper loads a specific file; see their original
releases for licences.

{chr(10).join(table)}

Full sha256 sums and the original cluster paths are in `MANIFEST.txt`.
"""
open(os.path.join(src, "README.md"), "w").write(card)
api = HfApi()
api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
api.upload_folder(folder_path=src, repo_id=repo, repo_type="model", commit_message="paper checkpoints")
print("uploaded to https://huggingface.co/" + repo)
PY
