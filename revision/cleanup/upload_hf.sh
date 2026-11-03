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
PUBLISHED = ("uwyk_published_", "cpfn_published_")       # not re-hosted: linked to their original releases
for r in rows:
    left, _, _ = r.partition(" | ")
    sha, size, name, *model = left.split()
    if name.startswith(PUBLISHED):
        continue
    table.append(f"| `{name}` | {' '.join(model)} | {int(size) / 1e6:.0f} | `{sha[:16]}…` |")
card = f"""---
license: other
tags: [causal-inference, prior-data-fitted-networks, tabular]
---
# Checkpoints: Joint Learning of Factual and Counter-factual Worlds for Causal PFNs

Every checkpoint we trained for the results of the paper, including the fixed-SCM Do-PFN runs. The training and
evaluation code, and how each model was trained and evaluated, are in the GitHub repository.

{chr(10).join(table)}

## Published models (not re-hosted)

The paper also evaluates published checkpoints, which are available from their original releases:

- **Do-PFN** (Robertson et al., 2025): https://github.com/jr2021/Do-PFN, file
  `artifacts/model_submitit_0ccc_id_171b69db_epoch_-1.cpkt`.
- **UWYK** (Reuter et al., 2026): https://github.com/ArikReuter/Graphs4CausalFoundationModels, branch
  `reproduce-realcause-results` (commit `c27fba6`, via `git lfs pull`), file
  `experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0/best_model.pt`
  with its `best_model_config.yaml`.
- **CausalPFN** (Balazadeh et al., 2025), the published two-stage checkpoint: `causalpfn_v0.pt` from
  https://huggingface.co/vdblm/causalpfn (code: https://github.com/vdblm/CausalPFN).

Their sha256 sums, as used in the paper, are in `MANIFEST.txt`, so a downloaded copy can be checked against them.
Full sha256 sums and the original cluster paths of all files are in `MANIFEST.txt`.
"""
open(os.path.join(src, "README.md"), "w").write(card)
api = HfApi()
api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
api.upload_folder(folder_path=src, repo_id=repo, repo_type="model", commit_message="paper checkpoints",
                  ignore_patterns=[p + "*" for p in PUBLISHED], delete_patterns=[p + "*" for p in PUBLISHED])
print("uploaded to https://huggingface.co/" + repo)
PY
