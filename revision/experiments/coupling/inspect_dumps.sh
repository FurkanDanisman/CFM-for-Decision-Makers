#!/bin/bash
# E3 prep: record the format of one dump per (model, benchmark) and push it (revision/results/d262t1/e3_dump_keys.txt).
set -uo pipefail
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"; source "$KIT/venv/bin/activate"
export CS_DATA="${CS_DATA:-$KIT/case_study_data/d_variation}"
python "$REPO/revision/experiments/coupling/inspect_dumps.py" "$REPO/revision/results/d262t1/e3_dump_keys.txt" || exit 1
cd "$REPO" && git add revision/results/d262t1/e3_dump_keys.txt && git commit -q -m "d262t1: dump format report for E3" \
  && git pull -q --rebase && git push -q && echo pushed
