#!/bin/bash
#SBATCH --job-name=uwyk-abl
#SBATCH --output=logs_uwyk_ablation/train_%j.out
#SBATCH --error=logs_uwyk_ablation/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --requeue

# UWYK ablations on THEIR code, unmodified.
#
# Starts from the config that produced the released full-conditioning
# checkpoint (experiments/checkpoints/full_conditioned_model/config.yaml),
# applies only the requested keys, and VERIFIES SEMANTICALLY -- by diffing the
# parsed YAML, not the text -- that exactly those leaf paths changed. Anything
# else differing is a hard abort. That is what makes "only X changed" a checked
# claim instead of an assumption.
#
#   A (baseline)   BINARIZE=1.0
#   B (bins)       BINARIZE=1.0 NUM_BARS=32
#   C (both arms)  BINARIZE=1.0 BOTH_ARMS=1     (also needs the code patch:
#                  rpfn_patches/uwyk_both_arms/install.sh)
#
# NOTE binarize_treatment_prob is absent from their config, so it defaults to
# 0.0 -- the released model NEVER binarizes the treatment. Setting it to 1.0 is
# a genuine change to the prior, so A is a new baseline, not a reproduction of
# the published checkpoint. Compare A/B/C to each other, not to that.
#
# Usage:
#   UWYK_ROOT=$DEPLOY_ROOT/external/g4cfm BINARIZE=1.0 NUM_BARS=32 \
#     sbatch --account=aip-rgrosse --gres=gpu:h100:4 \
#            training_uwyk_variants/submit_train_uwyk_ablation.sbatch
#
# Env:
#   UWYK_ROOT    (required) Graphs4CausalFoundationModels checkout
#   NUM_BARS     BarDistribution bin count       (unset = leave at 1000)
#   BINARIZE     binarize_treatment_prob         (unset = leave at default 0.0)
#   BOTH_ARMS    1 = enable the both-arms patch  (unset = upstream behaviour)
#   TAG          suffix for config/experiment names
#   VENV         venv to activate                (default $UWYK_ROOT/venv)
#   NUM_WORKERS  dataloader workers (default: leave their 50)
#                Each spawned worker initialises its own CUDA context at
#                ~518 MiB, so 50 of them claim ~23 GB on top of the ~53 GB the
#                model needs and an 80 GB H100 OOMs in the first forward:
#                  "Tried to allocate 412.00 MiB ... 295.19 MiB is free
#                   ... Process NNN has 518.00 MiB memory in use" x45
#                This only became reachable once the pickle shim let workers
#                spawn at all -- before that everything ran in one process.
#                16 keeps loading parallel while leaving the GPU ~8 GB of
#                worker contexts.
set -euo pipefail
mkdir -p logs_uwyk_ablation

UWYK_ROOT="${UWYK_ROOT:?UWYK_ROOT required}"
VENV="${VENV:-$UWYK_ROOT/venv}"
SRC_CFG="$UWYK_ROOT/experiments/checkpoints/full_conditioned_model/config.yaml"
TAG="${TAG:-bars${NUM_BARS:-1000}_bin${BINARIZE:-default}_both${BOTH_ARMS:-0}}"

[ -f "$SRC_CFG" ] || { echo "FATAL: released config not found: $SRC_CFG" >&2; exit 1; }
[ -f "$VENV/bin/activate" ] || { echo "FATAL: venv not found: $VENV" >&2; exit 1; }
source "$VENV/bin/activate"

OUT_CFG="$UWYK_ROOT/configs_ablation/config_${TAG}.yaml"
mkdir -p "$(dirname "$OUT_CFG")"

NUM_BARS="${NUM_BARS:-}" BINARIZE="${BINARIZE:-}" NUM_WORKERS="${NUM_WORKERS:-}" \
python3 - "$SRC_CFG" "$OUT_CFG" <<'PYEOF'
import os, re, sys, yaml

src, dst = sys.argv[1], sys.argv[2]
num_bars = os.environ.get('NUM_BARS') or None
binarize = os.environ.get('BINARIZE') or None
workers  = os.environ.get('NUM_WORKERS') or None

lines = open(src).read().splitlines(keepends=True)
expected = set()

# --- num_bars: rewrite the `value:` line FOLLOWING the num_bars key. A blind
# --- s/value: 1000/ would also hit max_fit_items and max_fit_batches.
if num_bars:
    out, hits, armed = [], 0, False
    for ln in lines:
        if armed and re.match(r'^\s*value:', ln):
            out.append(re.sub(r'(^\s*value:\s*)\S+', rf'\g<1>{num_bars}', ln)); hits += 1; armed = False
            continue
        if re.match(r'^\s*num_bars:', ln):
            armed = True
        out.append(ln)
    assert hits == 1, f'num_bars: expected 1 value line, rewrote {hits}'
    lines = out
    expected.add('model_config.num_bars.value')

# --- binarize_treatment_prob: absent upstream, so APPEND under dataset_config.
if binarize:
    txt = ''.join(lines)
    assert 'binarize_treatment_prob' not in txt, 'key already present — upstream changed'
    m = re.search(r'^dataset_config:\s*$', txt, re.M)
    assert m, 'dataset_config section not found'
    nxt = re.search(r'^\S', txt[m.end():], re.M)
    ins = m.end() + (nxt.start() if nxt else 0)
    block = (f'\n  # ABLATION: always binarize the treatment (upstream default 0.0)\n'
             f'  binarize_treatment_prob:\n    value: {binarize}\n')
    lines = (txt[:ins] + block + txt[ins:]).splitlines(keepends=True)
    expected.add('dataset_config.binarize_treatment_prob.value')

# --- num_workers: present upstream (50); rewrite the value line after the key.
if workers:
    out, hits, armed = [], 0, False
    for ln in lines:
        if armed and re.match(r'^\s*value:', ln):
            out.append(re.sub(r'(^\s*value:\s*)\S+', rf'\g<1>{workers}', ln)); hits += 1; armed = False
            continue
        if re.match(r'^\s*num_workers:', ln):
            armed = True
        out.append(ln)
    assert hits == 1, f'num_workers: expected 1 value line, rewrote {hits}'
    lines = out
    expected.add('training_config.num_workers.value')

open(dst, 'w').writelines(lines)

# --- semantic verification: parsed-YAML leaf diff must equal `expected` ------
def flat(d, p=''):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flat(v, f'{p}.{k}' if p else str(k))
    else:
        yield p, d

a = dict(flat(yaml.safe_load(open(src))))
b = dict(flat(yaml.safe_load(open(dst))))
changed = {k for k in set(a) | set(b) if a.get(k, '<absent>') != b.get(k, '<absent>')}
print(f'[config] intended changes : {sorted(expected) or "(none)"}')
print(f'[config] actual  changes  : {sorted(changed) or "(none)"}')
for k in sorted(changed):
    print(f'           {k}: {a.get(k, "<absent>")} -> {b.get(k, "<absent>")}')
assert changed == expected, f'UNEXPECTED CONFIG DRIFT: {sorted(changed ^ expected)}'
print('[config] verified: exactly the intended keys changed')
PYEOF

export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
# Read by the both-arms patch (rpfn_patches/uwyk_both_arms). Unset => upstream.
export UWYK_BOTH_ARMS="${BOTH_ARMS:-0}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] UWYK ablation TAG=$TAG"
echo "  NUM_BARS=${NUM_BARS:-<released 1000>}  BINARIZE=${BINARIZE:-<default 0.0>}  BOTH_ARMS=$UWYK_BOTH_ARMS  NUM_WORKERS=${NUM_WORKERS:-<released 50>}"
echo "  CONFIG=$OUT_CFG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# checkpoint_dir / save_dir in their config are relative, so cwd decides output.
cd "$UWYK_ROOT"
python -u src/training/run.py --config "$OUT_CFG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
