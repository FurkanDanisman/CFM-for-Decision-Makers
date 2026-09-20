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
#   BATCH_SIZE   micro-batch     (default: leave their 8)
#   ACCUM        grad accumulation (default: leave their 4)
#                EFFECTIVE BATCH IS batch_size x accumulate_grad_batches = 32.
#                Peak activation memory is driven by the MICRO-batch, so
#                BATCH_SIZE=4 ACCUM=8 halves peak memory while training on an
#                identical effective batch -- the run stays comparable.
#                Motivation: backward() hangs with the GPU at 100% and ~65 GB
#                resident while the dataloader workers sit idle in queue.get.
#                Task size is random (n_test_samples_per_dataset ~ U{1,500},
#                train up to 1000), so a large task at high occupancy drives the
#                allocator into repeated free/realloc.
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

# Their checkpoint_dir is relative (./checkpoints) and the run subdirectory
# name comes from the config's experiment name, NOT from TAG -- so every
# ablation writes to the same path and whichever finishes second destroys
# the first one's checkpoints. Give each TAG its own absolute root.
CKPT_DIR="${CKPT_DIR:-$UWYK_ROOT/checkpoints_ablation/$TAG}"
mkdir -p "$CKPT_DIR"

NUM_BARS="${NUM_BARS:-}" BINARIZE="${BINARIZE:-}" NUM_WORKERS="${NUM_WORKERS:-}" \
BATCH_SIZE="${BATCH_SIZE:-}" ACCUM="${ACCUM:-}" CKPT_DIR="$CKPT_DIR" \
python3 - "$SRC_CFG" "$OUT_CFG" <<'PYEOF'
import os, re, sys, yaml

src, dst = sys.argv[1], sys.argv[2]
num_bars = os.environ.get('NUM_BARS') or None
binarize = os.environ.get('BINARIZE') or None
workers  = os.environ.get('NUM_WORKERS') or None
bsz      = os.environ.get('BATCH_SIZE') or None
accum    = os.environ.get('ACCUM') or None
ckpt_dir = os.environ.get('CKPT_DIR') or None

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

# --- batch_size / accumulate_grad_batches: rewrite the value line after each key
for _key, _val, _path in (('batch_size', bsz, 'training_config.batch_size.value'),
                          ('accumulate_grad_batches', accum,
                           'training_config.accumulate_grad_batches.value')):
    if not _val:
        continue
    out, hits, armed = [], 0, False
    for ln in lines:
        if armed and re.match(r'^\s*value:', ln):
            out.append(re.sub(r'(^\s*value:\s*)\S+', rf'\g<1>{_val}', ln)); hits += 1; armed = False
            continue
        if re.match(rf'^\s*{_key}:', ln):
            armed = True
        out.append(ln)
    assert hits == 1, f'{_key}: expected 1 value line, rewrote {hits}'
    lines = out
    expected.add(_path)

# --- checkpoint_dir: per-TAG absolute root so ablations cannot overwrite each
# --- other. Same rewrite shape as num_bars: the `value:` line after the key.
if ckpt_dir:
    out, hits, armed = [], 0, False
    for ln in lines:
        if armed and re.match(r'^\s*value:', ln):
            out.append(re.sub(r'(^\s*value:\s*)\S+', rf'\g<1>{ckpt_dir}', ln)); hits += 1; armed = False
            continue
        if re.match(r'^\s*checkpoint_dir:', ln):
            armed = True
        out.append(ln)
    assert hits == 1, f'checkpoint_dir: expected 1 value line, rewrote {hits}'
    lines = out

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
if ckpt_dir:
    _hit = [k for k in b if k.endswith('checkpoint_dir.value')]
    assert len(_hit) == 1, f'checkpoint_dir: {len(_hit)} matching keys, expected 1'
    expected.add(_hit[0])
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

# ── THREAD OVERSUBSCRIPTION ──────────────────────────────────────────────────
# ONE thread per process, not one per core. Each DataLoader worker inherits
# these, and the SCM prior runs numpy / XGBoost / sklearn inside the workers --
# so OMP_NUM_THREADS=32 with 24 workers asks for 768 threads on 32 cores. The
# symptom is a step whose reported compute time is ~1.8 s while wall-clock per
# step is ~30 s: the GPU waits while the workers fight each other for cores.
# Oversubscription gets WORSE as workers are added, which disguises itself as
# "data generation is slow".
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
# Read by the both-arms patch (rpfn_patches/uwyk_both_arms). Unset => upstream.
export UWYK_BOTH_ARMS="${BOTH_ARMS:-0}"
# Read by rpfn_patches/uwyk_log_every. Upstream logs steps 1..10 then nothing
# until step 5000, and save_every is 5000 too, so a healthy run is
# indistinguishable from a hung one for ~2.5 h. 50 makes progress visible.
export UWYK_LOG_EVERY="${UWYK_LOG_EVERY:-50}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] UWYK ablation TAG=$TAG"
echo "  NUM_BARS=${NUM_BARS:-<released 1000>}  BINARIZE=${BINARIZE:-<default 0.0>}  BOTH_ARMS=$UWYK_BOTH_ARMS  NUM_WORKERS=${NUM_WORKERS:-<released 50>}  LOG_EVERY=$UWYK_LOG_EVERY  BATCH=${BATCH_SIZE:-<8>}x${ACCUM:-<4>}"
echo "  CONFIG=$OUT_CFG"
echo "  CKPT_DIR=$CKPT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# checkpoint_dir / save_dir in their config are relative, so cwd decides output.
cd "$UWYK_ROOT"

# ── RESUME ───────────────────────────────────────────────────────────────────
# Requires rpfn_patches/uwyk_resume. Upstream cannot continue a run: load_model
# never restored global_step, so a reloaded checkpoint trained a further
# max_steps instead of finishing the original run. With the patch, a job that
# hits the walltime cap requeues and carries on from its last checkpoint --
# which is what makes a ~23h/50k run safe on a cluster with a 24h cap.
#
# Pick the checkpoint with the HIGHEST recorded step, not the newest mtime:
# interrupted.pt is written last on SIGTERM but may predate the final periodic
# step_N.pt if the signal arrived just after one was saved.
RESUME_ARGS=()
if [ "${RESUME:-1}" = "1" ]; then
    BEST_CKPT="$(python3 - "$CKPT_DIR" <<'PYRESUME'
import glob, os, sys, torch
best, best_step = None, -1
for f in glob.glob(os.path.join(sys.argv[1], '**', '*.pt'), recursive=True):
    try:
        md = torch.load(f, map_location='cpu', weights_only=False).get('metadata', {})
        step = int(md.get('step', md.get('global_step', -1)))
    except Exception:
        continue
    if step > best_step:
        best, best_step = f, step
if best:
    print(best)
PYRESUME
)" || BEST_CKPT=""
    if [ -n "$BEST_CKPT" ]; then
        RESUME_ARGS+=(--resume "$BEST_CKPT")
        echo "[resume] continuing from $BEST_CKPT"
    else
        echo "[resume] no checkpoint in $CKPT_DIR — starting fresh"
    fi
fi

# Forward SIGTERM so their signal handler writes interrupted.pt, then requeue.
CHILD=""
_on_term() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SIGTERM — forwarding to child pid=$CHILD"
    [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null || true
    wait "$CHILD" 2>/dev/null || true
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap _on_term SIGTERM

python -u src/training/run.py --config "$OUT_CFG" "${RESUME_ARGS[@]}" &
CHILD=$!
wait "$CHILD"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
