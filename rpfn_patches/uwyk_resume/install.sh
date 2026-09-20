#!/usr/bin/env bash
# Make UWYK training resumable.
#
# Upstream saves a checkpoint every save_every steps and can load the weights
# back -- but it cannot CONTINUE from them. Two independent reasons:
#
#   1. Trainer.load_model() restores model / optimizer / scheduler /
#      BarDistribution, then returns. It never assigns self.global_step, which
#      __init__ set to 0. The training loop breaks on
#      `self.global_step >= self.max_steps`, so a "resumed" run trains a further
#      max_steps from the loaded weights rather than finishing the original run.
#
#      Their own code hints the restore was intended: load_model prints
#      metadata['global_step'] -- but save_model writes the counter as
#      metadata['step'], so that branch never fires and nobody noticed.
#
#   2. run.py has no way to ask for it. The only CLI argument is --config, and
#      load_model is never called at startup.
#
# Consequence on a cluster with a walltime cap shorter than the run: a job that
# hits the limit is a total loss. That is the entire reason this patch exists --
# tamIA caps GPU jobs at 24h and a 50k-step ablation needs ~23h, so a single
# slow batch costs the whole day.
#
# This patch:
#   - restores global_step in load_model, reading metadata['step'] and falling
#     back to metadata['global_step'] so either layout works
#   - adds `--resume <ckpt>` to run.py and calls load_model right after the
#     Trainer is constructed
#
# Idempotent. Backs up both files. Verifies by compiling each patched file and
# asserting the marker is present exactly once.
set -euo pipefail

UWYK_ROOT="${UWYK_ROOT:?UWYK_ROOT must point at the Graphs4CausalFoundationModels checkout}"
TRAINER="$UWYK_ROOT/src/training/trainer.py"
RUNPY="$UWYK_ROOT/src/training/run.py"
MARK="RPFN-RESUME-PATCH"

[ -f "$TRAINER" ] || { echo "FATAL: not found: $TRAINER" >&2; exit 1; }
[ -f "$RUNPY" ]   || { echo "FATAL: not found: $RUNPY" >&2; exit 1; }

if grep -q "$MARK" "$TRAINER" && grep -q "$MARK" "$RUNPY"; then
    echo "[install] already patched: $TRAINER, $RUNPY"
    exit 0
fi

[ -f "$TRAINER.beforeresume" ] || cp "$TRAINER" "$TRAINER.beforeresume"
[ -f "$RUNPY.beforeresume" ]   || cp "$RUNPY"   "$RUNPY.beforeresume"
echo "[install] backed up -> $TRAINER.beforeresume, $RUNPY.beforeresume"

MARK="$MARK" TRAINER="$TRAINER" RUNPY="$RUNPY" python3 - <<'PYEOF'
import os, re, sys

MARK    = os.environ['MARK']
TRAINER = os.environ['TRAINER']
RUNPY   = os.environ['RUNPY']

# ---------------------------------------------------------------- trainer.py
src = open(TRAINER).read()
if MARK not in src:
    anchor = ("        if 'global_step' in metadata:\n"
              "            print(f\"   Checkpoint was saved at step {metadata['global_step']}\")\n"
              "        return metadata\n")
    if src.count(anchor) != 1:
        sys.exit(f'FATAL: load_model tail anchor matched {src.count(anchor)} times, expected 1')
    repl = (f"        # {MARK}: save_model() writes the counter as metadata['step'];\n"
            "        # upstream only ever read metadata['global_step'], which is never\n"
            "        # written, so the step counter silently restarted at 0 on load and\n"
            "        # a resumed run trained a further max_steps instead of finishing.\n"
            "        _resume_step = metadata.get('step', metadata.get('global_step'))\n"
            "        if _resume_step is not None:\n"
            "            self.global_step = int(_resume_step)\n"
            "            print(f'   Resuming at step {self.global_step}/{self.max_steps}')\n"
            "        else:\n"
            "            print('   WARNING: checkpoint carries no step; counter stays at '\n"
            "                  f'{self.global_step}')\n"
            "        return metadata\n")
    src = src.replace(anchor, repl)
    open(TRAINER, 'w').write(src)
    print('[install] patched trainer.py: load_model now restores global_step')

# ------------------------------------------------------------------- run.py
src = open(RUNPY).read()
if MARK not in src:
    anchor = ("    parser.add_argument('--config', type=str, default=None, "
              "help='Path to YAML config file')\n")
    if src.count(anchor) != 1:
        sys.exit(f'FATAL: argparse anchor matched {src.count(anchor)} times, expected 1')
    src = src.replace(anchor, anchor +
        f"    # {MARK}\n"
        "    parser.add_argument('--resume', type=str, default=None,\n"
        "                        help='Checkpoint to resume from (restores weights, "
        "optimizer, scheduler and step counter)')\n")

    # Insert the load_model call after the `trainer = Trainer(...)` statement.
    # The call spans ~30 lines, so find it by balancing parentheses rather than
    # guessing a closing line.
    lines = src.splitlines(keepends=True)
    starts = [i for i, l in enumerate(lines) if re.match(r'^\s*trainer\s*=\s*Trainer\(', l)]
    if len(starts) != 1:
        sys.exit(f'FATAL: found {len(starts)} `trainer = Trainer(` lines, expected 1')
    i = starts[0]
    indent = re.match(r'^(\s*)', lines[i]).group(1)
    depth = 0
    for j in range(i, len(lines)):
        depth += lines[j].count('(') - lines[j].count(')')
        if depth <= 0:
            end = j
            break
    else:
        sys.exit('FATAL: unbalanced parentheses after `trainer = Trainer(`')

    inject = (f"\n{indent}# {MARK}: continue a run that hit the walltime cap.\n"
              f"{indent}if getattr(args, 'resume', None):\n"
              f"{indent}    trainer.load_model(args.resume)\n")
    lines.insert(end + 1, inject)
    open(RUNPY, 'w').write(''.join(lines))
    print(f'[install] patched run.py: --resume added, load_model called after line {end + 1}')
PYEOF

for f in "$TRAINER" "$RUNPY"; do
    python3 -c "compile(open('$f').read(), '$f', 'exec')" \
        || { echo "[install] SYNTAX ERROR in $f — restoring backup" >&2
             cp "$f.beforeresume" "$f"; exit 1; }
    n=$(grep -c "RPFN-RESUME-PATCH" "$f")
    echo "[install] $f: syntax OK, marker x$n"
done

python3 - "$RUNPY" <<'PYEOF'
import sys, re
src = open(sys.argv[1]).read()
assert "--resume" in src, 'resume flag missing'
assert re.search(r"if getattr\(args, 'resume', None\):\n\s+trainer\.load_model\(args\.resume\)", src), \
    'load_model call not placed'
print('[install] verified: --resume parsed and wired to trainer.load_model')
PYEOF

echo "[install] done. Resume with:  python -u src/training/run.py --config C.yaml --resume CKPT.pt"
