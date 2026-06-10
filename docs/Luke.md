# Training & Bug Diagnosis Guide

A hand-off document for someone investigating a `Fatal Python error:
none_dealloc` crash that occurs during training of this repository's
2D paired-outcome Causal Foundation Model.

Read [`PROBLEM_TRAINING.md`](PROBLEM_TRAINING.md) first for the problem
statement and a list of patterns we have already tried and rejected.
This document tells you how to actually run the training, observe the
crash, and gather useful diagnostic data.

---

## 1. Repository layout (only the parts you need)

```
train_cfm.py                       Production training entry point
submit_smoke.sh                    200-step CPU/GPU smoke (5 min)
submit_scaledup.sh                 500-step UWYK-config smoke (30 min)
submit_train.sh                    Full 50K-step training, 3 × 20h chunks

models/InterventionalPFN.py        UWYK body verbatim + learned null T_intv
losses/BarDistribution2D.py        2D BarDistribution loss + 9-region structure
data/PairedInterventionalDataset.py   Streaming SCM dataset — THE BUG IS HERE

docs/PROBLEM_TRAINING.md           Problem log + forbidden patterns (READ FIRST)
docs/Luke.md                       This file
docs/CFM_2D_BarDistribution_Design.md   Full design of the 2D head
docs/prior_summary.md              SCM prior + proposal annotations
```

---

## 2. Setup

### 2.1 Clone the repo and the UWYK source

UWYK's `SCM`, `SCMSampler`, and `BinarizingMechanism` are imported by
our streaming dataset. The data-generation half of the codebase is
unrunnable without them.

```bash
# This repository
git clone https://github.com/FurkanDanisman/CFM-for-Decision-Makers.git
cd CFM-for-Decision-Makers

# UWYK source — required for the SCM library
git clone --depth 1 https://github.com/ArikReuter/Graphs4CausalFoundationModels.git $HOME/g4cfm
export UWYK_SRC=$HOME/g4cfm/src    # used by data/PairedInterventionalDataset.py
```

On clusters where compute nodes lack internet (e.g., Trillium), clone
both from a login node — compute nodes only see what the filesystem
already contains.

### 2.2 Python environment

Python 3.11 is what we tested with. CUDA 12+ for the GPU side.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy scipy matplotlib clarabel
```

On Trillium we used:
```bash
module load python/3.11 scipy-stack
pip install --no-index torch numpy scipy matplotlib clarabel
```
(`--no-index` is the Alliance-Canada convention to use Compute Canada's
prebuilt wheelhouse, which gives a custom build called
`torch-2.12.0+computecanada`.)

### 2.3 Sanity check

```bash
export UWYK_SRC=$HOME/g4cfm/src
python -c "
import sys; sys.path.insert(0, '.')
from models.InterventionalPFN import InterventionalPFN
from losses.BarDistribution2D import total_params, make_edges, neg_log_prob_2d
from data.PairedInterventionalDataset import PairedInterventionalDataset
import torch

print('torch       :', torch.__version__)
print('cuda        :', torch.cuda.is_available())
print('total_params:', total_params(100))
ds = PairedInterventionalDataset(seed_base=42)
item = ds._generate_one(0)
print('sample keys :', list(item.keys()))
print('Y_do0 shape :', tuple(item['Y_do0'].shape))
print('OK')
"
```

If this prints `OK` and no traceback, the dataset can generate one
sample. That is necessary but does not prove anything about the bug —
the bug only surfaces after thousands of samples.

---

## 3. The three SLURM-style jobs

All three exist as `submit_*.sh` files in the repo. Each is a standard
shell script with `#SBATCH` directives at the top. They share the same
training script (`train_cfm.py`) and the same `data/`, `models/`, and
`losses/` packages — only environment variables differ.

| Script | Steps | Wall-time | What it tests |
|---|---|---|---|
| `submit_smoke.sh` | 200 | 30 min limit (uses ~5) | Demo: CPU vs GPU, imports, single H100 working |
| `submit_scaledup.sh` | 500 | 30 min limit (uses ~28) | UWYK Appendix G config (`d_model=256, depth=8`), bf16, activation checkpointing, MICROBATCH=4, GRAD_ACCUM=8 |
| `submit_train.sh` | 50,000 | 20 h × 3 chained jobs (~45 h total) | Production run. Resumes from checkpoints. |

The 20-hour chunks chain via `--dependency=afterany` and resume
automatically (`RESUME=1`); the script catches SIGTERM at wall-clock
and writes an interrupt checkpoint before exiting.

### Submitting

On Trillium, GPU jobs MUST be submitted from the GPU login node
(`trig-login01`), not the CPU login node (`tri-login04`). On other
clusters, check your scheduler docs for similar restrictions.

```bash
# Single job
sbatch submit_scaledup.sh

# Chained training run (three 20-hour chunks)
FIRST=$(sbatch --parsable submit_train.sh)
for i in 2 3; do
    FIRST=$(sbatch --parsable --dependency=afterany:$FIRST submit_train.sh)
done
squeue --me
```

---

## 4. The healthy run — what success looks like

`submit_scaledup.sh` should complete in ~27 minutes with this exact
shape of output (verified once on Trillium under different starting
state, before the bug started appearing):

```
=== Node info ===
NVIDIA H100 80GB HBM3
=================
Device:        cuda
GPU:           NVIDIA H100 80GB HBM3
Precision:     bf16 autocast
J:             100  (output_dim = 10013)
Model:         d_model=256  depth=8  heads=8  hidden_mult=4
Training:      steps=500  microbatch=4  grad_accum=8
                effective_batch = 32
                N_context=1000  N_query=250

[stream] starting DataLoader (workers=8, microbatch=4)
[BarDistribution2D] Fitting edges from 16000 Y_obs values
Model parameters: 16,119,072

   step       loss          lr      wall
─────────────────────────────────────────
     1   X.XXXX     2.00e-06     7.XXs
    20   X.XXXX     4.00e-05    XX.Xs
   100   X.XXXX     9.73e-05   3XX.Xs
   ...
   500   X.XXXX     1.00e-05  16XX.Xs

Total wall: 1620–1700s   (≈ 3.2 s/step, projects to 45 h for 50K steps)
```

Exit code `0:0`. `.err` file contains only the harmless
`gentoo/2023` module note (on Trillium). Per-step time ~3.2 s.

If your cluster reproduces those numbers, the model and the GPU side
are fine. The bug is in long-running runs only.

---

## 5. The bug

### 5.1 Symptom

```
Fatal Python error: none_dealloc: deallocating None: bug likely caused
by a refcount error in a C extension
```

Process aborts. SLURM reports `ExitCode=1:0` or `ExitCode=134:0` (134
= 128 + SIGABRT). The crash traceback points at
`data/PairedInterventionalDataset.py` inside `__getitem__` or
`_generate_one`.

### 5.2 When it appears

| Scenario | Samples generated before crash |
|---|---|
| Local laptop (macOS, mainline PyTorch, 1 process, sequential) | **20,000+ no crash** |
| Trillium compute (Linux, CC `torch-2.12.0+computecanada`, 1 process) | ~3,000 |
| Trillium compute (Linux, CC `torch-2.12.0+computecanada`, 8 forked workers) | ~16,000 |

Sample count = (training steps) × (effective batch). At
MICROBATCH=4 × GRAD_ACCUM=8, every training step generates 32 samples.

### 5.3 What this means

- **The Python code is correct.** It runs 20K+ iterations on macOS
  with the same configuration that crashes after 3K on Trillium.
- **The bug is in some C-extension on Trillium**, NOT in the
  algorithm.
- **`none_dealloc` is a refcount underflow on Python's `None`
  singleton.** This is always a bug in compiled C code — pure Python
  cannot cause it.

### 5.4 What we have NOT yet identified

We do not know **which** C extension. Candidates:
- PyTorch (CC's custom `torch-2.12.0+computecanada` build, which
  is a pre-release version not on PyPI)
- numpy
- One of the SCM library's mechanism implementations (XGBoost?
  Inside UWYK's `SampleMLPMechanism` or `SampleXGBoostMechanism`?)
- glibc malloc on the cluster's Linux distribution

Speculating without a real stack trace led us in circles. The current
plan (section 6) is to capture the trace.

---

## 6. Diagnostic procedure with `faulthandler`

Python's stdlib `faulthandler` module traps fatal signals (SIGSEGV,
SIGABRT) and dumps the full **C-level** stack — function names and
library names — before the process exits. This is what we need to
localize the bug.

`faulthandler.enable()` is already called at two places in this
repo (commit `9568bbe`):

- `train_cfm.py` top-of-file (covers the main process)
- `data/PairedInterventionalDataset._worker_init_fn` (covers each
  forked DataLoader worker)

### 6.1 Reproduce the crash

```bash
sbatch submit_scaledup.sh           # 500 steps, ~30 min on H100
squeue --me                          # wait for it to finish
```

### 6.2 If the run completes (no crash)

Move to `submit_train.sh` for 50K steps. The bug needs more samples
to surface; observe whether it appears.

### 6.3 If it crashes — read the `.err` file carefully

```bash
cat logs/scaled_<jobid>.err
```

Expected output now (with faulthandler) looks like:

```
Fatal Python error: none_dealloc: deallocating None: bug likely caused
by a refcount error in a C extension

Current thread 0x7f... (most recent call first):
  File "data/PairedInterventionalDataset.py", line N in __getitem__
  ...
  Native frames:
  #0  some_function_name (lib_name.so + 0xXXXXX)
  #1  another_function   (lib_name.so + 0xXXXXX)
  ...
```

The **Native frames** section is the key. It will name the library
(`.so` file) where the refcount bug fired.

| Native frame points at | Investigation direction |
|---|---|
| `libtorch_python.so` / `libtorch.so` | PyTorch internal — try a mainline (PyPI) torch build to see if the bug is specific to CC's `+computecanada` build |
| `_multiarray_umath.cpython*.so` / `libnpymath*.so` | numpy — try mainline numpy from PyPI |
| `xgboost*.so` | XGBoost mechanism path — see UWYK's `SampleXGBoostMechanism` |
| `libpython*.so` directly | Likely an issue in cffi / ctypes binding inside a mechanism, or a thread issue |
| `libc.so.6` (malloc/free) | Heap corruption, very generic — would need valgrind / asan-style tooling |

### 6.4 What to send back

If you cannot pin down the cause yourself, please send back:
1. The full `.err` file (the native frames are the part we need).
2. The exact `pip freeze` output of the venv (`pip freeze > pip_freeze.txt`).
3. `nvidia-smi` from the compute node where the crash happened.
4. The Linux distribution + libc version (`cat /etc/os-release`, `ldd --version`).

That is enough for us to triangulate from another machine.

---

## 7. Things explicitly ruled out — do not retry these

These are dead ends from prior sessions. They either don't fix the bug,
trade it for a worse problem, or violate the architectural commitments
of the project.

1. **Pre-generating a corpus of SCMs to disk** ("disk corpus"). The
   model needs to train on streaming SCMs, not a finite cached set.
2. **Switching to UWYK's 1D BarDistribution head.** Abandons the 2D
   paired-outcome design that is the entire point.
3. **`STREAM_WORKERS=0` as a normal config.** ~31 s/step instead of
   ~3 s/step. Acceptable as a one-shot diagnostic, never production.
4. **Try-except around `_generate_one`.** A fatal C error abort()s
   the process; Python try/except cannot catch it.
5. **Swapping libraries on speculation.** Library swaps must be
   justified by a native frame pointing at the library in question.

See [`PROBLEM_TRAINING.md`](PROBLEM_TRAINING.md) for the full forbidden-
patterns list and the verification standard for proposed fixes.

---

## 8. Quick reference

### Train (full run)

```bash
ssh <GPU_login_node>
cd <repo>
# Optional: warm-start by submitting submit_smoke.sh first
FIRST=$(sbatch --parsable submit_train.sh)
for i in 2 3; do
    FIRST=$(sbatch --parsable --dependency=afterany:$FIRST submit_train.sh)
done
squeue --me
```

### Monitor

```bash
squeue --me                                   # job status
tail -f logs/train_<jobid>.out                # live log
ssh <compute_node> nvidia-smi                 # GPU utilization (should be 90-100%)
ls -lt checkpoints/                           # latest saved checkpoint
```

### Inspect a crash

```bash
ls -lt logs/ | head -3
cat logs/<jobname>_<jobid>.err                # this is where faulthandler writes
cat logs/<jobname>_<jobid>.out | tail -40     # last training output
grep -E "^ *[0-9]+ +-?[0-9.]+" logs/<jobname>_<jobid>.out   # loss table
```

### Resume after crash or wall-clock

Just resubmit; the script auto-loads `checkpoints/step_*.pt` because
`RESUME=1`:

```bash
sbatch submit_train.sh
```

---

## 9. Contact

Owner: Furkan Danişman (`furkandanisman@gmail.com`)
Repo: https://github.com/FurkanDanisman/CFM-for-Decision-Makers
