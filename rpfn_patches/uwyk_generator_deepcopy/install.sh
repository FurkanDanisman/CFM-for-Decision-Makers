#!/usr/bin/env bash
# Stop UWYK's DataLoader workers hard-aborting on deepcopy(scm).
#
# THE BUG (PyTorch, confirmed 2.6.0+cu124). copy.deepcopy of a torch.Generator
# over-decrements the refcount of the None singleton by exactly 1 per call.
#
# WHERE UWYK HITS IT. InterventionalDataset._get_item_internal does
#     org_scm = deepcopy(scm)          # line ~635
# whenever return_scm / return_adjacency_matrix / return_ancestor_matrix is set
# -- and the released config sets return_ancestor_matrix: true, so this runs on
# EVERY task. Each sampled SCM holds one torch.Generator per mechanism (tens of
# them), leaking ~30-40 None-refs per task. After a few thousand tasks a worker
# drives None's refcount to zero and the process aborts with
#     Fatal Python error: none_dealloc
# which surfaces as dead workers, "Core dump ... failed" in dmesg, and a main
# process blocked forever on an empty prefetch queue.
#
# THE FIX, already proven in this repo. training/data/PairedInterventionalDataset.py
# hit exactly this (its comment: "killing training ~step 500") and solved it by
# registering a state-based clone in copy._deepcopy_dispatch. That table is
# consulted BEFORE any __deepcopy__ hook and works on torch.Generator, a C type
# that cannot take a monkeypatched __deepcopy__ -- so one registration fixes
# every Generator anywhere in the SCM object graph, in every process that
# imports the module. A fresh Generator seeded via set_state(get_state()) is
# verified leak-free.
#
# graph2d trains the same model through PairedInterventionalDataset with 8
# workers and does not exhibit this, which is what identified the cause.
#
# Idempotent; backs up to InterventionalDataset.py.beforegenfix.
set -euo pipefail

UWYK_ROOT="${UWYK_ROOT:?UWYK_ROOT must point at the Graphs4CausalFoundationModels checkout}"
DST="$UWYK_ROOT/src/priordata_processing/Datasets/InterventionalDataset.py"
[ -f "$DST" ] || { echo "FATAL: not found: $DST" >&2; exit 1; }

if grep -q "GENERATOR DEEPCOPY FIX" "$DST"; then
    echo "[install] already patched: $DST"; exit 0
fi

cp -n "$DST" "$DST.beforegenfix" 2>/dev/null || true
echo "[install] backed up → $DST.beforegenfix"

python3 - "$DST" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()

OLD = "from copy import deepcopy\n"
NEW = '''from copy import deepcopy

# ── GENERATOR DEEPCOPY FIX ───────────────────────────────────────────────────
# copy.deepcopy of a torch.Generator over-decrements the None singleton's
# refcount by 1 per call (PyTorch, confirmed 2.6.0+cu124). _get_item_internal
# deepcopies the SCM on every task when return_ancestor_matrix is set, and each
# SCM holds one Generator per mechanism, so a worker leaks ~30-40 None-refs per
# task and aborts with "Fatal Python error: none_dealloc" after a few thousand.
# The visible symptoms are dead workers, "Core dump ... failed" in dmesg, and a
# main process blocked forever on an empty prefetch queue.
#
# copy._deepcopy_dispatch is consulted before any __deepcopy__ hook and works on
# torch.Generator (a C type that cannot take a monkeypatched __deepcopy__), so
# one registration covers every Generator in the SCM object graph. Same fix as
# training/data/PairedInterventionalDataset.py in the R-PFN repo, which is why
# graph2d trains through that dataset with 8 workers without aborting.
import copy as _rpfn_copy


def _rpfn_deepcopy_generator(g, memo):
    import torch as _t
    ng = _t.Generator(device=g.device)
    ng.set_state(g.get_state())
    memo[id(g)] = ng
    return ng


def _rpfn_register_generator_deepcopy():
    import torch as _t
    if _t.Generator not in _rpfn_copy._deepcopy_dispatch:
        _rpfn_copy._deepcopy_dispatch[_t.Generator] = _rpfn_deepcopy_generator


_rpfn_register_generator_deepcopy()
# ─────────────────────────────────────────────────────────────────────────────
'''
assert OLD in s, "deepcopy import not found — upstream changed?"
s = s.replace(OLD, NEW, 1)
open(p, "w").write(s)
print("[install] patched InterventionalDataset: torch.Generator deepcopy is now leak-free")
PYEOF

python3 -c "import sys; compile(open(sys.argv[1]).read(), sys.argv[1], 'exec'); print('[install] syntax OK')" "$DST"

# prove the leak is gone, the same way the original fix was verified
python3 - <<'PYEOF'
import sys
try:
    import torch, copy, gc
except ImportError:
    print("[install] (torch unavailable here — skipping runtime leak check)"); sys.exit(0)

def none_refs():
    gc.collect(); return sys.getrefcount(None)

def clone(g, memo):
    ng = torch.Generator(device=g.device); ng.set_state(g.get_state()); memo[id(g)] = ng; return ng

g = torch.Generator(); g.manual_seed(0)
before = none_refs()
for _ in range(200): copy.deepcopy(g)
leak_unpatched = none_refs() - before

copy._deepcopy_dispatch[torch.Generator] = clone
before = none_refs()
for _ in range(200): copy.deepcopy(g)
leak_patched = none_refs() - before

print(f"[install] None-refcount delta over 200 deepcopies: "
      f"unpatched {leak_unpatched:+d}, patched {leak_patched:+d}")
assert leak_patched == 0, "patched clone still leaks None references"
print("[install] verified: patched path leaks nothing")
PYEOF
echo "[install] done."
