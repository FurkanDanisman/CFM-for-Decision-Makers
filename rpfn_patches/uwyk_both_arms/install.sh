#!/usr/bin/env bash
# Patch UWYK's InterventionalDataset so each interventional draw yields BOTH
# treatment arms from the SAME noise, instead of one arm per independent draw.
#
# WHY. Upstream samples fresh noise and propagates once; the intervened node's
# value comes from a resampling distribution over its observational marginal:
#     scm.sample_exogenous(n); scm.sample_endogenous(n)
#     interv1_raw = scm.propagate(n)
# So two rows never share a noise draw, and the dataset contains no
# potential-outcome pairs. This is the UWYK analogue of the CausalPFN both-arms
# change, where each query point is emitted at t=0 and t=1.
#
# WHAT. Under UWYK_BOTH_ARMS=1 the noise is drawn ONCE for n//2 samples, then
# propagated twice with the intervened node forced to t0 and to t1. Every other
# exogenous and endogenous term is shared, so the two rows are a genuine
# potential-outcome pair. Row count is preserved (2 * (n//2)), so C differs
# from A only in the PAIRING, not in how much data the model sees.
#
# HOW IT WORKS. scm.intervene() cuts incoming edges and replaces the mechanism
# with InterventionMechanism (identity on noise), so after intervention the
# node's VALUE is its noise. _sample_fast reads scm._fixed_exogenous[v] and
# scm._fixed_endogenous[v] directly, so overwriting those cached tensors in
# place is exactly what propagate() consumes.
#
# t0/t1 come from BinarizingMechanism and must be captured BEFORE intervene()
# replaces the mechanism. Requires binarize_treatment_prob > 0 (the released
# config omits it, so it defaults to 0.0 and no arms exist); when a sample is
# not binarized, upstream behaviour is used for that sample.
#
# OPT-IN: without UWYK_BOTH_ARMS=1 this file is byte-equivalent in behaviour to
# upstream. Idempotent; backs up to InterventionalDataset.py.beforebotharms.
set -euo pipefail

UWYK_ROOT="${UWYK_ROOT:?UWYK_ROOT must point at the Graphs4CausalFoundationModels checkout}"
DST="$UWYK_ROOT/src/priordata_processing/Datasets/InterventionalDataset.py"
[ -f "$DST" ] || { echo "FATAL: not found: $DST" >&2; exit 1; }

if grep -q "BOTH-ARMS PATCH" "$DST"; then
    echo "[install] already patched: $DST"; exit 0
fi

cp -n "$DST" "$DST.beforebotharms" 2>/dev/null || true
echo "[install] backed up → $DST.beforebotharms"

python3 - "$DST" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p).read()

# ── 1. capture t0/t1 before intervene() discards the binarized mechanism ─────
OLD1 = "            scm.intervene(node = intervention_node) # intervene on the chosen node\n"
NEW1 = '''            # ── BOTH-ARMS PATCH ── capture the binarized treatment levels BEFORE
            # intervene() swaps the mechanism for InterventionMechanism. None
            # when this sample was not binarized; the pairing is skipped then.
            _ba_mech = scm.mechanisms.get(intervention_node, None)
            _ba_t0 = getattr(_ba_mech, "t0", None)
            _ba_t1 = getattr(_ba_mech, "t1", None)

            scm.intervene(node = intervention_node) # intervene on the chosen node
'''
assert OLD1 in s, "intervene() call not found — upstream changed?"
s = s.replace(OLD1, NEW1, 1)

# ── 2. paired propagation ────────────────────────────────────────────────────
OLD2 = """            # Sample new noise for interventional scenario
            scm.sample_exogenous(num_samples=number_test_samples)
            scm.sample_endogenous(num_samples=number_test_samples)

            interv1_raw = scm.propagate(num_samples=number_test_samples)  # interventional data (post-intervention)
"""
NEW2 = '''            # Sample new noise for interventional scenario
            if (_os.environ.get("UWYK_BOTH_ARMS", "0") == "1"
                    and _ba_t0 is not None and _ba_t1 is not None):
                # Draw the noise ONCE, then propagate twice with the intervened
                # node forced to t0 and to t1. Everything else is shared, so the
                # two rows are a potential-outcome pair rather than independent
                # interventional draws.
                #
                # Noise is drawn at the FULL number_test_samples, not n//2: the
                # noise distributions (and the downstream padding to
                # max_number_test_samples) are sized against that value, and
                # sampling fewer produces
                #   "expanded size of the tensor (1000) must match the existing
                #    size (500)".
                # We propagate twice at full size and then slice, so the row
                # count out of this block is exactly number_test_samples and
                # nothing downstream has to change.
                scm.sample_exogenous(num_samples=number_test_samples)
                scm.sample_endogenous(num_samples=number_test_samples)

                def _ba_force(_val):
                    # _sample_fast reads these dicts directly, so mutating the
                    # cached tensors in place is what propagate() consumes.
                    _hit = False
                    _fe = getattr(scm, "_fixed_exogenous", None)
                    if _fe is not None and intervention_node in _fe:
                        _fe[intervention_node].fill_(float(_val)); _hit = True
                    _fn = getattr(scm, "_fixed_endogenous", None)
                    if _fn is not None and intervention_node in _fn:
                        _fn[intervention_node].fill_(float(_val)); _hit = True
                    if not _hit:
                        raise RuntimeError(
                            "[uwyk-both-arms] intervened node %r absent from cached "
                            "noise — cannot force an arm" % (intervention_node,))

                _ba_force(_ba_t0)
                _ba_arm0 = {k: (v.clone() if torch.is_tensor(v) else v)
                            for k, v in scm.propagate(num_samples=number_test_samples).items()}
                _ba_force(_ba_t1)
                _ba_arm1 = {k: (v.clone() if torch.is_tensor(v) else v)
                            for k, v in scm.propagate(num_samples=number_test_samples).items()}

                # Row i of arm0 and row i of arm1 share a noise draw, so taking
                # the leading rows of each keeps the pairs intact. Total is
                # exactly number_test_samples -- unchanged from upstream.
                _ba_h = number_test_samples // 2
                _ba_r = number_test_samples - _ba_h
                interv1_raw = {k: torch.cat([_ba_arm0[k][:_ba_h], _ba_arm1[k][:_ba_r]], dim=0)
                               for k in _ba_arm0}
                global _BA_ANNOUNCED
                if not _BA_ANNOUNCED:
                    _BA_ANNOUNCED = True
                    print("[uwyk-both-arms] ACTIVE: %d rows = %d at t0=%.4g + %d at t1=%.4g "
                          "(rows i and i share a noise draw)"
                          % (number_test_samples, _ba_h, _ba_t0, _ba_r, _ba_t1), flush=True)
            else:
                scm.sample_exogenous(num_samples=number_test_samples)
                scm.sample_endogenous(num_samples=number_test_samples)

                interv1_raw = scm.propagate(num_samples=number_test_samples)  # interventional data (post-intervention)
'''
assert OLD2 in s, "sample/propagate block not found — upstream changed?"
s = s.replace(OLD2, NEW2, 1)

# ── 3. module-level imports / announce flag, before the first top-level import
if "import os as _os" not in s:
    # Anchor AFTER any __future__ imports: those must be the first statement in
    # the file, and ast.parse does NOT enforce that -- only compile() does.
    fut = list(re.finditer(r"^from __future__ import .*$", s, re.MULTILINE))
    if fut:
        at = s.index("\n", fut[-1].end()) + 1
    else:
        m = re.search(r"^(?:import |from )", s, re.MULTILINE)
        assert m, "no top-level import to anchor"
        at = m.start()
    s = s[:at] + "import os as _os\n_BA_ANNOUNCED = False\n" + s[at:]

open(p, "w").write(s)
print("[install] patched InterventionalDataset: paired-arm propagation (opt-in)")
PYEOF

# compile(), not ast.parse(): only compile() rejects a misplaced __future__ import
python3 -c "import sys; compile(open(sys.argv[1]).read(), sys.argv[1], 'exec'); print('[install] syntax OK')" "$DST"
echo "[install] done. Enable with UWYK_BOTH_ARMS=1"
