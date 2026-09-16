#!/usr/bin/env bash
# Patch CausalPFN's icl_model.cepo_losses to optionally supervise BOTH arms of
# every query point instead of one coin-flipped arm.
#
# WHY. Upstream does:
#     random_treatments = torch.randint(0, 2, E_y0_query.shape, ...)
#     Ey_target = where(random_treatments == 0, E_y0_std, E_y1_std)
# so each query point contributes ONE outcome value and its other potential
# outcome is discarded. (Their own comment above that line says "evaluating
# each of the query points at both t=1 and t=0" -- the code does not.)
#
# cpfn2d supervises BOTH arms of all N_q query points, so at equal steps the 1D
# head sees half the outcome values from the same covariates. With
# CPFN_QUERY_BOTH_ARMS=1 every query point is emitted twice -- t=0 targeting
# E_y0, t=1 targeting E_y1 -- giving 2*N_q rows from the same N_q covariates.
# The two heads then see identical data at identical step counts.
#
# OPT-IN. Without CPFN_QUERY_BOTH_ARMS=1 the coin flip is untouched, so this
# file is safe to install permanently: existing runs and evals are unaffected.
#
# Idempotent; backs up the original to icl_model.py.beforebotharms.
set -euo pipefail

CAUSALPFN="${CAUSALPFN:?CAUSALPFN must point at the causalpfn checkout}"
DST="$CAUSALPFN/src/causalpfn/models/icl_model.py"
[ -f "$DST" ] || { echo "FATAL: not found: $DST" >&2; exit 1; }

if grep -q "BOTH-ARMS PATCH" "$DST"; then
    echo "[install] already patched: $DST"
    exit 0
fi

cp -n "$DST" "$DST.beforebotharms" 2>/dev/null || true
echo "[install] backed up → $DST.beforebotharms"

python3 - "$DST" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p).read()

OLD_FLIP = """        # get access to both the factual and counterfactual predictions by evaluating each of the query points at
        # both t=1 and t=0
        random_treatments = torch.randint(0, 2, E_y0_query.shape, device=E_y0_query.device)
"""
NEW_FLIP = '''        # ── BOTH-ARMS PATCH ──────────────────────────────────────────────
        # Upstream drew ONE arm per query point and discarded that point's
        # other potential outcome, so a step supervised N_q outcome values from
        # N_q covariates. cpfn2d supervises BOTH arms of all N_q, i.e. 2*N_q
        # values from the same N_q covariates.
        #
        # CPFN_QUERY_BOTH_ARMS=1 emits every query point twice -- once at t=0
        # targeting E_y0, once at t=1 targeting E_y1 -- so the 1D head sees
        # exactly what the 2D head sees at the same step count. Query rows
        # carry no positional encoding and do not attend to one another, so
        # blocking [all t=0 | all t=1] is equivalent to interleaving the pairs.
        #
        # Unset (the default) restores upstream's coin flip exactly.
        if _os.environ.get("CPFN_QUERY_BOTH_ARMS", "0") == "1":
            X_query_eff = torch.cat([X_query, X_query], dim=1)
            query_treatments = torch.cat(
                [torch.zeros_like(E_y0_query), torch.ones_like(E_y1_query)], dim=1
            )
            Ey_target = torch.cat([E_y0_standardized, E_y1_standardized], dim=1)
        else:
            X_query_eff = X_query
            query_treatments = torch.randint(0, 2, E_y0_query.shape, device=E_y0_query.device)
            Ey_target = torch.where(query_treatments == 0, E_y0_standardized, E_y1_standardized)
'''
assert OLD_FLIP in s, "coin-flip block not found — upstream changed?"
s = s.replace(OLD_FLIP, NEW_FLIP, 1)

OLD_Q = """        x_and_t_query = torch.cat(
            [
                random_treatments.unsqueeze(-1),
                X_query,
            ],
            dim=2,
        )  # shape: (batch_size,  query_len , num_features + 1)

        Ey_target = torch.where(
            random_treatments == 0, E_y0_standardized, E_y1_standardized
        )  # shape: (batch_size, query_len)
"""
NEW_Q = """        x_and_t_query = torch.cat(
            [
                query_treatments.unsqueeze(-1).to(X_query_eff.dtype),
                X_query_eff,
            ],
            dim=2,
        )  # shape: (batch_size,  query_len , num_features + 1)
        # query_len is 2*N_q under CPFN_QUERY_BOTH_ARMS=1, N_q otherwise.
        # Fail loudly rather than let a target/row mismatch train silently.
        assert Ey_target.shape == x_and_t_query.shape[:2], (
            f"Ey_target {tuple(Ey_target.shape)} does not match query rows "
            f"{tuple(x_and_t_query.shape[:2])}"
        )
"""
assert OLD_Q in s, "query-assembly block not found — upstream changed?"
s = s.replace(OLD_Q, NEW_Q, 1)

# `os` under an alias, so we never collide with an existing name in this module.
# Insert before the FIRST top-level import, not at line 0 -- inserting ahead of
# a module docstring would demote it from __doc__ to a bare expression.
if "import os as _os" not in s:
    m = re.search(r"^(?:import |from )", s, re.MULTILINE)
    assert m, "no top-level import found to anchor the os import"
    s = s[: m.start()] + "import os as _os\n" + s[m.start() :]

open(p, "w").write(s)
print("[install] patched cepo_losses: both-arms query expansion (opt-in)")
PYEOF

python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read()); print('[install] syntax OK')" "$DST"
echo "[install] done. Enable with CPFN_QUERY_BOTH_ARMS=1"
