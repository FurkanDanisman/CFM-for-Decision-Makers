#!/bin/bash
# Slice to query positions BEFORE the output head in CausalPFN's model.forward.
#
# WHY. models/model.py does:
#
#     pred = self.head(src)                                   # FULL sequence
#     pred = 30 * torch.tanh(pred / (7.5 * src.size(-1)**0.5))
#     return pred[context_length:]                            # queries only
#
# Every CONTEXT position pays the full head projection and is then discarded.
# With the 1D head (nbins ~ 1024) that waste is invisible. With our 2D joint
# head at J=1024 the head emits J**2 + 13 = 1,048,599 logits per position, and
# the wasted work is what OOMs:
#
#     32,768 positions x 1,048,599 x 4 B = 128 GiB   <- observed failure
#
# Both nn.Linear and tanh are POINTWISE over positions, so computing them on
# src[context_length:] gives bit-identical output for the positions that are
# actually returned. This is a pure waste-removal, not an approximation.
# (src.size(-1) is the feature dim, unaffected by slicing positions.)
#
# Idempotent. Backs up the original as model.py.beforeheadslice on first run.
#
# Usage:  DEPLOY_ROOT=/path/to/deploy ./install.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -d "$DEPLOY_ROOT/external/causalpfn" ]; then
    for _ in 1 2 3; do
        [ -d "$DEPLOY_ROOT/external/causalpfn" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
DST="$CAUSALPFN/src/causalpfn/models/model.py"
[ -f "$DST" ] || { echo "FATAL: $DST not found" >&2; exit 1; }
[ -f "${DST}.beforeheadslice" ] || { echo "[head-slice] backing up -> ${DST}.beforeheadslice"; cp "$DST" "${DST}.beforeheadslice"; }

python3 - "$DST" <<'PY'
import sys, re
p = sys.argv[1]
s = open(p).read()

MARK = '[rpfn head-slice]'
if MARK in s:
    print('[head-slice] already patched; nothing to do')
    sys.exit(0)

old = """        pred = self.head(src)

        pred = 30 * torch.tanh(pred / (7.5 * src.size(-1) ** 0.5))
        if return_log_act_norms:
            return pred[context_length:], log_act_norms
        else:
            return pred[context_length:]"""

new = """        # [rpfn head-slice] Only query positions are returned below, and both
        # the head Linear and the tanh are POINTWISE over positions -- so
        # slicing first is exactly equivalent and skips the projection on every
        # context position. At J=1024 the 2D head emits 1,048,599 logits per
        # position, which is the difference between a 128 GiB allocation and a
        # (n_query / (context + n_query)) fraction of it.
        pred = self.head(src[context_length:])

        pred = 30 * torch.tanh(pred / (7.5 * src.size(-1) ** 0.5))
        if return_log_act_norms:
            return pred, log_act_norms
        else:
            return pred"""

if old not in s:
    print('FATAL: the expected forward() body was not found in', p, file=sys.stderr)
    print('       upstream may have changed; patch NOT applied.', file=sys.stderr)
    sys.exit(2)

n = s.count(old)
if n != 1:
    print(f'FATAL: expected 1 match, found {n}', file=sys.stderr); sys.exit(3)

open(p, 'w').write(s.replace(old, new, 1))
print('[head-slice] patched', p)
PY

python3 -c "
import ast,sys; ast.parse(open('$DST').read()); print('[head-slice] $DST parses OK')"
echo "[head-slice] done. Revert with: cp ${DST}.beforeheadslice $DST"
