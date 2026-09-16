# Shared environment for every entry point in Reproduce/.
#   source "$(dirname "$0")/../env.sh"     (depth varies; see wrappers)
#
# Resolves the three third-party dependencies to the VENDORED copies inside
# this repo, so a fresh clone needs no external checkouts. An explicit
# CAUSALPFN / DOPFN_ROOT / UWYK in the environment still wins, for anyone
# deliberately testing against an upstream working copy.
#
# These are exported, so they also override the ${VAR:-default} fallbacks
# inside the underlying benchmarks/cluster scripts.

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO="${REPO:-$(cd "$_here/.." && pwd)}"
export VENDOR="${VENDOR:-$REPO/vendor}"

export CAUSALPFN="${CAUSALPFN:-$VENDOR/causalpfn}"
export DOPFN_ROOT="${DOPFN_ROOT:-$VENDOR/dopfn}"
export DOPFN="${DOPFN:-$DOPFN_ROOT}"
export UWYK="${UWYK:-$VENDOR/uwyk}"
export UWYK_SRC="${UWYK_SRC:-$UWYK/src}"
export UWYK_ROOT="${UWYK_ROOT:-$UWYK}"

# DEPLOY_ROOT is where the venv and the scratch outputs live. It is NOT a
# code dependency -- only the venv and the run outputs.
export DEPLOY_ROOT="${DEPLOY_ROOT:-$(cd "$REPO/.." && pwd)}"

# The faiss shim: CausalPFN imports faiss transitively but never calls it.
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

for _d in "$CAUSALPFN" "$DOPFN_ROOT" "$UWYK"; do
    if [ ! -d "$_d" ]; then
        echo "WARNING: missing dependency $_d" >&2
        echo "         run Reproduce/vendor_externals.sh once, or set the" >&2
        echo "         matching env var to an existing checkout." >&2
    fi
done
unset _here _d
