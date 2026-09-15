"""Prove the RealCause eval's arm_centered path inverts the training transform.

A checkpoint trained with y_scaling_mode='arm_centered' predicts each arm
relative to THAT ARM's own context mean. If the eval standardises pooled
instead, every CATE comes out short by exactly (m1 - m0) -- a result that looks
entirely plausible (right shape, right spread) but is uniformly biased. That is
the same failure mode the cpfn1d per-arm density dumps hit, where predictions
landed ~2.0 off and coverage fell to 43%.

This runs the REAL forward_pmats with a stub backbone, so it tests the shipped
code path rather than a re-implementation of it.

    python -u benchmarks/eval_causalpfn2d/verify_arm_centered_scaling.py

Needs CAUSALPFN set (same as any eval job). No checkpoint required.
"""
from __future__ import annotations
import os
import sys

os.environ.setdefault('CKPT', '/dev/null')      # module reads these at import;
os.environ.setdefault('OUT', '/tmp/_verify')    # neither is touched here
os.environ.pop('STD_MODE', None)
os.environ.pop('Y_STD_MODE_EVAL', None)

import numpy as np
import torch

# causalpfn.__init__ pulls in faiss/transformers/wandb at import time, none of
# which this check touches. The Table-1 shim stubs them via a meta-path finder
# that is normally enabled by putting its directory on PYTHONPATH (the eval
# sbatch does exactly that). This script is run by hand, so install the finder
# here instead of relying on the caller's environment.
_SHIMS = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'uwyk_table1', 'shims'))
if os.path.isdir(_SHIMS):
    sys.path.insert(0, _SHIMS)
    try:
        import sitecustomize  # noqa: F401  (installs the stub finder on import)
    except ImportError:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_cpfn2d_realcause as E
from training_causalpfn2d.model_causalpfn_2d import CausalPFN2DHead

J = 32
RTOL = 1e-6


class StubBackbone:
    """Captures the standardised y the eval hands the network."""
    def __init__(self):
        self.y_std = None

    def _forward_logits(self, X_ctx, T_ctx, y_std, X_q):
        self.y_std = y_std.detach().cpu().double().numpy().copy()
        return torch.zeros(1, X_q.shape[1], J * J + 13)


def make_context(n=400, d=5, arm_offset=2.0, seed=0):
    """Context where the two arms differ by a KNOWN mean offset."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    T = (rng.random(n) < 0.5).astype(np.float32)
    # control arm ~ N(5, 1.5), treated arm shifted by arm_offset
    Y = (5.0 + 1.5 * rng.normal(size=n) + arm_offset * T).astype(np.float32)
    return X, T, Y


def check(name, ok, detail=''):
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'  {detail}' if detail else ''))
    return ok


def main():
    failures = 0
    X, T, Y = make_context(arm_offset=2.0)
    Xq = X[:20]

    # ── 1. eval's standardisation == training's, elementwise ────────────────
    print('\n1. eval arm_centered standardisation vs training _arm_centered_y_stats')
    stub = StubBackbone()
    _, _, stats = E.forward_pmats(stub, X, T, Y, Xq, J, 'arm_centered', 'arm_centered')

    yc = torch.from_numpy(Y).unsqueeze(0)
    tc = torch.from_numpy(T).unsqueeze(0)
    m0_t, m1_t, sc_t = CausalPFN2DHead._arm_centered_y_stats(yc, tc)
    m0_t = float(m0_t.item()); m1_t = float(m1_t.item()); sc_t = float(sc_t.item())

    y_std_train = ((yc - torch.where(tc > 0.5, torch.full_like(tc, m1_t),
                                     torch.full_like(tc, m0_t))) / sc_t)
    y_std_train = y_std_train.double().numpy()
    gap = float(np.max(np.abs(stub.y_std - y_std_train)))
    failures += not check('y_std matches training transform', gap < 1e-6,
                          f'max|delta|={gap:.3e}')
    failures += not check('m0 matches', abs(stats['y0s'] - m0_t) < 1e-6,
                          f"eval={stats['y0s']:.6f} train={m0_t:.6f}")
    failures += not check('m1 matches', abs(stats['y1s'] - m1_t) < 1e-6,
                          f"eval={stats['y1s']:.6f} train={m1_t:.6f}")
    failures += not check('scale is SHARED across arms',
                          stats['y0sc'] == stats['y1sc'],
                          f"y0sc={stats['y0sc']:.6f} y1sc={stats['y1sc']:.6f}")
    failures += not check('scale matches training (pooled context std)',
                          abs(stats['y0sc'] - sc_t) < 1e-6,
                          f"eval={stats['y0sc']:.6f} train={sc_t:.6f}")

    # ── 2. round trip: un-scaling recovers the true CATE ─────────────────────
    print('\n2. round trip  cate = (e1-e0)*scale + (m1-m0)')
    rng = np.random.default_rng(7)
    E_y0 = rng.normal(5.0, 1.0, size=20)          # arbitrary raw arm means
    E_y1 = E_y0 + rng.normal(2.0, 0.5, size=20)   # arbitrary raw CATE
    true_cate = E_y1 - E_y0
    # what the model sees / predicts, in its own standardised units
    e0_std = (E_y0 - m0_t) / sc_t
    e1_std = (E_y1 - m1_t) / sc_t
    # exactly the un-scaling evaluate() applies on stats['mode'] == 'per_arm'
    got = (e1_std * stats['y1sc'] + stats['y1s']) - (e0_std * stats['y0sc'] + stats['y0s'])
    err = float(np.max(np.abs(got - true_cate)))
    failures += not check('CATE recovered', err < 1e-9, f'max|delta|={err:.3e}')

    # ── 3. pooled scoring of the same checkpoint is wrong by exactly (m1-m0) ─
    print('\n3. what pooled scoring would have done (the bug this prevents)')
    stub_p = StubBackbone()
    _, _, sp = E.forward_pmats(stub_p, X, T, Y, Xq, J, 'arm_centered', 'pooled')
    pooled_cate = (e1_std - e0_std) * sp['scale']
    bias = float(np.mean(pooled_cate - true_cate))
    offset = m1_t - m0_t
    failures += not check('pooled bias == -(m1-m0)', abs(bias + offset) < 1e-6,
                          f'bias={bias:+.6f}  -(m1-m0)={-offset:+.6f}')
    print(f'       -> pooled scoring would understate every CATE by {offset:.3f} '
          f'(true ATE {true_cate.mean():.3f} -> {pooled_cate.mean():.3f})')

    # ── 4. degenerate arm falls back, as training does ──────────────────────
    print('\n4. degenerate arm (all-control context)')
    T0 = np.zeros_like(T)
    _, _, sd = E.forward_pmats(StubBackbone(), X, T0, Y, Xq, J,
                               'arm_centered', 'arm_centered')
    failures += not check('m1 falls back to m0', abs(sd['y1s'] - sd['y0s']) < 1e-9,
                          f"m0={sd['y0s']:.6f} m1={sd['y1s']:.6f}")
    failures += not check('offset is 0, not NaN', abs(sd['arm_offset']) < 1e-9)

    print(f'\n{"ALL CHECKS PASSED" if not failures else f"{failures} CHECK(S) FAILED"}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
