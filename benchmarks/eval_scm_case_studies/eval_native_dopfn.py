"""Native Do-PFN eval on the 6 synthetic case studies.

Uses DoPFNRegressor from the dopfn upstream repo. Per realization:
  - fit on training rows (T is col 0 of X)
  - predict_cate on test X
  - report PEHE + ATE_err (matching our other eval scripts)

Env vars:
  DATASET     one of the 6 case-study names (Observed_Confounder, ...)
  OUT         per-realization NPZ dir
  DOPFN_ROOT  path to dopfn_upstream repo (has scripts/, artifacts/)
  DOPFN_DATA_ROOT  parent of prior_sampling/  (default: $DOPFN_ROOT/data/prior_sampling)
  MAX_REAL    optional cap
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=os.environ.get('DATASET', 'Observed_Confounder'))
args, _ = parser.parse_known_args()
DATASET   = args.dataset
OUT       = os.environ['OUT']
DOPFN_ROOT = os.environ['DOPFN_ROOT']
CAUSALPFN  = os.environ.get('CAUSALPFN', '')  # required for RealCause loaders (from `benchmarks import IHDPDataset`)
MAX_REAL  = os.environ.get('MAX_REAL', '')

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, os.path.join(REPO_SRC, 'benchmarks'))   # top-level import
sys.path.insert(0, DOPFN_ROOT)
# CausalPFN owns the `benchmarks` package that ships IHDPDataset/ACIC2016Dataset/...
if CAUSALPFN:
    sys.path.insert(0, CAUSALPFN)
    sys.path.insert(0, CAUSALPFN + '/src')

# sklearn renamed check_array's finite-check kwarg: force_all_finite became
# ensure_all_finite in 1.6, and the old name was removed in 1.8. Do-PFN's
# base.py and various callers use BOTH spellings depending on vintage, so
# translate toward whichever the INSTALLED sklearn accepts rather than assuming
# a direction -- a one-way shim breaks on the other half of the version range:
#     old sklearn + ensure_all_finite -> TypeError
#     new sklearn + force_all_finite  -> TypeError   (what Fir hit)
import inspect as _inspect  # noqa: E402
import sklearn.utils as _sku  # noqa: E402
_orig_ca = _sku.check_array
try:
    _ca_params = _inspect.signature(_orig_ca).parameters
    if any(p.kind is _inspect.Parameter.VAR_KEYWORD for p in _ca_params.values()):
        raise ValueError                      # **kwargs tells us nothing
    _CA_WANT = ('ensure_all_finite' if 'ensure_all_finite' in _ca_params
                else 'force_all_finite')
except Exception:                             # fall back on the version
    import sklearn as _sk  # noqa: E402
    _maj, _min = (int(x) for x in _sk.__version__.split('.')[:2])
    _CA_WANT = 'ensure_all_finite' if (_maj, _min) >= (1, 6) else 'force_all_finite'
_CA_OTHER = ('force_all_finite' if _CA_WANT == 'ensure_all_finite'
             else 'ensure_all_finite')


def _patched_check_array(*a, **kw):
    if _CA_OTHER in kw:
        kw.setdefault(_CA_WANT, kw.pop(_CA_OTHER))
    return _orig_ca(*a, **kw)


_sku.check_array = _patched_check_array
print(f"[sklearn-shim] check_array finite kwarg -> {_CA_WANT!r}", flush=True)
# Also patch in the sklearn.utils.validation namespace (where check_array lives)
import sklearn.utils.validation as _skuv  # noqa: E402
_skuv.check_array = _patched_check_array

from scm_case_study_dataset import SCMCaseStudyDataset  # noqa: E402

# RealCause loaders (importing here is safe — they're lightweight).
_REALCAUSE = ('IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal')


# ── UWYK_Fig3_4 ComplexMech PEHE benchmark hook ──────────────────────────────
# Dispatches dataset names like CMECH_n20_nonzero to
# benchmarks/uwyk_fig34_dataset.py. Path-robust: this file may sit in
# benchmarks/<sub>/ or realcause_eval/<sub>/.
def _cmech_bench_dir():
    import os as _os, sys as _sys
    _d = _os.path.dirname(_os.path.abspath(__file__))
    for _ in range(5):
        _c = _os.path.join(_d, 'benchmarks')
        if _os.path.isdir(_c):
            if _c not in _sys.path:
                _sys.path.insert(0, _c)
            return _c
        _d = _os.path.dirname(_d)
    return None


def _cmech_names():
    _cmech_bench_dir()
    try:
        from uwyk_fig34_dataset import dataset_names
    except ImportError:
        return ()
    return tuple(dataset_names())


def _cmech_dataset(name):
    """UWYK_Fig3_4 ComplexMech dataset for `name`, or None if not one of ours."""
    _cmech_bench_dir()
    try:
        from uwyk_fig34_dataset import UWYKFig34Dataset, parse_name
    except ImportError:
        return None
    return UWYKFig34Dataset(name) if parse_name(name) else None


_CMECH_CASES = _cmech_names()
# ─────────────────────────────────────────────────────────────────────────────

def _get_dataset(name: str):
    if name in _REALCAUSE:
        from benchmarks import (IHDPDataset, ACIC2016Dataset,
                                 RealCauseLalondeCPSDataset,
                                 RealCauseLalondePSIDDataset)
        return {
            'IHDP':     IHDPDataset(),
            'ACIC':     ACIC2016Dataset(),
            'CPS':      RealCauseLalondeCPSDataset(),
            'PSID':     RealCauseLalondePSIDDataset(),
            'PSID_bal': RealCauseLalondePSIDDataset(),
        }[name]
    _cm = _cmech_dataset(name)
    if _cm is not None:
        return _cm
    return SCMCaseStudyDataset(name)


def _cate_ds_from(ds, r: int, name: str):
    """Return the CATE-style dataset for realization r."""
    if name in _REALCAUSE:
        # RealCause loaders: ds[r] returns (cate_ds, meta) or just cate_ds.
        got = ds[r]
        return got[0] if isinstance(got, tuple) else got
    # SCMCaseStudyDataset: ds[r] returns (cate_ds, meta).
    return ds[r][0]


def _n_tables(ds, name: str):
    return int(getattr(ds, 'n_tables', len(ds)))

# DoPFNRegressor loads artifacts by relative path, so cwd matters.
_prev_cwd = os.getcwd()
os.chdir(DOPFN_ROOT)
try:
    from scripts.transformer_prediction_interface.base import DoPFNRegressor  # noqa: E402
    # DoPFNRegressor defaults to device='cpu'. Some versions of the upstream
    # accept `device=` in __init__, some only expose it as an attribute. Try
    # kwarg first, fall back to setattr so the model is moved on first fit().
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = DoPFNRegressor(device=_device)
    except TypeError:
        model = DoPFNRegressor()
        model.device = _device
    print(f'[dopfn_native] instantiated on device={_device}', flush=True)
finally:
    os.chdir(_prev_cwd)


# ── DOPFN_CKPT: run this pipeline with OUR weights ──────────────────────────
# DoPFNRegressor loads the released artifacts. Pointing DOPFN_CKPT at a
# training_dopfn_repro dopfn_1d checkpoint swaps our trained weights into the
# same regressor, so all of Do-PFN's own preprocessing -- y normalisation, T in
# column 0, the predict_cate do(1)-do(0) difference -- is applied exactly as the
# repro was trained to expect, instead of being re-implemented and risking a
# silent convention mismatch.
#
# The repro mutates the PerFeatureTransformer in place (decoder_dict['standard']
# swapped, model.criterion replaced), so its state dict has the SAME key layout
# as the regressor's own model, including criterion.borders -- which is where
# the 1-D bar borders live, since install_criterion returns None for dopfn_1d
# and the checkpoint's 'edges' field is empty.
# The repro's own 2D decoder, so the logit layout is not re-derived here:
# pred[..., :J*J] -> softmax -> (J, J), then 9 region weights + 4 tail scales.
try:
    from losses.BarDistribution2D import unpack_pred as _unpack_pred_2d
except Exception:                                    # only needed for joint_2d
    _unpack_pred_2d = None

_DOPFN_CKPT = os.environ.get('DOPFN_CKPT', '')
_REPRO_SD = None
_IS_2D = False
_CKPT_EDGES = None
_J2D = None
_REPRO_BORDERS = None
if _DOPFN_CKPT:
    _blob = torch.load(_DOPFN_CKPT, map_location='cpu', weights_only=False)
    _REPRO_SD = _blob.get('model', _blob.get('model_state_dict'))
    if _REPRO_SD is None:
        raise SystemExit(f'DOPFN_CKPT {_DOPFN_CKPT} has no model/model_state_dict')
    _prov = _blob.get('provenance') or {}
    _variant = _prov.get('variant') if isinstance(_prov, dict) else None
    # joint_2d runs through THIS pipeline too, so DoPFN's own preprocessing --
    # its y normalisation above all -- is applied exactly as the repro was
    # trained to expect. The alternative (eval_dopfn_bb_raw) re-implements the
    # scaling with a --y-scaling flag, and its grid is min-max [-1,1] while the
    # repro's edges span [-2.4865, +3.7314]: the data then lands in ~3 of 10
    # bins.
    if _variant not in (None, 'dopfn_1d', 'joint_2d'):
        raise SystemExit(f'DOPFN_CKPT variant is {_variant!r}; this path handles '
                         f'dopfn_1d and joint_2d')
    _IS_2D = (_variant == 'joint_2d')
    _CKPT_EDGES = None
    if _IS_2D:
        _e = _blob.get('edges')
        if _e is None:
            raise SystemExit('joint_2d checkpoint has no edges; the 2D grid is required')
        _CKPT_EDGES = np.asarray(_e.tolist() if hasattr(_e, 'tolist') else _e,
                                 dtype=np.float64)
        _J2D = _CKPT_EDGES.size - 1            # 11 edges -> J = 10
        print(f'[dopfn_native] joint_2d: J={_J2D} edges=[{_CKPT_EDGES[0]:+.4f}, '
              f'{_CKPT_EDGES[-1]:+.4f}] (training y-space)', flush=True)
    print(f'[dopfn_native] DOPFN_CKPT={_DOPFN_CKPT} step={_blob.get("step")} '
          f'variant={_variant} ({len(_REPRO_SD)} tensors)', flush=True)


# ── J mismatch: rebuild the bin-shaped tensors before loading ────────────────
# The released regressor is constructed at ITS OWN J -- 100 bins, so borders is
# 101 wide. A repro checkpoint trained at J=10 carries a 10-row head and an
# 11-point border grid, and load_state_dict rejects five tensors on shape:
#   decoder_dict.standard.2.{weight,bias}
#   criterion.{borders,bucket_widths,losses_per_bucket}
# Every one is fully specified BY the checkpoint, so replacing the target's
# tensors with correctly shaped ones loses nothing -- the load that follows
# overwrites their contents. What we must not do is resize anything else, or a
# real architecture difference would be silently reshaped away instead of
# raising. Hence the allowlist: only the output head's final layer and the bar
# distribution's own buffers may change width.
_RESIZABLE = ('criterion.',)
_RESIZABLE_SUFFIX = ('decoder_dict.standard.2.weight', 'decoder_dict.standard.2.bias')


def _resizable(key):
    return key.startswith(_RESIZABLE) or key.endswith(_RESIZABLE_SUFFIX)


def _resize_bins_to_ckpt(target, sd):
    """Widen/narrow the allowlisted bin-shaped tensors to match `sd`.

    Returns the list of (key, old_shape, new_shape) actually changed, so the
    caller can report it: a silent J change would be exactly the kind of thing
    that makes a model score against the wrong grid.
    """
    import torch.nn as _nn
    tgt = target.state_dict()
    changed = []
    for k, v in sd.items():
        if k not in tgt or tuple(tgt[k].shape) == tuple(v.shape):
            continue
        if not _resizable(k):
            continue                      # leave it to load_state_dict to reject
        owner = target
        for part in k.split('.')[:-1]:
            owner = owner[int(part)] if part.isdigit() else getattr(owner, part)
        leaf = k.split('.')[-1]
        cur = getattr(owner, leaf, None)
        # clone, not empty_like: the tensor then already holds the right
        # values, so nothing depends on the subsequent load touching it.
        #
        # ...and follow the module's device/dtype, not the checkpoint's. The
        # checkpoint is loaded with map_location='cpu' while the regressor may
        # already be on cuda, so a bare clone leaves the new head on CPU and the
        # first matmul dies with "mat1 is on cuda:0, different from other
        # tensors on cpu".
        new = v.detach().clone()
        if cur is not None and hasattr(cur, 'device'):
            new = new.to(device=cur.device, dtype=cur.dtype)
        if isinstance(cur, _nn.Parameter):
            setattr(owner, leaf, _nn.Parameter(
                new, requires_grad=bool(cur.requires_grad)))
        elif leaf in getattr(owner, '_buffers', {}):
            owner._buffers[leaf] = new
        else:
            setattr(owner, leaf, new)
        if isinstance(owner, _nn.Linear) and leaf == 'weight':
            owner.out_features = int(v.shape[0])
        changed.append((k, tuple(tgt[k].shape), tuple(v.shape)))

    # Derived bin counts cached on the criterion would otherwise stay at the old
    # J and be used to reshape logits later.
    nb = None
    for k, v in sd.items():
        if k.endswith('criterion.borders') or k == 'criterion.borders':
            nb = int(v.shape[0]) - 1
    if nb is not None:
        crit = getattr(target, 'criterion', None)
        for attr in ('num_bars', 'num_buckets', 'n_bars', 'num_classes'):
            if crit is not None and isinstance(getattr(crit, attr, None), int):
                setattr(crit, attr, nb)
    return changed


def _inject_weights(reg):
    """Load our state dict into whichever nn.Module the regressor holds.

    The attribute name differs across upstream versions, so find it by matching
    state-dict keys rather than hard-coding one, and load strict=True so a
    partial or mismatched load is an error and not a silently half-trained model.
    """
    if _REPRO_SD is None:
        return
    import torch.nn as _nn
    want = set(_REPRO_SD)
    best = None
    for _name in dir(reg):
        if _name.startswith('__'):
            continue
        try:
            obj = getattr(reg, _name)
        except Exception:
            continue
        if isinstance(obj, _nn.Module):
            have = set(obj.state_dict())
            if not have:
                continue
            overlap = len(want & have) / max(len(want), 1)
            if best is None or overlap > best[0]:
                best = (overlap, _name, obj)
    if best is None:
        raise SystemExit('[dopfn_native] no nn.Module found on the regressor to load into')
    overlap, name, target = best
    if overlap < 0.99:
        raise SystemExit(
            f'[dopfn_native] best match {name!r} shares only {overlap:.1%} of keys '
            f'with the checkpoint — refusing to load a mismatched model')
    # The regressor builds its PerFeatureTransformer WITH attention biases; the
    # pickle the repro trained from has none, so ~48 in_proj_bias/out_proj.bias
    # keys are absent from our checkpoint. A bias-free layer is exactly a layer
    # whose bias is zero, so load non-strict and zero those tensors -- this is
    # an equivalence, not an approximation.
    #
    # Guarded hard: every missing key must be a bias, and nothing we carry may
    # be unexpected. Anything else means the architectures genuinely differ and
    # a partial load would quietly evaluate a half-released, half-ours model.
    _resized = _resize_bins_to_ckpt(target, _REPRO_SD)
    if _resized and not getattr(_inject_weights, '_said_resize', False):
        _inject_weights._said_resize = True
        print(f'[dopfn_native] checkpoint J differs from the released model; '
              f'rebuilt {len(_resized)} bin-shaped tensor(s):', flush=True)
        for _k, _o, _n in _resized:
            print(f'    {_k}: {_o} -> {_n}', flush=True)
    missing, unexpected = target.load_state_dict(_REPRO_SD, strict=False)
    if unexpected:
        raise SystemExit(
            f'[dopfn_native] checkpoint has {len(unexpected)} keys the model does '
            f'not accept, e.g. {sorted(unexpected)[:3]} — architectures differ')
    non_bias = [k for k in missing if not k.endswith('bias')]
    if non_bias:
        raise SystemExit(
            f'[dopfn_native] {len(non_bias)} missing key(s) are NOT biases, e.g. '
            f'{sorted(non_bias)[:3]} — refusing to load a partially trained model')
    if missing:
        _tgt_sd = target.state_dict()
        with torch.no_grad():
            for k in missing:
                _tgt_sd[k].zero_()
    global _REPRO_BORDERS
    _b = getattr(getattr(target, 'criterion', None), 'borders', None)
    if _b is not None:
        _REPRO_BORDERS = _b.detach().cpu().numpy().astype('float64').copy()
    if not getattr(_inject_weights, '_announced', False):
        _inject_weights._announced = True
        print(f'[dopfn_native] loaded DOPFN_CKPT weights into regressor.{name} '
              f'({overlap:.1%} key match; zeroed {len(missing)} absent bias '
              f'tensors, 0 unexpected)', flush=True)


def _neutralise_criterion_summaries(crit):
    """Let predict_full run on 2-D logits.

    predict_full builds a summary dict -- mean, median, quantiles -- and every one
    of those interprets the logits as 1-D bar-distribution parameters. With a
    joint_2d head they are 113 wide against a 100-bin criterion, so each fails in
    its own way:
        mean   -> size mismatch, got input (75), mat (75x113), vec (100)
        median -> icdf -> IndexError: index 110 is out of bounds ... size 101
    Gating them one at a time just moves the error, so the whole family is gated
    together.

    Only two things are needed from predict_full: the raw logits, and the borders
    it RESCALED by data_std/data_mean -- which is how the 2-D grid reaches raw Y
    units. None of the summaries is read on the 2-D path; the point estimate comes
    from the joint itself. Every patch is WIDTH-GATED, so a genuine 1-D call runs
    the original code, and they are applied at class level so they survive the
    deepcopy predict_full makes of the criterion.
    """
    cls = type(crit)
    if getattr(cls, '_j2d_patched', False):
        return
    names = ('mean', 'median', 'mode', 'icdf', 'quantile', 'cdf', 'ucb', 'sample')
    patched = []
    for name in names:
        orig = getattr(cls, name, None)
        if not callable(orig):
            continue

        def _make(orig_fn):
            def wrapper(self, logits, *args, **kwargs):
                nb = getattr(self, 'borders', None)
                n_bins = (int(nb.shape[0]) - 1) if nb is not None else None
                if n_bins is not None and int(logits.shape[-1]) != n_bins:
                    # Width says these are not 1-D bar parameters. Return a
                    # correctly shaped placeholder instead of raising.
                    return torch.zeros(logits.shape[:-1], dtype=logits.dtype,
                                       device=logits.device)
                return orig_fn(self, logits, *args, **kwargs)
            return wrapper

        setattr(cls, name, _make(orig))
        patched.append(name)
    cls._j2d_patched = True
    print(f'[dopfn_native][2d] width-gated {cls.__name__}: {", ".join(patched)}',
          flush=True)


def _predict_joint2d(model, X_test_full, y_train=None):
    """CATE and the joint density for a joint_2d head, inside DoPFN's own pipeline.

    Two things differ from the 1-D path and both are forced, not stylistic:

    ONE FORWARD, NOT TWO. The 2-D head emits the JOINT p(Y0, Y1) for a query, so
    there is no do(0)/do(1) pair to difference -- the treatment column is zeroed,
    exactly as in training.

    predict_cate IS UNUSABLE HERE. It computes criterion.mean(logits) with the
    1-D criterion (100 bins) against 113-wide 2-D logits, which is meaningless.
    The point estimate therefore comes from the joint itself,
    E[tau] = sum_ij p_ij (c_j - c_i), which is the same quantity the density
    represents rather than a second, inconsistent estimator.

    Un-normalisation is recovered, not assumed: predict_full rescales the
    criterion's borders as borders * data_std + data_mean, so comparing the
    borders before and after that call pins the affine map exactly, and the same
    map carries the 2-D grid from training space into raw Y units. That is the
    whole reason for routing joint_2d through this pipeline -- no --y-scaling
    choice to get wrong.
    """
    if _unpack_pred_2d is None:
        raise SystemExit('joint_2d needs losses/BarDistribution2D.py on the path')
    _crit = getattr(model, 'criterion', None) or getattr(
        getattr(model, 'model_processed_', None), 'criterion', None)
    if _crit is not None:
        _neutralise_criterion_summaries(_crit)
    Xq = X_test_full.copy()
    Xq[:, 0] = 0.0                                   # treatment zeroed for queries
    fq = model.predict_full(torch.from_numpy(Xq.astype(np.float32)))
    logits = np.asarray(fq['logits'], dtype=np.float64)          # (N_q, J*J+13)
    J = int(_J2D)
    need = J * J + 13
    if logits.shape[-1] != need:
        raise SystemExit(f'joint_2d expected {need} logits for J={J}, '
                         f'got {logits.shape[-1]}')

    # Affine map predict_full applied to the borders; two distinct points fix it.
    B_after = np.asarray(fq['criterion'].borders.detach().cpu().numpy(), dtype=np.float64)
    b_before = _REPRO_BORDERS
    if b_before is None or b_before.size != B_after.size:
        raise SystemExit('cannot recover the y normalisation: borders before/after '
                         'predict_full do not correspond')
    span = float(b_before[-1] - b_before[0])
    if abs(span) < 1e-12:
        raise SystemExit('degenerate borders; cannot recover data_std')
    data_std = float(B_after[-1] - B_after[0]) / span
    data_mean = float(B_after[0]) - float(b_before[0]) * data_std
    edges_raw = _CKPT_EDGES * data_std + data_mean

    p_mat = _unpack_pred_2d(torch.from_numpy(logits), J,
                            float(_CKPT_EDGES[1] - _CKPT_EDGES[0]))[0]
    p_mat = np.asarray(p_mat.detach().cpu().numpy(), dtype=np.float64)   # (N_q, J, J)
    p_mat = p_mat / np.maximum(p_mat.sum(axis=(1, 2), keepdims=True), 1e-300)

    centers = 0.5 * (edges_raw[:-1] + edges_raw[1:])
    # axis 1 indexes Y0, axis 2 indexes Y1 (BarDistribution2D._compute_rho:
    # marg0 = p_mat.sum(dim=-1), marg1 = p_mat.sum(dim=-2)).
    cate = (p_mat.sum(axis=1) @ centers) - (p_mat.sum(axis=2) @ centers)

    if not getattr(_predict_joint2d, '_said', False):
        _predict_joint2d._said = True
        print(f'[dopfn_native][2d] data_std={data_std:.6g} data_mean={data_mean:.6g}  '
              f'edges_raw=[{edges_raw[0]:+.4f}, {edges_raw[-1]:+.4f}]', flush=True)
        # Separate "the joint is placed correctly" from "the model predicts tau
        # badly". E[Y0] and E[Y1] are ABSOLUTE outcome levels, so they can be
        # checked against the observed y without knowing any counterfactual: if
        # they do not sit near the data, the decode or the grid is wrong; if they
        # do and tau is still poor, the model is weak and the pipeline is fine.
        _m0 = float((p_mat.sum(axis=2) @ centers).mean())
        _m1 = float((p_mat.sum(axis=1) @ centers).mean())
        _mass_in = float(p_mat.sum(axis=(1, 2)).mean())
        # rho of the joint, same construction as BarDistribution2D._compute_rho.
        _a0 = p_mat.sum(axis=2); _a1 = p_mat.sum(axis=1)
        _E0 = _a0 @ centers; _E1 = _a1 @ centers
        _V0 = np.maximum(_a0 @ (centers ** 2) - _E0 ** 2, 1e-12)
        _V1 = np.maximum(_a1 @ (centers ** 2) - _E1 ** 2, 1e-12)
        _E01 = (p_mat * centers[None, :, None] * centers[None, None, :]).sum(axis=(1, 2))
        _rho = float(np.mean((_E01 - _E0 * _E1) / np.sqrt(_V0 * _V1)))
        print(f'[dopfn_native][2d] joint: E[Y0]={_m0:+.4f} E[Y1]={_m1:+.4f} '
              f'ATE={_m1 - _m0:+.4f}  rho={_rho:+.4f}  inner_mass={_mass_in:.4f}',
              flush=True)
        if y_train is not None:
            _y = np.asarray(y_train, dtype=np.float64).reshape(-1)
            print(f'[dopfn_native][2d] observed y: mean={_y.mean():+.4f} '
                  f'sd={_y.std():.4f} range=[{_y.min():+.4f}, {_y.max():+.4f}]',
                  flush=True)
            if not (_y.min() - 3 * _y.std() < _m0 < _y.max() + 3 * _y.std()):
                print('[dopfn_native][2d][WARN] E[Y0] lies far outside the observed '
                      'y range -- the grid or the decode is wrong, not the model.',
                      flush=True)
    dens = dict(
        edges=edges_raw.astype(np.float32),
        p_joint_scaled=p_mat.astype(np.float32),
        y_shift=np.float32(0.0),
        y_scale=np.float32(1.0),                     # edges already in raw units
    )
    return cate.astype(np.float64), dens


def evaluate(r: int, ds):
    cate_ds = _cate_ds_from(ds, r, DATASET)
    X_train_full = np.hstack([cate_ds.t_train.reshape(-1, 1), cate_ds.X_train])
    y_train = cate_ds.y_train
    X_test_full = np.hstack([np.zeros((cate_ds.X_test.shape[0], 1), dtype=np.float32),
                              cate_ds.X_test])   # T-col placeholder (predict_cate handles both)

    # Optional random context subsampling (matches cpfn1d/graph2d convention).
    _eval_cap = os.environ.get('EVAL_MAX_CONTEXT', '')
    if _eval_cap:
        cap = int(_eval_cap)
        n_ctx = X_train_full.shape[0]
        if cap < n_ctx:
            _seed = int(os.environ.get('EVAL_CONTEXT_SEED', '0'))
            rng = np.random.default_rng(_seed + r)
            idx = rng.choice(n_ctx, size=cap, replace=False)
            X_train_full = X_train_full[idx]
            y_train = y_train[idx]

    # DoPFNRegressor.fit() expects X with T in col 0; predict_cate does the do(1)-do(0) diff.
    # predict_cate internally calls X.cpu().detach().numpy(), so pass a torch tensor.
    os.chdir(DOPFN_ROOT)
    try:
        model.fit(X_train_full, y_train)
        _inject_weights(model)   # after fit: the module may be built there
        X_test_t = torch.from_numpy(X_test_full.astype(np.float32))
        if _IS_2D:
            cate_pred, _dens2d = _predict_joint2d(model, X_test_full, y_train)
        else:
            cate_pred, _dens2d = model.predict_cate(X_test_t), None
        # For density dump: get raw bin probs via predict_full on both arms.
        dens = _dens2d
        if (not _IS_2D) and os.environ.get('DENSITY_DUMP', '0') == '1':
            X0 = X_test_full.copy(); X0[:, 0] = 0.0
            X1 = X_test_full.copy(); X1[:, 0] = 1.0
            X0_t = torch.from_numpy(X0.astype(np.float32))
            X1_t = torch.from_numpy(X1.astype(np.float32))
            full0 = model.predict_full(X0_t)
            full1 = model.predict_full(X1_t)
            logits0 = np.asarray(full0['logits'])                 # (N_q, nbins)
            logits1 = np.asarray(full1['logits'])
            edges = np.asarray(full0['criterion'].borders)        # (nbins+1,) — natural Y units
            p_y0 = np.exp(logits0 - logits0.max(axis=-1, keepdims=True))
            p_y0 = p_y0 / p_y0.sum(axis=-1, keepdims=True)
            p_y1 = np.exp(logits1 - logits1.max(axis=-1, keepdims=True))
            p_y1 = p_y1 / p_y1.sum(axis=-1, keepdims=True)
            # criterion.borders ARE in raw Y units — measured, not assumed:
            # regressing cate_pred on the density mean gives slope 1.038 with
            # R^2 = 0.9992 on well-behaved realizations. So y_scale stays 1.0.
            #
            # (An earlier commit rescaled by this slope on the theory that
            # TabPFN-derived models normalise the target internally. That was
            # wrong for DoPFN and is reverted: the slope is ~1, and forcing a
            # per-realization rescale would have corrupted the density using a
            # factor that is 7.9 on the realizations where the fit is poor.)
            #
            # BUCKET MEANS EXACTLY AS DoPFN COMPUTES THEM.
            #
            # predict_cate -> predict_cid -> predict -> predict_full()['mean']
            # = criterion.mean(logits), and criterion is a
            # FullSupportBarDistribution whose mean() is
            #     bucket_means = borders[:-1] + bucket_widths/2
            #     bucket_means[0]  = -HalfNormal(w[0]/icdf).mean  + borders[1]
            #     bucket_means[-1] =  HalfNormal(w[-1]/icdf).mean + borders[-2]
            #
            # The subtlety: BarDistribution registers `bucket_widths` as a
            # BUFFER at __init__, and predict_full rescales ONLY `borders`
            #     criterion.borders = criterion.borders * data_std + data_mean
            # leaving bucket_widths in NORMALISED units. So DoPFN's own mean
            # adds a normalised half-width to a raw-unit border, and sets the
            # tail scales from normalised widths. That mixed-unit arithmetic is
            # a bug upstream, but it IS what predict_cate returns and therefore
            # what the reported PEHE describes, so the dumped density has to be
            # checked against the same quantity or the two describe different
            # estimators.
            #
            # Using raw-unit centres for every bucket -- or even raw-unit tail
            # scales -- disagrees with it: a persistent ~1.03 slope (the
            # data_std factor) plus blow-ups to slope 3.2 where tail mass is
            # large. That is what the density_scale_r2 gate was silently
            # discarding, ~25% of dopfn_native realizations on the case studies
            # and the same defect behind the sd_ratio 18-59 rows on ComplexMech
            # and RealCause.
            #
            # `full0['criterion']` is the deepcopy predict_full mutated, so it
            # already carries rescaled borders WITH stale normalised widths --
            # calling its own mean() reproduces predict_cate exactly rather
            # than re-deriving it and risking another mismatch.
            _crit0 = full0['criterion']
            _crit1 = full1['criterion']
            _mean0 = _crit0.mean(torch.from_numpy(logits0)).detach().numpy()
            _mean1 = _crit1.mean(torch.from_numpy(logits1)).detach().numpy()
            # Bucket means on the SAME convention, stored so downstream users
            # of p_y0/p_y1 can reproduce this mean instead of assuming centres.
            _bw_norm = _crit0.bucket_widths.detach().cpu().numpy()
            _centers = edges[:-1] + _bw_norm / 2.0
            _ICDF_HALFNORMAL_HALF = 0.6744897501960817   # HalfNormal(1).icdf(0.5)
            _SQRT_2_OVER_PI = 0.7978845608028654         # E[HalfNormal(1)]
            _centers[0] = edges[1] - (_bw_norm[0] / _ICDF_HALFNORMAL_HALF) * _SQRT_2_OVER_PI
            _centers[-1] = edges[-2] + (_bw_norm[-1] / _ICDF_HALFNORMAL_HALF) * _SQRT_2_OVER_PI
            # Score the DUMPED arrays, not the criterion call. p_y0/p_y1 and
            # bucket_means are what land on disk and what every downstream
            # consumer reads; `_mean0/_mean1` would make this check compare
            # criterion.mean() against predict_cate, which IS criterion.mean()
            # -- true by construction and therefore blind to a broken dump.
            # Going through p @ bucket_means validates the softmax, the stored
            # bucket means and the arrays together.
            _cate_dens = ((p_y1 @ _centers) - (p_y0 @ _centers)).astype(np.float64)
            _ref_max = float(np.max(np.abs((_mean1 - _mean0) - _cate_dens)))
            if _ref_max > 1e-3:
                print(f'  [density][WARN] dumped p @ bucket_means differs from '
                      f'criterion.mean() by {_ref_max:.3g} -- the saved arrays '
                      f'do not reproduce the model\'s own estimator.', flush=True)
            _cp = np.asarray(cate_pred, dtype=np.float64).reshape(-1)
            _den = float(np.dot(_cate_dens, _cate_dens))
            _scale = float(np.dot(_cate_dens, _cp) / _den) if _den > 0 else 1.0
            _resid = _cp - _scale * _cate_dens
            _ss = float(np.dot(_cp - _cp.mean(), _cp - _cp.mean()))
            _r2 = 1.0 - float(np.dot(_resid, _resid)) / _ss if _ss > 0 else float('nan')
            print(f'  [density] cate_pred vs density mean: slope={_scale:.6g} '
                  f'R^2={_r2:.6f}  (slope~1 confirms raw borders)', flush=True)
            if not (_r2 > 0.99):
                print('  [density][WARN] the dumped density mean does not track '
                      'cate_pred on this realization — its calibration numbers '
                      'are suspect. Filter on density_scale_r2.', flush=True)
            dens = dict(
                edges=edges.astype(np.float32),
                p_y0_scaled=p_y0.astype(np.float32),
                p_y1_scaled=p_y1.astype(np.float32),
                y_shift=np.float32(0.0),
                y_scale=np.float32(1.0),          # borders are raw; see above
                density_scale_slope=np.float32(_scale),
                density_scale_r2=np.float32(_r2),
                # DoPFN's own bucket means (rescaled borders + STALE normalised
                # widths, half-normal tail means at the two ends). A consumer
                # that takes 0.5*(edges[:-1]+edges[1:]) does NOT reproduce
                # predict_cate; use these instead.
                bucket_means=_centers.astype(np.float32),
            )
    finally:
        os.chdir(_prev_cwd)

    cate_pred = np.asarray(cate_pred, dtype=np.float32).reshape(-1)
    true_cate = np.asarray(cate_ds.true_cate, dtype=np.float32).reshape(-1)
    pehe = float(np.sqrt(np.mean((cate_pred - true_cate) ** 2)))
    ate_true = float(true_cate.mean())
    ate_hat  = float(cate_pred.mean())
    # Case-study datasets: report err as pure L1. RealCause: relative-error.
    if DATASET not in _REALCAUSE:
        err = abs(ate_hat - ate_true)
    else:
        err = abs(ate_hat - ate_true) / max(abs(ate_true), 0.1)
    row = {'dataset': DATASET, 'realization': r,
           'true_ate': ate_true, 'ate_pred': ate_hat,
           'pehe_raw': pehe, 'err_raw': err}
    if dens is not None:
        row.update(dens)
        row['true_cate_per_query'] = true_cate.astype(np.float32)
    return row


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = _get_dataset(DATASET)
    n_full = _n_tables(ds, DATASET)
    n = n_full if not MAX_REAL else min(n_full, int(MAX_REAL))
    print(f'[bootstrap] native DoPFN  {DATASET}  n={n}', flush=True)
    rows = []; t0 = time.time()
    for r in range(n):
        row = evaluate(r, ds)
        rows.append(row)
        np.savez(os.path.join(OUT, f'r{r:03d}.npz'), **{k: np.array(v) for k, v in row.items()})
        print(f'r={r:03d}  pehe={row["pehe_raw"]:6.3f}  err={row["err_raw"]:5.3f}  '
              f'ate={row["ate_pred"]:+5.2f} vs true {row["true_ate"]:+5.2f}  '
              f'({time.time()-t0:.0f}s)', flush=True)
    def _ms(k):
        v = np.array([r[k] for r in rows]); return v.mean(), v.std(ddof=1)/np.sqrt(len(v))
    print(f'\n══ {DATASET}  native DoPFN  n={len(rows)} ══')
    for k in ('pehe_raw', 'err_raw'):
        m, s = _ms(k); print(f'  {k:10s} = {m:8.3f} ± {s:6.3f}')


if __name__ == '__main__':
    main()
