"""OURS — InterventionalPFN + 2D MALC + OT barycenter.

Produces six CATE estimates per (dataset, realization), all from the same
model forward pass:

  ours_mean          : E[τ] under raw p_mat marginals   (no MALC smoothing)
  ours_malc_mean     : E[τ] under MALC-smoothed p(τ)    (raw joint)
  ours_malc_mean_msk : E[τ] under MALC-smoothed masked p(τ) (diag masked)
  ours_malc_mode     : argmax MALC-smoothed p(τ)        (raw joint)
  ours_malc_mode_msk : argmax MALC-smoothed masked p(τ) (diag masked)
  ours_ot_mode       : mode of the W2 barycenter of masked per-query densities
                        — a *population* ATE estimate, not per-query CATE

Diagonal masking suppresses the τ=0 attractor of the joint p(Y_do0, Y_do1) —
see benchmarks/plots/plot_mask_example.py for the intuition.
"""
from __future__ import annotations
import hashlib, os, sys
from multiprocessing import get_context
import numpy as np
import torch


def _to_np(a):
    if isinstance(a, torch.Tensor): return a.numpy()
    return np.asarray(a)


def _pad(arr, L):
    if L is None or L <= 0:                     # per-feature attention backbone: no cap
        return arr
    if arr.shape[1] >= L: return arr[:, :L]
    z = np.zeros((arr.shape[0], L - arr.shape[1]), dtype=np.float32)
    return np.concatenate([arr, z], axis=1)


def _mask_diag(p_mat_np, J, band=1):
    p = p_mat_np.copy()
    for j0 in range(J):
        for j1 in range(max(0, j0 - band), min(J, j0 + band + 1)):
            p[j0, j1] = 0.0
    p /= max(p.sum(), 1e-12)
    return p


# ── MALC parallel-worker plumbing ────────────────────────────────────────────
_GLOBAL = {}


def _init_worker(edges_np, J, bin_width, N_EVAL, MALC_B, MALC_MAX_K, repo, malc_dir):
    """Pool initializer: import MALC once per worker + cache eval grids."""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    if repo not in sys.path: sys.path.insert(0, repo)
    if malc_dir not in sys.path: sys.path.insert(0, malc_dir)
    from losses.BarDistribution2D import fit_malc_inner
    from malc_2d import dmalc_2d
    _GLOBAL['fit'] = fit_malc_inner
    _GLOBAL['dmalc'] = dmalc_2d
    _GLOBAL['edges'] = edges_np
    _GLOBAL['J'] = J
    _GLOBAL['bw'] = bin_width
    _GLOBAL['MALC_B'] = MALC_B
    _GLOBAL['MALC_MAX_K'] = MALC_MAX_K
    xs = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
    ys = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    _GLOBAL['xs'] = xs; _GLOBAL['ys'] = ys
    _GLOBAL['eval_pts'] = np.column_stack([XX.ravel(), YY.ravel()])
    _GLOBAL['dy0'] = xs[1] - xs[0]; _GLOBAL['dy1'] = ys[1] - ys[0]
    _GLOBAL['tau'] = np.linspace(ys[0] - xs[-1], ys[-1] - xs[0], 401)
    _GLOBAL['dtau'] = _GLOBAL['tau'][1] - _GLOBAL['tau'][0]


def _em_tau_from_fit(fit):
    """τ = E[Y1] − E[Y0] via pi-weighted per-component EM means.
    Returns NaN if fit is None or has no valid components."""
    if fit is None:
        return float('nan')
    mu_list = [c.mu_hat for c in fit.fits if c is not None]
    if not mu_list:
        return float('nan')
    mus = np.stack(mu_list)
    weights = np.asarray(fit.pi, dtype=np.float64)[:len(mus)]
    s = weights.sum()
    if s <= 0:
        return float('nan')
    weights = weights / s
    mu_y0 = float((weights * mus[:, 0]).sum())
    mu_y1 = float((weights * mus[:, 1]).sum())
    return mu_y1 - mu_y0


def _fit_and_marginalize(p_mat_np, seed, return_fit=False):
    """Fit 2D MALC to p_mat, then marginalize to a 1D p(τ) on the tau grid.
    If return_fit=True, also returns the underlying MALC2DFit so the caller
    can extract EM-adjusted mu_hats."""
    fit = _GLOBAL['fit'](p_mat_np.T, _GLOBAL['edges'], _GLOBAL['edges'],
                          B_fit=_GLOBAL['MALC_B'], B_select=_GLOBAL['MALC_B'],
                          max_K=_GLOBAL['MALC_MAX_K'], seed=seed, parallel=False)
    density = _GLOBAL['dmalc'](fit, _GLOBAL['eval_pts']).reshape(len(_GLOBAL['xs']), len(_GLOBAL['ys']))
    tau = _GLOBAL['tau']
    xs = _GLOBAL['xs']; ys = _GLOBAL['ys']; dy0 = _GLOBAL['dy0']; dy1 = _GLOBAL['dy1']
    out = np.zeros_like(tau)
    for k, t in enumerate(tau):
        y1 = xs + t; v = (y1 >= ys[0]) & (y1 <= ys[-1])
        if not np.any(v): continue
        # Match l2_ihdp/methods_densities.py (fix in 3cf191d): searchsorted(side='left')
        # already returns the correct column index k for xs[v][i] == xs[k].
        # The earlier "- 1" was a genuine off-by-one that biased MALC-mean by
        # one bin on the scaled y-axis.
        col = np.clip(np.searchsorted(xs, xs[v]), 0, len(xs) - 1)
        rf = (y1[v] - ys[0]) / dy1
        rlo = np.clip(np.floor(rf).astype(int), 0, len(ys) - 2)
        rhi = rlo + 1; whi = rf - rlo; wlo = 1.0 - whi
        f = wlo * density[rlo, col] + whi * density[rhi, col]
        out[k] = f.sum() * dy0
    s = out.sum() * _GLOBAL['dtau']
    if s > 0: out /= s
    if return_fit:
        return out, fit
    return out


def _worker_one_query(args):
    """Per-query task: fit MALC on raw and masked p_mat, return p(τ)'s + EM τ's.

    Returns (i, p_raw, p_msk, em_mix_tau, em_k1_tau) where the last two are
    scaled-axis EM-mean CATE point estimates: pi-weighted mixture-fit mean and
    forced-K=1 fit mean respectively. NaN if the corresponding fit failed.
    """
    i, p_mat_np = args
    from losses.BarDistribution2D import fit_malc_inner
    p_masked = _mask_diag(p_mat_np, _GLOBAL['J'], band=1)
    seed_raw = int(hashlib.md5(f"q{i}r".encode()).hexdigest()[:8], 16) % (10**8)
    seed_msk = int(hashlib.md5(f"q{i}m".encode()).hexdigest()[:8], 16) % (10**8)
    # Mixture fit (up to _GLOBAL['MALC_MAX_K']) — used for p(τ) via density
    # integration AND for the EM-K-selection mean.
    p_raw, fit_mix = _fit_and_marginalize(p_mat_np, seed_raw, return_fit=True)
    em_mix_tau = _em_tau_from_fit(fit_mix)
    p_msk = _fit_and_marginalize(p_masked, seed_msk)
    # Separate forced-K=1 fit for the EM-K1 mean-adjusted estimate.
    em_k1_tau = float('nan')
    try:
        fit_k1 = fit_malc_inner(p_mat_np.T, _GLOBAL['edges'], _GLOBAL['edges'],
                                 B_fit=_GLOBAL['MALC_B'], B_select=_GLOBAL['MALC_B'],
                                 max_K=1, seed=seed_raw, parallel=False)
        em_k1_tau = _em_tau_from_fit(fit_k1)
    except Exception:
        pass
    return i, p_raw, p_msk, em_mix_tau, em_k1_tau


def _raw_p_tau_from_p_mat(p_mat_np, J, bw, tau):
    """p(τ) from raw p_mat via marginal convolution (independence assumption).
    Matches Do-PFN's recipe. Interpolated onto the given tau grid.
    Callable from the main process (no _GLOBAL dependency)."""
    dtau = tau[1] - tau[0]
    p_y0 = p_mat_np.sum(axis=1); p_y1 = p_mat_np.sum(axis=0)
    s0 = p_y0.sum(); s1 = p_y1.sum()
    if s0 <= 0 or s1 <= 0:
        return np.zeros_like(tau)
    p_y0 = p_y0 / s0; p_y1 = p_y1 / s1
    dens_native = np.correlate(p_y1, p_y0, mode='full') / bw
    tau_native = np.arange(-(J - 1), J) * bw
    out = np.interp(tau, tau_native, dens_native, left=0.0, right=0.0)
    s = out.sum() * dtau
    if s > 0: out /= s
    return out


# ── main pipeline entry point ────────────────────────────────────────────────
def ours_pipeline(cate_dataset, our_model, edges_np, J, bin_width, NUM_FEATURES,
                   centers, args, wasserstein_barycenter_1d):
    """Run OURS on one cate_dataset, return dict of six CATE variants.

    Parameters
    ----------
    cate_dataset            CATE_Dataset (X_train, t_train, y_train, X_test)
    our_model               loaded InterventionalPFN in eval mode
    edges_np, J, bin_width, centers   grid metadata from the checkpoint
    NUM_FEATURES            feature-slot count the model was trained with
    args                    argparse namespace (needs .workers, .n_eval, .malc_B,
                             .malc_max_K, .repo)
    wasserstein_barycenter_1d  callable — MALC/Optimal_Transport/ot_barycenter.py
    """
    from losses.BarDistribution2D import unpack_pred

    Xtr = _to_np(cate_dataset.X_train).astype(np.float32)
    tt  = _to_np(cate_dataset.t_train).astype(np.float32).reshape(-1, 1)
    yt  = _to_np(cate_dataset.y_train).astype(np.float32).reshape(-1, 1)
    Xte = _to_np(cate_dataset.X_test).astype(np.float32)

    Xtr_p = _pad(Xtr, NUM_FEATURES); Xte_p = _pad(Xte, NUM_FEATURES)

    # For the DoPFN backbone (NUM_FEATURES <= 0), match Do-PFN's own inference
    # preprocessing chain (DoPFNRegressor uses PreprocessorConfig("power") from
    # TabPFN internals): apply sklearn's Yeo-Johnson PowerTransformer to X
    # BEFORE the model. Fitted on training-context X only, applied to both
    # context and test. This reshapes skewed real-world features (income, wage,
    # etc.) toward Gaussian — the SCM distribution the model was trained on.
    # Skip for the InterventionalPFN path (NUM_FEATURES > 0) since that model
    # was not evaluated under Do-PFN's inference recipe.
    if NUM_FEATURES is not None and NUM_FEATURES <= 0:
        try:
            from sklearn.preprocessing import PowerTransformer
            pt = PowerTransformer(method='yeo-johnson', standardize=True)
            pt.fit(Xtr_p)
            Xtr_p = pt.transform(Xtr_p).astype(np.float32)
            Xte_p = pt.transform(Xte_p).astype(np.float32)
        except Exception as _e:
            print(f'[warn] PowerTransformer failed ({type(_e).__name__}: {_e}); '
                  f'falling back to z-score only', flush=True)

    mu = Xtr_p.mean(0, keepdims=True); sd = Xtr_p.std(0, keepdims=True); sd[sd < 1e-6] = 1.0
    Xtr_s = (Xtr_p - mu) / sd; Xte_s = (Xte_p - mu) / sd
    y_min = float(yt.min()); y_max = float(yt.max()); y_rng = max(y_max - y_min, 1e-8)
    yt_s = 2 * (yt - y_min) / y_rng - 1.0

    # ── Hierarchical clustering — verbatim from UWYK's
    #    PreprocessingGraphConditionedPFN._hierarchical_cluster + _assign_to_clusters
    # so OURS and UWYK use identical partitioning when N_train exceeds
    # MAX_N_TRAIN. Algorithm:
    #   1. k_initial = N // max_n_train + 1
    #   2. KMeans on X_train (random_state=42, n_init=10)
    #   3. For each oversized cluster: split with KMeans(k=2). If still
    #      oversized, random-permutation-split into ceil(size/max_n_train) parts.
    #   4. Test assignment: centroid = mean of assigned training points (NOT
    #      the KMeans centers); nearest-centroid by Euclidean distance.
    import gc as _gc
    MAX_N_TRAIN   = int(getattr(args, 'ours_max_n_train', 1000))
    RANDOM_STATE  = 42
    N_train = int(Xtr_s.shape[0])

    def _uwyk_hierarchical_cluster(X_train, max_n_train, random_state=42):
        from sklearn.cluster import KMeans
        N = X_train.shape[0]
        k_initial = N // max_n_train + 1
        rng = np.random.RandomState(random_state)
        km = KMeans(n_clusters=k_initial, random_state=random_state, n_init=10)
        initial_labels = km.fit_predict(X_train)
        cluster_assignments = np.copy(initial_labels)
        next_cluster_id = k_initial
        for cluster_id in range(k_initial):
            cluster_mask = initial_labels == cluster_id
            cluster_size = int(cluster_mask.sum())
            if cluster_size <= max_n_train:
                continue
            X_cluster = X_train[cluster_mask]
            km_split = KMeans(n_clusters=2, random_state=random_state, n_init=10)
            sub_labels = km_split.fit_predict(X_cluster)
            sub_sizes = [int((sub_labels == 0).sum()), int((sub_labels == 1).sum())]
            cluster_indices = np.where(cluster_mask)[0]
            if max(sub_sizes) <= max_n_train:
                cluster_assignments[cluster_indices[sub_labels == 0]] = cluster_id
                cluster_assignments[cluster_indices[sub_labels == 1]] = next_cluster_id
                next_cluster_id += 1
            else:
                n_sub = (cluster_size + max_n_train - 1) // max_n_train
                shuffled = rng.permutation(cluster_indices)
                for i, idx in enumerate(shuffled):
                    subcluster_id = cluster_id if (i % n_sub) == 0 else next_cluster_id + (i % n_sub) - 1
                    cluster_assignments[idx] = subcluster_id
                next_cluster_id += n_sub - 1
        return cluster_assignments

    def _uwyk_assign_to_clusters(X_test, X_train, cluster_assignments):
        unique_clusters = np.unique(cluster_assignments)
        centroids = np.zeros((len(unique_clusters), X_train.shape[1]))
        for i, cid in enumerate(unique_clusters):
            centroids[i] = X_train[cluster_assignments == cid].mean(axis=0)
        M = X_test.shape[0]
        test_assignments = np.zeros(M, dtype=int)
        for i in range(M):
            distances = np.linalg.norm(centroids - X_test[i], axis=1)
            test_assignments[i] = unique_clusters[int(np.argmin(distances))]
        return test_assignments

    if N_train > MAX_N_TRAIN:
        train_labels = _uwyk_hierarchical_cluster(Xtr_s, MAX_N_TRAIN, RANDOM_STATE)
        test_labels  = _uwyk_assign_to_clusters(Xte_s, Xtr_s, train_labels)
        cluster_blocks = []
        for cid in np.unique(train_labels):
            tr_mask = train_labels == cid
            te_mask = test_labels  == cid
            if te_mask.sum() == 0:
                continue
            cluster_blocks.append((int(cid), np.where(tr_mask)[0], np.where(te_mask)[0]))
        print(f"[ours_pipeline] N_train={N_train} > MAX_N_TRAIN={MAX_N_TRAIN} → "
              f"{len(cluster_blocks)} clusters (sizes: {[b[1].size for b in cluster_blocks]})",
              flush=True)
    else:
        cluster_blocks = [(0, np.arange(N_train), np.arange(int(Xte_s.shape[0])))]

    # Chunk over test queries inside each cluster + force gc between chunks.
    M = int(Xte_s.shape[0])
    chunk = int(getattr(args, 'ours_query_chunk', 20))
    est_mean = np.zeros(M)
    p_mats = np.zeros((M, J, J), dtype=np.float32)
    for cid, tr_idx, te_idx in cluster_blocks:
        Xtr_t = torch.from_numpy(Xtr_s[tr_idx]).unsqueeze(0)
        tt_t  = torch.from_numpy(tt[tr_idx]).unsqueeze(0)
        yt_t  = torch.from_numpy(yt_s[tr_idx]).unsqueeze(0)
        for start in range(0, len(te_idx), chunk):
            stop = min(start + chunk, len(te_idx))
            queries = te_idx[start:stop]
            Xte_chunk = torch.from_numpy(Xte_s[queries]).unsqueeze(0)
            with torch.no_grad():
                pred_chunk = our_model(Xtr_t, tt_t, yt_t, Xte_chunk)['predictions'][0]
            for j in range(pred_chunk.shape[0]):
                p_mat, *_ = unpack_pred(pred_chunk[j], J, bin_width)
                p_np = p_mat.detach().cpu().numpy().astype(np.float32)
                i = int(queries[j])
                p_mats[i] = p_np
                E_y0 = (centers[:, None] * p_np).sum()
                E_y1 = (centers[None, :] * p_np).sum()
                est_mean[i] = float(E_y1 - E_y0)
            del pred_chunk, Xte_chunk
            _gc.collect()
        del Xtr_t, tt_t, yt_t
        _gc.collect()
    # Release scaled arrays before spawning MALC workers.
    del Xtr_s, Xte_s, yt_s
    _gc.collect()

    worker_args = [(i, p_mats[i]) for i in range(M)]
    p_taus_raw = np.zeros((M, 401)); p_taus_msk = np.zeros((M, 401))
    em_mix_scaled = np.full(M, np.nan, dtype=np.float64)
    em_k1_scaled  = np.full(M, np.nan, dtype=np.float64)
    if args.workers > 1:
        ctx = get_context('spawn')
        with ctx.Pool(processes=args.workers, initializer=_init_worker,
                      initargs=(edges_np, J, bin_width, args.n_eval,
                                args.malc_B, args.malc_max_K,
                                args.repo, os.path.join(args.repo, 'MALC'))) as pool:
            for (i, pr, pm, emm, emk1) in pool.imap_unordered(
                    _worker_one_query, worker_args, chunksize=1):
                p_taus_raw[i] = pr; p_taus_msk[i] = pm
                em_mix_scaled[i] = emm
                em_k1_scaled[i]  = emk1
    else:
        _init_worker(edges_np, J, bin_width, args.n_eval, args.malc_B, args.malc_max_K,
                      args.repo, os.path.join(args.repo, 'MALC'))
        for a in worker_args:
            i, pr, pm, emm, emk1 = _worker_one_query(a)
            p_taus_raw[i] = pr; p_taus_msk[i] = pm
            em_mix_scaled[i] = emm
            em_k1_scaled[i]  = emk1

    tau = np.linspace(edges_np[0] - edges_np[-1], edges_np[-1] - edges_np[0], 401)
    dtau_ = tau[1] - tau[0]
    est_malc_mode     = tau[p_taus_raw.argmax(axis=1)]
    est_malc_mode_msk = tau[p_taus_msk.argmax(axis=1)]
    est_malc_mean     = (tau[None, :] * p_taus_raw).sum(axis=1) * dtau_
    est_malc_mean_msk = (tau[None, :] * p_taus_msk).sum(axis=1) * dtau_

    scale = y_rng / 2.0
    ours_mean          = est_mean          * scale
    ours_malc_mean     = est_malc_mean     * scale
    ours_malc_mean_msk = est_malc_mean_msk * scale
    ours_malc_mode     = est_malc_mode     * scale
    ours_malc_mode_msk = est_malc_mode_msk * scale
    ours_em_mix_mean   = em_mix_scaled     * scale        # EM-K-selection
    ours_em_k1_mean    = em_k1_scaled      * scale        # EM-K1

    # OT-mode + OT-mean: population-level ATE from W2 barycenter of masked
    # per-query densities. Mode = argmax, Mean = expectation under barycenter.
    ate_bary_scaled = wasserstein_barycenter_1d(p_taus_msk, tau)
    tau_raw = tau * scale
    ate_bary_raw = ate_bary_scaled / scale
    ate_ot_mode_scalar = float(tau_raw[ate_bary_raw.argmax()])
    d_tau_raw = tau_raw[1] - tau_raw[0]
    bary_norm = ate_bary_raw / max(ate_bary_raw.sum() * d_tau_raw, 1e-12)
    ate_ot_mean_scalar = float((tau_raw * bary_norm).sum() * d_tau_raw)

    # RAW-OT-mean: W2 barycenter of per-query p(τ) derived from raw p_mat
    # marginals via convolution (no MALC smoothing). Ablates MALC's role in
    # the population ATE point estimate. Wrapped so a failure here doesn't
    # throw away all the MALC results computed above.
    try:
        p_taus_raw_bin = np.stack([
            _raw_p_tau_from_p_mat(p_mats[i], J, bin_width, tau) for i in range(M)])
        ate_bary_raw_scaled = wasserstein_barycenter_1d(p_taus_raw_bin, tau)
        ate_bary_raw_raw = ate_bary_raw_scaled / scale
        bary_raw_norm = ate_bary_raw_raw / max(ate_bary_raw_raw.sum() * d_tau_raw, 1e-12)
        ate_ot_mean_raw_scalar = float((tau_raw * bary_raw_norm).sum() * d_tau_raw)
    except Exception as e:
        print(f'[warn] raw-OT-mean failed ({type(e).__name__}: {e}); '
              f'setting ate_ours_ot_mean_raw=NaN', flush=True)
        ate_ot_mean_raw_scalar = float('nan')

    return dict(
        ours_mean          = ours_mean,
        ours_malc_mean     = ours_malc_mean,
        ours_malc_mean_msk = ours_malc_mean_msk,
        ours_malc_mode     = ours_malc_mode,
        ours_malc_mode_msk = ours_malc_mode_msk,
        ours_em_mix_mean   = ours_em_mix_mean,         # EM K-selection (per-query CATE)
        ours_em_k1_mean    = ours_em_k1_mean,          # EM K=1        (per-query CATE)
        ours_ot_mode_ate   = ate_ot_mode_scalar,       # population ATE, not per-query
        ours_ot_mean_ate   = ate_ot_mean_scalar,       # population ATE, MALC-smoothed
        ours_ot_mean_ate_raw = ate_ot_mean_raw_scalar, # population ATE, raw p_mat (no MALC)
    )
