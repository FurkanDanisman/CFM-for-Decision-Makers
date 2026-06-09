"""Fast 2D log-concave MLE via direct CLARABEL call (no CVXPY DSL overhead).

Same problem as log_concave_2d.mlelcd_2d, but builds the sparse conic program
matrices directly and calls CLARABEL.solve() — avoiding CVXPY's per-call
canonicalization which dominates wall-time on small/medium exp-cone problems.

Standard CLARABEL form:
    minimize    (1/2) x' P x + q' x
    subject to  A x + s = b,   s ∈ K

Our variables x = [y (n); t (m)] where y_i = log f(x_i) at the n synthetic
points and t_q is an epigraph variable per (triangle, quadrature) pair.

Objective q = [-w (n); area_T * w_q (m,)].  (P = 0.)

Constraints:
    For each (T, q): want t_q >= exp(bary_q · y_T). Represent as the slack
    triple (s1, s2, s3) ∈ ExponentialCone defined by
        s1 = bary_q · y_T   (linear in y)
        s2 = 1
        s3 = t_q
    Standard form: A x + s = b, so we put -bary_q in the y columns
    of row s1 (b_s1 = 0); zeros in row s2 (b_s2 = 1); -e_t in row s3
    (b_s3 = 0).
    Note the convention here:  the cone constraint is  y * exp(x/y) <= z,
    with the triple (s1, s2, s3) = (x, y, z) = (bary_q·y_T, 1, t_q),
    which yields  1 * exp(bary_q · y_T / 1) <= t_q  ⇔  exp(...) <= t_q. ✓

    For each concavity edge: linear (c · y) >= 0. As slack:
        s = -(c · y) ∈ NonnegativeCone (i.e., -c·y >= 0 ⇔ c·y <= 0)
    Wait, we want bcoef · y_T - y_apex >= 0 (the concavity constraint).
    As slack s = b - Ax, we want s = bcoef·y_T - y_apex (in NonnegativeCone).
    So put -bcoef in y_T columns of that row, +1 in y_apex column, b = 0.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import clarabel
import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay


# --- Dunavant order-5 quadrature on a triangle ---


def _dunavant5():
    a1 = (6.0 - np.sqrt(15.0)) / 21.0
    a2 = (6.0 + np.sqrt(15.0)) / 21.0
    w1 = 9.0 / 40.0
    w2 = (155.0 - np.sqrt(15.0)) / 1200.0
    w3 = (155.0 + np.sqrt(15.0)) / 1200.0
    bary = np.array(
        [
            [1 / 3, 1 / 3, 1 / 3],
            [a1, a1, 1 - 2 * a1],
            [a1, 1 - 2 * a1, a1],
            [1 - 2 * a1, a1, a1],
            [a2, a2, 1 - 2 * a2],
            [a2, 1 - 2 * a2, a2],
            [1 - 2 * a2, a2, a2],
        ]
    )
    w = np.array([w1, w2, w2, w2, w3, w3, w3])
    return bary, w


_QUAD_BARY, _QUAD_W = _dunavant5()


@dataclass
class LogConcaveDensity2D:
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    tri: Delaunay
    hull_edges: np.ndarray
    midpoint: np.ndarray
    outnorm: np.ndarray
    outdist: np.ndarray
    loglik: float
    integral: float


def mlelcd_2d_fast(
    x: np.ndarray,
    w: np.ndarray | None = None,
    verbose: bool = False,
    jitter: float = 1e-10,
    seed: int | None = None,
    tol_gap: float = 1e-8,
    tol_feas: float = 1e-8,
    max_iter: int = 200,
) -> LogConcaveDensity2D:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("x must be (n, 2)")
    n = x.shape[0]
    if n < 3:
        raise ValueError("need at least 3 points for 2D MLE")

    if w is None:
        w = np.full(n, 1.0 / n)
    else:
        w = np.asarray(w, dtype=float)
        w = w / w.sum()

    if jitter > 0:
        rng = np.random.default_rng(seed)
        x_in = x + jitter * rng.standard_normal(x.shape)
    else:
        x_in = x

    tri = Delaunay(x_in)
    simplices = tri.simplices  # (n_tri, 3)
    n_tri = simplices.shape[0]

    v0 = x_in[simplices[:, 0]]
    v1 = x_in[simplices[:, 1]]
    v2 = x_in[simplices[:, 2]]
    cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    areas = 0.5 * np.abs(cross)

    nq = len(_QUAD_W)
    m = n_tri * nq  # epigraph variables t_q

    # ---- Objective q ----
    # Variables order: [y (n); t (m)]
    q_vec = np.concatenate([-w, np.repeat(areas, nq) * np.tile(_QUAD_W, n_tri)])

    # ---- Constraints ----
    # 1. Concavity (NonnegativeCone)
    edges_rows = []  # list of (col_idx_list, val_list)
    neighbors = tri.neighbors  # (n_tri, 3)
    for ti in range(n_tri):
        T = simplices[ti]
        for k in range(3):
            nb = neighbors[ti, k]
            if nb == -1 or nb < ti:
                continue
            T2 = simplices[nb]
            apex_T2 = int([v for v in T2 if v not in (T[0], T[1], T[2])][0])
            v_T1 = x_in[T]
            A_bary = np.vstack([v_T1.T, np.ones(3)])
            rhs = np.concatenate([x_in[apex_T2], [1.0]])
            bcoef = np.linalg.solve(A_bary, rhs)
            # constraint: bcoef·y_T - y_apex >= 0
            # slack s = b - Ax = bcoef·y_T - y_apex in NonnegativeCone
            # → A row has -bcoef in y_T columns and +1 in y_apex column, b = 0
            cols = [int(T[0]), int(T[1]), int(T[2]), apex_T2]
            vals = [-float(bcoef[0]), -float(bcoef[1]), -float(bcoef[2]), 1.0]
            # de-duplicate if apex_T2 happens to equal one of T's vertices
            # (shouldn't happen in a Delaunay triangulation, but defensive)
            edges_rows.append((cols, vals))

    n_nonneg = len(edges_rows)

    # 2. Exp cone (3 rows per (T, q))
    # We build A as COO triplets, then csc.
    rows = []
    cols = []
    vals = []
    b_list = []

    # Concavity rows first (rows 0 .. n_nonneg-1)
    for r, (cc, vv) in enumerate(edges_rows):
        for c, v in zip(cc, vv):
            rows.append(r)
            cols.append(c)
            vals.append(v)
        b_list.append(0.0)

    # Exp cone rows (3 per quadrature point per triangle)
    row_offset = n_nonneg
    n_total_vars = n + m

    for ti in range(n_tri):
        T = simplices[ti]
        for qi in range(nq):
            bary = _QUAD_BARY[qi]
            r1 = row_offset  # s1 = bary·y_T
            r2 = row_offset + 1  # s2 = 1
            r3 = row_offset + 2  # s3 = t_q
            row_offset += 3

            # s1 row: A puts -bary in y_T cols → s1 = 0 - (-bary·y_T) = bary·y_T ✓
            for j, b_ in zip(T, bary):
                rows.append(r1)
                cols.append(int(j))
                vals.append(-float(b_))
            b_list.append(0.0)

            # s2 row: A is all zero → s2 = 1 - 0 = 1 ✓
            b_list.append(1.0)

            # s3 row: A puts -1 in t_q col → s3 = 0 - (-t_q) = t_q ✓
            rows.append(r3)
            cols.append(n + ti * nq + qi)
            vals.append(-1.0)
            b_list.append(0.0)

    n_rows = row_offset
    A_mat = sp.csc_matrix((vals, (rows, cols)), shape=(n_rows, n_total_vars))
    b_vec = np.array(b_list, dtype=float)
    P_mat = sp.csc_matrix((n_total_vars, n_total_vars))

    cones: list = [clarabel.NonnegativeConeT(n_nonneg)]
    for _ in range(m):
        cones.append(clarabel.ExponentialConeT())

    settings = clarabel.DefaultSettings()
    settings.verbose = verbose
    settings.tol_gap_abs = tol_gap
    settings.tol_gap_rel = tol_gap
    settings.tol_feas = tol_feas
    settings.max_iter = max_iter

    solver = clarabel.DefaultSolver(P_mat, q_vec, A_mat, b_vec, cones, settings)
    sol = solver.solve()

    status_str = str(sol.status)
    if "Solved" not in status_str:
        raise RuntimeError(f"clarabel failed: status={status_str}")

    x_opt = np.asarray(sol.x)
    y_opt = x_opt[:n]

    integral_val = 0.0
    for qi in range(nq):
        bary = _QUAD_BARY[qi]
        wq = _QUAD_W[qi]
        affine = bary[0] * y_opt[simplices[:, 0]] + bary[1] * y_opt[simplices[:, 1]] + bary[2] * y_opt[simplices[:, 2]]
        integral_val += float(np.sum(wq * areas * np.exp(affine)))

    loglik = float(np.sum(w * y_opt))

    hull_edges = tri.convex_hull
    midpoint = x_in.mean(axis=0)
    n_facets = len(hull_edges)
    outnorm = np.zeros((n_facets, 2))
    outdist = np.zeros(n_facets)
    for i, edge in enumerate(hull_edges):
        va = x_in[edge[0]]
        vb = x_in[edge[1]]
        ev = vb - va
        normal = np.array([ev[1], -ev[0]])
        if normal @ (va - midpoint) < 0:
            normal = -normal
        normal = normal / np.linalg.norm(normal)
        outnorm[i] = normal
        outdist[i] = float(normal @ (va - midpoint))

    return LogConcaveDensity2D(
        x=x_in,
        y=y_opt,
        w=w,
        tri=tri,
        hull_edges=hull_edges,
        midpoint=midpoint,
        outnorm=outnorm,
        outdist=outdist,
        loglik=loglik,
        integral=integral_val,
    )


def dlcd_2d_fast(query_pts: np.ndarray, density: LogConcaveDensity2D, eps: float = 1e-10) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(query_pts, dtype=float))
    if pts.shape[1] != 2:
        raise ValueError("query_pts must be (m, 2)")

    centered = pts - density.midpoint
    facet_vals = centered @ density.outnorm.T - density.outdist
    is_out = facet_vals.max(axis=1) > eps

    simplex_idx = density.tri.find_simplex(pts)

    log_dens = np.full(pts.shape[0], -np.inf)
    inside = (~is_out) & (simplex_idx >= 0)
    if np.any(inside):
        inside_idx = np.where(inside)[0]
        sidx = simplex_idx[inside_idx]
        T = density.tri.simplices[sidx]
        transform = density.tri.transform[sidx]
        diff = pts[inside_idx] - transform[:, 2, :]
        b01 = np.einsum("ijk,ik->ij", transform[:, :2, :], diff)
        b2 = 1.0 - b01.sum(axis=1)
        bary = np.column_stack([b01, b2])
        y_T = density.y[T]
        log_dens[inside_idx] = np.sum(bary * y_T, axis=1)

    return np.exp(log_dens)
