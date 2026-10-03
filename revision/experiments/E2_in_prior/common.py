"""Write one world in the ComplexMech npz format the paper's harnesses read, plus its truths."""
import os
import numpy as np

N_CTX, N_Q, SHARD = 1000, 100, 100             # observational units, query units, worlds per shard


def write_world(root, r, X_ctx, T_ctx, Y_ctx, X_q, truths):
    """root/shard<k>/complexmech/5node/path_TY/hide_0.0/r###.npz  (harness input, k = r // SHARD)
       root/truths/r###.npz                                       (every truth, scored later)
    true_cate in the harness input is delta_shared; the scorer reads every truth from truths/."""
    f = lambda a: np.asarray(a, dtype=np.float32)
    d = os.path.join(root, f"shard{r // SHARD}", "complexmech", "5node", "path_TY", "hide_0.0")
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(root, "truths"), exist_ok=True)
    np.savez(os.path.join(d, f"r{r:03d}.npz"),
             X_train=f(X_ctx), T_train=f(T_ctx).reshape(-1), Y_train=f(Y_ctx).reshape(-1),
             X_test=f(X_q), true_cate=f(truths["delta_shared"]).reshape(-1),
             n_real_features=np.int64(X_ctx.shape[1]))
    np.savez(os.path.join(root, "truths", f"r{r:03d}.npz"), **{k: f(v).reshape(-1) for k, v in truths.items()})
