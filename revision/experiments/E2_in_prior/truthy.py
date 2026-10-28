"""Truth-only mode of the gen_*.py generators (E2_TRUTHY=1): instead of writing a dataset, check that the redrawn
world reproduces the saved ITE truths of dataset r and save its query units' true outcomes to truths_y/r###.npz.
Keys: y0, y1 (the prior's own pair, shared noise), y0_ind, y1_ind (the independent-noise pair), plus any extras."""
import os
import numpy as np

ON = os.environ.get("E2_TRUTHY") == "1"


def done(out, r):
    return os.path.exists(os.path.join(out, "truths_y", f"r{r:03d}.npz"))


def save(out, r, y0, y1, y0_ind, y1_ind, **extra):
    f = lambda a: np.asarray(a, dtype=np.float64).reshape(-1)
    y0, y1, y0_ind, y1_ind = f(y0), f(y1), f(y0_ind), f(y1_ind)
    t = np.load(os.path.join(out, "truths", f"r{r:03d}.npz"))
    for name, mine in (("delta_shared", y1 - y0), ("delta_indep", y1_ind - y0_ind)):
        if name in t.files and not np.allclose(mine.astype(np.float32), t[name], rtol=1e-4, atol=1e-5):
            raise SystemExit(f"{out} r={r}: redraw does not reproduce the saved {name} -- stopping")
    os.makedirs(os.path.join(out, "truths_y"), exist_ok=True)
    np.savez(os.path.join(out, "truths_y", f"r{r:03d}.npz"), y0=y0.astype(np.float32), y1=y1.astype(np.float32),
             y0_ind=y0_ind.astype(np.float32), y1_ind=y1_ind.astype(np.float32),
             **{k: f(v).astype(np.float32) for k, v in extra.items()})
