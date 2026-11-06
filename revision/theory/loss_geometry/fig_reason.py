"""The reason, measured: loss of a prediction that is wrong only in the effect, under the 1D (separate) and the
2D (joint) loss, as the correlation of the two outcomes' noise grows; and the 2D loss predicted from the 1D loss.
Prediction: m0_hat = m0 - 0.5, m1_hat = m1 + 0.5 (effect off by 1, average level right). Noise sd 1, correlation rho.
Losses per unit, without the constant term that does not depend on the prediction (log of the noise sizes):
  1D: (r0^2 + r1^2) / 2          2D: r^T Sigma^-1 r / 2        (r = Y - prediction)
Theory: a correct prediction has loss 1 under both; an effect error adds (1D loss - 1) / (1 - rho) under 2D.
   .venv/bin/python revision/theory/loss_geometry/fig_reason.py OUT_DIR"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]; rng = np.random.default_rng(0); N = 200_000
e = np.array([-0.5, 0.5])
rhos = np.linspace(0, 0.99, 100); L1, L2 = [], []
for r in rhos:
    S = np.array([[1, r], [r, 1]]); z = rng.standard_normal((N, 2)) @ np.linalg.cholesky(S).T
    res = z - e                                         # Y - prediction = noise - error
    L1.append(0.5 * (res ** 2).sum(1).mean())
    L2.append(0.5 * np.einsum("ni,ij,nj->n", res, np.linalg.inv(S), res).mean())
L1, L2 = np.array(L1), np.array(L2)
fine = np.linspace(0, 0.99, 400)
theory = 1 + (L1.mean() - 1) / (1 - fine)
plt.rcParams.update({"font.size": 13})
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(rhos, L1, "-", color="#1f4e9c", lw=3.5, label="1D loss")
ax.plot(rhos, L2, "-", color="#c0392b", lw=3.5, label="2D loss")
ax.plot(fine, theory, "--", color="k", lw=1.8, label="Theory prediction")
ax.set(xlabel="correlation ρ of the two outcomes", ylabel="loss", xlim=(0, 1),
       title="Loss of a prediction whose effect is wrong by 1")
ax.legend(fontsize=13); fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"reason.{ext}"), dpi=150)
for r, a, b in list(zip(rhos, L1, L2))[::11]: print(f"rho {r:.2f}  1D {a:.3f}  2D {b:.3f}  predicted {1 + (a - 1) / (1 - r):.3f}")
