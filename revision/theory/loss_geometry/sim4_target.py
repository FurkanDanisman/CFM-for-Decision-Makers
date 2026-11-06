"""Toy for the CausalPFN-C case: the baseline learns the noiseless expectations, the 2D model the noisy outcomes.

Data as in sim.py: x ~ U[-2, 2], m0(x) = sin(2x) + 0.5x, tau(x) = 0.5 sin(3x), m1 = m0 + tau.
Noise as in CausalPFN-C's 2D prior: ONE draw eta ~ N(0, 1) shared by both arms (eta1 := eta0), so rho = 1 and
Y1 - Y0 = tau exactly. Same network (shared trunk, outputs (m0_hat, m1_hat)), same data stream, same initialisation.
  1D on m_t     : 1/2 (r0^2 + r1^2) with labels (m0, m1)            (CausalPFN-C 1D / matched label: E_y0, E_y1)
  2D, rho = c   : 1/2 r^T Sigma_c^{-1} r with labels (Y0, Y1)        (joint Gaussian loss, correlation c in the loss)
c = 0 is a 2D model that does not use the dependence; c -> 1 one that uses it fully (Sigma_c = [[1, c], [c, 1]]).
Reported at the last step: RMSE of tau_hat (PEHE), of m0_hat and of m1_hat on a grid, mean (SE) over seeds.
   .venv/bin/python revision/theory/loss_geometry/sim4_target.py OUT_DIR
"""
import os, sys
import numpy as np
import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)
SEEDS = range(10)
STEPS, BATCH = 3000, 256
WIDTHS = {"wide": 64, "narrow": 6}
MODELS = [("1D on m_t", None), ("2D, rho in loss 0", 0.0), ("2D, rho in loss 0.5", 0.5),
          ("2D, rho in loss 0.9", 0.9), ("2D, rho in loss 0.99", 0.99)]
torch.set_num_threads(1)

m0 = lambda x: torch.sin(2 * x) + 0.5 * x
tau = lambda x: 0.5 * torch.sin(3 * x)
XG = torch.linspace(-2, 2, 401).unsqueeze(1)

def net(width):
    return torch.nn.Sequential(torch.nn.Linear(1, width), torch.nn.Tanh(), torch.nn.Linear(width, width),
                               torch.nn.Tanh(), torch.nn.Linear(width, 2))

def run(c, seed, width):
    g = torch.Generator().manual_seed(1000 + seed)
    torch.manual_seed(seed)
    f = net(width); opt = torch.optim.Adam(f.parameters(), lr=1e-3)
    Si = None if c is None else torch.linalg.inv(torch.tensor([[1.0, c], [c, 1.0]]))
    for _ in range(STEPS):
        x = torch.rand(BATCH, 1, generator=g) * 4 - 2
        eta = torch.randn(BATCH, 1, generator=g)
        m = torch.cat([m0(x), m0(x) + tau(x)], 1)
        if c is None:
            r = f(x) - m; l = 0.5 * (r ** 2).sum(1).mean()
        else:
            r = f(x) - (m + eta); l = 0.5 * ((r @ Si) * r).sum(1).mean()
        opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        p = f(XG); e = lambda a, b: (a - b).pow(2).mean().sqrt().item()
        return e(p[:, 1] - p[:, 0], tau(XG)[:, 0]), e(p[:, 0], m0(XG)[:, 0]), e(p[:, 1], (m0(XG) + tau(XG))[:, 0])

lines = ["Toy, CausalPFN-C case: shared noise (rho = 1); 1D learns m_t, 2D learns Y_t with correlation c in its loss",
         "RMSE on a grid after %d steps, mean (SE) over %d seeds; ratio = 2D / 1D on m_t" % (STEPS, len(SEEDS))]
for wn, w in WIDTHS.items():
    R = {name: np.array([run(c, s, w) for s in SEEDS]) for name, c in MODELS}
    b = R["1D on m_t"].mean(0)
    lines += ["", f"network: {wn} (width {w})",
              f"   {'model':22s} {'PEHE (tau)':>17s} {'m0':>17s} {'m1':>17s}   ratio tau   ratio m0   ratio m1"]
    for name, _ in MODELS:
        A = R[name]; mu, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A))
        lines.append(f"   {name:22s} " + " ".join(f"{mu[k]:8.4f} ({se[k]:.4f})" for k in range(3)) +
                     "   " + "  ".join(f"{mu[k] / b[k]:9.2f}" for k in range(3)))
    print("\n".join(lines[-len(MODELS) - 2:]), flush=True)
open(os.path.join(OUT, "sim4_summary.txt"), "w").write("\n".join(lines) + "\n")
print("wrote", os.path.join(OUT, "sim4_summary.txt"))
