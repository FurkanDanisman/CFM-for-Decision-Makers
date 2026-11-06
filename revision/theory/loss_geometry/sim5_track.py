"""One toy, tracked over training: does the joint loss give a lower PEHE when the correlation VARIES with x and the
model has to LEARN it (nothing is given to it)?

Data (fresh draws every step): x ~ U[-2, 2]; true outcomes m0(x) = sin(2x) + 0.5x, m1(x) = m0(x) + tau(x),
tau(x) = 0.5 sin(3x); observed Y0 = m0 + e0, Y1 = m1 + e1, noise sd 1, correlation rho(x) rising from 0.1 at x = -2
to 0.95 at x = 2.
Model: one network, x -> (m0_hat, m1_hat, log s0, log s1, rho_hat). Two copies, same data, same start, same steps:
  separate: -log N(Y0 | m0_hat, s0) - log N(Y1 | m1_hat, s1)              (rho_hat unused)
  joint   : -log N2(Y0, Y1 | m0_hat, m1_hat, s0, s1, rho_hat)            (rho_hat learned by the loss)
Tracked every 100 steps on a grid of x: PEHE = RMSE of m1_hat - m0_hat vs tau, error of m0_hat and m1_hat, the
learned rho_hat(x). 10 seeds; both networks (narrow: width 6, wide: width 64).
   .venv/bin/python revision/theory/loss_geometry/sim5_track.py OUT_DIR
"""
import json, os, sys
import numpy as np
import torch
from multiprocessing import Pool

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
STEPS, BATCH, EVERY, SEEDS = 4000, 256, 100, range(10)
WIDTHS = {"narrow": 6, "wide": 64}
m0 = lambda x: torch.sin(2 * x) + 0.5 * x
tau = lambda x: 0.5 * torch.sin(3 * x)
rho = lambda x: 0.10 + 0.85 * (x + 2) / 4
XG = torch.linspace(-2, 2, 401).unsqueeze(1)


def net(w):
    return torch.nn.Sequential(torch.nn.Linear(1, w), torch.nn.Tanh(), torch.nn.Linear(w, w), torch.nn.Tanh(),
                               torch.nn.Linear(w, 5))


def heads(o):
    return o[:, 0], o[:, 1], o[:, 2].exp(), o[:, 3].exp(), 0.999 * torch.tanh(o[:, 4])


def nll(o, y, loss):
    a, b, s0, s1, r = heads(o)
    z0, z1 = (y[:, 0] - a) / s0, (y[:, 1] - b) / s1
    if loss == "separate":
        return (0.5 * z0 ** 2 + s0.log() + 0.5 * z1 ** 2 + s1.log()).mean()
    return ((z0 ** 2 - 2 * r * z0 * z1 + z1 ** 2) / (2 * (1 - r ** 2)) + s0.log() + s1.log()
            + 0.5 * torch.log(1 - r ** 2)).mean()


def run(args):
    wname, seed, loss = args
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(1000 + seed); torch.manual_seed(seed)
    f = net(WIDTHS[wname]); opt = torch.optim.Adam(f.parameters(), lr=1e-3)
    track = []
    for step in range(1, STEPS + 1):
        x = torch.rand(BATCH, 1, generator=g) * 4 - 2
        r = rho(x)[:, 0]
        z0, z1 = torch.randn(BATCH, generator=g), torch.randn(BATCH, generator=g)
        y = torch.stack([m0(x)[:, 0] + z0, (m0(x) + tau(x))[:, 0] + r * z0 + torch.sqrt(1 - r ** 2) * z1], 1)
        l = nll(f(x), y, loss)
        opt.zero_grad(); l.backward(); opt.step()
        if step % EVERY == 0:
            with torch.no_grad():
                a, b, _, _, rh = heads(f(XG))
                e = lambda p, q: (p - q).pow(2).mean().sqrt().item()
                track.append(dict(step=step, loss=l.item(), pehe=e(b - a, tau(XG)[:, 0]), m0=e(a, m0(XG)[:, 0]),
                                  m1=e(b, (m0(XG) + tau(XG))[:, 0])))
    with torch.no_grad():
        a, b, _, _, rh = heads(f(XG))
    final = dict(rho_hat=rh.tolist(), tau_err=((b - a) - tau(XG)[:, 0]).abs().tolist(), tau_hat=(b - a).tolist())
    return args, dict(track=track, final=final)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = [(w, s, l) for w in WIDTHS for s in SEEDS for l in ("separate", "joint")]
    with Pool(8) as pool:
        res = dict(pool.map(run, jobs))
    json.dump({"|".join(map(str, k)): v for k, v in res.items()}, open(os.path.join(OUT, "sim5_results.json"), "w"))
    L = ["Toy: correlation rising from 0.1 to 0.95 across x, learned by the model; separate vs joint loss, same data,",
         f"same start, {STEPS} steps; mean (SE) over {len(SEEDS)} seeds of the final errors; ratio = joint / separate per seed"]
    for w in WIDTHS:
        L += ["", f"network {w} (width {WIDTHS[w]})"]
        for k in ("pehe", "m0", "m1"):
            S = np.array([res[(w, s, "separate")]["track"][-1][k] for s in SEEDS])
            J = np.array([res[(w, s, "joint")]["track"][-1][k] for s in SEEDS]); q = J / S
            se = lambda a: a.std(ddof=1) / np.sqrt(len(a))
            L.append(f"   {k:5s} separate {S.mean():.4f} ({se(S):.4f})  joint {J.mean():.4f} ({se(J):.4f})  "
                     f"ratio {q.mean():.3f} ({se(q):.3f})  joint lower in {int((J < S).sum())}/{len(SEEDS)}")
        rh = np.array([res[(w, s, "joint")]["final"]["rho_hat"] for s in SEEDS]).mean(0)
        xs = XG[:, 0].numpy()
        L.append("   learned rho at x = -2, -1, 0, 1, 2: " + ", ".join(f"{rh[np.argmin(abs(xs - v))]:.2f}" for v in (-2, -1, 0, 1, 2))
                 + "   (true: " + ", ".join(f"{0.10 + 0.85 * (v + 2) / 4:.2f}" for v in (-2, -1, 0, 1, 2)) + ")")
    open(os.path.join(OUT, "sim5_summary.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
