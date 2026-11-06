"""CausalPFN version of sim6_groups.py: the same setup, only the 1D target differs.
Same x ~ U[-2, 2], m0 = sin(2x) + 0.5x, tau = 0.5 sin(3x), network (width 6, outputs m0, m1, two sds, a correlation),
Adam, 4000 steps, batch 256, 10 seeds. Two noise terms:
  u   (the variation left in CausalPFN's E_y_t, not controlled), sd SD_U: corr(u0, u1) = rho_u; RHO_U = None draws one
      random value per network (per seed and range, the same for its 1D and 2D network) from U[0, 1), a number fixes it
  eta (the outcome noise CausalPFN's 1D target removes, controlled), sd 1: corr(eta0, eta1) = rho_eta, drawn per unit from the
      range, independently of x: [0.1, 0.4), [0.4, 0.7), [0.7, 0.95], and [0.1, 0.95]
  1D target: m_t(x) + u_t           (separate loss)
  2D target: m_t(x) + u_t + eta_t   (joint loss)
The actual corr(Y0, Y1 | x) of the 2D target is (rho_u SD_U^2 + rho_eta) / (SD_U^2 + 1), of the 1D target rho_u.
Setting: SD_U = 2, RHO_U = 0 (u dominates and is independent), output sim7_results.json; the earlier run with SD_U = 1,
RHO_U = None is kept as sim7_results_rhou_random.json.
Tracked every 100 steps: training loss on fixed test units; the loss of a perfect prediction (true m, sds, correlation)
on the same units; after training: PEHE against tau(x) on a grid of x.
   .venv/bin/python revision/theory/loss_geometry/sim7_cpfn.py OUT_DIR
"""
import importlib.util, json, math, os, sys
import numpy as np
import torch
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s5", os.path.join(HERE, "sim5_track.py"))
S5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S5)
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
STEPS, BATCH, EVERY, SEEDS, WIDTH = 4000, 256, 100, range(10), 6
SD_U, RHO_U = 2.0, 0.0
RANGES = {"low": (0.1, 0.4), "medium": (0.4, 0.7), "high": (0.7, 0.95), "full": (0.1, 0.95)}
m0, tau, XG = S5.m0, S5.tau, S5.XG


def rho_u_of(setting, seed):
    if RHO_U is not None:
        return RHO_U
    return float(torch.rand(1, generator=torch.Generator().manual_seed(777 + 31 * seed + list(RANGES).index(setting))))


def pair(n, r, g):
    z0, z1 = torch.randn(n, generator=g), torch.randn(n, generator=g)
    return z0, r * z0 + torch.sqrt(1 - r ** 2) * z1


def draw(x, setting, rho_u, loss, g):
    n = len(x); lo, hi = RANGES[setting]
    u0, u1 = pair(n, torch.full((n,), rho_u), g); u0, u1 = SD_U * u0, SD_U * u1
    r_eta = lo + (hi - lo) * torch.rand(n, generator=g)
    e0, e1 = pair(n, r_eta, g)
    y0, y1 = m0(x)[:, 0] + u0, (m0(x) + tau(x))[:, 0] + u1
    if loss == "joint":
        y0, y1 = y0 + e0, y1 + e1
    return torch.stack([y0, y1], 1), r_eta


def test_units(setting, rho_u, loss):
    g = torch.Generator().manual_seed(12345)
    x = torch.rand(30000, 1, generator=g) * 4 - 2
    y, r_eta = draw(x, setting, rho_u, loss, g)
    return x, y, r_eta


def perfect(x, y, r_eta, rho_u, loss):
    n = len(x)
    if loss == "joint":
        v = SD_U ** 2 + 1
        ls, r = torch.full((n,), 0.5 * math.log(v)), (rho_u * SD_U ** 2 + r_eta) / v
    else:
        ls, r = torch.full((n,), math.log(SD_U)), torch.zeros(n)
    o = torch.stack([m0(x)[:, 0], (m0(x) + tau(x))[:, 0], ls, ls, torch.atanh(r / 0.999)], 1)
    return S5.nll_units(o, y, loss).mean().item()


def run(args):
    setting, seed, loss = args
    torch.set_num_threads(1)
    rho_u = rho_u_of(setting, seed)
    XT, YT, RT = test_units(setting, rho_u, loss)
    g = torch.Generator().manual_seed(1000 + seed); torch.manual_seed(seed)
    f = S5.net(WIDTH); opt = torch.optim.Adam(f.parameters(), lr=1e-3)
    track = []
    for step in range(1, STEPS + 1):
        x = torch.rand(BATCH, 1, generator=g) * 4 - 2
        l = S5.nll(f(x), draw(x, setting, rho_u, loss, g)[0], loss)
        opt.zero_grad(); l.backward(); opt.step()
        if step % EVERY == 0:
            with torch.no_grad():
                track.append(dict(step=step, loss=S5.nll_units(f(XT), YT, loss).mean().item()))
    with torch.no_grad():
        a, b, _, _, rh = S5.heads(f(XG))
        pehe = ((b - a) - tau(XG)[:, 0]).pow(2).mean().sqrt().item()
    lo, hi = RANGES[setting]
    actual = (rho_u * SD_U ** 2 + (lo + hi) / 2) / (SD_U ** 2 + 1) if loss == "joint" else rho_u
    return args, dict(track=track, pehe=pehe, rho_hat=rh.mean().item(), rho_u=rho_u, actual_corr=actual,
                      perfect=perfect(XT, YT, RT, rho_u, loss))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = [(st, s, l) for st in RANGES for s in SEEDS for l in ("separate", "joint")]
    with Pool(8) as pool:
        res = dict(pool.map(run, jobs))
    json.dump({"ranges": RANGES, "runs": {"|".join(map(str, k)): v for k, v in res.items()}},
              open(os.path.join(OUT, "sim7_results.json"), "w"))
    for st in RANGES:
        row = []
        for l in ("separate", "joint"):
            P = np.array([res[(st, s, l)]["pehe"] for s in SEEDS])
            A = np.mean([res[(st, s, l)]["actual_corr"] for s in SEEDS])
            H = np.mean([res[(st, s, l)]["rho_hat"] for s in SEEDS])
            row.append(f"{l} PEHE {P.mean():.4f} ({P.std(ddof=1) / np.sqrt(len(P)):.4f}) actual corr {A:.2f} learned {H:.2f}")
        print(f"{st:6s}: " + "   ".join(row))
