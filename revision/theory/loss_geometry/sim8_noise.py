"""Toys varying the SIZE of the noise (same functions, network, Adam, 4000 steps, batch 256 units, 30 seeds as
sim6_groups.py / sim7_cpfn.py). The size is drawn per unit from the network's range, independently of x:
sd in [0.25, 0.75), [0.75, 1.25), [1.25, 1.75], and [0.25, 1.75].
  mode cpfn  (CausalPFN): u sd 2, independent between the arms; eta with sd from the range and corr(eta0, eta1) drawn per
             unit from U[0.1, 0.95]. 1D target m + u (both outcomes, separate loss); 2D target m + u + eta (joint loss).
  mode dopfn (UWYK / Do-PFN, noise cancellation): eps with sd from the range and corr(eps0, eps1) drawn per unit from
             U[0.1, 0.95]. 1D target: ONE randomly chosen outcome per unit, m_t + eps_t (separate loss on that outcome);
             2D target: both outcomes m + eps (joint loss).
Tracked every 100 steps: training loss on fixed test units (each network's own loss), the loss of a perfect prediction
(true m, sds and correlation of each unit) on the same units; after training: PEHE against tau(x) on a grid of x.
   .venv/bin/python revision/theory/loss_geometry/sim8_noise.py MODE OUT_DIR
"""
import importlib.util, json, math, os, sys
import numpy as np
import torch
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s5", os.path.join(HERE, "sim5_track.py"))
S5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S5)
MODE = sys.argv[1] if len(sys.argv) > 1 else "cpfn"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
STEPS, BATCH, EVERY, SEEDS, WIDTH = 4000, 256, 100, range(30), 6
RANGES = {"low": (0.25, 0.75), "medium": (0.75, 1.25), "high": (1.25, 1.75), "full": (0.25, 1.75)}
SD_U = 2.0
m0, tau, XG = S5.m0, S5.tau, S5.XG


def pair(r, g):
    z0, z1 = torch.randn(len(r), generator=g), torch.randn(len(r), generator=g)
    return z0, r * z0 + torch.sqrt(1 - r ** 2) * z1


def draw(x, setting, loss, g):
    """targets (n, 2), the arm each unit shows (1D dopfn only, else None), and each unit's true (sd0, sd1, corr)"""
    n = len(x); lo, hi = RANGES[setting]
    sd = lo + (hi - lo) * torch.rand(n, generator=g)
    r = 0.1 + 0.85 * torch.rand(n, generator=g)
    a, b = pair(r, g)
    m = torch.stack([m0(x)[:, 0], (m0(x) + tau(x))[:, 0]], 1)
    if MODE == "cpfn":
        u0, u1 = pair(torch.zeros(n), g)
        y = m + SD_U * torch.stack([u0, u1], 1)
        if loss == "joint":
            y = y + sd[:, None] * torch.stack([a, b], 1)
            v = SD_U ** 2 + sd ** 2
            return y, None, (v.sqrt(), v.sqrt(), r * sd ** 2 / v)
        return y, None, (torch.full((n,), SD_U), torch.full((n,), SD_U), torch.zeros(n))
    y = m + sd[:, None] * torch.stack([a, b], 1)
    arm = torch.randint(0, 2, (n,), generator=g) if loss == "separate" else None
    return y, arm, (sd, sd, r)


def unit_loss(o, y, arm, loss):
    if arm is None:
        return S5.nll_units(o, y, loss)
    A, B, s0, s1, _ = S5.heads(o)
    mu, s = torch.where(arm == 0, A, B), torch.where(arm == 0, s0, s1)
    yy = torch.where(arm == 0, y[:, 0], y[:, 1])
    return 0.5 * ((yy - mu) / s) ** 2 + s.log()


def test_units(setting, loss):
    g = torch.Generator().manual_seed(12345)
    x = torch.rand(30000, 1, generator=g) * 4 - 2
    return (x,) + draw(x, setting, loss, g)


def perfect(setting, loss):
    x, y, arm, (s0, s1, r) = test_units(setting, loss)
    o = torch.stack([m0(x)[:, 0], (m0(x) + tau(x))[:, 0], s0.log(), s1.log(), torch.atanh(r.clamp(-0.998, 0.998) / 0.999)], 1)
    return unit_loss(o, y, arm, loss).mean().item()


def run(args):
    setting, seed, loss = args
    torch.set_num_threads(1)
    XT, YT, AT, _ = test_units(setting, loss)
    g = torch.Generator().manual_seed(1000 + seed); torch.manual_seed(seed)
    f = S5.net(WIDTH); opt = torch.optim.Adam(f.parameters(), lr=1e-3)
    track = []
    for step in range(1, STEPS + 1):
        x = torch.rand(BATCH, 1, generator=g) * 4 - 2
        y, arm, _ = draw(x, setting, loss, g)
        l = unit_loss(f(x), y, arm, loss).mean()
        opt.zero_grad(); l.backward(); opt.step()
        if step % EVERY == 0:
            with torch.no_grad():
                track.append(dict(step=step, loss=unit_loss(f(XT), YT, AT, loss).mean().item()))
    with torch.no_grad():
        a, b, _, _, rh = S5.heads(f(XG))
        pehe = ((b - a) - tau(XG)[:, 0]).pow(2).mean().sqrt().item()
    return args, dict(track=track, pehe=pehe, rho_hat=rh.mean().item())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = [(st, s, l) for st in RANGES for s in SEEDS for l in ("separate", "joint")]
    with Pool(8) as pool:
        res = dict(pool.map(run, jobs))
    json.dump({"mode": MODE, "ranges": RANGES, "perfect": {f"{st}|{l}": perfect(st, l) for st in RANGES for l in ("separate", "joint")},
               "runs": {"|".join(map(str, k)): v for k, v in res.items()}},
              open(os.path.join(OUT, f"sim8_{MODE}_results.json"), "w"))
    from scipy import stats
    for st in RANGES:
        a = np.array([res[(st, s, "separate")]["pehe"] for s in SEEDS]); b = np.array([res[(st, s, "joint")]["pehe"] for s in SEEDS])
        se = lambda v: v.std(ddof=1) / np.sqrt(len(v))
        print(f"{MODE} {st:6s}: 1D {a.mean():.4f} ({se(a):.4f})  2D {b.mean():.4f} ({se(b):.4f})  2D/1D {np.mean(b / a):.3f}  "
              f"2D better in {int((b < a).sum())}/{len(a)}  t p {stats.ttest_rel(b, a).pvalue:.2g}  Wilcoxon p {stats.wilcoxon(b, a).pvalue:.2g}  "
              f"2D learned rho {np.mean([res[(st, s, 'joint')]['rho_hat'] for s in SEEDS]):.2f}")
