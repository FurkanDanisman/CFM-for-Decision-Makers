"""Separate networks per correlation range: same toy as sim5_track.py (same functions, x ~ U[-2, 2], same network,
4000 steps, 10 seeds), but the correlation of the two outcomes' noise rises linearly over x only within one range:
  [0.1, 0.4), [0.4, 0.7), [0.7, 0.95].
For each range, a 1D (separate loss) and a 2D (joint loss) network are trained, each learning its own correlation.
Tracked every 100 steps: training loss on fixed test units of that range; the loss of a perfect prediction (true m0,
m1, sd 1, true rho) on the same units; after training, the effect error on a grid of x (with its rho).
   .venv/bin/python revision/theory/loss_geometry/sim6_groups.py OUT_DIR
"""
import importlib.util, json, os, sys
import torch
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s5", os.path.join(HERE, "sim5_track.py"))
S5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S5)
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
STEPS, BATCH, EVERY, SEEDS, WIDTH = 4000, 256, 100, range(10), 6
RANGES = {"low": (0.1, 0.4), "medium": (0.4, 0.7), "high": (0.7, 0.95)}
m0, tau, XG = S5.m0, S5.tau, S5.XG


def rho_fn(lo, hi):
    return lambda x: lo + (hi - lo) * (x + 2) / 4


def draw(x, r, g):
    z0, z1 = torch.randn(len(x), generator=g), torch.randn(len(x), generator=g)
    return torch.stack([m0(x)[:, 0] + z0, (m0(x) + tau(x))[:, 0] + r * z0 + torch.sqrt(1 - r ** 2) * z1], 1)


def test_units(rf):
    g = torch.Generator().manual_seed(12345)
    x = torch.rand(30000, 1, generator=g) * 4 - 2
    r = rf(x)[:, 0]
    return x, r, draw(x, r, g)


def run(args):
    grp, seed, loss = args
    torch.set_num_threads(1)
    rf = rho_fn(*RANGES[grp]); XT, RT, YT = test_units(rf)
    g = torch.Generator().manual_seed(1000 + seed); torch.manual_seed(seed)
    f = S5.net(WIDTH); opt = torch.optim.Adam(f.parameters(), lr=1e-3)
    track = []
    for step in range(1, STEPS + 1):
        x = torch.rand(BATCH, 1, generator=g) * 4 - 2
        l = S5.nll(f(x), draw(x, rf(x)[:, 0], g), loss)
        opt.zero_grad(); l.backward(); opt.step()
        if step % EVERY == 0:
            with torch.no_grad():
                track.append(dict(step=step, loss=S5.nll_units(f(XT), YT, loss).mean().item()))
    with torch.no_grad():
        a, b, _, _, _ = S5.heads(f(XG))
        tau_err = ((b - a) - tau(XG)[:, 0]).abs().tolist()
    return args, dict(track=track, tau_err=tau_err)


def perfect(grp, loss):
    rf = rho_fn(*RANGES[grp]); XT, RT, YT = test_units(rf); n = len(XT)
    o = torch.stack([m0(XT)[:, 0], (m0(XT) + tau(XT))[:, 0], torch.zeros(n), torch.zeros(n), torch.atanh(RT / 0.999)], 1)
    return S5.nll_units(o, YT, loss).mean().item()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = [(g, s, l) for g in RANGES for s in SEEDS for l in ("separate", "joint")]
    with Pool(8) as pool:
        res = dict(pool.map(run, jobs))
    out = {"ranges": RANGES, "rho_grid": {g: rho_fn(*RANGES[g])(XG)[:, 0].tolist() for g in RANGES},
           "perfect": {f"{g}|{l}": perfect(g, l) for g in RANGES for l in ("separate", "joint")},
           "runs": {"|".join(map(str, k)): v for k, v in res.items()}}
    json.dump(out, open(os.path.join(OUT, "sim6_results.json"), "w"))
    import numpy as np
    for g in RANGES:
        for l in ("separate", "joint"):
            E = np.array([res[(g, s, l)]["tau_err"] for s in SEEDS]) ** 2
            P = np.sqrt(E.mean(1)); L = np.array([res[(g, s, l)]["track"][-1]["loss"] for s in SEEDS])
            print(f"{g:6s} {l:8s}: PEHE {P.mean():.4f} ({P.std(ddof=1) / np.sqrt(len(P)):.4f})  final loss {L.mean():.3f}  "
                  f"perfect {out['perfect'][f'{g}|{l}']:.3f}")
