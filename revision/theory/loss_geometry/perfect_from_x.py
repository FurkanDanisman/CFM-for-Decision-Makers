"""Perfect-prediction loss = the best loss possible from x: the true m0(x), m1(x) and ONE Gaussian for the noise, fitted
by maximum likelihood on the test units (its sds and correlation are the overall sds and correlation of Y - m). The
noise size and correlation of a unit are drawn independently of x, so no network can do better than this from x.
Adds "perfect_x" to sim6_results.json, sim7_results.json and sim8_{cpfn,dopfn}_results.json (same test units as tracked).
   .venv/bin/python revision/theory/loss_geometry/perfect_from_x.py OUT_DIR
"""
import importlib.util, json, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]


def load(name, argv):
    sys.argv = [name] + argv
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


S5 = load("sim5_track", [OUT])


def best(x, y, arm, loss, unit_loss):
    m = torch.stack([S5.m0(x)[:, 0], (S5.m0(x) + S5.tau(x))[:, 0]], 1); res = y - m; n = len(x)
    if arm is None:
        s0, s1 = res[:, 0].std(), res[:, 1].std()
        r = torch.corrcoef(res.T)[0, 1] if loss == "joint" else torch.tensor(0.0)
    else:
        s0, s1 = res[arm == 0, 0].std(), res[arm == 1, 1].std(); r = torch.tensor(0.0)
    o = torch.stack([m[:, 0], m[:, 1], s0.log().expand(n), s1.log().expand(n), torch.atanh(r / 0.999).expand(n)], 1)
    return unit_loss(o, y, arm, loss).mean().item()


nll = lambda o, y, arm, loss: S5.nll_units(o, y, loss)
LOSSES = ("separate", "joint")

S6 = load("sim6_groups", [OUT]); p = os.path.join(OUT, "sim6_results.json"); R = json.load(open(p))
R["perfect_x"] = {}
for st in S6.RANGES:
    for l in LOSSES:
        x, y, _ = S6.test_units(st); R["perfect_x"][f"{st}|{l}"] = best(x, y, None, l, nll)
json.dump(R, open(p, "w")); print("sim6", R["perfect_x"])

S7 = load("sim7_cpfn", [OUT]); p = os.path.join(OUT, "sim7_results.json"); R = json.load(open(p))
R["perfect_x"] = {}
for st in S7.RANGES:
    for l in LOSSES:
        x, y, _ = S7.test_units(st, S7.rho_u_of(st, 0), l); R["perfect_x"][f"{st}|{l}"] = best(x, y, None, l, nll)
json.dump(R, open(p, "w")); print("sim7", R["perfect_x"])

for mode in ("cpfn", "dopfn"):
    S8 = load("sim8_noise", [mode, OUT]); p = os.path.join(OUT, f"sim8_{mode}_results.json"); R = json.load(open(p))
    R["perfect_x"] = {}
    for st in S8.RANGES:
        for l in LOSSES:
            x, y, arm, _ = S8.test_units(st, l); R["perfect_x"][f"{st}|{l}"] = best(x, y, arm, l, S8.unit_loss)
    json.dump(R, open(p, "w")); print("sim8", mode, R["perfect_x"])
