"""The in-prior "same" SCM (revision/experiments/E2_in_prior/gen_dopfn.py, seed 20261003) as a training source.

Built once exactly as the in-prior data were drawn (sample_batch(20261003, seq_len=1,500,000, batch_size=1,
on_nonfinite='resample')). Fixed: graph, T / Y roles, structural weights, activations, the covariate set x_keys,
the hyperparameters, and the binarization (threshold and the two treatment levels t1 < t2 of that draw). Every
visit draws only fresh noise -- exogenous roots N(0, exo_std) and each equation's additive noise N(0, noise_std) --
at the training shape (batch_size, seq_len), then runs the observational pass and both pure arms as prior.py does.

Which steps use it (pure functions of the absolute step index, so a resume is consistent):
  natural   never
  p10       each step with probability 0.10, instead of that step's prior SCM
  once      exactly one step, at a fixed random position in [0, steps)
`chunks` datasets-of-4 per such step (1 = the usual 4 datasets; 250 = 1000 datasets, one optimizer update).
"""
from __future__ import annotations

import numpy as np
import torch

from training_dopfn_repro import prior as P

SEED = 20261003
BUILD_ROWS = 1_500_000            # gen_dopfn.py same mode: R1 * 1500 rows, R1 = 1000
P10_SALT, ONCE_SALT = 777_001, 777_002


def uses_fixed(mode: str, step: int, steps: int) -> bool:
    if mode == "natural":
        return False
    if mode == "p10":
        return bool(np.random.default_rng([P10_SALT, step]).random() < 0.10)
    if mode == "once":
        return step == int(np.random.default_rng(ONCE_SALT).integers(0, steps))
    raise ValueError(mode)


class FixedSCM:
    def __init__(self, data_r000: str):
        """data_r000: the in-prior same-SCM dataset 0 (gen_dopfn.py same: .../dopfn/shard0/complexmech/5node/
        path_TY/hide_0.0/r000.npz). Y and the covariates are read off it by exact value match: prior.py picks them
        through Python set order, which depends on the hash seed of the process, so the seed alone does not fix
        them. Graph, weights, noise and treatment levels are fixed by the seed."""
        cap = {}
        arm, stack = P._propagate_arm, P._stack_features

        def capture(scm, graph, t_key, value, shared):
            cap["scm"], cap["graph"] = scm, graph
            return arm(scm, graph, t_key, value, shared)

        def capture_obs(sample, keys):
            cap.setdefault("obs", {k: v[0, :1000].clone() for k, v in sample.items()})   # first call: obs pass
            return stack(sample, keys)
        P._propagate_arm, P._stack_features = capture, capture_obs
        try:
            rec = P.sample_batch(SEED, P.PriorConfig(seq_len=BUILD_ROWS, batch_size=1, on_nonfinite="resample"))
        finally:
            P._propagate_arm, P._stack_features = arm, stack
        self.meta = rec["meta"]
        self.scm, self.graph = cap["scm"], cap["graph"]
        self.t_key = self.meta["t_key"]
        self.y_key, self.x_keys = self._roles_from_data(cap["obs"], rec["x_obs"][:1000, 0, 0], data_r000)
        self.meta.update(y_key=self.y_key, x_keys=self.x_keys)
        self.thresh = (self.scm.t_threshs[0], self.scm.t1s[0], self.scm.t2s[0])
        self.scm.set_binarization_params = lambda treatment: None     # binarization stays the in-prior one

    def _roles_from_data(self, obs, t01, path):
        z = np.load(path)
        X, T, Y = z["X_train"], z["T_train"], z["Y_train"]
        if not np.array_equal(T, t01.numpy().astype(np.float32)):
            raise RuntimeError(f"{path}: treatment column does not match the rebuilt SCM")
        match = lambda col: [k for k, v in obs.items()
                             if k != self.t_key and np.allclose(v.numpy().astype(np.float32), col, rtol=1e-6, atol=1e-6)]
        ys = match(Y)
        xs = [match(X[:, j]) for j in range(X.shape[1])]
        if len(ys) != 1 or any(len(m) != 1 for m in xs) or X.shape[1] != self.meta["num_features"]:
            raise RuntimeError(f"{path}: no exact match to the rebuilt SCM (Y {ys}, X {xs})")
        return ys[0], [self.t_key] + [m[0] for m in xs]

    def describe(self) -> str:
        m = self.meta
        par = {v: sorted(self.graph.predecessors(v)) for v in self.graph.nodes}
        return (f"fixed SCM seed {SEED}: attempts {m['attempts']} F {m['num_features']} U {m['num_unobserved']} "
                f"exo_std {m['exo_std']:.6f} noise_std {m['noise_std']:.6f} edge_prob {m['edge_prob']:.6f} "
                f"T {self.t_key} Y {self.y_key} x_keys {self.x_keys} t1 {float(self.thresh[1]):.4f} "
                f"t2 {float(self.thresh[2]):.4f} parents {par}")

    def sample(self, seed: int, cfg: P.PriorConfig) -> dict:
        """One record in sample_batch's format: the fixed SCM, fresh noise of shape (batch_size, seq_len)."""
        B, S = cfg.batch_size, cfg.seq_len
        scm, m = self.scm, self.meta
        P._seed_everything(seed)
        for k in scm.exogenous_distributions:
            scm.exogenous_distributions[k] = (P._load_dopfn_components()[2]((B, S), m["exo_std"]), {})
        for k, (f, _) in scm.functions.items():
            f.additive_noise = torch.normal(0.0, m["noise_std"], (B, S))
        th, t1, t2 = self.thresh
        scm.t_threshs, scm.t1s, scm.t2s = th.repeat(B), t1.repeat(B), t2.repeat(B)

        endo, exo = scm.get_next_sample(binarize=True, graph=self.graph)
        s_obs = endo | exo
        t1s, t2s = scm.t1s[:, None].expand(-1, S), scm.t2s[:, None].expand(-1, S)
        raw1, raw0 = (t1s, t2s) if P._ARM1_IS_LOWER_RAW_LEVEL else (t2s, t1s)
        s1 = P._propagate_arm(scm, self.graph, self.t_key, raw1, exo)
        s0 = P._propagate_arm(scm, self.graph, self.t_key, raw0, exo)

        x_obs, x1, x0 = (P._stack_features(s, self.x_keys) for s in (s_obs, s1, s0))
        coin = torch.randint(0, 2, (B, S))
        t_int01 = (coin == 0).float().T
        if cfg.zero_one_treatment:
            mid = ((scm.t1s + scm.t2s) / 2.0).view(1, -1)
            x_obs[:, :, 0] = (x_obs[:, :, 0] < mid).float()
            x1[:, :, 0], x0[:, :, 0] = 1.0, 0.0
        y_obs, y1, y0 = (s[self.y_key].T.float() for s in (s_obs, s1, s0))
        sel = t_int01.bool()
        rec = {"x_obs": x_obs, "y_obs": y_obs, "t_int01": t_int01, "y_int": torch.where(sel, y1, y0),
               "y_do0": y0, "y_do1": y1, "x_int": torch.where(sel.unsqueeze(-1), x1, x0),
               "meta": {**m, "seed": seed, "fixed_scm": True, "attempts": 1}}
        rec["meta"]["nonfinite"] = not P._is_finite(rec)
        if rec["meta"]["nonfinite"]:
            for k in ("x_obs", "y_obs", "y_int", "y_do0", "y_do1", "x_int"):
                rec[k] = torch.full_like(rec[k], P.SENTINEL)
        return rec
