import os
from pathlib import Path

import networkx as nx
import pytest
import torch


_LOCAL_DOPFN = Path(__file__).resolve().parents[3] / "Do-PFN"
if _LOCAL_DOPFN.exists():
    os.environ.setdefault("DOPFN_SRC", str(_LOCAL_DOPFN))

from training.data.PairedDoPFNDataset import PairedDoPFNDataset


def _config(graph, t_idx, y_idx, x_idcs):
    return {
        "graph": graph,
        "t_idx": t_idx,
        "y_idx": y_idx,
        "x_idcs": x_idcs,
        "noise_std": 0.05,
        "noise_dist": "gaussian",
        "exo_std": 1.0,
        "exo_dist": "gaussian",
        "nonlins": "id",
        "max_hidden_layers": 0,
        "binary_strategy": "mean",
    }


@pytest.mark.parametrize(
    ("graph", "t_idx", "y_idx", "x_idcs"),
    [
        # Exogenous treatment U0; X1 outcome; U2 observed covariate.
        (nx.DiGraph([(0, 1), (2, 1)]), 0, 1, [2]),
        # Endogenous treatment X1 between root U0 and outcome X2.
        (nx.DiGraph([(0, 1), (1, 2)]), 1, 2, [0]),
    ],
)
def test_joint_contract_shared_noise_and_arm_order(graph, t_idx, y_idx, x_idcs):
    dataset = PairedDoPFNDataset(
        dopfn_config=_config(graph, t_idx, y_idx, x_idcs),
        n_train=32,
        n_test=16,
        num_features=1,
        seed_base=7,
        min_target_variance=None,
        min_unique_target_fraction=None,
    )
    sample = dataset[0]

    assert sample["X_obs"].shape == (32, 1)
    assert sample["T_obs"].shape == (32, 1)
    assert sample["Y_obs"].shape == (32, 1)
    assert sample["X_intv"].shape == (16, 1)
    assert sample["Y_do0"].shape == sample["Y_do1"].shape == (16, 1)
    assert set(sample["T_obs"].flatten().tolist()) == {0.0, 1.0}
    assert torch.isclose(sample["Y_obs"].min(), torch.tensor(-1.0))
    assert torch.isclose(sample["Y_obs"].max(), torch.tensor(1.0))

    debug = dataset.last_debug
    treatment = debug["treatment"]
    non_treatment_exogenous = set(debug["scm"].exogenous_vars) - {treatment}
    for key in non_treatment_exogenous:
        assert torch.equal(debug["do0"][key], debug["do1"][key])

    # Preserve Do-PFN's original model-facing encoding: lower raw treatment is
    # 1 and upper raw treatment is 0.
    assert torch.all(debug["raw_do0"] > debug["raw_do1"])
    assert torch.equal(debug["do0"][treatment], debug["raw_do0"])
    assert torch.equal(debug["do1"][treatment], debug["raw_do1"])
    midpoint = (debug["raw_do0"][0, 0] + debug["raw_do1"][0, 0]) / 2
    expected_t_obs = (
        debug["obs"][treatment][0, : dataset.n_train] < midpoint
    ).float()
    assert torch.equal(sample["T_obs"].flatten(), expected_t_obs)

    # With identity mechanisms and shared noise, each individual's effect is
    # constant within this SCM. Independent arm noise would violate this.
    treatment_effect = sample["Y_do1"] - sample["Y_do0"]
    assert treatment_effect.std() < 1e-5
    assert treatment_effect.abs().max() > 1e-5


def test_dataset_is_deterministic_by_index():
    graph = nx.DiGraph([(0, 1), (2, 1)])
    dataset = PairedDoPFNDataset(
        dopfn_config=_config(graph, 0, 1, [2]),
        n_train=24,
        n_test=8,
        num_features=1,
        seed_base=11,
        min_target_variance=None,
        min_unique_target_fraction=None,
    )

    first = dataset[3]
    repeated = dataset[3]
    assert set(first) == {
        "X_obs",
        "T_obs",
        "Y_obs",
        "X_intv",
        "Y_do0",
        "Y_do1",
    }
    for key in first:
        assert torch.equal(first[key], repeated[key])
