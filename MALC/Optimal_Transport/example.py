"""Runnable demonstration of the OT-CATE-ATE pipeline.

Three scenarios:
    A. K=2 ground-truth comparison at N=50 (the canonical experiment)
    B. K=3 ground-truth comparison at N=50
    C. Sample-size ablation for K=2 across N ∈ {10, 30, 50, 100}

Outputs four PNGs:  A.png, B.png, ablation_K2.png
"""

from __future__ import annotations

import numpy as np

from simulator import make_dgp_K2, make_dgp_K3
from validation import run_ground_truth_check, run_sample_size_ablation


def main():
    # Common output d-grid: wide enough to cover the support of any plausible
    # CATE under K=2 or K=3 with shifts.
    d_grid = np.linspace(-6, 6, 161)

    print("=== A. Ground truth at N=50, K=2 (asym 0.4/0.6) ===")
    dgp2 = make_dgp_K2(asym=0.6, sd=0.6, shift_sd=0.3)
    run_ground_truth_check(
        dgp2, d_grid, N=50, B=10000, seed=20180621,
        title="K=2 ground-truth comparison (N=50)",
        savepath="A_K2_N50.png",
    )

    print("\n=== B. Ground truth at N=50, K=3 (0.3/0.4/0.3) ===")
    dgp3 = make_dgp_K3(sd=0.6, shift_sd=0.3)
    run_ground_truth_check(
        dgp3, d_grid, N=50, B=10000, seed=20180621,
        title="K=3 ground-truth comparison (N=50)",
        savepath="B_K3_N50.png",
    )

    print("\n=== C. Sample-size ablation, K=2 ===")
    run_sample_size_ablation(
        dgp2, d_grid, Ns=[10, 25, 50, 100], n_reps=2, B=10000,
        savepath="C_ablation_K2.png",
    )


if __name__ == "__main__":
    main()
