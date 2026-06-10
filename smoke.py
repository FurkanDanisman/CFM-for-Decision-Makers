"""
Tiny smoke test — verifies the model + loss imports and runs one
forward+backward pass on CPU or GPU. No real training.

Run:  python smoke.py
"""
import sys
import torch

sys.path.insert(0, '.')
from losses.BarDistribution2D import total_params, make_edges, neg_log_prob_2d
from models.InterventionalPFN import InterventionalPFN


def main():
    J = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  free mem: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    model = InterventionalPFN(
        num_features=50, d_model=32, depth=2,
        heads_feat=4, heads_samp=4, output_dim=total_params(J),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    X_obs  = torch.randn(1, 20, 50, device=device)
    T_obs  = torch.zeros(1, 20, 1, device=device)
    Y_obs  = torch.randn(1, 20, device=device)
    X_intv = torch.randn(1, 5, 50, device=device)
    Y_do0  = torch.randn(1, 5, device=device).clamp(-0.9, 0.9)
    Y_do1  = torch.randn(1, 5, device=device).clamp(-0.9, 0.9)

    out = model(X_obs, T_obs, Y_obs, X_intv)
    print(f"output shape: {tuple(out['predictions'].shape)}")

    edges = make_edges(J).to(device)
    loss = neg_log_prob_2d(out['predictions'], Y_do0, Y_do1, J, edges)
    print(f"loss: {loss.item():.4f}")

    loss.backward()
    total_grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"grad norm sum: {total_grad:.4f}")

    print("SMOKE TEST OK")


if __name__ == '__main__':
    main()
