import torch
import os
import wandb
from model import PINNResult
from physics import navier_stokes_2d

def test_pipeline():
    print("Testing PINN Pipeline...")
    
    # Mock settings
    os.environ["WANDB_MODE"] = "offline"
    wandb.init(project="pinn-test", mode="offline")
    
    device = "cpu"
    model = PINNResult(hidden_features=32, hidden_layers=2).to(device)
    
    # Create dummy data
    x = torch.rand(10, 1, requires_grad=True, device=device)
    y = torch.rand(10, 1, requires_grad=True, device=device)
    
    # Test Forward Pass
    out = model(torch.cat([x, y], dim=1))
    print(f"Forward pass shape: {out.shape}")
    assert out.shape == (10, 3)
    
    # Test Physics Gradients
    f_u, f_v, f_cont = navier_stokes_2d(model, x, y)
    print(f"Physics residuals shape: {f_u.shape}")
    assert f_u.shape == (10, 1)
    
    # Test Backward Pass
    loss = torch.mean(f_u**2)
    loss.backward()
    print("Backward pass successful.")
    
    wandb.finish()
    print("Pipeline Verified!")

if __name__ == "__main__":
    test_pipeline()
