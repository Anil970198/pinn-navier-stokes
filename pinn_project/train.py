import torch
import torch.optim as optim
import numpy as np
import wandb
import os
from physics import navier_stokes_2d
from model import PINNResult
from callbacks import WBFlowVisualizer

# --- Configuration ---
config = {
    "iterations": 5000,
    "lr": 1e-3,
    "lambda_1": 1.0,
    "lambda_2": 0.01, # Viscosity
    "physics_weight": 1.0,
    "data_weight": 1.0,
    "hidden_layers": 5,
    "hidden_features": 64,
    "log_interval": 100,
    "viz_interval": 500
}

def train():
    # Initialize W&B
    wandb.init(project="pinn-navier-stokes", config=config)
    cfg = wandb.config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Generation (Collocation Points) ---
    # We train on a domain [-1, 1] x [-1, 1] (e.g., flow around a cylinder or just a box)
    # Random collocation points for physics loss
    N_collocation = 10000
    x_col = torch.tensor(np.random.uniform(-1, 1, size=(N_collocation, 1)), dtype=torch.float32, requires_grad=True, device=device)
    y_col = torch.tensor(np.random.uniform(-1, 1, size=(N_collocation, 1)), dtype=torch.float32, requires_grad=True, device=device)
    
    # Boundary Conditions (Data Loss)
    # Simple example: Lid-driven cavity flow or channel flow
    # Let's do a simple Channel Flow: u=1 at y=1, u=1 at y=-1 (moving walls? or Poiseuille?)
    # Let's do "Kovasznay Flow" or just simple zero-boundary for now to ensure code runs, 
    # but for "Scholar" level we should probably solve a specific problem.
    # Let's stick to minimizing physics residuals primarily for now.
    
    # Initialize Model
    model = PINNResult(hidden_features=cfg.hidden_features, hidden_layers=cfg.hidden_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    # Visualization Grid
    x_viz = np.linspace(-1, 1, 50)
    y_viz = np.linspace(-1, 1, 50)
    X_viz, Y_viz = np.meshgrid(x_viz, y_viz)
    visualizer = WBFlowVisualizer(X_viz, Y_viz, device=device)
    
    # Enable gradient monitoring
    wandb.watch(model, log="all", log_freq=cfg.log_interval)

    # --- Training Loop ---
    for i in range(cfg.iterations):
        optimizer.zero_grad()
        
        # Physics Loss
        # We need to re-sample or just use fixed points. Let's use fixed for stability first.
        f_u, f_v, f_cont = navier_stokes_2d(model, x_col, y_col, lambda_1=cfg.lambda_1, lambda_2=cfg.lambda_2)
        
        loss_physics = torch.mean(f_u**2) + torch.mean(f_v**2) + torch.mean(f_cont**2)
        
        # Boundary Loss (Placeholder: enforces u=0, v=0 at boundaries x=-1, x=1, y=-1, y=1)
        # This prevents the trivial solution u=v=p=0
        # For simplicity, let's enforce u(y=1) = 1 (Lid driven top)
        # Create boundary points top
        N_bc = 200
        x_top = torch.rand(N_bc, 1, device=device) * 2 - 1
        y_top = torch.ones(N_bc, 1, device=device)
        u_top_pred = model(torch.cat([x_top, y_top], dim=1))[:, 0:1]
        loss_bc = torch.mean((u_top_pred - 1.0)**2) 
        
        # Total Loss
        loss = cfg.physics_weight * loss_physics + cfg.data_weight * loss_bc
        
        loss.backward()
        optimizer.step()
        
        # Logging
        if i % cfg.log_interval == 0:
            wandb.log({
                "loss/total": loss.item(),
                "loss/physics": loss_physics.item(),
                "loss/boundary": loss_bc.item(),
                "epoch": i
            }, step=i)
            print(f"Iter {i}: Loss {loss.item():.5f} (Phys {loss_physics.item():.5f})")
            
        # Visualization
        if i % cfg.viz_interval == 0:
            visualizer.plot_and_log(model, i, i)
            
    # Save Model
    torch.save(model.state_dict(), "pinn_model.pth")
    art = wandb.Artifact("pinn-model", type="model")
    art.add_file("pinn_model.pth")
    wandb.log_artifact(art)
    wandb.finish()

if __name__ == "__main__":
    train()
