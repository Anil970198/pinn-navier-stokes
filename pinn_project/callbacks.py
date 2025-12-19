import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import io
from PIL import Image

class WBFlowVisualizer:
    def __init__(self, x_grid, y_grid, device='cpu'):
        """
        Args:
            x_grid, y_grid: Meshgrid coordinates (numpy arrays) for visualization.
        """
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.device = device
        
        # Prepare tensor inputs for the model
        self.x_tensor = torch.tensor(x_grid.flatten()[:, None], dtype=torch.float32, device=device)
        self.y_tensor = torch.tensor(y_grid.flatten()[:, None], dtype=torch.float32, device=device)
        self.inputs = torch.cat([self.x_tensor, self.y_tensor], dim=1)

    def plot_and_log(self, model, epoch, step):
        model.eval()
        with torch.no_grad():
            outputs = model(self.inputs)
            # outputs: (N, 3) -> u, v, p
            u = outputs[:, 0].cpu().numpy().reshape(self.x_grid.shape)
            v = outputs[:, 1].cpu().numpy().reshape(self.x_grid.shape)
            p = outputs[:, 2].cpu().numpy().reshape(self.x_grid.shape)
            
            velocity_magnitude = np.sqrt(u**2 + v**2)

        # Create a figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Velocity Magnitude (Heatmap)
        c1 = axes[0].contourf(self.x_grid, self.y_grid, velocity_magnitude, levels=50, cmap='viridis')
        fig.colorbar(c1, ax=axes[0])
        axes[0].set_title(f"Velocity Magnitude (Step {step})")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        
        # Plot 2: Pressure Field
        c2 = axes[1].contourf(self.x_grid, self.y_grid, p, levels=50, cmap='plasma')
        fig.colorbar(c2, ax=axes[1])
        axes[1].set_title(f"Pressure Field (Step {step})")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")

        # Plot 3: Vector Field (Quiver)
        # Downsample for clearer quiver plot if grid is dense
        skip = (slice(None, None, 4), slice(None, None, 4))
        axes[2].quiver(self.x_grid[skip], self.y_grid[skip], u[skip], v[skip], color='black')
        axes[2].set_title(f"Flow Vectors (Step {step})")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        
        plt.tight_layout()
        
        # Convert to W&B Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        
        # Log to W&B
        wandb.log({
            "flow_visualization": wandb.Image(image, caption=f"Epoch {epoch}"),
            "epoch": epoch
        }, step=step)
        
        plt.close(fig)
        
    def plot_residuals(self, physics_loss, data_loss, epoch):
        # This could be a custom chart if we accumulated history, 
        # but W&B handles scalar logging automatically.
        pass
