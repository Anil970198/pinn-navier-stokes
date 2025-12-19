import torch

def navier_stokes_2d(model, x, y, lambda_1=1.0, lambda_2=0.01):
    """
    Computes the residuals of the 2D Navier-Stokes equations.
    
    Args:
        model: The PINN model that takes (x, y, t) or (x, y) and returns (u, v, p).
        x: Input x coordinates (requires_grad=True).
        y: Input y coordinates (requires_grad=True).
        lambda_1: Scalar parameter (usually 1.0).
        lambda_2: Scalar parameter (viscosity, 1/Re).
        
    Returns:
        f_u: Residual for x-momentum equation.
        f_v: Residual for y-momentum equation.
        f_continuity: Residual for continuity equation.
    """
    # Enable gradient computation for inputs
    # We assume x and y are already tensors with requires_grad=True
    
    # Forward pass
    # Concatenate inputs to pass to the model
    # Model expected to output: psi (stream function) and p (pressure) OR u, v, p
    # For this standard implementation, let's assume the model outputs (u, v, p) directly
    # Input shape: (N, 2) -> (x, y)
    inputs = torch.cat([x, y], dim=1)
    outputs = model(inputs)
    
    u = outputs[:, 0:1]
    v = outputs[:, 1:2]
    p = outputs[:, 2:3]
    
    # First derivatives
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    
    # Second derivatives
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    
    # Navier-Stokes Equations
    # Momentum u: u*u_x + v*u_y + p_x - lambda_2*(u_xx + u_yy) = 0
    f_u = lambda_1 * (u * u_x + v * u_y) + p_x - lambda_2 * (u_xx + u_yy)
    
    # Momentum v: u*v_x + v*v_y + p_y - lambda_2*(v_xx + v_yy) = 0
    f_v = lambda_1 * (u * v_x + v * v_y) + p_y - lambda_2 * (v_xx + v_yy)
    
    # Continuity: u_x + v_y = 0
    f_continuity = u_x + v_y
    
    return f_u, f_v, f_continuity
