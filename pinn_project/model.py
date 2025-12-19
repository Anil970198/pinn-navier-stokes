import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    # See paper: Implicit Neural Representations with Periodic Activation Functions (Sitzmann et al., 2020)
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                            1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                            np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        # sin(omega_0 * (Wx + b))
        return torch.sin(self.omega_0 * self.linear(input))

class PINNResult(nn.Module):
    def __init__(self, in_features=2, hidden_features=64, hidden_layers=3, out_features=3):
        """
        Args:
            in_features: 2 for (x, y)
            out_features: 3 for (u, v, p)
        """
        super().__init__()
        
        layers = []
        # First layer
        layers.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=30.0))
        
        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=30.0))
            
        # Output layer (Linear, as we want arbitrary range for u, v, p)
        # Note: Some papers use Sine for output too, but Linear is often safer for regression specific ranges
        self.net = nn.Sequential(*layers)
        self.final_linear = nn.Linear(hidden_features, out_features)
        
        with torch.no_grad():
            self.final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / 30.0, 
                                              np.sqrt(6 / hidden_features) / 30.0)
            
    def forward(self, x):
        # x shape (N, 2)
        out = self.net(x)
        out = self.final_linear(out)
        return out
