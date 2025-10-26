"""
MLP velocity field model for flow matching.

This module implements a small MLP that takes concatenated [x,t] inputs
and outputs a 2D velocity field v_θ(x,t), with exact divergence computation
via autograd.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VelocityFieldMLP(nn.Module):
    """
    MLP velocity field v_θ: ℝ² × [0,1] → ℝ².
    
    Architecture: 2-3 hidden layers, width 64-128, SiLU/Softplus activation.
    Input: concatenated [x, t] where x ∈ ℝ², t ∈ [0,1]
    Output: velocity field v_θ(x,t) ∈ ℝ²
    """
    
    def __init__(self, hidden_dim: int = 128, num_layers: int = 3, activation: str = "silu"):
        """
        Initialize the velocity field MLP.
        
        Args:
            hidden_dim: Hidden layer dimension (64-128)
            num_layers: Number of hidden layers (2-3)
            activation: Activation function ("silu" or "softplus")
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        
        # Input: [x, t] where x ∈ ℝ², t ∈ ℝ
        input_dim = 3
        
        # Build layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output: velocity field v_θ(x,t) ∈ ℝ²
        layers.append(nn.Linear(hidden_dim, 2))
        
        self.layers = nn.ModuleList(layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function."""
        if self.activation == "silu":
            return F.silu(x)
        elif self.activation == "softplus":
            return F.softplus(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: v_θ(x,t).
        
        Args:
            x: Spatial coordinates of shape (..., 2)
            t: Time coordinates of shape (..., 1) or scalar
            
        Returns:
            Velocity field of shape (..., 2)
        """
        # Ensure t has the same batch dimensions as x
        if t.dim() == 0:
            t = t.expand(x.shape[:-1] + (1,))
        elif t.dim() == 1 and x.dim() > 1:
            t = t.unsqueeze(-1)
        
        # Concatenate [x, t]
        inputs = torch.cat([x, t], dim=-1)
        
        # Forward through layers
        h = inputs
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h)
            h = self._activation(h)
        
        # Final layer (no activation)
        output = self.layers[-1](h)
        
        return output
    
    def divergence(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute divergence ∇·v_θ(x,t) = Σᵢ ∂v_{θ,i}/∂xᵢ via autograd.
        
        Args:
            x: Spatial coordinates of shape (..., 2)
            t: Time coordinates of shape (..., 1) or scalar
            
        Returns:
            Divergence of shape (...,)
        """
        # Enable gradients for x
        x.requires_grad_(True)
        
        # Compute velocity field
        v = self.forward(x, t)
        
        # Compute divergence via autograd
        div = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
        
        for i in range(2):  # For each component of velocity
            # Compute gradient of v_i w.r.t. x
            grad_v_i = torch.autograd.grad(
                v[..., i].sum(), x, 
                create_graph=True, retain_graph=True
            )[0]
            
            # Add ∂v_i/∂x_i to divergence
            div += grad_v_i[..., i]
        
        return div


def test_velocity_field():
    """Test velocity field model."""
    print("Testing velocity field model...")
    
    # Create model
    model = VelocityFieldMLP(hidden_dim=128, num_layers=3, activation="silu")
    model = model.double()  # Use float64
    
    # Test inputs
    batch_size = 100
    x = torch.randn(batch_size, 2, dtype=torch.float64)
    t = torch.rand(batch_size, dtype=torch.float64)
    
    # Test forward pass
    v = model(x, t)
    print(f"Input x shape: {x.shape}")
    print(f"Input t shape: {t.shape}")
    print(f"Output v shape: {v.shape}")
    print(f"Velocity range: [{v.min():.4f}, {v.max():.4f}]")
    
    # Test divergence computation
    div = model.divergence(x, t)
    print(f"Divergence shape: {div.shape}")
    print(f"Divergence range: [{div.min():.4f}, {div.max():.4f}]")
    print(f"Divergence mean: {div.mean():.4f}")
    
    # Test gradient flow
    loss = (v**2).sum()
    loss.backward()
    
    # Check that gradients exist
    has_gradients = any(p.grad is not None for p in model.parameters())
    print(f"Model has gradients: {has_gradients}")
    
    # Test with scalar time
    t_scalar = torch.tensor(0.5, dtype=torch.float64)
    v_scalar = model(x, t_scalar)
    print(f"Scalar time output shape: {v_scalar.shape}")
    
    # Test divergence with scalar time
    div_scalar = model.divergence(x, t_scalar)
    print(f"Scalar time divergence shape: {div_scalar.shape}")


if __name__ == "__main__":
    test_velocity_field()
