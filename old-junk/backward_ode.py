"""
Backward ODE solver for evaluating q_t(x) and ∇log q_t(x).

This module implements the core numerical routine for integrating backward
in time to recover x_0 and compute log-densities and score functions
for the learned flow q_t.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint
from typing import Tuple, Callable
import numpy as np

# Optional import for type hints
try:
    from model import VelocityFieldMLP
except ImportError:
    VelocityFieldMLP = nn.Module  # Fallback for standalone execution


class BackwardODESystem(nn.Module):
    """
    ODE system for backward integration to evaluate q_t(x).
    
    Integrates backward from time t to 0:
    - State ODE: ẋ_s = -v_θ(x_s, s) with x_t = x
    - Log-density accumulator: ℓ̇_s = +∇·v_θ(x_s, s) with ℓ_0 = 0
    """
    
    def __init__(self, velocity_field: VelocityFieldMLP):
        """
        Initialize backward ODE system.
        
        Args:
            velocity_field: Trained velocity field model
        """
        super().__init__()
        self.velocity_field = velocity_field
    
    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ODE integration.
        
        Args:
            t: Current time (scalar)
            y: State vector [x, log_density] of shape (..., 3)
               where x ∈ ℝ² and log_density ∈ ℝ
        
        Returns:
            Derivative dy/dt of shape (..., 3)
        """
        # Split state
        x = y[..., :2]  # Spatial coordinates
        log_density = y[..., 2]  # Log-density accumulator
        
        # Compute velocity field
        v = self.velocity_field(x, t)
        
        # Compute divergence
        div_v = self.velocity_field.divergence(x, t)
        
        # State ODE: ẋ = -v_θ(x, t) (backward in time)
        dx_dt = -v
        
        # Log-density ODE: ℓ̇ = +∇·v_θ(x, t)
        dlog_density_dt = div_v
        
        # Combine derivatives
        # Ensure dlog_density_dt has the right shape for concatenation
        if dlog_density_dt.dim() == 0:
            # Zero-dimensional tensor (scalar) - make it 1D to match dx_dt
            dlog_density_dt = dlog_density_dt.unsqueeze(0)
        elif dlog_density_dt.dim() == 1:
            # 1D tensor - add dimension to match dx_dt
            dlog_density_dt = dlog_density_dt.unsqueeze(-1)
        
        dy_dt = torch.cat([dx_dt, dlog_density_dt], dim=-1)
        
        return dy_dt


class QTEvaluator:
    """
    Evaluator for q_t(x) and ∇log q_t(x) using backward ODE integration.
    """
    
    def __init__(self, 
                 velocity_field: VelocityFieldMLP,
                 rtol: float = 1e-6,
                 atol: float = 1e-8,
                 method: str = "dopri5"):
        """
        Initialize q_t evaluator.
        
        Args:
            velocity_field: Trained velocity field model
            rtol: Relative tolerance for ODE solver
            atol: Absolute tolerance for ODE solver
            method: ODE solver method ("dopri5", "tsit5", etc.)
        """
        self.velocity_field = velocity_field
        self.rtol = rtol
        self.atol = atol
        self.method = method
        
        # Create ODE system
        self.ode_system = BackwardODESystem(velocity_field)
    
    def evaluate_q_t(self, 
                    x_t: torch.Tensor, 
                    t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate q_t(x), ∇log q_t(x), and recovered initial states at given points.
        
        Args:
            x_t: Terminal states of shape (..., 2)
            t: Terminal times of shape (...,) or scalar
        
        Returns:
            Tuple of (log_q_t, grad_log_q_t, x_0) where:
            - log_q_t: Log-density of shape (...,)
            - grad_log_q_t: Score function of shape (..., 2)
            - x_0: Recovered initial states of shape (..., 2)
        """
        # Convert t to tensor if it's a scalar
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=x_t.dtype, device=x_t.device)
        
        # Ensure t has proper shape
        if t.dim() == 0:
            t = t.expand(x_t.shape[:-1])
        elif t.dim() == 1 and x_t.dim() > 1:
            t = t.expand(x_t.shape[:-1])
        
        # Initial state: [x_t, 0] (log-density starts at 0)
        y0 = torch.cat([x_t, torch.zeros(x_t.shape[:-1] + (1,), 
                                       dtype=x_t.dtype, device=x_t.device)], dim=-1)
        
        # Time points for integration (backward from t to 0)
        # odeint expects a 1D time vector, so we need to handle batch integration differently
        if t.dim() == 0:
            # Single time point
            if t.item() == 0.0:
                # Special case: if t=0, we're already at the initial time
                # Return the initial state without integration
                solution = y0.unsqueeze(0)  # Shape: (1, ..., 3)
            else:
                t_points = torch.tensor([t.item(), 0.0], dtype=y0.dtype, device=y0.device)
                solution = odeint(
                    self.ode_system,
                    y0,
                    t_points,
                    rtol=self.rtol,
                    atol=self.atol,
                    method=self.method
                )
        else:
            # Batch of time points - integrate each separately
            batch_size = t.shape[0]
            solutions = []
            for i in range(batch_size):
                if t[i].item() == 0.0:
                    # Special case: if t=0, return two steps to match (2,1,3) shape
                    yi = y0[i:i+1]
                    sol = torch.stack([yi, yi], dim=0)  # Shape: (2, 1, 3)
                else:
                    t_points = torch.tensor([t[i].item(), 0.0], dtype=y0.dtype, device=y0.device)
                    sol = odeint(
                        self.ode_system,
                        y0[i:i+1],  # Single sample
                        t_points,
                        rtol=self.rtol,
                        atol=self.atol,
                        method=self.method
                    )
                solutions.append(sol)
            # Stack solutions
            solution = torch.stack(solutions, dim=1).squeeze(2)  # Shape: (2, batch_size, 3)
        
        # Extract final state (at time 0)
        y_final = solution[-1]  # Shape: (..., 3)
        x_0 = y_final[..., :2]  # Recovered initial state
        log_density_accumulated = y_final[..., 2]  # Accumulated log-density
        
        # Compute log q_t(x) = log p_0(x_0) + ℓ_t
        # where p_0 = N(0, I₂), so log p_0(x_0) = -d/2 log(2π) - |x_0|²/2
        two_pi = x_0.new_tensor(2 * np.pi)
        log_p_0 = -1.0 * torch.log(two_pi) - torch.sum(x_0**2, dim=-1) / 2
        log_q_t = log_p_0 + log_density_accumulated
        
        # Compute ∇log q_t(x) via autograd on the whole log_q_t
        # This correctly captures the Jacobian (∂x_0/∂x_t) through the ODE
        if x_t.requires_grad:
            # compute score once, do not retain graph
            grad_log_q_t = torch.autograd.grad(
                log_q_t.sum(), x_t, create_graph=False, retain_graph=False
            )[0]
        else:
            grad_log_q_t = torch.zeros_like(x_t)
        
        # make the outputs leaf, non-graph tensors (important when called in a loop)
        log_q_t = log_q_t.detach()
        x_0 = x_0.detach()
        grad_log_q_t = grad_log_q_t.detach()
        
        return log_q_t, grad_log_q_t, x_0
    
    def evaluate_grad_log_q_t(self, 
                             x_t: torch.Tensor, 
                             t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate ∇log q_t(x) using autograd through the ODE solver.
        
        Args:
            x_t: Terminal states of shape (..., 2)
            t: Terminal times of shape (...,) or scalar
        
        Returns:
            Score function ∇log q_t(x) of shape (..., 2)
        """
        # Enable gradients for x_t
        x_t.requires_grad_(True)
        
        # Special case: if t=0, q_t = p_0, so grad log q_t = grad log p_0 = -x_t
        if t.dim() == 0 and t.item() == 0.0:
            return -x_t
        elif t.dim() > 0 and torch.all(t == 0.0):
            return -x_t
        
        # Evaluate log q_t(x_t) with gradients enabled
        log_q_t, grad_log_q_t, _ = self.evaluate_q_t(x_t, t)
        
        return grad_log_q_t
    
    def evaluate_full(self, 
                     x_t: torch.Tensor,
                     t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate both log q_t(x) and ∇log q_t(x) efficiently.
        
        Args:
            x_t: Terminal states of shape (..., 2)
            t: Terminal times of shape (...,) or scalar
        
        Returns:
            Tuple of (log_q_t, grad_log_q_t, x_0) where:
            - log_q_t: Log-density of shape (...,)
            - grad_log_q_t: Score function of shape (..., 2)
            - x_0: Recovered initial states of shape (..., 2)
        """
        # Enable gradients for x_t
        x_t.requires_grad_(True)
        
        # Evaluate all quantities efficiently
        log_q_t, grad_log_q_t, x_0 = self.evaluate_q_t(x_t, t)
        
        return log_q_t, grad_log_q_t, x_0


def test_backward_ode():
    """Test backward ODE implementation."""
    print("Testing backward ODE implementation...")
    
    # Set random seed
    torch.manual_seed(42)
    
    # Create a simple velocity field (identity-like)
    class SimpleVelocityField(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.5))
        
        def forward(self, x, t):
            return self.weight * x
        
        def divergence(self, x, t):
            return (2 * self.weight) * torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device)
    
    # Create model and evaluator
    model = SimpleVelocityField().double()
    evaluator = QTEvaluator(model, rtol=1e-6, atol=1e-8)
    
    # Test points
    x_test = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    t_test = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
    
    print(f"Test points: {x_test}")
    print(f"Test times: {t_test}")
    
    # Evaluate q_t(x)
    log_q_t, grad_log_q_t, x_0 = evaluator.evaluate_full(x_test, t_test)
    
    print(f"Log q_t(x): {log_q_t}")
    print(f"Grad log q_t(x): {grad_log_q_t}")
    print(f"Recovered x_0: {x_0}")
    
    # Test reversibility: integrate forward from x_0 to t (simplified for single sample)
    def forward_ode(t, y):
        x = y[:2]  # Single sample
        v = model(x.unsqueeze(0), t).squeeze(0)  # Remove batch dimension
        div_v = model.divergence(x.unsqueeze(0), t).squeeze(0)  # Remove batch dimension
        dx_dt = v  # Forward in time
        dlog_density_dt = div_v
        return torch.cat([dx_dt, dlog_density_dt.unsqueeze(0)], dim=0)
    
    # Test reversibility with first sample only
    x_0_single = x_0[0]  # Shape: (2,)
    y0_forward = torch.cat([x_0_single, torch.zeros(1, dtype=x_0.dtype)], dim=0)  # Shape: (3,)
    t_points_forward = torch.tensor([0.0, t_test[0].item()], dtype=torch.float64)
    
    solution_forward = odeint(
        forward_ode,
        y0_forward,
        t_points_forward,
        rtol=1e-6,
        atol=1e-8,
        method="dopri5"
    )
    
    x_t_recovered = solution_forward[-1][:2]
    x_test_single = x_test[0]
    print(f"Forward recovered x_t: {x_t_recovered}")
    print(f"Original x_t: {x_test_single}")
    print(f"Reversibility error: {torch.norm(x_t_recovered - x_test_single)}")


if __name__ == "__main__":
    test_backward_ode()
