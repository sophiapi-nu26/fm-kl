"""
Ground truth path implementation for KL evolution identity validation.

This module implements the three scalar schedules a(t) and their corresponding
Gaussian paths p_t = N(0, σ_p(t)² I₂) with closed-form expressions for
log-densities and score functions.
"""

import torch
import numpy as np
from typing import Callable, Tuple


class GroundTruthPath:
    """Ground truth Gaussian path driven by velocity field u(x,t) = a(t)x."""
    
    def __init__(self, schedule_type: str = "sin_pi"):
        """
        Initialize ground truth path with one of three schedules.
        
        Args:
            schedule_type: One of "sin_pi", "sin_2pi", "linear"
        """
        self.schedule_type = schedule_type
        self.d = 2  # Dimension
        
        # Define the scalar schedule a(t)
        if schedule_type == "sin_pi":
            self.a_func = lambda t: torch.sin(np.pi * t)
        elif schedule_type == "sin_2pi":
            self.a_func = lambda t: 0.3 * torch.sin(2 * np.pi * t) + 0.2
        elif schedule_type == "linear":
            self.a_func = lambda t: t - 0.5
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    def A(self, t: torch.Tensor) -> torch.Tensor:
        """Compute A(t) = ∫₀ᵗ a(s) ds."""
        if self.schedule_type == "sin_pi":
            return (1 - torch.cos(np.pi * t)) / np.pi
        elif self.schedule_type == "sin_2pi":
            return 0.3 * (1 - torch.cos(2 * np.pi * t)) / (2 * np.pi) + 0.2 * t
        elif self.schedule_type == "linear":
            return 0.5 * t**2 - 0.5 * t
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def sigma_p(self, t: torch.Tensor) -> torch.Tensor:
        """Compute σ_p(t) = exp(A(t))."""
        return torch.exp(self.A(t))
    
    def a(self, t: torch.Tensor) -> torch.Tensor:
        """Compute a(t)."""
        return self.a_func(t)
    
    def u(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """True velocity field u(x,t) = a(t)x."""
        a_t = self.a(t)
        # Ensure proper broadcasting: a_t should have shape (..., 1) for (..., 2) x
        if a_t.dim() == 1 and x.dim() > 1:
            a_t = a_t.unsqueeze(-1)
        return a_t * x
    
    def log_p(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Log-density of p_t(x) = N(0, σ_p(t)² I₂).
        
        log p_t(x) = -d/2 log(2π) - d log σ_p(t) - |x|²/(2σ_p(t)²)
        """
        sigma = self.sigma_p(t)
        return (-self.d / 2 * torch.log(torch.tensor(2 * np.pi, dtype=torch.float64)) 
                - self.d * torch.log(sigma) 
                - torch.sum(x**2, dim=-1) / (2 * sigma**2))
    
    def grad_log_p(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Score function ∇log p_t(x) = -x/σ_p(t)².
        """
        sigma = self.sigma_p(t)
        return -x / (sigma**2)
    
    def sample_p(self, t: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Sample from p_t = N(0, σ_p(t)² I₂).
        
        Args:
            t: Time tensor of shape (batch_size,) or scalar
            batch_size: Number of samples
            
        Returns:
            Samples of shape (batch_size, 2)
        """
        if t.dim() == 0:
            t = t.expand(batch_size)
        
        sigma = self.sigma_p(t)
        z = torch.randn(batch_size, self.d, dtype=torch.float64)
        return sigma.unsqueeze(-1) * z


def test_ground_truth():
    """Test ground truth implementation."""
    print("Testing ground truth implementation...")
    
    for schedule in ["sin_pi", "sin_2pi", "linear"]:
        print(f"\nTesting schedule: {schedule}")
        gt = GroundTruthPath(schedule)
        
        # Test at specific time points
        t_test = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        
        print(f"a(t): {gt.a(t_test)}")
        print(f"A(t): {gt.A(t_test)}")
        print(f"σ_p(t): {gt.sigma_p(t_test)}")
        
        # Test sampling and log-density
        x_samples = gt.sample_p(t_test[1], 1000)
        log_p_vals = gt.log_p(x_samples, t_test[1])
        grad_log_p_vals = gt.grad_log_p(x_samples, t_test[1])
        
        print(f"Sample mean: {x_samples.mean(dim=0)}")
        print(f"Sample std: {x_samples.std(dim=0)}")
        print(f"Expected std: {gt.sigma_p(t_test[1])}")
        print(f"Log-p mean: {log_p_vals.mean()}")
        print(f"Grad log-p shape: {grad_log_p_vals.shape}")


if __name__ == "__main__":
    test_ground_truth()
