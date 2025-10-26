"""
Evaluation grid and KL/RHS computation for KL evolution identity validation.

This module implements the evaluation grid (time × samples matrix) and
computes the KL divergence and RHS integrand at each time point.
"""

import torch
import numpy as np
from typing import Tuple, Dict, List, TYPE_CHECKING
import time

from ground_truth import GroundTruthPath
from backward_ode import QTEvaluator

if TYPE_CHECKING:
    from model import VelocityFieldMLP


class KLEvolutionEvaluator:
    """
    Evaluator for KL evolution identity validation.
    
    Computes KL(p_t|q_t) and ∫₀ᵗ E[(u-v_θ)ᵀ(∇log p_s - ∇log q_s)] ds
    on a time × samples evaluation grid.
    """
    
    def __init__(self,
                 ground_truth: GroundTruthPath,
                 velocity_field: "VelocityFieldMLP",
                 rtol: float = 1e-6,
                 atol: float = 1e-8):
        """
        Initialize evaluator.
        
        Args:
            ground_truth: Ground truth path
            velocity_field: Trained velocity field model
            rtol: Relative tolerance for ODE solver
            atol: Absolute tolerance for ODE solver
        """
        self.ground_truth = ground_truth
        self.velocity_field = velocity_field
        self.q_evaluator = QTEvaluator(velocity_field, rtol=rtol, atol=atol)
        
    def create_evaluation_grid(self, 
                              K: int = 101, 
                              N: int = 2000,
                              seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create evaluation grid: time axis × sample axis.
        
        Args:
            K: Number of time points (K=101)
            N: Number of samples per time point (N=2000)
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (t_grid, X_grid) where:
            - t_grid: Time points of shape (K,)
            - X_grid: Samples of shape (K, N, 2)
        """
        torch.manual_seed(seed)
        
        # Time grid: t_k = k/(K-1) for k = 0, ..., K-1
        t_grid = torch.linspace(0.0, 1.0, K, dtype=torch.float64)
        
        # Sample grid: for each t_k, sample N points from p_{t_k}
        X_grid = torch.zeros(K, N, 2, dtype=torch.float64)
        
        # Cast/move to model's device/dtype
        dev = next(self.velocity_field.parameters()).device
        dtype = next(self.velocity_field.parameters()).dtype
        t_grid = t_grid.to(device=dev, dtype=dtype)
        X_grid = X_grid.to(device=dev, dtype=dtype)
        
        print(f"Creating evaluation grid: {K} time points × {N} samples")
        for k in range(K):
            t_k = t_grid[k]
            xk = self.ground_truth.sample_p(t_k, N)
            X_grid[k] = xk.to(device=dev, dtype=dtype)
        
        return t_grid, X_grid
    
    def compute_grid_quantities(self, 
                               t_grid: torch.Tensor, 
                               X_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all quantities for the evaluation grid.
        
        Args:
            t_grid: Time points of shape (K,)
            X_grid: Samples of shape (K, N, 2)
        
        Returns:
            Dictionary containing all computed quantities
        """
        K, N, d = X_grid.shape
        print(f"Computing quantities for {K}×{N} grid...")
        
        # Initialize arrays
        dev = X_grid.device
        dtype = X_grid.dtype
        log_p = torch.zeros(K, N, device=dev, dtype=dtype)
        s_p = torch.zeros(K, N, d, device=dev, dtype=dtype)
        log_q = torch.zeros(K, N, device=dev, dtype=dtype)
        s_q = torch.zeros(K, N, d, device=dev, dtype=dtype)
        u = torch.zeros(K, N, d, device=dev, dtype=dtype)
        v = torch.zeros(K, N, d, device=dev, dtype=dtype)
        
        # Compute quantities for each time point
        for k in range(K):
            t_k = t_grid[k]
            x_k = X_grid[k]  # Shape: (N, 2)
            
            print(f"Processing time point {k+1}/{K} (t={t_k.item():.3f})")
            
            # Ground truth quantities (analytic)
            print("Computing ground truth quantities...")
            with torch.no_grad():
                log_p[k] = self.ground_truth.log_p(x_k, t_k)
                s_p[k] = self.ground_truth.grad_log_p(x_k, t_k)
                u[k] = self.ground_truth.u(x_k, t_k)
            print("Ground truth quantities computed."
                  f"log_p: {log_p[k].shape}\n"
                  f"s_p: {s_p[k].shape}\n"
                  f"u: {u[k].shape}\n")
            
            # Learned quantities (via backward ODE)
            # Note: We need gradients enabled for autograd through ODE solver
            print("Computing learned quantities...")
            x_k_req = x_k.detach().clone().requires_grad_(True)   # enable grads for score
            # grad is required; ensure grad mode is on only here
            with torch.enable_grad():
                log_q_temp, s_q_temp, _ = self.q_evaluator.evaluate_full(x_k_req, t_k)
            # evaluate_q_t already returns detached tensors per (1)
            log_q[k] = log_q_temp.reshape(-1)  # always (N,)
            s_q[k] = s_q_temp
            print("Learned quantities computed."
                  f"log_q: {log_q[k].shape}\n"
                  f"s_q: {s_q[k].shape}\n")
            
            # Velocity field evaluation
            t_in = t_k.expand(x_k.shape[0])  # shape (N,)
            with torch.no_grad():
                v[k] = self.velocity_field(x_k, t_in)
        
        return {
            'log_p': log_p,
            's_p': s_p,
            'log_q': log_q,
            's_q': s_q,
            'u': u,
            'v': v
        }
    
    def compute_kl_estimator(self, log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
        """
        Compute KL estimator: KL̂(t_k) = (1/N) Σᵢ [log p_{t_k}(X[k,i]) - log q_{t_k}(X[k,i])]
        
        Args:
            log_p: Log-densities of shape (K, N)
            log_q: Log-densities of shape (K, N)
        
        Returns:
            KL estimates of shape (K,)
        """
        return torch.mean(log_p - log_q, dim=1)
    
    def compute_rhs_integrand(self, 
                             u: torch.Tensor, 
                             v: torch.Tensor,
                             s_p: torch.Tensor, 
                             s_q: torch.Tensor) -> torch.Tensor:
        """
        Compute RHS integrand: ĝ(t_k) = (1/N) Σᵢ [(u[k,i] - v[k,i])ᵀ(s_p[k,i] - s_q[k,i])]
        
        Args:
            u: True velocities of shape (K, N, 2)
            v: Learned velocities of shape (K, N, 2)
            s_p: True scores of shape (K, N, 2)
            s_q: Learned scores of shape (K, N, 2)
        
        Returns:
            RHS integrand estimates of shape (K,)
        """
        # Compute dot products: (u - v)ᵀ(s_p - s_q)
        dot_products = torch.sum((u - v) * (s_p - s_q), dim=-1)  # Shape: (K, N)
        
        # Average over samples
        return torch.mean(dot_products, dim=1)  # Shape: (K,)
    
    def compute_rhs_cumulative(self, 
                             rhs_integrand: torch.Tensor, 
                             t_grid: torch.Tensor) -> torch.Tensor:
        """
        Compute cumulative RHS integral using trapezoidal rule.
        
        Args:
            rhs_integrand: RHS integrand estimates of shape (K,)
            t_grid: Time points of shape (K,)
        
        Returns:
            Cumulative RHS integral of shape (K,)
        """
        K = len(t_grid)
        rhs_cumulative = torch.zeros(K, device=t_grid.device, dtype=t_grid.dtype)
        
        # R[0] = 0
        rhs_cumulative[0] = t_grid.new_tensor(0.0)
        
        # Trapezoidal rule: R[m] = R[m-1] + 0.5 * (t_m - t_{m-1}) * (ĝ(t_m) + ĝ(t_{m-1}))
        for m in range(1, K):
            dt = t_grid[m] - t_grid[m-1]
            rhs_cumulative[m] = (rhs_cumulative[m-1] + 
                               0.5 * dt * (rhs_integrand[m] + rhs_integrand[m-1]))
        
        return rhs_cumulative
    
    def compute_relative_error(self, 
                              kl_estimator: torch.Tensor, 
                              rhs_cumulative: torch.Tensor) -> torch.Tensor:
        """
        Compute relative error: |R[k] - KL̂(t_k)| / max(10⁻⁸, KL̂(t_k))
        
        Args:
            kl_estimator: KL estimates of shape (K,)
            rhs_cumulative: Cumulative RHS integral of shape (K,)
        
        Returns:
            Relative errors of shape (K,)
        """
        numerator = torch.abs(rhs_cumulative - kl_estimator)
        eps = kl_estimator.new_tensor(1e-8)
        denominator = torch.maximum(eps, torch.abs(kl_estimator))
        return numerator / denominator
    
    def evaluate(self, 
                 K: int = 101, 
                 N: int = 2000,
                 seed: int = 42) -> Dict[str, torch.Tensor]:
        """
        Complete evaluation of KL evolution identity.
        
        Args:
            K: Number of time points
            N: Number of samples per time point
            seed: Random seed
        
        Returns:
            Dictionary containing all evaluation results
        """
        print(f"Starting KL evolution evaluation...")
        print(f"Schedule: {self.ground_truth.schedule_type}")
        print(f"Grid: {K} time points × {N} samples")
        
        start_time = time.time()
        
        # Create evaluation grid
        t_grid, X_grid = self.create_evaluation_grid(K, N, seed)
        
        # Compute all quantities
        quantities = self.compute_grid_quantities(t_grid, X_grid)
        
        # Compute estimators
        kl_estimator = self.compute_kl_estimator(quantities['log_p'], quantities['log_q'])
        rhs_integrand = self.compute_rhs_integrand(
            quantities['u'], quantities['v'],
            quantities['s_p'], quantities['s_q']
        )
        rhs_cumulative = self.compute_rhs_cumulative(rhs_integrand, t_grid)
        relative_error = self.compute_relative_error(kl_estimator, rhs_cumulative)
        
        elapsed_time = time.time() - start_time
        print(f"Evaluation completed in {elapsed_time:.2f} seconds")
        
        # Compute error statistics
        median_error = torch.median(relative_error).item()
        max_error = torch.max(relative_error).item()
        mean_error = torch.mean(relative_error).item()
        
        print(f"Error statistics:")
        print(f"  Median relative error: {median_error:.4f}")
        print(f"  Mean relative error: {mean_error:.4f}")
        print(f"  Max relative error: {max_error:.4f}")
        
        return {
            't_grid': t_grid,
            'X_grid': X_grid,
            'kl_estimator': kl_estimator,
            'rhs_integrand': rhs_integrand,
            'rhs_cumulative': rhs_cumulative,
            'relative_error': relative_error,
            'quantities': quantities,
            'error_stats': {
                'median': median_error,
                'mean': mean_error,
                'max': max_error
            },
            'elapsed_time': elapsed_time
        }


def test_evaluation():
    """Test evaluation implementation."""
    print("Testing KL evolution evaluation...")
    
    # Set random seed
    torch.manual_seed(42)
    
    # Create ground truth and a simple model
    gt = GroundTruthPath("sin_pi")
    
    # Create a simple velocity field that's close to the true one
    class SimpleVelocityField(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.8))  # Close to sin(πt)
        
        def forward(self, x, t):
            return self.weight * x
        
        def divergence(self, x, t):
            return (2 * self.weight) * torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device)
    
    model = SimpleVelocityField().double()
    
    # Create evaluator
    evaluator = KLEvolutionEvaluator(gt, model)
    
    # Run evaluation with small grid for testing
    results = evaluator.evaluate(K=11, N=100, seed=42)
    
    print(f"Time grid shape: {results['t_grid'].shape}")
    print(f"KL estimator shape: {results['kl_estimator'].shape}")
    print(f"RHS cumulative shape: {results['rhs_cumulative'].shape}")
    print(f"Relative error shape: {results['relative_error'].shape}")
    
    print(
        f"KL estimator range: "
        f"[{results['kl_estimator'].min().item():.6f}, {results['kl_estimator'].max().item():.6f}]"
    )
    print(
        f"RHS cumulative range: "
        f"[{results['rhs_cumulative'].min().item():.6f}, {results['rhs_cumulative'].max().item():.6f}]"
    )


if __name__ == "__main__":
    test_evaluation()
