"""
Plotting and diagnostics functionality for KL evolution identity validation.

This module provides visualization tools for the experimental results and
diagnostic checks for numerical stability.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime

from ground_truth import GroundTruthPath
from model import VelocityFieldMLP
from backward_ode import QTEvaluator


class KLPlotter:
    """Plotting utilities for KL evolution experiments."""
    
    def __init__(self, save_dir: str = "plots"):
        """
        Initialize plotter.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Generate timestamp for this session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def plot_kl_evolution(self, 
                         results: Dict[str, torch.Tensor],
                         schedule_type: str,
                         save_path: Optional[str] = None) -> None:
        """
        Plot KL evolution curves: KL̂(t) vs R(t).
        
        Args:
            results: Evaluation results dictionary
            schedule_type: Type of schedule ("sin_pi", "sin_2pi", "linear")
            save_path: Optional path to save plot
        """
        # Convert to numpy arrays (handle both tensors and arrays)
        def to_numpy(x):
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
            return np.asarray(x)
        
        t_grid = to_numpy(results['t_grid'])
        kl_estimator = to_numpy(results['kl_estimator'])
        rhs_cumulative = to_numpy(results['rhs_cumulative'])
        relative_error = to_numpy(results['relative_error'])
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Main plot: KL curves
        ax1.plot(t_grid, kl_estimator, 'b-', linewidth=2, label='KL̂(t)', alpha=0.8)
        ax1.plot(t_grid, rhs_cumulative, 'r--', linewidth=2, label='R(t)', alpha=0.8)
        ax1.fill_between(t_grid, kl_estimator, rhs_cumulative, alpha=0.2, color='gray')
        
        ax1.set_xlabel('Time t')
        ax1.set_ylabel('KL Divergence')
        ax1.set_title(f'KL Evolution Identity Validation - {schedule_type}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        
        # Relative error plot
        ax2.semilogy(t_grid, relative_error, 'g-', linewidth=1.5, alpha=0.8)
        ax2.axhline(y=0.03, color='orange', linestyle=':', alpha=0.7, label='3% threshold')
        ax2.axhline(y=0.08, color='red', linestyle=':', alpha=0.7, label='8% threshold')
        
        ax2.set_xlabel('Time t')
        ax2.set_ylabel('Relative Error')
        ax2.set_title('Relative Error: |R(t) - KL̂(t)| / max(10⁻⁸, KL̂(t))')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        
        # Add error statistics text
        error_stats = results['error_stats']
        stats_text = (f"Median: {error_stats['median']:.3f}\n"
                     f"Mean: {error_stats['mean']:.3f}\n"
                     f"Max: {error_stats['max']:.3f}")
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.save_dir, f'{schedule_type}_kl_evolution_{self.timestamp}.pdf')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()  # Close figure instead of showing
        print(f"Plot saved to: {save_path}")
    
    def plot_training_curves(self, 
                           train_losses: List[float],
                           val_losses: List[float],
                           schedule_type: str,
                           save_path: Optional[str] = None) -> None:
        """
        Plot training and validation curves.
        
        Args:
            train_losses: Training losses
            val_losses: Validation losses
            schedule_type: Type of schedule
            save_path: Optional path to save plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epochs = range(len(train_losses))
        ax.plot(epochs, train_losses, label='Training Loss', alpha=0.7, linewidth=1)
        
        if val_losses:
            val_epochs = np.linspace(0, len(train_losses)-1, len(val_losses))
            ax.plot(val_epochs, val_losses, label='Validation Loss', 
                   marker='o', markersize=3, linewidth=1.5)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Training Progress - {schedule_type}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.save_dir, f'{schedule_type}_training_{self.timestamp}.pdf')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()  # Close figure instead of showing
        print(f"Training plot saved to: {save_path}")
    
    def plot_velocity_field_comparison(self, 
                                     ground_truth: GroundTruthPath,
                                     velocity_field: VelocityFieldMLP,
                                     t_values: List[float],
                                     save_path: Optional[str] = None) -> None:
        """
        Plot velocity field comparison at different time points.
        
        Args:
            ground_truth: Ground truth path
            velocity_field: Learned velocity field
            t_values: Time points to plot
            save_path: Optional path to save plot
        """
        n_times = len(t_values)
        fig, axes = plt.subplots(2, n_times, figsize=(4*n_times, 8))
        
        if n_times == 1:
            axes = axes.reshape(2, 1)
        
        # Create spatial grid
        x_range = np.linspace(-3, 3, 20)
        y_range = np.linspace(-3, 3, 20)
        X, Y = np.meshgrid(x_range, y_range)
        points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float64)
        
        for i, t in enumerate(t_values):
            t_tensor = torch.tensor(t, dtype=torch.float64)
            
            # True velocity field
            with torch.no_grad():
                u_true = ground_truth.u(points, t_tensor)
                u_pred = velocity_field(points, t_tensor)
            
            # Reshape for plotting
            U_true = u_true[:, 0].reshape(X.shape).numpy()
            V_true = u_true[:, 1].reshape(X.shape).numpy()
            U_pred = u_pred[:, 0].reshape(X.shape).numpy()
            V_pred = u_pred[:, 1].reshape(X.shape).numpy()
            
            # Plot true velocity field
            axes[0, i].quiver(X, Y, U_true, V_true, alpha=0.7)
            axes[0, i].set_title(f'True Velocity Field (t={t:.2f})')
            axes[0, i].set_xlabel('x₁')
            axes[0, i].set_ylabel('x₂')
            axes[0, i].grid(True, alpha=0.3)
            
            # Plot learned velocity field
            axes[1, i].quiver(X, Y, U_pred, V_pred, alpha=0.7)
            axes[1, i].set_title(f'Learned Velocity Field (t={t:.2f})')
            axes[1, i].set_xlabel('x₁')
            axes[1, i].set_ylabel('x₂')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.save_dir, f'velocity_field_comparison_{self.timestamp}.pdf')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()  # Close figure instead of showing
        print(f"Velocity field plot saved to: {save_path}")


class KLDiagnostics:
    """Diagnostic utilities for numerical validation."""
    
    def __init__(self, ground_truth: GroundTruthPath, velocity_field: VelocityFieldMLP):
        """
        Initialize diagnostics.
        
        Args:
            ground_truth: Ground truth path
            velocity_field: Learned velocity field
        """
        self.ground_truth = ground_truth
        self.velocity_field = velocity_field
        self.q_evaluator = QTEvaluator(velocity_field)
    
    def test_reversibility(self, 
                          num_tests: int = 10,
                          tolerance: float = 1e-6) -> Dict[str, float]:
        """
        Test reversibility: backward → forward integration.
        
        Args:
            num_tests: Number of reversibility tests
            tolerance: Tolerance for reversibility check
        
        Returns:
            Dictionary with reversibility statistics
        """
        print(f"Testing reversibility with {num_tests} random points...")
        
        errors = []
        
        for i in range(num_tests):
            # Random test point
            x_t = torch.randn(2, dtype=torch.float64) * 2
            t = torch.rand(1, dtype=torch.float64).item()
            
            # Backward integration to get x_0
            print(f"Debug test {i+1}: About to call evaluate_full")
            try:
                _, _, x_0 = self.q_evaluator.evaluate_full(x_t.unsqueeze(0), t)
                print(f"Debug test {i+1}: evaluate_full succeeded")
                x_0 = x_0.squeeze(0).detach()  # Detach to avoid keeping computation graph
                print(f"Debug test {i+1}: x_0 shape={x_0.shape}, requires_grad={x_0.requires_grad}")
            except Exception as e:
                print(f"Debug test {i+1}: evaluate_full failed with error: {e}")
                import traceback
                print(f"Debug test {i+1}: Full traceback:")
                traceback.print_exc()
                raise
            
            # Forward integration from x_0 to t
            def forward_ode(t_val, y):
                x = y[:2]
                # Convert t_val to tensor if it's a scalar
                if not torch.is_tensor(t_val):
                    t_val = torch.tensor(t_val, dtype=x.dtype, device=x.device)
                v = self.velocity_field(x.unsqueeze(0), t_val).squeeze(0)
                div_v = self.velocity_field.divergence(x.unsqueeze(0), t_val).squeeze(0)
                dx_dt = v
                dlog_density_dt = div_v
                return torch.cat([dx_dt, dlog_density_dt.unsqueeze(0)])
            
            from torchdiffeq import odeint
            
            y0 = torch.cat([x_0, torch.zeros(1, dtype=x_0.dtype, device=x_0.device)])
            t_points = torch.tensor([0.0, t], dtype=x_0.dtype, device=x_0.device)
            print(f"Debug test {i+1}: y0 shape={y0.shape}, requires_grad={y0.requires_grad}")
            print(f"Debug test {i+1}: t_points shape={t_points.shape}")
            
            try:
                solution = odeint(forward_ode, y0, t_points, rtol=1e-6, atol=1e-8)
                print(f"Debug test {i+1}: odeint succeeded, solution shape={solution.shape}")
                x_t_recovered = solution[-1][:2]
                print(f"Debug test {i+1}: x_t_recovered shape={x_t_recovered.shape}")
            except Exception as e:
                print(f"Debug test {i+1}: odeint failed with error: {e}")
                import traceback
                print(f"Debug test {i+1}: Full traceback:")
                traceback.print_exc()
                raise
            
            # Compute error
            error = torch.norm(x_t_recovered - x_t).item()
            errors.append(error)
            
            if i < 3:  # Print first few results
                print(f"  Test {i+1}: error = {error:.2e}")
        
        errors = np.array(errors)
        
        stats = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'passed_tolerance': np.sum(errors < tolerance),
            'total_tests': num_tests
        }
        
        print(f"Reversibility test results:")
        print(f"  Mean error: {stats['mean_error']:.2e}")
        print(f"  Std error: {stats['std_error']:.2e}")
        print(f"  Max error: {stats['max_error']:.2e}")
        print(f"  Tests within tolerance: {stats['passed_tolerance']}/{stats['total_tests']}")
        
        return stats
    
    def test_convergence(self, 
                        K_values: List[int] = [5, 11, 21],
                        N_values: List[int] = [50, 100, 200],
                        seed: int = 42) -> Dict[str, Dict]:
        """
        Test convergence with different grid sizes.
        
        Args:
            K_values: Different numbers of time points
            N_values: Different numbers of samples
            seed: Random seed
        
        Returns:
            Dictionary with convergence results
        """
        print("Testing convergence with different grid sizes...")
        
        from evaluation import KLEvolutionEvaluator
        
        evaluator = KLEvolutionEvaluator(self.ground_truth, self.velocity_field)
        
        results = {}
        
        # Test different K values
        print("Testing different K values (time points)...")
        K_results = {}
        for K in K_values:
            print(f"  K = {K}")
            result = evaluator.evaluate(K=K, N=100, seed=seed)  # Use smaller N for testing
            K_results[K] = result['error_stats']
        
        results['K_convergence'] = K_results
        
        # Test different N values
        print("Testing different N values (samples)...")
        N_results = {}
        for N in N_values:
            print(f"  N = {N}")
            result = evaluator.evaluate(K=11, N=N, seed=seed)  # Use smaller K for testing
            N_results[N] = result['error_stats']
        
        results['N_convergence'] = N_results
        
        return results
    
    def test_tolerance_sensitivity(self, 
                                  rtol_values: List[float] = [1e-4, 1e-5, 1e-6, 1e-7],
                                  atol_values: List[float] = [1e-6, 1e-7, 1e-8, 1e-9],
                                  K: int = 51, N: int = 1000) -> Dict[str, Dict]:
        """
        Test sensitivity to ODE solver tolerances.
        
        Args:
            rtol_values: Different relative tolerances
            atol_values: Different absolute tolerances
            K: Number of time points
            N: Number of samples
        
        Returns:
            Dictionary with tolerance sensitivity results
        """
        print("Testing tolerance sensitivity...")
        
        from evaluation import KLEvolutionEvaluator
        
        results = {}
        
        # Test rtol sensitivity
        print("Testing rtol sensitivity...")
        rtol_results = {}
        for rtol in rtol_values:
            print(f"  rtol = {rtol}")
            evaluator = KLEvolutionEvaluator(self.ground_truth, self.velocity_field, 
                                           rtol=rtol, atol=1e-8)
            result = evaluator.evaluate(K=K, N=N, seed=42)
            rtol_results[rtol] = result['error_stats']
        
        results['rtol_sensitivity'] = rtol_results
        
        # Test atol sensitivity
        print("Testing atol sensitivity...")
        atol_results = {}
        for atol in atol_values:
            print(f"  atol = {atol}")
            evaluator = KLEvolutionEvaluator(self.ground_truth, self.velocity_field, 
                                           rtol=1e-6, atol=atol)
            result = evaluator.evaluate(K=K, N=N, seed=42)
            atol_results[atol] = result['error_stats']
        
        results['atol_sensitivity'] = atol_results
        
        return results


def test_plotting():
    """Test plotting functionality."""
    print("Testing plotting functionality...")
    
    # Create dummy data
    t_grid = torch.linspace(0, 1, 21)
    kl_estimator = torch.exp(-t_grid) * 0.1  # Decaying KL
    rhs_cumulative = kl_estimator + torch.randn_like(kl_estimator) * 0.01
    relative_error = torch.abs(rhs_cumulative - kl_estimator) / torch.maximum(
        torch.tensor(1e-8), torch.abs(kl_estimator))
    
    results = {
        't_grid': t_grid,
        'kl_estimator': kl_estimator,
        'rhs_cumulative': rhs_cumulative,
        'relative_error': relative_error,
        'error_stats': {
            'median': torch.median(relative_error).item(),
            'mean': torch.mean(relative_error).item(),
            'max': torch.max(relative_error).item()
        }
    }
    
    # Test plotter
    plotter = KLPlotter()
    plotter.plot_kl_evolution(results, "test_schedule")


if __name__ == "__main__":
    test_plotting()
