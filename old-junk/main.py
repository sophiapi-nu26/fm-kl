"""
Main experiment script for KL evolution identity validation.

This script runs the complete experiment for all three scalar schedules:
- sin(πt)
- 0.3 sin(2πt) + 0.2  
- t - 0.5

It trains velocity field models, evaluates the KL evolution identity,
and generates all required plots and diagnostics.
"""

import torch
import numpy as np
import os
import json
import argparse
from datetime import datetime
import subprocess
from typing import Dict

from ground_truth import GroundTruthPath
from model import VelocityFieldMLP
from training import FlowMatchingTrainer
from evaluation import KLEvolutionEvaluator
from plotting import KLPlotter, KLDiagnostics


def _to_numpy(x):
    """Convert tensor or array to numpy array, handling device/dtype."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True)
        return result.stdout.strip()[:8]
    except:
        return "unknown"


def setup_experiment(seed: int = 42, device: str = "cpu") -> Dict:
    """
    Set up experiment environment.
    
    Args:
        seed: Random seed
        device: Device to use
    
    Returns:
        Experiment metadata
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set PyTorch default dtype to float64
    torch.set_default_dtype(torch.float64)
    
    # Validate device
    if device.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA requested but not available"
        torch.cuda.manual_seed_all(seed)
    
    # Create output directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Get experiment metadata
    metadata = {
        'seed': seed,
        'device': device,
        'timestamp': datetime.now().isoformat(),
        'git_hash': get_git_hash(),
        'torch_version': torch.__version__,
        'numpy_version': np.__version__
    }
    
    print(f"Experiment setup:")
    print(f"  Seed: {seed}")
    print(f"  Device: {device}")
    print(f"  Git hash: {metadata['git_hash']}")
    print(f"  Timestamp: {metadata['timestamp']}")
    
    return metadata


def train_model(schedule_type: str, 
                metadata: Dict,
                num_epochs: int = 2000,
                learning_rate: float = 1e-3,
                batch_size: int = 1024,
                target_val_loss: float = 1e-4) -> Dict:
    """
    Train velocity field model for given schedule.
    
    Args:
        schedule_type: Type of schedule ("sin_pi", "sin_2pi", "linear")
        metadata: Experiment metadata
        num_epochs: Maximum number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        target_val_loss: Target validation loss
    
    Returns:
        Training results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Training model for schedule: {schedule_type}")
    print(f"{'='*60}")
    
    # Create ground truth and model
    ground_truth = GroundTruthPath(schedule_type)
    model = VelocityFieldMLP(hidden_dim=128, num_layers=3, activation="silu")
    model = model.double()
    
    # Create trainer
    trainer = FlowMatchingTrainer(
        model=model,
        ground_truth=ground_truth,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=metadata['device']
    )
    
    # Train model
    training_history = trainer.train(
        num_epochs=num_epochs,
        val_freq=100,
        target_val_loss=target_val_loss,
        patience=200
    )
    
    # Save model
    checkpoint_path = f"checkpoints/model_{schedule_type}.pt"
    trainer.save_model(checkpoint_path)
    
    # Plot training curves
    plotter = KLPlotter()
    plotter.plot_training_curves(
        training_history['train_losses'],
        training_history['val_losses'],
        schedule_type
    )
    
    # Test model performance
    dev = next(model.parameters()).device
    x_test = torch.randn(1000, 2, dtype=torch.float64, device=dev)
    t_test = torch.rand(1000, dtype=torch.float64, device=dev)
    
    with torch.no_grad():
        u_pred = model(x_test, t_test)
        u_true = ground_truth.u(x_test, t_test)
        test_mse = torch.nn.MSELoss()(u_pred, u_true).item()
    
    print(f"Final test MSE: {test_mse:.6f}")
    
    return {
        'model': model,
        'ground_truth': ground_truth,
        'training_history': training_history,
        'test_mse': test_mse,
        'checkpoint_path': checkpoint_path
    }


def evaluate_model(training_results: Dict,
                  schedule_type: str,
                  K: int = 101,
                  N: int = 2000,
                  rtol: float = 1e-6,
                  atol: float = 1e-8) -> Dict:
    """
    Evaluate KL evolution identity for trained model.
    
    Args:
        training_results: Results from training
        schedule_type: Type of schedule
        K: Number of time points
        N: Number of samples per time point
        rtol: Relative tolerance for ODE solver
        atol: Absolute tolerance for ODE solver
    
    Returns:
        Evaluation results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Evaluating KL evolution identity for: {schedule_type}")
    print(f"{'='*60}")
    
    model = training_results['model']
    ground_truth = training_results['ground_truth']
    
    # Create evaluator
    evaluator = KLEvolutionEvaluator(
        ground_truth=ground_truth,
        velocity_field=model,
        rtol=rtol,
        atol=atol
    )
    
    # Set evaluator to same device as model
    dev = next(model.parameters()).device
    if hasattr(evaluator, 'to'):
        evaluator.to(dev)
    
    # Run evaluation
    results = evaluator.evaluate(K=K, N=N, seed=42)
    
    # Plot results
    plotter = KLPlotter()
    plotter.plot_kl_evolution(
        {
            't_grid': _to_numpy(results['t_grid']),
            'kl_estimator': _to_numpy(results['kl_estimator']),
            'rhs_integrand': _to_numpy(results['rhs_integrand']),
            'rhs_cumulative': _to_numpy(results['rhs_cumulative']),
            'relative_error': _to_numpy(results['relative_error']),
            'error_stats': results['error_stats'],  # stays a dict
            'elapsed_time': results['elapsed_time'],
        },
        schedule_type
    )
    
    # Save results
    results_path = f"results/evaluation_{schedule_type}.npz"
    error_stats_json = json.dumps(results['error_stats'], default=float)
    np.savez(results_path,
             t_grid=_to_numpy(results['t_grid']),
             kl_estimator=_to_numpy(results['kl_estimator']),
             rhs_integrand=_to_numpy(results['rhs_integrand']),
             rhs_cumulative=_to_numpy(results['rhs_cumulative']),
             relative_error=_to_numpy(results['relative_error']),
             error_stats=error_stats_json,
             elapsed_time=float(results['elapsed_time']))
    
    print(f"Results saved to: {results_path}")
    
    return results


def run_diagnostics(training_results: Dict,
                   schedule_type: str) -> Dict:
    """
    Run diagnostic checks for numerical validation.
    
    Args:
        training_results: Results from training
        schedule_type: Type of schedule
    
    Returns:
        Diagnostics results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Running diagnostics for: {schedule_type}")
    print(f"{'='*60}")
    
    model = training_results['model']
    ground_truth = training_results['ground_truth']
    
    # Create diagnostics
    dev = next(model.parameters()).device
    diagnostics = KLDiagnostics(ground_truth, model)
    if hasattr(diagnostics, 'to'):
        diagnostics.to(dev)
    
    # Test reversibility
    reversibility_stats = diagnostics.test_reversibility(num_tests=20)
    
    # Test convergence
    convergence_results = diagnostics.test_convergence(
        K_values=[5, 11, 21],
        N_values=[50, 100, 200]
    )
    
    # Test tolerance sensitivity
    tolerance_results = diagnostics.test_tolerance_sensitivity(
        rtol_values=[1e-5, 1e-6, 1e-7],
        atol_values=[1e-7, 1e-8, 1e-9]
    )
    
    # Save diagnostics
    diagnostics_path = f"results/diagnostics_{schedule_type}.json"
    diagnostics_data = {
        'reversibility': reversibility_stats,
        'convergence': convergence_results,
        'tolerance_sensitivity': tolerance_results
    }
    
    with open(diagnostics_path, 'w') as f:
        json.dump(diagnostics_data, f, indent=2, default=str)
    
    print(f"Diagnostics saved to: {diagnostics_path}")
    
    return diagnostics_data


def run_single_schedule(schedule_type: str,
                       metadata: Dict,
                       config: Dict) -> Dict:
    """
    Run complete experiment for a single schedule.
    
    Args:
        schedule_type: Type of schedule
        metadata: Experiment metadata
        config: Experiment configuration
    
    Returns:
        Complete results for this schedule
    """
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT FOR SCHEDULE: {schedule_type.upper()}")
    print(f"{'='*80}")
    
    # Train model
    training_results = train_model(
        schedule_type=schedule_type,
        metadata=metadata,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        target_val_loss=config['target_val_loss']
    )
    
    # Evaluate KL evolution identity
    evaluation_results = evaluate_model(
        training_results=training_results,
        schedule_type=schedule_type,
        K=config['K'],
        N=config['N'],
        rtol=config['rtol'],
        atol=config['atol']
    )
    
    # Run diagnostics
    diagnostics_results = run_diagnostics(training_results, schedule_type)
    
    # Check acceptance criteria
    error_stats = evaluation_results['error_stats']
    passed_criteria = (
        error_stats['median'] <= 0.03 and  # Median relative error ≤ 3%
        error_stats['max'] <= 0.08         # Max relative error ≤ 8%
    )
    
    print(f"\nAcceptance criteria check:")
    print(f"  Median relative error: {error_stats['median']:.4f} (≤ 0.03: {'✓' if error_stats['median'] <= 0.03 else '✗'})")
    print(f"  Max relative error: {error_stats['max']:.4f} (≤ 0.08: {'✓' if error_stats['max'] <= 0.08 else '✗'})")
    print(f"  Overall: {'PASSED' if passed_criteria else 'FAILED'}")
    
    return {
        'schedule_type': schedule_type,
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'diagnostics_results': diagnostics_results,
        'passed_criteria': passed_criteria
    }
    
    # # DEBUG: exit gracefully
    # exit(0)


def main():
    """Main experiment function."""
    parser = argparse.ArgumentParser(description='KL Evolution Identity Validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--schedules', nargs='+', 
                       default=['sin_pi', 'sin_2pi', 'linear'],
                       help='Schedules to run')
    parser.add_argument('--num_epochs', type=int, default=2000, help='Training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--target_val_loss', type=float, default=1e-4, help='Target validation loss')
    parser.add_argument('--K', type=int, default=101, help='Number of time points')
    parser.add_argument('--N', type=int, default=2000, help='Number of samples per time point')
    parser.add_argument('--rtol', type=float, default=1e-6, help='ODE solver relative tolerance')
    parser.add_argument('--atol', type=float, default=1e-8, help='ODE solver absolute tolerance')
    
    args = parser.parse_args()
    
    # Setup experiment
    metadata = setup_experiment(seed=args.seed, device=args.device)
    
    # Experiment configuration
    config = {
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'target_val_loss': args.target_val_loss,
        'K': args.K,
        'N': args.N,
        'rtol': args.rtol,
        'atol': args.atol
    }
    
    print(f"\nExperiment configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Run experiments for all schedules
    all_results = {}
    
    for schedule_type in args.schedules:
        try:
            results = run_single_schedule(schedule_type, metadata, config)
            all_results[schedule_type] = results
        except Exception as e:
            print(f"Error running schedule {schedule_type}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    for schedule_type, results in all_results.items():
        error_stats = results['evaluation_results']['error_stats']
        passed = results['passed_criteria']
        print(f"{schedule_type:12s}: Median={error_stats['median']:.4f}, "
              f"Max={error_stats['max']:.4f}, {'PASSED' if passed else 'FAILED'}")
    
    # Save complete results
    summary_path = "results/experiment_summary.json"
    summary_data = {
        'metadata': metadata,
        'config': config,
        'results': {
            schedule: {
                'error_stats': results['evaluation_results']['error_stats'],
                'passed_criteria': results['passed_criteria'],
                'test_mse': results['training_results']['test_mse']
            }
            for schedule, results in all_results.items()
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    print(f"\nComplete results saved to: {summary_path}")
    print("Experiment completed!")


if __name__ == "__main__":
    main()
