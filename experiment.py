"""
Main experiment script to verify KL divergence identity.

Usage:
    python experiment.py --schedule a1 --seed 42
    python experiment.py --schedule a2
    python experiment.py --schedule a3 --num_samples 4000 --num_times 201
"""

import argparse
import numpy as np
import torch
from tqdm import tqdm

from utils import (
    set_seed, get_device, ensure_dirs, save_checkpoint, 
    load_checkpoint, save_results, compute_relative_error
)
from model import VelocityMLP
from true_path import Schedule, schedule_to_enum, sample_p_t
from train import train_velocity_model, validate_model
from eval import compute_kl_lhs, compute_rhs_integrand, integrate_rhs


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Verify KL divergence identity via flow matching')
    
    # Schedule
    parser.add_argument('--schedule', type=str, choices=['a1', 'a2', 'a3'], required=True,
                        help='Schedule to use')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    
    # Evaluation parameters
    parser.add_argument('--num_samples', type=int, default=2000,
                        help='Number of samples per time point')
    parser.add_argument('--num_times', type=int, default=101,
                        help='Number of time points in grid')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to saved model checkpoint (skip training)')
    
    # Model architecture
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='Hidden layer dimensions')
    parser.add_argument('--activation', type=str, default='silu',
                        choices=['silu', 'softplus'],
                        help='Activation function')
    
    # Training control
    parser.add_argument('--num_batches_per_epoch', type=int, default=512,
                        help='Batches per epoch')
    parser.add_argument('--val_times', type=int, default=64,
                        help='Number of time points for validation')
    parser.add_argument('--val_samples_per_time', type=int, default=2048,
                        help='Samples per time point for validation')
    
    return parser.parse_args()


def main():
    """Main experiment loop."""
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    ensure_dirs()
    
    # Convert schedule string to enum
    schedule_enum = schedule_to_enum(args.schedule)
    
    print(f"=" * 60)
    print(f"KL Divergence Identity Verification")
    print(f"Schedule: {args.schedule}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"=" * 60)
    
    # Create model
    model = VelocityMLP(
        input_dim=3,
        hidden_dims=args.hidden_dims,
        output_dim=2,
        activation=args.activation
    )
    
    print(f"\nModel created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training or loading
    model_path = f"data/models/vtheta_schedule_{args.schedule}.pth"
    
    if args.load_model is not None:
        print(f"\nLoading model from {args.load_model}")
        load_checkpoint(model, args.load_model, device)
    else:
        print(f"\n{'='*60}")
        print("Training phase")
        print(f"{'='*60}")
        
        # Train model
        best_mse, history = train_velocity_model(
            model=model,
            schedule=schedule_enum,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_batches_per_epoch=args.num_batches_per_epoch,
            val_times=args.val_times,
            val_samples_per_time=args.val_samples_per_time,
            device=device
        )
        
        print(f"\nTraining history: val_mse min = {best_mse:.6f}")
        
        # Save checkpoint
        save_checkpoint(model, model_path)
        
        # Quick validation
        val_mse, val_nmse = validate_model(
            model, schedule_enum, 
            args.val_times, args.val_samples_per_time,
            device
        )
        print(f"Final validation: MSE = {val_mse:.6e}, NMSE = {val_nmse:.6e}")
    
    # Evaluation phase
    print(f"\n{'='*60}")
    print("Evaluation phase")
    print(f"{'='*60}")
    
    # Time grid
    t_grid = np.linspace(0, 1, args.num_times)
    
    # Storage for results
    kl_curve = np.zeros(args.num_times)
    rhs_integrand = np.zeros(args.num_times)
    
    # Evaluate at each time point
    for k in tqdm(range(args.num_times), desc="Evaluating"):
        t_k = t_grid[k]
        
        # Sample from p_t_k
        x_batch = sample_p_t(t_k, args.num_samples, schedule_enum, device=device)
        
        # LHS: KL(p_t|q_t)
        kl_k = compute_kl_lhs(x_batch, t_k, schedule_enum, model)
        kl_curve[k] = kl_k
        
        # RHS: integrand ĝ(t_k)
        g_k = compute_rhs_integrand(x_batch, t_k, schedule_enum, model)
        rhs_integrand[k] = g_k
    
    # Integrate RHS over time (trapezoidal rule)
    rhs_cumulative = integrate_rhs(rhs_integrand, t_grid)
    
    # Compute relative error
    error_stats = compute_relative_error(rhs_cumulative, kl_curve)
    
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Median relative error: {error_stats['median']:.3f}%")
    print(f"Max relative error: {error_stats['max']:.3f}%")
    print(f"Mean relative error: {error_stats['mean']:.3f}%")
    
    # Acceptance criteria
    accepted = (error_stats['median'] <= 3.0) and (error_stats['max'] <= 8.0)
    print(f"\nAcceptance: {'PASS' if accepted else 'FAIL'}")
    if not accepted:
        print("Thresholds: median ≤ 3%, max ≤ 8%")
    
    # Save results
    metadata = {
        'schedule': args.schedule,
        'seed': args.seed,
        'num_samples': args.num_samples,
        'num_times': args.num_times,
        'rtol': 1e-6,
        'atol': 1e-8,
        'median_rel_error': error_stats['median'],
        'max_rel_error': error_stats['max'],
        'mean_rel_error': error_stats['mean'],
        'accepted': accepted
    }
    
    save_results(
        t_grid, kl_curve, rhs_integrand, rhs_cumulative,
        args.schedule, metadata
    )
    
    print(f"\n{'='*60}")
    print("Experiment complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

