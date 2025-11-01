"""
Main experiment script for Part-2 (Learning) bound verification.

Trains a velocity MLP and validates the bound: KL(p_1|q^θ_1) ≤ ε_θ√S_θ

Usage:
    python experiment_learn_pt2.py --schedule a2 --epochs 200 --batch_size 128
    python experiment_learn_pt2.py --schedule a1 --eval_checkpoints "final,best,50,100" --eval_only
"""

import argparse
import json
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from utils import set_seed, get_device, ensure_dirs
from true_path import Schedule, schedule_to_enum
from model_learn_pt2 import VelocityMLP
from train_learn_pt2 import train_velocity_model
from eval_learn_pt2 import (
    compute_epsilon_learn, compute_score_gap_integral_learn, 
    compute_kl_at_t1_learn
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Verify KL bound via learned velocity model')
    
    # Schedule
    parser.add_argument('--schedule', type=str, choices=['a1', 'a2', 'a3'], required=True,
                        help='Schedule to use')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=400,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size (small default to slow training and capture diverse checkpoints)')
    parser.add_argument('--batches_per_epoch', type=int, default=2,
                        help='Number of batches per epoch (small default to slow training)')
    parser.add_argument('--val_times', type=int, default=64,
                        help='Number of time points for training validation')
    parser.add_argument('--val_samples_per_time', type=int, default=2048,
                        help='Samples per time point for training validation')
    parser.add_argument('--save_epochs', type=int, nargs='+', default=[10, 20, 40, 80, 160, 320],
                        help='Epoch numbers to save checkpoints regardless of improvement')
    
    # Evaluation parameters (separate from training)
    parser.add_argument('--eval_checkpoints', type=str, default='final,best',
                        help='Comma-separated list: final,best, or epoch numbers (e.g., "final,best,50,100")')
    parser.add_argument('--eval_val_times', type=int, default=101,
                        help='Number of time points for ε_θ computation')
    parser.add_argument('--eval_val_samples_per_time', type=int, default=1024,
                        help='Samples per time point for ε_θ computation')
    parser.add_argument('--eval_K_S', type=int, default=101,
                        help='Number of time points for S_θ computation')
    parser.add_argument('--eval_N_S', type=int, default=512,
                        help='Samples per time point for S_θ computation (reduced for speed)')
    parser.add_argument('--eval_N_kl', type=int, default=5000,
                        help='Number of samples for KL computation (reduced for speed)')
    parser.add_argument('--eval_rtol', type=float, default=1e-6,
                        help='Relative tolerance for ODE solver (evaluation)')
    parser.add_argument('--eval_atol', type=float, default=1e-8,
                        help='Absolute tolerance for ODE solver (evaluation)')
    parser.add_argument('--eval_chunk_size', type=int, default=1024,
                        help='Chunk size for batched evaluation')
    parser.add_argument('--tight_eval', action='store_true',
                        help='Use tighter tolerances for evaluation (rtol=3e-7, atol=3e-9)')
    
    # Seeds
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for training')
    parser.add_argument('--eval_seed', type=int, default=12345,
                        help='Random seed for evaluation (independent from training)')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    
    # Output
    parser.add_argument('--outdir', type=str, default='data/part-2-learn',
                        help='Output directory root')
    
    # Mode
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training and evaluate existing checkpoints')
    
    return parser.parse_args()


def ensure_learn_dirs(outdir, schedule):
    """Ensure Part-2-learn output directories exist."""
    base_dir = Path(outdir) / schedule
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (base_dir / 'results').mkdir(parents=True, exist_ok=True)
    (base_dir / 'plots').mkdir(parents=True, exist_ok=True)
    (base_dir / 'logs').mkdir(parents=True, exist_ok=True)
    return base_dir


def plot_bound_verification(results, schedule, save_path):
    """
    Plot LHS vs RHS scatter with y=x reference line.
    
    Args:
        results: List of dicts with 'kl_hat', 'rhs', 'epoch' keys
        schedule: Schedule string
        save_path: Path to save plot
    """
    if not results:
        print("No results to plot")
        return
    
    lhs_list = [r['kl_hat'] for r in results]
    rhs_list = [r['rhs'] for r in results]
    epochs = [r['epoch'] for r in results]
    
    plt.figure(figsize=(10, 8))
    
    # Scatter plot with epoch labels
    scatter = plt.scatter(rhs_list, lhs_list, s=100, alpha=0.7, c=epochs, 
                         cmap='viridis', label='Checkpoints', zorder=3)
    plt.colorbar(scatter, label='Epoch')
    
    # Add epoch labels
    for r in results:
        plt.annotate(f"E{r['epoch']}", (r['rhs'], r['kl_hat']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Reference line y=x
    max_rhs = max(rhs_list) if rhs_list else 0
    max_lhs = max(lhs_list) if lhs_list else 0
    max_val = max(max_rhs, max_lhs, 1e-6) * 1.1
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x (bound)', zorder=1)
    
    plt.xlabel('RHS = ε_θ√S_θ', fontsize=12)
    plt.ylabel('LHS = KL(p₁|q₁^θ)', fontsize=12)
    plt.title(f'Bound Verification: KL(p₁|q₁^θ) ≤ ε_θ√S_θ\nSchedule {schedule.upper()}', 
              fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved scatter plot to {save_path}")
    plt.close()


def plot_fhat_curves(fhat_data_list, labels, schedule, save_path):
    """
    Plot f̂(t) curves for multiple checkpoints.
    
    Args:
        fhat_data_list: List of (t_grid, f_vals) tuples
        labels: List of labels (e.g., epoch numbers)
        schedule: Schedule string
        save_path: Path to save plot
    """
    if not fhat_data_list:
        return
    
    plt.figure(figsize=(10, 6))
    
    for (t_grid, f_vals), label in zip(fhat_data_list, labels):
        plt.plot(t_grid, f_vals, label=f"Epoch {label}", linewidth=1.5, alpha=0.7)
    
    plt.xlabel('Time t', fontsize=12)
    plt.ylabel('f̂(t) = E|s_p - s_q^θ|²', fontsize=12)
    plt.title(f'Score Gap Integrand f̂(t)\nSchedule {schedule.upper()}', 
              fontsize=13, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved f̂(t) curve to {save_path}")
    plt.close()


def load_checkpoint_metadata(checkpoint_path):
    """Load checkpoint metadata JSON."""
    metadata_path = checkpoint_path.with_suffix('.json')
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None


def main():
    """Main experiment loop."""
    args = parse_args()
    
    # Setup
    schedule_enum = schedule_to_enum(args.schedule)
    base_dir = ensure_learn_dirs(args.outdir, args.schedule)
    checkpoint_dir = base_dir / 'checkpoints'
    
    # Apply tight eval if requested
    if args.tight_eval:
        args.eval_rtol = 3e-7
        args.eval_atol = 3e-9
    
    # Set default dtype
    torch.set_default_dtype(torch.float64)
    
    print("=" * 80)
    print("Part-2 (Learning): KL Bound Verification")
    print("=" * 80)
    print(f"Schedule: {args.schedule}")
    print(f"Device: {args.device}")
    print(f"Training seed: {args.seed}")
    print(f"Eval seed: {args.eval_seed}")
    print("=" * 80)
    
    model = None
    
    # Training phase
    if not args.eval_only:
        print("\n" + "=" * 80)
        print("Training Phase")
        print("=" * 80)
        
        # Set training seed
        set_seed(args.seed)
        device = get_device(args.device)
        
        # Initialize model
        model = VelocityMLP(hidden_dims=[64, 64], activation='silu')
        
        # Auto batches_per_epoch if not specified
        batches_per_epoch = args.batches_per_epoch
        if batches_per_epoch is None:
            # Default: 2 batches per epoch to slow training and capture diverse checkpoints
            batches_per_epoch = 2
        
        # Train
        best_mse, history = train_velocity_model(
            model=model,
            schedule=schedule_enum,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_batches_per_epoch=batches_per_epoch,
            val_times=args.val_times,
            val_samples_per_time=args.val_samples_per_time,
            device=device,
            dtype=torch.float64,
            checkpoint_dir=str(checkpoint_dir),
            save_epochs=args.save_epochs,
            seed=args.seed
        )
        
        print(f"\nTraining complete. Best val MSE: {best_mse:.6e}")
        print(f"Checkpoints saved: {len(history.get('checkpoints', []))}")
    else:
        print("\nSkipping training (--eval_only mode)")
        device = get_device(args.device)
    
    # Evaluation phase
    print("\n" + "=" * 80)
    print("Evaluation Phase")
    print("=" * 80)
    
    # Parse checkpoint list
    checkpoint_names_raw = [name.strip() for name in args.eval_checkpoints.split(',')]
    
    # Handle "all" to evaluate all saved checkpoints
    if 'all' in checkpoint_names_raw:
        # Find all checkpoint .pt files (excluding aliases)
        all_checkpoints = []
        for ckpt_file in checkpoint_dir.glob("ckpt__*.pt"):
            # Extract epoch from filename
            try:
                # Format: ckpt__sched=a1__epoch=134__valmse=...
                parts = ckpt_file.stem.split('__')
                for part in parts:
                    if part.startswith('epoch='):
                        epoch_num = int(part.split('=')[1])
                        all_checkpoints.append(epoch_num)
                        break
            except (ValueError, IndexError):
                continue
        # Sort by epoch and convert to strings
        all_checkpoints = sorted(set(all_checkpoints))
        checkpoint_names = ['best', 'final'] + [str(e) for e in all_checkpoints]
        print(f"\nFound {len(all_checkpoints)} saved checkpoints. Evaluating all of them plus 'best' and 'final'.")
    else:
        checkpoint_names = checkpoint_names_raw
    
    results = []
    fhat_data_list = []
    labels = []
    
    for checkpoint_name in checkpoint_names:
        print(f"\nProcessing checkpoint: {checkpoint_name}")
        
        # Determine checkpoint path
        if checkpoint_name == 'final':
            ckpt_path = checkpoint_dir / 'final.pt'
        elif checkpoint_name == 'best':
            ckpt_path = checkpoint_dir / 'best.pt'
        else:
            # Try to find epoch-specific checkpoint
            try:
                epoch_num = int(checkpoint_name)
                # Find checkpoint file matching this epoch
                pattern = f"*epoch={epoch_num}*.pt"
                matches = list(checkpoint_dir.glob(pattern))
                if matches:
                    ckpt_path = matches[0]
                else:
                    print(f"  Warning: No checkpoint found for epoch {epoch_num}, skipping")
                    continue
            except ValueError:
                print(f"  Warning: Invalid checkpoint name '{checkpoint_name}', skipping")
                continue
        
        if not ckpt_path.exists():
            print(f"  Warning: Checkpoint {ckpt_path} not found, skipping")
            continue
        
        # Load checkpoint metadata
        metadata = load_checkpoint_metadata(ckpt_path)
        if metadata is None:
            print(f"  Warning: No metadata found for {ckpt_path}")
            epoch = int(checkpoint_name) if checkpoint_name.isdigit() else -1
            val_mse_train = None
        else:
            epoch = metadata['epoch']
            val_mse_train = metadata['val_mse']
        
        val_mse_str = f"{val_mse_train:.6e}" if val_mse_train is not None else 'N/A'
        print(f"  Epoch: {epoch}, Val MSE (train): {val_mse_str}")
        
        # Load model if needed
        if model is None:
            model = VelocityMLP(hidden_dims=[64, 64], activation='silu')
        
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        model = model.to(device)
        
        # Set eval seed
        set_seed(args.eval_seed)
        
        # Compute ε_θ
        print("  Computing ε_θ...")
        epsilon_hat = compute_epsilon_learn(
            v_theta=model,
            schedule=schedule_enum,
            val_times=args.eval_val_times,
            val_samples_per_time=args.eval_val_samples_per_time,
            device=device,
            dtype=torch.float64
        )
        print(f"    ε_θ = {epsilon_hat:.6e}")
        
        # Compute S_θ
        print("  Computing S_θ...")
        S_hat, (t_grid, f_vals) = compute_score_gap_integral_learn(
            v_theta=model,
            schedule=schedule_enum,
            K_S=args.eval_K_S,
            N_S=args.eval_N_S,
            rtol=args.eval_rtol,
            atol=args.eval_atol,
            eval_seed=args.eval_seed,
            device=device,
            dtype=torch.float64,
            chunk_size=args.eval_chunk_size
        )
        print(f"    S_θ = {S_hat:.6e}")
        
        # Compute KL at t=1
        print("  Computing KL(p₁|q₁^θ)...")
        KL_hat = compute_kl_at_t1_learn(
            v_theta=model,
            schedule=schedule_enum,
            N_kl=args.eval_N_kl,
            rtol=args.eval_rtol,
            atol=args.eval_atol,
            eval_seed=args.eval_seed,
            device=device,
            dtype=torch.float64,
            chunk_size=args.eval_chunk_size
        )
        print(f"    KL(p₁|q₁^θ) = {KL_hat:.6e}")
        
        # Compute RHS
        RHS = epsilon_hat * math.sqrt(S_hat)
        print(f"    RHS = ε_θ√S_θ = {RHS:.6e}")
        
        # Compute ratio
        if RHS < 1e-10:
            bound_satisfied = KL_hat < 1e-6
            ratio = float('inf') if not bound_satisfied else 0.0
        else:
            ratio = KL_hat / RHS
            bound_satisfied = KL_hat <= RHS
        
        status = "✓" if bound_satisfied else "✗"
        print(f"    Ratio (LHS/RHS) = {ratio:.6f}")
        print(f"    Bound satisfied: {status}")
        
        # Store results
        result = {
            'epoch': epoch,
            'ckpt_path': str(ckpt_path),
            'val_mse_train': val_mse_train,
            'eps_eval': float(epsilon_hat),
            'S_eval': float(S_hat),
            'KL_eval': float(KL_hat),
            'kl_hat': float(KL_hat),  # Alias for plotting
            'rhs': float(RHS),  # Alias for plotting
            'RHS': float(RHS),
            'ratio': float(ratio) if ratio != float('inf') else 'inf',
            'bound_satisfied': bool(bound_satisfied),
            'eval_params': {
                'eval_val_times': args.eval_val_times,
                'eval_val_samples_per_time': args.eval_val_samples_per_time,
                'eval_K_S': args.eval_K_S,
                'eval_N_S': args.eval_N_S,
                'eval_N_kl': args.eval_N_kl,
                'eval_rtol': args.eval_rtol,
                'eval_atol': args.eval_atol,
                'eval_chunk_size': args.eval_chunk_size
            },
            'train_params': metadata.get('train_params', {}) if metadata else {},
            'seeds': {'training': args.seed, 'eval': args.eval_seed}
        }
        
        results.append(result)
        fhat_data_list.append((t_grid, f_vals))
        labels.append(epoch)
    
    # Save results
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_path = base_dir / 'results' / f'bound_{args.schedule}_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump({
                'args': vars(args),
                'results': results
            }, f, indent=2)
        print(f"\nSaved results to {json_path}")
        
        # Save CSV
        csv_path = base_dir / 'results' / f'bound_{args.schedule}_{timestamp}.csv'
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('epoch,ckpt_path,val_mse_train,eps_eval,S_eval,KL_eval,RHS,ratio,bound_satisfied\n')
            for r in results:
                ratio_str = str(r['ratio']) if r['ratio'] != 'inf' else 'inf'
                f.write(f"{r['epoch']},{r['ckpt_path']},{r['val_mse_train'] or 'N/A'},"
                       f"{r['eps_eval']},{r['S_eval']},{r['KL_eval']},{r['RHS']},"
                       f"{ratio_str},{r['bound_satisfied']}\n")
        print(f"Saved CSV to {csv_path}")
        
        # Generate plots
        print("\nGenerating plots...")
        
        # Scatter plot
        scatter_path = base_dir / 'plots' / f'bound_scatter_{args.schedule}_{timestamp}.png'
        plot_bound_verification(results, args.schedule, scatter_path)
        
        # f̂(t) curves
        if fhat_data_list:
            fhat_path = base_dir / 'plots' / f'fhat_curves_{args.schedule}_{timestamp}.png'
            plot_fhat_curves(fhat_data_list, labels, args.schedule, fhat_path)
        
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"{'Epoch':<8} {'LHS':<15} {'RHS':<15} {'Ratio':<10} {'Bound'}")
        print("-" * 80)
        for r in results:
            status = "✓" if r['bound_satisfied'] else "✗"
            ratio_str = "inf" if r['ratio'] == 'inf' else f"{r['ratio']:.3f}"
            print(f"{r['epoch']:<8} {r['KL_eval']:<15.6e} {r['RHS']:<15.6e} "
                  f"{ratio_str:<10} {status}")
    
    print("\n" + "=" * 80)
    print("Experiment complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

