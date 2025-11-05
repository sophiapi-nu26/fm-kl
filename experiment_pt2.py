"""
Main experiment script for Part-2 bound verification.

Validates the bound: KL(p_1|q_1) ≤ ε√S

Usage:
    python experiment_pt2.py --mode synthetic --schedule a1 --delta_type constant --delta_beta 0.0 0.05 0.1 0.2
    python experiment_pt2.py --mode synthetic --schedule a2 --delta_type sine --delta_beta 0.05 0.1
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
from true_path import Schedule, schedule_to_enum, get_schedule_functions
from synthetic_velocity import SyntheticVelocity, constant_delta, sine_delta
from eval_pt2 import (
    compute_epsilon_pt2, compute_kl_at_t1_pt2, 
    compute_score_gap_integral_pt2
)
from plot_eps_curves import plot_lhs_rhs_vs_eps


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Verify KL bound via synthetic velocity fields')
    
    # Mode
    parser.add_argument('--mode', type=str, default='synthetic', choices=['synthetic'],
                        help='Experiment mode (only synthetic supported for Part-2)')
    
    # Schedule
    parser.add_argument('--schedule', type=str, choices=['a1', 'a2', 'a3'], required=True,
                        help='Schedule to use')
    
    # Delta configuration
    parser.add_argument('--delta_type', type=str, choices=['constant', 'sine'], required=True,
                        help='Type of perturbation δ')
    parser.add_argument('--delta_beta', type=float, nargs='+', required=True,
                        help='Beta values for δ (e.g., 0.0 0.05 0.1 0.2)')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    
    # Evaluation parameters
    parser.add_argument('--K_eps', type=int, default=101,
                        help='Number of time points for ε computation')
    parser.add_argument('--N_eps', type=int, default=4096,
                        help='Samples per time point for ε computation')
    parser.add_argument('--N_kl', type=int, default=20000,
                        help='Number of samples for KL computation')
    parser.add_argument('--K_S', type=int, default=101,
                        help='Number of time points for S computation')
    parser.add_argument('--N_S', type=int, default=2048,
                        help='Samples per time point for S computation')
    parser.add_argument('--rtol', type=float, default=1e-6,
                        help='Relative tolerance for ODE solver')
    parser.add_argument('--atol', type=float, default=1e-8,
                        help='Absolute tolerance for ODE solver')
    
    # Output
    parser.add_argument('--outdir', type=str, default='data/part-2',
                        help='Output directory')
    
    # Plotting
    parser.add_argument('--no_make_eps_curves', dest='make_eps_curves', action='store_false', default=True,
                        help='Disable ε-curves plot generation (default: enabled)')
    
    return parser.parse_args()


def ensure_part2_dirs(outdir='data/part-2'):
    """Ensure Part-2 output directories exist."""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    Path(f'{outdir}/plots').mkdir(parents=True, exist_ok=True)
    Path(f'{outdir}/results').mkdir(parents=True, exist_ok=True)


def plot_bound_verification(lhs_list, rhs_list, delta_labels, schedule, save_path):
    """
    Plot LHS vs RHS scatter with y=x reference line.
    
    Args:
        lhs_list: List of LHS values (KL(p_1|q_1))
        rhs_list: List of RHS values (ε√S)
        delta_labels: List of labels for each δ
        schedule: Schedule string
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(rhs_list, lhs_list, s=100, alpha=0.7, label='Experiments', zorder=3)
    
    # Add labels
    for i, label in enumerate(delta_labels):
        plt.annotate(label, (rhs_list[i], lhs_list[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Reference line y=x
    max_rhs = max(rhs_list) if rhs_list else 0
    max_lhs = max(lhs_list) if lhs_list else 0
    max_val = max(max_rhs, max_lhs, 1e-6) * 1.1  # Ensure at least 1e-6
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x (bound)', zorder=1)
    
    plt.xlabel('RHS = ε√S', fontsize=12)
    plt.ylabel('LHS = KL(p₁|q₁)', fontsize=12)
    plt.title(f'Bound Verification: KL(p₁|q₁) ≤ ε√S\nSchedule {schedule.upper()}', 
              fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved scatter plot to {save_path}")
    plt.close()


def plot_bar_chart(lhs_list, rhs_list, delta_labels, schedule, save_path):
    """
    Plot grouped bar chart of LHS and RHS per δ.
    
    Args:
        lhs_list: List of LHS values
        rhs_list: List of RHS values
        delta_labels: List of labels for each δ
        schedule: Schedule string
        save_path: Path to save plot
    """
    x = np.arange(len(delta_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, lhs_list, width, label='LHS = KL(p₁|q₁)', alpha=0.8)
    bars2 = ax.bar(x + width/2, rhs_list, width, label='RHS = ε√S', alpha=0.8)
    
    ax.set_xlabel('Perturbation δ', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'Bound Verification: LHS vs RHS per δ\nSchedule {schedule.upper()}', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(delta_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (lhs, rhs) in enumerate(zip(lhs_list, rhs_list)):
        ax.text(i - width/2, lhs, f'{lhs:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width/2, rhs, f'{rhs:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved bar chart to {save_path}")
    plt.close()


def plot_fhat_curves(fhat_data, delta_labels, schedule, save_path):
    """
    Plot f̂(t) curves (score gap over time) for each δ.
    
    Args:
        fhat_data: List of (t_grid, f_vals) tuples
        delta_labels: List of labels for each δ
        schedule: Schedule string
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 6))
    
    for i, ((t_grid, f_vals), label) in enumerate(zip(fhat_data, delta_labels)):
        plt.plot(t_grid, f_vals, label=f'δ: {label}', linewidth=2, alpha=0.8)
    
    plt.xlabel('Time t', fontsize=12)
    plt.ylabel('E[|s_p - s_q|²]', fontsize=12)
    plt.title(f'Score Gap f̂(t) Over Time\nSchedule {schedule.upper()}', 
              fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved f̂(t) curve to {save_path}")
    plt.close()


def main():
    """Main experiment loop."""
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    ensure_part2_dirs(args.outdir)
    
    # Convert schedule string to enum
    schedule_enum = schedule_to_enum(args.schedule)
    
    # Get schedule function
    a_fn, A_fn = get_schedule_functions(schedule_enum)
    
    print("=" * 80)
    print("Part-2: KL Bound Verification")
    print("=" * 80)
    print(f"Schedule: {args.schedule}")
    print(f"Delta type: {args.delta_type}")
    print(f"Delta betas: {args.delta_beta}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print("=" * 80)
    
    # Storage for results
    results = []
    lhs_list = []
    rhs_list = []
    delta_labels = []
    fhat_data = []
    
    # Loop over δ values
    for beta in tqdm(args.delta_beta, desc="Processing δ values"):
        print(f"\n{'='*60}")
        print(f"Processing δ: beta = {beta}")
        print(f"{'='*60}")
        
        # Build delta function
        if args.delta_type == 'constant':
            delta_fn = constant_delta(beta)
            delta_label = f"δ(t)={beta}"
        elif args.delta_type == 'sine':
            delta_fn = sine_delta(beta)
            delta_label = f"δ(t)={beta}·sin(2πt)"
        else:
            raise ValueError(f"Unknown delta_type: {args.delta_type}")
        
        # Construct synthetic velocity
        velocity = SyntheticVelocity(a_fn, delta_fn, dim=2)
        print(f"Velocity constructed: {delta_label}")
        
        # Compute ε (RMS flow-matching loss)
        print("Computing ε...")
        epsilon_hat = compute_epsilon_pt2(
            a_fn, delta_fn, schedule_enum,
            K_eps=args.K_eps, N_eps=args.N_eps, dim=2,
            device=device, dtype=torch.float64
        )
        print(f"  ε̂ = {epsilon_hat:.6e}")
        
        # Compute KL at t=1
        print("Computing KL(p_1|q_1)...")
        KL_hat = compute_kl_at_t1_pt2(
            velocity, schedule_enum,
            N_kl=args.N_kl,
            rtol=args.rtol, atol=args.atol,
            device=device, dtype=torch.float64
        )
        print(f"  KL(p_1|q_1) = {KL_hat:.6e}")
        
        # Compute S (score-gap integral)
        print("Computing S (score-gap integral)...")
        S_hat, (t_grid, f_vals) = compute_score_gap_integral_pt2(
            velocity, schedule_enum,
            K_S=args.K_S, N_S=args.N_S,
            rtol=args.rtol, atol=args.atol,
            device=device, dtype=torch.float64
        )
        print(f"  Ŝ = {S_hat:.6e}")
        
        # Compute RHS
        RHS = epsilon_hat * math.sqrt(S_hat)
        print(f"  RHS = ε̂√Ŝ = {RHS:.6e}")
        
        # Compute ratio (handle edge case when RHS ≈ 0)
        if RHS < 1e-10:
            # When δ=0, ε≈0 and S≈0, so RHS≈0
            # Bound KL ≤ 0 means KL must be ≈ 0 (since KL ≥ 0 always)
            # Check if KL is essentially zero within numerical precision
            bound_satisfied = KL_hat < 1e-6  # KL should be essentially 0 when v=u
            ratio = float('inf') if not bound_satisfied else 0.0
        else:
            ratio = KL_hat / RHS
            bound_satisfied = KL_hat <= RHS
        
        if ratio == float('inf'):
            print(f"  Ratio (LHS/RHS) = inf (RHS≈0, KL={KL_hat:.2e})")
        else:
            print(f"  Ratio (LHS/RHS) = {ratio:.6f}")
        print(f"  Bound satisfied: {'YES ✓' if bound_satisfied else 'NO ✗'}")
        
        # Store results
        result = {
            'schedule': args.schedule,
            'delta_type': args.delta_type,
            'beta': float(beta),
            'delta_label': delta_label,
            'epsilon_hat': float(epsilon_hat),
            'KL_hat': float(KL_hat),
            'S_hat': float(S_hat),
            'RHS': float(RHS),
            'ratio': float(ratio),
            'bound_satisfied': bool(bound_satisfied)
        }
        results.append(result)
        
        lhs_list.append(KL_hat)
        rhs_list.append(RHS)
        delta_labels.append(delta_label)
        fhat_data.append((t_grid, f_vals))
    
    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"{'δ':<30} {'LHS':<15} {'RHS':<15} {'Ratio':<10} {'Bound'}")
    print("-" * 80)
    for result in results:
        status = "✓" if result['bound_satisfied'] else "✗"
        ratio_str = "inf" if result['ratio'] == float('inf') else f"{result['ratio']:.3f}"
        print(f"{result['delta_label']:<30} {result['KL_hat']:<15.6e} {result['RHS']:<15.6e} "
              f"{ratio_str:<10} {status}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_path = Path(args.outdir) / 'results' / f'bound_{args.schedule}_{args.delta_type}_{timestamp}.json'
    with open(json_path, 'w') as f:
        # Convert inf to string for JSON compatibility
        results_json = []
        for result in results:
            result_copy = result.copy()
            if result_copy['ratio'] == float('inf'):
                result_copy['ratio'] = 'inf'
            results_json.append(result_copy)
        json.dump({
            'args': vars(args),
            'results': results_json
        }, f, indent=2)
    print(f"\nSaved results to {json_path}")
    
    # Save CSV (use UTF-8 encoding to handle Greek delta character)
    csv_path = Path(args.outdir) / 'results' / f'bound_{args.schedule}_{args.delta_type}_{timestamp}.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('schedule,delta_type,beta,delta_label,epsilon_hat,KL_hat,S_hat,RHS,ratio,bound_satisfied\n')
        for result in results:
            ratio_val = result['ratio']
            ratio_str = 'inf' if ratio_val == float('inf') else str(ratio_val)
            f.write(f"{result['schedule']},{result['delta_type']},{result['beta']},"
                   f'"{result["delta_label"]}",{result["epsilon_hat"]},{result["KL_hat"]},'
                   f'{result["S_hat"]},{result["RHS"]},{ratio_str},{result["bound_satisfied"]}\n')
    print(f"Saved CSV to {csv_path}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Scatter plot
    scatter_path = Path(args.outdir) / 'plots' / f'bound_scatter_{args.schedule}_{args.delta_type}_{timestamp}.png'
    plot_bound_verification(lhs_list, rhs_list, delta_labels, args.schedule, scatter_path)
    
    # Bar chart
    bar_path = Path(args.outdir) / 'plots' / f'bound_bars_{args.schedule}_{args.delta_type}_{timestamp}.png'
    plot_bar_chart(lhs_list, rhs_list, delta_labels, args.schedule, bar_path)
    
    # f̂(t) curves
    fhat_path = Path(args.outdir) / 'plots' / f'fhat_curves_{args.schedule}_{args.delta_type}_{timestamp}.png'
    plot_fhat_curves(fhat_data, delta_labels, args.schedule, fhat_path)
    
    # ε-curves plot
    if args.make_eps_curves:
        eps_curves_path = Path(args.outdir) / 'plots' / f'eps_curves_synthetic_{args.schedule}_{args.delta_type}_{timestamp}.png'
        try:
            plot_lhs_rhs_vs_eps(
                str(csv_path),
                str(eps_curves_path),
                schedule=args.schedule,
                ylog=True,
                annotate=True,
                title=f"Bound components vs ε — synthetic {args.delta_type} — schedule {args.schedule.upper()}"
            )
        except Exception as e:
            print(f"Warning: Failed to generate ε-curves plot: {e}")
    
    print(f"\n{'='*80}")
    print("Experiment complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

