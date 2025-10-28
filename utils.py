"""
Utility functions for setting up experiments: seed, device, plotting, etc.
"""

import torch
import numpy as np
import random
import os
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensure deterministic behavior (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(device_str='cpu'):
    """Get torch device from string."""
    if device_str == 'cpu':
        return torch.device('cpu')
    elif device_str.startswith('cuda'):
        if torch.cuda.is_available():
            return torch.device(device_str)
        else:
            print(f"Warning: CUDA not available, using CPU instead")
            return torch.device('cpu')
    else:
        raise ValueError(f"Unknown device: {device_str}")


def ensure_dirs():
    """Create necessary directories if they don't exist."""
    dirs = ['data', 'data/models', 'data/results', 'data/plots', 'data/plot-data']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_checkpoint(model, path, metadata=None):
    """Save model checkpoint."""
    ensure_dirs()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_dict = {'model_state_dict': model.state_dict()}
    if metadata:
        save_dict.update(metadata)
    torch.save(save_dict, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(model, path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {path}")
    return checkpoint.get('metadata', {})


def plot_comparison(t_grid, kl_curve, rhs_curve, schedule, save_path=None):
    """Plot LHS (KL) vs RHS (integrated) comparison."""
    plt.figure(figsize=(12, 5))
    
    # Left subplot: raw curves
    plt.subplot(1, 2, 1)
    plt.plot(t_grid, kl_curve, label='LHS: KL(p_t|q_t)', linewidth=3, alpha=0.8, marker='o', markersize=3)
    plt.plot(t_grid, rhs_curve, label='RHS: ∫(u-v)ᵀ(s_p-s_q)', linewidth=3, alpha=0.8, linestyle='--', marker='s', markersize=3)
    plt.xlabel('Time t', fontsize=18)
    plt.ylabel('KL Divergence', fontsize=18)
    plt.title('Raw Curves', fontsize=20, fontweight='bold')
    plt.legend(fontsize=15)
    plt.grid(True, alpha=0.3)
    
    # Right subplot: smoothed curves
    plt.subplot(1, 2, 2)
    kl_smooth = smooth_curve(kl_curve, window_size=5)
    rhs_smooth = smooth_curve(rhs_curve, window_size=5)
    plt.plot(t_grid, kl_smooth, label='LHS (smoothed)', linewidth=4, alpha=0.9)
    plt.plot(t_grid, rhs_smooth, label='RHS (smoothed)', linewidth=4, alpha=0.9, linestyle='--')
    plt.xlabel('Time t', fontsize=18)
    plt.ylabel('KL Divergence', fontsize=18)
    plt.title('Smoothed Curves', fontsize=20, fontweight='bold')
    plt.legend(fontsize=15)
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'KL Identity Verification - Schedule {schedule.upper()}', fontsize=21, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        ensure_dirs()
        plt.savefig(save_path, dpi=225, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    # plt.show()  # Commented out for automated runs
    plt.close()


def smooth_curve(y, window_size=5):
    """
    Apply moving average smoothing to reduce noise in curves.
    
    Args:
        y: Array of values to smooth
        window_size: Size of the moving average window (must be odd)
    
    Returns:
        Smoothed array
    """
    if window_size < 3:
        return y
    
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    half = window_size // 2
    smoothed = np.zeros_like(y)
    
    # Pad boundaries with edge values
    y_padded = np.pad(y, (half, half), mode='edge')
    
    # Apply moving average
    for i in range(len(y)):
        smoothed[i] = np.mean(y_padded[i:i+window_size])
    
    return smoothed


def compute_relative_error(rhs, kl):
    """Compute relative error between RHS and LHS."""
    # Avoid division by zero
    epsilon = 1e-8
    abs_error = np.abs(rhs - kl)
    rel_error = abs_error / np.maximum(epsilon, np.abs(kl))
    
    return {
        'median': float(np.median(rel_error) * 100),
        'max': float(np.max(rel_error) * 100),
        'mean': float(np.mean(rel_error) * 100),
        'abs_error': abs_error.tolist(),
        'rel_error': rel_error.tolist()
    }


def save_results(t_grid, kl_curve, rhs_integrand, rhs_cumulative, schedule, metadata, target_mse=None):
    """Save experimental results to files."""
    ensure_dirs()
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename suffix with target_mse if provided
    if target_mse is not None:
        mse_str = str(target_mse).replace('.', '-')
        suffix = f"_mse_{mse_str}_{timestamp}"
    else:
        suffix = f"_{timestamp}"
    
    # Save numpy arrays
    results_dir = Path('data/results')
    np.save(results_dir / f't_grid_{schedule}{suffix}.npy', t_grid)
    np.save(results_dir / f'kl_curve_{schedule}{suffix}.npy', kl_curve)
    np.save(results_dir / f'rhs_integrand_{schedule}{suffix}.npy', rhs_integrand)
    np.save(results_dir / f'rhs_cumulative_{schedule}{suffix}.npy', rhs_cumulative)
    
    # Save metadata
    metadata_path = results_dir / f'metadata_{schedule}{suffix}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved results to {results_dir}")
    
    # Plot comparison
    plot_path = Path('data/plots') / f'kl_comparison_{schedule}{suffix}.png'
    plot_comparison(t_grid, kl_curve, rhs_cumulative, schedule, plot_path)
    
    # Save plot data (for regeneration)
    plot_data_dir = Path('data/plot-data')
    plot_data = {
        't_grid': t_grid.tolist(),
        'kl_curve': kl_curve.tolist(),
        'rhs_cumulative': rhs_cumulative.tolist(),
        'schedule': schedule,
        'metadata': metadata
    }
    plot_data_path = plot_data_dir / f'kl_comparison_{schedule}{suffix}.json'
    with open(plot_data_path, 'w') as f:
        json.dump(plot_data, f, indent=2)
    print(f"Saved plot data to {plot_data_path}")


if __name__ == '__main__':
    # Test utilities
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    print("Utilities module loaded successfully")

