"""
Plotting utility for Part 2 (Learning and Synthetic) ε-curves.

Generates curves showing LHS (KL) and RHS (ε√S) vs ε (RMS flow-matching loss).
Works with both learned (Part 2 Learning) and synthetic (Part 2) CSVs.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_lhs_rhs_vs_eps(
    results_csv: str,
    out_png: str,
    schedule: str,
    ylog: bool = True,
    annotate_epochs: bool = True,
    annotate: bool = None,
    title: str = None,
):
    """
    Make the 'curves vs epsilon' plot:
      - x: eps (eval-time RMS flow-matching loss)
      - y1: KL(p1 || q1)  (LHS)
      - y2: eps * sqrt(S) (RHS)
    One line per quantity; points are checkpoints sorted by eps.
    
    Supports both learned CSVs (with eps_eval, KL_eval, S_eval) and 
    synthetic CSVs (with epsilon_hat, KL_hat, S_hat).
    
    Args:
        results_csv: Path to CSV file with results
        out_png: Path to save output PNG
        schedule: Schedule name (for filtering and title)
        ylog: Whether to use log scale for y-axis
        annotate_epochs: Whether to annotate points with epoch numbers (learned) - deprecated, use annotate
        annotate: Whether to annotate points (auto-detects epoch for learned, delta_label for synthetic)
        title: Optional custom title (default: auto-generated)
    """
    # Backward compatibility: if annotate not provided, use annotate_epochs
    if annotate is None:
        annotate = annotate_epochs
    
    # Read CSV
    df = pd.read_csv(results_csv)
    
    # Filter by schedule if 'schedule' column exists
    if 'schedule' in df.columns:
        df = df[df['schedule'] == schedule].copy()
    
    if df.empty:
        raise ValueError(f"No rows for schedule={schedule} in {results_csv}")

    # Auto-detect column names (learned vs synthetic)
    # Learned: eps_eval, KL_eval, S_eval
    # Synthetic: epsilon_hat, KL_hat, S_hat
    if 'eps_eval' in df.columns and 'KL_eval' in df.columns and 'S_eval' in df.columns:
        # Learned format
        eps_col = 'eps_eval'
        kl_col = 'KL_eval'
        s_col = 'S_eval'
        label_col = 'epoch' if 'epoch' in df.columns else None
    elif 'epsilon_hat' in df.columns and 'KL_hat' in df.columns and 'S_hat' in df.columns:
        # Synthetic format
        eps_col = 'epsilon_hat'
        kl_col = 'KL_hat'
        s_col = 'S_hat'
        label_col = 'delta_label' if 'delta_label' in df.columns else None
    else:
        # Try to find any combination
        eps_cols = [c for c in df.columns if 'eps' in c.lower()]
        kl_cols = [c for c in df.columns if 'kl' in c.lower()]
        s_cols = [c for c in df.columns if c.lower() in ['s', 's_eval', 's_hat']]
        
        if not eps_cols or not kl_cols or not s_cols:
            raise ValueError(f"Could not identify required columns in {results_csv}. "
                           f"Expected either (eps_eval, KL_eval, S_eval) for learned or "
                           f"(epsilon_hat, KL_hat, S_hat) for synthetic.")
        
        eps_col = eps_cols[0]
        kl_col = kl_cols[0]
        s_col = s_cols[0]
        label_col = None

    # Compute RHS if not present
    if 'RHS' not in df.columns:
        df['RHS'] = df[eps_col] * (df[s_col] ** 0.5)

    # Sort by epsilon (stable sort)
    df.sort_values(eps_col, inplace=True, kind='mergesort')

    # Extract data
    x = df[eps_col].values
    y_lhs = df[kl_col].values
    y_rhs = df['RHS'].values
    
    # Handle zero or very small epsilon values for log scale
    # Replace zero with a small value relative to the minimum non-zero epsilon
    min_nonzero_eps = x[x > 0].min() if (x > 0).any() else 1e-6
    x_plot = x.copy()
    x_plot[x_plot == 0] = min_nonzero_eps * 0.1  # Place zero points at 10% of min non-zero

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_plot, y_lhs, marker='o', linestyle='-', label='LHS: KL(p₁‖q₁)', 
            linewidth=2, color='darkgreen')
    ax.plot(x_plot, y_rhs, marker='s', linestyle='-', label='RHS: ε·√S', 
            linewidth=2, color='darkred')

    # Optionally annotate points (use x_plot for positioning)
    if annotate and label_col is not None and label_col in df.columns:
        labels = df[label_col].astype(str).values
        for xi, xi_plot, yi, lab in zip(x, x_plot, y_lhs, labels):
            # For epochs, convert to int string; for delta_label, use as-is
            if label_col == 'epoch':
                lab_str = str(int(float(lab))) if lab.replace('.', '').replace('-', '').isdigit() else str(lab)
            else:
                lab_str = str(lab)
            ax.annotate(lab_str, (xi_plot, yi), xytext=(3, 3), 
                       textcoords='offset points', fontsize=8)

    ax.set_xlabel('ε (RMS flow-matching loss)', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ttl = title or f"Bound components vs ε — schedule {schedule.upper()}"
    ax.set_title(ttl, fontsize=13, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10)

    # Set log scales
    ax.set_xscale('log')  # x-axis on log scale
    if ylog:
        ax.set_yscale('log')  # y-axis on log scale (if enabled)

    # Save
    out_dir = os.path.dirname(out_png)
    if out_dir:  # Only create directory if path contains a directory
        os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved ε-curves plot to {out_png}")

