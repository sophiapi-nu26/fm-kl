"""
Regenerate plots from saved plot data.

Usage:
    python regenerate_plots.py
    
This will regenerate all plots from the data/plot-data directory.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from utils import smooth_curve, plot_comparison, ensure_dirs


def regenerate_from_data(plot_data_path):
    """Regenerate a plot from saved data."""
    with open(plot_data_path, 'r') as f:
        data = json.load(f)
    
    t_grid = np.array(data['t_grid'])
    kl_curve = np.array(data['kl_curve'])
    rhs_cumulative = np.array(data['rhs_cumulative'])
    schedule = data['schedule']
    
    # Generate plot filename from data filename
    plot_data_path = Path(plot_data_path)
    plot_filename = plot_data_path.stem + '.png'  # Change .json to .png
    plot_path = Path('data/plots') / plot_filename
    
    # Regenerate plot
    plot_comparison(t_grid, kl_curve, rhs_cumulative, schedule, plot_path)
    
    return plot_path


def main():
    """Regenerate all plots from plot-data directory."""
    ensure_dirs()
    
    plot_data_dir = Path('data/plot-data')
    
    if not plot_data_dir.exists():
        print("No plot-data directory found!")
        return
    
    plot_data_files = list(plot_data_dir.glob('kl_comparison_*.json'))
    
    if not plot_data_files:
        print("No plot data files found!")
        return
    
    print(f"Found {len(plot_data_files)} plot data files to regenerate")
    
    for plot_data_path in plot_data_files:
        print(f"Regenerating: {plot_data_path.name}")
        plot_path = regenerate_from_data(plot_data_path)
        print(f"  Saved to: {plot_path}\n")
    
    print(f"Regenerated {len(plot_data_files)} plots")


if __name__ == '__main__':
    main()

