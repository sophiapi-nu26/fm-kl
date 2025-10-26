# KL Evolution Identity Validation

This repository implements a complete numerical validation of the KL evolution identity for flow matching, as described in §6.1 of the research specification.

## Objective

Numerically validate the KL evolution identity by showing, for all t ∈ [0,1]:

```
KL(p_t|q_t) ≈ ∫₀ᵗ E_{x∼p_s}[(u-v_θ)ᵀ(∇log p_s - ∇log q_s)] ds
```

where:
- `p_t` is a known Gaussian path driven by true velocity `u(x,t) = a(t)x`
- `q_t` is the path induced by a learned velocity field `v_θ(x,t)`

## Ground Truth Paths

Three scalar schedules `a(t)` are implemented:

1. **sin(πt)** - Simple sinusoidal schedule
2. **0.3 sin(2πt) + 0.2** - Modulated sinusoidal schedule  
3. **t - 0.5** - Linear schedule

Each schedule defines a Gaussian path `p_t = N(0, σ_p(t)² I₂)` where `σ_p(t) = exp(∫₀ᵗ a(s) ds)`.

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- torchdiffeq 0.2.3+
- NumPy 1.24+
- Matplotlib 3.7+
- SciPy 1.10+
- tqdm 4.65+

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd fm-kl

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete experiment for all three schedules:

```bash
python main.py
```

### Custom Configuration

```bash
python main.py \
    --seed 42 \
    --device cpu \
    --schedules sin_pi sin_2pi linear \
    --num_epochs 2000 \
    --learning_rate 1e-3 \
    --batch_size 1024 \
    --target_val_loss 1e-4 \
    --K 101 \
    --N 2000 \
    --rtol 1e-6 \
    --atol 1e-8
```

### Command Line Arguments

- `--seed`: Random seed for reproducibility (default: 42)
- `--device`: Device to use - "cpu" or "cuda" (default: cpu)
- `--schedules`: List of schedules to run (default: all three)
- `--num_epochs`: Maximum training epochs (default: 2000)
- `--learning_rate`: Learning rate for Adam optimizer (default: 1e-3)
- `--batch_size`: Training batch size (default: 1024)
- `--target_val_loss`: Target validation loss for early stopping (default: 1e-4)
- `--K`: Number of time points in evaluation grid (default: 101)
- `--N`: Number of samples per time point (default: 2000)
- `--rtol`: ODE solver relative tolerance (default: 1e-6)
- `--atol`: ODE solver absolute tolerance (default: 1e-8)

### Individual Components

You can also run individual components for testing:

```bash
# Test ground truth implementation
python ground_truth.py

# Test velocity field model
python model.py

# Test training
python training.py

# Test backward ODE solver
python backward_ode.py

# Test evaluation
python evaluation.py

# Test plotting
python plotting.py
```

## Project Structure

```
fm-kl/
├── main.py                 # Main experiment script
├── ground_truth.py         # Ground truth path implementation
├── model.py                # MLP velocity field model
├── training.py             # Flow matching training
├── backward_ode.py         # Backward ODE solver for q_t evaluation
├── evaluation.py           # KL evolution evaluation
├── plotting.py             # Plotting and diagnostics
├── requirements.txt        # Python dependencies
├── instructions.md         # Original research specification
├── checkpoints/            # Saved model checkpoints
├── results/               # Evaluation results and diagnostics
└── plots/                 # Generated plots
```

## Key Components

### 1. Ground Truth Path (`ground_truth.py`)

Implements the three scalar schedules with closed-form expressions for:
- Log-densities: `log p_t(x) = -d/2 log(2π) - d log σ_p(t) - |x|²/(2σ_p(t)²)`
- Score functions: `∇log p_t(x) = -x/σ_p(t)²`
- True velocity field: `u(x,t) = a(t)x`

### 2. Velocity Field Model (`model.py`)

Small MLP with:
- 2-3 hidden layers, width 64-128
- SiLU/Softplus activation
- Concatenated inputs `[x,t]`
- Exact divergence computation via autograd

### 3. Training (`training.py`)

Flow matching training with:
- MSE loss: `E|v_θ(x,t) - u(x,t)|²`
- Adam optimizer
- Early stopping based on validation loss
- Training curve visualization

### 4. Backward ODE Solver (`backward_ode.py`)

Core numerical routine for evaluating `q_t(x)` and `∇log q_t(x)`:
- Backward integration from time `t` to `0`
- State ODE: `ẋ_s = -v_θ(x_s, s)`
- Log-density accumulator: `ℓ̇_s = +∇·v_θ(x_s, s)`
- Adaptive RK solver (Dormand-Prince/Tsit5)

### 5. Evaluation (`evaluation.py`)

KL evolution identity validation:
- Time × samples evaluation grid
- KL estimator: `KL̂(t_k) = (1/N) Σᵢ [log p_{t_k}(X[k,i]) - log q_{t_k}(X[k,i])]`
- RHS integrand: `ĝ(t_k) = (1/N) Σᵢ [(u[k,i] - v[k,i])ᵀ(s_p[k,i] - s_q[k,i])]`
- Trapezoidal rule integration

### 6. Plotting & Diagnostics (`plotting.py`)

Visualization and validation:
- KL evolution curves
- Training progress plots
- Velocity field comparisons
- Reversibility tests
- Convergence analysis
- Tolerance sensitivity

## Output Files

### Checkpoints
- `checkpoints/model_{schedule}.pt` - Trained model weights

### Results
- `results/evaluation_{schedule}.npz` - Evaluation arrays
- `results/diagnostics_{schedule}.json` - Diagnostic results
- `results/experiment_summary.json` - Complete experiment summary

### Plots
- `plots/kl_evolution_{schedule}.pdf` - Main KL evolution plots
- `plots/training_{schedule}.pdf` - Training curves
- `plots/velocity_field_comparison.pdf` - Velocity field comparisons

## Acceptance Criteria

For each schedule, the experiment passes if:
- Median relative error ≤ 3%
- Maximum relative error ≤ 8%
- Curves `KL̂(t)` and `R(t)` overlap visually across [0,1]
- Diagnostics show numerical stability

## Environment Details

### Tested Versions
- Python: 3.8+
- PyTorch: 2.0.0+
- torchdiffeq: 0.2.3+
- NumPy: 1.24.0+
- Matplotlib: 3.7.0+
- SciPy: 1.10.0+

### Performance Notes
- 2D experiments run efficiently on CPU
- GPU optional for faster autograd through ODE
- Typical runtime: 10-30 minutes per schedule on modern hardware
- Memory usage: ~2-4 GB for full evaluation grid

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`

2. **CUDA errors**: Use `--device cpu` if CUDA is not available

3. **Memory issues**: Reduce `--N` (samples per time point) or `--K` (time points)

4. **Convergence issues**: Increase `--num_epochs` or adjust `--learning_rate`

5. **ODE solver errors**: Tighten tolerances with `--rtol 1e-7 --atol 1e-9`

### Debug Mode

Run individual components with smaller parameters for debugging:

```bash
python main.py --K 21 --N 100 --num_epochs 100
```

## Citation

If you use this code in your research, please cite the original paper and include a reference to this implementation.

## License

See LICENSE file for details.
