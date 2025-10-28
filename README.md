# Flow Matching KL Divergence Identity Verification

This project implements and verifies a mathematical identity relating the Kullback-Leibler divergence between two time-evolving distributions to an integral of local velocity field misalignments. The identity is:

```
KL(p_t|q_t) = ∫₀ᵗ E_x~p_t[(u(x,s) - v_θ(x,s))ᵀ(∇log p_s(x) - ∇log q_s(x))] ds
```

where:
- `p_t` evolves under velocity field `u(x,t) = a(t) x`
- `q_t` evolves under learned velocity field `v_θ(x,t)`
- Both start as standard Gaussians: `p_0 = q_0 = N(0, I)`

## Project Structure

```
fm-kl-2/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── Core Modules
│   ├── true_path.py            # True distribution p_t storming (schedules, sampling, densities)
│   ├── model.py                # Neural network v_θ(x,t)
│   ├── train.py                # Training loop for v_θ
│   ├── eval.py                 # Evaluation: backward ODE, KL LHS, RHS
│   ├── utils.py                # Utilities: plotting, saving, error computation
│   └── experiment.py           # Main orchestration script
│
├── Testing
│   ├── test_golden_path.py     # LHS pipeline tests (A1-A3, B1)
│   └── test_rhs.py             # RHS pipeline tests (R0-R3)
│
├── No-Learning Verification
│   ├── nolearning_test.py      # Closed-form KL identity tests
│   ├── run_all_nolearning.sh   # Run all schedule permutations
│   └── run_all_nolearning.ps1  # PowerShell version
│
├── Automated Experiments
│   ├── run_all_experiments.py  # Train and evaluate all schedules
│   ├── run_all_experiments.ps1 # PowerShell wrapper
│   ├── run_all_cross_eval.sh   # Cross-schedule evaluations
│   └── run_all_cross_eval.ps1  # PowerShell version
│
├── Plot Regeneration
│   └── regenerate_plots.py     # Regenerate plots from saved data
│
└── data/
    ├── models/                  # Saved trained models (.pth)
    ├── results/                 # Raw results (.npy, .json)
    ├── plots/                   # Generated plots (.png)
    └── plot-data/              # Plot data for regeneration (.json)
```

## Installation

1. **Clone the repository** (or navigate to the project directory)

2. **Create a conda environment:**
```bash
conda create -n flow-kl python=3.10
conda activate flow-kl
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run the closed-form no-learning test

This verifies the identity using analytic formulas (no neural networks):

```bash
conda activate flow-kl
python nolearning_test.py --schedule_p a1 --schedule_q aaffen
```

For all 6 schedule permutations:
```bash
.\run_all_nolearning.ps1  # Windows
bash run_all_nolearning.sh  # Linux/Mac
```

### 2. Train and evaluate a model

Train a model to learn velocity field `v_θ`:

```bash
python experiment.py --schedule a1 --target_mse 0.05
```

This will:
- Train a neural network to match the true velocity
- Save the model to `data/models/`
- Evaluate the KL identity
- Generate plots showing LHS vs RHS

### 3. Load a trained model and re-evaluate

```bash
python experiment.py --schedule a1 --load_model data/models/vtheta_schedule_a1_mse_0-05_TIMESTAMP.pth
```

## Core Concepts

### The Identity

For time-evolving distributions:
- **LHS**: KL divergence `KL(p_t|q_t)`
- **RHS**: Integral of local misalignment between velocity fields and score functions

The identity states these are equal for all `t ∈ [0,1]`.

### Three Velocity Schedules

The code supports three linear velocity schedules:

- **a1(t)**: `sin(πt)`
- **a2(t)**: `0.3 sin(2πt) + 0.2`
- **a3(t)**: `t - 1/2`

Each schedule generates a different time-evolving Gaussian distribution.

### Evaluation Pipeline

**LHS computation:**
1. Sample `x ~ p_t`
2. Solve backward ODE `ẋ(s) = v_θ(x(s), s)` from `s=t` to `s=0`
3. Accumulate divergence: `ℓ = ∫₀ᵗ ∇·v_θ ds`
4. Compute `log q_t(x) = log p_0(x₀) - ℓ`
5. Compute score `∇log q_t(x)` via autograd
6. Estimate `KL(p_t|q_t) = E[log p_t(x) - log q_t(x)]`

**RHS computation:**
1. Sample `x ~ p_t`
2. Compute true velocity `u(x,t)` and learned velocity `v_θ(x,t)`
3. Compute true score `∇log p_t(x)` and learned score `∇log q_t(x)`
4. Evaluate integrand `(u-v)ᵀ(s_p - s_q)`
5. Integrate over time using trapezoidal rule

## Command-Line Interface

### `experiment.py` - Main experiment script

**Basic usage:**
```bash
python experiment.py --schedule {a1,a2,a3} [OPTIONS]
```

**Key arguments:**
- `--schedule`: Which velocity schedule to use (required)
- `--seed`: Random seed (default: 42)
- `--target_mse`: Stop training when validation MSE reaches this value
- `--target_nmse`: Stop training when normalized MSE reaches this (default: 1e-2)
- `--load_model`: Path to saved model (skips training)
- `--epochs`: Max training epochs (default: 300)
- `--num_samples`: Samples per time point for evaluation (default: 2000)
- `--num_times`: Number of time points (default: 101)
- `--num_seeds`: Number of seeds to average over (default: 1)
- `--rtol`, `--atol`: ODE solver tolerances (default: 1e-6, 1e-8)

**Examples:**
```bash
# Train on schedule a1 with target MSE 0.05
python experiment.py --schedule a1 --target_mse 0.05

# High-resolution evaluation with tighter tolerances
python experiment.py --schedule a1 --num_samples 4000 --num_times 201 --rtol 1e-7 --atol 1e-9 \
    --load_model data/models/vtheta_schedule_a1_mse_0-05_TIMESTAMP.pth

# Reduce variance by averaging over 3 seeds
python experiment.py --schedule a1 --num_seeds 3 --load_model MODEL_PATH
```

### `nolearning_test.py` - Closed-form verification

**Usage:**
```bash
python nolearning_test.py --schedule_p {a1,a2,a3} --schedule_q {a1,a2,a3} [--skip_ode]
```

**Arguments:**
- `--schedule_p`: Schedule for distribution `p_t` (default: a1)
- `--schedule_q`: Schedule for distribution `q_t` (default: a2)
- `--skip_ode`: Skip ODE pipeline test for faster runs

## Testing

### LHS Pipeline Tests (`test_golden_path.py`)

Golden-path tests using `v_θ = u` (the true velocity):

- **A1**: Preimage computation (backward ODE)
- **A2**: Divergence accumulation sign
- **A3**: Log-density computation
- **B1**: Normalization check

```bash
python test_golden_path.py
```

### RHS Pipeline Tests (`test_rhs.py`)

Tests for the RHS integrand computation:

- **R0**: Trivial identity (`v = u` → `g(t) ≡ 0`)
- **R1(a)**: Analytic check with `v ≡ 0`
- **R1(b)**: Analytic check with scaled field `v = c·u`
- **R2**: Derivative consistency with LHS
- **R3**: Internal decomposition sanity

```bash
python test_rhs.py
```

### No-Learning Tests (`nolearning_test.py`)

Closed-form KL identity verification without learning:

- Computes LHS using analytical KL formula
- Computes RHS using analytical integrand
- Verifies they match up to quadrature error
- Checks derivative consistency

```bash
python nolearning_test.py --schedule_p a1 --schedule_q a2
```

## Automated Experiment Scripts

### Train models for all schedules and target MSEs

```bash
.\run_all_experiments.ps1
```

This trains models for schedules a1, a2, a3 with target MSEs: 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8.

### Cross-schedule evaluations

```bash
.\run_all_cross_eval.ps1
```

Evaluates each schedule with models trained on *different* schedules (e.g., a1 vs a2-trained model).

## Output Files

### Models
Saved to `data/models/`:
```
vtheta_schedule_{a1,a2,a3}_mse_{TARGET}_TIMESTAMP.pth
```

### Results
Saved to `data/results/`:
- `t_grid_*.npy`: Time grid
- `kl_curve_*.npy`: KL divergence values (LHS)
- `rhs_integrand_*.npy`: RHS integrand
- `rhs_cumulative_*.npy`: Integrated RHS
- `metadata_*.json`: Experiment parameters and errors

### Plots
Saved to `data/plots/`:
- `kl_comparison_*.png`: LHS vs RHS comparison (raw + smoothed)
- `training_curves_*.png`: Training/validation loss
- `velocity_comparison_*.png`: True vs learned velocity fields
- `nolearning_test_*.png`: Closed-form verification plots

### Plot Data
Saved to `data/plot-data/` (JSON format for regeneration):
```
kl_comparison_{schedule}_mse_{TARGET}_TIMESTAMP.json
```

To regenerate plots with different styling:
```bash
python regenerate_plots.py
```

## Key Implementation Details

### Divergence Computation
The divergence `∇·v_θ` is computed analytically by summing the diagonal of the Jacobian matrix:

```python
div = sum(dv_i/dx_i) for i in range(d)
```

This is exact for neural networks and differentiable.

### Backward ODE
To compute `q_t(x)`, we solve the backward ODE:
```
ẋ(s) = v_θ(x(s), s)  [forward direction]
```

using `odeint` with reversed time: integrating from `s=t` down to `s=0`.

### Log-Density Computation
```
log q_t(x) = log p_0(x₀) - ℓ
```
where:
- `x₀` is the preimage (from backward ODE)
- `ℓ = ∫₀ᵗ ∇·v_θ ds` is the accumulated divergence

Note the **subtraction** sign (not addition), which accounts for the change of variables.

### Score Computation
The score `∇log q_t(x)` is computed by:
1. Setting `x.requires_grad_(True)`
2. Computing `log_q = log_q_t(x, ...)`
3. Taking autograd gradient: `grad(log_q.sum(), x)`

This gives the full Jacobian in one backprop pass.

## Mathematical Background

### Time-Evolving Distributions

Given velocity field `a(t)`, the distribution evolves as:
```
p_t = N(0, σ_p(t)² I)
```
where `σ_p(t) = e^A(t)` and `A(t) = ∫₀ᵗ a(s) ds`.

This describes a Gaussian that spreads over time according to the velocity schedule.

### KL Divergence Between Gaussian Distributions

For `p = N(0, σ_p²I)` and `q = N(0, σ_q²I)`:
```
KL(p|q) = (d/2) [r - 1 - log(r)]
```
where `r = σ_p²/σ_q²` and `d` is the dimension.

### The Identity

The KL divergence identity relates the global divergence to local misalignment:
```
KL(p_t|q_t) = ∫₀ᵗ E_{x~p_s}[(u(x,s)-v_θ(x,s))ᵀ(∇log p_s(x)-∇log q_s(x))] ds
```

This is useful because:
- **LHS**: Requires solving ODEs and computing densities
- **RHS**: Only requires evaluating velocities and scores at sampled points

## Troubleshooting

### Model not converging

1. **Increase training epochs**: Use `--epochs 500`
2. **Lower target MSE**: Use `--target_mse 0.2` first
3. **Check model capacity**: Larger hidden dims may help

### Memory issues

1. **Reduce batch size**: Use `--batch_size 64`
2. **Reduce evaluation samples**: Use `--num_samples 1000`
3. **Process in chunks**: Modify evaluation loop

### ODE solver failures

1. **Loosen tolerances**: Use `--rtol 1e-5 --atol 1e-7`
2. **Reduce time span**: Check if issue is near t=0 or t=1
3. **Increase max steps**: Modify in `eval.py`

## File Naming Convention

All output files use a consistent naming scheme:

```
{filename}_{schedule}_mse_{TARGET}_{YYYYMMDD_HHMMSS}.{ext}
```

Example:
```
kl_comparison_a2_mse_0-05_20251027_231335.png
```

Where:
- `filename`: Plot/data name
- `schedule`: a1, a2, or a3
- `TARGET`: Target MSE (replaced `.` with `-`)
- `TIMESTAMP`: When the experiment was run

## Dependencies

- `torch>=2.0.0`: Neural networks and autograd
- `torchdiffeq>=0.2.3`: ODE solving
- `numpy>=1.24.0`: Numerical computation
- `matplotlib>=3.7.0`: Plotting
- `scipy>=1.10.0`: Scientific computing
- `tqdm>=4.65.0`: Progress bars
- `seaborn>=0.12.0`: Statistical plots

<!-- ## License

[Specify your license here]

## Citation

If you use this code, please cite:
```bibtex
[Add citation information]
```

## Contact

[Add contact information if relevant] -->

