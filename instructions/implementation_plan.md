# Implementation Plan

## Overview

This experiment verifies the KL-divergence identity by comparing:
- **LHS**: Measured KL(p_t|q_t) 
- **RHS**: Time-integrated local misalignment ∫(u-v)ᵀ(∇log p - ∇log q) ds

## File Structure

```
fm-kl-2/
├── true_path.py       # Schedules a(t), σ_p(t), sampling, log p_t, ∇log p_t
├── model.py           # MLP v_θ definition
├── train.py           # Training loop (flow matching)
├── eval.py            # Backward ODE, divergence, log q_t, ∇log q_t
├── experiment.py      # Main orchestrator (unified runner)
├── utils.py           # Seed, device, plotting, helpers
├── requirements.txt   # Dependencies
└── data/              # Saved models, results (created at runtime)
    ├── models/
    ├── results/
    └── plots/
```

## Component Descriptions

### 1. `true_path.py` - Analytical p_t Implementation

**Functions:**
- `a1(t)`, `a2(t)`, `a3(t)` - Schedule functions
- `A1(t)`, `A2(t)`, `A3(t)` - Closed-form integrals (∫₀ᵗ a(s) ds)
- `sigma_p(t, schedule)` - Compute σ_p(t) = exp(A(t))
- `log_p_t(x, t, schedule)` - Log density of p_t at point x
- `score_p_t(x, t, schedule)` - Score ∇log p_t(x) = -x/σ_p(t)²
- `velocity_u(x, t, schedule)` - True velocity u(x,t) = a(t) x
- `sample_p_t(t, batch_size, schedule)` - Sample x ~ p_t

**Key formulas:**
- A₁(t) = (1-cos(πt))/π
- A₂(t) = (0.3/(2π))(1-cos(2πt)) + 0.2t
- A₃(t) = (1/2) t² - (1/2) t
- log p_t(x) = -(d/2) log(2π) - d log σ_p(t) - |x|²/(2σ_p(t)²)

**Schedule enum:**
```python
from enum import Enum
class Schedule(Enum):
    A1 = "a1"  # sin(πt)
    A2 = "a2"  # 0.3 sin(2πt) + 0.2
    A3 = "a3"  # t - 1/2
```

---

### 2. `model.py` - Learned Velocity MLP

**Classes:**
- `VelocityMLP(input_dim=3, hidden_dims=[128, 128], output_dim=2)`
  - Input: [x₁, x₂, t] (concatenated)
  - Output: v_θ(x,t) ∈ ℝ²
  - Activation: SiLU or Softplus

**Functions:**
- `compute_divergence(v_θ, x, t)` - ∇·v_θ via autograd:
  ```python
  divergence = torch.autograd.grad(outputs.sum(), x, create_graph=True)[0].sum(dim=-1)
  ```

---

### 3. `train.py` - Flow Matching Training

**Function:**
```python
def train_velocity_model(
    model, 
    schedule, 
    epochs=300,
    lr=1e-3,
    batch_size=128,
    num_batches_per_epoch=512,
    val_times=64,
    val_samples_per_time=2048,
    early_stop_patience=20,
    target_nmse=1e-2
):
    """
    Returns:
        best_val_mse: float
        training_history: dict with 'train_mse', 'val_mse'
    """
```

**Training loop:**
- Sample t ~ Unif[0,1], x ~ p_t
- Target: u(x,t) = a(t) x
- Loss: MSE(v_θ(x,t) - u(x,t))
- Optimizer: Adam(lr, betas=(0.9, 0.999), weight_decay=1e-6)
- Scheduler: CosineAnnealingLR → final_lr=1e-4
- Validation: grid of 64 times × 2048 samples each
- Early stopping: val MSE plateau for 20 epochs OR NMSE ≤ 0.01

**Save checkpoint:** `data/models/vtheta_schedule_a1.pth`

---

### 4. `eval.py` - Learned Density q_t Evaluation

**Core Functions:**

1. **`backward_ode_and_divergence(v_θ, x, t)`**
   ```python
   """
   Returns: (x_0, ℓ(t))
   
   Solves backward ODE: dx/ds = -v_θ(x(s), s) for s in [t, 0]
   Simultaneously accumulates: ℓ = ∫₀ᵗ ∇·v_θ(x(s),s) ds
   
   State vector: [x₁, x₂, ℓ] (concatenated)
   """
   ```
   
2. **`log_q_t(x, t, v_θ, schedule)`**
   ```python
   """
   Compute log q_t(x) = log p_0(x_0) + ℓ(t)
   where x_0 = backward_ode terminal point
   """
   ```

3. **`score_q_t(x, t, v_θ, schedule)`**
   ```python
   """
   Compute ∇_x log q_t(x) via single autograd call:
   
   x.requires_grad_(True)
   log_q = log_q_t(x, t, v_θ, schedule)  # scalar
   score = torch.autograd.grad(log_q.sum(), x, create_graph=True)[0]
   """
   ```

4. **`compute_kl_lhs(x_batch, t, schedule, v_θ)`**
   ```python
   """
   Estimate KL(p_t|q_t) = E_x~p_t [log p_t(x) - log q_t(x)]
   """
   ```

5. **`compute_rhs_integrand(x_batch, t, schedule, v_θ)`**
   ```python
   """
   Compute ĝ(t) = (1/N) Σᵢ (u⁽ⁱ⁾ - v⁽ⁱ⁾)ᵀ (s_p⁽ⁱ⁾ - s_q⁽ⁱ⁾)
   where:
     u = velocity_u(x, t, schedule)
     v = v_θ(x, t)
     s_p = score_p_t(x, t, schedule)
     s_q = score_q_t(x, t, schedule)
   """
   ```

**ODE Implementation Details:**
- Use `torchdiffeq.odeint` with `method="dopri5"`
- Combined state: `z(s) = [x(s), ℓ(s)]` where x∈ℝ², ℓ∈ℝ
- Dynamics: 
  - `dx/ds = -v_θ(x, s)` for backward
  - `dℓ/ds = ∇·v_θ(x, s)` (accumulated)
- Timing: solve from s=t down to s=0

---

### 5. `experiment.py` - Main Orchestrator

**CLI Arguments:**
```python
parser.add_argument('--schedule', type=str, choices=['a1', 'a2', 'a3'], required=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--num_samples', type=int, default=2000)
parser.add_argument('--num_times', type=int, default=101)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--load_model', type=str, default=None)
```

**Main Flow:**
1. Parse arguments
2. Call `utils.set_seed(args.seed)`
3. Instantiate schedule from `true_path`
4. Create model from `model.py`
5. **Training phase:**
   - If `--load_model`: load checkpoint
   - Else: call `train_velocity_model()` → save checkpoint
6. **Evaluation phase:**
   - Load trained model
   - For each t_k in 101-point grid [0, 1]:
     - Sample N=2000 points from p_t_k
     - Compute LHS: KL(p_t_k|q_t_k)
     - Compute RHS integrand: ĝ(t_k)
   - Integrate RHS with trapezoidal rule
   - Compare LHS vs RHS curves
7. **Output:**
   - Plot overlay of KL(t) and R(t)
   - Print error table (median/max rel error)
   - Save arrays: `t_grid`, `KL_curve`, `rhs_integrand`, `rhs_cumulative`
   - Save metadata (seed, N, K, tolerances, rtol, atol)

**Acceptance Criteria:**
- Median relative error ≤ 3%
- Max relative error ≤ 8%

---

### 6. `utils.py` - Helpers

**Functions:**
- `set_seed(seed)` - Set random seeds for torch, numpy
- `get_device(device_str)` - Return torch.device
- `plot_comparison(t_grid, kl_curve, rhs_curve, schedule, save_path)` - Overlay plot
- `compute_relative_error(rhs, kl)` - Return median and max rel error
- `save_results(t_grid, kl_curve, rhs_integrand, rhs_cumulative, schedule, metadata)`
- `load_checkpoint(path)` / `save_checkpoint(model, path)`

**Plot Format:**
- X-axis: t ∈ [0, 1]
- Y-axis: KL divergence
- Two curves: KL(t) and R(t) overlaid
- Legend, labels, save to `data/plots/kl_comparison_a1.png`

---

## Data Flow

```
User runs: python experiment.py --schedule a1

1. Parse args, set seed (utils.py)
2. Load/create schedule functions (true_path.py)
3. Instantiate MLP (model.py)
4. Train v_θ (train.py)
   └─→ Save checkpoint
5. Load checkpoint
6. For each t_k in 101-point grid:
   a. Sample 2000 x ~ p_t_k (true_path.py)
   b. For each x:
      - backward_ode_and_divergence → x_0, ℓ(t)
      - log_q_t → scalar
      - score_q_t → ∇log q_t (autograd)
   c. Aggregate:
      - LHS: mean(log p_t - log q_t)
      - RHS: mean((u-v)ᵀ(s_p-s_q))
7. Integrate RHS → cumulative curve
8. Compare LHS vs RHS
9. Plot and save results
```

---

## Implementation Order (Coding Sequence)

1. **`utils.py`** - Seed, device, basic helpers
2. **`true_path.py`** - Schedule functions, analytic formulas
3. **`model.py`** - MLP architecture
4. **Test:** Sample p_t, compute log p_t, score p_t
5. **`train.py`** - Training loop with validation
6. **Test:** Train small model, verify MSE decreases
7. **`eval.py`** - Backward ODE with combined state
8. **Test:** Reversibility check (backward then forward)
9. **`experiment.py`** - Wire everything together
10. **Test:** Full run on schedule a1

---

## Key Technical Details

### ODE State for Backward + Divergence
```python
# Combined state vector
def ode_func(s, z):
    x = z[:, :2]  # spatial coords
    ell = z[:, 2:3]  # accumulator
    
    v = v_theta(x, s)
    div = compute_divergence(v_theta, x, s)
    
    dx_ds = -v  # backward flow
    dell_ds = div.unsqueeze(-1)  # divergence accumulation
    
    return torch.cat([dx_ds, dell_ds], dim=1)
```

### Score Computation (Critical)
```python
def score_q_t(x, t, v_theta, schedule):
    """MUST treat log_q_t as scalar and differentiate w.r.t. x"""
    x_grad = x.clone().detach().requires_grad_(True)
    log_q = log_q_t(x_grad, t, v_theta, schedule)  # scalar
    score = torch.autograd.grad(log_q.sum(), x_grad, create_graph=True)[0]
    return score
```

### Batch Processing
- All 2000 samples at time t_k → single ODE call
- State shape: `[B=2000, 3]` (last dim = [x₁, x₂, ℓ])
- If memory issues: chunk into 4×512

### Trapzoidal Integration
```python
rhs_cumulative[0] = 0
for m in range(1, len(t_grid)):
    dt = t_grid[m] - t_grid[m-1]
    rhs_cumulative[m] = rhs_cumulative[m-1] + (dt/2) * (rhs_integrand[m] + rhs_integrand[m-1])
```

---

## Testing Strategy

### Phase 1: True Path
- [ ] Verify σ_p(t) against definition
- [ ] MC check: E[|x|²/d] ≈ σ_p² (tolerance ~2-3%)
- [ ] Finite-diff check: ∇log p_t matches analytic (±1e-4)

### Phase 2: Training
- [ ] Training/val MSE decreases and stabilizes
- [ ] Val MSE/NMSE reasonable (nonzero but small)
- [ ] Early stopping triggers appropriately

### Phase 3: Evaluation Core
- [ ] Reversibility: backward then forward recovers x (±1e-5)
- [ ] Edge case: KL(t=0) ≈ 0
- [ ] Gradient check: autograd score vs finite-diff (±1e-3)

### Phase 4: Full Integration
- [ ] LHS and RHS curves visually match
- [ ] Doubling N changes KL by < 3%
- [ ] Doubling K changes RHS by < 2-3%
- [ ] Acceptance: median ≤ 3%, max ≤ 8%

---

## File Dependencies

```
experiment.py depends on:
  ├─ true_path.py
  ├─ model.py
  ├─ train.py
  ├─ eval.py
  └─ utils.py

eval.py depends on:
  ├─ model.py
  ├─ true_path.py
  └─ torchdiffeq

train.py depends on:
  ├─ model.py
  └─ true_path.py
```

---

## Notes

- Use float64 everywhere for numerical stability
- ODE solver: `torchdiffeq.odeint` with `method="dopri5"`, `rtol=1e-6`, `atol=1e-8`
- Max steps: 10000
- No adjoint method (need gradients w.r.t. x for score computation)
- Edge case: t=0 → skip ODE, return log p_0(x) directly

---

## Acceptance Checklist

After running `python experiment.py --schedule a1`:
- [ ] Model trains successfully
- [ ] Checkpoint saved to `data/models/`
- [ ] Evaluation completes without errors
- [ ] Reversibility test passed
- [ ] Gradient check passed
- [ ] KL(t=0) ≈ 0
- [ ] Median relative error ≤ 3%
- [ ] Max relative error ≤ 8%
- [ ] Plot generated and saved
- [ ] Arrays saved (pickle/numpy)
- [ ] Metadata saved (JSON)

Repeat for schedules a2 and a3.

