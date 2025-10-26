Below is a self-contained spec to implement §6.1 end-to-end. 

# Objective

Numerically validate the KL evolution identity by showing, for all (t\in[0,1]),
[
\mathrm{KL}(p_t|q_t);\approx;\int_0^t \mathbb{E}*{x\sim p_s}!\big[(u-v*\theta)^\top(\nabla\log p_s-\nabla\log q_s)\big],ds,
]
where (p_t) is a known Gaussian path driven by a true velocity (u(x,t)=a(t)x), and (q_t) is the path induced by a learned velocity field (v_\theta(x,t)).

# Ground truth path

* Dimension: (d=2).
* Three choices of scalar schedule (a(t)\in{\sin(\pi t),;0.3\sin(2\pi t)+0.2,;t-\tfrac12}).
* Define (A(t)=\int_0^t a(s),ds) and (\sigma_p(t)=\exp\big(A(t)\big)).
* True marginals: (p_t=\mathcal N(0,\sigma_p(t)^2 I_2)).
* Closed forms needed:

  * (\log p_t(x)=-\tfrac d2\log(2\pi)-d\log\sigma_p(t)-\tfrac{|x|^2}{2\sigma_p(t)^2}).
  * (\nabla\log p_t(x)=-x/\sigma_p(t)^2).
  * (u(x,t)=a(t),x).

# Model to train

* (v_\theta:\mathbb R^2\times[0,1]\to\mathbb R^2): small MLP (2–3 hidden layers, width 64–128, SiLU/Softplus), inputs concatenated ([x,t]).
* Divergence (\nabla!\cdot v_\theta(x,t)=\sum_{i=1}^2 \partial v_{\theta,i}/\partial x_i) via autograd (exact in 2D).

# Training (flow matching)

* Precision: use float64 everywhere.
* Sample minibatches: (t\sim\mathrm{Unif}[0,1]); sample (x\sim p_t) by drawing (z\sim\mathcal N(0,I_2)) and setting (x=\sigma_p(t),z).
* Supervision target: (u(x,t)=a(t)x).
* Loss: MSE (\mathbb E|v_\theta(x,t)-u(x,t)|^2).
* Optimizer: Adam; run to a small but nonzero validation error (we need imperfect fit so KL is nonzero).
* Save the trained weights; record the validation MSE (\widehat\varepsilon^2) (useful later).

# Evaluation grid (the “matrix” you’ll fill)

* Time axis (columns): choose a fixed grid (t_k=k/(K-1)), (k=0,\dots,K-1) (use (K=101)).
* Sample axis (rows): for each column (t_k), draw (N) i.i.d. samples (x^{(i)}*{t_k}\sim p*{t_k}) (use (N=2000)).
* You will compute, for every cell ((k,i)), the following quantities and then reduce:

  * (X[k,i]=x^{(i)}_{t_k})
  * (\log p[k,i]=\log p_{t_k}(X[k,i])) (analytic)
  * (s_p[k,i]=\nabla\log p_{t_k}(X[k,i])) (analytic)
  * (\log q[k,i]=\log q_{t_k}(X[k,i])) (via backward ODE; see next section)
  * (s_q[k,i]=\nabla\log q_{t_k}(X[k,i])) (autograd through backward ODE)
  * (u[k,i]=a(t_k),X[k,i])
  * (v[k,i]=v_\theta(X[k,i],t_k))

# Backward ODE change-of-variables for (q_t) (core numeric routine)

Purpose: evaluate (q_t(x)) and (\nabla_x\log q_t(x)) at arbitrary ((x,t)), not just along the learned flow.

* Input: terminal state (x_t=x), time (t), model (v_\theta).
* Integrate **backward in time** from (s=t) to (0) the system:

  * State ODE: (\dot x_s=-,v_\theta(x_s,s)) with (x_t=x).
  * Log-density accumulator: (\dot\ell_s=+,\nabla!\cdot v_\theta(x_s,s)), (\ell_0=0).
* Outputs:

  * The recovered (x_0) and accumulated (\ell_t=\int_0^t \nabla!\cdot v_\theta(x_s,s),ds).
  * (\log q_t(x)=\log p_0(x_0)+\ell_t) with (\log p_0(x_0)=-\tfrac d2\log(2\pi)-\tfrac{|x_0|^2}{2}).
  * The score (s_q=\nabla_x\log q_t(x)) obtained by differentiating the scalar (\log q_t(x)) w.r.t. the terminal (x) via autograd (ensure the ODE is run with `adjoint=False` so gradients propagate through the solver).
* Solver: adaptive RK (Dormand–Prince / Tsit5), tolerances rtol (=10^{-6}), atol (=10^{-8}).

# Estimators at each time (t_k)

1. **KL (left side)**
   [
   \widehat{\mathrm{KL}}(t_k)=\frac1N\sum_{i=1}^N\Big(\log p_{t_k}(X[k,i])-\log q_{t_k}(X[k,i])\Big).
   ]

2. **RHS pointwise integrand**
   [
   \hat g(t_k)=\frac1N\sum_{i=1}^N \big(u[k,i]-v[k,i]\big)^\top\Big(s_p[k,i]-s_q[k,i]\Big).
   ]

# Time integration (to get the RHS curve)

* Compute the cumulative integral on the grid using the trapezoidal rule:

  * (R[0]=0).
  * For (m=1,\dots,K-1),
    [
    R[m]=R[m-1]+\tfrac{1}{2}\big(t_m-t_{m-1}\big)\big(\hat g(t_m)+\hat g(t_{m-1})\big).
    ]
* Interpretation: (R[m]\approx \int_0^{t_m}\mathbb E_{p_s}[(u-v_\theta)^\top(\nabla\log p_s-\nabla\log q_s)]ds).

# Plots & outputs

* For each (a(t)), produce a single figure with two curves over (t\in[0,1]):

  * (t\mapsto \widehat{\mathrm{KL}}(t)) (LHS).
  * (t\mapsto R(t)) (integrated RHS).
* Include a small inset or second panel with relative error:
  [
  \text{RelErr}(t_k)=\frac{\big|R[k]-\widehat{\mathrm{KL}}(t_k)\big|}{\max(10^{-8},,\widehat{\mathrm{KL}}(t_k))}.
  ]
* Save arrays: `t_grid`, `KL_curve`, `rhs_integrand`, `rhs_cumulative`, plus run metadata (seed, tolerances, (N,K), (a(t)) choice, commit hash).

# Diagnostics (must implement)

* Sanity on training: plot validation (\mathbb E|v_\theta-u|^2) vs epochs; it should decrease and plateau.
* Numerical checks:

  * Increase (N) and/or refine (K) to confirm curves converge.
  * Tighten ODE tolerances; verify the curves move by less than a few percent.
  * Spot-check reversibility: take a random ((x,t)), run backward to (x_0), then forward with (+v_\theta) back to (t); ensure the terminal state is recovered within solver tolerance.

# Determinism & performance

* Set a fixed random seed; set PyTorch default dtype to float64.
* 2D runs are fine on CPU; GPU optional for faster autograd through ODE.
* Cache per-time Monte-Carlo batches and reuse the **same** samples for LHS and RHS at each (t_k) to reduce variance.

# Acceptance criteria

* For each (a(t)), the curves (t\mapsto \widehat{\mathrm{KL}}(t)) and (t\mapsto R(t)) overlap visually across ([0,1]); median relative error (\le 3%), max (\le 8%) after tightening tolerances and using (N\ge 2000), (K\ge 101).
* Diagnostics show stability under increased (N)/refined grid/tighter tolerances.

# Deliverables

* Trained model checkpoint(s) and evaluation artifacts for all three (a(t)).
* Saved arrays (`.npz` or similar) and plots (`.pdf/.png`) per schedule.
* A short README with exact command(s) used, environment (library versions), and measured error stats.
