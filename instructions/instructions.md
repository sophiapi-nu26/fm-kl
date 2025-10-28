Below is a complete, implementation-ready spec, written for a developer with no flow-matching background. It generalizes to any scalar schedule (a(t)) with velocity field (u(x,t)=a(t),x). Follow it in order. No code is required here—just do exactly what’s described.

---

# 0) Environment and global settings

* Precision: use float64 everywhere.
* Randomness: fix a global seed.
* Device: CPU is fine; GPU optional.
* ODE solver: adaptive RK (Dormand–Prince/Tsit5), rtol = 1e-6, atol = 1e-8.
* Dimension: (d=2).

---

# 1) Define the true path (p_t) and helpers

## 1.1 Schedules (a(t))

Implement the following schedules:

* (a_1(t)=\sin(\pi t))
* (a_2(t)=0.3\sin(2\pi t)+0.2)
* (a_3(t)=t-\tfrac12)

For each schedule, implement

* (A(t)=\int_0^t a(s),ds) (closed forms below)

  * (a_1): (A(t)=\tfrac{1-\cos(\pi t)}{\pi})
  * (a_2): (A(t)=\tfrac{0.3}{2\pi}\big(1-\cos(2\pi t)\big)+0.2t)
  * (a_3): (A(t)=\tfrac12 t^2-\tfrac12 t)
* (\sigma_p(t)=\exp(A(t)))

## 1.2 True Gaussian marginals and score

For any (t) and (x\in\mathbb R^2):

* (p_t = \mathcal N!\big(0,\sigma_p(t)^2 I_2\big))
* (\log p_t(x) = -\tfrac d2\log(2\pi)-d\log\sigma_p(t)-\tfrac{|x|^2}{2\sigma_p(t)^2})  (with (d=2))
* (\nabla\log p_t(x) = -,x/\sigma_p(t)^2)

## 1.3 True velocity (for supervision)

* (u(x,t)=a(t),x)

## 1.4 Sampling from (p_t)

To draw (x\sim p_t): sample (z\sim\mathcal N(0,I_2)), set (x=\sigma_p(t),z).

### Tests (unit)

* Verify (\sigma_p(t)) against its definition for several (t).
* Monte Carlo check: draw many (x\sim p_t), estimate (\mathbb E|x|^2/d) ≈ (\sigma_p(t)^2) (tolerance ~2–3% with 10k samples).
* Finite-difference check: (\nabla\log p_t(x)) matches analytic formula (central differences on (\log p_t) at random (x)) within 1e-4 absolute.

---

# 2) Define and train the learned velocity (v_\theta(x,t))

## 2.1 Model

* MLP taking concatenated ([x\in\mathbb R^2,; t\in\mathbb R]); 2–3 hidden layers, width 64–128, SiLU/Softplus; output (\in\mathbb R^2).
* Exact divergence via autograd: (\nabla!\cdot v_\theta(x,t)=\sum_{i=1}^2 \partial v_{\theta,i}/\partial x_i).

## 2.2 Training (flow-matching)

* Minibatch sampling:

  * (t\sim \mathrm{Unif}[0,1]) (draw a batch of times)
  * For each (t), sample (x\sim p_t) as in §1.4
* Supervise to (u(x,t)=a(t),x), minimize MSE: (|v_\theta(x,t)-u(x,t)|^2).
* Optimizer: Adam. Stop when validation MSE plateaus.

### Tests (training sanity)

* Show training/validation MSE decreases then stabilizes.
* On a held-out grid of ((t,x)), report (\mathbb E|v_\theta-u|^2). Expect small but nonzero (we want an imperfect fit to make KL nontrivial).

---

# 3) Time grid and evaluation batches

## 3.1 Time grid (columns)

* Define (K=101) time checkpoints: (t_k=k/(K-1)) for (k=0,\dots,K-1).

## 3.2 Samples per time (rows)

* For each (t_k), draw (N=2000) i.i.d. samples (x^{(i)}*{t_k}\sim p*{t_k}).
* Reuse the same batch at (t_k) for both LHS and RHS calculations.

### Tests

* Verify that for each (t_k), the empirical covariance of the batch is close to (\sigma_p(t_k)^2 I_2) (±5–10% component-wise with (N=2000)).

---

# 4) Backward flow and learned density (q_t)

This is the core routine used repeatedly below.

## 4.1 Backward IVP (rewind)

Given a **fixed** time (t) and terminal point (x):

* Solve the ODE backward in time: (\dot x(s) = -,v_\theta(x(s),s)), from (s=t) down to (0), with (x(t)=x).
* This produces (x_0:=x(0)), the unique preimage under the learned flow.

## 4.2 Accumulate log-volume change (divergence integral)

Along the same backward run, accumulate
[
\ell(t)=\int_{0}^{t} \nabla!\cdot v_\theta\big(x(s),s\big),ds.
]

## 4.3 Log-density and score of (q_t) at an arbitrary (x)

* Log-density:
  [
  \log q_t(x) = \log p_0(x_0) + \ell(t),\qquad \log p_0(x_0)=-\tfrac d2\log(2\pi)-\tfrac{|x_0|^2}{2}.
  ]
* Score (gradient w.r.t the terminal point (x)):

  * Treat (\log q_t(x)) as a scalar function of the terminal (x).
  * Compute (\nabla_x \log q_t(x)) via autograd by differentiating the scalar (\log q_t(x)) w.r.t. (x).
  * Do **not** attempt to hand-assemble the gradient from parts; rely on one gradient call.

### Tests (core correctness)

* **Reversibility:** pick random (x,t); backward to (x_0); then solve forward (\dot x=v_\theta) from (0\to t); the terminal point must match original (x) within ODE tolerance (‖error‖ ≤ 1e-5).
* **1D linear sanity (optional):** set (v_\theta(x,t)=\alpha(t)x) with a known (\alpha); check (\log q_t) against the analytic Gaussian (\mathcal N(0,e^{2A(t)}I)); score should be (-x/e^{2A(t)}) (error ≤ 1e-3).
* **Gradient check:** finite-difference (\nabla_x \log q_t(x)) at a few random points; compare to autograd (error ≤ 1e-3 with step (h=10^{-4})).

---

# 5) Left-hand side: (\mathrm{KL}(p_t|q_t)) for each (t_k)

For each (k):

1. Use the batch ({x^{(i)}*{t_k}}\sim p*{t_k}).
2. Compute (\log p_{t_k}(x^{(i)}_{t_k})) using §1.2.
3. Compute (\log q_{t_k}(x^{(i)}_{t_k})) using §4.
4. Estimate
   [
   \widehat{\mathrm{KL}}(t_k)=\frac{1}{N}\sum_{i=1}^N\Big[\log p_{t_k}\big(x^{(i)}*{t_k}\big)-\log q*{t_k}\big(x^{(i)}_{t_k}\big)\Big].
   ]

### Tests

* **Edge case (t=0):** (\widehat{\mathrm{KL}}(0)\approx 0) (|value| ≤ 1e-3) since (q_0=p_0) by construction.
* **Stability:** doubling (N) changes (\widehat{\mathrm{KL}}(t_k)) by < 3% for typical (t_k).

---

# 6) Right-hand side integrand: (\hat g(t_k))

For each (k):

1. For each (x^{(i)}_{t_k}) in the batch, compute:

   * (u^{(i)}=a(t_k),x^{(i)}_{t_k})
   * (v^{(i)}=v_\theta\big(x^{(i)}_{t_k},t_k\big))
   * (s_p^{(i)}=\nabla\log p_{t_k}\big(x^{(i)}*{t_k}\big)=-x^{(i)}*{t_k}/\sigma_p(t_k)^2)
   * (s_q^{(i)}=\nabla\log q_{t_k}\big(x^{(i)}_{t_k}\big)) from §4
2. Average:
   [
   \hat g(t_k)=\frac{1}{N}\sum_{i=1}^N \Big(u^{(i)}-v^{(i)}\Big)^\top \Big(s_p^{(i)}-s_q^{(i)}\Big).
   ]

### Tests

* **Zero-check with perfect teacher:** if you temporarily set (v_\theta\equiv u), then (\hat g(t_k)) should be ≈ 0 for all (k).
* **Variance sanity:** (\hat g(t_k)) should stabilize (std error (\sim N^{-1/2})); increasing (N) by 4 should halve its std dev.

---

# 7) Time integration of the RHS

Compute the cumulative integral on the grid:

* (R(0)=0)
* For (m=1,\dots,K-1):
  [
  R(t_m)=R(t_{m-1})+\frac{t_m-t_{m-1}}{2}\Big(\hat g(t_m)+\hat g(t_{m-1})\Big).
  ]

### Tests

* **Quadrature refinement:** If you double (K) (finer grid), the curve (R(t)) should change by < 2–3% pointwise (once ODE tolerances and (N) are not bottlenecks).

---

# 8) Final comparison and acceptance

* Plot both curves over (t\in[0,1]):

  * LHS: (t\mapsto \widehat{\mathrm{KL}}(t))
  * RHS: (t\mapsto R(t))
* Compute relative error at grid points:
  [
  \text{RelErr}(t_k)=\frac{\big|R(t_k)-\widehat{\mathrm{KL}}(t_k)\big|}{\max(10^{-8},\widehat{\mathrm{KL}}(t_k))}.
  ]

**Acceptance thresholds (per schedule (a(t))):**

* Median RelErr ≤ 3%
* Max RelErr ≤ 8%
* If thresholds are missed, tighten ODE tolerances, increase (N) and/or (K), ensure score of (q_t) is computed via the **single autograd gradient of the scalar (\log q_t(x))** (common pitfall).

---

# 9) Performance and batching notes

* **Batch over points at a fixed time:** the common case is one scalar (t) with a batch of (x). Solve the backward ODE **once** for the whole batch (the solver can carry batches).
* **Group by time:** if you must handle different (t) per point, group by unique times and process per group.
* **Detach wisely:** the evaluation phase only needs gradients w.r.t. terminal (x), not w.r.t. model parameters; avoid building unnecessary graphs across calls.

---

# 10) Deliverables

* For each schedule (a(t)): arrays `t_grid`, `KL_curve`, `rhs_integrand`, `rhs_cumulative`; the plot; metadata (seed, (N), (K), tolerances).
* A short report: training MSE trace, reversibility error stats, gradient check residuals, quadrature/MC convergence notes, and the final error table (median/max RelErr).

---

Follow this spec exactly. If anything deviates (e.g., LHS and RHS curves disagree), run the stepwise tests in §§4–7 to pinpoint and fix the issue.
