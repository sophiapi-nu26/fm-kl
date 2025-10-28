# Primer: What this experiment is doing (and why)

## Goal (what we’re validating)

We want to **empirically verify a precise identity** about how the Kullback–Leibler divergence between two time-evolving distributions changes over time. A “true” distribution (p_t) evolves under a known velocity field (u); a “learned” distribution (q_t) evolves under a learned field (v_\theta). The identity says the **rate of change of KL** is exactly the **inner product of two local errors**:
[
\frac{d}{dt},\mathrm{KL}(p_t|q_t)=
\mathbb E_{x\sim p_t}!\big[(u-v_\theta)^\top(\nabla\log p_t-\nabla\log q_t)\big].
]
Integrating from (0) to (t) (since (p_0=q_0)) gives
[
\mathrm{KL}(p_t|q_t)=\int_0^t \mathbb E_{x\sim p_s}!\big[(u-v_\theta)^\top(\nabla\log p_s-\nabla\log q_s)\big];ds.
]
Our experiment checks that **LHS (measured KL)** matches **RHS (time-integrated local misalignment)** across (t\in[0,1]).

## Setup (objects we control)

* **True path (p_t) in 2D.** Pick a scalar schedule (a(t)) and set (u(x,t)=a(t),x). This yields a zero-mean isotropic Gaussian path (p_t=\mathcal N(0,\sigma_p(t)^2 I_2)) with (\sigma_p(t)=\exp(\int_0^t a(s)ds)). Both (\log p_t(x)) and its score (\nabla\log p_t(x)=-x/\sigma_p^2(t)) are closed-form.
* **Learned field (v_\theta(x,t)).** A small MLP trained by **flow matching**: sample (t\sim\mathrm{Unif}[0,1]), sample (x\sim p_t), regress (v_\theta(x,t)) to the known target (u(x,t)=a(t)x) with MSE. This defines a learned flow (ODE) whose pushforward distributions are (q_t).

## The two things we must be able to evaluate at arbitrary ((x,t))

* **(\log q_t(x))** (learned density at time (t) at point (x)):
  Run the **backward flow** from ((x,t)) to (s=0) with (\dot x=-v_\theta(x,s)) to find the preimage (x_0). While rewinding, accumulate the **divergence integral** (\ell=\int_0^t \nabla!\cdot v_\theta(x(s),s),ds). Then
  [
  \log q_t(x)=\log p_0(x_0)+\ell.
  ]
  Intuition: (\log q_t) = **base log-density at the source point** (x_0) **plus total log volume change** along the path to (x).
* **(\nabla_x\log q_t(x))** (the learned score at time (t)):
  Treat (\log q_t(x)) as a scalar function of the terminal (x) and take its gradient w.r.t. (x). Intuition: how the log-probability at time (t) changes if you nudge the endpoint (x).

## The actual evaluation workflow (what you’ll compute)

1. **Train (v_\theta)** by regression to (u(x,t)=a(t)x). Save a checkpoint with nonzero but small MSE.
2. **Fix a time grid** (t_0,\dots,t_{K}) on ([0,1]) (e.g., 101 points). This is for numerical integration and plotting.
3. **For each time (t_k)**, draw a batch of (N) samples (x^{(i)}\sim p_{t_k}).
4. **At each sample (x^{(i)}):** compute

   * Analytic **(\log p_{t_k}(x^{(i)}))** and **(s_p=-x^{(i)}/\sigma_p(t_k)^2)**.
   * **(\log q_{t_k}(x^{(i)}))** by backward flow + divergence integral.
   * **(s_q=\nabla_x\log q_{t_k}(x^{(i)}))** by differentiating the scalar (\log q_{t_k}(x)) w.r.t. the terminal (x).
   * **(u=a(t_k),x^{(i)})** and **(v=v_\theta(x^{(i)},t_k))**.
5. **Aggregate per time (t_k):**

   * **LHS** (measured KL): (\widehat{\mathrm{KL}}(t_k)=\tfrac1N\sum_i[\log p_{t_k}(x^{(i)})-\log q_{t_k}(x^{(i)})]).
   * **RHS integrand:** (\hat g(t_k)=\tfrac1N\sum_i (u-v)^\top(s_p-s_q)).
6. **Integrate RHS over time** by trapezoidal rule on the grid:
   (R(t_m)\approx \sum_{k\le m}\tfrac{t_k-t_{k-1}}{2}\big(\hat g(t_k)+\hat g(t_{k-1})\big)).
7. **Compare curves:** plot (t\mapsto \widehat{\mathrm{KL}}(t)) vs (t\mapsto R(t)). They should coincide (within MC + ODE + quadrature error).

## What “success” looks like (and how we sanity-check)

* **Match:** median relative gap ≤ ~3%, max ≤ ~8% after tightening solver tolerances and using enough samples/grid points.
* **Numerical checks:**
  (i) **Reversibility** of backward+forward flow (endpoint recovered within tolerance).
  (ii) **Gradient check** for (\nabla_x\log q_t(x)) via finite differences.
  (iii) **Convergence** when you increase sample count (N) or refine the time grid (K).
* **Common pitfalls to avoid:**
  • Forgetting the chain rule when computing the score—always take the gradient of the **full scalar** (\log q_t(x)) w.r.t. (x).
  • Wrong sign on the divergence accumulator (it’s (+\int_0^t \nabla!\cdot v_\theta)).
  • Using (u(x,t)=0.3\sin(2\pi t)x+0.2) instead of ((0.3\sin(2\pi t)+0.2),x) (the velocity is **linear in (x)**, no constant offset).
  • Hard-coding the (-\log(2\pi)) constant of (\log p_0) for (d=2) and later changing (d).

That’s the north star: **turn the learned velocity into an evaluatable density (q_t) via backward flow and divergence; compute KL(t) directly under (p_t); compute the RHS by local inner products and integrate; show the two match across time.**
