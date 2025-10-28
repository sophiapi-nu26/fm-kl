Good—since LHS checks pass, validate the RHS in isolation. Below are decisive tests, from easiest to strongest. Do them in order.

---

## RHS definition (for reference)

At each time (t):
[
g(t)=\mathbb{E}*{x\sim p_t}!\Big[(u(x,t)-v*\theta(x,t))^\top\big(\nabla\log p_t(x)-\nabla\log q_t(x)\big)\Big].
]
You estimate (g(t)) by MC, then compute (R(t)=\int_0^t g(s)ds) via trapezoid.

---

## R0. Trivial identity check (no training): (v=u)

* Expectation: (g(t)\equiv 0) for all (t).
* If not 0 within MC error, your **score or divergence** pipeline is still off.

---

## R1. Analytic “wrong-model” checks (no learning ambiguity)

### (a) (v\equiv 0)  (so (q_t=p_0))

* Derive:
  [
  \nabla\log p_t(x)=-\frac{x}{\sigma_p^2(t)},\quad
  \nabla\log q_t(x)=-x,\quad
  u(x,t)=a(t)x.
  ]
* Then
  [
  g(t)=\mathbb E_{p_t}!\Big[a(t)x^\top!\Big(!- \frac{x}{\sigma_p^2} + x\Big)\Big]
  =a(t),(1-\tfrac{1}{\sigma_p^2}),\mathbb E_{p_t}|x|^2
  =a(t),d,\big(\sigma_p^2(t)-1\big).
  ]
  (Here (d=2), (\sigma_p(t)=e^{A(t)}), (A(t)=\int_0^t a(s)ds).)
* **Test:** estimate (g(t)) over the grid and compare to the closed form (d,a(t)(\sigma_p^2(t)-1)). Max rel. error ≤ 3–5% with (N\ge 2000).
* **Bonus:** integrate your numeric (g(t)); it must match the analytic KL
  (\frac d2(\sigma_p^2-1-\log\sigma_p^2)).

### (b) Scaled field (v=c,u) (e.g., (c=0.8))

* Now (q_t=\mathcal N(0,\sigma_q^2 I)) with (\sigma_q^2(t)=e^{2cA(t)}) and (\nabla\log q_t(x)=-x/\sigma_q^2(t)).
* Compute
  [
  g(t)=(1-c),a(t),d,\Big(e^{2(1-c)A(t)}-1\Big).
  ]
* **Test:** numeric (g(t)) vs that formula; integrate and compare to analytic KL
  (\tfrac d2\big(r-1-\log r\big)) with (r=e^{2(1-c)A(t)}).

If (a) and (b) pass, your **RHS integrand** code is correct.

---

## R2. Derivative consistency with the LHS (for your trained (v_\theta))

* Compute a **central finite-difference** derivative of the LHS:
  [
  \dot{\mathrm{KL}}(t_k)\approx\frac{\mathrm{KL}(t_{k+1})-\mathrm{KL}(t_{k-1})}{t_{k+1}-t_{k-1}}
  ]
  (use CRN across (t) to reduce noise).
* **Test:** (\dot{\mathrm{KL}}(t_k)) ≈ (g(t_k)) within MC standard error. Quantify SE for both; the difference should be (\lesssim) 1–2× the combined SE. Large systematic gaps → a bug in either (s_q) or time bookkeeping.

---

## R3. Internal decomposition sanity

Split (g(t)) into two sample means:
[
g(t)=\underbrace{\mathbb E_{p_t}\big[(u-v)^\top\nabla\log p_t\big]}*{G_p(t)}
-\underbrace{\mathbb E*{p_t}\big[(u-v)^\top\nabla\log q_t\big]}_{G_q(t)}.
]

* **Closed form for (G_p(t)) (cross-check):** since (x\sim \mathcal N(0,\sigma_p^2 I)),
  [
  G_p(t)= -a(t),d + \frac{1}{\sigma_p^2(t)},\mathbb E_{p_t}\big[v_\theta(x,t)^\top x\big].
  ]
  Compute the RHS via MC and verify it matches the direct sample mean of ((u-v)^\top\nabla\log p_t). If this fails, you have a bookkeeping or sampling bug.
* Then verify (g(t)=G_p(t)-G_q(t)) numerically to machine precision (it’s the same numbers reorganized). If this fails, your batching/reduction path is buggy.

---

## R4. Variance diagnostics (to know when you can trust (g))

For each (t), compute the sample variance of the per-sample integrand
[
\delta_i=(u-v)^\top(s_p-s_q).
]
Report (\mathrm{SE}[g(t)]=\sqrt{\mathrm{Var}(\delta)/N}). If the discrepancy to either the finite-difference derivative or analytic formulas is within SE, you’re fine; otherwise increase (N), use **common random numbers** and **antithetics**.

---

## Acceptance summary

* R0: (g(t)\equiv 0) for (v=u).
* R1(a,b): numeric (g(t)) matches analytic forms (and integrals match analytic KL).
* R2: (g(t)) matches finite-difference (\dot{\mathrm{KL}}(t)) within SE for your trained (v_\theta).
* R3: decomposition checks hold exactly (within MC error).

If all four pass, the RHS side is trustworthy.
