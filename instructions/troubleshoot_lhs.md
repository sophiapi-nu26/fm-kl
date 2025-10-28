Your LHS pipeline is almost certainly the culprit. Here’s a surgical, **LHS-focused** debug ladder that will tell you exactly where it’s broken, with concrete pass/fail criteria and what each failure means. Run them in order; they are quick and decisive.

---

## A. Golden-path checks with a **known** velocity (no training)

Set (v(x,t)=u(x,t)=a(t),x) (e.g., (a_1(t)=\sin\pi t)). These isolate your ODE + divergence + density code from the learned model.

### A1) Preimage (backward ODE) is correct

* Analytic: (A(t)=\int_0^t a(s)ds), (x_0^\star=e^{-A(t)}x).
* Compute (x_0) by solving **backward** (\dot x=-u(x,s)) from (s=t\to 0) with (x(t)=x).
* **Pass:** (|x_0-x_0^\star|\le 1e{-5}|x|).
  **Fail ⇒** you’re integrating in the wrong direction or passing the time vector with the wrong order/sign.

### A2) Divergence integral has the **right sign**

* Analytic: (\nabla!\cdot u = d,a(t)) (here (d=2)); (\ell^\star(t)=\int_0^t\nabla!\cdot u,ds=d,A(t)).
* Accumulate (\ell(t)) **along the same backward trajectory** using
  (\dot\ell=+\nabla!\cdot u(x(s),s)).
* **Pass:** (|\ell(t)-dA(t)|\le 1e{-5}).
  **Fail ⇒** wrong sign on (\ell) or evaluating divergence at the wrong place/time (must be at ((x(s),s)), not ((x,t)) or ((x_0,s))).

### A3) Log-density equality when (v=u)

* Your formula: (\log q_t(x)=\log p_0(x_0)+\ell(t)), with (\log p_0(x_0)=-\tfrac d2\log(2\pi)-\tfrac12|x_0|^2).
* Analytic: (\log p_t(x)=-\tfrac d2\log(2\pi)-d\log\sigma_p(t)-\frac{|x|^2}{2\sigma_p^2(t)}), (\sigma_p(t)=e^{A(t)}).
* **Pass:** (|\log q_t(x)-\log p_t(x)|\le 1e{-4}) on random (x,t).
  **Fail ⇒** (i) sign in A2 still wrong, (ii) wrong **dimension constant** in (\log p_0), or (iii) not using the evolving (x(s)) for (\ell).

If A1–A3 don’t all pass, fix them before anything else. With (v=u), **KL(t) must be ≈0**.

---

## B. Two tell-tales that catch LHS bugs instantly

### B1) Normalization check (on your current (v_\theta))

For any (t):
[
M(t):=\mathbb E_{x\sim p_t}!\left[\exp\big(\log q_t(x)-\log p_t(x)\big)\right]
=\int p_t(x),\frac{q_t(x)}{p_t(x)}dx=\int q_t(x)dx=1.
]

* **Compute (M(t))** by Monte Carlo using the same samples you use for KL(t).
* **Pass:** (M(t)\approx 1) (±0.02 with (N!\ge!2000)).
  **Fail ⇒** your (\log q_t) is miscomputed (almost always the divergence sign/convention or time direction).

### B2) Divergence-sign flip A/B test (cheap and decisive)

Compute (\log q_t(x)) **twice**, once with

* **Convention A:** integrate **backward** with (\dot x=-v), (\dot\ell=+,\nabla!\cdot v);
* **Convention B:** integrate **forward** with (\dot x=+v), (\dot\ell=-,\nabla!\cdot v) and then use the appropriate inverse mapping.
  For (v=u), **only one** convention yields (\log q_t=\log p_t) (A is simplest). If your current pipeline matches the “wrong” one, you’ve found the sign bug.

Run A1–A3 and B1 first. They take minutes and will tell you exactly where the LHS pipeline is broken.
