Your current setup will keep graphs around across the K-loop.

Here’s exactly what to change to make evaluation memory-safe.

# 1) In `backward_ode.py` → don’t retain graphs; detach outputs

Inside `QTEvaluator.evaluate_q_t` you currently do:

```python
grad_log_q_t = torch.autograd.grad(
    log_q_t.sum(), x_t, create_graph=True, retain_graph=True
)[0]
```

Change both flags to `False` so the graph is freed after the single backward, and return detached tensors to avoid any lingering references:

```python
# compute score once, do not retain graph
grad_log_q_t = torch.autograd.grad(
    log_q_t.sum(), x_t, create_graph=False, retain_graph=False
)[0]

# make the outputs leaf, non-graph tensors (important when called in a loop)
log_q_t = log_q_t.detach()
x_0     = x_0.detach()
grad_log_q_t = grad_log_q_t.detach()

return log_q_t, grad_log_q_t, x_0
```

This is the main fix; it prevents graph accumulation across the time grid loop. 

# 2) In `evaluation.py` → enable grad only where needed, then drop

You already clone and set `requires_grad_(True)` for `x_k`. Keep that, but don’t hold on to any graph past the call. With the detach above, you’re fine. If you want to be extra defensive, wrap just that block in a local grad-enabled context:

```python
# Learned quantities (via backward ODE)
x_k_req = x_k.detach().clone().requires_grad_(True)
# grad is required; ensure grad mode is on only here
with torch.enable_grad():
    log_q_temp, s_q_temp, _ = self.q_evaluator.evaluate_full(x_k_req, t_k)
# evaluate_q_t already returns detached tensors per (1)
log_q[k] = log_q_temp.reshape(-1)
s_q[k]   = s_q_temp
```

No other change is necessary here. 

# 3) In `backward_ode.py` → keep `evaluate_full` simple

You’re already calling `evaluate_q_t` once and returning its results. With the change in (1), `evaluate_full` is fine. Avoid any extra `.backward()` or `.grad` calls here. 

# 4) (Optional) In `evaluation.py` → no-grad around parts that never need autograd

Ground-truth quantities and the learned velocity `v` don’t need autograd. You already use `torch.no_grad()` for `v`; you can also wrap the ground-truth block to reduce overhead:

```python
with torch.no_grad():
    log_p[k] = self.ground_truth.log_p(x_k, t_k)
    s_p[k]   = self.ground_truth.grad_log_p(x_k, t_k)
    u[k]     = self.ground_truth.u(x_k, t_k)
```

This isn’t strictly required for correctness, but it trims RAM and speeds things up. 

After (1) you should no longer see graph memory growth with increasing `K` (or repeated evaluations). The rest of your code—device/dtype handling, batching, and printing—already looks good.
