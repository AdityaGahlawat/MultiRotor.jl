# Differential Flatness Controller

Trajectory tracking controller derived from the control-affine dynamics in `dynamics.md`.

---

## System

$$
\dot{x} = f(x) + g \cdot u
$$

Expanded dynamics:

$$
\dot{v}_b = R_{i2b} g_I - \omega \times v_b + \frac{1}{m} F_b \tag{1}
$$

$$
\dot{\omega} = -J^{-1}(\omega \times J\omega) + J^{-1} \tau_b \tag{2}
$$

$$
\dot{p} = R_{b2i} v_b \tag{3}
$$

$$
\dot{q} = \frac{1}{2} \Omega(\omega) q \tag{4}
$$

**State:** $x = [v_b, \omega, p, q]^T \in \mathbb{R}^{13}$

- $v_b \in \mathbb{R}^3$: body-frame velocity (ft/s)
- $\omega \in \mathbb{R}^3$: body-frame angular velocity (rad/s)
- $p \in \mathbb{R}^3$: inertial position in NED frame (ft)
- $q \in \mathbb{R}^4$: quaternion (inertial-to-body), $q = [q_0, q_1, q_2, q_3]^T$ with $q_0$ scalar

**Input:** $u = [F_b, \tau_b]^T \in \mathbb{R}^{6}$

- $F_b \in \mathbb{R}^3$: body-frame force (lbf)
- $\tau_b \in \mathbb{R}^3$: body-frame torque (ft-lbf)
---


## Control Objective

Given desired trajectory $p_d(t)$, find $u(x)$ such that $p \to p_d$.

---

## Controller Structure

The controller has two components:

$$
u = u_{\text{ff}} + u_{\text{fb}}
$$

- $u_{\text{ff}}$: **Feedforward** — computed from desired trajectory, assumes perfect tracking
- $u_{\text{fb}}$: **Feedback** — corrects for errors between actual and desired state

---

## Part 1: Feedforward Derivation (Mellinger & Kumar, 2011)

**Goal:** Given desired trajectory $p_d(t)$, derive what $F_b$ must be to achieve it.

### Step 1: Differentiate (3)

$$
\ddot{p} = \frac{d}{dt}(R_{b2i} v_b) = \dot{R}_{b2i} v_b + R_{b2i} \dot{v}_b
$$

### Step 2: Substitute (1) into Step 1

$$
\ddot{p} = \dot{R}_{b2i} v_b + R_{b2i} \left( R_{i2b} g_I - \omega \times v_b + \frac{F_b}{m} \right)
$$

### Step 3: Use rotation matrix derivative identity

For body-frame angular velocity $\omega$:

$$
\dot{R}_{b2i} = R_{b2i} \, [\omega]_\times
$$

where $[\omega]_\times$ is the skew-symmetric matrix of $\omega$.

Therefore:

$$
\dot{R}_{b2i} v_b = R_{b2i} (\omega \times v_b)
$$

### Step 4: Simplify

$$
\ddot{p} = R_{b2i} (\omega \times v_b) + R_{b2i} R_{i2b} g_I - R_{b2i} (\omega \times v_b) + R_{b2i} \frac{F_b}{m}
$$

The $\omega \times v_b$ terms cancel:

$$
\ddot{p} = g_I + R_{b2i} \frac{F_b}{m}
$$

### Step 5: Solve for $F_b$

$$
R_{b2i} \frac{F_b}{m} = \ddot{p} - g_I
$$

$$
\frac{F_b}{m} = R_{i2b} (\ddot{p} - g_I)
$$

$$
\boxed{F_b = m \cdot R_{i2b} \cdot (\ddot{p} - g_I)} \tag{5}
$$

Given any twice continuously differentiable inertial position $p(t)$, equation (5) gives the force $F_b$ that produces it.

### Step 6: Feedforward

Replacing $p(t)$ with any desired trajectory $p_d(t)$, the force input is:

$$
F_{b,\text{ff}} = m \cdot R_{i2b}(q) \cdot (\ddot{p}_d - g_I) \tag{6}
$$

where $\ddot{p}_d$ is obtained by differentiating $p_d(t)$ twice.

---

## Part 2: Feedback Derivation (Lee et al., 2010)

**Goal:** Correct for errors when actual position $p$ differs from desired $p_d$.

### Why feedback is needed

The feedforward assumes we are exactly on the trajectory. In practice:
- Initial conditions may not match
- Disturbances push us off trajectory
- Model inaccuracies exist

### Error dynamics

Define position error: $e = p - p_d$

We want stable error dynamics:

$$
\ddot{e} + K_{d,\text{position}} \dot{e} + K_{p,\text{position}} e = 0
$$

This is a damped second-order system that drives $e \to 0$.

### Feedback acceleration

Rearranging:

$$
\ddot{p} = \ddot{p}_d - K_{p,\text{position}} (p - p_d) - K_{d,\text{position}} (\dot{p} - \dot{p}_d)
$$

The feedback adds correction terms to the desired acceleration.

### Combined commanded acceleration

$$
\ddot{p}_{\text{cmd}} = \underbrace{\ddot{p}_d}_{\text{feedforward}} - \underbrace{K_{p,\text{position}} (p - p_d) - K_{d,\text{position}} (\dot{p} - \dot{p}_d)}_{\text{feedback}}
$$

where $\dot{p} = R_{b2i} v_b$ (current velocity in inertial frame).

---

## Combined Controller

### Body Force

$$
F_b = m \cdot R_{i2b}(q) \cdot (\ddot{p}_{\text{cmd}} - g_I)
$$

### Torque (attitude stabilization)

$$
\tau_b = -K_{p,\text{attitude}} \cdot e_q - K_{d,\text{attitude}} \cdot \omega
$$

where $e_q = [q_1, q_2, q_3]^T$ (quaternion vector part, stabilizes to level: $q_{\text{des}} = [1,0,0,0]$).

---



## Jacobian via ForwardDiff

The derivation above defines a control law mapping state to input:

$$
\texttt{cntrl\_DiffFlat}: x \rightarrow u
$$

For certain applications (e.g., ℓ₁-DRAC), we need the Jacobian of this control law:

$$
\nabla_x \, \texttt{cntrl\_DiffFlat} = \frac{\partial u}{\partial x} \in \mathbb{R}^{6 \times 13}
$$

ForwardDiff computes **exact** derivatives via automatic differentiation (not finite differences):

```julia
using ForwardDiff

function cntrl_DiffFlat_jacobian(t, x, traj, dp)
    f(x) = cntrl_DiffFlat(t, x, traj, dp)
    return ForwardDiff.jacobian(f, x)
end
```

The closure `f(x) = cntrl_DiffFlat(t, x, traj, dp)` captures `t`, `traj`, `dp` from the outer scope, exposing only `x` as the argument. `ForwardDiff.jacobian(f, x)` differentiates `f` with respect to its input argument, evaluated at `x`.

**Requirements for AD compatibility:**
- No branching on state values (`if x[1] > 0`)
- Regularize divisions: `a / (norm(b) + 1e-10)`
- Use Float64 for Jacobian computation

---

## References

1. Mellinger & Kumar (2011) - "Minimum Snap Trajectory Generation and Control for Quadrotors", ICRA
2. Lee et al. (2010) - "Geometric Tracking Control of a Quadrotor UAV on SE(3)", CDC
