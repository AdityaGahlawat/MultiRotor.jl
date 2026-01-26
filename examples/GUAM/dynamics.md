## Deterministic model
- #### standard multirotor represented via quaternions
- #### model parameters from the simplified GUAM Dynamics

--------------------
### State Vector (13D)

```math
\begin{align*}
x =& \begin{bmatrix} v_b^\top & \omega^\top & p^\top & q^\top \end{bmatrix}^\top 
    \\
=& \begin{bmatrix} v_x & v_y & v_z & \omega_x & \omega_y & \omega_z & p_x & p_y & p_z & q_0 & q_1 & q_2 & q_3 \end{bmatrix}^\top 
\in \mathbb{R}^{13}.
\end{align*}
```

where:

- $v_b \in \mathbb{R}^3$: body-frame velocity (ft/s)
- $\omega \in \mathbb{R}^3$: body-frame angular velocity (rad/s)
- $p \in \mathbb{R}^3$: inertial position in NED frame (ft)
- $q \in \mathbb{R}^4$: quaternion (inertial-to-body convention), $q = [q_0, q_1, q_2, q_3]^T$ with $q_0$ scalar

### Input Vector (6D)

```math
\begin{align*}
u   =& \begin{bmatrix} F_b^\top & \tau_b^\top \end{bmatrix}^\top
    \\
    =& \begin{bmatrix} F_x & F_y & F_z & \tau_x & \tau_y & \tau_z \end{bmatrix}^\top
    \in \mathbb{R}^6,  
\end{align*}
```
where

- $F_b \in \mathbb{R}^3$: body-frame force (lbf)
- $\tau_b \in \mathbb{R}^3$: body-frame moment (ft-lbf)

### Control-Affine Form

$$
\dot{x} = f(x) + g \cdot u
$$

#### Drift Term $f(x)$

$$
f(x) = \begin{bmatrix} R_{i2b} \, g_I - \omega \times v_b \\ -J^{-1}(\omega \times J\omega) \\ R_{b2i} \, v_b \\ \dot{q}(\omega) \end{bmatrix}
$$

where:
- $g_I = [0, 0, 32.17]^T$ ft/s² (gravity in NED, down is positive)
- $R_{i2b}$: rotation matrix from inertial to body (from quaternion $q$)
- $R_{b2i} = R_{i2b}^T$: rotation matrix from body to inertial
- $J$: inertia matrix
- $\dot{q}(\omega)$: quaternion kinematics (see below)

#### Input Matrix $g$ (constant)

$$
g = \begin{bmatrix} \frac{1}{m} I_3 & 0_{3\times3} \\ 0_{3\times3} & J^{-1} \\ 0_{3\times3} & 0_{3\times3} \\ 0_{4\times3} & 0_{4\times3} \end{bmatrix} \in \mathbb{R}^{13 \times 6}
$$

### Quaternion Kinematics

$$
\dot{q} = \frac{1}{2} \Omega(\omega) \, q - k_s (||q||^2 - 1) \, q
$$

where $k_s = 0.1$ is a stabilization gain and:

$$
\Omega(\omega) = \begin{bmatrix} 0 & -\omega_x & -\omega_y & -\omega_z \\ \omega_x & 0 & \omega_z & -\omega_y \\ \omega_y & -\omega_z & 0 & \omega_x \\ \omega_z & \omega_y & -\omega_x & 0 \end{bmatrix}
$$

### Quaternion to direction cosine matrix (DCM) 

$$
R_{i2b} = \begin{bmatrix} 1 - 2(q_2^2 + q_3^2) & 2(q_1 q_2 + q_0 q_3) & 2(q_1 q_3 - q_0 q_2) \\ 2(q_1 q_2 - q_0 q_3) & 1 - 2(q_1^2 + q_3^2) & 2(q_2 q_3 + q_0 q_1) \\ 2(q_1 q_3 + q_0 q_2) & 2(q_2 q_3 - q_0 q_1) & 1 - 2(q_1^2 + q_2^2) \end{bmatrix}
$$

### Vehicle Parameters

| Parameter | Value | Units |
|-----------|-------|-------|
| Mass $m$ | 181.79 | slugs |
| Gravity $g$ | 32.17 | ft/s² |

$$
J = \begin{bmatrix} 13052 & 58 & -2969 \\ 58 & 16661 & -986 \\ -2969 & -986 & 24735 \end{bmatrix} \text{ slug-ft}^2
$$

### Reference Frames

- **Inertial (NED)**: North-East-Down, fixed to Earth
- **Body**: Fixed to vehicle, x-forward, y-right, z-down