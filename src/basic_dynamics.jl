# Basic multirotor dynamics
# PURE DRIFT and CONTROL INPUT OPERATOR (separated)
#
# Control-affine form: ẋ = f_quad(t,x,dp) + g(t,x,dp)·u
#
# Functions:
#   f_quad(t, x, dp)  - pure drift dynamics (gravity, Coriolis, kinematics)
#   g(t, x, dp)       - input operator (13×6)
#   g_perp(t, x, dp)  - null space of g' (13×7)
#   quat_to_rotmat(q) - quaternion to rotation matrix (inertial-to-body)
#
# User defines separately: controller u(t,x,dp), diffusion p(t,x,dp), uncertainties
#
# State vector x ∈ ℝ¹³:
#   x[1:3]   = v_b    (body velocity, ft/s)
#   x[4:6]   = ω      (angular velocity, rad/s)
#   x[7:9]   = p      (inertial position NED, ft)
#   x[10:13] = q      (quaternion, scalar-first: q0,q1,q2,q3)
#
# Input vector u ∈ ℝ⁶:
#   u[1:3] = F_b  (body force, lbf)
#   u[4:6] = τ_b  (body torque, ft-lbf)
#
# Required fields in dp:
#   dp.mass    - vehicle mass (slugs)
#   dp.gravity - gravitational acceleration (ft/s²)
#   dp.J       - inertia matrix 3×3 (slug-ft²)
#   dp.J_inv   - inverse of J


# Quaternion to rotation matrix (inertial-to-body)
# Convention: q = [q0, q1, q2, q3] with q0 scalar
function quat_to_rotmat(q)
    q0, q1, q2, q3 = q
    @SMatrix [
        1-2*(q2^2+q3^2)   2*(q1*q2+q0*q3)   2*(q1*q3-q0*q2);
        2*(q1*q2-q0*q3)   1-2*(q1^2+q3^2)   2*(q2*q3+q0*q1);
        2*(q1*q3+q0*q2)   2*(q2*q3-q0*q1)   1-2*(q1^2+q2^2)
    ]
end

# Pure drift dynamics (NO control terms)
# Contains: gravity, Coriolis/gyroscopic, position kinematics, quaternion kinematics
# Usage: full dynamics = f_quad(t,x,dp) + g(t,x,dp) * u 
function f_quad(t, x, dp)
    v_b = SVector{3}(x[1], x[2], x[3])
    ω = SVector{3}(x[4], x[5], x[6])
    q = SVector{4}(x[10], x[11], x[12], x[13])

    R_i2b = quat_to_rotmat(q)
    R_b2i = R_i2b'

    # Gravity in body frame
    g_inertial = @SVector [0f0, 0f0, dp.gravity]
    g_body = R_i2b * g_inertial

    # Linear acceleration (gravity + Coriolis, NO F_b/m term)
    v_dot = g_body - cross(ω, v_b)

    # Angular acceleration (gyroscopic only, NO τ_b term)
    Jω = dp.J * ω
    ω_dot = -dp.J_inv * cross(ω, Jω)

    # Position kinematics
    p_dot = R_b2i * v_b

    # Quaternion kinematics with norm stabilization
    q0, q1, q2, q3 = q
    q0_dot = 0.5f0 * (-q1*ω[1] - q2*ω[2] - q3*ω[3])
    q1_dot = 0.5f0 * ( q0*ω[1] - q3*ω[2] + q2*ω[3])
    q2_dot = 0.5f0 * ( q3*ω[1] + q0*ω[2] - q1*ω[3])
    q3_dot = 0.5f0 * (-q2*ω[1] + q1*ω[2] + q0*ω[3])
    norm_err = (q0^2 + q1^2 + q2^2 + q3^2) - 1f0
    k_stab = 0.1f0
    q_dot = SVector(q0_dot, q1_dot, q2_dot, q3_dot) - k_stab * norm_err * q

    return vcat(v_dot, ω_dot, p_dot, q_dot)
end

# Input operator g (13×6)
# Maps control input u = [F_b; τ_b] to state derivative contribution
# g·u adds: [F_b/m; J⁻¹τ_b; 0; 0] to the state derivative
function g(t, x, dp)
    I3_m = SMatrix{3,3,Float32}(1f0/dp.mass * I)
    Z3 = @SMatrix zeros(Float32, 3, 3)
    Z4 = @SMatrix zeros(Float32, 4, 3)
    return vcat(
        hcat(I3_m, Z3),      # v̇ contribution: F_b/m
        hcat(Z3, dp.J_inv),  # ω̇ contribution: J⁻¹τ_b
        hcat(Z3, Z3),        # ṗ: no direct control effect
        hcat(Z4, Z4)         # q̇: no direct control effect
    )
end

# Null space of g' (13×7)
# States not directly affected by control input: position (3) + quaternion (4)
function g_perp(t, x, dp)
    Z6 = @SMatrix zeros(Float32, 6, 7)
    I7 = SMatrix{7,7,Float32}(1f0*I)
    return vcat(Z6, I7)
end
