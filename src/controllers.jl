# Controllers for multirotor dynamics
#
# Available controllers:
#   cntrl_PD(t, x, dp; gains) - PD tracking controller for waypoint following
#
# All controllers follow signature: cntrl(t, x, dp; kwargs...) -> SVector{6, Float32}
# where output is [F_b; τ_b] (body force and torque)

# Default PD gains
const DEFAULT_PD_GAINS = (
    Kp_position = 5f0,
    Kd_position = 20f0,
    Kp_attitude = 10000f0,
    Kd_attitude = 5000f0,
)

# PD tracking controller for waypoint following
#
# Required fields in dp:
#   mass, gravity - vehicle parameters
#   waypoints     - tuple of Waypoint structs (use MultiRotor.waypoints() constructor)
#
function cntrl_PD(t, x, gain, dp)
    # gain = (Kp_position, Kd_position, Kp_attitude, Kd_attitude)

    # Extract state
    v_b = SVector{3,Float32}(x[1], x[2], x[3])
    ω = SVector{3,Float32}(x[4], x[5], x[6])
    pos = SVector{3,Float32}(x[7], x[8], x[9])
    q = SVector{4,Float32}(x[10], x[11], x[12], x[13])

    # Rotation matrices
    R_i2b = quat_to_rotmat(q)
    R_b2i = R_i2b'

    # Get target waypoint
    p_target = get_current_waypoint(t, dp.waypoints)

    # Position PD in inertial frame
    v_inertial = R_b2i * v_b
    F_inertial = gain.Kp_position * (p_target - pos) - gain.Kd_position * v_inertial
    F_b = R_i2b * F_inertial + @SVector [0f0, 0f0, -dp.mass * dp.gravity]

    # Attitude PD (stabilize to level: q_desired = [1,0,0,0])
    q_err = @SVector Float32[q[2], q[3], q[4]]  # vector part of quaternion error
    τ_b = -gain.Kp_attitude * q_err - gain.Kd_attitude * ω

    return vcat(F_b, τ_b)
end

# TODO: cntrl_DiffFlatness(t, x, dp, trajectory)
# Differential flatness controller for trajectory tracking
# Flat outputs: position (x,y,z) + yaw
# References: Mellinger & Kumar ICRA 2011, Lee et al. CDC 2010
