# Data logging for neural network training
#
# Extract training data from L1DRAC solutions
#
# Returns: Vector of length Ntraj
#   training_data[i] for i ∈ {1, ..., Ntraj}:
#     .t = time points
#     .x = states over time
#     .u = cntrl_PD(x) over time
#     .J = cntrl_PD_jacobian(x) over time

function extract_training_data(solutions, gain, dp)
    training_data = []

    for ensemble_sol in solutions.nominal_sol
        for traj in ensemble_sol
            T = length(traj.t)

            x_data = [SVector{13}(traj.u[k]) for k in 1:T]
            u_data = [cntrl_PD(traj.t[k], x_data[k], gain, dp) for k in 1:T]
            J_data = [cntrl_PD_jacobian(traj.t[k], x_data[k], gain, dp) for k in 1:T]

            push!(training_data, (t=traj.t, x=x_data, u=u_data, J=J_data))
        end
    end

    return training_data
end
