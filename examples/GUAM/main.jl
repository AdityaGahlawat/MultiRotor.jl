# Main script to simulate the GUAM multirotor example

using Revise
import MultiRotor
import L1DRAC
using LinearAlgebra
using StaticArrays
using CUDA
using Distributions
using JLD2



function setup_system(; Ntraj=10)
    # Simulation Parameters
    tspan = (0.0, 50.0)
    Δₜ = 5e-3
    Δ_saveat = 1e2 * Δₜ
    simulation_parameters = L1DRAC.sim_params(tspan, Δₜ, Ntraj, Δ_saveat)

    # System Dimensions
    n, m, d = 13, 6, 4
    system_dimensions = L1DRAC.sys_dims(n, m, d)

    # Vehicle Parameters
    mass = 181.79f0    # slugs
    gravity = 32.17f0  # ft/s²
    J = @SMatrix Float32[13052 58 -2969; 58 16661 -986; -2969 -986 24735]  # slug-ft²
    J_inv = inv(J)

    # Waypoints (position in NED frame, arrival time)
    waypoints = MultiRotor.waypoints([
        ([0f0, 0f0, -100f0], 12.5f0),
        ([-500f0, 200f0, -100f0], 25.0f0),
        ([-500f0, -200f0, -150f0], 37.5f0),
        ([0f0, 0f0, -100f0], 50.0f0),
    ])

    dp = (; mass, gravity, J, J_inv, waypoints = waypoints)

    ## BASELINE CONTROLLER =====================================
    # PD controller gains
    gain = (Kp_position = 5f0, Kd_position = 20f0, Kp_attitude = 10000f0, Kd_attitude = 5000f0)

    # Baseline controller using PD tracking
    baseline_input(t, x, dp) = MultiRotor.cntrl_PD(t, x, gain, dp)
    ## ==============================================================

    # Nominal Closed-loop drift dynamics
    function f(t, x, dp)
        u = baseline_input(t, x, dp)
        return MultiRotor.f_quad(t, x, dp) + MultiRotor.g(t, x, dp) * u
    end

    # Nominal Diffusion
    p_m(t, x, dp) = @SMatrix zeros(Float32, 6, 4)   # m × d (matched)
    p_um(t, x, dp) = @SMatrix zeros(Float32, 7, 4)  # (n-m) × d (unmatched)
    p(t, x, dp) = vcat(p_m(t, x, dp), p_um(t, x, dp))  # n × d

    # Drift Uncertainty
    Λμ_m(t, x, dp) = @SVector zeros(Float32, 6)   # m (matched)
    Λμ_um(t, x, dp) = @SVector zeros(Float32, 7)  # (n-m) (unmatched)
    Λ_μ(t, x, dp) = vcat(Λμ_m(t, x, dp), Λμ_um(t, x, dp))  # n

    # Diffusion Uncertainty
    Λσ_m(t, x, dp) = @SMatrix zeros(Float32, 6, 4)   # m × d (matched)
    Λσ_um(t, x, dp) = @SMatrix zeros(Float32, 7, 4)  # (n-m) × d (unmatched)
    Λ_σ(t, x, dp) = vcat(Λσ_m(t, x, dp), Λσ_um(t, x, dp))  # n × d

    # L1DRAC setup
    nominal_components = L1DRAC.nominal_vector_fields(f, MultiRotor.g, MultiRotor.g_perp, p, dp)
    uncertain_components = L1DRAC.uncertain_vector_fields(Λ_μ, Λ_σ, dp)

    # Initial distributions
    x₀_mean = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    nominal_ξ₀ = MvNormal(x₀_mean, 1e-4 * I(13))
    true_ξ₀ = MvNormal(x₀_mean, 1e-4 * I(13))
    initial_distributions = L1DRAC.init_dist(nominal_ξ₀, true_ξ₀)

    nominal_system = L1DRAC.nom_sys(system_dimensions, nominal_components, initial_distributions)
    true_system = L1DRAC.true_sys(system_dimensions, nominal_components, uncertain_components, initial_distributions)

    return (
        simulation_parameters = simulation_parameters,
        nominal_system = nominal_system,
        true_system = true_system,
        waypoints = waypoints,
    )
end

function main(; Ntraj=10, max_GPUs=1, systems=[:nominal_sys])
    @info "Warmup run for JIT compilation"
    warmup_setup = setup_system(; Ntraj=10)
    L1DRAC.run_simulations(warmup_setup; max_GPUs=max_GPUs, systems=systems)

    @info "Complete run for Ntraj=$Ntraj"
    setup = setup_system(; Ntraj=Ntraj)
    solutions = L1DRAC.run_simulations(setup; max_GPUs=max_GPUs, systems=systems)
    return solutions, setup
end

function visualize(solutions, setup;
    linewidth=0.3, linealpha=0.1, save_dir=@__DIR__)
    MultiRotor.plot_trajectory_3D(solutions.nominal_sol, setup.waypoints;
                                   linewidth = linewidth, linealpha = linealpha, save_path = joinpath(save_dir, "trajectory_3D.png"))
    MultiRotor.plot_states(solutions.nominal_sol;
                           linewidth = linewidth, linealpha = linealpha, save_path = joinpath(save_dir, "states.png"))
end


## Commands to run simulations
# solutions, setup = main(; Ntraj= Int(1e3), max_GPUs= 1 , systems=[:nominal_sys])
# visualize(solutions, setup; linewidth=0.3, linealpha=0.1, save_dir=@__DIR__)

# ================== SAVE/LOAD =====================
function save_solutions(solutions, setup; save_dir=@__DIR__, filename="saved_solutions.jld2")
    filepath = joinpath(save_dir, filename)
    @save filepath solutions setup
    @info "Saved solutions to $filepath"
end

function load_solutions(; save_dir=@__DIR__, filename="saved_solutions.jld2")
    filepath = joinpath(save_dir, filename)
    @load filepath solutions setup
    @info "Loaded solutions from $filepath"
    return solutions, setup
end

# ================== DATA LOGGING =====================
function get_training_data(solutions, setup)
    gain = (Kp_position=5f0, Kd_position=20f0, Kp_attitude=10000f0, Kd_attitude=5000f0)
    dp = (; mass=181.79f0, gravity=32.17f0, waypoints=setup.waypoints)
    return MultiRotor.extract_training_data(solutions, gain, dp)
end

# ================== TESTS =====================
# Test PD controller Jacobian computation
function test_jacobian()
    mass = 181.79f0
    gravity = 32.17f0
    J = @SMatrix Float32[13052 58 -2969; 58 16661 -986; -2969 -986 24735]
    J_inv = inv(J)

    waypoints = MultiRotor.waypoints([
        ([0f0, 0f0, -100f0], 12.5f0),
        ([-500f0, 200f0, -100f0], 25.0f0),
    ])

    dp = (; mass, gravity, J, J_inv, waypoints)
    gain = (Kp_position = 5f0, Kd_position = 20f0, Kp_attitude = 10000f0, Kd_attitude = 5000f0)

    # Test state (hover at origin)
    x = @SVector Float32[0,0,0, 0,0,0, 0,0,0, 1,0,0,0]

    # Compute Jacobian
    J_u = MultiRotor.cntrl_PD_jacobian(5.0, x, gain, dp)

    @info "Jacobian size: $(size(J_u))"
    @info "Jacobian computed successfully"

    return J_u
end

# Test training data extraction
function test_training_data()
    solutions, setup = main(; Ntraj=10, max_GPUs=1, systems=[:nominal_sys])
    training_data = get_training_data(solutions, setup)

    @info "Number of trajectories: $(length(training_data))"
    @info "Timesteps per trajectory: $(length(training_data[1].t))"
    @info "State size: $(length(training_data[1].x[1]))"
    @info "Control size: $(length(training_data[1].u[1]))"
    @info "Jacobian size: $(size(training_data[1].J[1]))"

    return training_data
end


