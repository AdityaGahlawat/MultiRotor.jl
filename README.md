# Codebase for Multirotor sim and viz
- Generalized - can be used as a stochatsic system or a deterministic system

## Project Structure

```
MultiRotor/
├── examples/
│   └── GUAM/
├── src/
│   ├── MultiRotor.jl      # Main module
│   ├── types.jl           # Type definitions (Waypoint)
│   ├── basic_dynamics.jl  # Quadrotor dynamics (f_quad, g, g_perp)
│   ├── controllers.jl     # PD controller + Jacobian
│   ├── datalogging.jl     # Training data extraction
│   └── Viz.jl             # Plotting utilities
├── Manifest.toml
├── Project.toml
└── README.md
```

## Setup
Recommended to launch with multi thread support
```bash
julia -t auto
```
Add codebase to working environment
```julia
] add https://github.com/AdityaGahlawat/MultiRotor.jl
```

## Ready-to-use Systems 
- ### GUAM 
    - Run `examples/GUAM/main.jl`
    - System description in `examples/GUAM/dynamics.md`

## Example
**Source:** [`examples/GUAM/main.jl`](examples/GUAM/main.jl)

### Setup
```julia
import MultiRotor, L1DRAC
using Revise
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
    J = @SMatrix Float32[13052 58 -2969; 58 16661 -986; -2969 -986 24735]
    J_inv = inv(J)

    # Waypoints (position in NED frame, arrival time)
    waypoints = MultiRotor.waypoints([
        ([0f0, 0f0, -100f0], 12.5f0),
        ([-500f0, 200f0, -100f0], 25.0f0),
        ([-500f0, -200f0, -150f0], 37.5f0),
        ([0f0, 0f0, -100f0], 50.0f0),
    ])

    dp = (; mass, gravity, J, J_inv, waypoints = waypoints)

    # Baseline PD controller
    gain = (Kp_position = 5f0, Kd_position = 20f0, Kp_attitude = 10000f0, Kd_attitude = 5000f0)
    baseline_input(t, x, dp) = MultiRotor.cntrl_PD(t, x, gain, dp)

    # Nominal closed-loop dynamics
    function f(t, x, dp)
        u = baseline_input(t, x, dp)
        return MultiRotor.f_quad(t, x, dp) + MultiRotor.g(t, x, dp) * u
    end

    # ... (diffusion and uncertainty terms)

    return (
        simulation_parameters = simulation_parameters,
        nominal_system = nominal_system,
        true_system = true_system,
        waypoints = waypoints,
    )
end
```

### Main
```julia
function main(; Ntraj=10, max_GPUs=1, systems=[:nominal_sys])
    @info "Warmup run for JIT compilation"
    warmup_setup = setup_system(; Ntraj=10)
    L1DRAC.run_simulations(warmup_setup; max_GPUs=max_GPUs, systems=systems)

    @info "Complete run for Ntraj=$Ntraj"
    setup = setup_system(; Ntraj=Ntraj)
    solutions = L1DRAC.run_simulations(setup; max_GPUs=max_GPUs, systems=systems)
    return solutions, setup
end
```

### Data Logging for TaSIL (first order) Training

Run the nominal system to generate trajectory data:
```julia
solutions, setup = main(; Ntraj=Int(1e3), max_GPUs=1, systems=[:nominal_sys]);
```

Extract training data. Returns a vector of length `Ntraj` where each element contains:
- `.t` - time points
- `.x` - states over time
- `.u` - `cntrl_PD(x)` over time
- `.J` - `cntrl_PD_jacobian(x)` over time

```julia
training_data = get_training_data(solutions, setup);

# Access trajectory i ∈ {1, ..., Ntraj}, timestep k:
training_data[i].x[k]  # state
training_data[i].u[k]  # control input
training_data[i].J[k]  # Jacobian (6×13)
```

### Save/Load (if needed)
```julia
save_solutions(solutions, setup)
solutions, setup = load_solutions();
```

### Visualize (if needed)
```julia
visualize(solutions, setup)
```

## TO DO
- Differential flatness controller (if needed)
