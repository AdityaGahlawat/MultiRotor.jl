# Main module

module MultiRotor

using LinearAlgebra
using StaticArrays
using LaTeXStrings
using CairoMakie
# GLMakie will not load on a headless remote
try
    using GLMakie 
catch 
    @warn """
    GLMakie could not be loaded. 
    Likely culprit: headless remote (investigate error above). 
    Continuing without GLMakie.
    Only plotting affected.
    """ 
    @warn "---> SIMULATION CAN PROCEED"
end 
using ColorSchemes
using ForwardDiff


## Include source files
include("types.jl")
include("basic_dynamics.jl")
include("controllers.jl")
include("datalogging.jl")
include("Viz.jl")

## Exports
# Waypoints
export Waypoint, waypoints, get_current_waypoint


# Dynamics functions
export f_quad, g, g_perp, quat_to_rotmat

# Controllers
export cntrl_PD, cntrl_PD_jacobian

# Visualization
export plot_trajectory_3D, plot_states, plot_trajectory_planned

# Data logging
export extract_training_data

end
