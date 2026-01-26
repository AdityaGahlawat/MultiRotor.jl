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


## Include source files
include("types.jl")
include("basic_dynamics.jl")
include("controllers.jl")
include("Viz.jl")

## Exports
# Constructors
export waypoints
export get_current_waypoint

# Dynamics functions
export f_quad, g, g_perp, quat_to_rotmat

# Controllers
export cntrl_PD

# Visualization
export plot_trajectory_3D, plot_states

end
