# Visualization for multirotor trajectories

# Activate specified Makie backend
function activate_backend(backend::Symbol)
    if backend == :cairo
        CairoMakie.activate!()
    elseif backend == :gl
        GLMakie.activate!()
    else
        error("Unknown backend: $backend. Use :cairo or :gl")
    end
end

# Plot 3D trajectory from ensemble solution
function plot_trajectory_3D(ensemble, waypoints;
                            save_path = nothing,
                            title = L"Multirotor Trajectory",
                            backend = :cairo,
                            linewidth = 0.5,
                            linealpha = 0.3)
    activate_backend(backend)
    fig = Figure(size = (800, 600))
    ax = Axis3(fig[1, 1];
               title = title,
               xlabel = L"North (ft)",
               ylabel = L"East (ft)",
               zlabel = L"Altitude (ft)",
               azimuth = 1.275π,
               elevation = π/8)

    # Plot trajectories
    # L1DRAC.run_simulations returns: nominal_sol, true_sol, L1_sol
    # Each is a Vector{EnsembleSolution} of size:
    #   - 1 when max_GPUs = 0 (CPU) or max_GPUs = 1 (single GPU)
    #   - N when max_GPUs = N > 1 (multi-GPU)
    for ensemble_sol in ensemble
        for traj in ensemble_sol
            north = [u[7] for u in traj.u]
            east = [u[8] for u in traj.u]
            alt = [-u[9] for u in traj.u]  # NED to altitude (flip sign)
            lines!(ax, north, east, alt; color = ColorSchemes.tableau_10[1], linewidth = linewidth, alpha = linealpha)
        end
    end

    # Plot waypoints
    wp_north = [wp.position[1] for wp in waypoints]
    wp_east = [wp.position[2] for wp in waypoints]
    wp_alt = [-wp.position[3] for wp in waypoints]  # NED to altitude
    scatter!(ax, wp_north, wp_east, wp_alt; color = ColorSchemes.tableau_10[2], markersize = 15)

    if save_path !== nothing
        save(save_path, fig)
        @info "Saved 3D trajectory to $save_path"
    end

    return fig
end

# Plot state time series (9 subplots: position, velocity, angular velocity)
function plot_states(ensemble;
                     save_path = nothing,
                     title = L"State Time Series",
                     backend = :cairo,
                     linewidth = 0.5,
                     linealpha = 0.5)
    activate_backend(backend)
    fig = Figure(size = (800, 1200))

    state_labels = [L"p_x \, \mathrm{(ft)}", L"p_y \, \mathrm{(ft)}", L"p_z \, \mathrm{(ft)}",
                    L"v_x \, \mathrm{(ft/s)}", L"v_y \, \mathrm{(ft/s)}", L"v_z \, \mathrm{(ft/s)}",
                    L"\omega_x \, \mathrm{(rad/s)}", L"\omega_y \, \mathrm{(rad/s)}", L"\omega_z \, \mathrm{(rad/s)}"]
    state_indices = [7, 8, 9, 1, 2, 3, 4, 5, 6]  # position, velocity, angular velocity
    tableau = ColorSchemes.tableau_10
    state_colors = [tableau[1], tableau[1], tableau[1], tableau[2], tableau[2], tableau[2], tableau[3], tableau[3], tableau[3]]

    axes = [Axis(fig[i, 1]; ylabel = state_labels[i]) for i in 1:9]

    # Make plots span full width
    colsize!(fig.layout, 1, Relative(1))

    # L1DRAC.run_simulations returns: nominal_sol, true_sol, L1_sol
    # Each is a Vector{EnsembleSolution} of size:
    #   - 1 when max_GPUs = 0 (CPU) or max_GPUs = 1 (single GPU)
    #   - N when max_GPUs = N > 1 (multi-GPU)
    for ensemble_sol in ensemble
        for traj in ensemble_sol
            t = traj.t
            for (i, idx) in enumerate(state_indices)
                y = [u[idx] for u in traj.u]
                # Flip sign for altitude (index 9) to show positive up
                if idx == 9
                    y = -y
                end
                lines!(axes[i], t, y; color = state_colors[i], linewidth = linewidth, alpha = linealpha)
            end
        end
    end

    axes[9].xlabel = L"Time (s)"
    Label(fig[0, 1], title; fontsize = 16)

    if save_path !== nothing
        save(save_path, fig)
        @info "Saved state plots to $save_path"
    end

    return fig
end
