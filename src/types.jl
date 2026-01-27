# Waypoint struct for trajectory tracking
# Contains position (NED frame) and arrival time
struct Waypoint
    position::SVector{3, Float32}   # position in NED frame (ft)
    time::Float32              # cumulative arrival time (s)
end
# Constructor for desired waypoints
function waypoints(wp_list)     
    # wp_list = [(pos1, time1), (pos2, time2), ...]                                      
    return ntuple(i -> Waypoint(SVector{3,Float32}(wp_list[i][1]), Float32(wp_list[i][2])), length(wp_list))
end
# Get current target waypoint position based on time
function get_current_waypoint(t, waypoints)
    for wp in waypoints
        t < wp.time && return wp.position
    end
    return waypoints[end].position
end

