# plotting.py (FINAL VERSION WITH WAYPOINT-BASED ANNOTATIONS)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def find_closest_discrete_point_index(point, discretized_points):
    """Helper function to find the index of the discrete point closest to a waypoint."""
    return np.argmin(np.linalg.norm(discretized_points - point, axis=1))

def plot_scenario(gu_locations, trajectory, d_h, detailed_results):
    """
    Visualizes the scenario and annotates each major waypoint-to-waypoint segment
    with detailed performance info.
    """
    fig, ax = plt.subplots(figsize=(16, 12)) # Increased figure size for better readability
    ax.set_title("UAV Data Collection with Detailed Waypoint Segment Analysis")
    ax.set_xlabel("X-axis (meters)")
    ax.set_ylabel("Y-axis (meters)")
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    # --- Plot static elements: GUs and their ranges ---
    ax.scatter(gu_locations[:, 0], gu_locations[:, 1], c='blue', marker='o', label='Ground Users (GUs)')
    for i, loc in enumerate(gu_locations):
        ax.text(loc[0] + 30, loc[1] + 30, f'GU-{i}', fontsize=9, color='black')
    for gu in gu_locations:
        circle = patches.Circle(gu, d_h, color='blue', alpha=0.1, zorder=1)
        ax.add_patch(circle)

    # --- Plot main trajectory line (waypoints) and Start/End markers ---
    # We plot the waypoints themselves, not the discretized path here.
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-o', markersize=5, alpha=0.8, label='UAV Trajectory (Waypoints)')
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'gs', markersize=10, label='Start', zorder=5)
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'kx', markersize=10, mew=3, label='End', zorder=5)
    
    # --- Extract detailed results for analysis ---
    delta = detailed_results.get("delta_per_segment", np.array([]))
    schedule = detailed_results.get("schedule_matrix", np.array([]))
    discretized_points = detailed_results.get("discretized_points", np.array([]))

    # --- NEW LOGIC: Annotate based on MAJOR (waypoint-to-waypoint) segments ---
    if len(delta) > 0 and len(trajectory) > 1:
        
        # First, map each major waypoint to its corresponding index in the discretized path
        waypoint_indices_in_discrete_path = [find_closest_discrete_point_index(wp, discretized_points) for wp in trajectory]
        
        fly_counter = 1
        gu_counter = 1

        # Iterate through each major segment (from one waypoint to the next)
        for i in range(len(trajectory) - 1):
            p_start_major = trajectory[i]
            p_end_major = trajectory[i+1]
            
            # --- Identify the micro-segments that constitute this major segment ---
            start_idx = waypoint_indices_in_discrete_path[i]
            end_idx = waypoint_indices_in_discrete_path[i+1]
            
            if start_idx >= end_idx: # This can happen if a segment is too short to be discretized
                continue

            # --- Aggregate stats from the micro-segments ---
            major_segment_len = np.linalg.norm(p_end_major - p_start_major)
            total_flight_time = np.sum(delta[start_idx:end_idx])
            
            # Calculate total collection time on this major segment
            total_collection_time = 0
            # Sum over all GUs (axis 0) and all micro-segments in this major segment (axis 1)
            scheduled_times = schedule[:, start_idx:end_idx] * delta[start_idx:end_idx]
            total_collection_time = np.sum(scheduled_times)

            avg_velocity = major_segment_len / total_flight_time if total_flight_time > 1e-6 else 0
            
            # --- Determine segment name and type ---
            # Even-indexed segments (0, 2, 4...) are Fly segments (e.g., Start->S1, E1->S2)
            # Odd-indexed segments (1, 3, 5...) are GU segments (e.g., S1->E1, S2->E2)
            if i % 2 == 0:
                segment_name = f"Fly-{fly_counter}"
                fly_counter += 1
            else:
                segment_name = f"GU-{gu_counter}"
                gu_counter += 1

            annotation_text = (
                f"--- {segment_name} ---\n"
                f"Flight Time: {total_flight_time:.2f}s\n"
                f"Collect Time: {total_collection_time:.2f}s\n"
                f"Avg Speed: {avg_velocity:.1f}m/s"
            )

            # --- Place annotation on the plot at the midpoint of the major segment ---
            mid_point_major = (p_start_major + p_end_major) / 2
            ax.text(mid_point_major[0], mid_point_major[1] + 40, annotation_text, fontsize=8,
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', fc='cyan', alpha=0.8))

    ax.legend()
    fig.tight_layout()
    plt.show()