# velocity_scheduler.py
import numpy as np
from scipy.optimize import linprog
from config import UAV_MAX_SPEED_VMAX, DATA_TRANSMISSION_RATE_R, DEFAULT_DATA_FILE_SIZE_MBITS

def discretize_trajectory(trajectory, segment_length=20.0):
    """
    Discretizes a given trajectory into smaller segments of roughly equal length.
    
    Args:
        trajectory (np.array): An array of waypoints (N x 2).
        segment_length (float): The desired length for each small segment.

    Returns:
        np.array: An array of the new, more numerous points defining the discretized path.
    """
    points = [trajectory[0]]
    total_dist = 0.0
    for i in range(len(trajectory) - 1):
        p1, p2 = trajectory[i], trajectory[i + 1]
        segment_vec = p2 - p1
        length = np.linalg.norm(segment_vec)
        if length < 1e-6:
            continue
            
        direction = segment_vec / length
        
        # Calculate how many new points to add in this major segment
        num_new_points = int(np.floor((total_dist + length) / segment_length)) - int(np.floor(total_dist / segment_length))
        
        for j in range(1, num_new_points + 1):
            dist_from_p1 = (int(np.floor(total_dist / segment_length)) + j) * segment_length - total_dist
            if dist_from_p1 <= length:
                points.append(p1 + dist_from_p1 * direction)
        total_dist += length
    
    # Ensure the final point of the trajectory is always included
    if np.linalg.norm(points[-1] - trajectory[-1]) > 1e-6:
        points.append(trajectory[-1])
        
    return np.array(points)

def solve_velocity_and_scheduling(trajectory, gu_locations, d_h):
    """
    Implements the robust Block Coordinate Descent (BCD) algorithm to jointly optimize
    velocity (via time allocation delta_j) and link scheduling (I_nj),
    as described in Section IV-B and Algorithm 3 of the paper.

    Args:
        trajectory (np.array): The optimized trajectory waypoints.
        gu_locations (np.array): The locations of the Ground Users.
        d_h (float): The maximum communication radius.

    Returns:
        dict: A dictionary containing summary time stats and detailed per-segment results
              (delta, schedule matrix, and discretized points) for plotting and analysis.
    """
    print("Step 3: Jointly optimizing velocity and link scheduling using BCD...")

    # --- 1. Discretization and Parameter Setup ---
    traj_points = discretize_trajectory(trajectory, segment_length=20.0)
    J = len(traj_points) - 1  # Number of discretized segments
    if J <= 0:
        # Handle very short or non-existent trajectories
        mission_time = np.linalg.norm(trajectory[-1] - trajectory[0]) / UAV_MAX_SPEED_VMAX if len(trajectory) > 1 else 0
        return {
            "time_stats": {"total_mission_time": mission_time, "min_flying_time": mission_time, "min_communication_time": 0},
            "delta_per_segment": np.array([]),
            "schedule_matrix": np.array([]),
            "discretized_points": traj_points
        }

    N = len(gu_locations)  # Number of GUs
    d_j = np.linalg.norm(traj_points[1:] - traj_points[:-1], axis=1)  # Length of each discretized segment

    # Determine which GU is in communication range for each segment
    in_range = np.zeros((N, J), dtype=bool)
    for j in range(J):
        mid_point = (traj_points[j] + traj_points[j+1]) / 2
        for n in range(N):
            if np.linalg.norm(mid_point - gu_locations[n]) <= d_h:
                in_range[n, j] = True

    # --- 2. Robust BCD Initialization ---
    # This is a critical step. A naive initialization can trap the BCD algorithm.
    # We start with a realistic guess for the mission time to ensure feasibility.
    t_fly_min = np.sum(d_j / UAV_MAX_SPEED_VMAX)
    total_data = N * DEFAULT_DATA_FILE_SIZE_MBITS * 1e6
    t_comm_min = total_data / DATA_TRANSMISSION_RATE_R
    
    # Start with a feasible time, adding a small buffer to help the solver.
    initial_total_time = max(t_fly_min, t_comm_min) * 1.05
    
    # Distribute this total time proportionally across segments based on their length
    total_len = np.sum(d_j)
    delta = (d_j / total_len) * initial_total_time if total_len > 0 else np.zeros(J)
    # Ensure the initial delta still respects the maximum velocity constraint
    delta = np.maximum(delta, d_j / UAV_MAX_SPEED_VMAX)
    
    # --- 3. BCD Algorithm Iteration ---
    max_iters = 15
    tolerance = 1e-3
    prev_mission_time = np.inf
    I = np.zeros((N, J))  # Initialize schedule matrix

    for r in range(max_iters):
        # === STEP 3a: Optimize Link Scheduling (I) for a given time allocation (delta) ===
        # This is a Linear Program (LP). We solve for a flattened array of I variables.
        c_I = np.tile(delta, N)
        
        # Constraint 1: sum_n(I_nj) <= 1 for each segment j (at most one GU served at a time).
        A_ub_I_sum = np.zeros((J, N*J))
        b_ub_I_sum = np.ones(J)
        for j in range(J):
            A_ub_I_sum[j, j::J] = 1  # Selects I_0j, I_1j, I_2j, ...
        
        # Constraint 2: sum_j(I_nj * R * delta_j) >= M_n for each GU n (collect all data).
        A_ub_I_data = np.zeros((N, N*J))
        data_per_gu = DEFAULT_DATA_FILE_SIZE_MBITS * 1e6
        b_ub_I_data = np.full(N, -data_per_gu)  # Using -A*x <= -b for a >= constraint
        for n in range(N):
            A_ub_I_data[n, n*J:(n+1)*J] = -delta * DATA_TRANSMISSION_RATE_R
        
        A_ub_I = np.vstack([A_ub_I_sum, A_ub_I_data])
        b_ub_I = np.concatenate([b_ub_I_sum, b_ub_I_data])
        
        # Bounds: I_nj must be 0 if not in range, otherwise between 0 and 1.
        bounds_I = [(0, 1) if in_range.flatten()[i] else (0, 0) for i in range(N*J)]
        
        res_I = linprog(c_I, A_ub=A_ub_I, b_ub=b_ub_I, bounds=bounds_I, method='highs')
        # If solver succeeds, update I. Otherwise, keep the previous I.
        I = res_I.x.reshape(N, J) if res_I.success else I

        # === STEP 3b: Optimize Time Allocation (delta) for a given link schedule (I) ===
        # This is also an LP. Objective: min sum(delta_j).
        c_delta = np.ones(J)
        
        # Constraint 1: sum_j(I_nj * R * delta_j) >= M_n for each GU n.
        A_ub_delta_data = -I * DATA_TRANSMISSION_RATE_R
        b_ub_delta_data = np.full(N, -data_per_gu)
        
        # Bounds: delta_j must be >= d_j / V_max (velocity constraint).
        bounds_delta = [(dj / UAV_MAX_SPEED_VMAX, None) for dj in d_j]
        
        res_delta = linprog(c_delta, A_ub=A_ub_delta_data, b_ub=b_ub_delta_data, bounds=bounds_delta, method='highs')

        if res_delta.success:
            delta = res_delta.x
        
        # --- 4. Check for Convergence ---
        current_mission_time = np.sum(delta)
        if abs(current_mission_time - prev_mission_time) < tolerance:
            print(f"  BCD converged at iteration {r+1}.")
            break
        prev_mission_time = current_mission_time
    
    final_mission_time = np.sum(delta)
    print(f"BCD finished. Minimum mission time calculated: {final_mission_time:.2f} s")
    
    time_stats = {
        "total_mission_time": final_mission_time,
        "min_flying_time": t_fly_min,
        "min_communication_time": t_comm_min
    }

    # Return a dictionary with all detailed results for advanced analysis and plotting
    return {
        "time_stats": time_stats,
        "delta_per_segment": delta,
        "schedule_matrix": I,
        "discretized_points": traj_points
    }