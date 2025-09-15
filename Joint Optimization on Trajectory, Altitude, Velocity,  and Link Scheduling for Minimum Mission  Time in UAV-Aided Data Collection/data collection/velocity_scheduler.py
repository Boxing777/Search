# velocity_scheduler.py
import numpy as np
from scipy.optimize import linprog
from config import UAV_MAX_SPEED_VMAX, DATA_TRANSMISSION_RATE_R, DEFAULT_DATA_FILE_SIZE_MBITS

def discretize_trajectory(trajectory, segment_length=10.0):
    """Discretizes the trajectory into points with roughly equal spacing."""
    points = [trajectory[0]]
    total_dist = 0
    for i in range(len(trajectory) - 1):
        p1, p2 = trajectory[i], trajectory[i+1]
        segment_vec = p2 - p1
        length = np.linalg.norm(segment_vec)
        if length < 1e-6: continue
        
        direction = segment_vec / length
        
        num_new_points = int(np.floor((total_dist + length) / segment_length)) - int(np.floor(total_dist / segment_length))
        
        for j in range(1, num_new_points + 1):
            dist_from_p1 = (int(np.floor(total_dist / segment_length)) + j) * segment_length - total_dist
            if dist_from_p1 <= length:
                points.append(p1 + dist_from_p1 * direction)
        total_dist += length
    
    if np.linalg.norm(points[-1] - trajectory[-1]) > 1e-6:
        points.append(trajectory[-1])
        
    return np.array(points)

def solve_velocity_and_scheduling(trajectory, gu_locations, d_h):
    """Implements the BCD algorithm for velocity and link scheduling (Problem P3.1)."""
    print("Step 3: Optimizing velocity and link scheduling...")

    traj_points = discretize_trajectory(trajectory, segment_length=10.0)
    J = len(traj_points) - 1
    if J == 0:
        print("Trajectory too short for discretization. Mission time is 0.")
        return 0
    
    N = len(gu_locations)
    d_j = np.linalg.norm(traj_points[1:] - traj_points[:-1], axis=1)

    in_range = np.zeros((N, J), dtype=bool)
    for j in range(J):
        mid_point = (traj_points[j] + traj_points[j+1]) / 2
        for n in range(N):
            if np.linalg.norm(mid_point - gu_locations[n]) <= d_h:
                in_range[n, j] = True

    # BCD Initialization
    delta = d_j / UAV_MAX_SPEED_VMAX
    
    max_iters = 10
    tolerance = 1e-3
    prev_time = np.inf

    for r in range(max_iters):
        # --- Step 1: Optimize I for given delta ---
        c = np.tile(delta, N)
        
        # *** MAJOR FIX HERE ***
        # Constraint sum_n(I_nj) <= 1 for each j
        A_ub_sum_I = np.zeros((J, N*J))
        for j in range(J):
            for n in range(N):
                A_ub_sum_I[j, n*J + j] = 1
        b_ub_sum_I = np.ones(J)
        
        # Constraint sum_j(I_nj * delta_j * R) >= M_n for each n
        A_ub_data = np.zeros((N, N*J))
        data_to_collect = DEFAULT_DATA_FILE_SIZE_MBITS * 1e6
        b_ub_data = np.full(N, -data_to_collect)
        for n in range(N):
            A_ub_data[n, n*J:(n+1)*J] = -delta * DATA_TRANSMISSION_RATE_R
        
        A_ub = np.vstack([A_ub_sum_I, A_ub_data])
        b_ub = np.concatenate([b_ub_sum_I, b_ub_data])

        bounds = [(0, 1) for _ in range(N*J)]
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        I = res.x.reshape(N, J) if res.success else np.zeros((N, J))

        # --- Step 2: Optimize delta for given I ---
        c_delta = np.ones(J)
        
        A_ub_data_delta = -I * DATA_TRANSMISSION_RATE_R
        b_ub_data_delta = np.full(N, -data_to_collect)
        
        bounds_delta = [(dj / UAV_MAX_SPEED_VMAX, None) for dj in d_j]
        
        res = linprog(c_delta, A_ub=A_ub_data_delta, b_ub=b_ub_data_delta, bounds=bounds_delta, method='highs')

        if res.success:
            delta = res.x
        
        total_mission_time = np.sum(delta)
        if abs(total_mission_time - prev_time) < tolerance:
            print(f"  BCD converged at iteration {r+1}.")
            break
        prev_time = total_mission_time

    total_mission_time = np.sum(delta)
    print(f"BCD finished. Minimum mission time: {total_mission_time:.2f} s")
    return total_mission_time