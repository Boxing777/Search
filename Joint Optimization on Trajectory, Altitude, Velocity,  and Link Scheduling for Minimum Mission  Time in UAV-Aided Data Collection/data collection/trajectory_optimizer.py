# trajectory_optimizer.py
import numpy as np
from scipy.optimize import minimize
from python_tsp.heuristics import solve_tsp_simulated_annealing
from config import START_POS, END_POS

def solve_tsp_for_order(start_point, points):
    """Determines the visit order of GUs starting from a specific point."""
    if not points.any():
        return []

    all_points = np.vstack([start_point, points, END_POS])
    
    num_points = len(all_points)
    dist_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            dist_matrix[i, j] = np.linalg.norm(all_points[i] - all_points[j])

    # Trick for open TSP: create a dummy node that connects start and end
    # A simpler way: force the path to be S -> ... -> E
    # Prevent returning to start (0) and leaving end (num_points - 1)
    dist_matrix[:, 0] = 1e9 # High cost to return to start
    dist_matrix[num_points-1, :] = 1e9 # High cost to leave end
    dist_matrix[num_points-1, 0] = 0 # Except allow tour to close from end to start for solver

    permutation, _ = solve_tsp_simulated_annealing(dist_matrix, x0=list(range(num_points)))
    
    start_idx_in_perm = permutation.index(0)
    
    # Reorder the permutation to start with 0
    ordered_perm = permutation[start_idx_in_perm:] + permutation[:start_idx_in_perm]
    
    # Exclude start (0) and end (num_points-1) and map back to original indices
    gu_order_indices = [p - 1 for p in ordered_perm if p != 0 and p != (num_points - 1)]
    return gu_order_indices


def optimize_waypoints(segment_start_pos, gu_locations_ordered, d_h):
    """
    Solves problem (P2.1) for a given GU order and a dynamic start position.
    """
    num_gus = len(gu_locations_ordered)
    if num_gus == 0:
        return np.array([segment_start_pos, END_POS]), np.linalg.norm(END_POS - segment_start_pos)
    
    def objective_func(vars):
        total_length = 0
        waypoints = vars.reshape(-1, 2)
        
        total_length += np.linalg.norm(waypoints[0] - segment_start_pos)
        
        for i in range(num_gus):
            s_i, e_i = waypoints[2*i], waypoints[2*i + 1]
            total_length += np.linalg.norm(e_i - s_i)
            if i < num_gus - 1:
                s_next = waypoints[2*(i+1)]
                total_length += np.linalg.norm(s_next - e_i)
                
        total_length += np.linalg.norm(END_POS - waypoints[-1])
        return total_length

    # *** MAJOR FIX HERE ***
    # Use default arguments in lambda to capture the value of loop variables correctly.
    constraints = []
    for i in range(num_gus):
        gu_loc = gu_locations_ordered[i]
        constraints.append({
            'type': 'ineq',
            'fun': lambda vars, i=i, gl=gu_loc: d_h**2 - np.sum((vars.reshape(-1, 2)[2*i] - gl)**2)
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda vars, i=i, gl=gu_loc: d_h**2 - np.sum((vars.reshape(-1, 2)[2*i+1] - gl)**2)
        })
        
    initial_guess = np.repeat(gu_locations_ordered, 2, axis=0).flatten()

    result = minimize(objective_func, initial_guess, method='SLSQP', constraints=constraints, options={'maxiter': 1000})
    
    if not result.success:
        print("  Warning: Waypoint optimization did not converge.")

    waypoints = result.x.reshape(-1, 2)
    
    full_trajectory_segment = [segment_start_pos]
    for i in range(num_gus):
        full_trajectory_segment.append(waypoints[2*i])
        full_trajectory_segment.append(waypoints[2*i+1])
    full_trajectory_segment.append(END_POS)
    
    return np.array(full_trajectory_segment), result.fun

def stoa_algorithm(gu_locations, d_h):
    """Implements the Segment-based Trajectory Optimization Algorithm (STOA)."""
    print("Step 2: Optimizing trajectory using STOA...")
    
    remaining_gus_map = {i: loc for i, loc in enumerate(gu_locations)}
    final_waypoints = [START_POS]
    current_pos = START_POS

    while len(remaining_gus_map) > 0:
        print(f"  STOA iteration: {len(remaining_gus_map)} GUs remaining.")
        
        remaining_indices = list(remaining_gus_map.keys())
        remaining_locs = np.array(list(remaining_gus_map.values()))
        
        order_in_remaining = solve_tsp_for_order(current_pos, remaining_locs)
        ordered_gus_locs = remaining_locs[order_in_remaining]
        
        # *** MAJOR FIX HERE ***
        # Optimize segment starting from the current position
        segment_trajectory, _ = optimize_waypoints(current_pos, ordered_gus_locs, d_h)
        
        first_segment_path = segment_trajectory[0:3] # current_pos, S_o1, E_o1
        
        covered_original_indices = set()
        for i, original_idx in enumerate(remaining_indices):
            gu_loc = remaining_locs[i]
            p1, p2 = first_segment_path[1], first_segment_path[2] # The S_o1 -> E_o1 line
            
            # Check if this GU's circle intersects with the S_o1 -> E_o1 segment
            d = np.linalg.norm(p2 - p1)
            if d > 1e-6:
                t = np.dot(gu_loc - p1, p2 - p1) / d**2
                t = np.clip(t, 0, 1)
                closest_point = p1 + t * (p2 - p1)
                dist_to_segment = np.linalg.norm(gu_loc - closest_point)
                
                if dist_to_segment < d_h:
                    covered_original_indices.add(original_idx)

        if not covered_original_indices:
            first_gu_original_idx = remaining_indices[order_in_remaining[0]]
            covered_original_indices.add(first_gu_original_idx)

        final_waypoints.extend(first_segment_path[1:3]) # Add S_o1 and E_o1
        current_pos = first_segment_path[2]
        
        for idx in covered_original_indices:
            remaining_gus_map.pop(idx)

    final_waypoints.append(END_POS)
    final_trajectory = np.array(final_waypoints)
    
    total_length = np.sum(np.linalg.norm(final_trajectory[1:] - final_trajectory[:-1], axis=1))
    print(f"STOA finished. Final trajectory length: {total_length:.2f} m")

    return final_trajectory, total_length