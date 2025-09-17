# trajectory_optimizer_lookahead.py
import numpy as np
import time
from config import START_POS, END_POS
from trajectory_optimizer import solve_tsp_for_order, optimize_waypoints

def stoa_lookahead_algorithm(gu_locations, d_h, lookahead_k=3):
    """
    Implements a Lookahead-enhanced version of the STOA algorithm.
    *** VERSION 2: Corrected Cost Evaluation Model ***
    """
    print(f"Step 2: Optimizing trajectory using STOA with Lookahead (k={lookahead_k})...")
    
    remaining_gus_map = {i: loc for i, loc in enumerate(gu_locations)}
    final_waypoints = [START_POS]
    current_pos = START_POS

    while len(remaining_gus_map) > 0:
        print(f"  Lookahead iteration: {len(remaining_gus_map)} GUs remaining.")
        
        remaining_indices = list(remaining_gus_map.keys())
        remaining_locs = np.array(list(remaining_gus_map.values()))
        
        initial_order_indices = solve_tsp_for_order(current_pos, remaining_locs)
        candidates_to_evaluate = initial_order_indices[:lookahead_k]
        
        best_candidate_idx = -1
        min_estimated_total_cost = float('inf')
        best_segment_to_execute = None

        print(f"    - Evaluating {len(candidates_to_evaluate)} candidates...")
        for candidate_local_idx in candidates_to_evaluate:
            candidate_original_idx = remaining_indices[candidate_local_idx]
            candidate_loc = remaining_locs[candidate_local_idx]
            
            # --- Simulation Step ---
            # a. Temporarily optimize a path to serve only this candidate
            temp_segment, _ = optimize_waypoints(current_pos, np.array([candidate_loc]), d_h)
            
            # *** MAJOR FIX HERE: Correctly define the cost of the current move ***
            # The cost is ONLY the path from current_pos to the candidate's exit point.
            # temp_segment is [current_pos, S_cand, E_cand, END_POS]
            current_move_cost = np.linalg.norm(temp_segment[1] - temp_segment[0]) + np.linalg.norm(temp_segment[2] - temp_segment[1])
            simulated_next_pos = temp_segment[2] # The exit point E_cand

            # b. Create the list of GUs that would remain *after* this move
            simulated_remaining_locs = np.delete(remaining_locs, candidate_local_idx, axis=0)
            
            # --- Evaluation Step ---
            # c. Estimate the future cost from the simulated next position
            if len(simulated_remaining_locs) > 0:
                future_order = solve_tsp_for_order(simulated_next_pos, simulated_remaining_locs)
                future_path_points = np.vstack([simulated_next_pos, simulated_remaining_locs[future_order], END_POS])
                future_cost = np.sum(np.linalg.norm(future_path_points[1:] - future_path_points[:-1], axis=1))
            else:
                future_cost = np.linalg.norm(END_POS - simulated_next_pos)

            # d. Total estimated cost = cost of this move + estimated future cost
            total_estimated_cost = current_move_cost + future_cost
            
            print(f"      - Candidate GU #{candidate_original_idx}: Current Move Cost={current_move_cost:.2f}, Future Cost={future_cost:.2f}, Total Estimated Cost={total_estimated_cost:.2f}")

            # e. Keep track of the best candidate
            if total_estimated_cost < min_estimated_total_cost:
                min_estimated_total_cost = total_estimated_cost
                best_candidate_idx = candidate_local_idx
                best_segment_to_execute = temp_segment

        # ... (The rest of the function remains the same) ...
        chosen_original_idx = remaining_indices[best_candidate_idx]
        print(f"    -> Decision: Choose GU #{chosen_original_idx} as the next target.")
        
        first_segment_path = best_segment_to_execute[0:3]
        
        covered_original_indices = set()
        if len(first_segment_path) > 2:
            p1, p2 = first_segment_path[1], first_segment_path[2]
            d = np.linalg.norm(p2 - p1)
            if d > 1e-6:
                for original_idx, gu_loc in remaining_gus_map.items():
                    t = np.dot(gu_loc - p1, p2 - p1) / d**2
                    t = np.clip(t, 0, 1)
                    closest_point_on_segment = p1 + t * (p2 - p1)
                    dist_to_segment = np.linalg.norm(gu_loc - closest_point_on_segment)
                    if dist_to_segment < d_h:
                        covered_original_indices.add(original_idx)

        if not covered_original_indices:
            covered_original_indices.add(chosen_original_idx)

        final_waypoints.extend(first_segment_path[1:3])
        current_pos = first_segment_path[2]
        
        for idx in covered_original_indices:
            remaining_gus_map.pop(idx, None)

    final_waypoints.append(END_POS)
    final_trajectory = np.array(final_waypoints)
    
    total_length = np.sum(np.linalg.norm(final_trajectory[1:] - final_trajectory[:-1], axis=1))
    print(f"STOA with Lookahead finished. Final trajectory length: {total_length:.2f} m")

    return final_trajectory, total_length