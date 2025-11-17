# bob_planner.py (Revised for BOB-V Method)
# ==============================================================================
#           BOB-V (BOB with V-Shape-like Optimization) Planner
#
# Core Idea:
# 1. Pre-computation: Run Convex + CMC logic to determine the ideal FIP_cmc
#    and FOP_cmc for all GNs. These serve as intelligent anchor points.
# 2. Sequential Optimization: For each GN_i in the sequence:
#    a. The starting point is the actual optimized FOP from the previous step (FOP'_{i-1}).
#    b. The target anchor for the *next* leg is the pre-computed FIP_cmc of GN_{i+1}.
#    c. Perform a 2D search (like JOFC) for the best FIP'_i and FOP'_i on the
#       current GN_i's circle that minimizes the total leg time:
#       t_flight_in + t_collection + t_flight_out.
# ==============================================================================

import numpy as np
from typing import List, Dict

# Import necessary components
from trajectory_optimizer import TrajectoryOptimizer
from convex_trajectory_planner import ConvexTrajectoryPlanner
import parameters as params

class BOBPlanner:
    """
    Implements the BOB-V planning algorithm.
    """

    def __init__(self,
                 gns: np.ndarray,
                 data_center_pos: np.ndarray,
                 traj_optimizer: TrajectoryOptimizer,
                 convex_planner: ConvexTrajectoryPlanner):
        self.all_gns = gns
        self.data_center_pos = data_center_pos
        self.traj_optimizer = traj_optimizer
        self.convex_planner = convex_planner
        self.comm_radius = self.traj_optimizer.comm_radius_d
        self.uav_speed = params.UAV_MAX_SPEED

    # Helper functions copied from cmc_planner
    def _get_line_circle_intersections(self, p1, p2, center, radius):
        # ... (This function remains unchanged, same as in cmc_planner)
        p1_local, p2_local = p1 - center, p2 - center
        d = p2_local - p1_local
        dr_sq = np.dot(d, d)
        if dr_sq < 1e-9: return [p1] if np.linalg.norm(p1_local) <= radius else []
        D = np.linalg.det(np.vstack([p1_local, p2_local]))
        delta = radius**2 * dr_sq - D**2
        if delta < 0: return [p1, p2] if np.linalg.norm(p1_local) <= radius and np.linalg.norm(p2_local) <= radius else []
        intersections = []
        sqrt_delta = np.sqrt(delta)
        sgn = np.sign(d[1]) if d[1] != 0 else 1.0
        for sign in [-1, 1]:
            x = (D * d[1] + sign * sgn * d[0] * sqrt_delta) / dr_sq
            y = (-D * d[0] + sign * abs(d[1]) * sqrt_delta) / dr_sq
            intersection_local = np.array([x, y])
            dot_product = np.dot(intersection_local - p1_local, d)
            if -1e-9 <= dot_product <= dr_sq + 1e-9: intersections.append(intersection_local + center)
        def is_point_in_list(point, point_list): return bool(point_list) and np.any(np.all(np.isclose(point, np.array(point_list)), axis=1))
        if np.linalg.norm(p1_local) <= radius and not is_point_in_list(p1, intersections): intersections.append(p1)
        if np.linalg.norm(p2_local) <= radius and not is_point_in_list(p2, intersections): intersections.append(p2)
        return intersections

    def _get_closest_point_on_segment(self, p1, p2, point):
        d = p2 - p1
        if np.all(d == 0): return p1
        t = np.dot(point - p1, d) / np.dot(d, d)
        t = np.clip(t, 0, 1)
        return p1 + t * d

    def plan_path(self, ordered_gn_indices: List[int], required_data_per_gn: float) -> Dict:
        """
        Plans the BOB-V trajectory for a given, fixed sequence of GNs.
        """
        if not ordered_gn_indices:
            return {"segments": [], "total_time": 0.0, "total_length": 0.0}

        print("\n--- Planning with BOB-V (Hybrid CMC-JOFC) Method ---")

        # Step 1: Pre-compute all FIP_cmc anchors
        print("  - Step 1: Calculating FIP_cmc anchors for all GNs...")
        convex_result = self.convex_planner.plan_shortest_path_for_sequence(ordered_gn_indices)
        if not convex_result["path"].any():
            return {"segments": [], "total_time": 0.0, "total_length": 0.0}
        shortest_path = convex_result["path"]
        
        fip_cmc_anchors = {}
        for gn_index in ordered_gn_indices:
            gn_coord = self.all_gns[gn_index]
            all_intersections = []
            for i in range(len(shortest_path) - 1):
                p1, p2 = shortest_path[i], shortest_path[i+1]
                intersections = self._get_line_circle_intersections(p1, p2, gn_coord, self.comm_radius)
                all_intersections.extend(intersections)
            
            if all_intersections:
                path_progress = []
                current_path_dist = 0
                unique_intersections = np.unique(np.array(all_intersections), axis=0)
                for i in range(len(shortest_path) - 1):
                    p_start, p_end = shortest_path[i], shortest_path[i+1]
                    for point in unique_intersections:
                        proj_point = self._get_closest_point_on_segment(p_start, p_end, point)
                        if np.linalg.norm(proj_point - point) < 1e-6:
                             progress = current_path_dist + np.linalg.norm(proj_point - p_start)
                             if not any(np.isclose(progress, p[0]) for p in path_progress):
                                path_progress.append((progress, point))
                    current_path_dist += np.linalg.norm(p_end - p_start)
                
                if path_progress:
                    path_progress.sort(key=lambda x: x[0])
                    fip_cmc_anchors[gn_index] = path_progress[0][1] # Get the first point

        # Step 2: Perform sequential JOFC-like optimization using the FIP_cmc anchors
        print("  - Step 2: Performing sequential optimization...")
        
        bob_path_segments = []
        total_mission_time = 0.0
        total_path_length = 0.0
        
        previous_fop = self.data_center_pos

        for i, gn_index in enumerate(ordered_gn_indices):
            current_gn_coord = self.all_gns[gn_index]
            
            # Determine the target anchor for the t_flight_out calculation
            is_last_gn = (i == len(ordered_gn_indices) - 1)
            if is_last_gn:
                next_target_anchor = self.data_center_pos
            else:
                next_gn_index = ordered_gn_indices[i+1]
                # Use the pre-computed FIP_cmc of the *next* GN as the anchor
                next_target_anchor = fip_cmc_anchors.get(next_gn_index, self.all_gns[next_gn_index])
            
            # This logic is now identical to the JOFC loop in main.py
            min_local_leg_time = float('inf')
            best_local_config = {}

            sp = previous_fop
            is_overlapping = np.linalg.norm(sp - current_gn_coord) <= self.comm_radius
            
            num_angles = 36 # Adjust for precision
            angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
            fop_candidates = [current_gn_coord + self.comm_radius * np.array([np.cos(a), np.sin(a)]) for a in angles]
            fip_candidates = [sp] if is_overlapping else [current_gn_coord + self.comm_radius * np.array([np.cos(a), np.sin(a)]) for a in angles]

            for fip in fip_candidates:
                flight_time_in = np.linalg.norm(sp - fip) / self.uav_speed
                for fop in fop_candidates:
                    c_max = self.traj_optimizer.calculate_fm_max_capacity(fip, fop, current_gn_coord)
                    
                    if required_data_per_gn <= c_max:
                        optimal_oh, t_collect_theoretical = self.traj_optimizer.find_optimal_fm_trajectory(
                            fip, fop, current_gn_coord, required_data_per_gn, is_overlapping=is_overlapping
                        )
                    else:
                        optimal_oh = current_gn_coord
                        collection_flight_time = (np.linalg.norm(fip - optimal_oh) + np.linalg.norm(fop - optimal_oh)) / self.uav_speed
                        hover_time = (required_data_per_gn - c_max) / self.traj_optimizer.hover_datarate
                        t_collect_theoretical = collection_flight_time + hover_time
                    
                    physical_dist = np.linalg.norm(optimal_oh - fip) + np.linalg.norm(fop - optimal_oh)
                    physical_time = physical_dist / self.uav_speed
                    t_collect = max(t_collect_theoretical, physical_time)

                    flight_time_out = np.linalg.norm(next_target_anchor - fop) / self.uav_speed
                    total_leg_time = flight_time_in + t_collect + flight_time_out

                    if total_leg_time < min_local_leg_time:
                        min_local_leg_time = total_leg_time
                        best_local_config = {
                            'fip': fip, 'fop': fop, 'oh': optimal_oh,
                            'gn_index': gn_index, 'service_time': flight_time_in + t_collect,
                            'fly_in_dist': np.linalg.norm(sp - fip),
                            'collection_dist': physical_dist
                        }

            print(f"    -> Optimized for GN {gn_index}. Local Leg Time: {best_local_config['service_time']:.2f}s")
            total_mission_time += best_local_config['service_time']
            total_path_length += best_local_config['fly_in_dist'] + best_local_config['collection_dist']
            
            bob_path_segments.append(best_local_config)
            previous_fop = best_local_config['fop']

        final_flight_dist = np.linalg.norm(self.data_center_pos - previous_fop)
        final_flight_time = final_flight_dist / self.uav_speed
        
        total_mission_time += final_flight_time
        total_path_length += final_flight_dist

        print("  - BOB-V planning complete.")

        return {
            "segments": bob_path_segments,
            "total_time": total_mission_time,
            "total_length": total_path_length
        }