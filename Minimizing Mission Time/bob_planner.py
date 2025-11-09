# bob_planner.py (Revised for BOB-C Method)
# ==============================================================================
#           BOB-C (Balanced Optimization Benchmark based on CMC) Planner
#
# File Objective:
# Implements the revised BOB-C method. This version improves upon the original
# BOB by using more realistic anchor points for its local optimization.
#
# Core Idea:
# 1. Use the Convex Planner to generate the shortest geometric path.
# 2. Use CMC's logic to find the true maximal collection exit points (FOP_cmc)
#    for each GN along this shortest path. These points are better anchors
#    as they represent the latest possible departure time along the optimal path.
# 3. For each GN segment, FIX the exit point to its FOP_cmc.
# 4. Re-optimize the entry point (So') and the collection path (OH) for the
#    current segment to minimize the local leg time (t_in + t_collection).
# ==============================================================================

import numpy as np
from typing import List, Dict

# Import necessary components from your existing modules
from trajectory_optimizer import TrajectoryOptimizer
from convex_trajectory_planner import ConvexTrajectoryPlanner
import parameters as params

class BOBPlanner:
    """
    Implements the BOB-C planning algorithm.
    """

    def __init__(self,
                 gns: np.ndarray,
                 data_center_pos: np.ndarray,
                 traj_optimizer: TrajectoryOptimizer,
                 convex_planner: ConvexTrajectoryPlanner):
        """
        Initializes the BOB-C planner.
        """
        self.all_gns = gns
        self.data_center_pos = data_center_pos
        self.traj_optimizer = traj_optimizer
        self.convex_planner = convex_planner
        self.comm_radius = self.traj_optimizer.comm_radius_d
        self.uav_speed = params.UAV_MAX_SPEED

    # We need the helper functions from CMC planner here to find the anchors
    def _get_line_circle_intersections(self, p1, p2, center, radius):
        # This function should be identical to the one in cmc_planner.py
        p1_local, p2_local = p1 - center, p2 - center
        d = p2_local - p1_local
        dr_sq = np.dot(d, d)
        if dr_sq < 1e-9:
            return [p1] if np.linalg.norm(p1_local) <= radius else []
        D = np.linalg.det(np.vstack([p1_local, p2_local]))
        delta = radius**2 * dr_sq - D**2
        if delta < 0:
            return [p1, p2] if np.linalg.norm(p1_local) <= radius and np.linalg.norm(p2_local) <= radius else []
        intersections = []
        sqrt_delta = np.sqrt(delta)
        sgn = np.sign(d[1]) if d[1] != 0 else 1.0
        for sign in [-1, 1]:
            x = (D * d[1] + sign * sgn * d[0] * sqrt_delta) / dr_sq
            y = (-D * d[0] + sign * abs(d[1]) * sqrt_delta) / dr_sq
            intersection_local = np.array([x, y])
            dot_product = np.dot(intersection_local - p1_local, d)
            if -1e-9 <= dot_product <= dr_sq + 1e-9:
                 intersections.append(intersection_local + center)
        def is_point_in_list(point, point_list):
            return bool(point_list) and np.any(np.all(np.isclose(point, np.array(point_list)), axis=1))
        if np.linalg.norm(p1_local) <= radius and not is_point_in_list(p1, intersections):
            intersections.append(p1)
        if np.linalg.norm(p2_local) <= radius and not is_point_in_list(p2, intersections):
            intersections.append(p2)
        return intersections

    def _get_closest_point_on_segment(self, p1, p2, point):
        # This function should be identical to the one in cmc_planner.py
        d = p2 - p1
        if np.all(d == 0): return p1
        t = np.dot(point - p1, d) / np.dot(d, d)
        t = np.clip(t, 0, 1)
        return p1 + t * d

    def plan_path(self, ordered_gn_indices: List[int], required_data_per_gn: float) -> Dict:
        """
        Plans the BOB-C trajectory for a given, fixed sequence of GNs.
        """
        if not ordered_gn_indices:
            return {"segments": [], "total_time": 0.0, "total_length": 0.0}

        print("\n--- Planning with BOB-C (BOB based on CMC anchors) Method ---")

        # Step 1: Get the shortest path from the Convex Planner
        print("  - Step 1: Calculating geometric skeleton using Convex Planner...")
        convex_result = self.convex_planner.plan_shortest_path_for_sequence(ordered_gn_indices)
        if not convex_result["path"].any():
            print("  - WARNING: Convex planning failed. BOB-C planning aborted.")
            return {"segments": [], "total_time": 0.0, "total_length": 0.0}
        shortest_path = convex_result["path"]

        # Step 2: Calculate the FOP_cmc for each GN to use as fixed anchors
        print("  - Step 2: Calculating FOP_cmc anchors for each GN...")
        fixed_fop_anchors = {}
        for gn_index in ordered_gn_indices:
            gn_coord = self.all_gns[gn_index]
            
            all_intersections = []
            for i in range(len(shortest_path) - 1):
                p1, p2 = shortest_path[i], shortest_path[i+1]
                intersections = self._get_line_circle_intersections(p1, p2, gn_coord, self.comm_radius)
                all_intersections.extend(intersections)

            if len(all_intersections) >= 1:
                # Find the latest intersection point along the path to be the FOP anchor
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
                    fixed_fop_anchors[gn_index] = path_progress[-1][1] # Get the last point
        
        # Step 3: Perform segment-wise local optimization using FOP_cmc anchors
        print("  - Step 3: Performing segment-wise local optimization...")
        
        bob_path_segments = []
        total_mission_time = 0.0
        total_path_length = 0.0
        
        previous_fop_anchor = self.data_center_pos

        for i, gn_index in enumerate(ordered_gn_indices):
            current_gn_coord = self.all_gns[gn_index]
            
            fixed_fop = fixed_fop_anchors.get(gn_index)
            if fixed_fop is None:
                print(f"  - WARNING: No FOP_cmc anchor found for GN {gn_index}. Using Convex Eo as fallback.")
                # Fallback to original BOB's Eo if no intersection is found
                fixed_fop = [seg['end'] for seg in convex_result['collection_segments'] if seg['gn_index'] == gn_index][0]

            min_local_leg_time = float('inf')
            best_local_config = {}

            num_angles = 36
            angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
            fip_candidates = [current_gn_coord + self.comm_radius * np.array([np.cos(a), np.sin(a)]) for a in angles]

            for fip_candidate in fip_candidates:
                t_in = np.linalg.norm(fip_candidate - previous_fop_anchor) / self.uav_speed
                c_max = self.traj_optimizer.calculate_fm_max_capacity(fip_candidate, fixed_fop, current_gn_coord)
                
                if required_data_per_gn <= c_max:
                    optimal_oh, t_collect_theoretical = self.traj_optimizer.find_optimal_fm_trajectory(
                        fip_candidate, fixed_fop, current_gn_coord, required_data_per_gn, is_overlapping=True
                    )
                else:
                    optimal_oh = current_gn_coord
                    collection_flight_time = (np.linalg.norm(fip_candidate - optimal_oh) + np.linalg.norm(fixed_fop - optimal_oh)) / self.uav_speed
                    hover_time = (required_data_per_gn - c_max) / self.traj_optimizer.hover_datarate
                    t_collect_theoretical = collection_flight_time + hover_time
                
                physical_dist = np.linalg.norm(optimal_oh - fip_candidate) + np.linalg.norm(fixed_fop - optimal_oh)
                physical_time = physical_dist / self.uav_speed
                t_collect = max(t_collect_theoretical, physical_time)

                t_local = t_in + t_collect

                if t_local < min_local_leg_time:
                    min_local_leg_time = t_local
                    best_local_config = {
                        'fip': fip_candidate, 'fop': fixed_fop, 'oh': optimal_oh,
                        'gn_index': gn_index, 'service_time': t_local,
                        'fly_in_dist': np.linalg.norm(fip_candidate - previous_fop_anchor),
                        'collection_dist': physical_dist
                    }

            print(f"    -> Optimized for GN {gn_index}. Local Leg Time: {min_local_leg_time:.2f}s")
            total_mission_time += best_local_config['service_time']
            total_path_length += best_local_config['fly_in_dist'] + best_local_config['collection_dist']
            
            bob_path_segments.append(best_local_config)
            previous_fop_anchor = best_local_config['fop']

        final_flight_dist = np.linalg.norm(self.data_center_pos - previous_fop_anchor)
        final_flight_time = final_flight_dist / self.uav_speed
        
        total_mission_time += final_flight_time
        total_path_length += final_flight_dist

        print("  - BOB-C planning complete.")

        return {
            "segments": bob_path_segments,
            "total_time": total_mission_time,
            "total_length": total_path_length
        }