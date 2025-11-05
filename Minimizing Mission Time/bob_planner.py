# bob_planner.py
# ==============================================================================
#           BOB (Balanced Optimization Benchmark) Planner
#
# File Objective:
# Implements the BOB (Balanced Optimization Benchmark) method. This approach
# seeks a balance between the geometrically optimal path (from Convex Planner)
# and the time-optimal path (from V-Shape Planner).
#
# Core Idea:
# 1. Use the Convex Planner to generate a globally optimal "geometric skeleton"
#    defined by the So (start) and Eo (end) collection points.
# 2. For each GN segment, FIX the exit point (Eo) from the skeleton, as it is
#    geometrically well-positioned for the next leg of the journey.
# 3. Re-optimize the entry point (So') and the collection path (OH) for the
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
    Implements the BOB planning algorithm.
    """

    def __init__(self,
                 gns: np.ndarray,
                 data_center_pos: np.ndarray,
                 traj_optimizer: TrajectoryOptimizer,
                 convex_planner: ConvexTrajectoryPlanner):
        """
        Initializes the BOB planner.

        Args:
            gns (np.ndarray): Array of ALL GN coordinates, shape (N, 2).
            data_center_pos (np.ndarray): The start and end point of the mission.
            traj_optimizer (TrajectoryOptimizer): An instance of your trajectory optimizer
                                                  to calculate collection times.
            convex_planner (ConvexTrajectoryPlanner): An instance of your convex planner
                                                      to get the geometric skeleton.
        """
        self.all_gns = gns
        self.data_center_pos = data_center_pos
        self.traj_optimizer = traj_optimizer
        self.convex_planner = convex_planner
        self.comm_radius = self.traj_optimizer.comm_radius_d
        self.uav_speed = params.UAV_MAX_SPEED

    def plan_path(self, ordered_gn_indices: List[int], required_data_per_gn: float) -> Dict:
        """
        Plans the BOB trajectory for a given, fixed sequence of GNs.

        Args:
            ordered_gn_indices (List[int]): The pre-determined indices of GNs to visit.
            required_data_per_gn (float): The amount of data to collect from each GN.

        Returns:
            Dict: A dictionary containing the final path segments, total mission time,
                  and total path length.
        """
        if not ordered_gn_indices:
            return {"segments": [], "total_time": 0.0, "total_length": 0.0}

        print("\n--- Planning with BOB (Balanced Optimization Benchmark) Method ---")

        # Step 1: Get the geometric skeleton from the Convex Planner
        print("  - Step 1: Calculating geometric skeleton using Convex Planner...")
        convex_result = self.convex_planner.plan_shortest_path_for_sequence(ordered_gn_indices)
        if not convex_result["collection_segments"]:
            print("  - WARNING: Convex planning failed. BOB planning aborted.")
            return {"segments": [], "total_time": 0.0, "total_length": 0.0}
        
        # Extract Eo points from the skeleton, these are our fixed anchors
        # The structure is {gn_index: eo_coord} for easy lookup
        fixed_eo_points = {seg['gn_index']: seg['end'] for seg in convex_result['collection_segments']}

        # Step 2: Perform segment-wise local optimization
        print("  - Step 2: Performing segment-wise local optimization...")
        
        bob_path_segments = []
        total_mission_time = 0.0
        total_path_length = 0.0
        
        previous_eo = self.data_center_pos

        for i, gn_index in enumerate(ordered_gn_indices):
            current_gn_coord = self.all_gns[gn_index]
            
            # a. Fix the exit point for this segment
            fixed_fop = fixed_eo_points[gn_index]
            
            min_local_leg_time = float('inf')
            best_local_config = {}

            # b. Search for the best new entry point (So') on the circle's boundary
            num_angles = 36 # Discretization for the search space
            angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
            fip_candidates = [current_gn_coord + self.comm_radius * np.array([np.cos(a), np.sin(a)]) for a in angles]

            for fip_candidate in fip_candidates:
                # c. For each candidate, calculate the local leg time
                
                # i. Calculate fly-in time
                t_in = np.linalg.norm(fip_candidate - previous_eo) / self.uav_speed
                
                # ii. Calculate collection time with fixed FIP and FOP
                c_max = self.traj_optimizer.calculate_fm_max_capacity(fip_candidate, fixed_fop, current_gn_coord)
                
                if required_data_per_gn <= c_max:
                    # FM Mode
                    optimal_oh, t_collect_theoretical = self.traj_optimizer.find_optimal_fm_trajectory(
                        fip_candidate, fixed_fop, current_gn_coord, required_data_per_gn, is_overlapping=True # Treat as general case
                    )
                else:
                    # HM Mode
                    optimal_oh = current_gn_coord
                    collection_flight_time = (np.linalg.norm(fip_candidate - optimal_oh) + np.linalg.norm(fixed_fop - optimal_oh)) / self.uav_speed
                    hover_time = (required_data_per_gn - c_max) / self.traj_optimizer.hover_datarate
                    t_collect_theoretical = collection_flight_time + hover_time
                
                # Apply physical time constraint
                physical_dist = np.linalg.norm(optimal_oh - fip_candidate) + np.linalg.norm(fixed_fop - optimal_oh)
                physical_time = physical_dist / self.uav_speed
                t_collect = max(t_collect_theoretical, physical_time)

                # iii. Total local leg time
                t_local = t_in + t_collect

                # d. Keep track of the best local solution
                if t_local < min_local_leg_time:
                    min_local_leg_time = t_local
                    best_local_config = {
                        'fip': fip_candidate,
                        'fop': fixed_fop, # The exit point is fixed
                        'oh': optimal_oh,
                        'gn_index': gn_index,
                        'service_time': t_local,
                        'fly_in_dist': np.linalg.norm(fip_candidate - previous_eo),
                        'collection_dist': physical_dist
                    }

            # e. Accumulate results from the best local configuration
            print(f"    -> Optimized for GN {gn_index}. Local Leg Time: {min_local_leg_time:.2f}s")
            total_mission_time += best_local_config['service_time']
            total_path_length += best_local_config['fly_in_dist'] + best_local_config['collection_dist']
            
            bob_path_segments.append(best_local_config)
            
            # f. Update the starting point for the next iteration
            previous_eo = best_local_config['fop']

        # Step 3: Add final flight back to the data center
        final_flight_dist = np.linalg.norm(self.data_center_pos - previous_eo)
        final_flight_time = final_flight_dist / self.uav_speed
        
        total_mission_time += final_flight_time
        total_path_length += final_flight_dist

        print("  - BOB planning complete.")

        return {
            "segments": bob_path_segments,
            "total_time": total_mission_time,
            "total_length": total_path_length
        }