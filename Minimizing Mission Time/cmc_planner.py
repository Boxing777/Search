# cmc_planner.py
# ==============================================================================
#      CMC (Convex-Maximal-Collection) Planner & Time Estimator
#
# File Objective:
# Implements the CMC method. This approach takes the optimal geometric path
# from the Convex Planner and provides a more accurate and fair estimation of
# the mission time by utilizing the *entire* collection opportunity along that path.
#
# Core Idea:
# 1. Use the Convex Planner to get the absolute shortest path.
# 2. For each GN, instead of using the restrictive So -> Eo segment, find the
#    true intersection points (FIP_cmc and FOP_cmc) where the shortest path
#    enters and exits the GN's communication circle.
# 3. Calculate the total data collected along this maximal segment (FIP_cmc -> FOP_cmc).
# 4. Calculate any necessary hover time based on the data shortfall, hovering
#    at the point of highest data rate (closest point to the GN).
# 5. The final mission time is the total flight time (from the shortest path)
#    plus the sum of all calculated hover times.
# ==============================================================================

import numpy as np
from typing import List, Dict, Tuple, Optional

# Import necessary components from your existing modules
from trajectory_optimizer import TrajectoryOptimizer
from convex_trajectory_planner import ConvexTrajectoryPlanner
import parameters as params

class CMCPlanner:
    """
    Implements the CMC planning and time estimation algorithm.
    It refines the mission time calculation for a given convex-optimized path.
    """

    def __init__(self,
                 traj_optimizer: TrajectoryOptimizer,
                 convex_planner: ConvexTrajectoryPlanner):
        """
        Initializes the CMC planner.

        Args:
            traj_optimizer (TrajectoryOptimizer): An instance of your trajectory optimizer.
            convex_planner (ConvexTrajectoryPlanner): An instance of your convex planner.
        """
        self.traj_optimizer = traj_optimizer
        self.convex_planner = convex_planner
        self.uav_speed = params.UAV_MAX_SPEED

    def _get_line_circle_intersections(self, p1: np.ndarray, p2: np.ndarray,
                                       center: np.ndarray, radius: float) -> List[np.ndarray]:
        """
        Calculates the intersection points of a line segment (p1-p2) and a circle.
        """
        p1_local, p2_local = p1 - center, p2 - center
        d = p2_local - p1_local
        dr_sq = np.dot(d, d)
        
        if dr_sq < 1e-9: # The segment is a point
            # If the single point is inside the circle, it is an "intersection"
            if np.linalg.norm(p1_local) <= radius:
                return [p1]
            return []

        D = np.linalg.det(np.vstack([p1_local, p2_local]))
        delta = radius**2 * dr_sq - D**2

        if delta < 0: # No intersection
            # But what if the whole segment is inside?
            p1_inside = np.linalg.norm(p1_local) <= radius
            p2_inside = np.linalg.norm(p2_local) <= radius
            if p1_inside and p2_inside:
                return [p1, p2]
            return []

        intersections = []
        sqrt_delta = np.sqrt(delta)
        
        sgn = np.sign(d[1]) if d[1] != 0 else 1.0

        for sign in [-1, 1]:
            x = (D * d[1] + sign * sgn * d[0] * sqrt_delta) / dr_sq
            y = (-D * d[0] + sign * abs(d[1]) * sqrt_delta) / dr_sq
            
            intersection_local = np.array([x, y])
            
            # Check if the intersection point lies on the segment p1-p2
            dot_product = np.dot(intersection_local - p1_local, d)
            if -1e-9 <= dot_product <= dr_sq + 1e-9:
                 intersections.append(intersection_local + center)
        
        # --- COMPLETELY REVISED LOGIC FOR ENDPOINTS ---
        
        def is_point_in_list(point, point_list):
            if not point_list:
                return False
            return np.any(np.all(np.isclose(point, np.array(point_list)), axis=1))

        p1_inside = np.linalg.norm(p1_local) <= radius
        p2_inside = np.linalg.norm(p2_local) <= radius
        
        # If an endpoint is inside the circle and it hasn't been added yet, add it.
        # This correctly handles cases where the segment starts or ends inside the circle.
        if p1_inside and not is_point_in_list(p1, intersections):
            intersections.append(p1)
        if p2_inside and not is_point_in_list(p2, intersections):
            intersections.append(p2)
            
        return intersections

    def _get_closest_point_on_segment(self, p1: np.ndarray, p2: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Finds the point on the line segment p1-p2 that is closest to a given point."""
        d = p2 - p1
        if np.all(d == 0): return p1
        
        t = np.dot(point - p1, d) / np.dot(d, d)
        t = np.clip(t, 0, 1) # Clip t to be between 0 and 1 to stay on the segment
        
        return p1 + t * d

    def estimate_mission_time(self, ordered_gn_indices: List[int], required_data_per_gn: float) -> Dict:
        """
        Estimates the mission time for a given sequence using the CMC method.
        ...
        """
        if not ordered_gn_indices:
            # This part is unchanged
            return {"total_time": 0.0, "flight_time": 0.0, "hover_time": 0.0, "path_length": 0.0, "plot_points": {}}

        print("\n--- Estimating Mission Time with CMC (Convex-Maximal-Collection) Method ---")

        # This part is unchanged
        convex_result = self.convex_planner.plan_shortest_path_for_sequence(ordered_gn_indices)
        if not convex_result["path"].any():
            return {"total_time": 0.0, "flight_time": 0.0, "hover_time": 0.0, "path_length": 0.0, "plot_points": {}}
        
        shortest_path = convex_result["path"]
        shortest_path_length = convex_result["length"]
        total_flight_time = shortest_path_length / self.uav_speed

        total_hover_time = 0.0
        gns_coords = self.convex_planner.all_gns
        comm_radius = self.convex_planner.comm_radius

        # Prepare lists for visualization points
        all_fips_cmc = []
        all_fops_cmc = []

        for gn_index in ordered_gn_indices:
            gn_coord = gns_coords[gn_index]
            
            all_intersections = []
            for i in range(len(shortest_path) - 1):
                p1, p2 = shortest_path[i], shortest_path[i+1]
                intersections = self._get_line_circle_intersections(p1, p2, gn_coord, comm_radius)
                all_intersections.extend(intersections)

            fip_cmc, fop_cmc = None, None
            if len(all_intersections) >= 2:
                # To sort intersections correctly, we find their progress along the path
                unique_intersections = np.unique(np.array(all_intersections), axis=0)
                path_progress = []
                current_path_dist = 0
                for i in range(len(shortest_path) - 1):
                    p_start, p_end = shortest_path[i], shortest_path[i+1]
                    for point in unique_intersections:
                        proj_point = self._get_closest_point_on_segment(p_start, p_end, point)
                        if np.linalg.norm(proj_point - point) < 1e-6:
                             progress = current_path_dist + np.linalg.norm(proj_point - p_start)
                             # Avoid adding duplicate points from shared segment endpoints
                             if not any(np.isclose(progress, p[0]) for p in path_progress):
                                path_progress.append((progress, point))
                    current_path_dist += np.linalg.norm(p_end - p_start)

                if path_progress:
                    path_progress.sort(key=lambda x: x[0])
                    fip_cmc = path_progress[0][1]
                    fop_cmc = path_progress[-1][1]

            data_collected_on_segment = 0
            hover_time_for_gn = 0
            
            if fip_cmc is not None and fop_cmc is not None:
                all_fips_cmc.append(fip_cmc)
                all_fops_cmc.append(fop_cmc)
                
                # For simplicity, we assume the collection path is a straight line.
                # A more precise method would integrate over the actual convex path segments
                # that lie within the circle, but this is a good approximation.
                data_collected_on_segment = self.traj_optimizer._calculate_collected_data(
                    fip_cmc, fop_cmc, gn_coord
                )
            
            data_shortfall = required_data_per_gn - data_collected_on_segment
            
            if data_shortfall > 0:
                # +++ START OF MODIFICATION +++
                hover_point = None
                if fip_cmc is not None and fop_cmc is not None:
                    # Case 1: A valid collection segment exists.
                    # Find the closest point on this segment to the GN for hovering.
                    hover_point = self._get_closest_point_on_segment(fip_cmc, fop_cmc, gn_coord)
                else:
                    # Case 2: No valid collection segment was found (fip_cmc is None).
                    # The UAV must hover at the point on its *entire* path that is closest to the GN.
                    min_dist_sq = float('inf')
                    for i in range(len(shortest_path) - 1):
                        p1, p2 = shortest_path[i], shortest_path[i+1]
                        closest_p_on_segment = self._get_closest_point_on_segment(p1, p2, gn_coord)
                        dist_sq = np.sum((closest_p_on_segment - gn_coord)**2)
                        if dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                            hover_point = closest_p_on_segment
                
                # Calculate hover time using the determined hover_point
                if hover_point is not None:
                    rate_at_hover_point = self.traj_optimizer.calculate_hover_rate_at_point(hover_point, gn_coord)
                    hover_time_for_gn = data_shortfall / rate_at_hover_point if rate_at_hover_point > 1e-6 else float('inf')
                else:
                    # This case should ideally not be reached, but as a fallback:
                    hover_time_for_gn = float('inf')
                # +++ END OF MODIFICATION +++
                
            total_hover_time += hover_time_for_gn
            print(f"    -> CMC for GN {gn_index}: Flight Collection Data: {data_collected_on_segment/1e6:.2f} Mbits, Required Hover Time: {hover_time_for_gn:.2f}s")

        total_mission_time = total_flight_time + total_hover_time
        print("  - CMC time estimation complete.")
        
        return {
            "total_time": total_mission_time,
            "flight_time": total_flight_time,
            "hover_time": total_hover_time,
            "path_length": shortest_path_length,
            "plot_points": {"fips": all_fips_cmc, "fops": all_fops_cmc}
        }