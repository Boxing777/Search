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
        Estimates mission time using the CMC method with strict sequential constraints.
        This version correctly handles segments that are fully inside a communication circle.
        """
        # --- Step 1 remains unchanged ---
        if not ordered_gn_indices:
            return {"total_time": 0.0, "flight_time": 0.0, "hover_time": 0.0, "path_length": 0.0, "plot_points": {}}

        print("\n--- Estimating Mission Time with Sequential CMC Method ---")

        convex_result = self.convex_planner.plan_shortest_path_for_sequence(ordered_gn_indices)
        if not convex_result["path"].any():
            return {"total_time": 0.0, "flight_time": 0.0, "hover_time": 0.0, "path_length": 0.0, "plot_points": {}}
        
        shortest_path = convex_result["path"]
        shortest_path_length = convex_result["length"]
        total_flight_time = shortest_path_length / self.uav_speed

        gns_coords = self.convex_planner.all_gns
        comm_radius = self.convex_planner.comm_radius

        # --- Step 2: Generate Event List (Revised Logic) ---
        events = []
        current_path_dist = 0
        for i in range(len(shortest_path) - 1):
            p1, p2 = shortest_path[i], shortest_path[i+1]
            segment_len = np.linalg.norm(p2 - p1)
            if segment_len < 1e-9: continue

            for gn_index in ordered_gn_indices:
                gn_coord = gns_coords[gn_index]
                
                p1_inside = np.linalg.norm(p1 - gn_coord) <= comm_radius
                p2_inside = np.linalg.norm(p2 - gn_coord) <= comm_radius
                
                intersections = self._get_line_circle_intersections(p1, p2, gn_coord, comm_radius)

                # --- START OF FINAL CORRECTION ---
                if p1_inside and p2_inside:
                    # Case 1: Entire segment is inside.
                    events.append({'progress': current_path_dist, 'point': p1, 'type': 'START_INTERNAL', 'gn_index': gn_index})
                    events.append({'progress': current_path_dist + segment_len, 'point': p2, 'type': 'END_INTERNAL', 'gn_index': gn_index})
                
                elif p1_inside and not p2_inside:
                    # Case 2: Exiting. Find the single intersection.
                    if intersections:
                        # Should be only one intersection, but find the one closest to p1 just in case
                        point = min(intersections, key=lambda p: np.linalg.norm(p - p1))
                        progress = current_path_dist + np.linalg.norm(point - p1)
                        events.append({'progress': progress, 'point': point, 'type': 'EXIT', 'gn_index': gn_index})

                elif not p1_inside and p2_inside:
                    # Case 3: Entering. Find the single intersection.
                    if intersections:
                        point = min(intersections, key=lambda p: np.linalg.norm(p - p1))
                        progress = current_path_dist + np.linalg.norm(point - p1)
                        events.append({'progress': progress, 'point': point, 'type': 'ENTER', 'gn_index': gn_index})

                elif not p1_inside and not p2_inside and len(intersections) == 2:
                    # Case 4: Crossing from outside to outside.
                    p_a, p_b = intersections
                    prog_a = current_path_dist + np.linalg.norm(p_a - p1)
                    prog_b = current_path_dist + np.linalg.norm(p_b - p1)
                    
                    if prog_a > prog_b:
                        prog_a, prog_b = prog_b, prog_a
                        p_a, p_b = p_b, p_a
                    
                    events.append({'progress': prog_a, 'point': p_a, 'type': 'ENTER', 'gn_index': gn_index})
                    events.append({'progress': prog_b, 'point': p_b, 'type': 'EXIT', 'gn_index': gn_index})
                # --- END OF MODIFICATION ---

            current_path_dist += segment_len
        
        events.sort(key=lambda x: x['progress'])

        # Step 3: Scan the event list to determine sequential service segments
        collection_periods = {gn_index: [] for gn_index in ordered_gn_indices}
        gn_map = {gn_index: i for i, gn_index in enumerate(ordered_gn_indices)}
        
        currently_serving_gn = None
        current_fip = None
        
        for event in events:
            event_gn_idx = event['gn_index']
            event_type = event['type']
            event_point = event['point']

            if currently_serving_gn is None:
                # Can start serving if we enter a new zone
                if event_type in ['ENTER', 'START_INTERNAL']:
                    currently_serving_gn = event_gn_idx
                    current_fip = event_point
            else: # Currently serving a GN
                if event_gn_idx == currently_serving_gn:
                    # Event belongs to the GN we are currently serving
                    if event_type in ['EXIT', 'END_INTERNAL']:
                        collection_periods[currently_serving_gn].append((current_fip, event_point))
                        currently_serving_gn = None
                        current_fip = None
                else: # Event belongs to a different GN
                    # Check if the new GN is next in the service order
                    if gn_map.get(event_gn_idx, -1) > gn_map.get(currently_serving_gn, -1):
                        if event_type in ['ENTER', 'START_INTERNAL']:
                            # Overlapping case: handover service
                            collection_periods[currently_serving_gn].append((current_fip, event_point))
                            currently_serving_gn = event_gn_idx
                            current_fip = event_point

        if currently_serving_gn is not None:
             collection_periods[currently_serving_gn].append((current_fip, shortest_path[-1]))

        # Step 4 & 5 remain unchanged as they correctly iterate over the generated segments.
        # ... (The rest of the function is identical to your last correct version) ...
        total_hover_time = 0.0
        cmc_points_for_plot = []

        for gn_index in ordered_gn_indices:
            gn_coord = gns_coords[gn_index]
            data_collected_on_segment = 0
            
            segments_for_gn = collection_periods[gn_index]
            
            if segments_for_gn:
                final_fip = segments_for_gn[0][0]
                final_fop = segments_for_gn[-1][1]
                
                cmc_points_for_plot.append({
                    "gn_index": gn_index,
                    "fip": final_fip,
                    "fop": final_fop
                })

                for seg_start, seg_end in segments_for_gn:
                     # This logic correctly handles multiple segments
                     data_collected_on_segment += self.traj_optimizer._calculate_collected_data(
                        seg_start, seg_end, gn_coord
                    )
            
            hover_time_for_gn = 0
            data_shortfall = required_data_per_gn - data_collected_on_segment
            if data_shortfall > 0:
                hover_point, min_dist_sq = None, float('inf')
                if segments_for_gn:
                    for seg_start, seg_end in segments_for_gn:
                        closest_p = self._get_closest_point_on_segment(seg_start, seg_end, gn_coord)
                        dist_sq = np.sum((closest_p - gn_coord)**2)
                        if dist_sq < min_dist_sq:
                            min_dist_sq, hover_point = dist_sq, closest_p
                else:
                    for i in range(len(shortest_path) - 1):
                        closest_p = self._get_closest_point_on_segment(shortest_path[i], shortest_path[i+1], gn_coord)
                        dist_sq = np.sum((closest_p - gn_coord)**2)
                        if dist_sq < min_dist_sq:
                           min_dist_sq, hover_point = dist_sq, closest_p
                
                if hover_point is not None:
                    rate = self.traj_optimizer.calculate_hover_rate_at_point(hover_point, gn_coord)
                    hover_time_for_gn = data_shortfall / rate if rate > 1e-6 else float('inf')
                else:
                    hover_time_for_gn = float('inf')
            
            total_hover_time += hover_time_for_gn
            print(f"    -> SeqCMC for GN {gn_index}: Flight Collection: {data_collected_on_segment/1e6:.2f} Mbits, Hover: {hover_time_for_gn:.2f}s")
        
        total_mission_time = total_flight_time + total_hover_time
        print("  - Sequential CMC time estimation complete.")
        
        return {
            "total_time": total_mission_time,
            "flight_time": total_flight_time,
            "hover_time": total_hover_time,
            "path_length": shortest_path_length,
            "plot_points": cmc_points_for_plot
        }