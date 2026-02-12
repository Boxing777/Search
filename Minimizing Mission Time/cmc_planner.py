# cmc_planner.py
# ==============================================================================
#      CMC (Convex-Maximal-Collection) Planner & Time Estimator (Robust Version)
# ==============================================================================

import numpy as np
from typing import List, Dict, Tuple

# Import necessary components
from trajectory_optimizer import TrajectoryOptimizer
from convex_trajectory_planner import ConvexTrajectoryPlanner
import parameters as params

class CMCPlanner:
    """
    Implements the CMC planning and time estimation algorithm.
    Refines mission time by utilizing the *entire* collection opportunity along 
    the convex shortest path, with robust handling for floating-point boundaries.
    """

    def __init__(self,
                 traj_optimizer: TrajectoryOptimizer,
                 convex_planner: ConvexTrajectoryPlanner):
        self.traj_optimizer = traj_optimizer
        self.convex_planner = convex_planner
        self.uav_speed = params.UAV_MAX_SPEED
        # Tolerance for floating point comparisons (e.g. is point on boundary?)
        self.TOLERANCE = 1e-4 

    def _get_line_circle_intersections(self, p1: np.ndarray, p2: np.ndarray,
                                       center: np.ndarray, radius: float) -> List[np.ndarray]:
        """
        Calculates the intersection points of a line segment (p1-p2) and a circle.
        Includes robust handling for points exactly on the boundary or tangent.
        """
        p1_local, p2_local = p1 - center, p2 - center
        d = p2_local - p1_local
        dr_sq = np.dot(d, d)
        
        # Case: Segment is a single point
        if dr_sq < 1e-9: 
            if np.linalg.norm(p1_local) <= radius + self.TOLERANCE:
                return [p1]
            return []

        D = np.linalg.det(np.vstack([p1_local, p2_local]))
        delta = radius**2 * dr_sq - D**2

        # Case: No intersection (mathematically)
        # We allow a tiny negative delta to account for floating point tangent errors
        if delta < -1e-9: 
            return []
        
        if delta < 0: delta = 0 # Snap to 0 if slightly negative (tangent)

        intersections = []
        sqrt_delta = np.sqrt(delta)
        
        sgn = np.sign(d[1]) if d[1] != 0 else 1.0

        for sign in [-1, 1]:
            x = (D * d[1] + sign * sgn * d[0] * sqrt_delta) / dr_sq
            y = (-D * d[0] + sign * abs(d[1]) * sqrt_delta) / dr_sq
            
            intersection_local = np.array([x, y])
            
            # Check if the intersection point lies on the segment p1-p2
            # We use a dot product projection to check linearity
            # 0 <= dot(AP, AB) / dot(AB, AB) <= 1
            # A=p1_local, B=p2_local, P=intersection_local
            P_minus_A = intersection_local - p1_local
            proj = np.dot(P_minus_A, d) / dr_sq

            if -1e-5 <= proj <= 1.0 + 1e-5: # Allow small epsilon for segment ends
                intersections.append(intersection_local + center)
        
        # Deduplicate points that are extremely close
        unique_intersections = []
        for p in intersections:
            is_new = True
            for existing in unique_intersections:
                if np.linalg.norm(p - existing) < 1e-5:
                    is_new = False
                    break
            if is_new:
                unique_intersections.append(p)
                
        return unique_intersections

    def _get_closest_point_on_segment(self, p1: np.ndarray, p2: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Finds the point on the line segment p1-p2 that is closest to a given point."""
        d = p2 - p1
        if np.all(d == 0): return p1
        
        t = np.dot(point - p1, d) / np.dot(d, d)
        t = np.clip(t, 0, 1)
        return p1 + t * d

    def estimate_mission_time(self, ordered_gn_indices: List[int], required_data_per_gn: float) -> Dict:
        """
        Estimates mission time using the CMC method.
        """
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

        # --- Step 2: Generate Event List (Robust Logic) ---
        events = []
        current_path_dist = 0
        
        for i in range(len(shortest_path) - 1):
            p1, p2 = shortest_path[i], shortest_path[i+1]
            segment_len = np.linalg.norm(p2 - p1)
            
            # Skip zero-length segments
            if segment_len < 1e-6: continue

            for gn_index in ordered_gn_indices:
                gn_coord = gns_coords[gn_index]
                
                # Robust Inside Checks using TOLERANCE
                dist_p1 = np.linalg.norm(p1 - gn_coord)
                dist_p2 = np.linalg.norm(p2 - gn_coord)
                
                p1_inside = dist_p1 <= (comm_radius + self.TOLERANCE)
                p2_inside = dist_p2 <= (comm_radius + self.TOLERANCE)
                
                intersections = self._get_line_circle_intersections(p1, p2, gn_coord, comm_radius)

                # Collect all relevant points on this segment for this GN
                points_on_segment = []
                
                # 1. Add Endpoints if they are inside
                if p1_inside: points_on_segment.append(p1)
                if p2_inside: points_on_segment.append(p2)
                
                # 2. Add geometric intersections (entries/exits)
                points_on_segment.extend(intersections)
                
                # 3. Sort points by distance from p1 to establish sequence
                # Remove duplicates based on distance
                unique_points = []
                seen_dists = []
                
                sorted_candidates = sorted(points_on_segment, key=lambda p: np.linalg.norm(p - p1))
                
                for p in sorted_candidates:
                    d_from_p1 = np.linalg.norm(p - p1)
                    # Simple duplicate check
                    if not any(abs(d - d_from_p1) < 1e-5 for d in seen_dists):
                        seen_dists.append(d_from_p1)
                        unique_points.append(p)

                # 4. Generate Events from the valid segment
                # If we have 2 or more points, we span some distance inside the circle
                if len(unique_points) >= 2:
                    # The first point is the Entry (or Start Internal)
                    # The last point is the Exit (or End Internal)
                    # We create a simplified event stream: ENTER at first, EXIT at last.
                    
                    start_pt = unique_points[0]
                    end_pt = unique_points[-1]
                    
                    prog_start = current_path_dist + np.linalg.norm(start_pt - p1)
                    prog_end = current_path_dist + np.linalg.norm(end_pt - p1)
                    
                    # Add ENTER event
                    events.append({
                        'progress': prog_start, 
                        'point': start_pt, 
                        'type': 'ENTER', 
                        'gn_index': gn_index
                    })
                    # Add EXIT event
                    events.append({
                        'progress': prog_end, 
                        'point': end_pt, 
                        'type': 'EXIT', 
                        'gn_index': gn_index
                    })
                
                # Special Case: Single point inside (Touching boundary or just one point)
                # This contributes 0 collection but marks availability. 
                # Usually redundant if we handle segments, but kept for logic safety.
                elif len(unique_points) == 1 and p1_inside and p2_inside:
                     # Entire tiny segment is inside
                     events.append({'progress': current_path_dist, 'point': p1, 'type': 'ENTER', 'gn_index': gn_index})
                     events.append({'progress': current_path_dist + segment_len, 'point': p2, 'type': 'EXIT', 'gn_index': gn_index})

            current_path_dist += segment_len
        
        events.sort(key=lambda x: x['progress'])

        # --- Step 3: Sequential Service Logic ---
        collection_periods = {gn_index: [] for gn_index in ordered_gn_indices}
        gn_map = {gn_index: i for i, gn_index in enumerate(ordered_gn_indices)}
        
        currently_serving_gn = None
        current_fip = None
        
        # We need to filter out noise. Sometimes floating point gives Enter->Enter.
        # We track state per GN.
        gn_inside_status = {gn_id: False for gn_id in ordered_gn_indices}

        for event in events:
            event_gn_idx = event['gn_index']
            event_type = event['type']
            event_point = event['point']

            # Update status
            if event_type == 'ENTER':
                gn_inside_status[event_gn_idx] = True
            elif event_type == 'EXIT':
                gn_inside_status[event_gn_idx] = False

            # Determine who we are serving
            # Priority: The GN that appears earliest in the ordered list AND we are currently inside it.
            
            # Find candidate GNs (those we are currently 'inside')
            candidates = [gn for gn, is_in in gn_inside_status.items() if is_in]
            
            # Sort candidates by their required order
            candidates.sort(key=lambda x: gn_map[x])
            
            best_candidate = candidates[0] if candidates else None

            # State transition logic
            if currently_serving_gn != best_candidate:
                # We are switching GNs (or stopping service)
                
                # 1. Close current service
                if currently_serving_gn is not None:
                    # We stop serving 'currently_serving_gn' at this event_point
                    collection_periods[currently_serving_gn].append((current_fip, event_point))
                
                # 2. Start new service
                if best_candidate is not None:
                    # We start serving 'best_candidate' at this event_point
                    currently_serving_gn = best_candidate
                    current_fip = event_point
                else:
                    currently_serving_gn = None
                    current_fip = None

        # --- Step 4 & 5: Calculate Data and Hover Time ---
        total_hover_time = 0.0
        cmc_points_for_plot = []

        for gn_index in ordered_gn_indices:
            gn_coord = gns_coords[gn_index]
            data_collected_on_segment = 0
            
            segments_for_gn = collection_periods[gn_index]
            
            if segments_for_gn:
                # Visualization points
                final_fip = segments_for_gn[0][0]
                final_fop = segments_for_gn[-1][1]
                cmc_points_for_plot.append({
                    "gn_index": gn_index,
                    "fip": final_fip,
                    "fop": final_fop
                })

                for seg_start, seg_end in segments_for_gn:
                     dist = np.linalg.norm(seg_end - seg_start)
                     if dist > 1e-6:
                        data_collected_on_segment += self.traj_optimizer._calculate_collected_data(
                            seg_start, seg_end, gn_coord
                        )
            
            hover_time_for_gn = 0
            data_shortfall = required_data_per_gn - data_collected_on_segment
            
            if data_shortfall > 1e-6: # Use epsilon for float zero check
                # Find best hover point (closest to GN) on the valid service segments
                hover_point, min_dist_sq = None, float('inf')
                
                search_segments = segments_for_gn if segments_for_gn else []
                # Fallback: if no valid segments found (rare, but possible if path just touches boundary),
                # use the closest point on the entire path.
                if not search_segments:
                     for i in range(len(shortest_path) - 1):
                        closest_p = self._get_closest_point_on_segment(shortest_path[i], shortest_path[i+1], gn_coord)
                        # Only allow hovering if within radius (robust check)
                        if np.linalg.norm(closest_p - gn_coord) <= comm_radius + self.TOLERANCE:
                             search_segments.append((closest_p, closest_p))

                for seg_start, seg_end in search_segments:
                    closest_p = self._get_closest_point_on_segment(seg_start, seg_end, gn_coord)
                    dist_sq = np.sum((closest_p - gn_coord)**2)
                    if dist_sq < min_dist_sq:
                        min_dist_sq, hover_point = dist_sq, closest_p
                
                if hover_point is not None:
                    rate = self.traj_optimizer.calculate_hover_rate_at_point(hover_point, gn_coord)
                    hover_time_for_gn = data_shortfall / rate if rate > 1e-6 else float('inf')
                else:
                    # Should theoretically not happen if geometry is correct, but safe fallback
                    # Hover at edge closest to GN on the path
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