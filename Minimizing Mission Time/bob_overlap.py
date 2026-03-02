# bob_overlap.py
# ==============================================================================
#      BOB-F (BOB with Flexible Handover) Planner
#
# File Objective:
# Implements the BOB-F strategy in a standalone file. This planner extends the
# BOB-V logic by introducing a "Flexible Handover" mechanism for overlapping GNs.
#
# Core Strategies:
# 1. Strategy 1 (Non-Overlapping): Same as BOB-V. Uses FIP_cmc of the *next*
#    GN as the guidance anchor. Optimizes FIP/FOP on the circle boundary.
# 2. Strategy 2 (Overlapping): Triggered when GN_i and GN_{i+1} overlap.
#    - Breaks the boundary constraint.
#    - Searches for an optimal "Internal Switching Point" (P_flex) inside the
#      intersection area (sampled along the line connecting centers).
#    - Performs joint optimization of the path:
#      SP -> FIP_i -> OH_i -> P_flex -> OH_{i+1} -> FOP_{i+1} -> Anchor_{i+2}
# ==============================================================================

import numpy as np
from typing import List, Dict

# Import necessary components
from trajectory_optimizer import TrajectoryOptimizer
from convex_trajectory_planner import ConvexTrajectoryPlanner
import parameters as params

class BOBOverlapPlanner:
    """
    Implements the BOB-F (Flexible Handover) planning algorithm.
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

    # --- Helper Functions ---
    def _get_line_circle_intersections(self, p1, p2, center, radius):
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

    def _generate_p_flex_candidates(self, center_i: np.ndarray, center_next: np.ndarray, num_samples: int = 10) -> List[np.ndarray]:
        dist = np.linalg.norm(center_next - center_i)
        if dist == 0: return [center_i]
        
        R = self.comm_radius
       
        start_dist = max(0.0, dist - R)
        end_dist = min(dist, R)
        
        if start_dist > end_dist: 
            return []

       
        start_ratio = start_dist / dist
        end_ratio = end_dist / dist
        
        t_values = np.linspace(start_ratio, end_ratio, num_samples)
        
        candidates = []
        vec = center_next - center_i
        for t in t_values:
            candidates.append(center_i + t * vec)
            
        return candidates

    # --- Main Planning Logic ---
    def plan_path(self, ordered_gn_indices: List[int], required_data_per_gn: float) -> Dict:
        if not ordered_gn_indices:
            return {"segments": [], "total_time": 0.0, "total_length": 0.0}

        print("\n--- Planning with BOB-F (Flexible Handover) Method ---")

        # Step 1: Pre-compute Global Anchors (FIP_cmc) using Convex + CMC logic
        print("  - Step 1: Calculating Global Anchors (FIP_cmc)...")
        convex_result = self.convex_planner.plan_shortest_path_for_sequence(ordered_gn_indices)
        if not convex_result["path"].any():
            return {"segments": [], "total_time": 0.0, "total_length": 0.0}
        
        shortest_path = convex_result["path"]
        fip_cmc_anchors = {}
        
        # Reuse logic to find FIP_cmc for each GN
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
                    fip_cmc_anchors[gn_index] = path_progress[0][1] # FIP is the first intersection

        # Step 2: Sequential Optimization with Flexible Handover
        print("  - Step 2: Performing sequential optimization with overlap detection...")
        
        bob_path_segments = []
        total_mission_time = 0.0
        total_path_length = 0.0
        
        previous_fop = self.data_center_pos
        
        i = 0
        while i < len(ordered_gn_indices):
            gn_index = ordered_gn_indices[i]
            current_gn_coord = self.all_gns[gn_index]
            
            # Check for overlap with the NEXT node
            has_overlap = False
            if i < len(ordered_gn_indices) - 1:
                next_gn_index = ordered_gn_indices[i+1]
                next_gn_coord = self.all_gns[next_gn_index]
                if np.linalg.norm(current_gn_coord - next_gn_coord) < 2 * self.comm_radius:
                    has_overlap = True

            # --- Strategy 2: Overlapping Case (Flexible Handover) ---
            if has_overlap:
                print(f"    -> Overlap detected between GN {gn_index} and GN {next_gn_index}. Using Flexible Handover.")
                
                # 1. Define Global Anchor for the end of the sequence (i+2)
                if i + 2 < len(ordered_gn_indices):
                    target_gn_idx = ordered_gn_indices[i+2]
                    global_anchor = fip_cmc_anchors.get(target_gn_idx, self.all_gns[target_gn_idx])
                else:
                    global_anchor = self.data_center_pos

                # 2. Generate Candidate Sets
                # FIP candidates for GN i (on boundary)
                angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
                fip_i_candidates = [current_gn_coord + self.comm_radius * np.array([np.cos(a), np.sin(a)]) for a in angles]
                
                # FOP candidates for GN i+1 (on boundary)
                fop_next_candidates = [next_gn_coord + self.comm_radius * np.array([np.cos(a), np.sin(a)]) for a in angles]
                
                # P_flex candidates (inside overlap)
                p_flex_candidates = self._generate_p_flex_candidates(current_gn_coord, next_gn_coord)

                # 3. Decoupled Optimization
                min_combined_cost = float('inf')
                best_combined_config = {}

                # Loop through P_flex (the pivot)
                for p_flex in p_flex_candidates:
                    
                    # A. Optimize First Leg: SP -> FIP_i -> OH_i -> P_flex
                    #    P_flex acts as the FOP for GN i (inside circle)
                    best_leg_i_cost = float('inf')
                    best_leg_i_data = {}
                    
                    # Check if SP is already inside
                    is_sp_inside = np.linalg.norm(previous_fop - current_gn_coord) <= self.comm_radius
                    # If SP is inside, we can just start from SP (FIP=SP)
                    current_fip_candidates = [previous_fop] if is_sp_inside else fip_i_candidates

                    for fip in current_fip_candidates:
                        t_in = np.linalg.norm(fip - previous_fop) / self.uav_speed
                        
                        # Find V-shape time. FIP is boundary/SP, FOP (P_flex) is inside.
                        # Phase 3 solver handles this asymmetry.
                        c_max = self.traj_optimizer.calculate_fm_max_capacity(fip, p_flex, current_gn_coord)
                        
                        if required_data_per_gn <= c_max:
                            opt_oh, t_collect_theo = self.traj_optimizer.find_optimal_fm_trajectory(
                                fip, p_flex, current_gn_coord, required_data_per_gn, is_overlapping=True
                            )
                        else:
                            opt_oh = current_gn_coord
                            t_flight = (np.linalg.norm(fip - opt_oh) + np.linalg.norm(p_flex - opt_oh)) / self.uav_speed
                            t_hover = (required_data_per_gn - c_max) / self.traj_optimizer.hover_datarate
                            t_collect_theo = t_flight + t_hover
                        
                        # Physical Constraint
                        phy_dist = np.linalg.norm(opt_oh - fip) + np.linalg.norm(p_flex - opt_oh)
                        t_collect = max(t_collect_theo, phy_dist / self.uav_speed)
                        
                        cost = t_in + t_collect
                        if cost < best_leg_i_cost:
                            best_leg_i_cost = cost
                            best_leg_i_data = {
                                'fip': fip, 'oh': opt_oh, 'fop': p_flex, 
                                'time': cost, 'collect_time': t_collect, 
                                'dist_in': np.linalg.norm(fip - previous_fop), 'dist_col': phy_dist
                            }

                    # B. Optimize Second Leg: P_flex -> OH_next -> FOP_next -> Anchor
                    #    P_flex acts as the FIP for GN i+1 (inside circle)
                    best_leg_next_cost = float('inf')
                    best_leg_next_data = {}

                    for fop in fop_next_candidates:
                        # Find V-shape time. FIP (P_flex) is inside, FOP is on boundary.
                        c_max = self.traj_optimizer.calculate_fm_max_capacity(p_flex, fop, next_gn_coord)
                        
                        if required_data_per_gn <= c_max:
                            opt_oh, t_collect_theo = self.traj_optimizer.find_optimal_fm_trajectory(
                                p_flex, fop, next_gn_coord, required_data_per_gn, is_overlapping=True
                            )
                        else:
                            opt_oh = next_gn_coord
                            t_flight = (np.linalg.norm(p_flex - opt_oh) + np.linalg.norm(fop - opt_oh)) / self.uav_speed
                            t_hover = (required_data_per_gn - c_max) / self.traj_optimizer.hover_datarate
                            t_collect_theo = t_flight + t_hover
                        
                        phy_dist = np.linalg.norm(opt_oh - p_flex) + np.linalg.norm(fop - opt_oh)
                        t_collect = max(t_collect_theo, phy_dist / self.uav_speed)
                        
                        # Flight out to Global Anchor
                        t_out = np.linalg.norm(global_anchor - fop) / self.uav_speed
                        
                        cost = t_collect + t_out
                        if cost < best_leg_next_cost:
                            best_leg_next_cost = cost
                            best_leg_next_data = {
                                'fip': p_flex, 'oh': opt_oh, 'fop': fop,
                                'time': cost, 'collect_time': t_collect, 
                                'dist_col': phy_dist
                            }
                    
                    # C. Combine
                    total_cost = best_leg_i_cost + best_leg_next_cost
                    if total_cost < min_combined_cost:
                        min_combined_cost = total_cost
                        best_combined_config = {
                            'leg_i': best_leg_i_data,
                            'leg_next': best_leg_next_data
                        }

                # 4. Save Results & Update
                if not best_combined_config:
                    print(f"  - CRITICAL WARNING: Strategy 2 failed to find a valid handover for GN {gn_index}. Falling back to Strategy 1 logic.")
                    raise RuntimeError("Strategy 2 optimization failed. Check data requirements or network geometry.")
                
                # Save Leg i
                res_i = best_combined_config['leg_i']
                bob_path_segments.append({
                    'fip': res_i['fip'], 'fop': res_i['fop'], 'oh': res_i['oh'],
                    'gn_index': gn_index, 'service_time': res_i['collect_time'] + res_i['dist_in']/self.uav_speed,
                    'fly_in_dist': res_i['dist_in'], 'collection_dist': res_i['dist_col']
                })
                total_mission_time += res_i['time'] # time = t_in + t_col
                total_path_length += res_i['dist_in'] + res_i['dist_col']

                # Save Leg i+1
                res_next = best_combined_config['leg_next']
                bob_path_segments.append({
                    'fip': res_next['fip'], 'fop': res_next['fop'], 'oh': res_next['oh'],
                    'gn_index': next_gn_index, 'service_time': res_next['collect_time'], 
                    'fly_in_dist': 0.0, # Internal handover, no fly_in
                    'collection_dist': res_next['dist_col']
                })
                # Note: res_next['time'] includes t_out, but we only add collection time here.
                # t_out will be accounted for as t_in for the NEXT leg (or final return).
                total_mission_time += res_next['collect_time']
                total_path_length += res_next['dist_col']

                previous_fop = res_next['fop']
                i += 2 # Skip the next GN as it's already processed

            # --- Strategy 1: Non-Overlapping Case (Standard BOB-V) ---
            else:
                # Determine target anchor
                if i == len(ordered_gn_indices) - 1:
                    next_target_anchor = self.data_center_pos
                else:
                    next_idx = ordered_gn_indices[i+1]
                    next_target_anchor = fip_cmc_anchors.get(next_idx, self.all_gns[next_idx])

                min_local_leg_time = float('inf')
                best_local_config = {}

                sp = previous_fop
                is_sp_inside = np.linalg.norm(sp - current_gn_coord) <= self.comm_radius
                
                angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
                fop_candidates = [current_gn_coord + self.comm_radius * np.array([np.cos(a), np.sin(a)]) for a in angles]
                fip_candidates = [sp] if is_sp_inside else [current_gn_coord + self.comm_radius * np.array([np.cos(a), np.sin(a)]) for a in angles]

                for fip in fip_candidates:
                    flight_time_in = np.linalg.norm(sp - fip) / self.uav_speed
                    for fop in fop_candidates:
                        c_max = self.traj_optimizer.calculate_fm_max_capacity(fip, fop, current_gn_coord)
                        
                        if required_data_per_gn <= c_max:
                            opt_oh, t_col_theo = self.traj_optimizer.find_optimal_fm_trajectory(
                                fip, fop, current_gn_coord, required_data_per_gn, is_overlapping=is_sp_inside
                            )
                        else:
                            opt_oh = current_gn_coord
                            t_flight = (np.linalg.norm(fip - opt_oh) + np.linalg.norm(fop - opt_oh)) / self.uav_speed
                            t_hover = (required_data_per_gn - c_max) / self.traj_optimizer.hover_datarate
                            t_col_theo = t_flight + t_hover
                        
                        phy_dist = np.linalg.norm(opt_oh - fip) + np.linalg.norm(fop - opt_oh)
                        t_collect = max(t_col_theo, phy_dist / self.uav_speed)

                        t_out = np.linalg.norm(next_target_anchor - fop) / self.uav_speed
                        total_leg = flight_time_in + t_collect + t_out

                        if total_leg < min_local_leg_time:
                            min_local_leg_time = total_leg
                            best_local_config = {
                                'fip': fip, 'fop': fop, 'oh': opt_oh,
                                'gn_index': gn_index, 'service_time': flight_time_in + t_collect,
                                'fly_in_dist': np.linalg.norm(sp - fip),
                                'collection_dist': phy_dist
                            }

                print(f"    -> Optimized for GN {gn_index}. Local Leg Time: {best_local_config['service_time']:.2f}s")
                total_mission_time += best_local_config['service_time']
                total_path_length += best_local_config['fly_in_dist'] + best_local_config['collection_dist']
                
                bob_path_segments.append(best_local_config)
                previous_fop = best_local_config['fop']
                i += 1

        # Final return flight
        final_flight_dist = np.linalg.norm(self.data_center_pos - previous_fop)
        final_flight_time = final_flight_dist / self.uav_speed
        
        total_mission_time += final_flight_time
        total_path_length += final_flight_dist

        print("  - BOB-F planning complete.")

        return {
            "segments": bob_path_segments,
            "total_time": total_mission_time,
            "total_length": total_path_length
        }