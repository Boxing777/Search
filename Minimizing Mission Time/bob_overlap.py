# bob_overlap.py
# ==============================================================================
#      BOB-F (BOB with Flexible Handover via Dynamic Programming)
#
# File Objective:
# Implements the ultimate BOB-F strategy capable of handling overlapping chains
# of ANY length (1, 2, 3, 4, etc.) using Dynamic Programming on a Layered Graph.
#
# Core Strategy:
# 1. Identifies continuous overlapping clusters (groups) in the sequence.
# 2. For each group, constructs a layered search space:
#    [SP] -> [FIP Cands] -> [P_flex_1 Cands] -> ... -> [FOP Cands] -> [Anchor]
# 3. Uses a DP forward pass to find the minimum time to reach each node in each layer.
# 4. Uses a DP backward pass to extract the globally optimal continuous trajectory 
#    through the entire overlapping cluster.
# ==============================================================================

import numpy as np
from typing import List, Dict

# Import necessary components
from trajectory_optimizer import TrajectoryOptimizer
from convex_trajectory_planner import ConvexTrajectoryPlanner
import parameters as params

class BOBOverlapPlanner:
    """
    Implements the BOB-F planning algorithm using Dynamic Programming for arbitrary overlaps.
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

    def _generate_p_flex_candidates(self, center_i: np.ndarray, center_next: np.ndarray, sp: np.ndarray, total_samples: int = 10) -> List[np.ndarray]:
        """T-Shape Cross-Sampling Logic"""
        dist_centers = np.linalg.norm(center_next - center_i)
        if dist_centers == 0 or dist_centers >= 2 * self.comm_radius: 
            return [center_i]

        R = self.comm_radius
        candidates = []
        
        num_center_samples = max(2, total_samples // 2)
        num_bisect_samples = max(1, total_samples - num_center_samples)

        # 1. Center-line
        start_dist = max(0.0, dist_centers - R)
        end_dist = min(dist_centers, R)
        
        if start_dist <= end_dist:
            start_ratio = start_dist / dist_centers
            end_ratio = end_dist / dist_centers
            t_values = np.linspace(start_ratio, end_ratio, num_center_samples)
            vec_centers = center_next - center_i
            for t in t_values:
                candidates.append(center_i + t * vec_centers)

        # 2. Bisector (T-shape)
        d = dist_centers
        a = d / 2
        h = np.sqrt(max(0, R**2 - a**2))
        midpoint = center_i + a * (center_next - center_i) / d
        dx = (center_next[0] - center_i[0]) / d
        dy = (center_next[1] - center_i[1]) / d
        
        tip1 = np.array([midpoint[0] - h * dy, midpoint[1] + h * dx])
        tip2 = np.array([midpoint[0] + h * dy, midpoint[1] - h * dx])
        
        dist_sp_tip1 = np.linalg.norm(sp - tip1)
        dist_sp_tip2 = np.linalg.norm(sp - tip2)
        target_tip = tip1 if dist_sp_tip1 < dist_sp_tip2 else tip2
        
        bisect_t_values = np.linspace(0, 1, num_bisect_samples + 1)[1:]
        vec_bisect = target_tip - midpoint
        
        for t in bisect_t_values:
            candidates.append(midpoint + t * vec_bisect)
            
        return candidates

    # --- Main Planning Logic ---
    def plan_path(self, ordered_gn_indices: List[int], data_reqs: Dict[int, float]) -> Dict:
        if not ordered_gn_indices:
            return {"segments": [], "total_time": 0.0, "total_length": 0.0}

        print("\n--- Planning with BOB-F (DP Group Optimization) Method ---")

        # Step 1: Pre-compute Global Anchors (FIP_cmc)
        print("  - Step 1: Calculating Global Anchors (FIP_cmc)...")
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
                    fip_cmc_anchors[gn_index] = path_progress[0][1] 

        # Step 2: Auto-Group Overlapping Nodes
        groups = []
        curr_group = [ordered_gn_indices[0]]
        for i in range(len(ordered_gn_indices) - 1):
            idx_curr = ordered_gn_indices[i]
            idx_next = ordered_gn_indices[i+1]
            dist = np.linalg.norm(self.all_gns[idx_curr] - self.all_gns[idx_next])
            if dist < 2 * self.comm_radius:
                curr_group.append(idx_next)
            else:
                groups.append(curr_group)
                curr_group = [idx_next]
        groups.append(curr_group)

        # Step 3: DP Optimization per Group
        print(f"  - Step 2: Dynamic Programming on {len(groups)} overlapping clusters...")
        
        bob_path_segments = []
        total_mission_time = 0.0
        total_path_length = 0.0
        previous_fop = self.data_center_pos

        for group_idx, group in enumerate(groups):
            print(f"    -> Optimizing Group {group_idx+1}: Nodes {group}")
            
            # Determine Global Anchor for this group
            if group_idx == len(groups) - 1:
                global_anchor = self.data_center_pos
            else:
                next_group_first_gn = groups[group_idx + 1][0]
                #global_anchor = self.all_gns[next_group_first_gn] # next node center for anchor
                global_anchor = fip_cmc_anchors.get(next_group_first_gn, self.all_gns[next_group_first_gn]) # ideal_entry_point for anchor

            # Construct Layered Graph
            layers = []
            layers.append([previous_fop]) # Layer 0: SP

            # Layer 1: FIPs of the first GN in the group
            first_gn_coord = self.all_gns[group[0]]
            is_sp_inside = np.linalg.norm(previous_fop - first_gn_coord) <= self.comm_radius
            angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
            if is_sp_inside:
                layers.append([previous_fop])
            else:
                layers.append([first_gn_coord + self.comm_radius * np.array([np.cos(a), np.sin(a)]) for a in angles])

            # Layer 2 to N: P_flex switching points inside overlaps
            for k in range(len(group) - 1):
                c_curr = self.all_gns[group[k]]
                c_next = self.all_gns[group[k+1]]
                # For T-shape orientation, use the previous node's center as a rough reference
                ref_sp = previous_fop if k == 0 else self.all_gns[group[k-1]]
                p_flex_cands = self._generate_p_flex_candidates(c_curr, c_next, ref_sp, total_samples=10)
                if not p_flex_cands: # Fallback safety
                    p_flex_cands = [c_curr + 0.5 * (c_next - c_curr)]
                layers.append(p_flex_cands)

            # Layer N+1: FOPs of the last GN in the group
            last_gn_coord = self.all_gns[group[-1]]
            layers.append([last_gn_coord + self.comm_radius * np.array([np.cos(a), np.sin(a)]) for a in angles])

            # Layer N+2: Global Anchor
            layers.append([global_anchor])

            # Initialize DP Tables
            num_layers = len(layers)
            dp_cost = [[float('inf')] * len(layer) for layer in layers]
            dp_backtrack = [[-1] * len(layer) for layer in layers]
            dp_data = [[None] * len(layer) for layer in layers]
            
            dp_cost[0][0] = 0.0 # Cost to reach SP is 0

            # Forward Pass: Compute minimum cost transitions
            for l in range(1, num_layers):
                for i_curr, p_curr in enumerate(layers[l]):
                    for i_prev, p_prev in enumerate(layers[l-1]):
                        cost_transition = 0.0
                        seg_info = {}

                        if l == 1: # Fly-in leg
                            cost_transition = np.linalg.norm(p_curr - p_prev) / self.uav_speed
                            seg_info = {'fly_in_dist': np.linalg.norm(p_curr - p_prev)}
                        
                        elif l == num_layers - 1: # Fly-out leg
                            cost_transition = np.linalg.norm(p_curr - p_prev) / self.uav_speed
                            # We don't store fly_out_dist as a segment, just use it for cost
                            
                        else: # Collection leg within GN
                            gn_idx = group[l - 2]
                            gn_coord = self.all_gns[gn_idx]
                            
                            req_data_i = data_reqs[gn_idx]
                            
                            dist_prev = np.linalg.norm(p_prev - gn_coord)
                            dist_curr = np.linalg.norm(p_curr - gn_coord)
                            
                            use_elliptical_solver = not (np.isclose(dist_prev, self.comm_radius, atol=1e-2) and 
                                                         np.isclose(dist_curr, self.comm_radius, atol=1e-2))
                            
                            c_max = self.traj_optimizer.calculate_fm_max_capacity(p_prev, p_curr, gn_coord)
                            if req_data_i <= c_max:
                                opt_oh, t_col_theo = self.traj_optimizer.find_optimal_fm_trajectory(
                                    p_prev, p_curr, gn_coord, req_data_i, is_overlapping=use_elliptical_solver)
                            else:
                                opt_oh = gn_coord
                                t_flight = (np.linalg.norm(p_prev - opt_oh) + np.linalg.norm(p_curr - opt_oh)) / self.uav_speed
                                t_hover = (req_data_i - c_max) / self.traj_optimizer.hover_datarate
                                t_col_theo = t_flight + t_hover
                            
                            phy_dist = np.linalg.norm(opt_oh - p_prev) + np.linalg.norm(p_curr - opt_oh)
                            t_collect = max(t_col_theo, phy_dist / self.uav_speed)
                            
                            cost_transition = t_collect
                            seg_info = {'oh': opt_oh, 'collect_time': t_collect, 'collection_dist': phy_dist}

                        total_cost = dp_cost[l-1][i_prev] + cost_transition
                        if total_cost < dp_cost[l][i_curr]:
                            dp_cost[l][i_curr] = total_cost
                            dp_backtrack[l][i_curr] = i_prev
                            dp_data[l][i_curr] = seg_info

            # Backward Pass: Reconstruct optimal trajectory
            curr_idx = 0 # End point is Anchor (only 1 option)
            path_indices = [0] * num_layers
            
            for l in range(num_layers - 1, -1, -1):
                path_indices[l] = curr_idx
                curr_idx = dp_backtrack[l][curr_idx]

            # Record segments for the group
            fly_in_dist = dp_data[1][path_indices[1]]['fly_in_dist']
            
            for k, gn_index in enumerate(group):
                l = k + 2 # Collection layer index
                p_start = layers[l-1][path_indices[l-1]]
                p_end = layers[l][path_indices[l]]
                data = dp_data[l][path_indices[l]]
                
                leg_service_time = data['collect_time'] + (fly_in_dist / self.uav_speed if k == 0 else 0.0)
                
                bob_path_segments.append({
                    'fip': p_start,
                    'fop': p_end,
                    'oh': data['oh'],
                    'gn_index': gn_index,
                    'service_time': leg_service_time,
                    'fly_in_dist': fly_in_dist if k == 0 else 0.0,
                    'collection_dist': data['collection_dist']
                })
                
                total_mission_time += leg_service_time
                total_path_length += data['collection_dist'] + (fly_in_dist if k == 0 else 0.0)

            # Update SP for the next group
            previous_fop = layers[-2][path_indices[-2]]

        # Final return flight
        final_flight_dist = np.linalg.norm(self.data_center_pos - previous_fop)
        final_flight_time = final_flight_dist / self.uav_speed
        
        total_mission_time += final_flight_time
        total_path_length += final_flight_dist

        print("  - BOB-F (DP) planning complete.")

        return {
            "segments": bob_path_segments,
            "total_time": total_mission_time,
            "total_length": total_path_length
        }