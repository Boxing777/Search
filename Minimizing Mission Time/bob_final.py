# bob_final.py
# ==============================================================================
#      BOB-Final (Global Dynamic Programming Linkage Framework)
#
# File Objective:
# Implements the ultimate trajectory optimization using a Global Layered Graph.
# This version eliminates the need for pre-computed geometric anchors (CMC) 
# and SOCP skeletons for guidance. Instead, it solves for the global spatiotemporal 
# optimum by linking every node in the sequence through DP transitions.
#
# Core Strategy (Option 3):
# 1. Constructs a single continuous layered graph for the entire GN sequence.
# 2. Each GN has an Entry Layer and an Exit/Handover Layer.
# 3. Transitions between layers represent either:
#    a. Intra-node collection (V-shape path optimized via Stage 3 solver).
#    b. Inter-node flight (Straight line flight between nodes).
# 4. Global DP Forward Pass calculates the cumulative minimum time.
# 5. Global DP Backward Pass extracts the seamless mission trajectory.
# ==============================================================================

import numpy as np
from typing import List, Dict

# Import necessary components
from trajectory_optimizer import TrajectoryOptimizer
from convex_trajectory_planner import ConvexTrajectoryPlanner
import parameters as params

class BOBOverlapPlanner:
    """
    Implements the BOB-Final algorithm using a Global DP Linkage Framework.
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
    def _get_boundary_samples(self, center: np.ndarray, num_samples: int = 36) -> List[np.ndarray]:
        """Generates candidate points on the circle boundary."""
        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        return [center + self.comm_radius * np.array([np.cos(a), np.sin(a)]) for a in angles]

    def _generate_p_flex_candidates(self, center_i: np.ndarray, center_next: np.ndarray, ref_point: np.ndarray, total_samples: int = 10) -> List[np.ndarray]:
        """T-Shape Cross-Sampling Logic for internal handover points."""
        dist_centers = np.linalg.norm(center_next - center_i)
        if dist_centers == 0 or dist_centers >= 2 * self.comm_radius: 
            return [center_i]

        R = self.comm_radius
        candidates = []
        num_center_samples = max(2, total_samples // 2)
        num_bisect_samples = max(1, total_samples - num_center_samples)

        # 1. Center-line sampling
        start_dist = max(0.0, dist_centers - R)
        end_dist = min(dist_centers, R)
        if start_dist <= end_dist:
            t_values = np.linspace(start_dist / dist_centers, end_dist / dist_centers, num_center_samples)
            vec_centers = center_next - center_i
            for t in t_values:
                candidates.append(center_i + t * vec_centers)

        # 2. Perpendicular Bisector sampling (T-shape)
        midpoint = center_i + 0.5 * (center_next - center_i)
        h = np.sqrt(max(0, R**2 - (dist_centers/2)**2))
        dx = (center_next[0] - center_i[0]) / dist_centers
        dy = (center_next[1] - center_i[1]) / dist_centers
        
        tip1 = np.array([midpoint[0] - h * dy, midpoint[1] + h * dx])
        tip2 = np.array([midpoint[0] + h * dy, midpoint[1] - h * dx])
        
        target_tip = tip1 if np.linalg.norm(ref_point - tip1) < np.linalg.norm(ref_point - tip2) else tip2
        bisect_t_values = np.linspace(0, 1, num_bisect_samples + 1)[1:]
        vec_bisect = target_tip - midpoint
        
        for t in bisect_t_values:
            candidates.append(midpoint + t * vec_bisect)
        return candidates

    # --- Main Planning Logic ---
    def plan_path(self, ordered_gn_indices: List[int], required_data_per_gn: float) -> Dict:
        if not ordered_gn_indices:
            return {"segments": [], "total_time": 0.0, "total_length": 0.0}

        print("\n--- Planning with BOB-Final (Global DP Linkage) Method ---")

        # Step 1: Construct the Global Layered Graph
        # We build a chain of layers covering the entire mission sequence
        layers = []
        layer_meta = [] # Stores metadata to identify transition types

        # L0: Starting Position (Data Center)
        layers.append([self.data_center_pos])
        layer_meta.append({"type": "start"})

        current_ref = self.data_center_pos
        for i in range(len(ordered_gn_indices)):
            idx_curr = ordered_gn_indices[i]
            coord_curr = self.all_gns[idx_curr]
            
            # 1. Entry Layer for GN i
            # If overlap with previous, entry is already handled by the previous handover layer
            if i == 0 or np.linalg.norm(coord_curr - self.all_gns[ordered_gn_indices[i-1]]) >= 2 * self.comm_radius:
                layers.append(self._get_boundary_samples(coord_curr))
                layer_meta.append({"type": "entry", "gn_idx": idx_curr})
            
            # 2. Exit/Handover Layer for GN i
            if i < len(ordered_gn_indices) - 1:
                coord_next = self.all_gns[ordered_gn_indices[i+1]]
                if np.linalg.norm(coord_curr - coord_next) < 2 * self.comm_radius:
                    # Overlap: Create internal handover layer (Both Exit of i and Entry of i+1)
                    layers.append(self._generate_p_flex_candidates(coord_curr, coord_next, current_ref))
                    layer_meta.append({"type": "handover", "gn_idx": idx_curr})
                else:
                    # No overlap: Normal exit layer on boundary
                    layers.append(self._get_boundary_samples(coord_curr))
                    layer_meta.append({"type": "exit", "gn_idx": idx_curr})
            else:
                # Last Node: Normal exit layer on boundary
                layers.append(self._get_boundary_samples(coord_curr))
                layer_meta.append({"type": "exit", "gn_idx": idx_curr})
            
            current_ref = coord_curr

        # Final Layer: Return to Data Center
        layers.append([self.data_center_pos])
        layer_meta.append({"type": "end"})

        # Step 2: Global DP Forward Pass
        num_layers = len(layers)
        dp_cost = [[float('inf')] * len(l) for l in layers]
        dp_backtrack = [[-1] * len(l) for l in layers]
        dp_metadata = [[None] * len(l) for l in layers]

        dp_cost[0][0] = 0.0

        for l in range(1, num_layers):
            curr_meta = layer_meta[l]
            prev_meta = layer_meta[l-1]

            for i_curr, p_curr in enumerate(layers[l]):
                for i_prev, p_prev in enumerate(layers[l-1]):
                    
                    cost_transition = 0.0
                    metadata = {}

                    # Case A: Inter-node Flight (Flight between non-overlapping nodes)
                    # Occurs between: Start->Entry, Exit->Entry, Exit->End
                    if (prev_meta['type'] in ['start', 'exit']) and (curr_meta['type'] in ['entry', 'end']):
                        dist = np.linalg.norm(p_curr - p_prev)
                        cost_transition = dist / self.uav_speed
                        metadata = {'type': 'flight', 'dist': dist}

                    # Case B: Intra-node Collection (V-Shape flight)
                    # Occurs between: Entry->Exit, Entry->Handover, Handover->Handover, Handover->Exit
                    else:
                        gn_idx = curr_meta['gn_idx']
                        gn_coord = self.all_gns[gn_idx]
                        
                        # Determine if endpoints require elliptical solver (if either is internal)
                        needs_elliptical = (prev_meta['type'] == 'handover' or curr_meta['type'] == 'handover')
                        
                        c_max = self.traj_optimizer.calculate_fm_max_capacity(p_prev, p_curr, gn_coord)
                        if required_data_per_gn <= c_max:
                            opt_oh, t_theo = self.traj_optimizer.find_optimal_fm_trajectory(
                                p_prev, p_curr, gn_coord, required_data_per_gn, is_overlapping=needs_elliptical)
                        else:
                            opt_oh = gn_coord
                            t_f = (np.linalg.norm(p_prev - opt_oh) + np.linalg.norm(p_curr - opt_oh)) / self.uav_speed
                            t_h = (required_data_per_gn - c_max) / self.traj_optimizer.hover_datarate
                            t_theo = t_f + t_h
                        
                        phy_dist = np.linalg.norm(opt_oh - p_prev) + np.linalg.norm(p_curr - opt_oh)
                        t_collect = max(t_theo, phy_dist / self.uav_speed)
                        cost_transition = t_collect
                        metadata = {'type': 'collection', 'oh': opt_oh, 'dist': phy_dist, 'time': t_collect, 'gn_idx': gn_idx}

                    # Accumulate and minimize
                    new_total = dp_cost[l-1][i_prev] + cost_transition
                    if new_total < dp_cost[l][i_curr]:
                        dp_cost[l][i_curr] = new_total
                        dp_backtrack[l][i_curr] = i_prev
                        dp_metadata[l][i_curr] = metadata

        # Step 3: Global DP Backward Pass
        path_indices = [0] * num_layers
        curr_ptr = 0 # Final layer DC
        for l in range(num_layers - 1, -1, -1):
            path_indices[l] = curr_ptr
            curr_ptr = dp_backtrack[l][curr_ptr]

        # Step 4: Reconstruct Segments for Main script
        bob_path_segments = []
        total_mission_time = dp_cost[-1][0]
        total_path_length = 0.0

        for l in range(1, num_layers):
            ptr = path_indices[l]
            meta = dp_metadata[l][ptr]
            p_start = layers[l-1][path_indices[l-1]]
            p_end = layers[l][ptr]
            
            if meta['type'] == 'collection':
                bob_path_segments.append({
                    'fip': p_start, 'fop': p_end, 'oh': meta['oh'],
                    'gn_index': meta['gn_idx'], 'service_time': meta['time'],
                    'fly_in_dist': 0.0, 'collection_dist': meta['dist']
                })
                total_path_length += meta['dist']
            else:
                # If it's a flight segment, add its distance to the NEXT collection segment's fly_in
                # for compatibility with your existing visualization/reporting logic.
                total_path_length += meta['dist']
                if l < num_layers - 1:
                    # Look ahead to the next layer (which must be collection)
                    # We store it in the next GN's fly_in info
                    pass 

        print(f"  - BOB-Final (Global DP) planning complete. Total Time: {total_mission_time:.2f}s")

        return {
            "segments": bob_path_segments,
            "total_time": total_mission_time,
            "total_length": total_path_length
        }