# convex_trajectory_planner.py
# ==============================================================================
#            Convex Optimization-Based Trajectory Planner
#
# File Objective:
# Implements the trajectory optimization method described in the paper "Joint
# Optimization on Trajectory, Altitude, Velocity..." by Jiaxun Li et al.
# This approach models the problem of finding the shortest path through a series
# of circular regions as a convex optimization problem (P2.1).
#
# This version takes a PRE-DETERMINED visiting order.
# ==============================================================================

import numpy as np
import cvxpy as cp
from typing import List, Dict

class ConvexTrajectoryPlanner:
    """
    Finds the shortest geometric path for a FIXED sequence of circular communication
    zones using convex optimization.
    """

    def __init__(self, gns: np.ndarray, data_center_pos: np.ndarray, comm_radius: float):
        """
        Initializes the planner with the environment details.

        Args:
            gns (np.ndarray): Array of ALL GN coordinates, shape (N, 2).
            data_center_pos (np.ndarray): The start and end point of the mission.
            comm_radius (float): The communication radius (D_H^*) of each GN.
        """
        self.all_gns = gns
        self.data_center_pos = data_center_pos
        self.comm_radius = comm_radius

    def plan_shortest_path_for_sequence(self, ordered_gn_indices: List[int]) -> Dict:
        """
        Plans the globally optimal shortest path for a given, fixed sequence of GNs.

        Args:
            ordered_gn_indices (List[int]): The pre-determined indices of GNs to visit.

        Returns:
            Dict: A dictionary containing the final path coordinates and total length.
        """
        if not ordered_gn_indices:
            return {"path": np.array([]), "length": 0.0, "collection_segments": []}
        
        print(f"  - Planning shortest path for fixed order: {ordered_gn_indices}")
        
        ordered_gns_coords = self.all_gns[ordered_gn_indices]
        N = len(ordered_gns_coords)

        # Step 1: Define the convex optimization problem using CVXPY
        
        # Variables: So_i (Start of collection) and Eo_i (End of collection) points
        So = cp.Variable((N, 2), name="So_points")
        Eo = cp.Variable((N, 2), name="Eo_points")

        # Objective Function: Minimize the sum of all path segment lengths
        path_len_inside_circles = cp.sum([cp.norm(So[i] - Eo[i]) for i in range(N)])
        path_len_between_circles = cp.sum([cp.norm(Eo[i] - So[i+1]) for i in range(N - 1)])
        path_len_start_leg = cp.norm(self.data_center_pos - So[0])
        path_len_end_leg = cp.norm(Eo[N-1] - self.data_center_pos)
        
        total_path_length = path_len_inside_circles + path_len_between_circles + path_len_start_leg + path_len_end_leg
        
        objective = cp.Minimize(total_path_length)

        # Constraints: So_i and Eo_i must be within their respective GN's circle
        constraints = []
        for i in range(N):
            gn_coord = ordered_gns_coords[i]
            constraints += [
                cp.norm(So[i] - gn_coord) <= self.comm_radius,
                cp.norm(Eo[i] - gn_coord) <= self.comm_radius
            ]

        # Step 2: Solve the problem
        problem = cp.Problem(objective, constraints)
        print("  - Solving the convex optimization problem...")
        problem.solve(solver=cp.ECOS, verbose=False) 

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"  - WARNING: Convex optimization failed. Status: {problem.status}")
            return {"path": np.array([]), "length": 0.0, "collection_segments": []}

        print("  - Convex optimization successful.")
        
        # Step 3: Extract the results
        optimal_so = So.value
        optimal_eo = Eo.value
        
        full_path = [self.data_center_pos]
        for i in range(N):
            full_path.append(optimal_so[i])
            full_path.append(optimal_eo[i])
        full_path.append(self.data_center_pos)
        
        final_path = np.array(full_path)
        final_length = problem.value
        
        collection_segments = []
        for i in range(N):
            collection_segments.append({
                "gn_index": ordered_gn_indices[i],
                "start": optimal_so[i],
                "end": optimal_eo[i]
            })

        return {
            "path": final_path,
            "length": final_length,
            "collection_segments": collection_segments
        }