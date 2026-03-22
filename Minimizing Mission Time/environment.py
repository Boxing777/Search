# ==============================================================================
#                      Simulation Environment Generation
#
# File Objective:
# This file establishes the physical simulation world. It generates Ground Nodes
# (GNs) randomly and uniformly within the area while maintaining specific 
# geometric constraints regarding boundaries, the Data Center, and 
# inter-node spacing.
# ==============================================================================

import numpy as np
from typing import Optional, List

# --- Core Functionality ---

def generate_gns(num_gns: int, area_width: float, area_height: float, 
                   margin: float, data_center_pos: np.ndarray, 
                   seed: Optional[int] = None) -> np.ndarray:
    """
    Generates 2D coordinates for GNs using a random uniform distribution
    subject to three constraints:
    1. Boundary: GNs and their comm radius remain within the simulation area.
    2. Data Center: GNs' comm ranges do not cover the Data Center.
    3. Minimum Spacing: The distance between any two GNs must be >= 0.7 * margin.

    Args:
        num_gns (int): Total number of nodes to generate.
        area_width (float): Width of the simulation area.
        area_height (float): Height of the simulation area.
        margin (float): Communication radius (D).
        data_center_pos (np.ndarray): Coordinates of the Data Center.
        seed (Optional[int]): Seed for reproducibility.

    Returns:
        np.ndarray: Array of shape (num_gns, 2).
    """
    if seed is not None:
        np.random.seed(seed)

    gn_positions: List[np.ndarray] = []
    
    # Maximum attempts to prevent infinite loops in crowded areas
    max_attempts = num_gns * 2000 
    
    # Valid generation bounds (margin ensures the circle is inside the area)
    low_x, high_x = margin, area_width - margin
    low_y, high_y = margin, area_height - margin

    if low_x >= high_x or low_y >= high_y:
        raise ValueError("Margin is too large for the area dimensions.")

    # Rule: Distance between any two nodes must be >= 0.7 * D
    min_inter_node_dist = 0.7 * margin

    for _ in range(max_attempts):
        if len(gn_positions) >= num_gns:
            break

        # Generate a candidate position uniformly
        candidate_pos = np.array([
            np.random.uniform(low_x, high_x),
            np.random.uniform(low_y, high_y)
        ])
        
        # Constraint 1: Check distance to Data Center (Must be > D)
        if np.linalg.norm(candidate_pos - data_center_pos) <= margin:
            continue

        # Constraint 2: Check distance to all existing nodes (Must be >= 0.7 * D)
        is_too_close = False
        for existing_pos in gn_positions:
            if np.linalg.norm(candidate_pos - existing_pos) < min_inter_node_dist:
                is_too_close = True
                break
        
        if not is_too_close:
            gn_positions.append(candidate_pos)

    # Error handling if the area is too crowded to satisfy constraints
    if len(gn_positions) < num_gns:
        raise RuntimeError(f"Failed to generate {num_gns} GNs. Only {len(gn_positions)} placed. "
                           "The area might be too small for the given min_dist constraint.")

    print(f"Successfully generated {num_gns} GNs using random uniform placement.")
    return np.array(gn_positions)


# --- Class Structure for Environment Management ---

class SimulationEnvironment:
    """
    Container for simulation environment data and node management.
    """
    def __init__(self, params, comm_radius: float):
        """
        Args:
            params: Configuration module containing AREA and NUM_GNS.
            comm_radius (float): The communication radius D.
        """
        self.area_width: float = params.AREA_WIDTH
        self.area_height: float = params.AREA_HEIGHT
        self.data_center_pos: np.ndarray = np.array(params.DATA_CENTER_POS)
        
        # Retrieve the random seed from parameters if available
        seed = getattr(params, 'RANDOM_SEED', None)

        # Generate the GN positions using the uniform distribution logic
        self.gn_positions: np.ndarray = generate_gns(
            num_gns=params.NUM_GNS,
            area_width=self.area_width,
            area_height=self.area_height,
            margin=comm_radius,
            data_center_pos=self.data_center_pos,
            seed=seed
        )

    def get_gn_position(self, gn_index: int) -> np.ndarray:
        """
        Retrieves the coordinates of a specific GN by its index.
        """
        if 0 <= gn_index < len(self.gn_positions):
            return self.gn_positions[gn_index]
        else:
            raise IndexError(f"GN index {gn_index} is out of bounds.")