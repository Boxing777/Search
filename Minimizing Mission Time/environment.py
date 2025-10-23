# ==============================================================================
#                      Simulation Environment Generation
#
# File Objective:
# This file establishes the physical simulation world, ensuring all GNs and
# their communication ranges are fully contained within the area, do not
# cover the Data Center, and maintain a minimum inter-node distance.
# ==============================================================================

# Import necessary libraries
import numpy as np
from typing import Optional, List

# --- Core Functionality ---

def generate_gns(num_gns: int, area_width: float, area_height: float, 
                   margin: float, data_center_pos: np.ndarray, 
                   seed: Optional[int] = None) -> np.ndarray:
    """
    Generates 2D coordinates for GNs with three constraints:
    1. GNs and their comm range are within the simulation area (margin from edge).
    2. GNs' comm ranges do not cover the data center (margin from DC).
    3. The distance between any two GNs is greater than 0.8 * margin.

    Args:
        num_gns (int): The total number of Ground Nodes (N).
        area_width (float): The width of the main area.
        area_height (float): The height of the main area.
        margin (float): The safety margin from edges and data center (comm_radius).
        data_center_pos (np.ndarray): The coordinates of the data center.
        seed (Optional[int], optional): A seed for the random number generator.

    Returns:
        np.ndarray: An array of shape (num_gns, 2) with the (x, y) coordinates.
    """
    if seed is not None:
        np.random.seed(seed)

    gn_positions: List[np.ndarray] = []
    
    # Add a max_attempts counter for robustness to prevent infinite loops.
    max_attempts = num_gns * 1000 
    
    # Define the valid generation area, constrained by the margin
    low_x, high_x = margin, area_width - margin
    low_y, high_y = margin, area_height - margin

    if low_x >= high_x or low_y >= high_y:
        raise ValueError(f"Margin ({margin}) is too large for the area dimensions ({area_width}x{area_height}). Cannot generate GNs.")
        
    # <<< MODIFICATION START: The entire generation loop is modified >>>
    # Define the minimum inter-node distance based on the new rule.
    min_dist = 0.7 * margin

    # Loop with max_attempts to generate nodes one by one with all checks.
    for _ in range(max_attempts):
        # Generate a candidate position within the allowed boundaries.
        candidate_pos = np.array([
            np.random.uniform(low_x, high_x),
            np.random.uniform(low_y, high_y)
        ])
        
        # Constraint 1: Check distance to data center. Must be > margin (D).
        if np.linalg.norm(candidate_pos - data_center_pos) <= margin:
            continue # Too close to data center, generate a new candidate.

        # Constraint 2: Check distance to already placed GNs.
        is_valid_spacing = True
        for existing_pos in gn_positions:
            # Must be > min_dist (0.8 * D).
            if np.linalg.norm(candidate_pos - existing_pos) < min_dist:
                is_valid_spacing = False
                break # Too close to another GN, generate a new candidate.
        
        # If both spacing constraints are met, accept the point.
        if is_valid_spacing:
            gn_positions.append(candidate_pos)
            # If we have generated enough GNs, exit the loop.
            if len(gn_positions) == num_gns:
                print(f"Successfully generated {num_gns} GNs with min inter-node distance > {min_dist:.2f}m.")
                return np.array(gn_positions)

    # If the loop finishes without generating enough GNs, it means it's too difficult.
    raise RuntimeError(f"Failed to generate {num_gns} GNs after {max_attempts} attempts. "
                       f"The area might be too small or the number of GNs too large "
                       f"for the given constraints (D={margin:.2f}m, min_dist={min_dist:.2f}m).")
    # <<< MODIFICATION END >>>


# --- Class Structure for Environment Management ---

class SimulationEnvironment:
    """
    A container for all static elements of the simulation world.
    """
    def __init__(self, params, comm_radius: float):
        """
        Constructs the SimulationEnvironment.

        Args:
            params: A module or object containing simulation parameters.
            comm_radius (float): The communication radius, used as a margin.
        """
        self.area_width: float = params.AREA_WIDTH
        self.area_height: float = params.AREA_HEIGHT
        self.data_center_pos: np.ndarray = np.array(params.DATA_CENTER_POS)
        
        seed = getattr(params, 'RANDOM_SEED', None)

        # <<< MODIFICATION: Updated the print statement for clarity >>>
        print(f"Generating GNs with margin D={comm_radius:.2f}m from edges/DC and min inter-node dist > {0.8*comm_radius:.2f}m.")
        
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
        