# ==============================================================================
#                      Simulation Environment Generation
#
# File Objective:
# This file establishes the physical simulation world, ensuring all GNs and
# their communication ranges are fully contained within the area and do not
# cover the Data Center.
# ==============================================================================

# Import necessary libraries
import numpy as np
from typing import Optional, List

# --- Core Functionality ---

def generate_gns(num_gns: int, area_width: float, area_height: float, 
                   margin: float, data_center_pos: np.ndarray, 
                   seed: Optional[int] = None) -> np.ndarray:
    """
    Generates 2D coordinates for GNs with two constraints:
    1. GNs and their communication range are within the simulation area.
    2. GNs' communication ranges do not cover the data center.

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
    
    # Define the valid generation area, constrained by the margin
    low_x, high_x = margin, area_width - margin
    low_y, high_y = margin, area_height - margin

    if low_x >= high_x or low_y >= high_y:
        raise ValueError(f"Margin ({margin}) is too large for the area dimensions ({area_width}x{area_height}). Cannot generate GNs.")

    # Generate GNs one by one, checking the data center distance constraint
    while len(gn_positions) < num_gns:
        # Generate a candidate position
        candidate_pos = np.array([
            np.random.uniform(low_x, high_x),
            np.random.uniform(low_y, high_y)
        ])
        
        # <<< NEW CONSTRAINT CHECK >>>
        # Calculate the distance from the candidate GN to the data center
        distance_to_dc = np.linalg.norm(candidate_pos - data_center_pos)
        
        # If the distance is greater than the margin (comm_radius), accept the point
        if distance_to_dc > margin:
            gn_positions.append(candidate_pos)
    
    return np.array(gn_positions)


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

        print(f"Generating GNs with a safety margin of {comm_radius:.2f} meters from edges and data center...")
        
        # <<< MODIFICATION: Pass data_center_pos to the generator >>>
        self.gn_positions: np.ndarray = generate_gns(
            num_gns=params.NUM_GNS,
            area_width=self.area_width,
            area_height=self.area_height,
            margin=comm_radius,
            data_center_pos=self.data_center_pos, # Pass data center position
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