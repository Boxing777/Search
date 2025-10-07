# ==============================================================================
#                      Simulation Environment Generation
#
# File Objective:
# This file establishes the physical simulation world, ensuring that all Ground
# Nodes and their communication ranges are fully contained within the defined area.
# ==============================================================================

# Import necessary libraries
import numpy as np
from typing import Optional, Tuple

# --- Core Functionality ---

def generate_gns(num_gns: int, area_width: float, area_height: float, 
                   margin: float = 0.0, seed: Optional[int] = None) -> np.ndarray:
    """
    Generates 2D coordinates for Ground Nodes within a constrained area.

    The placement is random and uniform, but constrained by a margin to ensure
    that circles of radius 'margin' around the GNs do not exit the main area.

    Args:
        num_gns (int): The total number of Ground Nodes (N).
        area_width (float): The width of the main area.
        area_height (float): The height of the main area.
        margin (float, optional): The safety margin from the edges. Defaults to 0.0.
        seed (Optional[int], optional): A seed for the random number generator.

    Returns:
        np.ndarray: An array of shape (num_gns, 2) with the (x, y) coordinates.
    """
    if seed is not None:
        np.random.seed(seed)

    # <<< MODIFIED: Constrain the generation area by the margin >>>
    # GNs will be generated in [margin, width - margin] and [margin, height - margin]
    # Add a check to prevent negative generation range if margin is too large
    low_x, high_x = margin, area_width - margin
    low_y, high_y = margin, area_height - margin

    if low_x >= high_x or low_y >= high_y:
        raise ValueError(f"Margin ({margin}) is too large for the area dimensions ({area_width}x{area_height}). Cannot generate GNs.")

    x_coords = np.random.uniform(low_x, high_x, num_gns)
    y_coords = np.random.uniform(low_y, high_y, num_gns)

    gn_positions = np.column_stack((x_coords, y_coords))
    return gn_positions


# --- Class Structure for Environment Management ---

class SimulationEnvironment:
    """
    A container for all static elements of the simulation world.
    """
    def __init__(self, params, comm_radius: float): # <<< MODIFIED: Added comm_radius
        """
        Constructs the SimulationEnvironment.

        Args:
            params: A module or object containing simulation parameters.
            comm_radius (float): The communication radius, used as a margin for GN placement.
        """
        self.area_width: float = params.AREA_WIDTH
        self.area_height: float = params.AREA_HEIGHT
        self.data_center_pos: np.ndarray = np.array(params.DATA_CENTER_POS)
        
        seed = getattr(params, 'RANDOM_SEED', None)

        # <<< MODIFICATION: Pass the communication radius as the margin >>>
        print(f"Generating GNs with a safety margin of {comm_radius:.2f} meters...")
        self.gn_positions: np.ndarray = generate_gns(
            num_gns=params.NUM_GNS,
            area_width=self.area_width,
            area_height=self.area_height,
            margin=comm_radius, # Use the radius as the safety margin
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