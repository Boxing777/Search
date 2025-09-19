# ==============================================================================
#                      Simulation Environment Generation
#
# File Objective:
# The purpose of this file is to establish the physical simulation world.
# It is responsible for creating and managing the static elements of the
# simulation scenario, primarily the locations of the Ground Nodes (GNs).
# It uses parameters defined in `parameters.py` to generate a reproducible
# and consistent environment for the UAV mission planning algorithms.
# ==============================================================================

# Import necessary libraries
import numpy as np
from typing import Optional, Tuple

# --- Core Functionality ---

def generate_gns(num_gns: int, area_width: float, area_height: float, seed: Optional[int] = None) -> np.ndarray:
    """
    Procedurally generates the 2D coordinates for a specified number of Ground Nodes.

    The placement of these nodes is random and uniformly distributed within a
    defined rectangular area.

    Args:
        num_gns (int): The total number of Ground Nodes to be generated (N).
        area_width (float): The width of the rectangular area for GN placement.
        area_height (float): The height of the rectangular area for GN placement.
        seed (Optional[int], optional): A seed for the random number generator
            to ensure reproducibility. Defaults to None.

    Returns:
        np.ndarray: A NumPy array of shape (num_gns, 2) containing the (x, y)
                    coordinates of all generated GNs.
    """
    # If a seed is provided, initialize the random number generator with it
    # for reproducible results.
    if seed is not None:
        np.random.seed(seed)

    # Generate random x-coordinates uniformly distributed between 0 and area_width.
    x_coords = np.random.uniform(0, area_width, num_gns)

    # Generate random y-coordinates uniformly distributed between 0 and area_height.
    y_coords = np.random.uniform(0, area_height, num_gns)

    # Stack the x and y coordinates into a single (num_gns, 2) array.
    gn_positions = np.column_stack((x_coords, y_coords))

    return gn_positions


# --- Class Structure for Environment Management ---

class SimulationEnvironment:
    """
    A container for all static elements of the simulation world.

    This class is initialized with simulation parameters and holds the state of
    the environment, such as GN positions and the data center location. It can
    be passed to other modules to ensure they operate on a consistent world model.

    Attributes:
        area_width (float): The width of the simulation area.
        area_height (float): The height of the simulation area.
        data_center_pos (np.ndarray): A (1, 2) array for the data center's coordinates.
        gn_positions (np.ndarray): An (N, 2) array storing the coordinates of the N GNs.
    """
    def __init__(self, params):
        """
        Constructs the SimulationEnvironment using parameters from a config object.

        Args:
            params: A module or object containing simulation parameters, expected
                    to have attributes like AREA_WIDTH, AREA_HEIGHT, NUM_GNS,
                    DATA_CENTER_POS, and an optional RANDOM_SEED.
        """
        # Store key environmental dimensions
        self.area_width: float = params.AREA_WIDTH
        self.area_height: float = params.AREA_HEIGHT

        # Store the data center position, ensuring it's a NumPy array for consistency
        self.data_center_pos: np.ndarray = np.array(params.DATA_CENTER_POS)

        # Safely get the random seed from parameters; defaults to None if not present
        # using getattr is a robust way to handle optional parameters.
        seed = getattr(params, 'RANDOM_SEED', None)

        # Generate the Ground Node positions by calling the utility function
        self.gn_positions: np.ndarray = generate_gns(
            num_gns=params.NUM_GNS,
            area_width=self.area_width,
            area_height=self.area_height,
            seed=seed
        )

    def get_gn_position(self, gn_index: int) -> np.ndarray:
        """
        A helper method to retrieve the coordinates of a specific GN by its index.

        Args:
            gn_index (int): The index of the desired Ground Node.

        Returns:
            np.ndarray: A (1, 2) array representing the GN's (x, y) coordinates.
        """
        if 0 <= gn_index < len(self.gn_positions):
            return self.gn_positions[gn_index]
        else:
            raise IndexError(f"GN index {gn_index} is out of bounds.")