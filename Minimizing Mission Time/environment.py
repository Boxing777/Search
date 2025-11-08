# environment.py 
import numpy as np
from typing import Optional, List

# --- Core Functionality ---

def generate_gns(num_gns: int, area_width: float, area_height: float, 
                   margin: float, data_center_pos: np.ndarray, 
                   seed: Optional[int] = None) -> np.ndarray:
    """
    Generates 2D coordinates for GNs in "overlapping pairs".
    Each pair consists of two GNs whose communication zones overlap.
    Different pairs are guaranteed to not overlap with each other.

    Args:
        num_gns (int): The total number of GNs (MUST BE AN EVEN NUMBER).
        area_width (float): The width of the main area.
        area_height (float): The height of the main area.
        margin (float): The safety margin (comm_radius, D).
        data_center_pos (np.ndarray): The coordinates of the data center.
        seed (Optional[int], optional): A seed for the random number generator.

    Returns:
        np.ndarray: An array of shape (num_gns, 2) with the (x, y) coordinates.
    """
    if num_gns % 2 != 0:
        raise ValueError("num_gns must be an even number for paired generation.")
        
    if seed is not None:
        np.random.seed(seed)

    gn_positions: List[np.ndarray] = []
    num_pairs = num_gns // 2
    
    max_attempts_total = num_pairs * 1000
    
    # Define the valid generation area for anchor points
    low_x, high_x = margin, area_width - margin
    low_y, high_y = margin, area_height - margin

    if low_x >= high_x or low_y >= high_y:
        raise ValueError(f"Margin ({margin}) is too large for the area dimensions.")

    for _ in range(max_attempts_total):
        if len(gn_positions) >= num_gns:
            break

        # --- Step 1: Generate a valid "Anchor" GN for the new pair ---
        anchor_candidate = None
        for _ in range(100): # Attempts to find a valid anchor
            pos = np.array([np.random.uniform(low_x, high_x), np.random.uniform(low_y, high_y)])
            
            if np.linalg.norm(pos - data_center_pos) <= margin:
                continue

            is_far_from_others = True
            for existing_pos in gn_positions:
                if np.linalg.norm(pos - existing_pos) < 2 * margin:
                    is_far_from_others = False
                    break
            
            if is_far_from_others:
                anchor_candidate = pos
                break
        
        if anchor_candidate is None:
            continue

        # --- Step 2: Generate a "Partner" GN that overlaps with the anchor ---
        partner_candidate = None
        for _ in range(100): # Attempts to find a valid partner
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0.7 * margin, 1.8 * margin)
            
            offset = np.array([np.cos(angle), np.sin(angle)]) * dist
            pos = anchor_candidate + offset

            if not (low_x <= pos[0] <= high_x and low_y <= pos[1] <= high_y):
                continue

            if np.linalg.norm(pos - data_center_pos) <= margin:
                continue
            
            is_far_from_others = True
            for existing_pos in gn_positions:
                if np.linalg.norm(pos - existing_pos) < 2 * margin:
                    is_far_from_others = False
                    break
            
            if is_far_from_others:
                partner_candidate = pos
                break
        
        if partner_candidate is not None:
            gn_positions.append(anchor_candidate)
            gn_positions.append(partner_candidate)
            print(f"Generated pair {len(gn_positions)//2}/{num_pairs}. "
                  f"Distance: {np.linalg.norm(anchor_candidate - partner_candidate):.2f}m")

    if len(gn_positions) < num_gns:
        raise RuntimeError(f"Failed to generate {num_gns} GNs in pairs after {max_attempts_total} attempts.")

    print(f"Successfully generated {len(gn_positions)} GNs in {num_pairs} overlapping pairs.")
    return np.array(gn_positions)


# +++ MISSING CLASS DEFINITION - ADD THIS BACK +++
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

        print(f"Generating GNs in pairs with safety margin D={comm_radius:.2f}m.")
        
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