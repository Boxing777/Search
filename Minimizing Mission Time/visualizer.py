# ==============================================================================
#                      Simulation Results Visualizer
#
# File Objective:
# This file is responsible for all graphical representations of the simulation
# results. It uses Matplotlib to create plots and charts that make the complex
# outputs of the simulation algorithms easy to understand, corresponding to the
# figures in the research paper.
# ==============================================================================

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

# A predefined list of colors for differentiating UAVs in plots.
UAV_COLORS = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']

# --- Core Plotting Functions ---

def plot_gn_environment(ax: plt.Axes, gns: np.ndarray, data_center_pos: Tuple[float, float],
                        area_width: float, area_height: float, comm_radius: float = 0.0):
    """
    Creates a base plot showing the initial setup of the simulation environment.

    This function draws the GNs, data center, and optional communication ranges
    onto a provided Matplotlib Axes object.

    Args:
        ax (plt.Axes): The Matplotlib axes on which to draw.
        gns (np.ndarray): The 2D coordinates of all Ground Nodes.
        data_center_pos (Tuple[float, float]): The 2D coordinates of the data center.
        area_width (float): The width of the simulation area.
        area_height (float): The height of the simulation area.
        comm_radius (float, optional): The communication radius D to draw around each GN.
                                       Defaults to 0.0 (no circle).
    """
    # Set plot limits and aspect ratio
    ax.set_xlim(0, area_width)
    ax.set_ylim(0, area_height)
    ax.set_aspect('equal', adjustable='box')

    # Plot the Data Center
    ax.plot(data_center_pos[0], data_center_pos[1], 'k*', markersize=15, label='Data Center', zorder=3)

    # Plot the Ground Nodes
    ax.plot(gns[:, 0], gns[:, 1], 'bo', markersize=6, label='Ground Nodes', zorder=3)

    # Optionally, draw the communication range around each GN
    if comm_radius > 0:
        for gn_pos in gns:
            circle = plt.Circle(gn_pos, comm_radius, color='gray', linestyle='--', fill=False, alpha=0.5)
            ax.add_artist(circle)
            
    ax.set_xlabel("X-coordinate (meters)")
    ax.set_ylabel("Y-coordinate (meters)")
    ax.grid(True, linestyle=':', alpha=0.6)

def plot_initial_routes(gns: np.ndarray, data_center_pos: Tuple[float, float],
                        uav_assignments: Dict[str, List[int]], area_width: float,
                        area_height: float, title: str = "Initial UAV Routes (MTSP Solution)"):
    """
    Visualizes the initial straight-line paths determined by the Genetic Algorithm.

    This corresponds to Figure 4 in the paper.

    Args:
        gns (np.ndarray): Coordinates of all GNs.
        data_center_pos (Tuple[float, float]): Coordinates of the data center.
        uav_assignments (Dict[str, List[int]]): Output from MissionAllocationGA.
        area_width (float): The width of the simulation area.
        area_height (float): The height of the simulation area.
        title (str, optional): The title for the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw the base environment (without communication radius for clarity)
    plot_gn_environment(ax, gns, data_center_pos, area_width, area_height)

    # Iterate through each UAV's assigned route
    for i, (uav_id, route_indices) in enumerate(uav_assignments.items()):
        if not route_indices:
            continue
            
        color = UAV_COLORS[i % len(UAV_COLORS)]
        
        # Construct the full path of coordinates for this UAV's tour
        path_coords = [data_center_pos] + [gns[idx] for idx in route_indices] + [data_center_pos]
        path_coords = np.array(path_coords)
        
        # Plot the straight-line path
        ax.plot(path_coords[:, 0], path_coords[:, 1], color=color, linestyle='--',
                marker='.', markersize=8, label=f'{uav_id} Path')

    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_final_trajectories(gns: np.ndarray, data_center_pos: Tuple[float, float],
                            final_trajectories: Dict[str, List[Dict]], area_width: float,
                            area_height: float, comm_radius: float,
                            title: str = "Final Optimized Trajectories"):
    """
    Visualizes the final, optimized V-shaped trajectories from the TrajectoryOptimizer.

    This corresponds to Figures 5 and 6 in the paper.

    Args:
        gns (np.ndarray): Coordinates of all GNs.
        data_center_pos (Tuple[float, float]): Coordinates of the data center.
        final_trajectories (Dict[str, List[Dict]]): Detailed path data for each UAV.
        area_width (float): The width of the simulation area.
        area_height (float): The height of the simulation area.
        comm_radius (float): The communication radius D.
        title (str, optional): The title for the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw the base environment with communication radii
    plot_gn_environment(ax, gns, data_center_pos, area_width, area_height, comm_radius)

    # Iterate through each UAV's final trajectory
    for i, (uav_id, segments) in enumerate(final_trajectories.items()):
        color = UAV_COLORS[i % len(UAV_COLORS)]
        
        for j, segment in enumerate(segments):
            # Use a single label for the legend for the entire UAV path
            label = f'{uav_id} Trajectory' if j == 0 else None
            
            if segment['type'] == 'flight':
                start, end = np.array(segment['start']), np.array(segment['end'])
                ax.plot([start[0], end[0]], [start[1], end[1]], color=color,
                        linestyle='-', linewidth=1.5, label=label)
            elif segment['type'] == 'collection':
                fip, oh, fop = np.array(segment['fip']), np.array(segment['oh']), np.array(segment['fop'])
         