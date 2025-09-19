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
    ax.set_xlim(0, area_width)
    ax.set_ylim(0, area_height)
    ax.set_aspect('equal', adjustable='box')
    ax.plot(data_center_pos[0], data_center_pos[1], 'k*', markersize=15, label='Data Center', zorder=3)
    ax.plot(gns[:, 0], gns[:, 1], 'bo', markersize=6, label='Ground Nodes', zorder=3)

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
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_gn_environment(ax, gns, data_center_pos, area_width, area_height)

    for i, (uav_id, route_indices) in enumerate(uav_assignments.items()):
        if not route_indices:
            continue
            
        color = UAV_COLORS[i % len(UAV_COLORS)]
        path_coords = [data_center_pos] + [gns[idx] for idx in route_indices] + [data_center_pos]
        path_coords = np.array(path_coords)
        
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
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_gn_environment(ax, gns, data_center_pos, area_width, area_height, comm_radius)

    for i, (uav_id, segments) in enumerate(final_trajectories.items()):
        color = UAV_COLORS[i % len(UAV_COLORS)]
        
        for j, segment in enumerate(segments):
            label = f'{uav_id} Trajectory' if j == 0 else None
            
            if segment['type'] == 'flight':
                start, end = np.array(segment['start']), np.array(segment['end'])
                ax.plot([start[0], end[0]], [start[1], end[1]], color=color,
                        linestyle='-', linewidth=1.5, label=label)
                        
            elif segment['type'] == 'collection':
                fip, oh, fop = np.array(segment['fip']), np.array(segment['oh']), np.array(segment['fop'])
                
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # ++ CODE COMPLETED BASED ON YOUR REVIEW AND SUGGESTIONS ++
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                
                # Combine the points to form the V-shaped path: [fip, oh, fop]
                v_shape_path = np.array([fip, oh, fop])
                
                # Plot the V-shaped path with a thicker line to highlight it
                ax.plot(v_shape_path[:, 0], v_shape_path[:, 1], color=color,
                        linestyle='-', linewidth=2.5, marker='.', markersize=5, label=label)

    # Final plot adjustments
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_performance_curve(x_data: Dict[str, List], y_data: Dict[str, List],
                           x_label: str, y_label: str, title: str):
    """
    Plots a 2D performance graph comparing multiple data series.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ['o', 's', '^', 'D', 'v', '*']
    
    for i, series_name in enumerate(x_data.keys()):
        if series_name in y_data:
            ax.plot(x_data[series_name], y_data[series_name],
                    marker=markers[i % len(markers)],
                    linestyle='-',
                    label=series_name)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.show()