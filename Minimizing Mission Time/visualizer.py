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

# --- Helper Function for Drawing Arrows ---

def _add_arrow_to_line(ax: plt.Axes, start: np.ndarray, end: np.ndarray, color: str):
    """
    Adds a direction arrow to a line segment on the given axes.
    The arrow is placed near the middle of the segment.

    Args:
        ax (plt.Axes): The Matplotlib axes to draw on.
        start (np.ndarray): The starting (x, y) coordinate of the line.
        end (np.ndarray): The ending (x, y) coordinate of the line.
        color (str): The color of the arrow.
    """
    # Use annotate to draw an arrow. We shrink it from both ends to place it
    # in the middle of the segment without touching the endpoints.
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            shrinkA=15, # Distance (in points) to shrink from the start
            shrinkB=15, # Distance (in points) to shrink from the end
            linewidth=1
        ),
        zorder=4 # Ensure arrows are drawn on top of lines
    )

# --- Core Plotting Functions ---

def plot_gn_environment(ax: plt.Axes, gns: np.ndarray, data_center_pos: Tuple[float, float],
                        area_width: float, area_height: float, comm_radius: float = 0.0):
    """
    Creates a base plot showing the initial setup of the simulation environment.
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
    Visualizes the initial straight-line paths with direction arrows.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_gn_environment(ax, gns, data_center_pos, area_width, area_height)

    for i, (uav_id, route_indices) in enumerate(uav_assignments.items()):
        if not route_indices:
            continue
            
        color = UAV_COLORS[i % len(UAV_COLORS)]
        path_coords = [data_center_pos] + [gns[idx] for idx in route_indices] + [data_center_pos]
        path_coords = np.array(path_coords)
        
        # Plot the straight-line path
        ax.plot(path_coords[:, 0], path_coords[:, 1], color=color, linestyle='--',
                marker='.', markersize=8, label=f'{uav_id} Path')
        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++ NEW: Add arrows to each segment of the initial path     ++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for j in range(len(path_coords) - 1):
            start_point = path_coords[j]
            end_point = path_coords[j+1]
            _add_arrow_to_line(ax, start_point, end_point, color)

    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_final_trajectories(gns: np.ndarray, data_center_pos: Tuple[float, float],
                            final_trajectories: Dict[str, List[Dict]], area_width: float,
                            area_height: float, comm_radius: float,
                            title: str = "Final Optimized Trajectories"):
    """
    Visualizes the final, optimized V-shaped trajectories with direction arrows.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_gn_environment(ax, gns, data_center_pos, area_width, area_height, comm_radius)

    for i, (uav_id, segments) in enumerate(final_trajectories.items()):
        color = UAV_COLORS[i % len(UAV_COLORS)]
        
        # Use a single label for the legend per UAV
        ax.plot([], [], color=color, linestyle='-', linewidth=2.0, label=f'{uav_id} Trajectory')

        for j, segment in enumerate(segments):
            
            if segment['type'] == 'flight':
                start, end = np.array(segment['start']), np.array(segment['end'])
                ax.plot([start[0], end[0]], [start[1], end[1]], color=color,
                        linestyle='-', linewidth=1.5, zorder=2)
                _add_arrow_to_line(ax, start, end, color)
                        
            elif segment['type'] == 'collection':
                fip, oh, fop = np.array(segment['fip']), np.array(segment['oh']), np.array(segment['fop'])
                v_shape_path = np.array([fip, oh, fop])
                
                ax.plot(v_shape_path[:, 0], v_shape_path[:, 1], color=color,
                        linestyle='-', linewidth=2.5, marker='.', markersize=5, zorder=2)
                
                _add_arrow_to_line(ax, fip, oh, color)
                _add_arrow_to_line(ax, oh, fop, color)

                # <<< NEW SECTION TO INDICATE HOVERING >>>
                # If the mode was HM, draw a filled circle at the hover point (OH)
                if segment.get('mode') == 'HM':
                    ax.plot(oh[0], oh[1], 'o', color=color, markersize=10, 
                            markeredgecolor='black', zorder=5, label=f'{uav_id} Hover Point' if j==0 else "")


    # Create a consolidated legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    ax.set_title(title)
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