# ==============================================================================
#                      Simulation Results Visualizer (MODIFIED FOR COMPARISON)
#
# File Objective:
# This file is responsible for all graphical representations of the simulation
# results. It has been modified to support the comparison of different
# trajectory optimization algorithms on the same plot.
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
                        area_height: float, save_path: str = None, title: str = "Initial UAV Routes (MTSP Solution)"):
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_gn_environment(ax, gns, data_center_pos, area_width, area_height)
    for i, (uav_id, route_indices) in enumerate(uav_assignments.items()):
        if not route_indices: continue
        color = UAV_COLORS[i % len(UAV_COLORS)]
        path_coords = np.array([data_center_pos] + [gns[idx] for idx in route_indices] + [data_center_pos])
        ax.plot(path_coords[:, 0], path_coords[:, 1], color=color, linestyle='--', marker='.', markersize=8, label=f'{uav_id} Path')
        for j in range(len(path_coords) - 1):
            _add_arrow_to_line(ax, path_coords[j], path_coords[j+1], color)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close(fig) # Close the figure to free up memory
    else:
        plt.show()


# <<< RENAMED & MODIFIED FUNCTION TO HANDLE COMPARISON >>>
def plot_final_comparison_trajectories(gns: np.ndarray, data_center_pos: Tuple[float, float],
                                       v_shaped_trajectories: Dict[str, List[Dict]],
                                       convex_trajectories: Dict[str, np.ndarray],
                                       area_width: float, area_height: float, comm_radius: float,
                                       save_path: str = None, title: str = "Final Optimized Trajectories Comparison"):
    fig, ax = plt.subplots(figsize=(14, 14))
    plot_gn_environment(ax, gns, data_center_pos, area_width, area_height, comm_radius)
    for i, (uav_id, segments) in enumerate(v_shaped_trajectories.items()):
        color = UAV_COLORS[i % len(UAV_COLORS)]
        ax.plot([], [], color=color, linestyle='-', linewidth=2.0, label=f'{uav_id} V-Shaped (Time-Optimal)')
        sequence_counter = 1
        for segment in segments:
            if segment['type'] == 'flight':
                start, end = np.array(segment['start']), np.array(segment['end'])
                ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linestyle='-', linewidth=1.5, zorder=2)
                _add_arrow_to_line(ax, start, end, color)
            elif segment['type'] == 'collection':
                fip, oh, fop = np.array(segment['fip']), np.array(segment['oh']), np.array(segment['fop'])
                v_path = np.array([fip, oh, fop])
                ax.plot(v_path[:, 0], v_path[:, 1], color=color, linestyle='-', linewidth=1.5, marker='.', markersize=4, zorder=2)
                ax.text(oh[0] + 50, oh[1] + 50, str(sequence_counter), color='white', 
                        fontsize=10, fontweight='bold', ha='center', va='center',
                        bbox=dict(facecolor=color, alpha=0.8, boxstyle='circle,pad=0.2'))
                if segment.get('mode') == 'HM':
                    ax.plot(oh[0], oh[1], 'o', color=color, markersize=8, markeredgecolor='black')
                sequence_counter += 1
    for i, (uav_id, path) in enumerate(convex_trajectories.items()):
        color = 'darkblue' if i==0 else 'darkgreen'
        if len(path) > 0:
            ax.plot(path[:, 0], path[:, 1], color=color, linestyle='--', linewidth=2.0, marker='x', markersize=6, label=f'{uav_id} Convex (Shortest Path)')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize='large')
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close(fig) # Close the figure to free up memory
    else:
        plt.show()

# Deprecated wrapper for backward compatibility. 
# It now calls the new comparison function.
def plot_final_trajectories(gns: np.ndarray, data_center_pos: Tuple[float, float],
                            final_trajectories: Dict[str, List[Dict]], area_width: float,
                            area_height: float, comm_radius: float,
                            title: str = "Final Optimized Trajectories"):
    print("Warning: plot_final_trajectories is deprecated. Using plot_final_comparison_trajectories instead.")
    plot_final_comparison_trajectories(
        gns=gns,
        data_center_pos=data_center_pos,
        v_shaped_trajectories=final_trajectories, # Assume the old call passes V-shaped data
        convex_trajectories={}, # Pass an empty dict for the convex path
        area_width=area_width,
        area_height=area_height,
        comm_radius=comm_radius,
        title=title
    )
    

def plot_convex_path_details(gns: np.ndarray, data_center_pos: Tuple[float, float],
                             convex_results: Dict, area_width: float, area_height: float,
                             comm_radius: float, save_path: str = None,
                             title: str = "Convex Path Details"):
    """
    Visualizes the detailed convex-optimized path, highlighting So and Eo points.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_gn_environment(ax, gns, data_center_pos, area_width, area_height, comm_radius)

    for i, (uav_id, path) in enumerate(convex_results.items()):
        color = 'darkblue'
        path_np = np.array(path)
        if len(path_np) > 0:
            # Plot the full path
            ax.plot(path_np[:, 0], path_np[:, 1], color=color, linestyle='--', linewidth=1.5,
                    label=f'{uav_id} Convex Path')
            
            # Highlight So and Eo points
            # Path structure is [DC, So_0, Eo_0, So_1, Eo_1, ..., So_N-1, Eo_N-1, DC]
            so_points = path_np[1:-1:2] # Selects So_0, So_1, ...
            eo_points = path_np[2:-1:2] # Selects Eo_0, Eo_1, ...
            
            ax.plot(so_points[:, 0], so_points[:, 1], 'x', color='green', markersize=8, label='Start of Collection (So)')
            ax.plot(eo_points[:, 0], eo_points[:, 1], 'o', color='purple', markersize=6, fillstyle='none', markeredgewidth=2, label='End of Collection (Eo)')

            # Add arrows to show direction
            for j in range(len(path_np) - 1):
                _add_arrow_to_line(ax, path_np[j], path_np[j+1], color)

    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
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
    