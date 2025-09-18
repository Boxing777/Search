# plotting.py (FINAL CLEAN VERSION, NO WAYPOINT MARKERS)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_scenario(gu_locations, trajectory, d_h, detailed_results):
    """
    Visualizes the scenario with the final trajectory, GUs, and ranges,
    removing all text labels and waypoint markers for a clean visualization.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("UAV Data Collection Scenario")
    ax.set_xlabel("X-axis (meters)")
    ax.set_ylabel("Y-axis (meters)")
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    # --- Plot static elements: GUs and their ranges ---
    ax.scatter(gu_locations[:, 0], gu_locations[:, 1], c='blue', marker='o', label='Ground Users (GUs)')
    
    # GU ID labels are removed
    # for i, loc in enumerate(gu_locations):
    #     ax.text(loc[0] + 30, loc[1] + 30, f'GU-{i}', fontsize=9, color='black')
        
    for gu in gu_locations:
        circle = patches.Circle(gu, d_h, color='blue', alpha=0.1, zorder=1)
        ax.add_patch(circle)

    # --- Plot main trajectory line (waypoints) and Start/End markers ---
    if trajectory is not None and len(trajectory) > 0:
        # *** MODIFICATION: Changed 'r-o' to 'r-' to remove the red dots (markers) ***
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=1.5, label='UAV Trajectory') # Note: label updated
        
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'gs', markersize=10, label='Start', zorder=5)
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'kx', markersize=10, mew=3, label='End', zorder=5)
    
    # Update the legend to reflect the change
    handles, labels = ax.get_legend_handles_labels()
    # Manually reorder legend items if needed
    order = [labels.index('Ground Users (GUs)'), labels.index('UAV Trajectory'), labels.index('Start'), labels.index('End')]
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    
    plt.show()