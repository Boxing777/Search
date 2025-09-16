# plotting.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_scenario(gu_locations, trajectory, d_h):
    # Enlarge the figure size slightly to make labels clearer
    fig, ax = plt.subplots(figsize=(12, 12)) 
    
    # Plot GUs
    ax.scatter(gu_locations[:, 0], gu_locations[:, 1], c='blue', marker='o', label='Ground Users (GUs)')
    
    # Plot transmission regions
    for gu in gu_locations:
        circle = patches.Circle(gu, d_h, color='blue', alpha=0.1, zorder=1)
        ax.add_patch(circle)
        
    # Plot trajectory and waypoints
    if trajectory is not None and len(trajectory) > 0:
        # Plot the trajectory line
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.8, label='UAV Trajectory', zorder=2)
        
        # Plot the start and end points
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'gs', markersize=10, label='Start', zorder=3)
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'kx', markersize=10, mew=3, label='End', zorder=3)

        # --- New Feature: Annotate entry and exit points ---
        # The structure of trajectory points is [Start, S1, E1, S2, E2, ..., End]
        # We need to annotate the points from index 1 to the second to last.
        waypoints = trajectory[1:-1]
        
        for i, point in enumerate(waypoints):
            # Determine if it's an S (entry) or E (exit) point based on index parity
            # i = 0 -> S1
            # i = 1 -> E1
            # i = 2 -> S2
            # ...
            if i % 2 == 0: # Even indices are entry points (S)
                label_num = i // 2 + 1
                label = f'S{label_num}'
                # Mark S points as green triangles
                ax.plot(point[0], point[1], 'g^', markersize=8, zorder=3)
            else: # Odd indices are exit points (E)
                label_num = (i - 1) // 2 + 1
                label = f'E{label_num}'
                # Mark E points as orange downward-facing triangles
                ax.plot(point[0], point[1], 'v', color='orange', markersize=8, zorder=3)
            
            # Add a text label next to the point
            ax.text(point[0] + 20, point[1] + 20, label, fontsize=9, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.5))

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("UAV-Aided Data Collection Scenario with Waypoints")
    ax.legend()
    ax.grid(True)
    # Ensure the X and Y axes have the same scale, so circles look like circles
    ax.set_aspect('equal', adjustable='box') 
    plt.show()