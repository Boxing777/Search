# plotting.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_scenario(gu_locations, trajectory, d_h):
    """Visualizes the simulation scenario, including GUs, transmission regions, and the UAV trajectory."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot Ground Users (GUs)
    ax.scatter(gu_locations[:, 0], gu_locations[:, 1], c='blue', marker='o', label='Ground Users (GUs)')
    
    # Plot transmission regions for each GU
    for gu in gu_locations:
        circle = patches.Circle(gu, d_h, color='blue', alpha=0.1, zorder=1)
        ax.add_patch(circle)
        
    # Plot the UAV trajectory and its waypoints
    if trajectory is not None and len(trajectory) > 0:
        # The 'o' in 'r-o' will draw small circles at each waypoint (S and E points)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-o', alpha=0.7, markersize=4, label='UAV Trajectory', zorder=2)
        
        # Mark start and end points distinctly
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'gs', markersize=10, label='Start', zorder=3)
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'kx', markersize=10, mew=3, label='End', zorder=3)

        # --- MODIFICATION: The block for annotating S/E text labels has been commented out ---
        # Annotate entry (S) and exit (E) waypoints
        # waypoints = trajectory[1:-1]
        # for i, point in enumerate(waypoints):
        #     # Waypoints are ordered [S1, E1, S2, E2, ...].
        #     # Even indices (0, 2, ...) are entry points (S).
        #     if i % 2 == 0: 
        #         label_num = i // 2 + 1
        #         label = f'S{label_num}'
        #     else: # Odd indices (1, 3, ...) are exit points (E).
        #         label_num = (i - 1) // 2 + 1
        #         label = f'E{label_num}'
            
        #     ax.text(point[0] + 30, point[1] + 30, label, fontsize=9, ha='left', va='bottom',
        #             bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.6))

    ax.set_xlabel("X-axis (meters)")
    ax.set_ylabel("Y-axis (meters)")
    ax.set_title("UAV Data Collection Scenario")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box') 
    plt.show()