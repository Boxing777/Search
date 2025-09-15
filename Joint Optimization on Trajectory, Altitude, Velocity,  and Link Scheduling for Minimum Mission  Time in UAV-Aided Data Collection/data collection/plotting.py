# plotting.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_scenario(gu_locations, trajectory, d_h):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot GUs
    ax.scatter(gu_locations[:, 0], gu_locations[:, 1], c='blue', marker='o', label='Ground Users (GUs)')
    
    # Plot transmission regions
    for gu in gu_locations:
        circle = patches.Circle(gu, d_h, color='blue', alpha=0.1)
        ax.add_patch(circle)
        
    # Plot trajectory
    if trajectory is not None and len(trajectory) > 0:
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-o', markersize=3, label='UAV Trajectory')
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'gs', markersize=8, label='Start')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'kx', markersize=8, label='End')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("UAV-Aided Data Collection Scenario")
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    plt.show()