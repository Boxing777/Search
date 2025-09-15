# main.py
import numpy as np
from config import AREA_WIDTH, AREA_HEIGHT
from altitude_optimizer import find_optimal_altitude
from trajectory_optimizer import stoa_algorithm
from velocity_scheduler import solve_velocity_and_scheduling
from plotting import plot_scenario

def main():
    # --- Scenario Setup ---
    num_gus = 35
    print(f"Simulating for N = {num_gus} Ground Users.")
    # Randomly deploy GUs in the area
    gu_locations = np.random.rand(num_gus, 2) * np.array([AREA_WIDTH, AREA_HEIGHT])

    # --- Step 1: Altitude Optimization ---
    optimal_h, max_d_h = find_optimal_altitude()
    
    # --- Step 2: Trajectory Optimization ---
    # Using STOA algorithm
    final_trajectory, trajectory_length = stoa_algorithm(gu_locations, max_d_h)
    
    # --- Step 3: Velocity and Link Scheduling ---
    min_mission_time = solve_velocity_and_scheduling(final_trajectory, gu_locations, max_d_h)

    # --- Final Results --
    print("\n--- Simulation Finished ---")
    print(f"Optimal Altitude: {optimal_h:.2f} m")
    print(f"Max Transmission Radius: {max_d_h:.2f} m")
    print(f"Optimized Trajectory Length (STOA): {trajectory_length:.2f} m")
    print(f"Minimum Mission Time: {min_mission_time:.2f} s")
    
    # --- Visualization ---
    plot_scenario(gu_locations, final_trajectory, max_d_h)


if __name__ == "__main__":
    main()