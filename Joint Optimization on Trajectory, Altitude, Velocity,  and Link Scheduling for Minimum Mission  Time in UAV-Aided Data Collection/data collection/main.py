# main.py
import numpy as np
from config import AREA_WIDTH, AREA_HEIGHT, START_POS, END_POS
from altitude_optimizer import find_optimal_altitude
from trajectory_optimizer import stoa_algorithm
from velocity_scheduler import solve_velocity_and_scheduling
from plotting import plot_scenario

def main():
    # --- Scenario Setup ---
    num_gus = 20
    print(f"Simulating for N = {num_gus} Ground Users.")
    
    # Set a random seed to ensure reproducible results for each run.
    np.random.seed(42) 
    gu_locations = np.random.rand(num_gus, 2) * np.array([AREA_WIDTH, AREA_HEIGHT])

    # --- Step 1: Altitude Optimization ---
    print("\n--- Step 1: Altitude Optimization ---")
    optimal_h, max_d_h = find_optimal_altitude()
    
    # --- Step 2: Trajectory Optimization ---
    print("\n--- Step 2: Trajectory Optimization (STOA) ---")
    final_trajectory, trajectory_length = stoa_algorithm(gu_locations, max_d_h)
    
    # --- Step 3: Velocity and Link Scheduling ---
    print("\n--- Step 3: Velocity and Link Scheduling (BCD) ---")
    time_results = solve_velocity_and_scheduling(final_trajectory, gu_locations, max_d_h)

    # --- Final Results ---
    print("\n" + "="*45)
    print("      UAV Data Collection - Final Results")
    print("="*45)
    print(f"Scenario Parameters:")
    print(f"  - Number of Ground Users: {num_gus}")
    print(f"  - Start Position: {np.array2string(START_POS, precision=1)}")
    print(f"  - End Position:   {np.array2string(END_POS, precision=1)}")
    print("-" * 45)
    print(f"Optimization Outputs:")
    print(f"  - Optimal Altitude (H*):          {optimal_h:.2f} m")
    print(f"  - Max Communication Radius (D_H*): {max_d_h:.2f} m")
    print(f"  - Optimized Trajectory Length:    {trajectory_length:.2f} m")
    print("-" * 45)
    print(f"Time Analysis:")
    print(f"  - Min. Theoretical Flying Time:   {time_results['min_flying_time']:.2f} s (at max speed)")
    print(f"  - Min. Theoretical Comm. Time:    {time_results['min_communication_time']:.2f} s")
    print(f"  - >>> FINAL MISSION TIME <<<:     {time_results['total_mission_time']:.2f} s")
    print("=" * 45 + "\n")
    
    # --- Visualization ---
    print("Displaying trajectory plot...")
    plot_scenario(gu_locations, final_trajectory, max_d_h)


if __name__ == "__main__":
    main()