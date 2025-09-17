# main.py
import numpy as np
from config import AREA_WIDTH, AREA_HEIGHT, START_POS, END_POS
from altitude_optimizer import find_optimal_altitude
from trajectory_optimizer import stoa_algorithm, gtoa_algorithm
from velocity_scheduler import solve_velocity_and_scheduling
from plotting import plot_scenario

def main():
    # --- ALGORITHM & SCENARIO SELECTION ---
    # Choose the trajectory optimization algorithm: 'STOA' or 'GTOA'
    TRAJECTORY_ALGORITHM = 'STOA'
    
    # Choose the distribution mode for Ground Users: 'UNIFORM' or 'NORMAL'
    DISTRIBUTION_MODE = 'NORMAL'

    # --- Scenario Setup ---
    num_gus = 2
    print(f"Simulating for N = {num_gus} Ground Users with {DISTRIBUTION_MODE} distribution.")
    
    # Set a random seed to ensure reproducible results for each run
    # np.random.seed(42)

    # --- Generation of GU Locations based on selected mode ---
    if DISTRIBUTION_MODE == 'UNIFORM':
        print("Generating UNIFORM distribution...")
        gu_locations = np.random.rand(num_gus, 2) * np.array([AREA_WIDTH, AREA_HEIGHT])
    
    elif DISTRIBUTION_MODE == 'NORMAL':
        print("Generating NORMAL (Gaussian) distribution...")
        center_point = np.array([AREA_WIDTH / 2, AREA_HEIGHT / 2])
        spread_std_dev = np.array([AREA_WIDTH / 6, AREA_HEIGHT / 6])
        gu_locations = np.random.normal(
            loc=center_point, 
            scale=spread_std_dev, 
            size=(num_gus, 2)
        )
        gu_locations = np.clip(gu_locations, 0, [AREA_WIDTH, AREA_HEIGHT])
    else:
        raise ValueError("Invalid distribution mode. Choose 'UNIFORM' or 'NORMAL'.")

    # --- Step 1: Altitude Optimization ---
    print("\n--- Step 1: Altitude Optimization ---")
    optimal_h, max_d_h = find_optimal_altitude()
    
    # --- Step 2: Trajectory Optimization ---
    print(f"\n--- Step 2: Trajectory Optimization ({TRAJECTORY_ALGORITHM}) ---")
    if TRAJECTORY_ALGORITHM == 'STOA':
        final_trajectory, trajectory_length = stoa_algorithm(gu_locations, max_d_h)
    elif TRAJECTORY_ALGORITHM == 'GTOA':
        final_trajectory, trajectory_length = gtoa_algorithm(gu_locations, max_d_h)
    else:
        raise ValueError("Invalid algorithm selected. Choose 'STOA' or 'GTOA'.")

    # --- Step 3: Velocity and Link Scheduling ---
    detailed_results = solve_velocity_and_scheduling(final_trajectory, gu_locations, max_d_h)
    time_results = detailed_results["time_stats"]

    # --- Final Results ---
    print("\n" + "="*50)
    print("      UAV Data Collection - Final Simulation Results")
    print("="*50)
    print(f"Scenario:")
    print(f"  - Number of GUs:      {num_gus}")
    print(f"  - Distribution:       {DISTRIBUTION_MODE}")
    print(f"  - Trajectory Method:  {TRAJECTORY_ALGORITHM}")
    print("-" * 50)
    print(f"Optimization Outputs:")
    print(f"  - Optimal Altitude (H*):          {optimal_h:.2f} m")
    print(f"  - Max Communication Radius (D_H*): {max_d_h:.2f} m")
    print(f"  - Optimized Trajectory Length:    {trajectory_length:.2f} m")
    print("-" * 50)
    print(f"Time Analysis:")
    print(f"  - Min. Theoretical Flying Time:   {time_results['min_flying_time']:.2f} s")
    print(f"  - Min. Theoretical Comm. Time:    {time_results['min_communication_time']:.2f} s")
    print(f"  - >>> FINAL MISSION TIME <<<:     {time_results['total_mission_time']:.2f} s")
    print("=" * 50)
    
    # --- Visualization ---
    print("\nDisplaying trajectory plot with detailed analysis...")
    plot_scenario(gu_locations, final_trajectory, max_d_h, detailed_results)


if __name__ == "__main__":
    main()