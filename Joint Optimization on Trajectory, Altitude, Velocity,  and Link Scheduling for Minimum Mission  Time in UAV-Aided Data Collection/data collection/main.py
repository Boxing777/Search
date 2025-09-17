# main.py
import numpy as np
import time
from config import AREA_WIDTH, AREA_HEIGHT, START_POS, END_POS
from altitude_optimizer import find_optimal_altitude
# Import both algorithms
from trajectory_optimizer import stoa_algorithm
from trajectory_optimizer_lookahead import stoa_lookahead_algorithm
from velocity_scheduler import solve_velocity_and_scheduling
from plotting import plot_scenario

def main():
    # --- Scenario Setup ---
    num_gus = 20
    print(f"Simulating for N = {num_gus} Ground Users.")
    
    np.random.seed(42) 
    gu_locations = np.random.rand(num_gus, 2) * np.array([AREA_WIDTH, AREA_HEIGHT])

    # --- Step 1: Altitude Optimization (Common for both algorithms) ---
    print("\n--- Step 1: Altitude Optimization ---")
    optimal_h, max_d_h = find_optimal_altitude()
    
    # --- Run and Compare Trajectory Algorithms ---

    # --- Algorithm 1: Original STOA ---
    print("\n" + "="*50)
    print("  Running Original STOA Algorithm")
    print("="*50)
    start_time_stoa = time.time()
    stoa_trajectory, stoa_length = stoa_algorithm(gu_locations, max_d_h)
    end_time_stoa = time.time()
    stoa_runtime = end_time_stoa - start_time_stoa
    
    # --- Algorithm 2: STOA with Lookahead ---
    print("\n" + "="*50)
    print("  Running STOA with Lookahead Algorithm")
    print("="*50)
    start_time_lookahead = time.time()
    # You can change lookahead_k to 2, 3, 4, etc. to see the effect
    lookahead_trajectory, lookahead_length = stoa_lookahead_algorithm(gu_locations, max_d_h, lookahead_k=3)
    end_time_lookahead = time.time()
    lookahead_runtime = end_time_lookahead - start_time_lookahead

    # --- Final Results and Comparison ---
    print("\n" + "="*50)
    print("      ALGORITHM COMPARISON RESULTS")
    print("="*50)
    print(f"Scenario Parameters:")
    print(f"  - Number of Ground Users: {num_gus}")
    print(f"  - Optimal Altitude (H*): {optimal_h:.2f} m")
    print(f"  - Max Comm. Radius (D_H*): {max_d_h:.2f} m")
    print("-" * 50)
    print("Original STOA:")
    print(f"  - Trajectory Length: {stoa_length:.2f} m")
    print(f"  - Runtime:           {stoa_runtime:.2f} s")
    print("-" * 50)
    print("STOA with Lookahead (k=3):")
    print(f"  - Trajectory Length: {lookahead_length:.2f} m")
    print(f"  - Runtime:           {lookahead_runtime:.2f} s")
    print("-" * 50)

    improvement = ((stoa_length - lookahead_length) / stoa_length) * 100 if stoa_length > 0 else 0
    print(f"Path Length Improvement with Lookahead: {improvement:.2f}%")
    
    # --- Final steps with the BETTER trajectory (usually the lookahead one) ---
    if lookahead_length < stoa_length:
        print("\nProceeding with the superior Lookahead trajectory...")
        final_trajectory = lookahead_trajectory
    else:
        print("\nProceeding with the original STOA trajectory...")
        final_trajectory = stoa_trajectory
    
    # --- Step 3: Velocity and Link Scheduling (using the best trajectory) ---
    print("\n--- Step 3: Velocity and Link Scheduling (BCD) ---")
    time_results = solve_velocity_and_scheduling(final_trajectory, gu_locations, max_d_h)
    print(f"\n>>> FINAL MISSION TIME (using best path): {time_results['total_mission_time']:.2f} s <<<")
    
    # --- Visualization ---
    print("\nDisplaying trajectory plot for the Lookahead algorithm...")
    plot_scenario(gu_locations, lookahead_trajectory, max_d_h)
    print("Displaying trajectory plot for the original STOA algorithm...")
    plot_scenario(gu_locations, stoa_trajectory, max_d_h)


if __name__ == "__main__":
    main()