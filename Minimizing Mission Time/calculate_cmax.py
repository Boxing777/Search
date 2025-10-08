# ==============================================================================
#                      Benchmark C_max Calculator
#
# File Objective:
# This script calculates the theoretical maximum data capacity for the Flying
# Mode (FM) under the current simulation parameters. This value, referred to
# as C_f_max in the paper, serves as the critical threshold for determining
# when the Hovering Mode (HM) becomes necessary.
# ==============================================================================

import numpy as np
import parameters as params
from trajectory_optimizer import TrajectoryOptimizer

def calculate_benchmark_cmax():
    """
    Calculates and prints the benchmark C_f_max based on current parameters.
    """
    print("======================================================")
    print("    Calculating Benchmark C_f_max (FM Capacity Limit)")
    print("======================================================")

    # 1. Initialize the optimizer to get system parameters and models
    #    Make sure your parameters.py has PATH_LOSS_EXPONENT = 2.0
    try:
        optimizer = TrajectoryOptimizer(params.__dict__)
    except Exception as e:
        print(f"\nError initializing TrajectoryOptimizer: {e}")
        print("Please ensure all your parameter and model files are correct.")
        return

    # 2. Get the calculated communication radius D
    D = optimizer.comm_radius_d
    if D <= 0:
        print("\nError: Calculated communication radius is zero or negative.")
        print("This might be due to incorrect power or channel parameters.")
        return
        
    print(f"\nUsing current parameters:")
    print(f"  - Communication Radius (D): {D:.2f} meters")
    print(f"  - UAV Altitude (H):         {params.UAV_ALTITUDE:.2f} meters")
    print(f"  - Path Loss Exponent (eta): {params.PATH_LOSS_EXPONENT}")

    # 3. Define the standardized path for C_f_max calculation:
    #    This path is a diameter of the communication circle, passing through the GN center.
    #    For this calculation, we can assume the GN is at the origin (0, 0) for simplicity.
    gn_coord = np.array([0.0, 0.0])
    
    # The FIP and FOP are at the opposite ends of a horizontal diameter.
    fip = np.array([-D, 0.0])
    fop = np.array([D, 0.0])
    
    # 4. Call the function that calculates the max capacity for the FIP -> GN -> FOP path
    try:
        benchmark_cmax_bits = optimizer.calculate_fm_max_capacity(fip, fop, gn_coord)
    except Exception as e:
        print(f"\nError during C_f_max calculation: {e}")
        return

    benchmark_cmax_mbits = benchmark_cmax_bits / 1e6

    print("\n--- Calculation Result ---")
    print(f"The theoretical maximum data capacity in FM mode (C_f_max) is: {benchmark_cmax_mbits:.2f} Mbits")
    print("--------------------------")
    
    print("\nThis means:")
    print(f"-> If your 'required_data_per_gn' is LESS THAN OR EQUAL to {benchmark_cmax_mbits:.2f} Mbits,")
    print("   the system will primarily use FM mode, and you will see V-shaped or straight-line trajectories.")
    print(f"-> If your 'required_data_per_gn' is GREATER THAN {benchmark_cmax_mbits:.2f} Mbits,")
    print("   the system will be forced to use HM mode for some (or all) paths,")
    print("   which involves hovering over the GN.")

if __name__ == "__main__":
    calculate_benchmark_cmax()