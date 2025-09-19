# ==============================================================================
#                      Main Simulation Execution Script
#
# File Objective:
# This file is the central entry point and master controller for the entire
# simulation. It orchestrates the complete workflow, from setting up the
# environment to running optimization algorithms and visualizing the results.
# ==============================================================================

# --- Step 1: Imports and Setup ---
import time
import numpy as np

# Import custom modules
import parameters as params
from environment import SimulationEnvironment
from mission_allocation_ga import MissionAllocationGA
from trajectory_optimizer import TrajectoryOptimizer
import visualizer as vis

def main():
    """
    Main function to orchestrate the entire simulation workflow.
    """
    # --- Step 2: Initialization ---
    start_time = time.time()
    print("======================================================")
    print("      UAV-Assisted Data Collection Simulation")
    print("======================================================")
    print("\n[Step 1/5] Initializing simulation environment...")

    # Create the simulation environment from parameters
    sim_env = SimulationEnvironment(params)
    print(f"Environment created: {params.AREA_WIDTH}x{params.AREA_HEIGHT}m area with {params.NUM_GNS} GNs.")

    # Define the data requirement for each GN (in bits)
    required_data_per_gn = 1000 * 1e6 # 100 Mbits, as a placeholder
    print(f"Data requirement per GN set to {required_data_per_gn / 1e6:.0f} Mbits.")

    # --- Step 3: Phase 1 - Run Mission Allocation (Genetic Algorithm) ---
    print("\n[Step 2/5] Running Mission Allocation (Genetic Algorithm)...")
    
    # Instantiate and run the GA solver for mission allocation
    ga_solver = MissionAllocationGA(
        gns=sim_env.gn_positions,
        num_uavs=params.NUM_UAVS,
        data_center_pos=sim_env.data_center_pos,
        params=params.__dict__ # Pass params as a dictionary
    )
    ga_results = ga_solver.solve()
    initial_assignment = ga_results['assignment']

    # Visualize the initial, unoptimized routes from the GA
    print("Visualizing initial routes from GA...")
    vis.plot_initial_routes(
        gns=sim_env.gn_positions,
        data_center_pos=sim_env.data_center_pos,
        uav_assignments=initial_assignment,
        area_width=params.AREA_WIDTH,
        area_height=params.AREA_HEIGHT
    )
    
    # --- Step 4: Phase 2 - Run Trajectory Optimization (JOFC) ---
    print("\n[Step 3/5] Running Trajectory Optimization (JOFC) for each UAV...")
    
    # Instantiate the trajectory optimizer
    traj_optimizer = TrajectoryOptimizer(params.__dict__)
    
    # Dictionaries to store the final results
    final_trajectories = {}
    uav_mission_times = {}

    # Main optimization loop: iterate through each UAV and its assigned GNs
    for uav_id, gn_indices_route in initial_assignment.items():
        if not gn_indices_route:
            print(f"{uav_id} has no assigned GNs. Mission time: 0s.")
            final_trajectories[uav_id] = []
            uav_mission_times[uav_id] = 0.0
            continue

        print(f"Optimizing trajectory for {uav_id} serving GNs: {gn_indices_route}")
        
        # Initialize state for this UAV's tour
        previous_fop = sim_env.data_center_pos
        uav_path_segments = []
        current_uav_time = 0.0

        # Inner loop: iterate through each GN in the UAV's assigned route
        num_gns_in_route = len(gn_indices_route)
        for i, gn_index in enumerate(gn_indices_route):
            current_gn_coord = sim_env.gn_positions[gn_index]
            
            # Determine the coordinate of the next stop
            if i < num_gns_in_route - 1:
                next_gn_coord = sim_env.gn_positions[gn_indices_route[i+1]]
            else: # This is the last GN, so the next stop is the data center
                next_gn_coord = sim_env.data_center_pos

            # Call the JOFC optimizer for the current GN
            print(f"  - Optimizing for GN {gn_index}...")
            result = traj_optimizer.run_jofc_for_gn(
                prev_fop=previous_fop,
                current_gn_coord=current_gn_coord,
                next_gn_coord=next_gn_coord, # Pass the true next stop for flight-out calc
                required_data=required_data_per_gn
            )
            
            # Store the optimized segments for this GN
            uav_path_segments.append({'type': 'flight', 'start': previous_fop, 'end': result['fip']})
            uav_path_segments.append({'type': 'collection', **result})
            
            # Update state for the next iteration
            current_uav_time += result['service_time']
            previous_fop = result['fop']

        # After visiting all GNs, add the final leg back to the data center
        final_flight_time = np.linalg.norm(previous_fop - sim_env.data_center_pos) / params.UAV_MAX_SPEED
        current_uav_time += final_flight_time
        uav_path_segments.append({'type': 'flight', 'start': previous_fop, 'end': sim_env.data_center_pos})
        
        # Store the final results for this UAV
        final_trajectories[uav_id] = uav_path_segments
        uav_mission_times[uav_id] = current_uav_time
        print(f"  - Optimization complete for {uav_id}. Total time: {current_uav_time:.2f}s")

    # --- Step 5: Analysis and Output ---
    print("\n[Step 4/5] Analyzing final results...")
    
    # Calculate the overall Mission Completion Time (MCT)
    system_mct = 0.0
    if uav_mission_times:
        system_mct = max(uav_mission_times.values())
        
    end_time = time.time()
    total_execution_time = end_time - start_time
    
    print("\n--- Simulation Summary ---")
    for uav_id, mission_time in uav_mission_times.items():
        print(f"  - {uav_id} Total Mission Time: {mission_time:.2f} seconds")
    print(f"\nSystem Mission Completion Time (MCT): {system_mct:.2f} seconds")
    print(f"Total script execution time: {total_execution_time:.2f} seconds")
    print("--------------------------")
    
    # --- Step 6: Final Visualization ---
    print("\n[Step 5/5] Visualizing final optimized trajectories...")
    vis.plot_final_trajectories(
        gns=sim_env.gn_positions,
        data_center_pos=sim_env.data_center_pos,
        final_trajectories=final_trajectories,
        area_width=params.AREA_WIDTH,
        area_height=params.AREA_HEIGHT,
        comm_radius=traj_optimizer.comm_radius_d
    )
    
    print("\nSimulation finished successfully.")
    print("======================================================")

# --- Script Entry Point ---
if __name__ == "__main__":
    main()