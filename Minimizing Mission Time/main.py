# ==============================================================================
#                      Main Simulation Execution Script (CORRECTED)
#
# File Objective:
# This file is the central entry point and master controller for the entire
# simulation. This corrected version implements the proper "look-ahead"
# optimization loop in the main function to prevent suboptimal, crossing
# trajectories, faithfully representing the paper's joint optimization model.
# ==============================================================================

# --- Step 1: Imports and Setup ---
import time
import numpy as np

# Import custom modules
import parameters as params
from environment import SimulationEnvironment
from mission_allocation_ga import MissionAllocationGA
from trajectory_optimizer import TrajectoryOptimizer
from convex_trajectory_planner import ConvexTrajectoryPlanner
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
    # Set a high value to test the Hovering Mode (HM)
    required_data_per_gn = 100 * 1e6 # 100 Mbits. Change this to see different behaviors.
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
    
    # Instantiate the trajectory optimizer to use its helper functions
    traj_optimizer = TrajectoryOptimizer(params.__dict__)
    
  
    convex_planner = ConvexTrajectoryPlanner(
        gns=sim_env.gn_positions,
        data_center_pos=sim_env.data_center_pos,
        comm_radius=traj_optimizer.comm_radius_d
    )
    
    # Dictionaries to store the final results from both methods
    final_trajectories = {} 
    # Dictionaries to store the final results
    final_trajectories = {}
    uav_mission_times = {}
    convex_trajectories = {}
    convex_path_lengths = {}

    # Main optimization loop: iterate through each UAV and its assigned GNs
    for uav_id, gn_indices_route in initial_assignment.items():
        if not gn_indices_route:
            print(f"{uav_id} has no assigned GNs. Mission time: 0s.")
            final_trajectories[uav_id] = []
            uav_mission_times[uav_id] = 0.0
            convex_trajectories[uav_id] = np.array([])
            convex_path_lengths[uav_id] = 0.0
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
            
            # Determine the coordinate of the next stop for "look-ahead" calculation
            if i < num_gns_in_route - 1:
                next_stop_coord = sim_env.gn_positions[gn_indices_route[i+1]]
            else: 
                next_stop_coord = sim_env.data_center_pos

            # =========================================================================
            # +++ CORE LOGIC CORRECTION: Implement the full P5 optimization in main +++
            # This loop finds the best (FIP, FOP) pair for the current GN by 
            # minimizing the total time for the entire leg: 
            # time(prev_fop -> FIP) + time_collection + time(FOP -> next_stop_coord)
            # =========================================================================
            print(f"  - Optimizing for GN {gn_index} (considering next stop at {np.round(next_stop_coord, 2)})...")
            
            min_total_leg_time = float('inf')
            best_result_for_leg = {}

            # Discretize the circle around the current GN to find candidate FIPs and FOPs
            num_angles = 16
            angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
            candidate_points = np.array([
                current_gn_coord + traj_optimizer.comm_radius_d * np.array([np.cos(theta), np.sin(theta)])
                for theta in angles
            ])
            
            for fip in candidate_points:
                for fop in candidate_points:
                    # --- Calculate the three components of time for this (FIP, FOP) pair ---
                    
                    # 1. Flight-in time
                    flight_time_in = np.linalg.norm(previous_fop - fip) / params.UAV_MAX_SPEED
                    
                    # 2. Collection time (using the helpers from TrajectoryOptimizer)
                    c_f_max = traj_optimizer._calculate_fm_max_throughput(fip, fop, current_gn_coord)
                    
                    if required_data_per_gn <= c_f_max:
                        mode = 'FM'
                        optimal_oh, collection_time = traj_optimizer._find_optimal_oh_for_fm(fip, fop, current_gn_coord, required_data_per_gn)
                    else:
                        mode = 'HM'
                        optimal_oh = current_gn_coord
                        collection_flight_time = (np.linalg.norm(fip - optimal_oh) + np.linalg.norm(optimal_oh - fop)) / params.UAV_MAX_SPEED
                        hover_data_needed = required_data_per_gn - c_f_max
                        hover_time = hover_data_needed / traj_optimizer.hover_datarate if traj_optimizer.hover_datarate > 0 else float('inf')
                        collection_time = collection_flight_time + hover_time
                    
                    # 3. Flight-out time (This is the crucial "look-ahead" part)
                    flight_time_out = np.linalg.norm(fop - next_stop_coord) / params.UAV_MAX_SPEED
                    
                    # --- Total time for this entire leg (the value to be minimized) ---
                    total_leg_time = flight_time_in + collection_time + flight_time_out
                    
                    if total_leg_time < min_total_leg_time:
                        min_total_leg_time = total_leg_time
                        best_result_for_leg = {
                            'fip': fip,
                            'fop': fop,
                            'oh': optimal_oh,
                            'mode': mode,
                            'flight_in_time': flight_time_in,
                            'collection_time': collection_time,
                        }

            # After searching, we have found the best result for the current leg
            result = best_result_for_leg
            
            if not result:
                print(f"FATAL ERROR: Could not find a valid trajectory for GN {gn_index}. Aborting.")
                break 

            # Store the optimized segments for visualization
            uav_path_segments.append({'type': 'flight', 'start': previous_fop, 'end': result['fip']})
            uav_path_segments.append({'type': 'collection', **result})
            
            # Update state for the next iteration
            service_time_for_gn = result['flight_in_time'] + result['collection_time']
            current_uav_time += service_time_for_gn
            previous_fop = result['fop']
            
            print(f"    -> Best FOP found at {np.round(previous_fop, 2)}. Service Time for GN: {service_time_for_gn:.2f}s")


        # After visiting all GNs, add the final leg back to the data center
        final_flight_time = np.linalg.norm(previous_fop - sim_env.data_center_pos) / params.UAV_MAX_SPEED
        current_uav_time += final_flight_time
        uav_path_segments.append({'type': 'flight', 'start': previous_fop, 'end': sim_env.data_center_pos})
        
        # Store the final results for this UAV
        final_trajectories[uav_id] = uav_path_segments
        uav_mission_times[uav_id] = current_uav_time
        print(f"  - Optimization complete for {uav_id}. Total mission time: {current_uav_time:.2f}s")
        print("  -> Running Convex Planner for the same sequence...")
        convex_result = convex_planner.plan_shortest_path_for_sequence(gn_indices_route)
        convex_trajectories[uav_id] = convex_result['path']
        convex_path_lengths[uav_id] = convex_result['length']
        print(f"     Convex Path Length: {convex_result['length']:.2f}m")
    # --- Step 5: Analysis and Output ---
    print("\n[Step 4/5] Analyzing final results...")
    
    # Calculate the overall Mission Completion Time (MCT)
    system_mct = 0.0
    if uav_mission_times:
        system_mct = max(uav_mission_times.values())
        
    end_time = time.time()
    total_execution_time = end_time - start_time
    
    print("\n--- Simulation Summary ---")
    for uav_id in initial_assignment.keys():
        if uav_id in uav_mission_times:
            print(f"  - {uav_id} V-Shaped Mission Time: {uav_mission_times.get(uav_id, 0):.2f} seconds")
            convex_len = convex_path_lengths.get(uav_id, 0)
            convex_flight_time = convex_len / params.UAV_MAX_SPEED if convex_len > 0 else 0
            print(f"  - {uav_id} Convex Path Length:    {convex_len:.2f} meters (Theoretical Flight Time: {convex_flight_time:.2f}s)")
    
    print(f"\nSystem Mission Completion Time (MCT) for V-Shaped: {system_mct:.2f} seconds")
    print(f"Total script execution time: {total_execution_time:.2f} seconds")
    print("--------------------------")
    
    # --- Step 6: Final Visualization ---
    print("\n[Step 5/5] Visualizing final optimized trajectories...")
    vis.plot_final_comparison_trajectories(
        gns=sim_env.gn_positions,
        data_center_pos=sim_env.data_center_pos,
        v_shaped_trajectories=final_trajectories,
        convex_trajectories=convex_trajectories,
        area_width=params.AREA_WIDTH,
        area_height=params.AREA_HEIGHT,
        comm_radius=traj_optimizer.comm_radius_d
    )
    
    print("\nSimulation finished successfully.")
    print("======================================================")

# --- Script Entry Point ---
if __name__ == "__main__":
    main()