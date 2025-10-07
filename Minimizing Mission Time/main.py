# ==============================================================================
<<<<<<< HEAD
#      Main Simulation Execution Script (FINAL AND DEFINITIVE CORRECT VERSION)
#
# File Objective:
# This version implements the definitive and correct logic. It uses the robust
# angle-based search for FIP/FOP combined with the full JOFC look-ahead
# objective (flight_time_in + collection_time + flight_time_out). This
# framework correctly and naturally handles all geometric scenarios,
# including overlaps, to produce smooth, globally aware trajectories
# that are truly aligned with the paper's core principles.
=======
#                      Main Simulation Execution Script (MODIFIED FOR COMPARISON)
#
# File Objective:
# This script runs the primary V-shaped trajectory optimization and, for comparison,
# also computes the shortest possible geometric path for the same GN visiting
# order using convex optimization. Both results are then visualized.
>>>>>>> parent of 555e802 (仍有小BUG 已改掉用角度找)
# ==============================================================================

import time
import numpy as np

import parameters as params
from environment import SimulationEnvironment
from mission_allocation_ga import MissionAllocationGA
from trajectory_optimizer import TrajectoryOptimizer
from convex_trajectory_planner import ConvexTrajectoryPlanner
import visualizer as vis

def main():
    start_time = time.time()
    print("======================================================")
    print("      UAV Trajectory Optimization Comparison")
    print("======================================================")
    print("\n[Step 1/5] Initializing simulation environment...")

    sim_env = SimulationEnvironment(params)
    print(f"Environment created: {params.AREA_WIDTH}x{params.AREA_HEIGHT}m area with {params.NUM_GNS} GNs.")
<<<<<<< HEAD
    required_data_per_gn = 30 * 1e6 # Using a smaller value to see V-shapes
=======

    # Define the data requirement for each GN (in bits)
    # Set a high value to test the Hovering Mode (HM)
    required_data_per_gn = 100 * 1e6 # 100 Mbits. Change this to see different behaviors.
>>>>>>> parent of 555e802 (仍有小BUG 已改掉用角度找)
    print(f"Data requirement per GN set to {required_data_per_gn / 1e6:.0f} Mbits.")

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
    
    print("\n[Step 3/5] Running Trajectory Optimizations for each UAV...")
    
    # Instantiate the trajectory optimizer to use its helper functions
    traj_optimizer = TrajectoryOptimizer(params.__dict__)
    
    # Instantiate the convex planner
    convex_planner = ConvexTrajectoryPlanner(
        gns=sim_env.gn_positions,
        data_center_pos=sim_env.data_center_pos,
        comm_radius=traj_optimizer.comm_radius_d
    )
    
    final_trajectories = {}
    uav_mission_times = {}
    convex_trajectories = {}
    convex_path_lengths = {}

    for uav_id, gn_indices_route in initial_assignment.items():
        if not gn_indices_route:
            continue

        print(f"\n--- Optimizing for {uav_id} with sequence: {gn_indices_route} ---")
        
<<<<<<< HEAD
        print("  -> Running V-Shaped Trajectory Optimizer (Full JOFC Angle Search)...")
=======
        # --- Method 1: V-Shaped Trajectory (Your existing logic) ---
        print("  -> Running V-Shaped Trajectory Optimizer...")
>>>>>>> parent of 555e802 (仍有小BUG 已改掉用角度找)
        previous_fop = sim_env.data_center_pos
        uav_path_segments = []
        current_uav_time = 0.0

        # Inner loop: iterate through each GN in the UAV's assigned route
        num_gns_in_route = len(gn_indices_route)
        for i, gn_index in enumerate(gn_indices_route):
            current_gn_coord = sim_env.gn_positions[gn_index]
            
<<<<<<< HEAD
            sp = previous_fop
            ep = sim_env.gn_positions[gn_indices_route[i+1]] if i < num_gns_in_route - 1 else sim_env.data_center_pos
            
            print(f"  - Optimizing for GN {gn_index} (SP: {np.round(sp,1)}, EP: {np.round(ep,1)})...")
=======
            # Determine the coordinate of the next stop for "look-ahead" calculation
            if i < num_gns_in_route - 1:
                next_stop_coord = sim_env.gn_positions[gn_indices_route[i+1]]
            else: 
                next_stop_coord = sim_env.data_center_pos

            print(f"  - Optimizing for GN {gn_index} (considering next stop at {np.round(next_stop_coord, 2)})...")
>>>>>>> parent of 555e802 (仍有小BUG 已改掉用角度找)
            
            min_total_leg_time = float('inf')
            best_result_for_leg = {}

<<<<<<< HEAD
            # <<< USING THE ROBUST, FULL JOFC ANGLE-BASED SEARCH >>>
            
            num_angles = 8
            angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
            candidate_points = np.array([
                current_gn_coord + traj_optimizer.comm_radius_d * np.array([np.cos(theta), np.sin(theta)])
                for theta in angles
            ])
            
            for fip in candidate_points:
                for fop in candidate_points:
                    flight_time_in = np.linalg.norm(sp - fip) / params.UAV_MAX_SPEED
                    flight_time_out = np.linalg.norm(ep - fop) / params.UAV_MAX_SPEED
                    
                    c_f_max = traj_optimizer.calculate_fm_max_capacity(fip, fop, current_gn_coord)
                    
                    if required_data_per_gn <= c_f_max:
                        mode = 'FM'
                        optimal_oh, collection_time = traj_optimizer.find_optimal_fm_trajectory(
                            fip, fop, current_gn_coord, required_data_per_gn
                        )
=======
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
>>>>>>> parent of 555e802 (仍有小BUG 已改掉用角度找)
                    else:
                        mode = 'HM'
                        optimal_oh = current_gn_coord
                        collection_flight_time = (np.linalg.norm(fip - optimal_oh) + np.linalg.norm(optimal_oh - fop)) / params.UAV_MAX_SPEED
<<<<<<< HEAD
                        data_collected_during_flight = (
                            traj_optimizer._calculate_collected_data(fip, optimal_oh, current_gn_coord) +
                            traj_optimizer._calculate_collected_data(optimal_oh, fop, current_gn_coord)
                        )
                        hover_data_needed = required_data_per_gn - data_collected_during_flight
                        hover_time = max(0, hover_data_needed / traj_optimizer.hover_datarate) if traj_optimizer.hover_datarate > 0 else float('inf')
                        collection_time = collection_flight_time + hover_time
                    
                    # Using the full look-ahead objective function
=======
                        hover_data_needed = required_data_per_gn - c_f_max
                        hover_time = hover_data_needed / traj_optimizer.hover_datarate if traj_optimizer.hover_datarate > 0 else float('inf')
                        collection_time = collection_flight_time + hover_time
                    
                    # 3. Flight-out time (This is the crucial "look-ahead" part)
                    flight_time_out = np.linalg.norm(fop - next_stop_coord) / params.UAV_MAX_SPEED
                    
                    # --- Total time for this entire leg (the value to be minimized) ---
>>>>>>> parent of 555e802 (仍有小BUG 已改掉用角度找)
                    total_leg_time = flight_time_in + collection_time + flight_time_out
                    
                    if total_leg_time < min_total_leg_time:
                        min_total_leg_time = total_leg_time
                        best_result_for_leg = {
<<<<<<< HEAD
                            'fip': fip, 'fop': fop, 'oh': optimal_oh, 'mode': mode
=======
                            'fip': fip,
                            'fop': fop,
                            'oh': optimal_oh,
                            'mode': mode,
                            'flight_in_time': flight_time_in,
                            'collection_time': collection_time,
>>>>>>> parent of 555e802 (仍有小BUG 已改掉用角度找)
                        }

            # After searching, we have found the best result for the current leg
            result = best_result_for_leg
            
            if not result:
<<<<<<< HEAD
                print(f"FATAL ERROR: Could not find a valid trajectory for GN {gn_index}.")
                break
            
            # Recalculate the exact times for the chosen path to avoid storing stale data
            final_flight_time_in = np.linalg.norm(sp - result['fip']) / params.UAV_MAX_SPEED
            if result['mode'] == 'FM':
                _, final_collection_time = traj_optimizer.find_optimal_fm_trajectory(
                    result['fip'], result['fop'], current_gn_coord, required_data_per_gn
                )
            else: # HM mode
                collection_flight_time = (np.linalg.norm(result['fip'] - result['oh']) + np.linalg.norm(result['oh'] - result['fop'])) / params.UAV_MAX_SPEED
                data_collected_during_flight = (
                    traj_optimizer._calculate_collected_data(result['fip'], result['oh'], current_gn_coord) +
                    traj_optimizer._calculate_collected_data(result['oh'], result['fop'], current_gn_coord)
                )
                hover_data_needed = required_data_per_gn - data_collected_during_flight
                hover_time = max(0, hover_data_needed / traj_optimizer.hover_datarate) if traj_optimizer.hover_datarate > 0 else float('inf')
                final_collection_time = collection_flight_time + hover_time

            service_time_for_gn = final_flight_time_in + final_collection_time
=======
                print(f"FATAL ERROR: Could not find a valid trajectory for GN {gn_index}. Aborting.")
                break 

            # Store the optimized segments for visualization
            uav_path_segments.append({'type': 'flight', 'start': previous_fop, 'end': result['fip']})
            uav_path_segments.append({'type': 'collection', **result})
            
            # Update state for the next iteration
            service_time_for_gn = result['flight_in_time'] + result['collection_time']
>>>>>>> parent of 555e802 (仍有小BUG 已改掉用角度找)
            current_uav_time += service_time_for_gn
            previous_fop = result['fop']

            uav_path_segments.append({'type': 'flight', 'start': sp, 'end': result['fip']})
            uav_path_segments.append({'type': 'collection', **result})
            
<<<<<<< HEAD
            print(f"    -> Best FOP found. Mode: {result['mode']}. Service Time for GN: {service_time_for_gn:.2f}s")
        
        # Add the final flight back to the data center
=======
            print(f"    -> Best FOP found at {np.round(previous_fop, 2)}. Service Time for GN: {service_time_for_gn:.2f}s")


        # After visiting all GNs, add the final leg back to the data center
>>>>>>> parent of 555e802 (仍有小BUG 已改掉用角度找)
        final_flight_time = np.linalg.norm(previous_fop - sim_env.data_center_pos) / params.UAV_MAX_SPEED
        current_uav_time += final_flight_time
        uav_path_segments.append({'type': 'flight', 'start': previous_fop, 'end': sim_env.data_center_pos})
        
        # Store the final results for this UAV
        final_trajectories[uav_id] = uav_path_segments
        uav_mission_times[uav_id] = current_uav_time
        print(f"  - Optimization complete. V-Shaped Total mission time: {current_uav_time:.2f}s")

        # --- Method 2: Convex Optimal Path ---
        print("  -> Running Convex Planner for the same sequence...")
        convex_result = convex_planner.plan_shortest_path_for_sequence(gn_indices_route)
        convex_trajectories[uav_id] = convex_result['path']
        convex_path_lengths[uav_id] = convex_result['length']
        print(f"     Convex Path Length: {convex_result['length']:.2f}m")

    # Analysis and Visualization
    # ... (This part is correct and remains unchanged) ...
    print("\n[Step 4/5] Analyzing final results...")
<<<<<<< HEAD
=======
    
    # <<< NEW SECTION: Calculate V-Shaped Path Lengths >>>
>>>>>>> parent of 555e802 (仍有小BUG 已改掉用角度找)
    v_shaped_path_lengths = {}
    for uav_id, segments in final_trajectories.items():
        total_length = 0.0
        for segment in segments:
            if not segment: continue
            if segment['type'] == 'flight':
                start, end = np.array(segment['start']), np.array(segment['end'])
                total_length += np.linalg.norm(end - start)
            elif segment['type'] == 'collection':
                fip, oh, fop = np.array(segment['fip']), np.array(segment['oh']), np.array(segment['fop'])
                total_length += np.linalg.norm(oh - fip) + np.linalg.norm(fop - oh)
        v_shaped_path_lengths[uav_id] = total_length
<<<<<<< HEAD
    system_mct = max(uav_mission_times.values()) if uav_mission_times else 0
    end_time = time.time()
    total_execution_time = end_time - start_time
=======

    # Calculate the overall Mission Completion Time (MCT)
    system_mct = 0.0
    if uav_mission_times:
        system_mct = max(uav_mission_times.values())
        
    end_time = time.time()
    total_execution_time = end_time - start_time
    
    # <<< MODIFIED: Update summary output to include both path lengths >>>
>>>>>>> parent of 555e802 (仍有小BUG 已改掉用角度找)
    print("\n--- Simulation Summary ---")
    for uav_id in initial_assignment.keys():
        if uav_id in uav_mission_times and uav_mission_times[uav_id] > 0:
            print(f"--- {uav_id} Results ---")
            print(f"  V-Shaped Mission Time: {uav_mission_times.get(uav_id, 0):.2f} seconds")
            print(f"  V-Shaped Path Length:    {v_shaped_path_lengths.get(uav_id, 0):.2f} meters")
            convex_len = convex_path_lengths.get(uav_id, 0)
            convex_flight_time = convex_len / params.UAV_MAX_SPEED if convex_len > 0 else 0
            print(f"  Convex Path Length:      {convex_len:.2f} meters (Theoretical Flight Time: {convex_flight_time:.2f}s)")
    print(f"\nSystem Mission Completion Time (MCT) for V-Shaped: {system_mct:.2f} seconds")
    print(f"Total script execution time: {total_execution_time:.2f} seconds")
    print("--------------------------")
    print("\n[Step 5/5] Visualizing final combined trajectories...")
    vis.plot_final_comparison_trajectories(
        gns=sim_env.gn_positions, data_center_pos=sim_env.data_center_pos,
        v_shaped_trajectories=final_trajectories, convex_trajectories=convex_trajectories,
        area_width=params.AREA_WIDTH, area_height=params.AREA_HEIGHT,
        comm_radius=traj_optimizer.comm_radius_d
    )
    print("\nSimulation finished successfully.")
    print("======================================================")

if __name__ == "__main__":
    main()