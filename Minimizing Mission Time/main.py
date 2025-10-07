# ==============================================================================
#      Main Simulation Execution Script (FINAL AND DEFINITIVE CORRECT VERSION)
#
# File Objective:
# This version implements the definitive and correct logic. It uses the robust
# angle-based search for FIP/FOP combined with the full JOFC look-ahead
# objective (flight_time_in + collection_time + flight_time_out). This
# framework correctly and naturally handles all geometric scenarios,
# including overlaps, to produce smooth, globally aware trajectories
# that are truly aligned with the paper's core principles.
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
    required_data_per_gn = 30 * 1e6 # Using a smaller value to see V-shapes
    print(f"Data requirement per GN set to {required_data_per_gn / 1e6:.0f} Mbits.")

    print("\n[Step 2/5] Running Mission Allocation (Genetic Algorithm)...")
    
    ga_solver = MissionAllocationGA(
        gns=sim_env.gn_positions,
        num_uavs=params.NUM_UAVS,
        data_center_pos=sim_env.data_center_pos,
        params=params.__dict__
    )
    ga_results = ga_solver.solve()
    initial_assignment = ga_results['assignment']

    print("Visualizing initial routes from GA...")
    vis.plot_initial_routes(
        gns=sim_env.gn_positions,
        data_center_pos=sim_env.data_center_pos,
        uav_assignments=initial_assignment,
        area_width=params.AREA_WIDTH,
        area_height=params.AREA_HEIGHT
    )
    
    print("\n[Step 3/5] Running Trajectory Optimizations for each UAV...")
    
    traj_optimizer = TrajectoryOptimizer(params.__dict__)
    
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
        
        print("  -> Running V-Shaped Trajectory Optimizer (Full JOFC Angle Search)...")
        previous_fop = sim_env.data_center_pos
        uav_path_segments = []
        current_uav_time = 0.0
        num_gns_in_route = len(gn_indices_route)
        for i, gn_index in enumerate(gn_indices_route):
            current_gn_coord = sim_env.gn_positions[gn_index]
            
            sp = previous_fop
            ep = sim_env.gn_positions[gn_indices_route[i+1]] if i < num_gns_in_route - 1 else sim_env.data_center_pos
            
            print(f"  - Optimizing for GN {gn_index} (SP: {np.round(sp,1)}, EP: {np.round(ep,1)})...")
            
            min_total_leg_time = float('inf')
            best_result_for_leg = {}

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
                    else:
                        mode = 'HM'
                        optimal_oh = current_gn_coord
                        collection_flight_time = (np.linalg.norm(fip - optimal_oh) + np.linalg.norm(optimal_oh - fop)) / params.UAV_MAX_SPEED
                        data_collected_during_flight = (
                            traj_optimizer._calculate_collected_data(fip, optimal_oh, current_gn_coord) +
                            traj_optimizer._calculate_collected_data(optimal_oh, fop, current_gn_coord)
                        )
                        hover_data_needed = required_data_per_gn - data_collected_during_flight
                        hover_time = max(0, hover_data_needed / traj_optimizer.hover_datarate) if traj_optimizer.hover_datarate > 0 else float('inf')
                        collection_time = collection_flight_time + hover_time
                    
                    # Using the full look-ahead objective function
                    total_leg_time = flight_time_in + collection_time + flight_time_out
                    
                    if total_leg_time < min_total_leg_time:
                        min_total_leg_time = total_leg_time
                        best_result_for_leg = {
                            'fip': fip, 'fop': fop, 'oh': optimal_oh, 'mode': mode
                        }

            result = best_result_for_leg
            
            if not result:
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
            current_uav_time += service_time_for_gn
            previous_fop = result['fop']

            uav_path_segments.append({'type': 'flight', 'start': sp, 'end': result['fip']})
            uav_path_segments.append({'type': 'collection', **result})
            
            print(f"    -> Best FOP found. Mode: {result['mode']}. Service Time for GN: {service_time_for_gn:.2f}s")
        
        # Add the final flight back to the data center
        final_flight_time = np.linalg.norm(previous_fop - sim_env.data_center_pos) / params.UAV_MAX_SPEED
        current_uav_time += final_flight_time
        uav_path_segments.append({'type': 'flight', 'start': previous_fop, 'end': sim_env.data_center_pos})
        
        final_trajectories[uav_id] = uav_path_segments
        uav_mission_times[uav_id] = current_uav_time
        print(f"  - Optimization complete. V-Shaped Total mission time: {current_uav_time:.2f}s")

        print("  -> Running Convex Planner for the same sequence...")
        convex_result = convex_planner.plan_shortest_path_for_sequence(gn_indices_route)
        convex_trajectories[uav_id] = convex_result['path']
        convex_path_lengths[uav_id] = convex_result['length']
        print(f"     Convex Path Length: {convex_result['length']:.2f}m")

    # Analysis and Visualization
    # ... (This part is correct and remains unchanged) ...
    print("\n[Step 4/5] Analyzing final results...")
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
    system_mct = max(uav_mission_times.values()) if uav_mission_times else 0
    end_time = time.time()
    total_execution_time = end_time - start_time
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