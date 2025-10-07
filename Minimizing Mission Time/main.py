# ==============================================================================
#      Main Simulation Execution Script (FINAL VERIFIED - WITH OVERLAP HANDLING)
#
# File Objective:
# This definitive version correctly handles overlapping GN communication zones
# by enforcing a zero-length flight segment between them, ensuring full
# compliance with the Min Li et al. paper.
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
from utility import get_circle_intersections

def main():
    """
    Main function to orchestrate the simulation and comparison workflow.
    """
    # --- Step 2: Initialization ---
    start_time = time.time()
    print("======================================================")
    print("      UAV Trajectory Optimization Comparison")
    print("======================================================")
    print("\n[Step 1/5] Initializing simulation environment...")

    sim_env = SimulationEnvironment(params)
    print(f"Environment created: {params.AREA_WIDTH}x{params.AREA_HEIGHT}m area with {params.NUM_GNS} GNs.")
    required_data_per_gn = 100 * 1e6
    print(f"Data requirement per GN set to {required_data_per_gn / 1e6:.0f} Mbits.")

    # --- Step 3: Phase 1 - Run Mission Allocation (Genetic Algorithm) ---
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
    
    # --- Step 4: Phase 2 - Run Trajectory Optimizations ---
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

    # Main optimization loop
    for uav_id, gn_indices_route in initial_assignment.items():
        if not gn_indices_route:
            # ... (omitted)
            continue

        print(f"\n--- Optimizing for {uav_id} with sequence: {gn_indices_route} ---")
        
        print("  -> Running V-Shaped Trajectory Optimizer (JOFC Algorithm)...")
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

            # <<< CORE LOGIC CORRECTION FOR OVERLAP HANDLING >>>
            
            # 1. Check for overlap between the previous GN's circle and the current one.
            is_overlap = False
            if i > 0:
                prev_gn_coord = sim_env.gn_positions[gn_indices_route[i-1]]
                dist_between_gns = np.linalg.norm(current_gn_coord - prev_gn_coord)
                if dist_between_gns <= 2 * traj_optimizer.comm_radius_d:
                    is_overlap = True
                    print("    -> Overlap detected with previous GN.")

            # 2. Define d_in_range based on overlap status
            step_size = params.JOFC_GRID_STEP_SIZE
            if is_overlap:
                # If overlapping, the flight-in distance is 0. FIP is the same as the previous FOP.
                d_in_range = np.array([0.0])
            else:
                dist_sp_gn = np.linalg.norm(sp - current_gn_coord)
                d1_min = max(0, dist_sp_gn - traj_optimizer.comm_radius_d)
                d1_max = dist_sp_gn + traj_optimizer.comm_radius_d
                d_in_range = np.arange(d1_min, d1_max + step_size, step_size)
                if not d_in_range.any(): d_in_range = np.array([d1_min])
            
            # Define d_out_range (this logic remains the same)
            dist_ep_gn = np.linalg.norm(ep - current_gn_coord)
            d_out_min = max(0, dist_ep_gn - traj_optimizer.comm_radius_d)
            d_out_max = dist_ep_gn + traj_optimizer.comm_radius_d
            d_out_range = np.arange(d_out_min, d_out_max + step_size, step_size)
            if not d_out_range.any(): d_out_range = np.array([d_out_min])

            # Iterate through discretized distances
            for d_in in d_in_range:
                if is_overlap:
                    fip_candidates = [sp] # FIP is fixed to the start point (previous FOP)
                else:
                    fip_candidates = get_circle_intersections(sp, d_in, current_gn_coord, traj_optimizer.comm_radius_d)
                
                if not fip_candidates: continue

                for d_out in d_out_range:
                    fop_candidates = get_circle_intersections(ep, d_out, current_gn_coord, traj_optimizer.comm_radius_d)
                    if not fop_candidates: continue
                    
                    for fip in fip_candidates:
                        for fop in fop_candidates:
                            flight_time_in = d_in / params.UAV_MAX_SPEED
                            flight_time_out = d_out / params.UAV_MAX_SPEED
                            
                            c_f_max = traj_optimizer.calculate_fm_max_capacity(fip, fop, current_gn_coord)
                            
                            if required_data_per_gn <= c_f_max:
                                mode = 'FM'
                                optimal_oh, collection_time = traj_optimizer.find_optimal_fm_trajectory(fip, fop, current_gn_coord, required_data_per_gn)
                            else:
                                mode = 'HM'
                                optimal_oh = current_gn_coord
                                collection_flight_time = (np.linalg.norm(fip - optimal_oh) + np.linalg.norm(optimal_oh - fop)) / params.UAV_MAX_SPEED
                                data_collected_during_flight = (traj_optimizer._calculate_collected_data(fip, optimal_oh, current_gn_coord) + traj_optimizer._calculate_collected_data(optimal_oh, fop, current_gn_coord))
                                hover_data_needed = required_data_per_gn - data_collected_during_flight
                                hover_time = max(0, hover_data_needed / traj_optimizer.hover_datarate) if traj_optimizer.hover_datarate > 0 else float('inf')
                                collection_time = collection_flight_time + hover_time
                            
                            total_leg_time = flight_time_in + collection_time + flight_time_out
                            
                            if total_leg_time < min_total_leg_time:
                                min_total_leg_time = total_leg_time
                                best_result_for_leg = {
                                    'fip': fip, 'fop': fop, 'oh': optimal_oh, 'mode': mode,
                                    'flight_in_time': flight_time_in,
                                    'collection_time': collection_time,
                                }

            result = best_result_for_leg
            
            if not result:
                print(f"FATAL ERROR: Could not find a valid trajectory for GN {gn_index}.")
                break 
                
            uav_path_segments.append({'type': 'flight', 'start': previous_fop, 'end': result['fip']})
            uav_path_segments.append({'type': 'collection', **result})
            
            service_time_for_gn = result['flight_in_time'] + result['collection_time']
            current_uav_time += service_time_for_gn
            previous_fop = result['fop']
            
            print(f"    -> Best FOP found. Mode: {result['mode']}. Service Time for GN: {service_time_for_gn:.2f}s")

        # ... (Rest of the file is unchanged) ...
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

    # ... Analysis and Visualization parts are unchanged and correct ...
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