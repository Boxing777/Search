# ==============================================================================
#         Main Simulation Execution Script (REFACTORED FOR JOFC ALGORITHM)
#
# File Objective:
# This version refactors the core optimization loop to strictly follow the JOFC
# (Algorithm 3) from the Min Li et al. paper, using a distance-based search
# for FIP and FOP instead of an angle-based search.
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
from utility import get_circle_intersections # <<< Ensure this import is present

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

    # Create the simulation environment from parameters
    sim_env = SimulationEnvironment(params)
    print(f"Environment created: {params.AREA_WIDTH}x{params.AREA_HEIGHT}m area with {params.NUM_GNS} GNs.")

    # Define the data requirement for each GN (in bits)
    required_data_per_gn = 100 * 1e6 # 100 Mbits. Change this to see different behaviors.
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
    
    # Dictionaries to store the final results from both methods
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

        print(f"\n--- Optimizing for {uav_id} with sequence: {gn_indices_route} ---")
        
        print("  -> Running V-Shaped Trajectory Optimizer (JOFC Algorithm)...")
        previous_fop = sim_env.data_center_pos
        uav_path_segments = []
        current_uav_time = 0.0
        num_gns_in_route = len(gn_indices_route)
        for i, gn_index in enumerate(gn_indices_route):
            current_gn_coord = sim_env.gn_positions[gn_index]
            
            # Define Start Point (SP) and End Point (EP) for this leg
            sp = previous_fop
            ep = sim_env.gn_positions[gn_indices_route[i+1]] if i < num_gns_in_route - 1 else sim_env.data_center_pos
            
            print(f"  - Optimizing for GN {gn_index} (SP: {np.round(sp,1)}, EP: {np.round(ep,1)})...")
            
            min_total_leg_time = float('inf')
            best_result_for_leg = {}

            # <<< REFACTORED LOGIC: Following Algorithm 3 (JOFC) with Distance Search >>>
            
            # 1. Calculate feasible range for d_i1 and d_i2 (flight distances)
            dist_sp_gn = np.linalg.norm(sp - current_gn_coord)
            d1_min = max(0, dist_sp_gn - traj_optimizer.comm_radius_d)
            d1_max = dist_sp_gn + traj_optimizer.comm_radius_d

            # For d2, we need the distance from FOP to the END POINT of the leg (ep)
            # The paper's d_i2 is the distance from FOP to the NEXT FIP, which is part of 'ep'
            # To be precise, d_i2 is distance from FOP_i to FIP_{i+1}. Let's call it d_out here.
            # So we will search over d_in (d1) and the FOP position.
            # To simplify and align with the spirit of JOFC, we will search d1 and d_out_to_ep
            dist_ep_gn = np.linalg.norm(ep - current_gn_coord)
            d_out_min = max(0, dist_ep_gn - traj_optimizer.comm_radius_d)
            d_out_max = dist_ep_gn + traj_optimizer.comm_radius_d

            # 2. Discretize the search space for d_in (d1) and d_out_to_ep
            step_size = params.JOFC_GRID_STEP_SIZE
            d_in_range = np.arange(d1_min, d1_max + step_size, step_size)
            d_out_range = np.arange(d_out_min, d_out_max + step_size, step_size)

            # 3. Iterate through discretized distances
            for d_in in d_in_range:
                fip_candidates = get_circle_intersections(sp, d_in, current_gn_coord, traj_optimizer.comm_radius_d)
                if not fip_candidates: continue

                for d_out in d_out_range:
                    fop_candidates = get_circle_intersections(ep, d_out, current_gn_coord, traj_optimizer.comm_radius_d)
                    if not fop_candidates: continue
                    
                    for fip in fip_candidates:
                        for fop in fop_candidates:
                            flight_time_in = d_in / params.UAV_MAX_SPEED
                            flight_time_out = d_out / params.UAV_MAX_SPEED
                            
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
                            
                            # The total time for the entire leg (the value to be minimized)
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
                print(f"FATAL ERROR: Could not find a valid trajectory for GN {gn_index}. Aborting.")
                break 
                
            uav_path_segments.append({'type': 'flight', 'start': previous_fop, 'end': result['fip']})
            uav_path_segments.append({'type': 'collection', **result})
            
            service_time_for_gn = result['flight_in_time'] + result['collection_time']
            current_uav_time += service_time_for_gn
            previous_fop = result['fop']
            
            print(f"    -> Best FOP found at {np.round(previous_fop, 2)}. Service Time for GN: {service_time_for_gn:.2f}s")


        final_flight_time = np.linalg.norm(previous_fop - sim_env.data_center_pos) / params.UAV_MAX_SPEED
        current_uav_time += final_flight_time
        uav_path_segments.append({'type': 'flight', 'start': previous_fop, 'end': sim_env.data_center_pos})
        
        final_trajectories[uav_id] = uav_path_segments
        uav_mission_times[uav_id] = current_uav_time
        print(f"  - Optimization complete for {uav_id}. V-Shaped Total mission time: {current_uav_time:.2f}s")

        print("  -> Running Convex Planner for the same sequence...")
        convex_result = convex_planner.plan_shortest_path_for_sequence(gn_indices_route)
        convex_trajectories[uav_id] = convex_result['path']
        convex_path_lengths[uav_id] = convex_result['length']
        print(f"     Convex Path Length: {convex_result['length']:.2f}m")

    # --- Step 5: Analysis and Output ---
    print("\n[Step 4/5] Analyzing final results...")
    
    v_shaped_path_lengths = {}
    for uav_id, segments in final_trajectories.items():
        total_length = 0.0
        for segment in segments:
            if segment['type'] == 'flight':
                start = np.array(segment['start'])
                end = np.array(segment['end'])
                total_length += np.linalg.norm(end - start)
            elif segment['type'] == 'collection':
                fip = np.array(segment['fip'])
                oh = np.array(segment['oh'])
                fop = np.array(segment['fop'])
                total_length += np.linalg.norm(oh - fip) + np.linalg.norm(fop - oh)
        v_shaped_path_lengths[uav_id] = total_length
    
    system_mct = max(uav_mission_times.values()) if uav_mission_times else 0
    end_time = time.time()
    total_execution_time = end_time - start_time
    
    print("\n--- Simulation Summary ---")
    for uav_id in initial_assignment.keys():
        if uav_id in uav_mission_times:
            print(f"--- {uav_id} Results ---")
            print(f"  V-Shaped Mission Time: {uav_mission_times.get(uav_id, 0):.2f} seconds")
            print(f"  V-Shaped Path Length:    {v_shaped_path_lengths.get(uav_id, 0):.2f} meters")
            
            convex_len = convex_path_lengths.get(uav_id, 0)
            convex_flight_time = convex_len / params.UAV_MAX_SPEED if convex_len > 0 else 0
            print(f"  Convex Path Length:      {convex_len:.2f} meters (Theoretical Flight Time: {convex_flight_time:.2f}s)")
    
    print(f"\nSystem Mission Completion Time (MCT) for V-Shaped: {system_mct:.2f} seconds")
    print(f"Total script execution time: {total_execution_time:.2f} seconds")
    print("--------------------------")
    
    # --- Step 6: Final Visualization ---
    print("\n[Step 5/5] Visualizing final combined trajectories...")
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