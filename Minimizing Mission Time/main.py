# ==============================================================================
#      Main Simulation Script (Corrected Objective Function in JOFC)
#
# File Objective:
# This version fundamentally corrects the objective function within the JOFC
# loop to align with the paper's description (Eq. 17), including the cost of
# flying away from the current FOP towards the next target. This intrinsically
# prevents path-folding without needing artificial constraints.
# ==============================================================================

import time
import numpy as np
import os
import sys
import visualizer as vis
import traceback
import pandas as pd
import reporter
import parameters as params

from environment import SimulationEnvironment
from mission_allocation_ga import MissionAllocationGA
from trajectory_optimizer import TrajectoryOptimizer
from convex_trajectory_planner import ConvexTrajectoryPlanner
from datetime import datetime
from bob_planner import BOBPlanner
from cmc_planner import CMCPlanner

# --- Helper Class and Functions (No changes needed here) ---
class Logger:
    """A simple logger to write output to both console and a file."""
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        try:
            self.log_file = open(log_file_path, "w", encoding='utf-8')
        except Exception as e:
            print(f"CRITICAL: Failed to open log file at {log_file_path}. Error: {e}")
            self.log_file = None

    def write(self, message):
        self.terminal.write(message)
        if self.log_file:
            self.log_file.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()
            
    def close(self):
        if self.log_file:
            self.log_file.close()

def get_circle_intersections(p1: np.ndarray, r1: float, p2: np.ndarray, r2: float) -> list:
    d = np.linalg.norm(p1 - p2)
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0: return []
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(max(0, r1**2 - a**2))
    p_mid = p1 + a * (p2 - p1) / d
    x_mid, y_mid = p_mid[0], p_mid[1]
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    inter1 = (x_mid + h * dy / d, y_mid - h * dx / d)
    inter2 = (x_mid - h * dy / d, y_mid + h * dx / d)
    if np.allclose(inter1, inter2): return [np.array(inter1)]
    return [np.array(inter1), np.array(inter2)]

# --- Main Simulation Logic for a Single Run ---
def run_single_simulation(run_prefix: str, output_dir: str):
    start_time = time.time()
    print("======================================================")
    print(f"      UAV Trajectory Opt. - {run_prefix.upper()}")
    print("======================================================")
    print("\n[Step 1/5] Initializing simulation environment...")

    traj_optimizer = TrajectoryOptimizer(params.__dict__)
    
    rate_at_edge = traj_optimizer.get_rate_at_comm_edge()

    # To calculate the benchmark C_f_max, we define a standard diameter path
    D = traj_optimizer.comm_radius_d
    # Assume a virtual GN at origin for this calculation
    virtual_gn_coord = np.array([0.0, 0.0])
    # FIP and FOP are at the ends of the diameter
    fip_benchmark = np.array([-D, 0.0])
    fop_benchmark = np.array([D, 0.0])
    
    # Calculate the max capacity on this ideal path
    c_f_max = traj_optimizer.calculate_fm_max_capacity(fip_benchmark, fop_benchmark, virtual_gn_coord)
    
    print(f"\n--- Key Performance Indicators ---")
    print(f"Communication Radius (D): {traj_optimizer.comm_radius_d:.2f} meters.")
    print(f"Data Rate at GN Center (max): {traj_optimizer.hover_datarate / 1e6:.2f} Mbps.")
    print(f"Data Rate at GN Edge (min):   {rate_at_edge / 1e6:.2f} Mbps.")
    print(f"Max Data in FM Mode (C_f_max): {c_f_max / 1e6:.2f} Mbits.") 
    print(f"---------------------------------")
    
    comm_radius = traj_optimizer.comm_radius_d
    sim_env = SimulationEnvironment(params, comm_radius=comm_radius)
    
    print(f"Environment created: {params.AREA_WIDTH}x{params.AREA_HEIGHT}m area with {params.NUM_GNS} GNs.")

    required_data_per_gn = 24 * 1e6 # Set to a high value to see V-shapes data size
    print(f"Data requirement per GN set to {required_data_per_gn / 1e6:.0f} Mbits.")
    
    print("\n[Step 2/5] Running Mission Allocation (Genetic Algorithm)...")
    ga_solver = MissionAllocationGA(
        gns=sim_env.gn_positions, num_uavs=params.NUM_UAVS, data_center_pos=sim_env.data_center_pos,
        transmission_radius_d=traj_optimizer.comm_radius_d, params=params.__dict__
    )
    initial_assignment = ga_solver.solve()['assignment']
    
    vis.plot_initial_routes(
        gns=sim_env.gn_positions, data_center_pos=sim_env.data_center_pos, uav_assignments=initial_assignment,
        area_width=params.AREA_WIDTH, area_height=params.AREA_HEIGHT,
        save_path=os.path.join(output_dir, f'{run_prefix}_initial_routes.png')
    )
    
    print("\n[Step 3/5] Running Trajectory Optimizations for each UAV...")
    convex_planner = ConvexTrajectoryPlanner(
        gns=sim_env.gn_positions, data_center_pos=sim_env.data_center_pos, comm_radius=traj_optimizer.comm_radius_d
    )
    
    bob_planner = BOBPlanner(
        gns=sim_env.gn_positions,
        data_center_pos=sim_env.data_center_pos,
        traj_optimizer=traj_optimizer,
        convex_planner=convex_planner
    )
    
    cmc_planner = CMCPlanner(
        traj_optimizer=traj_optimizer,
        convex_planner=convex_planner
    )
    
    final_trajectories, uav_mission_times = {}, {}
    convex_trajectories, convex_path_lengths = {}, {}
    
    convex_mission_times = {}
    
    bob_trajectories, bob_path_lengths = {}, {}
    bob_mission_times = {}
    
    cmc_mission_times = {}
    cmc_plot_points = {}

    for uav_id, gn_indices_route in initial_assignment.items():
        if not gn_indices_route:
            final_trajectories[uav_id], uav_mission_times[uav_id] = [], 0.0
            convex_trajectories[uav_id], convex_path_lengths[uav_id] = np.array([]), 0.0
            convex_mission_times[uav_id] = 0.0
            continue

        print(f"\n--- Optimizing for {uav_id} with sequence: {gn_indices_route} ---")
        previous_fop, uav_path_segments, current_uav_time = sim_env.data_center_pos, [], 0.0

        # <<< FUNDAMENTAL CORRECTION OF THE JOFC LOOP STARTS HERE >>>
        for i, gn_index in enumerate(gn_indices_route):
            sp = previous_fop
            current_gn_coord = sim_env.gn_positions[gn_index]
            
            is_last_gn = (i == len(gn_indices_route) - 1)
            next_target_anchor = sim_env.data_center_pos if is_last_gn else sim_env.gn_positions[gn_indices_route[i+1]]

            min_total_leg_time = float('inf')
            best_leg_config = {}
            
            # --- Determine if this is an overlapping case ---
            is_overlapping = np.linalg.norm(sp - current_gn_coord) <= comm_radius
            
            # --- Unified Search Logic for FIP and FOP candidates ---
            if is_overlapping:
                # FIP is fixed, only search for FOP
                fip_candidates = [sp]
                flight_time_in_base = 0.0
            else:
                # Standard non-overlapping case, search both FIP and FOP
                num_angles = 36 # You can adjust this for precision vs. speed
                angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
                fip_candidates = [current_gn_coord + comm_radius * np.array([np.cos(a), np.sin(a)]) for a in angles]

            num_angles_fop = 36
            angles_fop = np.linspace(0, 2 * np.pi, num_angles_fop, endpoint=False)
            fop_candidates = [current_gn_coord + comm_radius * np.array([np.cos(a), np.sin(a)]) for a in angles_fop]
            
            for fip in fip_candidates:
                # For non-overlapping, this is the flight time from the previous FOP to the current FIP
                flight_time_in = np.linalg.norm(sp - fip) / params.UAV_MAX_SPEED if not is_overlapping else 0.0
                
                for fop in fop_candidates:
                    # Calculate max data capacity for this FIP/FOP pair
                    c_max = traj_optimizer.calculate_fm_max_capacity(fip, fop, current_gn_coord)
                    
                    # Determine collection mode (FM or HM) and calculate theoretical collection time
                    if required_data_per_gn <= c_max:
                        # FM Mode
                        optimal_oh, collection_time_theoretical = traj_optimizer.find_optimal_fm_trajectory(
                            fip, fop, current_gn_coord, required_data_per_gn, is_overlapping=is_overlapping)
                    else:
                        # HM Mode
                        optimal_oh = current_gn_coord
                        collection_flight_time = (np.linalg.norm(fip - optimal_oh) + np.linalg.norm(optimal_oh - fop)) / params.UAV_MAX_SPEED
                        hover_time = (required_data_per_gn - c_max) / traj_optimizer.hover_datarate if traj_optimizer.hover_datarate > 0 else float('inf')
                        collection_time_theoretical = collection_flight_time + hover_time

                    # +++ KEY CORRECTION: APPLY PHYSICAL TIME CONSTRAINT +++
                    physical_collection_dist = np.linalg.norm(optimal_oh - fip) + np.linalg.norm(fop - optimal_oh)
                    physical_collection_time = physical_collection_dist / params.UAV_MAX_SPEED
                    collection_time_final = max(collection_time_theoretical, physical_collection_time)
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++

                    # Calculate the flight time to the *next* anchor point
                    flight_time_out = np.linalg.norm(next_target_anchor - fop) / params.UAV_MAX_SPEED
                    
                    # The objective function for JOFC
                    total_leg_time = flight_time_in + collection_time_final + flight_time_out
                    
                    if total_leg_time < min_total_leg_time:
                        min_total_leg_time = total_leg_time
                        best_leg_config = {
                            'fip': fip, 'fop': fop, 'oh': optimal_oh,
                            'flight_time_in': flight_time_in,
                            'collection_time': collection_time_final # Store the corrected time
                        }

            if not best_leg_config:
                print(f"WARNING: Fallback for GN {gn_index}.")
                oh = current_gn_coord
                flight_time_in = np.linalg.norm(sp - oh) / params.UAV_MAX_SPEED
                hover_time = required_data_per_gn / traj_optimizer.hover_datarate
                best_leg_config = {'fip': oh, 'fop': oh, 'oh': oh, 
                                   'flight_time_in': flight_time_in, 'collection_time': hover_time}

            # Accumulate the actual time spent (service time for this GN)
            service_time_for_leg = best_leg_config['flight_time_in'] + best_leg_config['collection_time']
            current_uav_time += service_time_for_leg
            
            # For reporting and data structure, create the service_time key
            best_leg_config['service_time'] = service_time_for_leg
            
            # The FOP from the best configuration becomes the SP for the next iteration
            previous_fop = best_leg_config['fop']
            
            uav_path_segments.extend([{'type': 'flight', 'start': sp, 'end': best_leg_config['fip']},
                                      {'type': 'collection', **best_leg_config}])
            
            # The printed "Service Time" should reflect the total time for this leg
            print(f"    -> Optimized for GN {gn_index}. Leg Time (t_in+t_collect): {service_time_for_leg:.2f}s. New FOP: {np.round(previous_fop, 1)}")

        # <<< FUNDAMENTAL CORRECTION OF THE JOFC LOOP ENDS HERE >>>

        final_flight_time = np.linalg.norm(previous_fop - sim_env.data_center_pos) / params.UAV_MAX_SPEED
        current_uav_time += final_flight_time
        uav_path_segments.append({'type': 'flight', 'start': previous_fop, 'end': sim_env.data_center_pos})
        
        final_trajectories[uav_id], uav_mission_times[uav_id] = uav_path_segments, current_uav_time
        print(f"  - Optimization complete for {uav_id}. Total mission time: {current_uav_time:.2f}s")
        
        print("  -> Running Convex Planner for the same sequence...")
        convex_result = convex_planner.plan_shortest_path_for_sequence(gn_indices_route)
        convex_trajectories[uav_id], convex_path_lengths[uav_id] = convex_result['path'], convex_result['length']
        print(f"     Convex Path Length: {convex_result['length']:.2f}m")
        print("  -> Calculating actual mission time for Convex Path...")
        
        convex_total_hover_time = 0.0
        # Iterate through the path segments that are inside communication zones
        for segment in convex_result['collection_segments']:
            gn_coord = sim_env.gn_positions[segment['gn_index']]
            
            # Calculate how much data can be collected just by flying through
            data_collected_on_segment = traj_optimizer._calculate_collected_data(
                segment['start'], segment['end'], gn_coord
            )
            
            # Determine if hovering is needed
            data_shortfall = required_data_per_gn - data_collected_on_segment
            
            collection_flight_time = np.linalg.norm(segment['end'] - segment['start']) / params.UAV_MAX_SPEED
            hover_time = 0.0

            if data_shortfall > 0:
                # hover_time = data_shortfall / traj_optimizer.hover_datarate if traj_optimizer.hover_datarate > 0 else float('inf')
                # convex_total_hover_time += hover_time
                exit_point_Eo = segment['end']
                rate_at_Eo = traj_optimizer.calculate_hover_rate_at_point(exit_point_Eo, gn_coord)
                hover_time = data_shortfall / rate_at_Eo if rate_at_Eo > 1e-6 else float('inf')
                
            convex_total_hover_time += hover_time
            
            # Add detailed time to the segment dict for the reporter module to use later
            segment['Total_Collection_Time (s)'] = collection_flight_time + hover_time
        
        # The total fair time is the flight time plus any required hover time
        convex_flight_time = convex_result['length'] / params.UAV_MAX_SPEED
        convex_actual_mission_time = convex_flight_time + convex_total_hover_time
        convex_mission_times[uav_id] = convex_actual_mission_time
        
        print(f"     Convex Path -> Flight Time: {convex_flight_time:.2f}s, Required Hover Time: {convex_total_hover_time:.2f}s, TOTAL FAIR TIME: {convex_actual_mission_time:.2f}s")

        print("\n  -> Running BOB Planner for the same sequence...")
        bob_result = bob_planner.plan_path(gn_indices_route, required_data_per_gn)
        
        
        bob_mission_times[uav_id] = bob_result['total_time']
        bob_path_lengths[uav_id] = bob_result['total_length']
        bob_trajectories[uav_id] = bob_result['segments'] 
        
        print(f"     BOB Mission Time: {bob_result['total_time']:.2f}s | Path Length: {bob_result['total_length']:.2f}m")
        print("\n  -> Running CMC Planner for the same sequence...")
        
        cmc_result = cmc_planner.estimate_mission_time(gn_indices_route, required_data_per_gn)
        cmc_mission_times[uav_id] = cmc_result['total_time']
        cmc_plot_points[uav_id] = cmc_result['plot_points']
        
        print(f"     CMC Path    -> Flight Time: {cmc_result['flight_time']:.2f}s, Required Hover Time: {cmc_result['hover_time']:.2f}s, TOTAL FAIR TIME: {cmc_result['total_time']:.2f}s")

        if gn_indices_route:
            reporter.generate_flight_log_report( # Changed function name
                run_prefix=run_prefix,
                output_dir=output_dir,
                uav_id=uav_id,
                v_shaped_segments=final_trajectories.get(uav_id, []),
                convex_result=convex_result,
                data_center_pos=sim_env.data_center_pos,
                uav_speed=params.UAV_MAX_SPEED
            )

    print("\n[Step 4/5] Analyzing final results...")
    v_shaped_path_lengths = {}
    for uav_id, segments in final_trajectories.items():
        total_length = 0
        for s in segments:
            total_length += np.linalg.norm(s.get('end', s.get('fop')) - s.get('start', s.get('fip'))) if s['type']=='flight' else np.linalg.norm(s['oh']-s['fip']) + np.linalg.norm(s['fop']-s['oh'])
        v_shaped_path_lengths[uav_id] = total_length
    
    system_mct_v_shaped = max(uav_mission_times.values()) if uav_mission_times else 0
    system_mct_convex = max(convex_mission_times.values()) if convex_mission_times else 0
    system_mct_bob = max(bob_mission_times.values()) if bob_mission_times else 0
    system_mct_cmc = max(cmc_mission_times.values()) if cmc_mission_times else 0
    total_execution_time = time.time() - start_time
    
    print("\n--- Simulation Summary ---")
    for uav_id in initial_assignment.keys():
        if uav_id in uav_mission_times:
            print(f"--- {uav_id} Results ---")
            print(f"  V-Shaped Mission Time: {uav_mission_times.get(uav_id, 0):.2f}s | Path Length: {v_shaped_path_lengths.get(uav_id, 0):.2f}m")
            print(f"  Convex Mission Time:   {convex_mission_times.get(uav_id, 0):.2f}s | Path Length: {convex_path_lengths.get(uav_id, 0):.2f}m")
            print(f"  CMC Mission Time:      {cmc_mission_times.get(uav_id, 0):.2f}s | Path Length: {convex_path_lengths.get(uav_id, 0):.2f}m")
            print(f"  BOB Mission Time:      {bob_mission_times.get(uav_id, 0):.2f}s | Path Length: {bob_path_lengths.get(uav_id, 0):.2f}m")
    
    print(f"\nSystem Mission Completion Time (MCT) for V-Shaped: {system_mct_v_shaped:.2f}s")
    print(f"System Mission Completion Time (MCT) for Convex:   {system_mct_convex:.2f}s")
    print(f"System Mission Completion Time (MCT) for CMC:      {system_mct_cmc:.2f}s")
    print(f"System Mission Completion Time (MCT) for BOB:   {system_mct_bob:.2f}s")
    print(f"Total script execution time: {total_execution_time:.2f}s")
    
    print("\n[Step 5/5] Visualizing final combined trajectories...")
    vis.plot_final_comparison_trajectories(
        gns=sim_env.gn_positions, data_center_pos=sim_env.data_center_pos,
        v_shaped_trajectories=final_trajectories, convex_trajectories=convex_trajectories,
        bob_trajectories=bob_trajectories, cmc_plot_points=cmc_plot_points, area_width=params.AREA_WIDTH, 
        area_height=params.AREA_HEIGHT,comm_radius=traj_optimizer.comm_radius_d,
        save_path=os.path.join(output_dir, f'{run_prefix}_final_trajectories.png')
    )
    
    print("\nSimulation finished successfully.")
    print("======================================================")

# --- Main Entry Point for Batch Execution (No changes from your version) ---
if __name__ == "__main__":
    NUMBER_OF_RUNS = 200 # Set to 1 for testing the fix
    BASE_RESULTS_DIR = "simulation_results"
    BATCH_SEED = 777 # You can choose any integer you like.
    
    if not os.path.exists(BASE_RESULTS_DIR):
        os.makedirs(BASE_RESULTS_DIR)
    
    batch_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_run_dir = os.path.join(BASE_RESULTS_DIR, f"run_{batch_timestamp}")
    os.makedirs(batch_run_dir)
    
    print(f"Starting batch of {NUMBER_OF_RUNS} runs with Master Seed: {BATCH_SEED}.")
    print(f"All results will be saved in: {batch_run_dir}")
    
    original_stdout = sys.stdout
    
    for i in range(NUMBER_OF_RUNS):
        run_prefix = f"run_{i+1}"
        log_path = os.path.join(batch_run_dir, f"{run_prefix}_log.txt")
        logger = Logger(log_path)
        sys.stdout = logger
        print(f"\n--- Starting {run_prefix.upper()} ---")
        try:
            run_seed = BATCH_SEED + i
            setattr(params, 'RANDOM_SEED', run_seed)
            print(f"Using random seed: {run_seed}") 
            run_single_simulation(run_prefix, batch_run_dir)
        except Exception as e:
            traceback.print_exc()
        finally:
            if isinstance(sys.stdout, Logger): sys.stdout.close()
            sys.stdout = original_stdout
        print(f"--- Finished {run_prefix.upper()} ---")

    print(f"\nAll runs completed. Check the '{batch_run_dir}' directory.")
    