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
from datetime import datetime
import traceback

# Import custom modules
import parameters as params
from environment import SimulationEnvironment
from mission_allocation_ga import MissionAllocationGA
from trajectory_optimizer import TrajectoryOptimizer
from convex_trajectory_planner import ConvexTrajectoryPlanner
import visualizer as vis

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
    comm_radius = traj_optimizer.comm_radius_d
    sim_env = SimulationEnvironment(params, comm_radius=comm_radius)
    
    print(f"Environment created: {params.AREA_WIDTH}x{params.AREA_HEIGHT}m area with {params.NUM_GNS} GNs.")

    required_data_per_gn = 80 * 1e6 # Set to a high value to see V-shapes data size
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
    
    final_trajectories, uav_mission_times = {}, {}
    convex_trajectories, convex_path_lengths = {}, {}
    convex_mission_times = {}

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
            
            # --- Check for Overlapping Case ---
            # Is the start point (previous FOP) already inside the current GN's comm range?
            if np.linalg.norm(sp - current_gn_coord) <= comm_radius:
                # --- SPECIAL HANDLING FOR OVERLAPPING REGIONS (Paper's explicit instruction) ---
                # FIP is forced to be the same as SP. t_flight_in is 0.
                fip = sp
                flight_time_in = 0.0
                
                # The search problem degenerates to a 1D search for the best FOP.
                num_angles = 36 # More points for 1D search
                angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
                fop_candidates = [current_gn_coord + comm_radius * np.array([np.cos(a), np.sin(a)]) for a in angles]

                for fop in fop_candidates:
                    # Objective function is now simpler
                    c_max = traj_optimizer.calculate_fm_max_capacity(fip, fop, current_gn_coord)
                    
                    if required_data_per_gn <= c_max:
                        optimal_oh, collection_time = traj_optimizer.find_optimal_fm_trajectory(
                            fip, fop, current_gn_coord, required_data_per_gn, is_overlapping=True)
                    else:
                        optimal_oh = current_gn_coord
                        collection_flight_time = (np.linalg.norm(fip - optimal_oh) + np.linalg.norm(optimal_oh - fop)) / params.UAV_MAX_SPEED
                        hover_time = (required_data_per_gn - c_max) / traj_optimizer.hover_datarate if traj_optimizer.hover_datarate > 0 else float('inf')
                        collection_time = collection_flight_time + hover_time

                    flight_time_out = np.linalg.norm(next_target_anchor - fop) / params.UAV_MAX_SPEED
                    total_leg_time = collection_time + flight_time_out # t_flight_in is 0

                    if total_leg_time < min_total_leg_time:
                        min_total_leg_time = total_leg_time
                        best_leg_config = {
                            'fip': fip, 'fop': fop, 'oh': optimal_oh,
                            'service_time': collection_time # For overlapping, service time is just collection time
                        }
            else:
                # --- STANDARD HANDLING FOR NON-OVERLAPPING REGIONS (Our previous corrected logic) ---
                num_angles = 36
                angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
                fip_candidates = [current_gn_coord + comm_radius * np.array([np.cos(a), np.sin(a)]) for a in angles]
                fop_candidates = [current_gn_coord + comm_radius * np.array([np.cos(a), np.sin(a)]) for a in angles]
                
                for fip in fip_candidates:
                    for fop in fop_candidates:
                        flight_time_in = np.linalg.norm(sp - fip) / params.UAV_MAX_SPEED
                        c_max = traj_optimizer.calculate_fm_max_capacity(fip, fop, current_gn_coord)
                        
                        if required_data_per_gn <= c_max:
                            optimal_oh, collection_time = traj_optimizer.find_optimal_fm_trajectory(
                                fip, fop, current_gn_coord, required_data_per_gn, is_overlapping=False)
                        else:
                            optimal_oh = current_gn_coord
                            collection_flight_time = (np.linalg.norm(fip - optimal_oh) + np.linalg.norm(optimal_oh - fop)) / params.UAV_MAX_SPEED
                            hover_time = (required_data_per_gn - c_max) / traj_optimizer.hover_datarate if traj_optimizer.hover_datarate > 0 else float('inf')
                            collection_time = collection_flight_time + hover_time

                        flight_time_out = np.linalg.norm(next_target_anchor - fop) / params.UAV_MAX_SPEED
                        total_leg_time = flight_time_in + collection_time + flight_time_out
                        
                        if total_leg_time < min_total_leg_time:
                            min_total_leg_time = total_leg_time
                            best_leg_config = {
                                'fip': fip, 'fop': fop, 'oh': optimal_oh,
                                'service_time': flight_time_in + collection_time
                            }

            if not best_leg_config:
                print(f"WARNING: Fallback for GN {gn_index}.")
                oh = current_gn_coord
                flight_time_in = np.linalg.norm(sp - oh) / params.UAV_MAX_SPEED
                hover_time = required_data_per_gn / traj_optimizer.hover_datarate
                min_service_time_for_gn = flight_time_in + hover_time
                best_leg_config = {'fip': oh, 'fop': oh, 'oh': oh, 'service_time': min_service_time_for_gn}

            # Accumulate the actual time spent (service time for this GN)
            current_uav_time += best_leg_config['service_time']
            # The FOP from the best configuration becomes the SP for the next iteration
            previous_fop = best_leg_config['fop']
            
            uav_path_segments.extend([{'type': 'flight', 'start': sp, 'end': best_leg_config['fip']},
                                      {'type': 'collection', **best_leg_config}])
            
            print(f"    -> Optimized for GN {gn_index}. Service Time: {best_leg_config['service_time']:.2f}s. New FOP: {np.round(previous_fop, 1)}")

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
            if data_shortfall > 0:
                hover_time = data_shortfall / traj_optimizer.hover_datarate if traj_optimizer.hover_datarate > 0 else float('inf')
                convex_total_hover_time += hover_time
        
        # The total fair time is the flight time plus any required hover time
        convex_flight_time = convex_result['length'] / params.UAV_MAX_SPEED
        convex_actual_mission_time = convex_flight_time + convex_total_hover_time
        convex_mission_times[uav_id] = convex_actual_mission_time
        
        print(f"     Convex Path -> Flight Time: {convex_flight_time:.2f}s, Required Hover Time: {convex_total_hover_time:.2f}s, TOTAL FAIR TIME: {convex_actual_mission_time:.2f}s")

    print("\n[Step 4/5] Analyzing final results...")
    v_shaped_path_lengths = {}
    for uav_id, segments in final_trajectories.items():
        total_length = 0
        for s in segments:
            total_length += np.linalg.norm(s.get('end', s.get('fop')) - s.get('start', s.get('fip'))) if s['type']=='flight' else np.linalg.norm(s['oh']-s['fip']) + np.linalg.norm(s['fop']-s['oh'])
        v_shaped_path_lengths[uav_id] = total_length
    
    system_mct_v_shaped = max(uav_mission_times.values()) if uav_mission_times else 0
    system_mct_convex = max(convex_mission_times.values()) if convex_mission_times else 0
    total_execution_time = time.time() - start_time
    
    print("\n--- Simulation Summary ---")
    for uav_id in initial_assignment.keys():
        if uav_id in uav_mission_times:
            print(f"--- {uav_id} Results ---")
            print(f"  V-Shaped Mission Time: {uav_mission_times.get(uav_id, 0):.2f}s | Path Length: {v_shaped_path_lengths.get(uav_id, 0):.2f}m")
            print(f"  Convex Mission Time:   {convex_mission_times.get(uav_id, 0):.2f}s | Path Length: {convex_path_lengths.get(uav_id, 0):.2f}m")
    
    print(f"\nSystem Mission Completion Time (MCT) for V-Shaped: {system_mct_v_shaped:.2f}s")
    print(f"System Mission Completion Time (MCT) for Convex:   {system_mct_convex:.2f}s")
    print(f"Total script execution time: {total_execution_time:.2f}s")
    
    print("\n[Step 5/5] Visualizing final combined trajectories...")
    vis.plot_final_comparison_trajectories(
        gns=sim_env.gn_positions, data_center_pos=sim_env.data_center_pos,
        v_shaped_trajectories=final_trajectories, convex_trajectories=convex_trajectories,
        area_width=params.AREA_WIDTH, area_height=params.AREA_HEIGHT,
        comm_radius=traj_optimizer.comm_radius_d,
        save_path=os.path.join(output_dir, f'{run_prefix}_final_trajectories.png')
    )
    
    print("\nSimulation finished successfully.")
    print("======================================================")

# --- Main Entry Point for Batch Execution (No changes from your version) ---
if __name__ == "__main__":
    NUMBER_OF_RUNS = 3 # Set to 1 for testing the fix
    BASE_RESULTS_DIR = "simulation_results"
    
    if not os.path.exists(BASE_RESULTS_DIR):
        os.makedirs(BASE_RESULTS_DIR)
    
    batch_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_run_dir = os.path.join(BASE_RESULTS_DIR, f"run_{batch_timestamp}")
    os.makedirs(batch_run_dir)
    
    print(f"Starting batch of {NUMBER_OF_RUNS} runs.")
    print(f"All results will be saved in: {batch_run_dir}")
    
    original_stdout = sys.stdout
    
    for i in range(NUMBER_OF_RUNS):
        run_prefix = f"run_{i+1}"
        log_path = os.path.join(batch_run_dir, f"{run_prefix}_log.txt")
        logger = Logger(log_path)
        sys.stdout = logger
        print(f"\n--- Starting {run_prefix.upper()} ---")
        try:
            setattr(params, 'RANDOM_SEED', int(time.time()) + i)
            run_single_simulation(run_prefix, batch_run_dir)
        except Exception as e:
            traceback.print_exc()
        finally:
            if isinstance(sys.stdout, Logger): sys.stdout.close()
            sys.stdout = original_stdout
        print(f"--- Finished {run_prefix.upper()} ---")

    print(f"\nAll runs completed. Check the '{batch_run_dir}' directory.")