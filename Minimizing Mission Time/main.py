# ==============================================================================
#      Main Simulation Script (with Automated Batch Execution)
#
# File Objective:
# This script orchestrates automated, sequential simulation runs. For each run,
# it creates a unique timestamped directory to save all outputs, including
# console logs and generated trajectory plots.
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

# --- Helper Class and Functions ---

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

def run_single_simulation(output_dir: str):
    start_time = time.time()
    print("======================================================")
    print("      UAV Trajectory Optimization Comparison")
    print("======================================================")
    print("\n[Step 1/5] Initializing simulation environment...")

    print("Pre-calculating communication radius D...")
    traj_optimizer = TrajectoryOptimizer(params.__dict__)
    comm_radius = traj_optimizer.comm_radius_d

    sim_env = SimulationEnvironment(params, comm_radius=comm_radius)
    
    print(f"Environment created: {params.AREA_WIDTH}x{params.AREA_HEIGHT}m area with {params.NUM_GNS} GNs.")

    required_data_per_gn = 50 * 1e6 # 500 Mbits
    print(f"Data requirement per GN set to {required_data_per_gn / 1e6:.0f} Mbits.")
    
    print("\n[Step 2/5] Running Mission Allocation (Genetic Algorithm)...")
    
    ga_solver = MissionAllocationGA(
        gns=sim_env.gn_positions,
        num_uavs=params.NUM_UAVS,
        data_center_pos=sim_env.data_center_pos,
        transmission_radius_d=traj_optimizer.comm_radius_d,
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
        area_height=params.AREA_HEIGHT,
        save_path=os.path.join(output_dir, 'initial_routes.png')
    )
    
    print("\n[Step 3/5] Running Trajectory Optimizations for each UAV...")
    
    convex_planner = ConvexTrajectoryPlanner(
        gns=sim_env.gn_positions,
        data_center_pos=sim_env.data_center_pos,
        comm_radius=traj_optimizer.comm_radius_d
    )
    
    final_trajectories, uav_mission_times = {}, {}
    convex_trajectories, convex_path_lengths = {}, {}

    for uav_id, gn_indices_route in initial_assignment.items():
        if not gn_indices_route:
            print(f"{uav_id} has no assigned GNs. Mission time: 0s.")
            final_trajectories[uav_id], uav_mission_times[uav_id] = [], 0.0
            convex_trajectories[uav_id], convex_path_lengths[uav_id] = np.array([]), 0.0
            continue

        print(f"\n--- Optimizing for {uav_id} with sequence: {gn_indices_route} ---")
        
        print("  -> Running V-Shaped Trajectory Optimizer (JOFC Algorithm)...")
        previous_fop, uav_path_segments, current_uav_time = sim_env.data_center_pos, [], 0.0

        for i, gn_index in enumerate(gn_indices_route):
            sp, current_gn_coord = previous_fop, sim_env.gn_positions[gn_index]
            min_service_time_for_gn, best_leg_config = float('inf'), {}
            num_angles = 24
            angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
            fip_candidates = [current_gn_coord + traj_optimizer.comm_radius_d * np.array([np.cos(a), np.sin(a)]) for a in angles]
            fop_candidates = [current_gn_coord + traj_optimizer.comm_radius_d * np.array([np.cos(a), np.sin(a)]) for a in angles]
            
            for fip in fip_candidates:
                for fop in fop_candidates:
                    flight_time_in = np.linalg.norm(sp - fip) / params.UAV_MAX_SPEED
                    c_max = traj_optimizer.calculate_fm_max_capacity(fip, fop, current_gn_coord)
                    
                    if required_data_per_gn <= c_max:
                        optimal_oh, collection_time = traj_optimizer.find_optimal_fm_trajectory(
                            fip, fop, current_gn_coord, required_data_per_gn, is_symmetric=False)
                    else:
                        optimal_oh = current_gn_coord
                        collection_flight_time = (np.linalg.norm(fip - optimal_oh) + np.linalg.norm(optimal_oh - fop)) / params.UAV_MAX_SPEED
                        hover_time = (required_data_per_gn - c_max) / traj_optimizer.hover_datarate if traj_optimizer.hover_datarate > 0 else float('inf')
                        collection_time = collection_flight_time + hover_time
                    
                    service_time = flight_time_in + collection_time
                    if service_time < min_service_time_for_gn:
                        min_service_time_for_gn, best_leg_config = service_time, {'fip': fip, 'fop': fop, 'oh': optimal_oh, 'service_time': service_time}

            if not best_leg_config:
                print(f"WARNING: Fallback for GN {gn_index}.")
                oh = current_gn_coord
                flight_time_in = np.linalg.norm(sp - oh) / params.UAV_MAX_SPEED
                hover_time = required_data_per_gn / traj_optimizer.hover_datarate
                min_service_time_for_gn = flight_time_in + hover_time
                best_leg_config = {'fip': oh, 'fop': oh, 'oh': oh, 'service_time': min_service_time_for_gn}

            current_uav_time += best_leg_config['service_time']
            previous_fop = best_leg_config['fop']
            uav_path_segments.extend([{'type': 'flight', 'start': sp, 'end': best_leg_config['fip']}, {'type': 'collection', **best_leg_config}])
            print(f"    -> Optimized for GN {gn_index}. Service Time: {best_leg_config['service_time']:.2f}s. New FOP: {np.round(previous_fop, 1)}")

        final_flight_time = np.linalg.norm(previous_fop - sim_env.data_center_pos) / params.UAV_MAX_SPEED
        current_uav_time += final_flight_time
        uav_path_segments.append({'type': 'flight', 'start': previous_fop, 'end': sim_env.data_center_pos})
        
        final_trajectories[uav_id], uav_mission_times[uav_id] = uav_path_segments, current_uav_time
        print(f"  - Optimization complete for {uav_id}. Total mission time: {current_uav_time:.2f}s")

        print("  -> Running Convex Planner for the same sequence...")
        convex_result = convex_planner.plan_shortest_path_for_sequence(gn_indices_route)
        convex_trajectories[uav_id], convex_path_lengths[uav_id] = convex_result['path'], convex_result['length']
        print(f"     Convex Path Length: {convex_result['length']:.2f}m")

    print("\n[Step 4/5] Analyzing final results...")
    v_shaped_path_lengths = {}
    for uav_id, segments in final_trajectories.items():
        total_length = 0
        for s in segments:
            total_length += np.linalg.norm(s.get('end', s.get('fop')) - s.get('start', s.get('fip'))) if s['type']=='flight' else np.linalg.norm(s['oh']-s['fip']) + np.linalg.norm(s['fop']-s['oh'])
        v_shaped_path_lengths[uav_id] = total_length
    
    system_mct = max(uav_mission_times.values()) if uav_mission_times else 0
    total_execution_time = time.time() - start_time
    
    print("\n--- Simulation Summary ---")
    for uav_id in initial_assignment.keys():
        if uav_id in uav_mission_times:
            print(f"--- {uav_id} Results ---")
            print(f"  V-Shaped Mission Time: {uav_mission_times.get(uav_id, 0):.2f}s | Path Length: {v_shaped_path_lengths.get(uav_id, 0):.2f}m")
            convex_len = convex_path_lengths.get(uav_id, 0)
            convex_time = convex_len / params.UAV_MAX_SPEED if convex_len > 0 else 0
            print(f"  Convex Mission Time (theor.): {convex_time:.2f}s | Path Length: {convex_len:.2f}m")
    
    print(f"\nSystem Mission Completion Time (MCT) for V-Shaped: {system_mct:.2f}s")
    print(f"Total script execution time: {total_execution_time:.2f}s")
    
    print("\n[Step 5/5] Visualizing final combined trajectories...")
    vis.plot_final_comparison_trajectories(
        gns=sim_env.gn_positions,
        data_center_pos=sim_env.data_center_pos,
        v_shaped_trajectories=final_trajectories,
        convex_trajectories=convex_trajectories,
        area_width=params.AREA_WIDTH,
        area_height=params.AREA_HEIGHT,
        comm_radius=traj_optimizer.comm_radius_d,
        save_path=os.path.join(output_dir, 'final_trajectories.png')
    )
    
    print("\nSimulation finished successfully.")
    print("======================================================")

# --- Main Entry Point for Batch Execution ---

if __name__ == "__main__":
    
    NUMBER_OF_RUNS = 3  # <--- SET HOW MANY TIMES YOU WANT TO RUN
    BASE_RESULTS_DIR = "simulation_results"
    
    if not os.path.exists(BASE_RESULTS_DIR):
        os.makedirs(BASE_RESULTS_DIR)
        
    original_stdout = sys.stdout
    
    for i in range(NUMBER_OF_RUNS):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(BASE_RESULTS_DIR, f"run_{i+1}_{timestamp}")
        os.makedirs(run_dir)
        
        log_path = os.path.join(run_dir, "log.txt")
        logger = Logger(log_path)
        sys.stdout = logger
        
        print(f"--- Starting Run {i+1}/{NUMBER_OF_RUNS} ---")
        print(f"Results will be saved in: {run_dir}")
        
        try:
            # Use a different random seed for each run to get different GN placements
            # This makes each run a unique experiment
            current_seed = int(time.time()) + i
            print(f"Using random seed: {current_seed}")
            params.RANDOM_SEED = current_seed
            
            run_single_simulation(run_dir)
        except Exception as e:
            print("\n" + "="*20 + " ERROR " + "="*20)
            print(f"An error occurred during run {i+1}:")
            traceback.print_exc()
            print("="*47)
        finally:
            # Ensure the logger is closed and stdout is restored even if errors occur
            if isinstance(sys.stdout, Logger):
                sys.stdout.close()
            sys.stdout = original_stdout

        print(f"--- Finished Run {i+1}/{NUMBER_OF_RUNS} ---\n")

    print(f"All {NUMBER_OF_RUNS} runs completed. Check the '{BASE_RESULTS_DIR}' directory.")