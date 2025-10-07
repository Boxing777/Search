# ==============================================================================
#      Main Simulation Script (Final Paper-Faithful JOFC Implementation)
#
# File Objective:
# This final version uses the most direct and faithful interpretation of the
# paper's Algorithm 3 (JOFC), employing a grid search over FIP and FOP
# candidates on the communication circle to greedily optimize the service
# time for each GN sequentially.
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

def get_circle_intersections(p1: np.ndarray, r1: float, p2: np.ndarray, r2: float) -> list:
    """Calculates the intersection points of two circles."""
    d = np.linalg.norm(p1 - p2)
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
        return []

    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(max(0, r1**2 - a**2))
    
    p_mid = p1 + a * (p2 - p1) / d
    x_mid, y_mid = p_mid[0], p_mid[1]
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]

    inter1 = (x_mid + h * dy / d, y_mid - h * dx / d)
    inter2 = (x_mid - h * dy / d, y_mid + h * dx / d)
    
    if np.allclose(inter1, inter2):
        return [np.array(inter1)]
    return [np.array(inter1), np.array(inter2)]


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

    required_data_per_gn = 100 * 1e6 # 100 Mbits
    print(f"Data requirement per GN set to {required_data_per_gn / 1e6:.0f} Mbits.")
    
    traj_optimizer = TrajectoryOptimizer(params.__dict__)
    
    # --- Step 3: Phase 1 - Run Mission Allocation (Genetic Algorithm) ---
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
        area_height=params.AREA_HEIGHT
    )
    
    # --- Step 4: Phase 2 - Run Trajectory Optimizations ---
    print("\n[Step 3/5] Running Trajectory Optimizations for each UAV...")
    
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

        for i, gn_index in enumerate(gn_indices_route):
            # <<< START OF FINAL, PAPER-FAITHFUL JOFC IMPLEMENTATION >>>
            sp = previous_fop
            current_gn_coord = sim_env.gn_positions[gn_index]
            
            min_service_time_for_gn = float('inf')
            best_leg_config = {}

            # 1. Discretize the search space for FIP and FOP on the communication circle.
            num_angles = 24  # Or can be a parameter
            angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
            
            fip_candidates = [current_gn_coord + traj_optimizer.comm_radius_d * np.array([np.cos(a), np.sin(a)]) for a in angles]
            fop_candidates = [current_gn_coord + traj_optimizer.comm_radius_d * np.array([np.cos(a), np.sin(a)]) for a in angles]
            
            # 2. Grid search for the best (FIP, FOP) pair.
            for fip in fip_candidates:
                for fop in fop_candidates:
                    
                    # Objective: Minimize service_time = flight_time_in + collection_time
                    flight_time_in = np.linalg.norm(sp - fip) / params.UAV_MAX_SPEED
                    
                    c_max = traj_optimizer.calculate_fm_max_capacity(fip, fop, current_gn_coord)
                    
                    optimal_oh = None
                    collection_time = float('inf')
                    
                    if required_data_per_gn <= c_max:
                        # FM Mode
                        optimal_oh, collection_time = traj_optimizer.find_optimal_fm_trajectory(
                            fip, fop, current_gn_coord, required_data_per_gn, is_symmetric=False)
                    else:
                        # HM Mode
                        optimal_oh = current_gn_coord
                        collection_flight_time = (np.linalg.norm(fip - optimal_oh) + np.linalg.norm(optimal_oh - fop)) / params.UAV_MAX_SPEED
                        hover_data_needed = required_data_per_gn - c_max
                        hover_time = hover_data_needed / traj_optimizer.hover_datarate if traj_optimizer.hover_datarate > 0 else float('inf')
                        collection_time = collection_flight_time + hover_time
                    
                    service_time = flight_time_in + collection_time
                    
                    if service_time < min_service_time_for_gn:
                        min_service_time_for_gn = service_time
                        best_leg_config = {
                            'fip': fip,
                            'fop': fop,
                            'oh': optimal_oh,
                            'service_time': service_time
                        }

            if not best_leg_config:
                print(f"WARNING: Could not find a valid trajectory for GN {gn_index}. Using fallback.")
                oh = current_gn_coord
                flight_time_in = np.linalg.norm(sp - oh) / params.UAV_MAX_SPEED
                hover_time = required_data_per_gn / traj_optimizer.hover_datarate
                min_service_time_for_gn = flight_time_in + hover_time
                best_leg_config = {'fip': oh, 'fop': oh, 'oh': oh, 'service_time': min_service_time_for_gn}

            # Update mission based on the best found configuration for THIS GN
            current_uav_time += best_leg_config['service_time']
            previous_fop = best_leg_config['fop']
            
            uav_path_segments.append({'type': 'flight', 'start': sp, 'end': best_leg_config['fip']})
            uav_path_segments.append({'type': 'collection', **best_leg_config})
            
            print(f"    -> Optimized for GN {gn_index}. Service Time: {best_leg_config['service_time']:.2f}s. New FOP: {np.round(previous_fop, 1)}")
            # <<< END OF FINAL, PAPER-FAITHFUL JOFC IMPLEMENTATION >>>

        # Add the final flight back to the data center
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

if __name__ == "__main__":
    main()