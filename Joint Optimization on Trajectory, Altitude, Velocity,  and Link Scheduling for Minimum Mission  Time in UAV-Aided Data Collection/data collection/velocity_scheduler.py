# velocity_scheduler.py (NEW, CORRECTED VERSION)
import numpy as np
from config import UAV_MAX_SPEED_VMAX, DATA_TRANSMISSION_RATE_R, DEFAULT_DATA_FILE_SIZE_MBITS

def solve_velocity_and_scheduling(trajectory, gu_locations, d_h):
    """
    Calculates the minimum mission time based on a clearer, physically accurate model.
    This replaces the previous BCD implementation to correctly handle hover time.
    The new model is: Mission Time = Flying Time + Hovering Time.
    """
    print("Step 3: Calculating mission time (Fly + Hover model)...")

    num_gus = len(gu_locations)
    if num_gus == 0:
        trajectory_length = np.linalg.norm(trajectory[-1] - trajectory[0])
        mission_time = trajectory_length / UAV_MAX_SPEED_VMAX
        return {
            "total_mission_time": mission_time,
            "min_flying_time": mission_time,
            "min_communication_time": 0,
            "hover_time": 0
        }

    # --- Step 1: Calculate Total Flying Time (T_fly) ---
    # This is the time required to traverse the trajectory at maximum speed.
    trajectory_length = np.sum(np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1))
    t_fly = trajectory_length / UAV_MAX_SPEED_VMAX

    # --- Step 2: Calculate Total Required Communication Time (T_comm_required) ---
    # This is the net time the communication module must be active.
    total_data_to_collect = num_gus * DEFAULT_DATA_FILE_SIZE_MBITS * 1e6 # in bits
    t_comm_required = total_data_to_collect / DATA_TRANSMISSION_RATE_R

    # --- Step 3: Calculate Available Communication Time During Flight ---
    # This is the most critical step. We calculate how much time the UAV spends
    # inside any communication circle while flying at V_max.
    t_comm_available_during_flight = 0.0
    
    # The trajectory is a series of waypoints [P0, P1, P2, ...].
    # P0 is Start, P_last is End. P1, P2... are the S/E points.
    # We analyze the segments between waypoints, e.g., P0->P1, P1->P2 etc.
    for i in range(len(trajectory) - 1):
        p_start = trajectory[i]
        p_end = trajectory[i+1]
        segment_vec = p_end - p_start
        segment_len = np.linalg.norm(segment_vec)
        if segment_len < 1e-6:
            continue # Skip zero-length segments

        # For each segment, check how much of it lies inside ANY GU's circle.
        # This is a simplified but effective check.
        # We check the midpoint of the segment. If it's inside a circle,
        # we assume the whole segment's flight time is available for communication.
        # This is a reasonable approximation for the "Fly-through" case.
        mid_point = (p_start + p_end) / 2
        is_in_any_circle = False
        for gu_loc in gu_locations:
            if np.linalg.norm(mid_point - gu_loc) <= d_h:
                is_in_any_circle = True
                break
        
        if is_in_any_circle:
            time_on_segment = segment_len / UAV_MAX_SPEED_VMAX
            t_comm_available_during_flight += time_on_segment

    # --- Step 4: Calculate Required Hover Time (T_hover) ---
    # If the time available during flight is not enough, the rest must be done by hovering.
    t_hover = max(0, t_comm_required - t_comm_available_during_flight)

    # --- Step 5: Calculate Final Mission Time ---
    t_mission = t_fly + t_hover

    print(f"Calculation finished.")
    print(f"  - Total flying time (T_fly): {t_fly:.2f} s")
    print(f"  - Time available for comm during flight: {t_comm_available_during_flight:.2f} s")
    print(f"  - Required comm time (T_comm): {t_comm_required:.2f} s")
    print(f"  - Calculated hover time (T_hover): {t_hover:.2f} s")
    
    time_stats = {
        "total_mission_time": t_mission,
        "min_flying_time": t_fly, # This name is now equivalent to t_fly
        "min_communication_time": t_comm_required, # This is t_comm_required
        "hover_time": t_hover
    }
    return time_stats