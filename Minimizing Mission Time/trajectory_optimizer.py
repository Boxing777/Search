# ==============================================================================
#                      UAV Trajectory Optimizer (MODIFIED)
#
# File Objective:
# This file is the computational core of the project, implementing the low-level,
# detailed trajectory planning for a single UAV visiting a single Ground Node (GN).
# It implements Algorithm 2 (OH Positioning) and Algorithm 3 (JOFC) from the
# research paper to find the optimal V-shaped path that minimizes the service
# time for a GN.
# ==============================================================================

# Import necessary libraries
import numpy as np
from typing import Dict, Tuple

# Import models and utilities
import models
from utility import dbm_to_watts, db_to_linear

class TrajectoryOptimizer:
    """
    Encapsulates the logic for V-shaped trajectory optimization (JOFC).
    """

    def __init__(self, params: Dict):
        """
        Initializes the optimizer with system parameters.
        """
        self.params = params
        self.uav_altitude = params['UAV_ALTITUDE']
        self.uav_max_speed = params['UAV_MAX_SPEED']
        self.integration_steps = params['NUMERICAL_INTEGRATION_STEPS']

        # Pre-calculate constants for communication models
        self.gn_tx_power_watts = dbm_to_watts(params['GN_TRANSMIT_POWER_DBM'])
        noise_spectral_density_watts = dbm_to_watts(params['NOISE_POWER_SPECTRAL_DENSITY_DBM'] - 30)
        self.noise_power_watts = noise_spectral_density_watts * params['BANDWIDTH']
        
        self.comm_radius_d = self._calculate_max_comm_radius_iterative()
        print(f"Calculated a more accurate communication radius D = {self.comm_radius_d:.2f} meters.")
        
        # <<< NEW >>> Pre-calculate hover rate for efficiency
        self.hover_datarate = self._calculate_hover_datarate()
        print(f"Calculated hover data rate at GN: {self.hover_datarate / 1e6:.2f} Mbps.")


    def _calculate_max_comm_radius_iterative(self) -> float:
        """
        Calculates a more accurate max horizontal communication radius D where SNR
        meets the threshold, using an iterative binary search approach.
        """
        snr_thresh_linear = db_to_linear(self.params['SNR_THRESHOLD_DB'])
        
        low_d, high_d = 0.0, self.params['AREA_WIDTH'] 

        for _ in range(50): 
            mid_d = (low_d + high_d) / 2.0
            if mid_d < 1e-6:
                mid_d = 1e-6

            dist_3d = np.sqrt(mid_d**2 + self.uav_altitude**2)
            elevation = np.degrees(np.arcsin(self.uav_altitude / dist_3d))
            
            path_loss = models.calculate_path_loss(dist_3d, elevation, self.params)
            snr = models.calculate_snr(self.gn_tx_power_watts, self.noise_power_watts, path_loss)
            
            if snr >= snr_thresh_linear:
                low_d = mid_d
            else:
                high_d = mid_d
        
        return low_d
    
    # <<< NEW FUNCTION >>>
    def _calculate_hover_datarate(self) -> float:
        """Calculates the data rate when UAV is hovering directly above a GN."""
        dist_3d = self.uav_altitude
        elevation_angle = 90.0
        
        path_loss = models.calculate_path_loss(dist_3d, elevation_angle, self.params)
        snr = models.calculate_snr(self.gn_tx_power_watts, self.noise_power_watts, path_loss)
        rate = models.calculate_transmission_rate(snr, self.params['BANDWIDTH'])
        return rate

    def _calculate_collected_data(self, start_point_2d: np.ndarray, end_point_2d: np.ndarray, gn_coord_2d: np.ndarray) -> float:
        """
        Numerically integrates to find the total data collected on a flight segment.
        """
        segment_vector = end_point_2d - start_point_2d
        segment_length = np.linalg.norm(segment_vector)
        if segment_length < 1e-6:
            return 0.0
        
        travel_time = segment_length / self.uav_max_speed
        delta_t = travel_time / self.integration_steps
        total_data = 0.0

        for i in range(self.integration_steps):
            frac = (i + 0.5) / self.integration_steps 
            uav_pos_2d = start_point_2d + frac * segment_vector
            
            dist_2d = np.linalg.norm(uav_pos_2d - gn_coord_2d)
            dist_3d = np.sqrt(dist_2d**2 + self.uav_altitude**2)
            elevation_angle = np.degrees(np.arcsin(self.uav_altitude / dist_3d))

            path_loss = models.calculate_path_loss(dist_3d, elevation_angle, self.params)
            snr = models.calculate_snr(self.gn_tx_power_watts, self.noise_power_watts, path_loss)
            rate = models.calculate_transmission_rate(snr, self.params['BANDWIDTH'])
            
            total_data += rate * delta_t
            
        return total_data

    def _find_optimal_oh_for_fm(self, fip: np.ndarray, fop: np.ndarray, gn_coord: np.ndarray, required_data: float) -> Tuple[np.ndarray, float]:
        """
        Finds the optimal OH location and minimum collection time for FM mode.
        (Previously named find_optimal_oh_and_time)
        """
        midpoint = (fip + fop) / 2.0
        vec_to_gn = gn_coord - midpoint
        
        dist_mid_to_gn = np.linalg.norm(vec_to_gn)
        
        if dist_mid_to_gn > 1e-6:
            bisector_vec = vec_to_gn / dist_mid_to_gn
        else:
            # If midpoint is on GN, bisector is perpendicular to FIP-FOP line
            perp_vec = np.array([fop[1] - fip[1], fip[0] - fop[0]])
            if np.linalg.norm(perp_vec) > 1e-6:
                 bisector_vec = perp_vec / np.linalg.norm(perp_vec)
            else: # FIP and FOP are the same, no direction
                bisector_vec = np.array([1, 0])


        # Search for OH on the bisector line segment towards the GN
        low_d, high_d = 0.0, dist_mid_to_gn
        optimal_oh = midpoint 
        
        # Binary search to find the minimum distance from midpoint (and thus shortest path)
        # that satisfies the data requirement.
        for _ in range(20):
            mid_d = (low_d + high_d) / 2.0
            candidate_oh = midpoint + mid_d * bisector_vec
            
            data_leg1 = self._calculate_collected_data(fip, candidate_oh, gn_coord)
            data_leg2 = self._calculate_collected_data(candidate_oh, fop, gn_coord)
            total_collected = data_leg1 + data_leg2

            if total_collected >= required_data:
                # This path works, try a shorter one (smaller d)
                optimal_oh = candidate_oh
                high_d = mid_d
            else:
                # This path is too short, need to go further (larger d)
                low_d = mid_d
        
        path_length = np.linalg.norm(fip - optimal_oh) + np.linalg.norm(optimal_oh - fop)
        min_collection_time = path_length / self.uav_max_speed
        
        return optimal_oh, min_collection_time
    
    # <<< NEW HELPER FUNCTION >>>
    def _calculate_fm_max_throughput(self, fip: np.ndarray, fop: np.ndarray, gn_coord: np.ndarray) -> float:
        """
        Calculates C_f_max: the maximum data collected in FM by flying through the GN.
        """
        data_in = self._calculate_collected_data(fip, gn_coord, gn_coord)
        data_out = self._calculate_collected_data(gn_coord, fop, gn_coord)
        return data_in + data_out

    # <<< MODIFIED MAIN FUNCTION >>>
    def run_jofc_for_gn(self, prev_fop: np.ndarray, current_gn_coord: np.ndarray, next_gn_coord: np.ndarray, required_data: float) -> Dict:
        """
        Performs Joint Optimization of Flight and Collection (JOFC) for a single GN.
        This now includes the logic to decide between Flying Mode (FM) and Hovering Mode (HM).
        """
        min_total_service_time = float('inf')
        best_trajectory = {}
        
        num_angles = 8
        angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
        candidate_points = np.array([
            current_gn_coord + self.comm_radius_d * np.array([np.cos(theta), np.sin(theta)])
            for theta in angles
        ])
        
        for fip in candidate_points:
            for fop in candidate_points:
                flight_time_in = np.linalg.norm(prev_fop - fip) / self.uav_max_speed
                
                # --- FM vs HM Decision Logic ---
                c_f_max = self._calculate_fm_max_throughput(fip, fop, current_gn_coord)
                
                if required_data <= c_f_max:
                    # --- Flying Mode (FM) Case ---
                    mode = 'FM'
                    optimal_oh, collection_time = self._find_optimal_oh_for_fm(
                        fip, fop, current_gn_coord, required_data
                    )
                    
                else:
                    # --- Hovering Mode (HM) Case ---
                    mode = 'HM'
                    oh_hm = current_gn_coord
                    
                    # Time spent flying during the collection phase (FIP -> GN -> FOP)
                    collection_flight_time = (np.linalg.norm(fip - oh_hm) + np.linalg.norm(oh_hm - fop)) / self.uav_max_speed
                    
                    # Data that must be collected while hovering
                    hover_data_needed = required_data - c_f_max
                    
                    # Time spent hovering
                    hover_time = hover_data_needed / self.hover_datarate if self.hover_datarate > 0 else float('inf')
                    
                    collection_time = collection_flight_time + hover_time
                    optimal_oh = oh_hm

                # --- Update Best Trajectory Found So Far ---
                total_time = flight_time_in + collection_time
                
                if total_time < min_total_service_time:
                    min_total_service_time = total_time
                    best_trajectory = {
                        'fip': fip,
                        'fop': fop,
                        'oh': optimal_oh,
                        'service_time': min_total_service_time,
                        'mode': mode # Store the mode used
                    }
        
        return best_trajectory