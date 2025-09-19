# ==============================================================================
#                      UAV Trajectory Optimizer
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

# Import models from the local module
import models
from utility import dbm_to_watts, watts_to_dbm, db_to_linear # Helper functions for conversions

class TrajectoryOptimizer:
    """
    Encapsulates the logic for V-shaped trajectory optimization (JOFC).

    This class provides methods to find the optimal Flying-In Point (FIP),
    Flying-Out Point (FOP), and Hovering Point (OH) for a UAV to minimize
    the service time at a given Ground Node.
    """

    def __init__(self, params: Dict):
        """
        Initializes the optimizer with system parameters.

        Args:
            params (Dict): A configuration dictionary containing all system parameters.
        """
        self.params = params
        self.uav_altitude = params['UAV_ALTITUDE']
        self.uav_max_speed = params['UAV_MAX_SPEED']
        self.integration_steps = params['NUMERICAL_INTEGRATION_STEPS']

        # Pre-calculate constants for communication models
        self.gn_tx_power_watts = dbm_to_watts(params['GN_TRANSMIT_POWER_DBM'])
        noise_spectral_density_watts = dbm_to_watts(params['NOISE_POWER_SPECTRAL_DENSITY_DBM'] - 30) # dBm/Hz to W/Hz
        self.noise_power_watts = noise_spectral_density_watts * params['BANDWIDTH']
        
        # Pre-calculate the maximum horizontal communication radius (D)
        self.comm_radius_d = self._calculate_max_comm_radius()

    def _calculate_max_comm_radius(self) -> float:
        """
        Calculates the maximum horizontal communication radius D where SNR meets the threshold.
        This is done by solving for the distance in the SNR equation.
        """
        snr_thresh_linear = db_to_linear(self.params['SNR_THRESHOLD_DB'])
        
        # Simplified path loss model for this calculation (assuming worst-case NLoS for robustness)
        # We need to find distance `d_3d` such that SNR = threshold.
        # Pt / (sigma^2 * L) = snr_thresh  => L = Pt / (sigma^2 * snr_thresh)
        required_path_loss_linear = self.gn_tx_power_watts / (self.noise_power_watts * snr_thresh_linear)
        required_path_loss_db = 10 * np.log10(required_path_loss_linear)

        # Now, we invert the FSPL formula to get a rough distance estimate
        # FSPL = 20*log10(d) + 20*log10(f) + C => 20*log10(d) = FSPL - 20*log10(f) - C
        fc = self.params['CARRIER_FREQUENCY']
        c = models.SPEED_OF_LIGHT
        const_term = 20 * np.log10(fc) + 20 * np.log10((4 * np.pi) / c) + self.params['NLOS_ADDITIONAL_LOSS_DB']
        
        log_d = (required_path_loss_db - const_term) / 20.0
        d_3d = 10**log_d
        
        # d_3d^2 = H^2 + D^2 => D = sqrt(d_3d^2 - H^2)
        if d_3d**2 > self.uav_altitude**2:
            return np.sqrt(d_3d**2 - self.uav_altitude**2)
        else:
            # This would happen if the UAV is too high to ever achieve the SNR threshold
            return 0.0

    def _calculate_collected_data(self, start_point_2d: np.ndarray, end_point_2d: np.ndarray, gn_coord_2d: np.ndarray) -> float:
        """
        Numerically integrates to find the total data collected on a flight segment.
        (Implementation of Appendix C).
        """
        segment_vector = end_point_2d - start_point_2d
        segment_length = np.linalg.norm(segment_vector)
        if segment_length == 0:
            return 0.0
        
        travel_time = segment_length / self.uav_max_speed
        delta_t = travel_time / self.integration_steps
        total_data = 0.0

        for i in range(1, self.integration_steps + 1):
            # UAV position at the midpoint of the time step
            frac = (i - 0.5) / self.integration_steps
            uav_pos_2d = start_point_2d + frac * segment_vector
            
            # Calculate 3D distance and elevation angle
            dist_2d = np.linalg.norm(uav_pos_2d - gn_coord_2d)
            dist_3d = np.sqrt(dist_2d**2 + self.uav_altitude**2)
            elevation_angle = np.degrees(np.arcsin(self.uav_altitude / dist_3d))

            # Calculate instantaneous data rate using the models
            path_loss = models.calculate_path_loss(dist_3d, elevation_angle, self.params)
            snr = models.calculate_snr(self.gn_tx_power_watts, self.noise_power_watts, path_loss)
            rate = models.calculate_transmission_rate(snr, self.params['BANDWIDTH'])
            
            total_data += rate * delta_t
            
        return total_data

    def find_optimal_oh_and_time(self, fip: np.ndarray, fop: np.ndarray, gn_coord: np.ndarray, required_data: float) -> Tuple[np.ndarray, float]:
        """
        Finds the optimal OH location and minimum collection time for a given FIP and FOP.
        (Implementation of Algorithm 2 using Binary Search).
        """
        midpoint = (fip + fop) / 2.0
        bisector_vec = gn_coord - midpoint
        
        # Normalize the bisector vector
        if np.linalg.norm(bisector_vec) > 0:
            bisector_vec /= np.linalg.norm(bisector_vec)
        else: # FIP, FOP, and GN are collinear
            # A simple heuristic: move perpendicular to the FIP-FOP line
            perp_vec = np.array([fop[1] - fip[1], fip[0] - fop[0]])
            perp_vec /= np.linalg.norm(perp_vec)
            bisector_vec = perp_vec

        # Binary search for the optimal distance 'd' of OH from the midpoint
        low_d, high_d = 0.0, self.comm_radius_d 
        optimal_oh = midpoint # Default if no hovering is needed
        
        for _ in range(20): # 20 iterations for good precision
            mid_d = (low_d + high_d) / 2.0
            candidate_oh = midpoint + mid_d * bisector_vec
            
            # Calculate total data collected on the FIP -> OH -> FOP path
            data_leg1 = self._calculate_collected_data(fip, candidate_oh, gn_coord)
            data_leg2 = self._calculate_collected_data(candidate_oh, fop, gn_coord)
            total_collected = data_leg1 + data_leg2

            if total_collected >= required_data:
                # We have enough data, try a shorter path (smaller d)
                optimal_oh = candidate_oh
                high_d = mid_d
            else:
                # Not enough data, need a longer path closer to the GN (larger d)
                low_d = mid_d
        
        # Calculate the collection time for the optimal V-shaped path
        path_length = np.linalg.norm(fip - optimal_oh) + np.linalg.norm(optimal_oh - fop)
        min_collection_time = path_length / self.uav_max_speed
        
        return optimal_oh, min_collection_time

    def run_jofc_for_gn(self, prev_fop: np.ndarray, current_gn_coord: np.ndarray, next_gn_coord: np.ndarray, required_data: float) -> Dict:
        """
        Performs Joint Optimization of Flight and Collection (JOFC) for a single GN.
        (Implementation of Algorithm 3 using Grid Search).
        """
        min_total_service_time = float('inf')
        best_trajectory = {}
        
        # Discretize the search space on the communication circle
        num_angles = 24 # Number of candidate points on the circle
        angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
        candidate_points = np.array([
            current_gn_coord + self.comm_radius_d * np.array([np.cos(theta), np.sin(theta)])
            for theta in angles
        ])
        
        # Grid search over all pairs of FIP and FOP candidates
        for fip in candidate_points:
            for fop in candidate_points:
                # Calculate flight time from previous FOP to this FIP
                flight_time_in = np.linalg.norm(prev_fop - fip) / self.uav_max_speed
                
                # Find the best collection time for this FIP-FOP pair
                optimal_oh, min_collection_time = self.find_optimal_oh_and_time(
                    fip, fop, current_gn_coord, required_data
                )
                
                # Total service time for this GN (flight in + collection)
                # The flight out time is handled by the next GN's optimization
                total_time = flight_time_in + min_collection_time
                
                # Update if we found a better trajectory
                if total_time < min_total_service_time:
                    min_total_service_time = total_time
                    best_trajectory = {
                        'fip': fip,
                        'fop': fop,
                        'oh': optimal_oh,
                        'service_time': min_total_service_time
                    }
        
        return best_trajectory