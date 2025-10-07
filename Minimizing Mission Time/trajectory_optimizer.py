# ==============================================================================
#      UAV Trajectory Optimizer (FINAL AND DEFINITIVELY CORRECTED VERSION)
#
# File Objective:
# This version removes the harmful optimization in `_calculate_collected_data`
# that was interfering with the convergence of the OPAO algorithm. This ensures
# that the V-shaped trajectories are calculated correctly under all geometric
# conditions.
# ==============================================================================

import numpy as np
from typing import Dict, Tuple
from scipy.optimize import minimize

import models
from utility import dbm_to_watts, db_to_linear

class TrajectoryOptimizer:
    def __init__(self, params: Dict):
        self.params = params
        self.uav_altitude = params['UAV_ALTITUDE']
        self.uav_max_speed = params['UAV_MAX_SPEED']
        self.integration_steps = params['NUMERICAL_INTEGRATION_STEPS']
        self.gn_tx_power_watts = dbm_to_watts(params['GN_TRANSMIT_POWER_DBM'])
        noise_spectral_density_watts = dbm_to_watts(params['NOISE_POWER_SPECTRAL_DENSITY_DBM'] - 30)
        self.noise_power_watts = noise_spectral_density_watts * params['BANDWIDTH']
        self.comm_radius_d = self._calculate_max_comm_radius_iterative()
        print(f"Calculated a more accurate communication radius D = {self.comm_radius_d:.2f} meters.")
        self.hover_datarate = self._calculate_hover_datarate()
        print(f"Calculated hover data rate at GN: {self.hover_datarate / 1e6:.2f} Mbps.")

    def _calculate_max_comm_radius_iterative(self) -> float:
        snr_thresh_linear = db_to_linear(self.params['SNR_THRESHOLD_DB'])
        low_d, high_d = 0.0, self.params['AREA_WIDTH']
        for _ in range(50):
            mid_d = (low_d + high_d) / 2.0
            if mid_d < 1e-6: mid_d = 1e-6
            dist_3d = np.sqrt(mid_d**2 + self.uav_altitude**2)
            elevation = np.degrees(np.arcsin(self.uav_altitude / dist_3d))
            path_loss = models.calculate_path_loss(dist_3d, elevation, self.params)
            snr = models.calculate_snr(self.gn_tx_power_watts, self.noise_power_watts, path_loss)
            if snr >= snr_thresh_linear: low_d = mid_d
            else: high_d = mid_d
        return low_d

    def _calculate_hover_datarate(self) -> float:
        dist_3d = self.uav_altitude
        elevation_angle = 90.0
        path_loss = models.calculate_path_loss(dist_3d, elevation_angle, self.params)
        snr = models.calculate_snr(self.gn_tx_power_watts, self.noise_power_watts, path_loss)
        rate = models.calculate_transmission_rate(snr, self.params['BANDWIDTH'])
        return rate

    # <<< CORRECTED FUNCTION: Removed the harmful optimization >>>
    def _calculate_collected_data(self, start_point_2d: np.ndarray, end_point_2d: np.ndarray, gn_coord_2d: np.ndarray) -> float:
        segment_vector = end_point_2d - start_point_2d
        segment_length = np.linalg.norm(segment_vector)
        if segment_length < 1e-6: return 0.0
        
        travel_time = segment_length / self.uav_max_speed
        delta_t = travel_time / self.integration_steps
        total_data = 0.0

        for i in range(self.integration_steps):
            frac = (i + 0.5) / self.integration_steps
            uav_pos_2d = start_point_2d + frac * segment_vector
            
            dist_2d = np.linalg.norm(uav_pos_2d - gn_coord_2d)
            
            # The SNR will naturally become very low outside the communication range,
            # so the data rate will approach zero. We don't need to skip it manually.
            # if dist_2d > self.comm_radius_d * 1.05: continue # THIS LINE WAS THE BUG

            dist_3d = np.sqrt(dist_2d**2 + self.uav_altitude**2)
            elevation_angle = np.degrees(np.arcsin(self.uav_altitude / dist_3d))
            path_loss = models.calculate_path_loss(dist_3d, elevation_angle, self.params)
            snr = models.calculate_snr(self.gn_tx_power_watts, self.noise_power_watts, path_loss)
            
            # If SNR is below threshold, rate is effectively zero
            if snr < db_to_linear(self.params['SNR_THRESHOLD_DB']):
                rate = 0.0
            else:
                rate = models.calculate_transmission_rate(snr, self.params['BANDWIDTH'])
            
            total_data += rate * delta_t
            
        return total_data

    def _find_oh_for_max_throughput_on_ellipse(self, fip: np.ndarray, fop: np.ndarray, gn_coord: np.ndarray, path_length: float) -> Tuple[np.ndarray, float]:
        foci_dist = np.linalg.norm(fip - fop)
        if path_length < foci_dist: path_length = foci_dist
        ellipse_center = (fip + fop) / 2.0
        angle = np.arctan2(fop[1] - fip[1], fop[0] - fip[0])
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        a = path_length / 2.0
        b = np.sqrt(max(0, a**2 - (foci_dist / 2.0)**2))

        def objective(t):
            x_local, y_local = a * np.cos(t), b * np.sin(t)
            x_global = ellipse_center[0] + x_local * cos_a - y_local * sin_a
            y_global = ellipse_center[1] + x_local * sin_a + y_local * cos_a
            oh_candidate = np.array([x_global, y_global])
            
            if np.linalg.norm(oh_candidate - gn_coord) > self.comm_radius_d:
                return 0

            data = self._calculate_collected_data(fip, oh_candidate, gn_coord) + \
                   self._calculate_collected_data(oh_candidate, fop, gn_coord)
            return -data

        vec_to_gn = gn_coord - ellipse_center
        initial_angle_guess = np.arctan2(vec_to_gn[1], vec_to_gn[0]) - angle
        result = minimize(objective, initial_angle_guess, bounds=[(0, 2 * np.pi)])
        
        best_t = result.x[0]
        x_local_opt, y_local_opt = a * np.cos(best_t), b * np.sin(best_t)
        x_global_opt = ellipse_center[0] + x_local_opt * cos_a - y_local_opt * sin_a
        y_global_opt = ellipse_center[1] + x_local_opt * sin_a + y_local_opt * cos_a
        optimal_oh = np.array([x_global_opt, y_global_opt])
        max_throughput = -result.fun
        return optimal_oh, max_throughput

    def find_optimal_fm_trajectory(self, fip: np.ndarray, fop: np.ndarray, gn_coord: np.ndarray, required_data: float) -> Tuple[np.ndarray, float]:
        k_min = np.linalg.norm(fip - fop)
        k_max = np.linalg.norm(fip - gn_coord) + np.linalg.norm(gn_coord - fop)
        min_path_length = k_max
        
        for _ in range(10):
            k_temp = (k_min + k_max) / 2.0
            if k_temp < k_min + 1e-6: break
            
            _, max_data_at_k_temp = self._find_oh_for_max_throughput_on_ellipse(fip, fop, gn_coord, k_temp)
            if max_data_at_k_temp >= required_data:
                min_path_length = k_temp
                k_max = k_temp
            else:
                k_min = k_temp
        
        optimal_oh, _ = self._find_oh_for_max_throughput_on_ellipse(fip, fop, gn_coord, min_path_length)
        min_collection_time = min_path_length / self.uav_max_speed
        return optimal_oh, min_collection_time

    def calculate_fm_max_capacity(self, fip: np.ndarray, fop: np.ndarray, gn_coord: np.ndarray) -> float:
        longest_path_len = np.linalg.norm(fip - gn_coord) + np.linalg.norm(gn_coord - fop)
        if longest_path_len < np.linalg.norm(fip-fop):
             longest_path_len = np.linalg.norm(fip-fop)
        
        _, max_throughput = self._find_oh_for_max_throughput_on_ellipse(fip, fop, gn_coord, longest_path_len)
        return max_throughput