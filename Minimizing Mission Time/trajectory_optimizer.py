# ==============================================================================
#      UAV Trajectory Optimizer (Final Geometrically Correct Version)
#
# File Objective:
# This definitive version correctly distinguishes between non-overlapping and
# overlapping scenarios. For non-overlapping cases (where FIP and FOP are on
# the same circle), it uses the efficient mid-perpendicular search (Lemma 1).
# For overlapping cases (where FIP and FOP are on different circles), it
# faithfully implements the general ellipse-based search (Algorithm 2).
# ==============================================================================

import numpy as np
from typing import Dict, Tuple

import models
from utility import dbm_to_watts, db_to_linear

class TrajectoryOptimizer:
    def __init__(self, params: Dict):
        self.params = params
        self.uav_altitude = params['UAV_ALTITUDE']
        self.uav_max_speed = params['UAV_MAX_SPEED']
        self.integration_steps = params['NUMERICAL_INTEGRATION_STEPS']
        
        self.gn_tx_power_watts = dbm_to_watts(params['GN_TRANSMIT_POWER_DBM'])
        noise_spectral_density_watts = dbm_to_watts(params['NOISE_POWER_SPECTRAL_DENSITY_DBM'])
        self.noise_power_watts = noise_spectral_density_watts * params['BANDWIDTH']
        
        self.comm_radius_d = self._calculate_max_comm_radius_iterative()
        # print(f"Calculated communication radius D = {self.comm_radius_d:.2f} meters.")
        self.hover_datarate = self._calculate_hover_datarate()
        # print(f"Calculated hover data rate at GN: {self.hover_datarate / 1e6:.2f} Mbps.")

    def _calculate_max_comm_radius_iterative(self) -> float:
        """Calculates the max horizontal communication radius D where SNR equals the threshold."""
        snr_thresh_linear = db_to_linear(self.params['SNR_THRESHOLD_DB'])
        low_d, high_d = 0.0, self.params['AREA_WIDTH']
        for _ in range(50):
            mid_d = (low_d + high_d) / 2.0
            if mid_d < 1e-6:
                snr = np.inf 
            else:
                dist_3d = np.sqrt(mid_d**2 + self.uav_altitude**2)
                if dist_3d == 0: continue
                elevation = np.degrees(np.arcsin(self.uav_altitude / dist_3d))
                path_loss = models.calculate_path_loss(dist_3d, elevation, self.params)
                snr = models.calculate_snr(self.gn_tx_power_watts, self.noise_power_watts, path_loss)
            
            if snr >= snr_thresh_linear:
                low_d = mid_d
            else:
                high_d = mid_d
        return low_d
    
    def calculate_hover_rate_at_point(self, point_2d: np.ndarray, gn_coord: np.ndarray) -> float:
        """
        Calculates the specific data rate when hovering at a given 2D point.
        This is used for calculating fair hover times at path edges.
        """
        dist_2d = np.linalg.norm(point_2d - gn_coord)
        dist_3d = np.sqrt(dist_2d**2 + self.uav_altitude**2)

        rate = 0.0
        if dist_3d > 1e-6:
            elevation = np.degrees(np.arcsin(self.uav_altitude / dist_3d))
            path_loss = models.calculate_path_loss(dist_3d, elevation, self.params)
            snr = models.calculate_snr(self.gn_tx_power_watts, self.noise_power_watts, path_loss)
            rate = models.calculate_transmission_rate(snr, self.params['BANDWIDTH'])
            
        return rate
    
    def get_rate_at_comm_edge(self) -> float:
        """
        Calculates the data rate exactly at the communication radius boundary (D).
        By definition of D, this should be very close to the rate at the SNR threshold.
        """
        # A point on the edge has a horizontal distance of D
        point_on_edge_2d = np.array([self.comm_radius_d, 0])
        gn_coord_at_origin = np.array([0, 0])
        
        # We can reuse the general-purpose function we created
        rate = self.calculate_hover_rate_at_point(point_on_edge_2d, gn_coord_at_origin)
        return rate

    def _calculate_hover_datarate(self) -> float:
        """Calculates the data rate when hovering directly above a GN."""
        dist_3d = self.uav_altitude
        elevation_angle = 90.0
        path_loss = models.calculate_path_loss(dist_3d, elevation_angle, self.params)
        snr = models.calculate_snr(self.gn_tx_power_watts, self.noise_power_watts, path_loss)
        rate = models.calculate_transmission_rate(snr, self.params['BANDWIDTH'])
        return rate

    def _calculate_collected_data(self, start_point_2d: np.ndarray, end_point_2d: np.ndarray, gn_coord_2d: np.ndarray) -> float:
        """Numerically integrates the data rate over a straight flight path."""
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
            dist_3d = np.sqrt(dist_2d**2 + self.uav_altitude**2)
            if dist_3d < 1e-6: continue
            elevation_angle = np.degrees(np.arcsin(self.uav_altitude / dist_3d))
            
            path_loss = models.calculate_path_loss(dist_3d, elevation_angle, self.params)
            snr = models.calculate_snr(self.gn_tx_power_watts, self.noise_power_watts, path_loss)
            rate = models.calculate_transmission_rate(snr, self.params['BANDWIDTH'])
            
            total_data += rate * delta_t
            
        return total_data
    
    def _get_closest_point_on_ellipse(self, fip: np.ndarray, fop: np.ndarray, path_length: float, point: np.ndarray) -> np.ndarray:
        """Approximates the point on an ellipse closest to an external point 'point' (the GN)."""
        foci_dist = np.linalg.norm(fip - fop)
        if path_length < foci_dist: path_length = foci_dist

        ellipse_center = (fip + fop) / 2.0
        angle = np.arctan2(fop[1] - fip[1], fop[0] - fop[0])
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        a = path_length / 2.0
        b = np.sqrt(max(1e-9, a**2 - (foci_dist / 2.0)**2))
        
        t_vals = np.linspace(0, 2 * np.pi, 100)
        points_on_ellipse_local = np.vstack((a * np.cos(t_vals), b * np.sin(t_vals))).T
        
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        points_on_ellipse_global = (rot_matrix @ points_on_ellipse_local.T).T + ellipse_center
        
        distances_sq = np.sum((points_on_ellipse_global - point)**2, axis=1)
        best_idx = np.argmin(distances_sq)
        
        return points_on_ellipse_global[best_idx]

    def _find_oh_on_mid_perpendicular(self, fip: np.ndarray, fop: np.ndarray, gn_coord: np.ndarray, required_data: float) -> Tuple[np.ndarray, float]:
        """Implements Lemma 1: Finds optimal OH on the mid-perpendicular via binary search."""
        q_point = (fip + fop) / 2.0
        if np.linalg.norm(fip - fop) < 1e-6:
            return q_point, float('inf') if required_data > 1e-6 else 0.0
            
        v_perp = np.array([fop[1] - fip[1], fip[0] - fop[0]])
        v_perp_unit = v_perp / np.linalg.norm(v_perp)
        
        low_d = 0.0
        high_d = np.linalg.norm(gn_coord - q_point) + self.comm_radius_d
        best_d = high_d

        for _ in range(20):
            mid_d = (low_d + high_d) / 2.0
            oh_candidate1 = q_point + mid_d * v_perp_unit
            oh_candidate2 = q_point - mid_d * v_perp_unit
            oh_candidate = oh_candidate1 if np.linalg.norm(oh_candidate1 - gn_coord) < np.linalg.norm(oh_candidate2 - gn_coord) else oh_candidate2

            data = self._calculate_collected_data(fip, oh_candidate, gn_coord) + \
                   self._calculate_collected_data(oh_candidate, fop, gn_coord)
            
            if data >= required_data:
                best_d = mid_d
                high_d = mid_d
            else:
                low_d = mid_d
        
        oh_final1 = q_point + best_d * v_perp_unit
        oh_final2 = q_point - best_d * v_perp_unit
        optimal_oh = oh_final1 if np.linalg.norm(oh_final1 - gn_coord) < np.linalg.norm(oh_final2 - gn_coord) else oh_final2
        
        path_length = np.linalg.norm(fip - optimal_oh) + np.linalg.norm(optimal_oh - fop)
        collection_time = path_length / self.uav_max_speed
        return optimal_oh, collection_time

    # <<< FUNDAMENTALLY CORRECTED LOGIC >>>
    def find_optimal_fm_trajectory(self, fip: np.ndarray, fop: np.ndarray, gn_coord: np.ndarray, required_data: float, is_overlapping: bool) -> Tuple[np.ndarray, float]:
        """
        Finds the optimal V-shaped trajectory for FM mode.
        - If NOT overlapping (is_overlapping=False), FIP/FOP are on the same circle 
          -> use simplified mid-perpendicular search (Lemma 1).
        - If overlapping (is_overlapping=True), FIP/FOP are on different circles
          -> use general ellipse-based search (Algorithm 2).
        """
        if not is_overlapping:
            # Non-overlapping case: FIP/FOP on same circle -> OH is on mid-perpendicular.
            return self._find_oh_on_mid_perpendicular(fip, fop, gn_coord, required_data)
        else:
            # Overlapping/General case: FIP/FOP not on same circle -> OH is not on mid-perpendicular.
            # Must use the full Algorithm 2: bisection on path length K.
            k_min = np.linalg.norm(fip - fop)
            k_max = np.linalg.norm(fip - gn_coord) + np.linalg.norm(gn_coord - fop)
            min_path_length = k_max

            for _ in range(10):
                k_temp = (k_min + k_max) / 2.0
                if k_temp >= k_max - 1e-6: break

                oh_candidate = self._get_closest_point_on_ellipse(fip, fop, k_temp, gn_coord)
                max_data_at_k_temp = self._calculate_collected_data(fip, oh_candidate, gn_coord) + \
                                       self._calculate_collected_data(oh_candidate, fop, gn_coord)

                if max_data_at_k_temp >= required_data:
                    min_path_length = k_temp
                    k_max = k_temp
                else:
                    k_min = k_temp
            
            final_optimal_oh = self._get_closest_point_on_ellipse(fip, fop, min_path_length, gn_coord)
            collection_time = min_path_length / self.uav_max_speed
            return final_optimal_oh, collection_time
    
    def calculate_fm_max_capacity(self, fip: np.ndarray, fop: np.ndarray, gn_coord: np.ndarray) -> float:
        """
        Calculates C_max: the max data capacity by flying through the GN's center.
        """
        data1 = self._calculate_collected_data(fip, gn_coord, gn_coord)
        data2 = self._calculate_collected_data(gn_coord, fop, gn_coord)
        return data1 + data2
    