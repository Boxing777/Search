import numpy as np
import time
from typing import Dict, Tuple, List

# ==============================================================================
# [1] INCLUDED MODULES: parameters.py & models.py (Simplified)
# ==============================================================================
SPEED_OF_LIGHT = 299792458.0

PARAMS = {
    'AREA_WIDTH': 2000.0, 'AREA_HEIGHT': 2000.0,
    'UAV_ALTITUDE': 50.0, 'UAV_MAX_SPEED': 20.0,
    'NOISE_POWER_SPECTRAL_DENSITY_DBM': -174.0,
    'GN_TRANSMIT_POWER_DBM': -40.0,
    'CARRIER_FREQUENCY': 2e9, 'BANDWIDTH': 2e6,
    'SNR_THRESHOLD_DB': 2.6, 'PATH_LOSS_EXPONENT': 1.3165,
    'LOS_ADDITIONAL_LOSS_DB': 3.0, 'NLOS_ADDITIONAL_LOSS_DB': 13.0,
    'LOS_PROBABILITY_PARAMS': {"a": 11.95, "beta": 0.14},
    'NUMERICAL_INTEGRATION_STEPS': 200
}

def dbm_to_watts(dbm: float) -> float: return 10**((dbm - 30) / 10.0)
def db_to_linear(db: float) -> float: return 10**(db / 10.0)

def calculate_los_probability(elevation_angle_degrees: float, a: float, b: float) -> float:
    exponent = -b * (elevation_angle_degrees - a)
    return 1.0 / (1.0 + a * np.exp(exponent))

def calculate_path_loss(distance_3d: float, elevation_angle_degrees: float, params: Dict) -> float:
    fc, eta = params['CARRIER_FREQUENCY'], params['PATH_LOSS_EXPONENT']
    fspl_linear = ((4 * np.pi * fc / SPEED_OF_LIGHT) ** eta) * (distance_3d ** eta)
    los_params = params['LOS_PROBABILITY_PARAMS']
    p_los = calculate_los_probability(elevation_angle_degrees, los_params['a'], los_params['beta'])
    xi_los_linear = 10**(params['LOS_ADDITIONAL_LOSS_DB'] / 10.0)
    xi_nlos_linear = 10**(params['NLOS_ADDITIONAL_LOSS_DB'] / 10.0)
    additional_loss_linear = p_los * (xi_los_linear - xi_nlos_linear) + xi_nlos_linear
    return fspl_linear * additional_loss_linear

def calculate_snr(gn_tx_power_w: float, noise_power_w: float, path_loss_lin: float) -> float:
    if path_loss_lin <= 0: return 0.0
    return gn_tx_power_w / (noise_power_w * path_loss_lin)

def calculate_transmission_rate(snr_linear: float, bandwidth_hz: float) -> float:
    if snr_linear < 0: return 0.0
    return bandwidth_hz * np.log2(1 + snr_linear)

# ==============================================================================
# [2] INCLUDED MODULE: trajectory_optimizer.py
# ==============================================================================
class TrajectoryOptimizer:
    def __init__(self, params: Dict):
        self.params = params
        self.uav_altitude = params['UAV_ALTITUDE']
        self.uav_max_speed = params['UAV_MAX_SPEED']
        self.integration_steps = params['NUMERICAL_INTEGRATION_STEPS']
        self.gn_tx_power_watts = dbm_to_watts(params['GN_TRANSMIT_POWER_DBM'])
        self.noise_power_watts = dbm_to_watts(params['NOISE_POWER_SPECTRAL_DENSITY_DBM']) * params['BANDWIDTH']
        self.comm_radius_d = self._calculate_max_comm_radius_iterative()
        self.hover_datarate = self._calculate_hover_datarate()

    def _calculate_max_comm_radius_iterative(self) -> float:
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
                path_loss = calculate_path_loss(dist_3d, elevation, self.params)
                snr = calculate_snr(self.gn_tx_power_watts, self.noise_power_watts, path_loss)
            if snr >= snr_thresh_linear: low_d = mid_d
            else: high_d = mid_d
        return low_d
    
    def calculate_hover_rate_at_point(self, point_2d: np.ndarray, gn_coord: np.ndarray) -> float:
        dist_2d = np.linalg.norm(point_2d - gn_coord)
        dist_3d = np.sqrt(dist_2d**2 + self.uav_altitude**2)
        rate = 0.0
        if dist_3d > 1e-6:
            elevation = np.degrees(np.arcsin(self.uav_altitude / dist_3d))
            path_loss = calculate_path_loss(dist_3d, elevation, self.params)
            snr = calculate_snr(self.gn_tx_power_watts, self.noise_power_watts, path_loss)
            rate = calculate_transmission_rate(snr, self.params['BANDWIDTH'])
        return rate

    def _calculate_hover_datarate(self) -> float:
        return self.calculate_hover_rate_at_point(np.array([0,0]), np.array([0,0]))

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
            dist_3d = np.sqrt(dist_2d**2 + self.uav_altitude**2)
            if dist_3d < 1e-6: continue
            elevation_angle = np.degrees(np.arcsin(self.uav_altitude / dist_3d))
            path_loss = calculate_path_loss(dist_3d, elevation_angle, self.params)
            snr = calculate_snr(self.gn_tx_power_watts, self.noise_power_watts, path_loss)
            rate = calculate_transmission_rate(snr, self.params['BANDWIDTH'])
            total_data += rate * delta_t
        return total_data
    
    def _get_closest_point_on_ellipse(self, fip: np.ndarray, fop: np.ndarray, path_length: float, point: np.ndarray) -> np.ndarray:
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
        q_point = (fip + fop) / 2.0
        if np.linalg.norm(fip - fop) < 1e-6: return q_point, float('inf') if required_data > 1e-6 else 0.0
        v_perp = np.array([fop[1] - fip[1], fip[0] - fop[0]])
        v_perp_unit = v_perp / np.linalg.norm(v_perp)
        low_d, high_d = 0.0, np.linalg.norm(gn_coord - q_point) + self.comm_radius_d
        best_d = high_d
        for _ in range(20):
            mid_d = (low_d + high_d) / 2.0
            oh_candidate1 = q_point + mid_d * v_perp_unit
            oh_candidate2 = q_point - mid_d * v_perp_unit
            oh_candidate = oh_candidate1 if np.linalg.norm(oh_candidate1 - gn_coord) < np.linalg.norm(oh_candidate2 - gn_coord) else oh_candidate2
            data = self._calculate_collected_data(fip, oh_candidate, gn_coord) + self._calculate_collected_data(oh_candidate, fop, gn_coord)
            if data >= required_data:
                best_d, high_d = mid_d, mid_d
            else: low_d = mid_d
        oh_final1 = q_point + best_d * v_perp_unit
        oh_final2 = q_point - best_d * v_perp_unit
        optimal_oh = oh_final1 if np.linalg.norm(oh_final1 - gn_coord) < np.linalg.norm(oh_final2 - gn_coord) else oh_final2
        collection_time = (np.linalg.norm(fip - optimal_oh) + np.linalg.norm(optimal_oh - fop)) / self.uav_max_speed
        return optimal_oh, collection_time

    def find_optimal_fm_trajectory(self, fip: np.ndarray, fop: np.ndarray, gn_coord: np.ndarray, required_data: float, is_overlapping: bool) -> Tuple[np.ndarray, float]:
        if not is_overlapping:
            return self._find_oh_on_mid_perpendicular(fip, fop, gn_coord, required_data)
        else:
            k_min = np.linalg.norm(fip - fop)
            k_max = np.linalg.norm(fip - gn_coord) + np.linalg.norm(gn_coord - fop)
            min_path_length = k_max
            for _ in range(10):
                k_temp = (k_min + k_max) / 2.0
                if k_temp >= k_max - 1e-6: break
                oh_candidate = self._get_closest_point_on_ellipse(fip, fop, k_temp, gn_coord)
                max_data_at_k_temp = self._calculate_collected_data(fip, oh_candidate, gn_coord) + self._calculate_collected_data(oh_candidate, fop, gn_coord)
                if max_data_at_k_temp >= required_data:
                    min_path_length, k_max = k_temp, k_temp
                else: k_min = k_temp
            final_optimal_oh = self._get_closest_point_on_ellipse(fip, fop, min_path_length, gn_coord)
            collection_time = min_path_length / self.uav_max_speed
            return final_optimal_oh, collection_time
    
    def calculate_fm_max_capacity(self, fip: np.ndarray, fop: np.ndarray, gn_coord: np.ndarray) -> float:
        return self._calculate_collected_data(fip, gn_coord, gn_coord) + self._calculate_collected_data(gn_coord, fop, gn_coord)

# ==============================================================================
# [3] SANDBOX ENVIRONMENT & TESTING LOGIC
# ==============================================================================

class SandboxTest:
    def __init__(self):
        print("Initializing Sandbox Environment...")
        self.traj_optimizer = TrajectoryOptimizer(PARAMS)
        self.R = self.traj_optimizer.comm_radius_d
        print(f"  - Comm Radius (D): {self.R:.2f} m")
        self.v_max = PARAMS['UAV_MAX_SPEED']

        # Setup Scenario
        self.gn1_coord = np.array([1000.0, 1000.0])
        # Place GN2 so they overlap (distance < 2R). Let's say dist = 1.2 * R
        self.gn2_coord = np.array([1000.0 + 1.2 * self.R, 1000.0])
        self.sp = np.array([500.0, 500.0]) # UAV comes from bottom-left
        self.anchor = np.array([1800.0, 500.0]) # UAV heads to top-right
        self.req_data = 40 * 1e6 # 40 Mbits

        print(f"  - GN1 at {self.gn1_coord}, GN2 at {self.gn2_coord}")
        print(f"  - Distance between GNs: {np.linalg.norm(self.gn1_coord - self.gn2_coord):.2f} m")

        # Generate FIP/FOP Candidates (Boundary points)
        angles = np.linspace(0, 2 * np.pi, 18, endpoint=False) 
        self.fip_cands = [self.gn1_coord + self.R * np.array([np.cos(a), np.sin(a)]) for a in angles]
        self.fop_cands = [self.gn2_coord + self.R * np.array([np.cos(a), np.sin(a)]) for a in angles]

    # --- P_flex Generator Helpers ---
    def _get_t_shape_candidates(self, num_samples: int) -> List[np.ndarray]:
        c1, c2 = self.gn1_coord, self.gn2_coord
        d = np.linalg.norm(c2 - c1)
        
        num_a = max(2, num_samples // 2)
        num_b = max(1, num_samples - num_a)
        cands = []

        # Line A (Center-line)
        start_dist = max(0.0, d - self.R)
        end_dist = min(d, self.R)
        for t in np.linspace(start_dist / d, end_dist / d, num_a):
            cands.append(c1 + t * (c2 - c1))

        # Line B (Bisector)
        a = d / 2
        h = np.sqrt(max(0, self.R**2 - a**2))
        mid = c1 + a * (c2 - c1) / d
        dx, dy = (c2[0] - c1[0]) / d, (c2[1] - c1[1]) / d
        tip1 = np.array([mid[0] - h * dy, mid[1] + h * dx])
        tip2 = np.array([mid[0] + h * dy, mid[1] - h * dx])
        target_tip = tip1 if np.linalg.norm(self.sp - tip1) < np.linalg.norm(self.sp - tip2) else tip2
        
        for t in np.linspace(0, 1, num_b + 1)[1:]:
            cands.append(mid + t * (target_tip - mid))
            
        return cands

    def _get_line_candidates(self, p_start: np.ndarray, p_end: np.ndarray, num_samples: int) -> List[np.ndarray]:
        cands = []
        vec = p_end - p_start
        for t in np.linspace(0, 1, num_samples):
            cands.append(p_start + t * vec)
        return cands

    # --- Core Evaluation Function ---
    def evaluate_p_flex(self, p_flex: np.ndarray) -> Tuple[float, Dict]:
        best_cost = float('inf')
        
        best_leg1 = float('inf')
        for fip in self.fip_cands:
            t_in = np.linalg.norm(fip - self.sp) / self.v_max
            c_max = self.traj_optimizer.calculate_fm_max_capacity(fip, p_flex, self.gn1_coord)
            if self.req_data <= c_max:
                _, t_col_theo = self.traj_optimizer.find_optimal_fm_trajectory(fip, p_flex, self.gn1_coord, self.req_data, True)
            else:
                t_flight = (np.linalg.norm(fip - self.gn1_coord) + np.linalg.norm(p_flex - self.gn1_coord)) / self.v_max
                t_col_theo = t_flight + (self.req_data - c_max) / self.traj_optimizer.hover_datarate
            
            t_col = max(t_col_theo, np.linalg.norm(fip - p_flex) / self.v_max)
            if t_in + t_col < best_leg1: best_leg1 = t_in + t_col

        best_leg2 = float('inf')
        for fop in self.fop_cands:
            c_max = self.traj_optimizer.calculate_fm_max_capacity(p_flex, fop, self.gn2_coord)
            if self.req_data <= c_max:
                _, t_col_theo = self.traj_optimizer.find_optimal_fm_trajectory(p_flex, fop, self.gn2_coord, self.req_data, True)
            else:
                t_flight = (np.linalg.norm(p_flex - self.gn2_coord) + np.linalg.norm(fop - self.gn2_coord)) / self.v_max
                t_col_theo = t_flight + (self.req_data - c_max) / self.traj_optimizer.hover_datarate
            
            t_col = max(t_col_theo, np.linalg.norm(p_flex - fop) / self.v_max)
            t_out = np.linalg.norm(self.anchor - fop) / self.v_max
            if t_col + t_out < best_leg2: best_leg2 = t_col + t_out
            
        return best_leg1 + best_leg2

    # --- Test Strategies ---
    def run_strategy_1_t_shape(self):
        # [MODIFIED] Set to 10 points (5 center + 5 bisect)
        print("\n--- Running Strategy 1: T-Shape (10 points total) ---")
        start_t = time.time()
        cands = self._get_t_shape_candidates(10) # <<< CHANGED from 20 to 10
        
        best_cost, best_p = float('inf'), None
        for p in cands:
            cost = self.evaluate_p_flex(p)
            if cost < best_cost: best_cost, best_p = cost, p
            
        exec_t = time.time() - start_t
        print(f"  Result: Cost = {best_cost:.4f} s | P_flex = ({best_p[0]:.1f}, {best_p[1]:.1f}) | Exec Time = {exec_t:.2f} s")
        return best_cost, best_p

    def run_strategy_2_triangular(self):
        # [MODIFIED] Set to 15 points total (5+5 axial + 5 bridge)
        print("\n--- Running Strategy 2: Triangular Progressive (5+5 -> 5 points total) ---")
        start_t = time.time()
        
        # Step A: Axial Search (5 on Center-line, 5 on Bisector)
        cands_axial = self._get_t_shape_candidates(10) 
        cands_A = cands_axial[:5] 
        cands_B = cands_axial[5:] 
        
        best_cost_A, best_p_A = float('inf'), None
        for p in cands_A:
            cost = self.evaluate_p_flex(p)
            if cost < best_cost_A: best_cost_A, best_p_A = cost, p
            
        best_cost_B, best_p_B = float('inf'), None
        for p in cands_B:
            cost = self.evaluate_p_flex(p)
            if cost < best_cost_B: best_cost_B, best_p_B = cost, p

        # Step B: Bridge Search
        dist_AB = np.linalg.norm(best_p_A - best_p_B)
        if dist_AB < 1.0:
            print("  [Bridge Skipped]: P_best_A and P_best_B are essentially the same point.")
            best_cost, best_p = best_cost_A, best_p_A
        else:
            print(f"  [Bridge Search]: Searching between ({best_p_A[0]:.0f},{best_p_A[1]:.0f}) and ({best_p_B[0]:.0f},{best_p_B[1]:.0f})")
            # [MODIFIED] Bridge search uses 5 points instead of 10
            cands_C = self._get_line_candidates(best_p_A, best_p_B, 5) # <<< CHANGED from 10 to 5
            best_cost_C, best_p_C = float('inf'), None
            for p in cands_C:
                cost = self.evaluate_p_flex(p)
                if cost < best_cost_C: best_cost_C, best_p_C = cost, p
            
            # Final Selection among A, B, C
            min_cost = min(best_cost_A, best_cost_B, best_cost_C)
            if min_cost == best_cost_C: best_cost, best_p = best_cost_C, best_p_C
            elif min_cost == best_cost_A: best_cost, best_p = best_cost_A, best_p_A
            else: best_cost, best_p = best_cost_B, best_p_B

        exec_t = time.time() - start_t
        print(f"  Result: Cost = {best_cost:.4f} s | P_flex = ({best_p[0]:.1f}, {best_p[1]:.1f}) | Exec Time = {exec_t:.2f} s")
        return best_cost, best_p

if __name__ == "__main__":
    sandbox = SandboxTest()
    cost1, p1 = sandbox.run_strategy_1_t_shape()
    cost2, p2 = sandbox.run_strategy_2_triangular()
    
    print("\n================ SUMMARY ================")
    print(f"Strategy 1 (T-Shape 10pts) Cost: {cost1:.4f} s")
    print(f"Strategy 2 (Triangular 15pts) Cost: {cost2:.4f} s")
    print(f"Time Saved by Strategy 2: {cost1 - cost2:.4f} s")
    if cost2 < cost1:
        print("CONCLUSION: Strategy 2 (Triangular) successfully found a better point!")
    else:
        print("CONCLUSION: Strategy 1 was sufficient for this geometry.")
    print("=========================================")