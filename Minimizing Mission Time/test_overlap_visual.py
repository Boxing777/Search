# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Dict, Tuple, List

# ==============================================================================
# [1] CORE PHYSICS & MODELS
# ==============================================================================
SPEED_OF_LIGHT = 299792458.0
PARAMS = {
    'UAV_ALTITUDE': 50.0, 
    'UAV_MAX_SPEED': 20.0,
    'GN_TRANSMIT_POWER_DBM': -40.0, 
    'NOISE_POWER_SPECTRAL_DENSITY_DBM': -174.0,
    'BANDWIDTH': 2e6, 
    'SNR_THRESHOLD_DB': 2.6, 
    'PATH_LOSS_EXPONENT': 1.3165,
    'LOS_ADDITIONAL_LOSS_DB': 3.0, 
    'NLOS_ADDITIONAL_LOSS_DB': 13.0,
    'LOS_PROBABILITY_PARAMS': {"a": 11.95, "beta": 0.14},
    'NUMERICAL_INTEGRATION_STEPS': 100,
    'CARRIER_FREQUENCY': 2e9
}

def db_to_linear(db: float): return 10**(db / 10.0)

class TrajectoryOptimizer:
    def __init__(self, params: Dict):
        self.params = params
        self.uav_alt = params['UAV_ALTITUDE']
        self.v_max = params['UAV_MAX_SPEED']
        self.steps = params['NUMERICAL_INTEGRATION_STEPS']
        self.tx_w = 10**((params['GN_TRANSMIT_POWER_DBM'] - 30) / 10.0)
        self.noise_w = 10**((params['NOISE_POWER_SPECTRAL_DENSITY_DBM'] - 30) / 10.0) * params['BANDWIDTH']
        self.comm_radius = self._calc_radius()

    def _calc_radius(self):
        thresh = db_to_linear(self.params['SNR_THRESHOLD_DB'])
        low, high = 0.0, 1000.0
        for _ in range(40):
            mid = (low + high) / 2
            d3d = np.sqrt(mid**2 + self.uav_alt**2)
            elev = np.degrees(np.arcsin(self.uav_alt / d3d))
            fc, eta = self.params['CARRIER_FREQUENCY'], self.params['PATH_LOSS_EXPONENT']
            fspl = ((4*np.pi*fc/SPEED_OF_LIGHT)**eta) * (d3d**eta)
            p_los = 1.0/(1.0+11.95*np.exp(-0.14*(elev-11.95)))
            xi_l, xi_n = db_to_linear(3.0), db_to_linear(13.0)
            pl = fspl * (p_los*(xi_l-xi_n)+xi_n)
            snr = self.tx_w / (self.noise_w * pl)
            if snr >= thresh: low = mid
            else: high = mid
        return low

    def calc_rate(self, p, gn):
        p_arr = np.array(p).flatten()
        gn_arr = np.array(gn).flatten()
        d2d = np.linalg.norm(p_arr - gn_arr)
        d3d = np.sqrt(d2d**2 + self.uav_alt**2)
        elev = np.degrees(np.arcsin(self.uav_alt / d3d))
        fc, eta = self.params['CARRIER_FREQUENCY'], self.params['PATH_LOSS_EXPONENT']
        fspl = ((4*np.pi*fc/SPEED_OF_LIGHT)**eta) * (d3d**eta)
        p_los = 1.0/(1.0+11.95*np.exp(-0.14*(elev-11.95)))
        xi_l, xi_n = db_to_linear(3.0), db_to_linear(13.0)
        pl = fspl * (p_los*(xi_l-xi_n)+xi_n)
        snr = self.tx_w / (self.noise_w * pl)
        return self.params['BANDWIDTH'] * np.log2(1 + snr)

    def _calc_data(self, start, end, gn):
        p1, p2 = np.array(start).flatten(), np.array(end).flatten()
        vec = p2 - p1
        dist = np.linalg.norm(vec)
        if dist < 1e-3: return 0
        dt = (dist/self.v_max)/self.steps
        total = 0
        for i in range(self.steps):
            total += self.calc_rate(p1 + ((i+0.5)/self.steps)*vec, gn) * dt
        return total

    def find_v_shape(self, fip, fop, gn, req):
        p1, p2 = np.array(fip).flatten(), np.array(fop).flatten()
        gn_pt = np.array(gn).flatten()
        q = (p1 + p2)/2.0
        if np.linalg.norm(p2 - p1) < 1e-6: return q
        v_perp = np.array([p2[1]-p1[1], p1[0]-p2[0]])
        v_perp = v_perp / np.linalg.norm(v_perp)
        low, high = 0.0, self.comm_radius
        best_d = high
        for _ in range(15):
            mid = (low + high)/2.0
            c1, c2 = q + mid*v_perp, q - mid*v_perp
            oh = c1 if np.linalg.norm(c1-gn_pt) < np.linalg.norm(c2-gn_pt) else c2
            if self._calc_data(p1, oh, gn_pt) + self._calc_data(oh, p2, gn_pt) >= req:
                best_d, high = mid, mid
            else: low = mid
        c_fin = q + best_d*v_perp
        c_fin2 = q - best_d*v_perp
        return c_fin if np.linalg.norm(c_fin-gn_pt) < np.linalg.norm(c_fin2-gn_pt) else c_fin2

# ==============================================================================
# [2] VISUAL SANDBOX CLASS
# ==============================================================================
class VisualSandbox:
    def __init__(self):
        self.opt = TrajectoryOptimizer(PARAMS)
        self.R = self.opt.comm_radius
        self.gn1 = np.array([1000.0, 1000.0])
        self.gn2 = np.array([1000.0 + 1.25 * self.R, 1000.0])
        self.sp = np.array([600.0, 700.0])
        self.anchor = np.array([1800.0, 700.0])
        self.req = 40 * 1e6

    def get_path_segments(self, p_flex):
        angles = np.linspace(0, 2*np.pi, 18, endpoint=False)
        fip_cands = [self.gn1 + self.R * np.array([np.cos(a), np.sin(a)]) for a in angles]
        fop_cands = [self.gn2 + self.R * np.array([np.cos(a), np.sin(a)]) for a in angles]
        best_t1, best_pts1 = float('inf'), None
        for fip in fip_cands:
            t_in = np.linalg.norm(fip - self.sp)/20.0
            oh = self.opt.find_v_shape(fip, p_flex, self.gn1, self.req)
            t_col = (np.linalg.norm(oh-fip) + np.linalg.norm(p_flex - oh))/20.0
            if t_in + t_col < best_t1:
                best_t1, best_pts1 = t_in + t_col, (fip, oh)
        best_t2, best_pts2 = float('inf'), None
        for fop in fop_cands:
            oh = self.opt.find_v_shape(p_flex, fop, self.gn2, self.req)
            t_col = (np.linalg.norm(oh-p_flex) + np.linalg.norm(fop - oh))/20.0
            t_out = np.linalg.norm(self.anchor - fop)/20.0
            if t_col + t_out < best_t2:
                best_t2, best_pts2 = t_col + t_out, (oh, fop)
        return best_t1 + best_t2, best_pts1, best_pts2

    def get_skeleton(self, n):
        c1, c2 = self.gn1, self.gn2
        d = np.linalg.norm(c2-c1)
        la = [c1 + t*(c2-c1) for t in np.linspace((d-self.R)/d, self.R/d, n//2)]
        mid = (c1+c2)/2.0
        v_perp = np.array([c2[1]-c1[1], c1[0]-c2[0]])
        v_perp = v_perp / np.linalg.norm(v_perp)
        h = np.sqrt(max(0, self.R**2 - (d/2.0)**2))
        tip = mid + h*v_perp
        if np.linalg.norm(self.sp - tip) > np.linalg.norm(self.sp - (mid - h*v_perp)): tip = mid - h*v_perp
        lb = [mid + t*(tip-mid) for t in np.linspace(0, 1, n - n//2 + 1)[1:]]
        return la, lb

    def animate_strategy(self, strategy_type, filename):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect('equal')
        ax.set_xlim(500, 1900); ax.set_ylim(600, 1400)
        ax.add_patch(plt.Circle(self.gn1, self.R, color='gray', fill=False, linestyle='--'))
        ax.add_patch(plt.Circle(self.gn2, self.R, color='gray', fill=False, linestyle='--'))
        ax.plot(self.sp[0], self.sp[1], 'ks', label='SP')
        ax.plot(self.anchor[0], self.anchor[1], 'k*', markersize=12, label='Anchor')

        line_path, = ax.plot([], [], 'r-', linewidth=1.5, alpha=0.6, label='Searching...')
        line_best, = ax.plot([], [], 'g-', linewidth=3.5, label='Final Best Route', zorder=10)
        scat_all = ax.scatter([], [], c='blue', s=15, alpha=0.3)
        scat_now = ax.scatter([], [], c='red', s=100, edgecolors='black', zorder=5)
        scat_best = ax.scatter([], [], c='yellow', marker='D', s=150, edgecolors='green', label='Best P_flex', zorder=11)
        bridge_line, = ax.plot([], [], 'cyan', linestyle=':', linewidth=2, label='Bridge')
        text_info = ax.text(0.02, 0.95, '', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

        frames_data = []
        global_best_p, global_min_t = None, float('inf')

        if strategy_type == 1:
            la, lb = self.get_skeleton(10)
            p_list = la + lb
            for p in p_list:
                t_val, _, _ = self.get_path_segments(p)
                if t_val < global_min_t: global_min_t, global_best_p = t_val, p
                frames_data.append({'p': p, 'msg': 'Skeleton Search', 'is_final': False})
        else:
            la, lb = self.get_skeleton(10)
            ba, bb, ma, mb = None, None, float('inf'), float('inf')
            for p in la:
                t_val, _, _ = self.get_path_segments(p)
                if t_val < global_min_t: global_min_t, global_best_p = t_val, p
                if t_val < ma: ma, ba = t_val, p
                frames_data.append({'p': p, 'msg': 'Axial Search (A)', 'is_final': False})
            for p in lb:
                t_val, _, _ = self.get_path_segments(p)
                if t_val < global_min_t: global_min_t, global_best_p = t_val, p
                if t_val < mb: mb, bb = t_val, p
                frames_data.append({'p': p, 'msg': 'Axial Search (B)', 'is_final': False})
            for t_val_bridge in np.linspace(0, 1, 7)[1:-1]:
                p = ba + t_val_bridge*(bb-ba)
                t_total, _, _ = self.get_path_segments(p)
                if t_total < global_min_t: global_min_t, global_best_p = t_total, p
                frames_data.append({'p': p, 'msg': 'Bridge Search', 'ba': ba, 'bb': bb, 'is_final': False})

        # Add 5 final frames to highlight the winner
        for _ in range(5):
            frames_data.append({'p': global_best_p, 'msg': 'FINAL SELECTION', 'is_final': True})

        def update(i):
            d = frames_data[i]
            p = d['p']
            total_t, pts1, pts2 = self.get_path_segments(p)
            xpath = [self.sp[0], pts1[0][0], pts1[1][0], p[0], pts2[0][0], pts2[1][0], self.anchor[0]]
            ypath = [self.sp[1], pts1[0][1], pts1[1][1], p[1], pts2[0][1], pts2[1][1], self.anchor[1]]

            if d['is_final']:
                line_best.set_data(xpath, ypath)
                scat_best.set_offsets([p])
                line_path.set_alpha(0) # Hide search line
                scat_now.set_alpha(0)
            else:
                line_path.set_data(xpath, ypath)
                scat_now.set_offsets([p])
                scat_all.set_offsets([fd['p'] for fd in frames_data[:i+1] if not fd['is_final']])

            if 'ba' in d: bridge_line.set_data([d['ba'][0], d['bb'][0]], [d['ba'][1], d['bb'][1]])
            else: bridge_line.set_data([], [])

            text_info.set_text(f"Strategy {strategy_type}\n{d['msg']}\nTime: {total_t:.2f}s")
            return line_path, line_best, scat_all, scat_now, scat_best, bridge_line, text_info

        ani = FuncAnimation(fig, update, frames=len(frames_data), blit=False)
        ax.legend(loc='lower right', fontsize='small')
        print(f"Generating: {filename}...")
        ani.save(filename, writer=PillowWriter(fps=2))
        plt.close(fig)

if __name__ == "__main__":
    sandbox = VisualSandbox()
    sandbox.animate_strategy(1, "strategy_1_t_shape.gif")
    sandbox.animate_strategy(2, "strategy_2_triangular.gif")
    print("\nSUCCESS: Highlighted GIFs created.")