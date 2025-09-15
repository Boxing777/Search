# channel_model.py
import numpy as np
from config import *

def calculate_plos(d_horizontal, h):
    """Calculates the Line-of-Sight (LoS) probability. Implements Equation (7)."""
    elevation_angle_rad = np.arctan(h / d_horizontal)
    elevation_angle_deg = np.degrees(elevation_angle_rad)
    return 1 / (1 + A_PARAM * np.exp(-B_PARAM * (elevation_angle_deg - A_PARAM)))

def calculate_path_loss(d_horizontal, h):
    """Calculates the average path loss. Implements Equations (5) and (6)."""
    distance_3d = np.sqrt(d_horizontal**2 + h**2)
    
    # Free Space Path Loss (FSPL) in dB
    fspl_db = 20 * np.log10(distance_3d) + 20 * np.log10(CARRIER_FREQ_FC) + 20 * np.log10(4 * np.pi / LIGHT_SPEED_C)
    
    path_loss_los_db = fspl_db + ETA_LOS
    path_loss_nlos_db = fspl_db + ETA_NLOS
    
    # Convert from dB to linear scale
    path_loss_los = 10**(path_loss_los_db / 10)
    path_loss_nlos = 10**(path_loss_nlos_db / 10)
    
    p_los = calculate_plos(d_horizontal, h)
    
    # Average path loss
    avg_path_loss = p_los * path_loss_los + (1 - p_los) * path_loss_nlos
    return avg_path_loss

def calculate_snr(d_horizontal, h):
    """Calculates the received SNR at the UAV."""
    path_loss = calculate_path_loss(d_horizontal, h)
    received_power = GU_TRANSMIT_POWER_P / path_loss
    snr = received_power / NOISE_POWER
    return snr

def get_transmission_radius(h):
    """
    Finds the horizontal distance D_H where the SNR equals the required threshold.
    This function is crucial for altitude optimization.
    """
    from scipy.optimize import fsolve

    # We need to solve: calculate_snr(D_H, h) = SNR_THRESHOLD * SNR_GAP
    # This is equivalent to finding the root of: f(D_H) = calculate_snr(D_H, h) - required_snr = 0
    required_snr = SNR_THRESHOLD * SNR_GAP
    
    def equation_to_solve(d_h):
        if d_h <= 0: return -np.inf # Avoid non-physical values
        return calculate_snr(d_h, h) - required_snr

    # Use an initial guess. A good guess might be h itself.
    initial_guess = h
    try:
        # fsolve finds the root of the equation
        d_h_solution = fsolve(equation_to_solve, initial_guess)[0]
        return d_h_solution if d_h_solution > 0 else 0
    except:
        return 0