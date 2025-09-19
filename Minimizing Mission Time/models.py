# ==============================================================================
#                      Scientific Models Library
#
# File Objective:
# This file is the core scientific library for the simulation. It implements
# all the mathematical and physical models described in the research paper.
# It contains a collection of pure, stateless, and deterministic functions
# that form the building blocks for the more complex optimization algorithms.
# ==============================================================================

# Import necessary libraries
import numpy as np
from typing import Dict

# Define physical constants
SPEED_OF_LIGHT = 299792458.0  # meters per second

# ==============================================================================
# Section 1: Wireless Channel and Communication Models
# ------------------------------------------------------------------------------
# Implements functions related to the air-to-ground wireless communication link.
# ==============================================================================

def calculate_los_probability(elevation_angle_degrees: float, a: float, b: float) -> float:
    """
    Computes the probability of a Line-of-Sight (LoS) link using Equation (4).

    Formula: P_LoS = 1 / (1 + a * exp(-b * (elevation_angle_degrees - a)))

    Args:
        elevation_angle_degrees (float): The elevation angle theta in degrees.
        a (float): The environment-dependent parameter 'a' from the LoS model.
        b (float): The environment-dependent parameter 'beta' from the LoS model.

    Returns:
        float: The probability of a LoS connection (between 0 and 1).
    """
    exponent = -b * (elevation_angle_degrees - a)
    p_los = 1.0 / (1.0 + a * np.exp(exponent))
    return p_los

def calculate_path_loss(distance_3d: float, elevation_angle_degrees: float, params: Dict) -> float:
    """
    Calculates the total path loss in decibels (dB) using Equation (3).

    This is calculated as the sum of Free Space Path Loss (FSPL) and an
    average additional loss based on LoS/NLoS conditions.

    Args:
        distance_3d (float): The 3D Euclidean distance (l_ij) in meters.
        elevation_angle_degrees (float): The elevation angle in degrees (for P_LoS).
        params (Dict): A dictionary containing communication parameters like
                       CARRIER_FREQUENCY, LOS_PROBABILITY_PARAMS, etc.

    Returns:
        float: The total path loss in decibels (dB).
    """
    fc = params['CARRIER_FREQUENCY']
    fspl_db = 20 * np.log10(distance_3d) + 20 * np.log10(fc) + 20 * np.log10((4 * np.pi) / SPEED_OF_LIGHT)

    los_params = params['LOS_PROBABILITY_PARAMS']
    p_los = calculate_los_probability(elevation_angle_degrees, los_params['a'], los_params['beta'])

    eta_los = params['LOS_ADDITIONAL_LOSS_DB']
    eta_nlos = params['NLOS_ADDITIONAL_LOSS_DB']
    additional_loss_db = p_los * eta_los + (1 - p_los) * eta_nlos

    total_path_loss_db = fspl_db + additional_loss_db

    return total_path_loss_db

def calculate_snr(gn_transmit_power_watts: float, total_noise_power_watts: float, path_loss_db: float) -> float:
    """
    Calculates the Signal-to-Noise Ratio (SNR) in a linear scale (not dB).

    Implements Equation (2): SNR = P_t / (sigma^2 * L_path_linear).

    Args:
        gn_transmit_power_watts (float): The GN's transmission power (Pt) in Watts.
        total_noise_power_watts (float): Total noise power (sigma^2) in Watts.
        path_loss_db (float): The path loss value in dB from calculate_path_loss.

    Returns:
        float: The SNR in a linear scale.
    """
    linear_path_loss = 10**(path_loss_db / 10.0)
    snr = gn_transmit_power_watts / (total_noise_power_watts * linear_path_loss)
    return snr

def calculate_transmission_rate(snr_linear: float, bandwidth_hz: float) -> float:
    """
    Calculates the instantaneous data transmission rate using the Shannon-Hartley theorem.

    Implements Equation (1): R = B * log2(1 + SNR).

    Args:
        snr_linear (float): The Signal-to-Noise Ratio in a linear scale.
        bandwidth_hz (float): The channel bandwidth (B) in Hertz.

    Returns:
        float: The achievable data rate in bits per second (bps).
    """
    if snr_linear < 0:
        return 0.0
    rate = bandwidth_hz * np.log2(1 + snr_linear)
    return rate

# ==============================================================================
# Section 2: UAV Physical Models
# ------------------------------------------------------------------------------
# Implements functions related to the UAV's physical state.
# ==============================================================================

def calculate_flight_power(velocity_ms: float, params: Dict) -> float:
    """
    Calculates UAV propulsion power based on flight speed, using model from [27].

    Formula: P(v) = P0*(1+3v^2/U_tip^2) + P1*(sqrt(1+v^4/(4v0^4)) - v^2/(2v0^2))^(1/2) + 0.5*d0*rho*s0*A*v^3

    Args:
        velocity_ms (float): The instantaneous speed (v) of the UAV in m/s.
        params (Dict): Dictionary containing the UAV's physical power model
                       parameters from UAV_FLIGHT_POWER_PARAMS.

    Returns:
        float: The total propulsion power P_f(v) in Watts.
    """
    p0 = params['P0']
    p1 = params['P1']
    u_tip = params['U_tip']
    v0 = params['v0']
    d0 = params['d0']
    rho = params['rho']
    s0 = params['s0']
    a_rotor = params['A']
    v = velocity_ms

    part1 = p0 * (1 + (3 * v**2) / u_tip**2)
    inner_sqrt_val = 1 + (v**4) / (4 * v0**4)
    inner_term = np.sqrt(inner_sqrt_val) - (v**2) / (2 * v0**2)
    part2 = p1 * np.sqrt(max(0, inner_term))
    part3 = 0.5 * d0 * rho * s0 * a_rotor * v**3

    total_power = part1 + part2 + part3
    return total_power

# ==============================================================================
# Section 3: Mission Cost and Utility Models
# ------------------------------------------------------------------------------
# Implements functions used by optimization algorithms to evaluate solutions.
# ==============================================================================

def calculate_initial_mission_cost(gn_coord_prev: np.ndarray, gn_coord_curr: np.ndarray,
                                     transmission_radius_d: float, a: float, b: float) -> float:
    """
    Calculates the initial weighted mission cost for a UAV path segment (Equation 9).

    This simplified metric is used for the initial mission allocation by the GA.

    Args:
        gn_coord_prev (np.ndarray): The 2D coordinates of the previous GN (or data center).
        gn_coord_curr (np.ndarray): The 2D coordinates of the current GN.
        transmission_radius_d (float): The max communication radius (D) in meters.
        a (float): The scaling factor for the flight mission (SCALING_FACTOR_A).
        b (float): The scaling factor for the collection mission (SCALING_FACTOR_B).

    Returns:
        float: The dimensionless mission cost (??_ij) for this path segment.
    """
    distance = np.linalg.norm(gn_coord_curr - gn_coord_prev)

    if distance > 2 * transmission_radius_d:
        cost = a * (distance - 2 * transmission_radius_d) + 2 * b * transmission_radius_d
    else:
        cost = b * distance

    return cost