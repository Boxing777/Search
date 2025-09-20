# ==============================================================================
#                      Simulation Parameters Configuration
#
# File Objective:
# This file, parameters.py, serves as the central configuration hub for our
# simulation. It defines all the static constants and parameters required to
# model the UAV-enabled wireless sensor network environment, the UAV's physical
# characteristics, the wireless communication channel, and the control settings
# for the optimization algorithms. By centralizing these values, we can easily
# modify and experiment with different simulation scenarios without changing the
# core logic of the program.
# ==============================================================================

# Import necessary libraries for type hinting and potential numerical constants
import numpy as np
from typing import Tuple, Dict

# ==============================================================================
# Section 1: Simulation Scenario Parameters
# ------------------------------------------------------------------------------
# This section defines the overall physical environment and the scale of the problem.
# ==============================================================================

# The width of the square simulation area. The simulation space is a 2D plane
# from (0, 0) to (AREA_WIDTH, AREA_HEIGHT).
AREA_WIDTH = 3000.0  # meters

# The height of the square simulation area. Typically same as AREA_WIDTH.
AREA_HEIGHT = 3000.0 # meters




# The total number of Ground Nodes (GNs) that need their data collected (N).
NUM_GNS =6 # integer

# The total number of Unmanned Aerial Vehicles (UAVs) for the mission (M).
NUM_UAVS = 1 # integer

# The 2D coordinates of the data center. UAVs start and end here.
DATA_CENTER_POS: Tuple[float, float] = (1500.0, 1500.0) # meters

# ==============================================================================
# Section 2: UAV Physical Parameters
# ------------------------------------------------------------------------------
# This section defines the physical attributes and constraints of the UAVs.
# ==============================================================================

# The fixed operational altitude of all UAVs (H).
UAV_ALTITUDE = 50.0 # meters

# The maximum cruising speed of the UAVs (v_max). Assumed travel speed between zones.
UAV_MAX_SPEED = 25.0 # meters/second

# The total energy budget available for each UAV for the entire mission (Eth).
UAV_ENERGY_BUDGET = 5e5 # Joules

# A collection of parameters for the UAV's propulsion power model, based on [27].
# These are used in a function `calculate_flight_power(velocity)`.
UAV_FLIGHT_POWER_PARAMS: Dict[str, float] = {
    "P0": 300.0,    # Blade profile power (Watts)
    "P1": 400.0,    # Induced power (Watts)
    "U_tip": 180.0, # Tip speed of the rotor blade (m/s)
    "v0": 5.0,      # Mean rotor induced velocity in hover (m/s)
    "d0": 0.5,      # Fuselage drag ratio (dimensionless)
    "s0": 0.05,     # Rotor solidity (dimensionless)
    "rho": 1.225,   # Air density (kg/m^3)
    "A": 0.8,       # Rotor disc area (m^2)
}

# The constant circuit power consumed by the UAV's communication hardware (Pc).
UAV_CIRCUIT_POWER = 1.0 # Watts

# ==============================================================================
# Section 3: Wireless Communication Parameters
# ------------------------------------------------------------------------------
# Defines the parameters for the air-to-ground wireless channel model,
# based on Table I of the paper.
# ==============================================================================

# The spectral density of the additive white Gaussian noise (AWGN).
NOISE_POWER_SPECTRAL_DENSITY_DBM = -174.0 # dBm/Hz

# The fixed transmission power of each Ground Node (Pt).
GN_TRANSMIT_POWER_DBM = -40.0 # dBm

# The carrier frequency of the wireless communication link (fc).
CARRIER_FREQUENCY = 2e9 # Hertz (2 GHz)

# The communication channel bandwidth (B).
BANDWIDTH = 2e6 # Hertz (2 MHz)

# The minimum required Signal-to-Noise Ratio (SNR) for successful reception.
SNR_THRESHOLD_DB = 2.6 # dB (gamma_thresh)

# The path loss exponent (eta). Note: The paper's Table I lists eta=1, which
# is followed here, but free-space path loss typically uses an exponent of 2.
PATH_LOSS_EXPONENT = 1.0 # dimensionless

# The average additional path loss for Line-of-Sight (LoS) links (xi_LoS).
LOS_ADDITIONAL_LOSS_DB = 3.0 # dB

# The average additional path loss for Non-Line-of-Sight (NLoS) links (xi_NLoS).
NLOS_ADDITIONAL_LOSS_DB = 13.0 # dB

# Parameters 'a' and 'beta' for the S-curve LoS probability model (equation 4).
LOS_PROBABILITY_PARAMS: Dict[str, float] = {
    "a": 11.95,     # (alpha in the paper)
    "beta": 0.14
}

# ==============================================================================
# Section 4: Algorithm Control Parameters
# ------------------------------------------------------------------------------
# Defines parameters that control the behavior of the optimization algorithms.
# ==============================================================================

# --- For Genetic Algorithm (Algorithm 1) ---
GA_POPULATION_SIZE = 80 # Number of individuals in each generation
GA_NUM_ITERATIONS = 100 # Total number of generations (ite)
GA_MUTATION_RATE = 0.02 # Probability of a gene undergoing mutation (0.0 to 1.0)
GA_CROSSOVER_RATE = 0.85 # Probability of crossover between parents (0.0 to 1.0)

# --- For Mission Allocation Model (Equation 9) ---
# Weighting factor for the flight mission component in the allocation cost function.
SCALING_FACTOR_A = 0.5 # (a)
# Weighting factor for the data collection mission component.
SCALING_FACTOR_B = 0.5 # (b)

# --- For Trajectory Optimizer (Algorithms 2 & 3) ---
# Step size 'k' for discretizing the search space for FIP and FOP in JOFC.
JOFC_GRID_STEP_SIZE = 30.0 # meters

# The number of discrete steps 'K' to approximate the integral for calculating
# total collected data (Appendix C, equation 19).
NUMERICAL_INTEGRATION_STEPS = 200 # integer