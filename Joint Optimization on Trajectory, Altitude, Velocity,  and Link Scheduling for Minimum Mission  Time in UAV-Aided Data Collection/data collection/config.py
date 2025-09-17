# config.py
import numpy as np

# --- Simulation Scenario Parameters ---
AREA_WIDTH = 3000  # meters
AREA_HEIGHT = 3000 # meters
START_POS = np.array([5.0, 5.0])
END_POS = np.array([3000.0, 15.0])

# --- Channel Model Parameters (Urban Scenario from paper) ---
# For P_LoS (Equation 7)
A_PARAM = 9.6117
B_PARAM = 0.1581

# For Path Loss (Equation 6)
ETA_LOS = 1.0       # Additional path loss for LoS
ETA_NLOS = 20.0     # Additional path loss for NLoS
CARRIER_FREQ_FC = 2.4e9  # 2.4 GHz
LIGHT_SPEED_C = 3e8

# --- System Parameters (from Table I) ---
SNR_THRESHOLD_DB = 7.0
SNR_THRESHOLD = 10**(SNR_THRESHOLD_DB / 10)  # Convert dB to linear scale
SNR_GAP_DB = 15.0
SNR_GAP = 10**(SNR_GAP_DB / 10)              # SNR gap for practical modulation
NOISE_POWER_DBM_PER_HZ = -174
BANDWIDTH_B = 2e6  # 2 MHz
NOISE_POWER = 10**((NOISE_POWER_DBM_PER_HZ - 30) / 10) * BANDWIDTH_B # in Watts

GU_TRANSMIT_POWER_DBM = -50
GU_TRANSMIT_POWER_P = 10**((GU_TRANSMIT_POWER_DBM - 30) / 10) # in Watts

# --- UAV Parameters ---
UAV_MAX_SPEED_VMAX = 20.0  # m/s
H_MIN = 50                 # Minimum altitude
H_MAX = 300                # Maximum altitude

# --- Data Collection Parameters ---
DEFAULT_DATA_FILE_SIZE_MBITS = 150 # Mbits
DATA_TRANSMISSION_RATE_R = BANDWIDTH_B * np.log2(1 + SNR_THRESHOLD) # bps