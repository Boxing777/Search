# ==============================================================================
#                      Utility Functions Module
#
# File Objective:
# This file provides common helper functions, especially for unit conversions
# (e.g., dBm to Watts, dB to linear scale), that are used across multiple
# modules in the simulation. This avoids code duplication and improves
# maintainability.
# ==============================================================================

import numpy as np

def dbm_to_watts(dbm: float) -> float:
    """
    Converts power from dBm to Watts.
    """
    return 10**((dbm - 30) / 10.0)

def watts_to_dbm(watts: float) -> float:
    """
    Converts power from Watts to dBm.
    """
    if watts <= 0:
        return -np.inf
    return 10 * np.log10(watts) + 30

def db_to_linear(db: float) -> float:
    """
    Converts a ratio from decibels (dB) to a linear scale.
    """
    return 10**(db / 10.0)

def linear_to_db(linear: float) -> float:
    """
    Converts a ratio from a linear scale to decibels (dB).
    """
    if linear <= 0:
        return -np.inf
    return 10 * np.log10(linear)
