# ==============================================================================
#                      Utility Functions Module (MODIFIED)
#
# File Objective:
# This file provides common helper functions. A new function to calculate
# circle intersections has been added to support the JOFC algorithm.
# ==============================================================================

import numpy as np
from typing import List

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

# <<< NEW FUNCTION >>>
def get_circle_intersections(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float) -> List[np.ndarray]:
    """
    Calculates the intersection points of two circles.
    
    Args:
        c1 (np.ndarray): Center of the first circle (x1, y1).
        r1 (float): Radius of the first circle.
        c2 (np.ndarray): Center of the second circle (x2, y2).
        r2 (float): Radius of the second circle.
        
    Returns:
        List[np.ndarray]: A list containing 0, 1, or 2 intersection points.
    """
    # Distance between centers
    d = np.linalg.norm(c1 - c2)
    
    # Check for non-intersecting, touching, or identical cases
    if d > r1 + r2 or d < abs(r1 - r2) or (d == 0 and r1 != r2):
        return [] # No intersection
    
    x1, y1 = c1
    x2, y2 = c2
    
    # Formula from https://mathworld.wolfram.com/Circle-CircleIntersection.html
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(max(0, r1**2 - a**2)) # Use max(0,...) to avoid floating point errors
    
    # Midpoint between the intersections
    x_mid = x1 + a * (x2 - x1) / d
    y_mid = y1 + a * (y2 - y1) / d
    
    # Calculate the two intersection points
    p1 = np.array([
        x_mid + h * (y2 - y1) / d,
        y_mid - h * (x2 - x1) / d
    ])
    
    if d == r1 + r2 or d == abs(r1-r2): # Circles touch at one point
        return [p1]
    
    p2 = np.array([
        x_mid - h * (y2 - y1) / d,
        y_mid + h * (x2 - x1) / d
    ])
    
    return [p1, p2]