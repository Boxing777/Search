# altitude_optimizer.py
from scipy.optimize import minimize_scalar
from channel_model import get_transmission_radius
from config import H_MIN, H_MAX

def find_optimal_altitude():
    """
    Finds the optimal altitude H that maximizes the transmission radius D_H.
    This corresponds to solving problem (P4) in the paper.
    """
    print("Step 1: Optimizing UAV altitude...")

    # We want to maximize D_H(H), which is equivalent to minimizing -D_H(H).
    def objective_function(h):
        return -get_transmission_radius(h)

    # `minimize_scalar` is perfect for this 1D optimization problem.
    result = minimize_scalar(
        objective_function,
        bounds=(H_MIN, H_MAX),
        method='bounded'
    )

    optimal_h = result.x
    max_d_h = -result.fun  # Negate back to get the maximum radius

    print(f"Optimal altitude H* = {optimal_h:.2f} m")
    print(f"Maximized transmission radius D_H* = {max_d_h:.2f} m")
    
    return optimal_h, max_d_h