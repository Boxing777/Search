import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def solve_and_plot_random_convex(num_gns=5, area_size=(2000, 2000), comm_radius=250.0):
    """
    Generates random GNs, solves the convex shortest path problem, and plots the result.
    
    Args:
        num_gns (int): Number of Ground Nodes to generate.
        area_size (tuple): Width and Height of the simulation area.
        comm_radius (float): Communication radius (D).
    """
    
    # 1. Setup Scenario
    # --- MODIFICATION START ---
    # Fixed Data Center (Start/End) position at the EXACT CENTER
    dc_pos = np.array([area_size[0] / 2, area_size[1] / 2]) 
    # --- MODIFICATION END ---
    
    # Randomly generate Ground Nodes within the area
    # We add padding (200m) from edges so they don't spawn too close to the border
    np.random.seed(None) # Ensure random results each time
    gns = np.random.uniform(
        low=[200, 200], 
        high=[area_size[0]-200, area_size[1]-200], 
        size=(num_gns, 2)
    )
    
    # Sort GNs by X-coordinate to determine the visiting order
    # (Note: In a real scenario, you'd solve TSP first, but here we fix the order by position)
    gns = gns[gns[:, 0].argsort()]

    N = len(gns)

    # 2. Convex Optimization Formulation
    # Variables: So (Entry points) and Eo (Exit points) for each GN
    So = cp.Variable((N, 2), name="So")
    Eo = cp.Variable((N, 2), name="Eo")
    
    cost = 0
    # Segment 1: Data Center -> First Entry Point
    cost += cp.norm(So[0] - dc_pos)
    
    for i in range(N):
        # Path inside the circle (So -> Eo)
        cost += cp.norm(Eo[i] - So[i])
        
        # Path between circles (Eo_current -> So_next)
        if i < N - 1:
            cost += cp.norm(So[i+1] - Eo[i])
            
    # Final Segment: Last Exit Point -> Data Center
    cost += cp.norm(dc_pos - Eo[N-1])
    
    # Constraints: Points must be within their respective circles
    constraints = []
    for i in range(N):
        constraints.append(cp.norm(So[i] - gns[i]) <= comm_radius)
        constraints.append(cp.norm(Eo[i] - gns[i]) <= comm_radius)
        
    # Solve
    print(f"Solving for {num_gns} random nodes...")
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    
    if prob.status != 'optimal':
        print("Optimization failed.")
        return

    # Extract results
    so_val = So.value
    eo_val = Eo.value

    # 3. Visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot Communication Circles
    for i, gn in enumerate(gns):
        circle = plt.Circle(gn, comm_radius, color='gray', fill=False, linestyle='--', alpha=0.5)
        ax.add_patch(circle)
        ax.plot(gn[0], gn[1], 'ko', markersize=4, alpha=0.5) # GN Center
        ax.text(gn[0], gn[1]+20, f'GN{i+1}', ha='center', fontsize=8, color='gray')

    # Plot Data Center
    ax.plot(dc_pos[0], dc_pos[1], 'k*', markersize=20, label='Data Center (Center)', zorder=10)
    
    # Construct trajectory arrays for plotting
    path_x = [dc_pos[0]]
    path_y = [dc_pos[1]]
    
    for i in range(N):
        # Segment: Previous -> So
        path_x.append(so_val[i,0])
        path_y.append(so_val[i,1])
        # Segment: So -> Eo
        path_x.append(eo_val[i,0])
        path_y.append(eo_val[i,1])
        
    # Segment: Last Eo -> DC
    path_x.append(dc_pos[0])
    path_y.append(dc_pos[1])
    
    # Plot Trajectory
    ax.plot(path_x, path_y, 'r-', linewidth=2.5, label='Optimal Path (Rubber Band)')

    # Plot Entry (So) and Exit (Eo) points
    ax.scatter(so_val[:,0], so_val[:,1], color='blue', marker='*', s=150, zorder=10, label='Entry Point ($So$)')
    ax.scatter(eo_val[:,0], eo_val[:,1], color='green', marker='*', s=150, zorder=10, label='Exit Point ($Eo$)')

    # Add text labels for specific points
    for i in range(N):
        ax.text(so_val[i,0], so_val[i,1]-40, f'$So_{{{i+1}}}$', color='blue', fontsize=9, ha='center')
        ax.text(eo_val[i,0], eo_val[i,1]+25, f'$Eo_{{{i+1}}}$', color='green', fontsize=9, ha='center')

    # Settings
    ax.set_aspect('equal')
    ax.set_title(f'Convex Path with Center Start/End ({num_gns} Nodes)', fontsize=16)
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Configure Legend
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    # Set fixed limits to show the center clearly
    ax.set_xlim(0, area_size[0])
    ax.set_ylim(0, area_size[1])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- CONFIGURATION ---
    NUMBER_OF_NODES = 5
    
    solve_and_plot_random_convex(
        num_gns=NUMBER_OF_NODES, 
        area_size=(2000, 2000), # Square area
        comm_radius=250.0
    )