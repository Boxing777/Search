# plot_final_performance.py
# ==============================================================================
#      Academic Plotter: Sensitivity Analysis under Varying Data Loads
#
# File Objective:
# Automatically parses statistical summaries from varying data loads (1MB to 32MB)
# and generates publication-quality performance curves comparing:
# 1. Convex (Geometric Baseline)
# 2. V-Shaped (Greedy Baseline)
# 3. BOB-F (Proposed Boundary-Overlap Strategy, Labeled as 'RPA-BO')
# ==============================================================================

import os
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_summary_file(file_path: str) -> dict:
    """
    Parses a single statistical summary text file to extract mean values
    for BOB-F, V-Shaped, and Convex metrics.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the line starting with "mean" inside the STATISTICAL SUMMARY block
    lines = content.split('\n')
    mean_line = None
    for line in lines:
        if line.strip().startswith('mean'):
            mean_line = line
            break
            
    if not mean_line:
        raise ValueError(f"Could not find 'mean' row in {file_path}")

    # Extract all numerical values from the mean row (handling commas in numbers)
    # Remove commas first to avoid splitting numbers like "6,191.67" into "6" and "191.67"
    clean_line = mean_line.replace(',', '')
    values = [float(val) for val in re.findall(r'[-+]?\d*\.\d+|\d+', clean_line)]

    # Based on the column indices in the analyzer's output:
    # 0: BOB_F_Time, 1: BOB_F_Length
    # 2: V_Shaped_Time, 3: V_Shaped_Length
    # 4: Convex_Time, 5: Convex_Length
    return {
        'BOB_F_Time': values[0],
        'BOB_F_Length': values[1],
        'V_Shaped_Time': values[2],
        'V_Shaped_Length': values[3],
        'Convex_Time': values[4],
        'Convex_Length': values[5]
    }

def generate_performance_curves():
    data_dir = "final_simulation"
    data_loads_mb = [1, 2, 4, 8, 16]
    
    # Storage for parsed metrics
    mct_data = {'Convex': [], 'V-Shaped': [], 'BOB-F': []}
    length_data = {'Convex': [], 'V-Shaped': [], 'BOB-F': []}
    
    print(f"Scanning directory '{data_dir}' for summary logs...")
    
    # Read and parse each file sequentially
    for load in data_loads_mb:
        file_name = f"{load}MB.txt"
        file_path = os.path.join(data_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"WARNING: Required file '{file_name}' not found in '{data_dir}'. Aborting.")
            return
            
        try:
            stats = parse_summary_file(file_path)
            
            # Map parsed data to corresponding structures (Corrected Underscore Keys)
            mct_data['Convex'].append(stats['Convex_Time'])
            mct_data['V-Shaped'].append(stats['V_Shaped_Time'])
            mct_data['BOB-F'].append(stats['BOB_F_Time'])
            
            length_data['Convex'].append(stats['Convex_Length'])
            length_data['V-Shaped'].append(stats['V_Shaped_Length'])
            length_data['BOB-F'].append(stats['BOB_F_Length'])
            
            print(f"  Successfully parsed {file_name}")
        except Exception as e:
            print(f"  ERROR parsing {file_name}: {e}")
            return

    # --- Plotting Style Configurations ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    
    # Figure 1: Data Requirement vs. Mission Completion Time (MCT)
    # Width is increased to 9.5 to accommodate the external legend cleanly
    fig1, ax1 = plt.subplots(figsize=(9.5, 6))
    
    # Line styles, colors, and markers aligned with visualizer.py
    ax1.plot(data_loads_mb, mct_data['Convex'], marker='d', linestyle=':', color='darkblue', 
             linewidth=2.0, markersize=8, label='Convex')
    ax1.plot(data_loads_mb, mct_data['V-Shaped'], marker='s', linestyle='--', color='green', 
             linewidth=2.0, markersize=8, label='V-Shaped')
    ax1.plot(data_loads_mb, mct_data['BOB-F'], marker='o', linestyle='-', color='red', 
             linewidth=2.5, markersize=8, label='RPA-BO')
    
    ax1.set_xlabel("Data Requirement per GN (MB)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Average Mission Completion Time (s)", fontsize=12, fontweight='bold')
    ax1.set_xscale('log', base=2) # Using log-scale for binary-increasing data loads
    ax1.set_xticks(data_loads_mb)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
 #   ax1.grid(True, which="both", linestyle=':', alpha=0.5)
    
    # Place the legend outside the axes bounding box on the upper right
    ax1.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0.0, fontsize='medium')
    
    save_path_mct = os.path.join(data_dir, "final_mct_comparison.png")
    plt.savefig(save_path_mct, dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # Figure 2: Data Requirement vs. Path Length
    # Width is increased to 9.5 to accommodate the external legend cleanly
    fig2, ax2 = plt.subplots(figsize=(9.5, 6))
    
    # Line styles, colors, and markers aligned with visualizer.py
    ax2.plot(data_loads_mb, length_data['Convex'], marker='d', linestyle=':', color='darkblue', 
             linewidth=2.0, markersize=8, label='Convex')
    ax2.plot(data_loads_mb, length_data['V-Shaped'], marker='s', linestyle='--', color='green', 
             linewidth=2.0, markersize=8, label='V-Shaped')
    ax2.plot(data_loads_mb, length_data['BOB-F'], marker='o', linestyle='-', color='red', 
             linewidth=2.5, markersize=8, label='RPA-BO')
    
    ax2.set_xlabel("Data Requirement per GN (MB)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Average Flight Path Distance (meters)", fontsize=12, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(data_loads_mb)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
 #   ax2.grid(True, which="both", linestyle=':', alpha=0.5)
    
    # Place the legend outside the axes bounding box on the upper right
    ax2.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0.0, fontsize='medium')
    
    save_path_len = os.path.join(data_dir, "final_length_comparison.png")
    plt.savefig(save_path_len, dpi=300, bbox_inches='tight')
    plt.close(fig2)

    print("\n==================== PERFORMANCE PLOTS SAVED ====================")
    print(f"Saved MCT comparison plot to: {save_path_mct}")
    print(f"Saved Path Length comparison plot to: {save_path_len}")
    print("=================================================================")

if __name__ == "__main__":
    generate_performance_curves()