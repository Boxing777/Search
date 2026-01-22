# analyze_results.py (Final Corrected Version)
# ==============================================================================
#                      Simulation Results Analyzer
#
# File Objective:
# This script post-processes the simulation results. It automatically finds
# the latest simulation run directory (or accepts a specific one), parses all
# log files within it, calculates descriptive statistics, and generates
# comparative plots (box plots) for key performance indicators.
# ==============================================================================

import os
import re
import sys # <<< ADDED for sys.exit()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def parse_log_file(file_path):
    """Parses a single log file and extracts the summary data."""
    results = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Warning: Log file not found at {file_path}")
        return None

    summary_pattern = re.compile(
        r"V-Shaped Mission Time:\s*([\d.]+).*?Path Length:\s*([\d.]+).*?"
        r"Convex Mission Time:\s*([\d.]+).*?Path Length:\s*([\d.]+).*?"
        r"CMC Mission Time:\s*([\d.]+).*?Path Length:\s*([\d.]+).*?"
        r"BOB Mission Time:\s*([\d.]+).*?Path Length:\s*([\d.]+)",
        re.DOTALL
    )
    
    match = summary_pattern.search(content)
    if match:
        try:
            results['V_Shaped_Time'] = float(match.group(1))
            results['V_Shaped_Length'] = float(match.group(2))
            results['Convex_Time'] = float(match.group(3))
            results['Convex_Length'] = float(match.group(4))
            results['CMC_Time'] = float(match.group(5))
            results['CMC_Length'] = float(match.group(6))
            results['BOB_Time'] = float(match.group(7))
            results['BOB_Length'] = float(match.group(8))
        except (ValueError, IndexError):
            print(f"Warning: Could not parse numbers from summary in {file_path}")
            return None
    else:
        # Fallback for logs that might not contain all four methods
        print(f"Warning: Could not find the full summary block in {file_path}. Skipping.")
        return None

    return results

def analyze_batch_results(batch_dir):
    """Analyzes all log files in a given batch directory."""
    all_results = []
    print(f"\nAnalyzing log files in: '{batch_dir}'")
    for filename in sorted(os.listdir(batch_dir)):
        if filename.endswith("_log.txt"):
            file_path = os.path.join(batch_dir, filename)
            run_results = parse_log_file(file_path)
            if run_results:
                all_results.append(run_results)
    
    if not all_results:
        print(f"No valid log files with complete summaries found in '{batch_dir}'")
        return

    df = pd.DataFrame(all_results)
    
    # --- 1. Print Statistical Summary ---
    print("\n" + "="*20 + " STATISTICAL SUMMARY " + "="*20)
    # Use describe() for a full statistical overview
    with pd.option_context('display.float_format', '{:,.2f}'.format):
        print(df.describe())
    print("="*63 + "\n")
    
    # Define the methods for plotting
    methods = ['V_Shaped', 'Convex', 'CMC', 'BOB']
    
    # --- 2. Generate Box Plot for Mission Completion Time ---
    time_df = df[[f'{m}_Time' for m in methods]]
    time_df.columns = [m.replace('_', '-') for m in methods]
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=time_df)
    plt.title('Mission Completion Time (MCT) Distribution', fontsize=16)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    save_path_mct = os.path.join(batch_dir, 'summary_mct_boxplot.png')
    plt.savefig(save_path_mct)
    print(f"Saved MCT boxplot to: '{save_path_mct}'")

    # --- 3. Generate Box Plot for Path Length ---
    length_df = df[[f'{m}_Length' for m in methods]]
    length_df.columns = [m.replace('_', '-') for m in methods]
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=length_df)
    plt.title('Total Path Length Distribution', fontsize=16)
    plt.ylabel('Distance (meters)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    save_path_len = os.path.join(batch_dir, 'summary_length_boxplot.png')
    plt.savefig(save_path_len)
    print(f"Saved Path Length boxplot to: '{save_path_len}'")
    
    # Show plots at the very end
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze simulation batch results.")
    parser.add_argument(
        "batch_dir", 
        type=str, 
        nargs='?', # Makes the argument optional
        default=None, 
        help="Optional: Path to a specific simulation batch directory. If not provided, the latest one in 'simulation_results' will be used."
    )
    args = parser.parse_args()
    
    target_dir = args.batch_dir

    # If no directory is provided by the user, find the latest one automatically
    if target_dir is None:
        print("No directory provided. Attempting to find the latest run...")
        base_results_dir = "simulation_results"
        
        if not os.path.isdir(base_results_dir):
            print(f"Error: Base results directory '{base_results_dir}' not found.")
            sys.exit(1)
        
        # <<< BUGFIX IS HERE: This logic block was incorrect/misplaced before >>>
        # Get all subdirectories in the base results directory that start with 'run_'
        all_run_dirs = [d for d in os.listdir(base_results_dir) if os.path.isdir(os.path.join(base_results_dir, d)) and d.startswith('run_')]
        
        if not all_run_dirs:
            print(f"Error: No run directories found in '{base_results_dir}'.")
            sys.exit(1)
            
        # Sort the directories alphabetically (which is also chronologically) and pick the last one
        latest_run_dir_name = sorted(all_run_dirs)[-1]
        target_dir = os.path.join(base_results_dir, latest_run_dir_name)
        print(f"-> Automatically selected the latest run: '{target_dir}'")

    # Final check to ensure the target directory exists before analysis
    if not os.path.isdir(target_dir):
        print(f"Error: Target directory not found at '{target_dir}'")
    else:
        analyze_batch_results(target_dir)