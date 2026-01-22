# analyze_results.py (Final Robust Version)
# ==============================================================================
import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def parse_log_file(file_path):
    """Parses a single log file and extracts the summary data robustly."""
    results = {}
    method_map = {
        'V-Shaped': 'V_Shaped',
        'Convex': 'Convex',
        'CMC': 'CMC',
        'BOB': 'BOB'
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                for display_name, key_name in method_map.items():
                    # Match lines like "V-Shaped Mission Time: 357.66s | Path Length: 7127.13m"
                    if line.strip().startswith(f"{display_name} Mission Time:"):
                        parts = re.findall(r"[\d.]+", line)
                        if len(parts) >= 2:
                            results[f'{key_name}_Time'] = float(parts[0])
                            results[f'{key_name}_Length'] = float(parts[1])
                        break # Move to the next line once a match is found
    except FileNotFoundError:
        print(f"Warning: Log file not found at {file_path}")
        return None
        
    # Ensure all required keys are present, otherwise the run is invalid
    for method_key in method_map.values():
        if f'{method_key}_Time' not in results:
            print(f"Warning: Incomplete summary in {file_path}. Skipping this file.")
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
    
    print("\n" + "="*20 + " STATISTICAL SUMMARY " + "="*20)
    with pd.option_context('display.float_format', '{:,.2f}'.format):
        print(df.describe())
    print("="*63 + "\n")
    
    methods = ['V_Shaped', 'Convex', 'CMC', 'BOB']
    
    # --- Box Plot for Mission Completion Time ---
    time_df = df[[f'{m}_Time' for m in methods]]
    time_df.columns = [m.replace('_', '-') for m in methods]
    
    plt.figure(figsize=(10, 6)); sns.boxplot(data=time_df)
    plt.title('Mission Completion Time (MCT) Distribution', fontsize=16)
    plt.ylabel('Time (seconds)', fontsize=12); plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(batch_dir, 'summary_mct_boxplot.png'))
    print(f"Saved MCT boxplot to '{os.path.join(batch_dir, 'summary_mct_boxplot.png')}'")

    # --- Box Plot for Path Length ---
    length_df = df[[f'{m}_Length' for m in methods]]
    length_df.columns = [m.replace('_', '-') for m in methods]
    
    plt.figure(figsize=(10, 6)); sns.boxplot(data=length_df)
    plt.title('Total Path Length Distribution', fontsize=16)
    plt.ylabel('Distance (meters)', fontsize=12); plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(batch_dir, 'summary_length_boxplot.png'))
    print(f"Saved Path Length boxplot to '{os.path.join(batch_dir, 'summary_length_boxplot.png')}'")
    
    # --- Relative Performance Bar Plot ---
    baseline_method, comparison_methods = 'Convex', ['V_Shaped', 'CMC', 'BOB']
    avg_improvements = {}
    for method in comparison_methods:
        df[f'{method}_Improvement'] = (df[f'{baseline_method}_Time'] - df[f'{method}_Time']) / df[f'{baseline_method}_Time'] * 100
        avg_improvements[method] = df[f'{method}_Improvement'].mean()
    improvement_df = pd.DataFrame(list(avg_improvements.items()), columns=['Method', 'Average Improvement (%)'])
    
    plt.figure(figsize=(10, 6)); barplot = sns.barplot(x='Method', y='Average Improvement (%)', data=improvement_df)
    plt.axhline(0, color='black', linewidth=0.8); plt.title(f'Average Time Improvement vs. {baseline_method}', fontsize=16)
    plt.ylabel('Time Saved (%)', fontsize=12)
    for p in barplot.patches: barplot.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    plt.savefig(os.path.join(batch_dir, 'summary_improvement_barplot.png')); print(f"Saved Improvement bar plot to '{os.path.join(batch_dir, 'summary_improvement_barplot.png')}'")

    plt.show()

if __name__ == "__main__":
    # ... (The main execution block is fine, no changes needed here) ...
    parser = argparse.ArgumentParser(description="Analyze simulation batch results.")
    parser.add_argument("batch_dir", type=str, nargs='?', default=None, help="Optional: Path to the simulation batch directory.")
    args = parser.parse_args()
    target_dir = args.batch_dir
    if target_dir is None:
        print("No directory provided. Attempting to find the latest run...")
        base_results_dir = os.path.join("..", "simulation_results")
        if not os.path.isdir(base_results_dir): print(f"Error: Base results directory not found."); sys.exit(1)
        all_run_dirs = [d for d in os.listdir(base_results_dir) if os.path.isdir(os.path.join(base_results_dir, d)) and d.startswith('run_')]
        if not all_run_dirs: print(f"Error: No run directories found."); sys.exit(1)
        latest_run_dir_name = sorted(all_run_dirs)[-1]
        target_dir = os.path.join(base_results_dir, latest_run_dir_name)
        print(f"-> Automatically selected the latest run: '{target_dir}'")
    if not os.path.isdir(target_dir): print(f"Error: Target directory not found at '{target_dir}'")
    else: analyze_batch_results(target_dir)