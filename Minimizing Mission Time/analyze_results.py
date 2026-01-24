# analyze_results.py (Final Version with Enhanced Plots)
# ==============================================================================
import os
import re
import sys
import pandas as pd
import numpy as np # <<< ADDED for np.arange, np.floor, np.ceil
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D # <<< ADDED for custom legend
import seaborn as sns
import argparse

# <<< (Logger class remains the same) >>>
class Logger:
    """A simple logger to write output to both console and a file."""
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.flush()
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    def close(self):
        self.log_file.close()

def parse_log_file(file_path):
    """Parses a single log file and extracts the summary data robustly."""
    results = {}
    method_map = {
        'V-Shaped': 'V_Shaped', 'Convex': 'Convex', 'CMC': 'CMC', 'BOB': 'BOB'
    }
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                for display_name, key_name in method_map.items():
                    if line.strip().startswith(f"{display_name} Mission Time:"):
                        parts = re.findall(r"[\d.]+", line)
                        if len(parts) >= 2:
                            results[f'{key_name}_Time'] = float(parts[0])
                            results[f'{key_name}_Length'] = float(parts[1])
                        break
    except FileNotFoundError:
        print(f"Warning: Log file not found at {file_path}"); return None
    for method_key in method_map.values():
        if f'{method_key}_Time' not in results:
            print(f"Warning: Incomplete summary in {os.path.basename(file_path)}. Skipping this file."); return None
    return results

def analyze_batch_results(batch_dir):
    """Analyzes all log files and generates summary stats and plots."""
    all_results = []
    print(f"\nAnalyzing log files in: '{batch_dir}'")
    for filename in sorted(os.listdir(batch_dir)):
        if filename.endswith("_log.txt"):
            file_path = os.path.join(batch_dir, filename)
            run_results = parse_log_file(file_path)
            if run_results:
                all_results.append(run_results)
    
    if not all_results:
        print(f"\nNo valid log files with complete summaries found in '{batch_dir}'"); return

    df = pd.DataFrame(all_results)
    
    print("\n" + "="*20 + " STATISTICAL SUMMARY " + "="*20)
    with pd.option_context('display.width', 1000, 'display.float_format', '{:,.2f}'.format): print(df.describe())
    print("="*63 + "\n")
    
    methods = ['V_Shaped', 'Convex', 'CMC', 'BOB']
    
    # <<< MODIFICATION START: Enhanced MCT Box Plot >>>
    time_df = df[[f'{m}_Time' for m in methods]]; time_df.columns = [m.replace('_', '-') for m in methods]
    plt.figure(figsize=(12, 8))
    ax_time = sns.boxplot(data=time_df, showmeans=True, meanline=True, meanprops={'color': 'cyan', 'linestyle': '--', 'linewidth': 2})
    
    means_time = time_df.mean()
    vertical_offset_time = means_time.min() * 0.01
    for i, method_name in enumerate(time_df.columns):
        mean_val = means_time[i]
        ax_time.text(i, mean_val + vertical_offset_time, f'{mean_val:.2f}', horizontalalignment='center', size='medium', color='black', weight='semibold')

    min_val_time, max_val_time = time_df.min().min(), time_df.max().max()
    tick_step_time = 20 # Time step of 20 seconds
    start_tick_time = np.floor(min_val_time / tick_step_time) * tick_step_time
    end_tick_time = np.ceil(max_val_time / tick_step_time) * tick_step_time
    ax_time.set_yticks(np.arange(start_tick_time, end_tick_time + tick_step_time, tick_step_time))

    plt.title('Mission Completion Time (MCT) Distribution', fontsize=16)
    plt.ylabel('Time (seconds)', fontsize=12); plt.grid(True, linestyle=':', alpha=0.6)
    legend_elements_time = [Line2D([0], [0], color='cyan', lw=2, label='Mean', linestyle='--')]
    ax_time.legend(handles=legend_elements_time)
    save_path_mct = os.path.join(batch_dir, 'summary_mct_boxplot.png'); plt.savefig(save_path_mct)
    print(f"Saved enhanced MCT boxplot to: '{save_path_mct}'")
    # <<< MODIFICATION END >>>

    # <<< MODIFICATION START: Enhanced Path Length Box Plot >>>
    length_df = df[[f'{m}_Length' for m in methods]]; length_df.columns = [m.replace('_', '-') for m in methods]
    plt.figure(figsize=(12, 8))
    ax_len = sns.boxplot(data=length_df, showmeans=True, meanline=True, meanprops={'color': 'cyan', 'linestyle': '--', 'linewidth': 2})
    
    means_len = length_df.mean()
    vertical_offset_len = means_len.min() * 0.01
    for i, method_name in enumerate(length_df.columns):
        mean_val = means_len[i]
        ax_len.text(i, mean_val + vertical_offset_len, f'{mean_val:.2f}', horizontalalignment='center', size='medium', color='black', weight='semibold')

    min_val_len, max_val_len = length_df.min().min(), length_df.max().max()
    tick_step_len = 250 # Distance step of 250 meters
    start_tick_len = np.floor(min_val_len / tick_step_len) * tick_step_len
    end_tick_len = np.ceil(max_val_len / tick_step_len) * tick_step_len
    ax_len.set_yticks(np.arange(start_tick_len, end_tick_len + tick_step_len, tick_step_len))

    plt.title('Total Path Length Distribution', fontsize=16)
    plt.ylabel('Distance (meters)', fontsize=12); plt.grid(True, linestyle=':', alpha=0.6)
    legend_elements_len = [Line2D([0], [0], color='cyan', lw=2, label='Mean', linestyle='--')]
    ax_len.legend(handles=legend_elements_len)
    save_path_len = os.path.join(batch_dir, 'summary_length_boxplot.png'); plt.savefig(save_path_len)
    print(f"Saved enhanced Path Length boxplot to: '{save_path_len}'")
    # <<< MODIFICATION END >>>
    
    baseline_method, comparison_methods = 'Convex', ['V_Shaped', 'CMC', 'BOB']
    avg_improvements = {}
    for method in comparison_methods:
        df[f'{method}_Improvement'] = (df[f'{baseline_method}_Time'] - df[f'{method}_Time']) / df[f'{baseline_method}_Time'] * 100
        avg_improvements[method.replace('_', '-')] = df[f'{method}_Improvement'].mean()
    improvement_df = pd.DataFrame(list(avg_improvements.items()), columns=['Method', 'Average Improvement (%)'])
    
    plt.figure(figsize=(10, 6)); barplot = sns.barplot(x='Method', y='Average Improvement (%)', data=improvement_df)
    plt.axhline(0, color='black', linewidth=0.8); plt.title(f'Average Time Improvement vs. {baseline_method.replace("_","-")}', fontsize=16)
    plt.ylabel(f'Time Saved vs. {baseline_method.replace("_","-")} (%)', fontsize=12)
    for p in barplot.patches: barplot.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    save_path_imp = os.path.join(batch_dir, 'summary_improvement_barplot.png'); plt.savefig(save_path_imp)
    print(f"Saved Improvement bar plot to: '{save_path_imp}'")

    plt.show()

if __name__ == "__main__":
    # ... (The main execution block remains the same, it is correct) ...
    parser = argparse.ArgumentParser(description="Analyze simulation batch results.")
    parser.add_argument("batch_dir", type=str, nargs='?', default=None, help="Optional: Path to the simulation batch directory.")
    args = parser.parse_args()
    target_dir = args.batch_dir
    if target_dir is None:
        print("No directory provided. Attempting to find the latest run...")
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # Assuming 'simulation_results' is at the same level as the script's parent folder
        # Adjust if your structure is different, e.g., base_results_dir = "simulation_results"
        base_results_dir = os.path.join(script_dir, "simulation_results")
        if not os.path.isdir(base_results_dir): print(f"Error: Base results directory not found at '{base_results_dir}'"); sys.exit(1)
        all_run_dirs = [d for d in os.listdir(base_results_dir) if os.path.isdir(os.path.join(base_results_dir, d)) and d.startswith('run_')]
        if not all_run_dirs: print(f"Error: No run directories found."); sys.exit(1)
        latest_run_dir_name = sorted(all_run_dirs)[-1]
        target_dir = os.path.join(base_results_dir, latest_run_dir_name)
        print(f"-> Automatically selected the latest run: '{target_dir}'")
    if not os.path.isdir(target_dir): print(f"Error: Target directory not found at '{target_dir}'")
    else:
        log_path = os.path.join(target_dir, "summary_analyzing.txt")
        original_stdout = sys.stdout; sys.stdout = Logger(log_path)
        try: analyze_batch_results(target_dir)
        except Exception as e: print("\n"+"="*20+" ANALYSIS ERROR "+"="*20); traceback.print_exc(); print("="*56)
        finally:
            if isinstance(sys.stdout, Logger): sys.stdout.close()
            sys.stdout = original_stdout; print(f"\nAnalysis complete. Summary saved to '{log_path}'")