# analyze_results.py (Final Version with Enhanced Plots & BOB-F)
# ==============================================================================
import os
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import argparse

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
    # <<< MODIFIED: Added BOB-F to the parsing map >>>
    # Note: Keys here match the print statements in main.py exactly
    method_map = {
        'V-Shaped': 'V_Shaped', 
        'Convex': 'Convex', 
        'CMC': 'CMC', 
        'BOB-V': 'BOB_V',   # Changed from 'BOB' to 'BOB-V' to match main.py
        'BOB-F': 'BOB_F'    # Added new method
    }
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                for display_name, key_name in method_map.items():
                    # We use startswith to catch "BOB-V Mission Time:" etc.
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
            print(f"Warning: Incomplete summary in {os.path.basename(file_path)} (missing {method_key}). Skipping this file."); return None
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
    
    # <<< MODIFIED: Methods list now includes BOB_V and BOB_F >>>
    methods = ['V_Shaped', 'Convex', 'CMC', 'BOB_V', 'BOB_F']
    
    # ---------------------------------------------------------
    # 1. Boxplots (Time & Length)
    # ---------------------------------------------------------
    time_df = df[[f'{m}_Time' for m in methods]]; time_df.columns = [m.replace('_', '-') for m in methods]
    plt.figure(figsize=(12, 8))
    ax_time = sns.boxplot(data=time_df, showmeans=True, meanline=True, meanprops={'color': 'cyan', 'linestyle': '--', 'linewidth': 2})
    
    means_time = time_df.mean()
    vertical_offset_time = means_time.min() * 0.01
    for i, method_name in enumerate(time_df.columns):
        mean_val = means_time.iloc[i]
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
    plt.close() # Close figure to free memory
    
    length_df = df[[f'{m}_Length' for m in methods]]; length_df.columns = [m.replace('_', '-') for m in methods]
    plt.figure(figsize=(12, 8))
    ax_len = sns.boxplot(data=length_df, showmeans=True, meanline=True, meanprops={'color': 'cyan', 'linestyle': '--', 'linewidth': 2})
    
    means_len = length_df.mean()
    vertical_offset_len = means_len.min() * 0.01
    for i, method_name in enumerate(length_df.columns):
        mean_val = means_len.iloc[i]
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
    plt.close()
    
    # ---------------------------------------------------------
    # 2. Improvement Barplots
    # ---------------------------------------------------------
    baseline_method = 'Convex'
    comparison_methods = ['V_Shaped', 'CMC', 'BOB_V', 'BOB_F']
    
    # --- Local Averages ---
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
    plt.close()
    
    # --- Global Averages ---
    print("\n" + "="*20 + " GLOBAL AVERAGE COMPARISON " + "="*20)
    baseline_mean_time = df[f'{baseline_method}_Time'].mean()
    print(f"Global Average Time for {baseline_method}: {baseline_mean_time:.2f}s")
    
    global_improvement_data = []
    for method in comparison_methods:
        method_mean_time = df[f'{method}_Time'].mean()
        imp_percent = (baseline_mean_time - method_mean_time) / baseline_mean_time * 100
        global_improvement_data.append({'Method': method.replace('_', '-'), 'Improvement of Averages (%)': imp_percent})
        print(f"{method.replace('_', '-')} Average: {method_mean_time:.2f}s -> Improvement: {imp_percent:.2f}%")
        
    df_global_imp = pd.DataFrame(global_improvement_data)
    plt.figure(figsize=(10, 6))
    barplot_global = sns.barplot(x='Method', y='Improvement of Averages (%)', data=df_global_imp)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(f'Improvement of Global Average Time vs. {baseline_method.replace("_","-")}', fontsize=16)
    plt.ylabel(f'Time Saved (Global Avg) (%)', fontsize=12)
    for p in barplot_global.patches:
        barplot_global.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    save_path_global_imp = os.path.join(batch_dir, 'summary_improvement_of_averages_barplot.png')
    plt.savefig(save_path_global_imp)
    print(f"Saved Improvement (Global Averages) bar plot to: '{save_path_global_imp}'")
    print("="*67 + "\n")
    plt.close()
    
    # ---------------------------------------------------------
    # 3. Head-to-Head Analysis (Pie Charts)
    # ---------------------------------------------------------
    
    def generate_h2h_pie_chart(df, method_a_key, method_b_key, display_name_a, display_name_b, filename):
        print(f"\n" + "="*20 + f" HEAD-TO-HEAD: {display_name_a} vs {display_name_b} " + "="*20)
        
        # Positive diff means B took longer (A won)
        diffs = df[f'{method_b_key}_Time'] - df[f'{method_a_key}_Time']
        
        wins = diffs[diffs > 0]
        losses = diffs[diffs < 0]
        draws = diffs[diffs == 0]
        
        num_wins, num_losses, num_draws = len(wins), len(losses), len(draws)
        total_runs = len(df)
        
        win_rate = (num_wins / total_runs) * 100
        loss_rate = (num_losses / total_runs) * 100
        
        avg_win_margin = wins.mean() if num_wins > 0 else 0.0
        avg_loss_margin = (-losses).mean() if num_losses > 0 else 0.0
        
        print(f"Total Runs: {total_runs}")
        print(f"{display_name_a} Wins:   {num_wins} ({win_rate:.1f}%)")
        print(f"{display_name_a} Losses: {num_losses} ({loss_rate:.1f}%)")
        print(f"Draws:          {num_draws}")
        print("-" * 40)
        print(f"When {display_name_a} wins, it saves an average of: {avg_win_margin:.2f} seconds")
        print(f"When {display_name_a} loses, it lags by an average of:  {avg_loss_margin:.2f} seconds")
        print("="*65 + "\n")

        plt.figure(figsize=(8, 8))
        labels = [f'{display_name_a} Wins\n({num_wins})', f'{display_name_b} Wins\n({num_losses})']
        sizes = [num_wins, num_losses]
        colors = ['#66b3ff', '#ff9999'] # Blue for A win, Red for B win
        
        if num_draws > 0:
            labels.append(f'Draws\n({num_draws})')
            sizes.append(num_draws)
            colors.append('#99ff99')

        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        
        plt.title(f'Head-to-Head Win Rate: {display_name_a} vs {display_name_b}\n(Total {total_runs} Runs)', fontsize=16)
        plt.tight_layout()
        
        save_path_pie = os.path.join(batch_dir, filename)
        plt.savefig(save_path_pie)
        print(f"Saved Head-to-Head pie chart to: '{save_path_pie}'")
        plt.close()

    # Generate Chart 1: BOB-V vs V-Shaped (Original)
    generate_h2h_pie_chart(df, 'BOB_V', 'V_Shaped', 'BOB-V', 'V-Shaped', 'summary_bob_v_vs_vshaped_pie.png')
    
    # Generate Chart 2: BOB-F vs V-Shaped (New)
    generate_h2h_pie_chart(df, 'BOB_F', 'V_Shaped', 'BOB-F', 'V-Shaped', 'summary_bob_f_vs_vshaped_pie.png')

    print("\nVisualizations generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze simulation batch results.")
    parser.add_argument("batch_dir", type=str, nargs='?', default=None, help="Optional: Path to the simulation batch directory.")
    args = parser.parse_args()
    target_dir = args.batch_dir
    if target_dir is None:
        print("No directory provided. Attempting to find the latest run...")
        script_dir = os.path.dirname(os.path.realpath(__file__))
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