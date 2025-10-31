# reporter.py (Version 6: Final Correct Name and Logic)
# ==============================================================================
#                      Simulation Report Generator
# ==============================================================================

import numpy as np
import pandas as pd
import os
from typing import Dict, List

# <<< RENAMED function to match the call in main.py >>>
def generate_flight_log_report(
    run_prefix: str,
    output_dir: str,
    uav_id: str,
    v_shaped_segments: List[Dict],
    convex_result: Dict,
    data_center_pos: np.ndarray,
    uav_speed: float,
    **kwargs
) -> None:
    """
    Generates a CSV report with a Travel/Service time breakdown for each method.
    """
    report_data = []

    # --- 1. Process V-Shaped Trajectory Data ---
    if v_shaped_segments:
        collection_segments_v = [s for s in v_shaped_segments if s['type'] == 'collection']
        
        previous_fop = data_center_pos
        for i, segment in enumerate(collection_segments_v):
            current_fip = v_shaped_segments[i * 2]['end']
            travel_time = np.linalg.norm(current_fip - previous_fop) / uav_speed
            
            # The service time is pre-calculated in the main loop
            service_time = segment['service_time']

            report_data.append({
                'Method': 'V-Shaped',
                'Sequence': i + 1,
                'GN_Index': segment.get('gn_index', 'N/A'),
                'Travel_Time (s)': travel_time,
                'Service_Time (s)': service_time,
            })
            previous_fop = segment['fop']
            
        # Add the final travel back to the data center
        final_travel_time = np.linalg.norm(data_center_pos - previous_fop) / uav_speed
        report_data.append({
            'Method': 'V-Shaped', 'Sequence': len(collection_segments_v) + 1, 'GN_Index': 'DC',
            'Travel_Time (s)': final_travel_time, 'Service_Time (s)': 0.0,
        })

    # --- 2. Process Convex Trajectory Data ---
    if convex_result and convex_result.get('collection_segments'):
        collection_segments_c = convex_result['collection_segments']
        previous_eo = data_center_pos
        
        for i, segment in enumerate(collection_segments_c):
            so_point = segment['start']
            travel_time = np.linalg.norm(so_point - previous_eo) / uav_speed
            service_time = segment.get('Total_Collection_Time (s)', 0)

            report_data.append({
                'Method': 'Convex',
                'Sequence': i + 1,
                'GN_Index': segment['gn_index'],
                'Travel_Time (s)': travel_time,
                'Service_Time (s)': service_time,
            })
            previous_eo = segment['end']
            
        # Add the final travel back to DC for Convex
        final_travel_time = np.linalg.norm(data_center_pos - previous_eo) / uav_speed
        report_data.append({
            'Method': 'Convex', 'Sequence': len(collection_segments_c) + 1, 'GN_Index': 'DC',
            'Travel_Time (s)': final_travel_time, 'Service_Time (s)': 0.0,
        })

    # --- 3. Generate DataFrame and Summary ---
    if not report_data:
        print("No data to generate report.")
        return

    df = pd.DataFrame(report_data)
    
    summary_list = []
    for method in ['V-Shaped', 'Convex']:
        method_df = df[df['Method'] == method]
        if not method_df.empty:
            total_time = method_df['Travel_Time (s)'].sum() + method_df['Service_Time (s)'].sum()
            summary_list.append({
                'Method': method,
                'Sequence': 'TOTAL',
                'GN_Index': '-',
                'Travel_Time (s)': method_df['Travel_Time (s)'].sum(),
                'Service_Time (s)': method_df['Service_Time (s)'].sum(),
                'Total_Mission_Time (s)': total_time,
            })
            
    df_summary = pd.DataFrame(summary_list)
    
    # --- 4. Save Report ---
    df_final = pd.concat([df, df_summary], ignore_index=True)
    df_final = df_final.sort_values(by=['Method', 'Sequence'], key=lambda x: pd.to_numeric(x, errors='coerce').fillna(float('inf')))
    
    cols = ['Method', 'Sequence', 'GN_Index', 'Travel_Time (s)', 'Service_Time (s)', 'Total_Mission_Time (s)']
    df_final = df_final.reindex(columns=cols)

    report_path = os.path.join(output_dir, f'{run_prefix}_{uav_id}_time_breakdown_report.csv')
    df_final.to_csv(report_path, index=False, float_format='%.2f')
    print(f"\nDetailed time breakdown report saved to: {os.path.basename(report_path)}")