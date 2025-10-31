# reporter.py (Version 3: Simplified Travel/Service Breakdown)
# ==============================================================================
#                      Simulation Report Generator
#
# File Objective:
# This module generates a detailed time breakdown report, separating the mission
# into 'Travel' (inter-GN flight) and 'Service' (intra-GN collection) segments.
# ==============================================================================

import numpy as np
import pandas as pd
import os
from typing import Dict, List

def generate_time_breakdown_report(
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
        # The first leg is always from Data Center
        travel_time = np.linalg.norm(v_shaped_segments[0]['end'] - v_shaped_segments[0]['start']) / uav_speed
        
        collection_segments_v = [s for s in v_shaped_segments if s['type'] == 'collection']
        for i, segment in enumerate(collection_segments_v):
            service_time = segment['service_time']
            report_data.append({
                'Method': 'V-Shaped',
                'Sequence': i + 1,
                'GN_Index': segment.get('gn_index', 'N/A'),
                'Travel_Time (s)': travel_time,
                'Service_Time (s)': service_time,
            })
            # The travel time for the next leg is the flight-out time of the current leg
            if i < len(collection_segments_v) - 1:
                current_fop = segment['fop']
                next_fip = v_shaped_segments[(i + 1) * 2]['end']
                travel_time = np.linalg.norm(next_fip - current_fop) / uav_speed
            else: # Add final travel back to DC
                last_fop = segment['fop']
                final_travel_time = np.linalg.norm(data_center_pos - last_fop) / uav_speed
                report_data.append({
                    'Method': 'V-Shaped', 'Sequence': i + 2, 'GN_Index': 'DC',
                    'Travel_Time (s)': final_travel_time, 'Service_Time (s)': 0.0,
                })

    # --- 2. Process Convex Trajectory Data ---
    if convex_result and convex_result.get('collection_segments'):
        collection_segments_c = convex_result['collection_segments']
        # The first leg is always from Data Center
        travel_time = np.linalg.norm(collection_segments_c[0]['start'] - data_center_pos) / uav_speed
        
        for i, segment in enumerate(collection_segments_c):
            service_time = segment.get('Total_Collection_Time (s)', 0)
            report_data.append({
                'Method': 'Convex',
                'Sequence': i + 1,
                'GN_Index': segment['gn_index'],
                'Travel_Time (s)': travel_time,
                'Service_Time (s)': service_time,
            })
            # The travel time for the next leg is from current Eo to next So
            if i < len(collection_segments_c) - 1:
                current_eo = segment['end']
                next_so = collection_segments_c[i+1]['start']
                travel_time = np.linalg.norm(next_so - current_eo) / uav_speed
            else: # Add final travel back to DC
                last_eo = segment['end']
                final_travel_time = np.linalg.norm(data_center_pos - last_eo) / uav_speed
                report_data.append({
                    'Method': 'Convex', 'Sequence': i + 2, 'GN_Index': 'DC',
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
                'Total_Mission_Time (s)': total_time
            })
            
    df_summary = pd.DataFrame(summary_list)
    
    # --- 4. Save Report ---
    cols = ['Method', 'Sequence', 'GN_Index', 'Travel_Time (s)', 'Service_Time (s)']
    df_final = pd.concat([df[cols], df_summary], ignore_index=True)
    df_final = df_final.sort_values(by=['Method', 'Sequence'], key=lambda x: pd.to_numeric(x, errors='coerce').fillna(float('inf')))
    
    report_path = os.path.join(output_dir, f'{run_prefix}_{uav_id}_time_breakdown_report.csv')
    df_final.to_csv(report_path, index=False, float_format='%.2f')
    print(f"\nDetailed time breakdown report saved to: {os.path.basename(report_path)}")