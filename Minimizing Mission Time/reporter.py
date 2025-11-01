# reporter.py (Version 7: Final Flight Log Format)
# ==============================================================================
#                      Simulation Report Generator
# ==============================================================================

import numpy as np
import pandas as pd
import os
from typing import Dict, List

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
    Generates a detailed flight log CSV report with specific distance and time breakdowns.
    """
    report_data = []

    # --- 1. Process V-Shaped Trajectory Data ---
    if v_shaped_segments:
        collection_segments_v = [s for s in v_shaped_segments if s['type'] == 'collection']
        
        previous_fop = data_center_pos
        for i, segment in enumerate(collection_segments_v):
            current_fip = v_shaped_segments[i * 2]['end']
            
            fly_in_dist = np.linalg.norm(current_fip - previous_fop)
            fly_in_time = fly_in_dist / uav_speed
            
            collection_dist = np.linalg.norm(segment['oh'] - segment['fip']) + np.linalg.norm(segment['fop'] - segment['oh'])
            
            # The service_time from main.py is either (t_in + t_collect) or just (t_collect).
            # We need to robustly extract just the collection part.
            is_overlapping = (fly_in_dist < 1e-6)
            collection_time = segment['service_time'] if is_overlapping else (segment['service_time'] - fly_in_time)
            
            report_data.append({
                'Method': 'V-Shaped',
                'Sequence': i + 1,
                'GN_Index': segment.get('gn_index', 'N/A'),
                'Fly_IN_Time (s)': fly_in_time,
                'Fly_IN_Distance (m)': fly_in_dist,
                'Collection_Time (s)': collection_time,
                'Collection_Distance (m)': collection_dist,
                'Fly_Back_Time (s)': 0.0 # Default to 0 for all but the last entry
            })
            previous_fop = segment['fop']
            
        # Add the final flight back time to the last entry for V-Shaped
        if report_data:
            fly_back_time = np.linalg.norm(data_center_pos - previous_fop) / uav_speed
            # Find the index of the last V-Shaped entry in the report_data list
            # It will be the last one we added.
            report_data[-1]['Fly_Back_Time (s)'] = fly_back_time

    # --- 2. Process Convex Trajectory Data ---
    if convex_result and convex_result.get('collection_segments'):
        collection_segments_c = convex_result['collection_segments']
        previous_eo = data_center_pos
        
        for i, segment in enumerate(collection_segments_c):
            so_point, eo_point = segment['start'], segment['end']
            
            fly_in_dist = np.linalg.norm(so_point - previous_eo)
            fly_in_time = fly_in_dist / uav_speed
            
            collection_dist = np.linalg.norm(eo_point - so_point)
            collection_time = segment.get('Total_Collection_Time (s)', 0)

            report_data.append({
                'Method': 'Convex',
                'Sequence': i + 1,
                'GN_Index': segment['gn_index'],
                'Fly_IN_Time (s)': fly_in_time,
                'Fly_IN_Distance (m)': fly_in_dist,
                'Collection_Time (s)': collection_time,
                'Collection_Distance (m)': collection_dist,
                'Fly_Back_Time (s)': 0.0 # Default to 0
            })
            previous_eo = segment['end']
            
        # Add the final flight back time to the last entry for Convex
        if report_data:
            fly_back_time = np.linalg.norm(data_center_pos - previous_eo) / uav_speed
            # Find the index of the last Convex entry
            last_convex_index = -1 # Find the last index with method 'Convex'
            for idx in range(len(report_data) - 1, -1, -1):
                if report_data[idx]['Method'] == 'Convex':
                    last_convex_index = idx
                    break
            if last_convex_index != -1:
                report_data[last_convex_index]['Fly_Back_Time (s)'] = fly_back_time

    # --- 3. Generate and Save Final Report ---
    if not report_data:
        print("No data to generate report.")
        return

    df = pd.DataFrame(report_data)
    df = df.sort_values(by=['Method', 'Sequence'])
    
    # Reorder columns to match the requested format
    cols = ['Method', 'Sequence', 'GN_Index', 
            'Fly_IN_Time (s)', 'Fly_IN_Distance (m)', 
            'Collection_Time (s)', 'Collection_Distance (m)', 
            'Fly_Back_Time (s)']
    df_final = df[cols]

    report_path = os.path.join(output_dir, f'{run_prefix}_{uav_id}_flight_log_report.csv')
    df_final.to_csv(report_path, index=False, float_format='%.2f')
    print(f"\nDetailed flight log report saved to: {os.path.basename(report_path)}")