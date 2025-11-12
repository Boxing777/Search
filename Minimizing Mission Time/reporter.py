# reporter.py (Version 8: Corrected Fly-Back Time Logic)
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
            
            is_overlapping = (fly_in_dist < 1e-6)
            # service_time from main.py is correct, but we need to extract collection_time for the report.
            # In non-overlapping, service_time = flight_in + collection.
            # In overlapping, service_time = collection.
            # So, collection_time can be derived from service_time and the calculated flight_in_time.
            calculated_flight_in_time_from_service = segment['service_time'] - segment['collection_time'] if 'collection_time' in segment and not is_overlapping else 0
            collection_time = segment.get('collection_time', segment['service_time'] - calculated_flight_in_time_from_service)

            
            report_data.append({
                'Method': 'V-Shaped',
                'Sequence': i + 1,
                'GN_Index': segment.get('gn_index', 'N/A'),
                'Fly_IN_Time (s)': fly_in_time,
                'Fly_IN_Distance (m)': fly_in_dist,
                'Collection_Time (s)': collection_time,
                'Collection_Distance (m)': collection_dist,
                'Fly_Back_Time (s)': 0.0,
                'Fly_Back_Distance (m)': 0.0 
            })
            previous_fop = segment['fop']
            
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
                'Fly_Back_Time (s)': 0.0,
                'Fly_Back_Distance (m)': 0.0 
            })
            previous_eo = segment['end']

    # --- 3. Generate and Save Final Report ---
    if not report_data:
        print("No data to generate report.")
        return

    df = pd.DataFrame(report_data)
    
    # <<< MODIFICATION START: Correct and robust calculation of Fly_Back_Time >>>
    # Original flawed logic is commented out below for reference
    # # if report_data:
    # #     fly_back_time = np.linalg.norm(data_center_pos - previous_fop) / uav_speed
    # #     report_data[-1]['Fly_Back_Time (s)'] = fly_back_time
    # # ... (and the complex index search for Convex) ...

    # New robust logic:
    if 'V-Shaped' in df['Method'].values:
        # Find the index of the row with the highest 'Sequence' number for V-Shaped
        last_v_shaped_row_index = df.loc[df['Method'] == 'V-Shaped', 'Sequence'].idxmax()
        
        # Get the corresponding FOP from the original data structure
        last_gn_sequence_index = df.loc[last_v_shaped_row_index, 'Sequence'] - 1
        last_collection_segment_v = [s for s in v_shaped_segments if s['type'] == 'collection'][last_gn_sequence_index]
        last_fop_v = last_collection_segment_v['fop']
        
        fly_back_dist_v = np.linalg.norm(data_center_pos - last_fop_v)
        fly_back_time_v = fly_back_dist_v / uav_speed
        df.loc[last_v_shaped_row_index, 'Fly_Back_Time (s)'] = fly_back_time_v
        df.loc[last_v_shaped_row_index, 'Fly_Back_Distance (m)'] = fly_back_dist_v

    if 'Convex' in df['Method'].values:
        # Find the index of the row with the highest 'Sequence' number for Convex
        last_convex_row_index = df.loc[df['Method'] == 'Convex', 'Sequence'].idxmax()

        # Get the corresponding Eo from the original data structure
        last_gn_sequence_index_c = df.loc[last_convex_row_index, 'Sequence'] - 1
        last_collection_segment_c = convex_result['collection_segments'][last_gn_sequence_index_c]
        last_eo_c = last_collection_segment_c['end']

        fly_back_dist_c = np.linalg.norm(data_center_pos - last_eo_c)
        fly_back_time_c = fly_back_dist_c / uav_speed
        df.loc[last_convex_row_index, 'Fly_Back_Time (s)'] = fly_back_time_c
        df.loc[last_convex_row_index, 'Fly_Back_Distance (m)'] = fly_back_dist_c

    df = df.sort_values(by=['Method', 'Sequence'])
    
    # Reorder columns to match the requested format
    cols = ['Method', 'Sequence', 'GN_Index', 
            'Fly_IN_Time (s)', 'Fly_IN_Distance (m)', 
            'Collection_Time (s)', 'Collection_Distance (m)', 
            'Fly_Back_Time (s)', 'Fly_Back_Distance (m)']
    df_final = df.reindex(columns=cols) # Use reindex to handle missing columns gracefully

    report_path = os.path.join(output_dir, f'{run_prefix}_{uav_id}_flight_log_report.csv')
    df_final.to_csv(report_path, index=False, float_format='%.2f')
    print(f"\nDetailed flight log report saved to: {os.path.basename(report_path)}")
    