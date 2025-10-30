# reporter.py (Robust Version)
# ==============================================================================
#                      Simulation Report Generator
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
    **kwargs # Accept extra arguments like total times
) -> None:
    """
    Generates a detailed CSV report comparing the time breakdown of V-Shaped
    and Convex path planning methods for a single UAV.
    """
    v_shaped_report_data = []
    convex_report_data = []

    # --- 1. Process V-Shaped Trajectory Data ---
    collection_segments_v = [s for s in v_shaped_segments if s['type'] == 'collection']
    for i, segment in enumerate(collection_segments_v):
        sequence_num = i + 1
        gn_index = segment.get('gn_index', 'N/A')
        
        # Find the preceding flight segment to get flight_in_time
        # The full list index for this collection segment is i*2 + 1
        flight_segment_index = i * 2
        flight_segment = v_shaped_segments[flight_segment_index]
        flight_in_time = np.linalg.norm(flight_segment['end'] - flight_segment['start']) / uav_speed
        
        is_overlapping = (flight_in_time < 1e-6)
        collection_time = segment['service_time'] if is_overlapping else (segment['service_time'] - flight_in_time)

        v_shaped_report_data.append({
            'Method': 'V-Shaped',
            'Sequence': sequence_num,
            'GN_Index': gn_index,
            'Flight_In_Time (s)': flight_in_time,
            'Collection_Time (s)': collection_time,
        })
            
    # --- 2. Process Convex Trajectory Data ---
    previous_eo = data_center_pos
    for i, segment in enumerate(convex_result.get('collection_segments', [])):
        so_point = segment['start']
        flight_in_time = np.linalg.norm(so_point - previous_eo) / uav_speed
        collection_time = segment.get('Total_Collection_Time (s)', 0)
        convex_report_data.append({
            'Method': 'Convex',
            'Sequence': i + 1,
            'GN_Index': segment['gn_index'],
            'Flight_In_Time (s)': flight_in_time,
            'Collection_Time (s)': collection_time,
        })
        previous_eo = segment['end']

    # --- 3. Calculate Flight-Out Times (Robust Method) ---
    if v_shaped_report_data:
        for i in range(len(v_shaped_report_data)):
            current_collection_segment = collection_segments_v[i]
            current_fop = current_collection_segment['fop']
            
            if i < len(v_shaped_report_data) - 1:
                # Flight-out is to the next collection segment's FIP
                # The next collection segment is at index (i+1)*2 + 1
                # The flight segment before it is at (i+1)*2
                next_flight_segment = v_shaped_segments[(i + 1) * 2]
                next_fip = next_flight_segment['end']
                flight_out_time = np.linalg.norm(next_fip - current_fop) / uav_speed
            else: 
                # Last GN, flight-out is to the data center
                flight_out_time = np.linalg.norm(data_center_pos - current_fop) / uav_speed
            
            v_shaped_report_data[i]['Flight_Out_Time (s)'] = flight_out_time

    if convex_report_data:
        collection_segments_c = convex_result.get('collection_segments', [])
        for i in range(len(convex_report_data)):
            current_eo = collection_segments_c[i]['end']
            if i < len(convex_report_data) - 1:
                next_so = collection_segments_c[i+1]['start']
                flight_out_time = np.linalg.norm(next_so - current_eo) / uav_speed
            else:
                flight_out_time = np.linalg.norm(data_center_pos - current_eo) / uav_speed
            convex_report_data[i]['Flight_Out_Time (s)'] = flight_out_time

    # --- 4. Combine, format, and save the report ---
    if v_shaped_report_data or convex_report_data:
        df_v = pd.DataFrame(v_shaped_report_data)
        df_c = pd.DataFrame(convex_report_data)
        
        # Add a summary row
        if not df_v.empty:
            summary_v = pd.DataFrame([{'Method': 'V-Shaped', 'Sequence': 'TOTAL', 'GN_Index': '-', 'Flight_In_Time (s)': df_v['Flight_In_Time (s)'].sum(), 'Collection_Time (s)': df_v['Collection_Time (s)'].sum(), 'Flight_Out_Time (s)': df_v['Flight_Out_Time (s)'].sum()}])
            df_v = pd.concat([df_v, summary_v], ignore_index=True)
        if not df_c.empty:
            summary_c = pd.DataFrame([{'Method': 'Convex', 'Sequence': 'TOTAL', 'GN_Index': '-', 'Flight_In_Time (s)': df_c['Flight_In_Time (s)'].sum(), 'Collection_Time (s)': df_c['Collection_Time (s)'].sum(), 'Flight_Out_Time (s)': df_c['Flight_Out_Time (s)'].sum()}])
            df_c = pd.concat([df_c, summary_c], ignore_index=True)

        df_combined = pd.concat([df_v, df_c]).sort_values(by=['Sequence', 'Method'], key=lambda x: pd.to_numeric(x, errors='coerce').fillna(float('inf')))
        
        cols = ['Method', 'Sequence', 'GN_Index', 'Flight_In_Time (s)', 'Collection_Time (s)', 'Flight_Out_Time (s)']
        df_combined = df_combined.reindex(columns=cols)
        
        report_path = os.path.join(output_dir, f'{run_prefix}_{uav_id}_time_breakdown_report.csv')
        df_combined.to_csv(report_path, index=False, float_format='%.2f')
        print(f"\nDetailed time breakdown report saved to: {os.path.basename(report_path)}")