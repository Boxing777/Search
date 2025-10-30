print("  -> Calculating actual mission time for Convex Path...")
        
        convex_total_hover_time = 0.0
        # Iterate through the path segments that are inside communication zones
        for segment in convex_result['collection_segments']:
            gn_coord = sim_env.gn_positions[segment['gn_index']]
            
            # Calculate how much data can be collected just by flying through
            data_collected_on_segment = traj_optimizer._calculate_collected_data(
                segment['start'], segment['end'], gn_coord
            )
            
            # Determine if hovering is needed
            data_shortfall = required_data_per_gn - data_collected_on_segment
            
            collection_flight_time = np.linalg.norm(segment['end'] - segment['start']) / params.UAV_MAX_SPEED
            hover_time = 0.0

            if data_shortfall > 0:
                # hover_time = data_shortfall / traj_optimizer.hover_datarate if traj_optimizer.hover_datarate > 0 else float('inf')
                # convex_total_hover_time += hover_time
                exit_point_Eo = segment['end']
                rate_at_Eo = traj_optimizer.calculate_hover_rate_at_point(exit_point_Eo, gn_coord)
                hover_time = data_shortfall / rate_at_Eo if rate_at_Eo > 1e-6 else float('inf')
                
            convex_total_hover_time += hover_time
            
            # Add detailed time to the segment dict for the reporter module to use later
            segment['Total_Collection_Time (s)'] = collection_flight_time + hover_time
        
        # The total fair time is the flight time plus any required hover time
        convex_flight_time = convex_result['length'] / params.UAV_MAX_SPEED
        convex_actual_mission_time = convex_flight_time + convex_total_hover_time
        convex_mission_times[uav_id] = convex_actual_mission_time
        
        print(f"     Convex Path -> Flight Time: {convex_flight_time:.2f}s, Required Hover Time: {convex_total_hover_time:.2f}s, TOTAL FAIR TIME: {convex_actual_mission_time:.2f}s")