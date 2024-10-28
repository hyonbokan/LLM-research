import os
import pandas as pd
import ast
from collections import defaultdict, Counter
import logging

numeric_columns = [
    'Total Routes', 'New Routes', 'Withdrawals', 'Origin Changes', 'Route Changes',
    'Maximum Path Length', 'Average Path Length', 'Maximum Edit Distance', 'Average Edit Distance',
    'Announcements', 'Unique Prefixes Announced', 'Average MED', 'Average Local Preference',
    'Total Communities', 'Unique Communities', 'Total Updates', 'Average Updates per Peer',
    'Max Updates from a Single Peer', 'Min Updates from a Single Peer', 'Std Dev of Updates',
    'Total Prefixes Announced', 'Average Announcements per Prefix',
    'Max Announcements for a Single Prefix', 'Min Announcements for a Single Prefix',
    'Std Dev of Announcements', 'Count of Unexpected ASNs in Paths',
    'Target Prefixes Withdrawn', 'Target Prefixes Announced', 'AS Path Changes',
    'Average Prefix Length', 'Max Prefix Length', 'Min Prefix Length',
    'AS Path Prepending'
]

def process_list_column(df, column_list, default_value):
    for col in column_list:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(
                lambda x: [item.strip() for item in x.strip("[]").replace("'", "").split(',') if item.strip()]
            )
        else:
            logging.warning(f"Column '{col}' not found in the DataFrame. Filling with default.")
            df[col] = [default_value for _ in range(len(df))]

def generate_overall_summary(df, summary_metrics, total_updates_per_peer, top_n_peers):
    overall_summary_text = "Overall Summary:\n\n"
    # Key statistics
    overall_summary_text += (
        f"During the observation period, Autonomous Systems reported various BGP metrics. "
        f"The total number of routes ranged from {int(summary_metrics['Total Routes']['min'])} to {int(summary_metrics['Total Routes']['max'])}, "
        f"with an average of {summary_metrics['Total Routes']['average']:.2f} routes. "
        f"Announcements varied between {int(summary_metrics['Announcements']['min'])} and {int(summary_metrics['Announcements']['max'])}, "
        f"averaging {summary_metrics['Announcements']['average']:.2f} announcements. "
        f"Withdrawals varied between {int(summary_metrics['Withdrawals']['min'])} and {int(summary_metrics['Withdrawals']['max'])}, "
        f"averaging {summary_metrics['Withdrawals']['average']:.2f} withdrawals. "
        f"The maximum path length observed was {int(summary_metrics['Maximum Path Length']['max'])} hops, "
        f"with an average path length of {summary_metrics['Average Path Length']['average']:.2f} hops. "
        f"Communities ranged from {int(summary_metrics['Total Communities']['min'])} to {int(summary_metrics['Total Communities']['max'])}, "
        f"with an average of {summary_metrics['Total Communities']['average']:.2f}. "
        f"The system observed an average prefix length of {summary_metrics['Average Prefix Length']['average']:.1f}, "
        f"with a maximum of {int(summary_metrics['Max Prefix Length']['max'])} and a minimum of {int(summary_metrics['Min Prefix Length']['min'])}."
    )
    overall_summary_text += "\n\n"

    # Top peers
    overall_summary_text += f"Top {top_n_peers} Peers with the Highest Number of Updates:\n"
    sorted_peers = sorted(total_updates_per_peer.items(), key=lambda x: x[1], reverse=True)[:top_n_peers]
    if sorted_peers:
        peer_details = ', '.join([f"AS{asn} ({updates:.2f} updates)" for asn, updates in sorted_peers])
        overall_summary_text += (
            f"The top {top_n_peers} peers contributing the most updates are {peer_details}.\n"
        )
    else:
        overall_summary_text += "No peer updates data available.\n"

    overall_summary_text += "\n\n"

    # Policy-Related Metrics Summary
    overall_summary_text += "Policy-Related Metrics Summary:\n\n"

    # Local Preference Summary
    overall_summary_text += (
        f"Local Preference values ranged from {summary_metrics['Average Local Preference']['min']:.2f} to {summary_metrics['Average Local Preference']['max']:.2f}, "
        f"with an average of {summary_metrics['Average Local Preference']['average']:.2f}.\n"
    )

    # MED Summary
    overall_summary_text += (
        f"MED values ranged from {summary_metrics['Average MED']['min']:.2f} to {summary_metrics['Average MED']['max']:.2f}, "
        f"with an average of {summary_metrics['Average MED']['average']:.2f}.\n"
    )

    # AS Path Prepending Summary
    total_prepending = df['AS Path Prepending'].sum()
    overall_summary_text += (
        f"AS Path Prepending was observed {int(total_prepending)} times during the observation period.\n"
    )

    overall_summary_text += "\n"

    return overall_summary_text


def generate_data_point_log(row, timestamp, as_number):
    log_entry = (
        f"On {timestamp}, Autonomous System {as_number} observed {int(row['Announcements'])} announcements. "
        f"There were {int(row['New Routes'])} new routes added. The total number of active routes was {int(row['Total Routes'])}. "
        f"The maximum path length observed was {int(row['Maximum Path Length'])} hops, with an average path length of {row['Average Path Length']:.2f} hops. "
        f"The maximum edit distance was {int(row['Maximum Edit Distance'])}, with an average of {row['Average Edit Distance']:.1f}. "
        f"The average MED was {row['Average MED']:.2f}. The average local preference was {row['Average Local Preference']:.2f}. "
        f"There were {int(row['Total Communities'])} total communities observed, with {int(row['Unique Communities'])} unique communities. "
        f"The number of unique peers was {int(row['Average Updates per Peer'])}, with an average of {row['Average Updates per Peer']:.2f} updates per peer. "
        f"The average prefix length was {row['Average Prefix Length']:.1f}, with a maximum of {int(row['Max Prefix Length'])} and a minimum of {int(row['Min Prefix Length'])}. "
        f"Number of prefixes announced: {int(row['Total Prefixes Announced'])}. Number of prefixes withdrawn: {int(row['Target Prefixes Withdrawn'])}."
    )
    return log_entry

def collect_prefix_events(row, idx, timestamp, as_number, prefix_announcements, prefix_withdrawals):
    # Announcements
    if row['Total Prefixes Announced'] > 0:
        prefixes_announced_str = row.get('Target Prefixes Announced', '0')
        prefixes_announced_list = parse_prefix_list(prefixes_announced_str, idx, 'Target Prefixes Announced')
        for prefix in prefixes_announced_list:
            announcement = (
                f"At {timestamp}, AS{as_number} announced the prefix: {prefix}. "
                f"Total prefixes announced: {int(row['Total Prefixes Announced'])}."
            )
            prefix_announcements.append(announcement)

    # Withdrawals
    if row['Target Prefixes Withdrawn'] > 0:
        prefixes_withdrawn_str = row.get('Target Prefixes Withdrawn', '0')
        prefixes_withdrawn_list = parse_prefix_list(prefixes_withdrawn_str, idx, 'Target Prefixes Withdrawn')
        for prefix in prefixes_withdrawn_list:
            withdrawal = (
                f"At {timestamp}, AS{as_number} withdrew the prefix: {prefix}. "
                f"Total prefixes withdrawn: {int(row['Target Prefixes Withdrawn'])}."
            )
            prefix_withdrawals.append(withdrawal)

def parse_prefix_list(prefixes_str, idx, column_name):
    prefixes_list = []
    if isinstance(prefixes_str, str) and prefixes_str.strip() != '0':
        try:
            parsed_list = ast.literal_eval(prefixes_str)
            if isinstance(parsed_list, list):
                for prefix in parsed_list:
                    if isinstance(prefix, list):
                        if len(prefix) == 1 and isinstance(prefix[0], str):
                            prefix = prefix[0]
                        else:
                            logging.warning(f"Found a nested list in '{column_name}' at index {idx}. Skipping.")
                            continue
                    if isinstance(prefix, str) and prefix.strip() != '0':
                        prefixes_list.append(prefix.strip())
        except (SyntaxError, ValueError):
            logging.warning(f"Could not parse '{column_name}' at index {idx}. Skipping.")
    return prefixes_list

def generate_updates_per_peer_info(row, timestamp, peers_nested):
    peers = [asn for peer_list in peers_nested for asn in peer_list]
    if peers:
        updates_info = f"At {timestamp}, updates per peer were as follows:"
        for peer_asn in peers:
            if isinstance(peer_asn, str) and peer_asn.isdigit():
                updates = row['Average Updates per Peer']
                updates_info += f" Peer AS{peer_asn} received {updates:.2f} updates."
    else:
        updates_info = f"At {timestamp}, no peer updates were observed."
    return updates_info

def detect_anomalies(row, anomaly_rules):
    anomalies_detected = []
    for rule in anomaly_rules:
        feature = rule['feature']
        threshold = rule['threshold']
        condition = rule['condition']
        anomaly_type = rule.get('type', 'Anomaly')
        description = rule.get('description', '')

        feature_value = row.get(feature, 0)

        if callable(condition):
            is_anomaly = condition(feature_value, threshold)
        elif condition == '>':
            is_anomaly = feature_value > threshold
        elif condition == '<':
            is_anomaly = feature_value < threshold
        else:
            raise ValueError(f"Unsupported condition '{condition}' in anomaly rule for feature '{feature}'")

        if is_anomaly:
            anomalies_detected.append({
                'type': anomaly_type,
                'feature': feature,
                'description': description,
                'value': feature_value,
                'threshold': threshold
            })

    return anomalies_detected

def generate_anomaly_log(anomaly):
    anomaly_log = (
        f"On {anomaly['timestamp']}, Autonomous System {anomaly['as_number']} experienced an anomaly:\n"
        f"  - {anomaly['type']} detected ({anomaly['description']})\n"
        f"    Feature: {anomaly['feature']}\n"
        f"    Value: {anomaly['value']}\n"
        f"    Threshold: {anomaly['threshold']}\n"
        "\n"
    )
    return anomaly_log

def generate_and_write_anomaly_summary(anomalies, output_dir, filename):
    anomaly_summary = "Anomaly Summary:\n\n"
    total_anomalies = len(anomalies)
    if total_anomalies > 0:
        anomaly_summary += f"A total of {total_anomalies} anomalies were detected during the observation period.\n\n"

        # Group anomalies by type and feature
        from collections import defaultdict

        anomaly_counts = defaultdict(int)
        anomaly_details = defaultdict(list)

        for anomaly in anomalies:
            key = f"{anomaly['type']} - {anomaly['feature']}"
            anomaly_counts[key] += 1
            detail = (
                f"- At {anomaly['timestamp']}, AS{anomaly['as_number']} had a value of {anomaly['value']} "
                f"for {anomaly['feature']} (Threshold: {anomaly['threshold']})"
            )
            anomaly_details[key].append(detail)

        # Create summary for each anomaly type and feature
        for key, count in anomaly_counts.items():
            anomaly_summary += f"{key}: {count} occurrences\n"
            # Include brief details (e.g., first few occurrences)
            details = anomaly_details[key][:5]  # Show up to first 5 details
            for detail in details:
                anomaly_summary += f"  {detail}\n"
            if count > 5:
                anomaly_summary += f"  ...and {count - 5} more occurrences.\n"
            anomaly_summary += "\n"
    else:
        anomaly_summary += "No anomalies were detected during the observation period.\n"

    # Write anomaly summary to file
    with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
        f.write(anomaly_summary)

    # Write detailed anomaly logs to 'anomalies.txt'
    anomalies_filename = 'anomalies.txt'
    with open(os.path.join(output_dir, anomalies_filename), 'w', encoding='utf-8') as f:
        for anomaly in anomalies:
            anomaly_log = generate_anomaly_log(anomaly)
            f.write(anomaly_log)

def collect_policy_info(row):
    policy_info = []
    # Local Preference
    avg_local_pref = row.get('Average Local Preference', None)
    if avg_local_pref is not None:
        policy_info.append(f"Average Local Preference: {avg_local_pref:.2f}")
    # MED
    avg_med = row.get('Average MED', None)
    if avg_med is not None:
        policy_info.append(f"Average MED: {avg_med:.2f}")
    # AS Path Length
    max_path_length = row.get('Maximum Path Length', None)
    if max_path_length is not None:
        policy_info.append(f"Maximum AS Path Length: {int(max_path_length)}")
    # AS Path Prepending
    as_path_prepending = row.get('AS Path Prepending', 0)
    if as_path_prepending > 0:
        policy_info.append(f"AS Path Prepending observed {int(as_path_prepending)} times")
    # Community Values
    community_values = row.get('Community Values', [])
    if community_values:
        policy_info.append(f"Community Values: {', '.join(community_values)}")
    return policy_info

def write_logs_to_file(logs, output_dir, filename):
    with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
        for log in logs:
            f.write(log)
            
            
def generate_policy_log(policy_info, timestamp, as_number):
    policy_log = f"On {timestamp}, Autonomous System {as_number} had the following policy-related observations:\n"
    for info in policy_info:
        policy_log += f"  - {info}\n"
    policy_log += "\n"
    return policy_log


def generate_policy_summary(df, summary_metrics):
    policy_summary = "\nPolicy Summary:\n\n"
    # Local Preference Summary
    policy_summary += (
        f"Local Preference values ranged from {summary_metrics['Average Local Preference']['min']:.2f} to "
        f"{summary_metrics['Average Local Preference']['max']:.2f}, with an average of "
        f"{summary_metrics['Average Local Preference']['average']:.2f}.\n"
    )
    # MED Summary
    policy_summary += (
        f"MED values ranged from {summary_metrics['Average MED']['min']:.2f} to "
        f"{summary_metrics['Average MED']['max']:.2f}, with an average of "
        f"{summary_metrics['Average MED']['average']:.2f}.\n"
    )
    # AS Path Prepending Summary
    total_prepending = df['AS Path Prepending'].sum()
    policy_summary += f"AS Path Prepending was observed {int(total_prepending)} times during the observation period.\n"
    # Community Values Summary
    all_communities = set()
    for communities in df['Community Values']:
        all_communities.update(communities)
    policy_summary += f"A total of {len(all_communities)} unique community values were observed.\n"
    return policy_summary


def process_bgp(
    df,
    output_dir='processed_output',
    overall_summary_filename='overall_summary.txt',
    data_point_logs_filename='data_point_logs.txt',
    prefix_announcements_filename='prefix_announcements.txt',
    prefix_withdrawals_filename='prefix_withdrawals.txt',
    updates_per_peer_filename='updates_per_peer.txt',
    anomaly_summary_filename='anomaly_summary.txt',
    anomalies_filename='anomalies.txt',
    policy_summary_filename='policy_summary.txt',
    top_n_peers=10,       # Number of top peers to include in the overall summary
    percentile=95        # Percentile for high anomalies
):
    global numeric_columns
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, 'process_bgp.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s'
    ) 

    # Convert numeric columns to float and fill missing columns
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            logging.warning(f"Column '{col}' not found in the DataFrame. Filling with 0.0.")
            df[col] = 0.0

    # Handle list columns (Top Peers and Top Prefixes)
    peer_columns = [f'Top Peer {i} ASN' for i in range(1, 6)]
    prefix_columns = [f'Top Prefix {i}' for i in range(1, 6)]

    # Process list columns
    process_list_column(df, peer_columns, [])
    process_list_column(df, prefix_columns, '')

    # Process 'Community Values' column
    if 'Community Values' in df.columns:
        df['Community Values'] = df['Community Values'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
        )
    else:
        logging.warning("Column 'Community Values' not found in the DataFrame. Filling with empty lists.")
        df['Community Values'] = [[] for _ in range(len(df))]

    # Initialize summary metrics
    summary_metrics = {
        col: {
            'min': df[col].min(),
            'max': df[col].max(),
            'average': df[col].mean(),
            'std_dev': df[col].std()
        }
        for col in numeric_columns
    }

    # Calculate percentile-based thresholds
    anomaly_thresholds = {}
    for col in numeric_columns:
        if col == 'Total Routes':
            anomaly_thresholds[col] = df[col].quantile(0.05)  # Lower 5th percentile for low values
        else:
            anomaly_thresholds[col] = df[col].quantile(percentile / 100)

    # Define anomaly rules
    anomaly_rules = []
    for feature, threshold in anomaly_thresholds.items():
        if feature == 'Total Routes':
            anomaly_rules.append({
                'feature': feature,
                'threshold': threshold,
                'condition': '<',
                'type': 'Low Value Anomaly',
                'description': f"Low value detected for '{feature}'."
            })
        else:
            anomaly_rules.append({
                'feature': feature,
                'threshold': threshold,
                'condition': '>',
                'type': 'High Value Anomaly',
                'description': f"High value detected for '{feature}'."
            })


    # Calculate total updates per peer
    total_updates_per_peer = defaultdict(float)
    for idx, row in df.iterrows():
        peers_nested = [row[col] for col in peer_columns]
        peers = [asn for peer_list in peers_nested for asn in peer_list]
        for peer_asn in peers:
            if isinstance(peer_asn, str) and peer_asn.isdigit():
                total_updates_per_peer[peer_asn] += row['Average Updates per Peer']

    # Generate Overall Summary
    overall_summary_text = generate_overall_summary(df, summary_metrics, total_updates_per_peer, top_n_peers)

    # Write Overall Summary to File
    with open(os.path.join(output_dir, overall_summary_filename), 'w', encoding='utf-8') as f:
        f.write(overall_summary_text)

    # Initialize logs and lists
    data_point_logs = []
    anomalies = []
    prefix_announcements = []
    prefix_withdrawals = []
    updates_per_peer_info = []
    policy_logs = []

    # Process each data point
    for idx, row in df.iterrows():
        timestamp = row.get('Timestamp', 'N/A')
        as_number = row.get('Autonomous System Number', 'N/A')

        # Generate data point log
        log_entry = generate_data_point_log(row, timestamp, as_number)
        data_point_logs.append(log_entry + "\n")

        # Collect prefix announcements and withdrawals
        collect_prefix_events(row, idx, timestamp, as_number, prefix_announcements, prefix_withdrawals)

        # Collect updates per peer information
        peers_nested = [row[col] for col in peer_columns]
        updates_info = generate_updates_per_peer_info(row, timestamp, peers_nested)
        updates_per_peer_info.append(updates_info + "\n")

        # Detect and store anomalies
        anomalies_detected = detect_anomalies(row, anomaly_rules)
        if anomalies_detected:
            for anomaly in anomalies_detected:
                anomaly_entry = {
                    'timestamp': timestamp,
                    'as_number': as_number,
                    'type': anomaly['type'],
                    'feature': anomaly['feature'],
                    'description': anomaly['description'],
                    'value': anomaly['value'],
                    'threshold': anomaly['threshold']
                }
                anomalies.append(anomaly_entry)

        # Collect policy-related information
        policy_info = collect_policy_info(row)
        if policy_info:
            policy_log = generate_policy_log(policy_info, timestamp, as_number)
            policy_logs.append(policy_log)

    # Write logs to files
    write_logs_to_file(data_point_logs, output_dir, data_point_logs_filename)
    write_logs_to_file(prefix_announcements, output_dir, prefix_announcements_filename)
    write_logs_to_file(prefix_withdrawals, output_dir, prefix_withdrawals_filename)
    write_logs_to_file(updates_per_peer_info, output_dir, updates_per_peer_filename)
    
    # Write detailed anomaly logs
    anomalies_filename = 'anomalies.txt'
    with open(os.path.join(output_dir, anomalies_filename), 'w', encoding='utf-8') as f:
        for anomaly in anomalies:
            anomaly_log = generate_anomaly_log(anomaly)
            f.write(anomaly_log)
            
    write_logs_to_file(policy_logs, output_dir, policy_summary_filename)

    # Generate and write Anomaly Summary
    generate_and_write_anomaly_summary(anomalies, output_dir, anomaly_summary_filename)

    # Generate and write Policy Summary
    policy_summary_text = generate_policy_summary(df, summary_metrics)
    with open(os.path.join(output_dir, policy_summary_filename), 'a', encoding='utf-8') as f:
        f.write(policy_summary_text)

    print(f"Data processing complete. Output files are saved in the '{output_dir}' directory.")