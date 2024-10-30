import os
import pandas as pd
import ast
from collections import defaultdict, Counter
import logging
import ast
from helper_functions import (
    generate_and_write_anomaly_summary,
    generate_data_point_log,
    collect_prefix_events,
    collect_policy_info,
    generate_policy_log,
    write_logs_to_file,
    generate_policy_summary,
    generate_as_path_changes_summary,
    generate_anomaly_log,
    generate_updates_per_peer_info,
    generate_updates_per_prefix_info,
    detect_anomalies,
    generate_overall_summary,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

anomaly_rules_config = [
    # Leak Anomaly
    {
        'type': 'Leak Anomaly',
        'description': "Excessive route withdrawals or unexpected route advertisements indicating possible route leaks.",
        'conditions': [
            {'feature': 'Withdrawals', 'operator': '>', 'threshold': None},
            {'feature': 'Route Changes', 'operator': '>', 'threshold': None}
        ]
    },
    # Flapping Anomaly
    {
        'type': 'Flapping Anomaly',
        'description': "Frequent changes in route announcements and withdrawals, indicating network instability.",
        'conditions': [
            {'feature': 'Std Dev of Updates', 'operator': '>', 'threshold': None},
            {'feature': 'Announcements', 'operator': '>', 'threshold': None}
        ]
    },
    # Hijack Anomaly
    {
        'type': 'Hijack Anomaly',
        'description': "Unauthorized ASNs appearing in the AS path, indicating potential route hijacking.",
        'conditions': [
            {'feature': 'Count of Unexpected ASNs in Paths', 'operator': '>', 'threshold': None}
        ]
    },
    # Path Manipulation Anomaly
    {
        'type': 'Path Manipulation Anomaly',
        'description': "Unusual modifications to the AS path, such as excessive AS path prepending.",
        'conditions': [
            {'feature': 'AS Path Prepending', 'operator': '>', 'threshold': None},
            {'feature': 'Maximum Path Length', 'operator': '>', 'threshold': None}
        ]
    },
    # Policy-Related Anomaly
    {
        'type': 'Policy-Related Anomaly',
        'description': "Deviations from defined routing policies, such as unexpected MED or Local Preference values.",
        'conditions': [
            {'feature': 'Average MED', 'operator': '>', 'threshold': None},
            {'feature': 'Average Local Preference', 'operator': '<', 'threshold': None}
        ]
    },
    # Add more anomaly categories and their conditions as needed
]

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
    'AS Path Prepending',
    'Total Peers', 'Total Paths', 'Total Prefixes Announced List', 'Total Prefixes Withdrawn List'
]
features_numeric = {
    "Total Routes",
    "New Routes",
    "Withdrawals",
    "Origin Changes",
    "Route Changes",
    "Maximum Path Length",
    "Average Path Length",
    "Maximum Edit Distance",
    "Average Edit Distance",
    "Announcements",
    "Unique Prefixes Announced",
    "Average MED",
    "Average Local Preference",
    "Total Communities",
    "Unique Communities",
    "Total Updates",
    "Average Updates per Peer",
    "Max Updates from a Single Peer",
    "Min Updates from a Single Peer",
    "Std Dev of Updates",
    "Total Prefixes Announced",
    "Average Announcements per Prefix",
    "Max Announcements for a Single Prefix",
    "Min Announcements for a Single Prefix",
    "Std Dev of Announcements",
    "Count of Unexpected ASNs in Paths",
    "Target Prefixes Withdrawn",
    "Target Prefixes Announced",
    "AS Path Changes",
    "Average Prefix Length",
    "Max Prefix Length",
    "Min Prefix Length",
    "AS Path Prepending",
    "Total Peers",
    "Total Paths",
    "Total Prefixes Announced List",
    "Total Prefixes Withdrawn List"
}

# Current numeric_columns set
numeric_columns_set = set(numeric_columns)

# Identify missing columns
missing_in_numeric = features_numeric - numeric_columns_set
extra_in_numeric = numeric_columns_set - features_numeric

if missing_in_numeric:
    logger.warning(f"The following numeric features are missing in 'numeric_columns': {missing_in_numeric}")
    # Optionally, you can automate adding them
    # numeric_columns.extend(list(missing_in_numeric))
if extra_in_numeric:
    logger.warning(f"The following columns in 'numeric_columns' are not defined in 'features': {extra_in_numeric}")
    # Optionally, you can remove them
    # numeric_columns = [col for col in numeric_columns if col not in extra_in_numeric]
if not missing_in_numeric and not extra_in_numeric:
    logger.info("All numeric features in 'features' are correctly included in 'numeric_columns'.")
    

def safe_parse_list(x, column_name, idx):
    try:
        parsed = ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
        # Ensure all elements are strings
        return [str(item).strip() for item in parsed]
    except (ValueError, SyntaxError):
        logging.warning(f"Failed to parse list from '{column_name}' at row {idx}. Filling with empty list.")
        return []

def process_bgp(
    df,
    output_dir='processed_output',
    overall_summary_filename='overall_summary.txt',
    data_point_logs_filename='data_point_logs.txt',
    prefix_announcements_filename='target_prefix_announcements.txt',
    prefix_withdrawals_filename='target_prefix_withdrawals.txt',
    updates_per_peer_filename='peer_asn.txt',
    updates_per_prefix_filename='prefixes.txt',
    anomaly_summary_filename='anomaly_summary.txt',
    as_path_changes_summary_filename='as_path_changes_summary.txt',
    anomalies_filename='anomalies.txt',
    policy_summary_filename='policy_summary.txt',
    percentile=90
):
    global numeric_columns
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, 'process_bgp.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s:%(message)s'
    )
    logger = logging.getLogger(__name__)

    # Define a list of columns to parse as lists
    list_columns = ['All Peers', 'All Paths', 'Community Values', 'All Prefixes Announced', 'All Prefixes Withdrawn']

    # Process list-type columns
    for col in list_columns:
        if col in df.columns:
            df[col] = df.apply(lambda row: safe_parse_list(row[col], col, row.name), axis=1)
        else:
            logger.warning(f"Column '{col}' not found in the DataFrame. Filling with empty lists.")
            df[col] = [[] for _ in range(len(df))]

    # Convert numeric columns to float and fill missing columns
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            logger.info(f"Converted column '{col}' to numeric.")
        else:
            logger.warning(f"Column '{col}' not found in the DataFrame. Filling with 0.0.")
            df[col] = 0.0

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

    # Assign thresholds to anomaly rules with validation
    anomaly_rules = []
    for anomaly_rule in anomaly_rules_config:
        rule_copy = anomaly_rule.copy()
        # Deep copy the conditions to avoid mutating the original config
        rule_copy['conditions'] = [condition.copy() for condition in anomaly_rule['conditions']]
        for condition in rule_copy['conditions']:
            feature = condition['feature']
            operator_str = condition['operator']
            if condition['threshold'] is None:
                if feature in anomaly_thresholds:
                    threshold = anomaly_thresholds[feature]
                    condition['threshold'] = threshold
                    logger.debug(f"Set threshold for feature '{feature}' to {threshold} for anomaly '{rule_copy['type']}'.")
                else:
                    logger.error(f"Feature '{feature}' not found in anomaly_thresholds. Cannot assign threshold.")
                    # Optionally, set a default threshold or skip the condition
                    condition['threshold'] = 0  # Example default
            else:
                threshold = condition['threshold']
                logger.debug(f"Using predefined threshold {threshold} for feature '{feature}' in anomaly '{rule_copy['type']}'.")
        anomaly_rules.append(rule_copy)
    logger.info(f"Defined {len(anomaly_rules)} anomaly rules.")

    # Calculate total updates per peer and prefix
    total_updates_per_peer = defaultdict(int)
    total_updates_per_prefix = defaultdict(int)

    # Additional data structures to track announcements
    total_announcements_per_peer = defaultdict(int)
    total_announcements_per_prefix = defaultdict(int)

    if 'Average Updates per Prefix' not in df.columns:
        df['Average Updates per Prefix'] = df['Total Updates'] / df['Total Prefixes Announced']
        df['Average Updates per Prefix'] = df['Average Updates per Prefix'].fillna(0.0)

    # Initialize logs and lists
    data_point_logs = []
    anomalies = []
    prefix_announcements = []
    prefix_withdrawals = []
    updates_per_peer_info = []
    updates_per_prefix_info = []
    policy_logs = []

    # Process each data point
    for idx, row in df.iterrows():
        try:
            timestamp = row.get('Timestamp', 'N/A')
            as_number = row.get('Autonomous System Number', 'N/A')

            # Generate data point log
            log_entry = generate_data_point_log(row, timestamp, as_number)
            data_point_logs.append(log_entry + "\n")

            # Collect prefix announcements and withdrawals
            collect_prefix_events(
                row, idx, timestamp, as_number,
                prefix_announcements, prefix_withdrawals,
                total_updates_per_prefix,
            )

            # Extract ASN values from 'All Peers' column
            peers_list = row.get('All Peers', [])
            peer_asns = [str(asn) for asn in peers_list if isinstance(asn, (int, str)) and str(asn).isdigit()]

            # **Update total_updates_per_peer here based on actual updates**
            peer_updates = row.get('Peer Updates', {})
            if isinstance(peer_updates, dict):
                for asn, updates in peer_updates.items():
                    try:
                        asn_str = str(asn)
                        updates_int = int(updates)
                        total_updates_per_peer[asn_str] += updates_int
                        logger.debug(f"Updated AS{asn_str} with {updates_int} updates. Total now: {total_updates_per_peer[asn_str]}")
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid update count for AS{asn} at row {idx}. Skipping.")
            else:
                # If 'Peer Updates' column does not exist, consider incrementing by 1 per occurrence
                for asn in peer_asns:
                    total_updates_per_peer[asn] += 1  # Increment by 1 per occurrence
                    logger.debug(f"Incremented AS{asn} updates by 1. Total now: {total_updates_per_peer[asn]}")

            # Track announcements per peer
            announcements_count = row.get('Announcements', 0)
            for asn in peer_asns:
                total_announcements_per_peer[asn] += announcements_count
                logger.debug(f"AS{asn} made {announcements_count} announcements. Total now: {total_announcements_per_peer[asn]}")

            # Generate updates per peer information with timestamp
            updates_info = generate_updates_per_peer_info(timestamp, peer_asns, total_updates_per_peer)
            updates_per_peer_info.append(updates_info + "\n")

            # Extract Top Prefixes
            top_prefixes = row.get('All Prefixes Announced', [])
            prefix_updates_list = [total_updates_per_prefix.get(prefix, 0) for prefix in top_prefixes]
            logger.debug(f"Row {idx}: Top Prefixes: {top_prefixes}")
            logger.debug(f"Row {idx}: Prefix Updates: {prefix_updates_list}")

            # Accumulate updates per prefix
            for prefix, updates in zip(top_prefixes, prefix_updates_list):
                if prefix:
                    total_updates_per_prefix[prefix] += updates
                    logger.debug(f"Accumulated {updates} updates for Prefix {prefix}. Total so far: {total_updates_per_prefix[prefix]}")
                else:
                    logger.debug(f"Encountered empty prefix at row {idx}.")

            # Track announcements per prefix
            for prefix in top_prefixes:
                if prefix:
                    total_announcements_per_prefix[prefix] += 1  # Increment by 1 per announcement
                    logger.debug(f"Prefix {prefix} announced. Total announcements: {total_announcements_per_prefix[prefix]}")
                else:
                    logger.debug(f"Empty prefix encountered in announcements at row {idx}.")

            # Generate updates per prefix information
            updates_prefix_info = generate_updates_per_prefix_info(timestamp, top_prefixes, prefix_updates_list)
            updates_per_prefix_info.append(updates_prefix_info + "\n")
            logger.debug(f"Row {idx}: Updates per prefix info added.")

            # Detect and store anomalies
            anomalies_detected = detect_anomalies(row, anomaly_rules, require_all_conditions=True)  # Assuming all conditions must be met
            if anomalies_detected:
                for anomaly in anomalies_detected:
                    anomaly_entry = {
                        'timestamp': timestamp,
                        'as_number': as_number,
                        'type': anomaly['type'],
                        'description': anomaly['description'],
                        'features_triggered': anomaly['features_triggered'],
                        'details': anomaly['details']
                    }
                    anomalies.append(anomaly_entry)
                    logger.info(f"Anomaly detected: {anomaly_entry}")

            # Collect policy-related information
            policy_info = collect_policy_info(row)
            if policy_info:
                policy_log = generate_policy_log(policy_info, timestamp, as_number)
                policy_logs.append(policy_log)
                logger.debug(f"Row {idx}: Policy info added.")

        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            continue

    # Write logs to files
    write_logs_to_file(data_point_logs, output_dir, data_point_logs_filename)
    write_logs_to_file(prefix_announcements, output_dir, prefix_announcements_filename)
    write_logs_to_file(prefix_withdrawals, output_dir, prefix_withdrawals_filename)
    write_logs_to_file(updates_per_peer_info, output_dir, updates_per_peer_filename)
    write_logs_to_file(updates_per_prefix_info, output_dir, updates_per_prefix_filename)
    
    # Generate Overall Summary
    overall_summary_text = generate_overall_summary(
        df, summary_metrics, total_updates_per_peer,
        total_updates_per_prefix, total_announcements_per_peer,
        total_announcements_per_prefix
    )

    # Write Overall Summary to File
    with open(os.path.join(output_dir, overall_summary_filename), 'w', encoding='utf-8') as f:
        f.write(overall_summary_text)

    # Write detailed anomaly logs
    with open(os.path.join(output_dir, anomalies_filename), 'w', encoding='utf-8') as f:
        for anomaly in anomalies:
            anomaly_log = generate_anomaly_log(anomaly, anomaly['timestamp'], anomaly['as_number'])
            f.write(anomaly_log)

    # Write anomaly summary
    generate_and_write_anomaly_summary(anomalies, output_dir, anomaly_summary_filename)

    # Write AS path changes summary
    generate_as_path_changes_summary(df, output_dir, as_path_changes_summary_filename)

    # Write policy summary
    policy_summary_text = generate_policy_summary(df, summary_metrics)
    with open(os.path.join(output_dir, policy_summary_filename), 'w', encoding='utf-8') as f:
        f.write(policy_summary_text)

    print(f"Data processing complete. Output files are saved in the '{output_dir}' directory.")
