import os
import pandas as pd
import ast
from collections import defaultdict, Counter
import logging
import operator

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

def process_list_column(df, column_list, default_value):
    for col in column_list:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(
                lambda x: [item.strip() for item in x.strip("[]").replace("'", "").split(',') if item.strip()]
            )
        else:
            logging.warning(f"Column '{col}' not found in the DataFrame. Filling with default.")
            df[col] = [default_value for _ in range(len(df))]

def generate_overall_summary(df, summary_metrics, total_updates_per_peer, total_updates_per_prefix,
                            total_announcements_per_peer, total_announcements_per_prefix):
    overall_summary_text = "BGP Monitoring Overview:\n\n"
    
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

    # **All Peers Summary**
    total_peers = df['All Peers'].explode().dropna().unique()
    num_peers = len(total_peers)
    overall_summary_text += f"Total Number of Unique Peers Observed: {num_peers}\n"
    if num_peers > 0:
        peers_list = ', '.join([f"AS{asn}" for asn in total_peers])
        overall_summary_text += f"List of Peers: {peers_list}\n"
    else:
        overall_summary_text += "No peer data available.\n"

    overall_summary_text += "\n"

    # **Total and Average Updates per Peer**
    total_updates = sum(total_updates_per_peer.values())
    average_updates = total_updates / len(total_updates_per_peer) if total_updates_per_peer else 0
    overall_summary_text += (
        f"Total Updates across all peers: {int(total_updates)}\n"
        f"Average Updates per Peer: {average_updates:.2f}\n"
    )
    overall_summary_text += "\n"

    # **Top Peers with Most Updates**
    top_n_peers = 5
    overall_summary_text += f"Top {top_n_peers} Peers with the Highest Number of Updates:\n"
    sorted_peers = sorted(total_updates_per_peer.items(), key=lambda x: x[1], reverse=True)[:top_n_peers]
    if sorted_peers and any(updates > 0 for _, updates in sorted_peers):
        for asn, updates in sorted_peers:
            if updates > 0:
                overall_summary_text += f" - AS{asn}: {int(updates)} updates\n"
            else:
                overall_summary_text += f" - AS{asn}: {int(updates)} updates (No updates recorded)\n"
    else:
        overall_summary_text += "No peer updates data available.\n"

    overall_summary_text += "\n"

    # **Top Prefixes with Most Updates**
    top_n_prefixes = 5
    overall_summary_text += f"Top {top_n_prefixes} Prefixes with the Highest Number of Updates:\n"
    sorted_prefixes = sorted(total_updates_per_prefix.items(), key=lambda x: x[1], reverse=True)[:top_n_prefixes]
    if sorted_prefixes and any(updates > 0 for _, updates in sorted_prefixes):
        for prefix, updates in sorted_prefixes:
            overall_summary_text += f" - Prefix {prefix}: {int(updates)} updates\n"
    else:
        overall_summary_text += "No prefix updates data available.\n"

    overall_summary_text += "\n"

    # **Top Announced Peers**
    overall_summary_text += f"Top {top_n_peers} Peers with the Highest Number of Announcements:\n"
    sorted_announced_peers = sorted(total_announcements_per_peer.items(), key=lambda x: x[1], reverse=True)[:top_n_peers]
    if sorted_announced_peers:
        for asn, announcements in sorted_announced_peers:
            overall_summary_text += f" - AS{asn}: {int(announcements)} announcements\n"
    else:
        overall_summary_text += "No peer announcements data available.\n"

    overall_summary_text += "\n"

    # **Top Announced Prefixes**
    overall_summary_text += f"Top {top_n_prefixes} Prefixes with the Highest Number of Announcements:\n"
    sorted_announced_prefixes = sorted(total_announcements_per_prefix.items(), key=lambda x: x[1], reverse=True)[:top_n_prefixes]
    if sorted_announced_prefixes:
        for prefix, announcements in sorted_announced_prefixes:
            overall_summary_text += f" - Prefix {prefix}: {int(announcements)} announcements\n"
    else:
        overall_summary_text += "No prefix announcements data available.\n"

    overall_summary_text += "\n"

    # **Policy-Related Metrics Summary**
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
    overall_summary_text += f"AS Path Prepending was observed {int(total_prepending)} times during the observation period.\n"

    # Community Values Summary
    all_communities = set()
    for communities in df['Community Values']:
        all_communities.update(communities)
    overall_summary_text += f"A total of {len(all_communities)} unique community values were observed.\n"

    return overall_summary_text

def generate_as_path_changes_summary(df, output_dir, as_path_changes_summary_filename):
    try:
        summary = "AS Path Changes Summary Report\n"
        summary += "="*30 + "\n\n"

        # Ensure 'AS Path Changes' and 'Autonomous System Number' columns exist
        missing_columns = []
        for col in ['AS Path Changes', 'Autonomous System Number', 'All Paths']:
            if col not in df.columns:
                missing_columns.append(col)

        if missing_columns:
            for col in missing_columns:
                logging.error(f"Column '{col}' not found in the DataFrame.")
                summary += f"Error: '{col}' column is missing from the data.\n"
            with open(os.path.join(output_dir, as_path_changes_summary_filename), 'w', encoding='utf-8') as f:
                f.write(summary)
            return

        # Calculate total AS path changes
        total_changes = df['AS Path Changes'].sum()
        summary += f"Total AS Path Changes Detected: {int(total_changes)}\n\n"

        # Aggregate AS path changes per ASN
        as_path_changes_per_asn = df.groupby('Autonomous System Number')['AS Path Changes'].sum()

        # Identify top 5 ASNs with the most AS path changes
        top_asns = as_path_changes_per_asn.sort_values(ascending=False).head(5)

        summary += "Top ASNs with Most AS Path Changes:\n"
        if not top_asns.empty:
            for asn, changes in top_asns.items():
                summary += f" - AS{asn}: {int(changes)} changes\n"
        else:
            summary += "No AS path changes detected.\n"

        summary += "\n"

        # **List All AS Paths Involved in Changes**
        # Extract all AS paths from the 'All Paths' column
        all_paths = df['All Paths'].explode().dropna().unique()
        summary += "List of All AS Paths Involved in Changes:\n"
        for path in all_paths:
            summary += f" - {path}\n"

        summary += "\n"

        # **Identify and List the Most Frequent AS Paths**
        # Flatten all paths and count their frequencies
        all_paths_flat = df['All Paths'].explode().dropna()
        path_frequencies = all_paths_flat.value_counts()
        most_frequent_paths = path_frequencies.head(5)

        summary += "Top 5 Most Frequent AS Paths:\n"
        if not most_frequent_paths.empty:
            for path, freq in most_frequent_paths.items():
                summary += f" - Path: {path} | Frequency: {int(freq)}\n"
        else:
            summary += "No AS path data available to determine frequencies.\n"

        summary += "\n"

        # Write summary to file
        summary_file_path = os.path.join(output_dir, as_path_changes_summary_filename)
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write(summary)

        logging.info(f"AS Path Changes Summary Report generated at '{summary_file_path}'.")

    except Exception as e:
        logging.error(f"Failed to generate AS Path Changes Summary: {e}")

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

def collect_prefix_events(row, idx, timestamp, as_number, prefix_announcements, prefix_withdrawals, total_updates_per_prefix, logger):
    # Announcements
    total_prefixes_announced = row.get('Total Prefixes Announced', 0)
    if total_prefixes_announced > 0:
        prefixes_announced_str = row.get('Target Prefixes Announced', '[]')
        prefixes_announced_list = parse_prefix_list(prefixes_announced_str, idx, 'Target Prefixes Announced')
        logger.debug(f"Row {idx}: Prefixes Announced: {prefixes_announced_list}")
        for prefix in prefixes_announced_list:
            announcement = (
                f"At {timestamp}, AS{as_number} announced the prefix: {prefix}. "
                f"Total prefixes announced: {int(total_prefixes_announced)}."
            )
            prefix_announcements.append(announcement)
            total_updates_per_prefix[prefix] += 1  # Increment updates per prefix
            logger.debug(f"Incremented updates for prefix {prefix}: {total_updates_per_prefix[prefix]}")

    # Withdrawals
    total_prefixes_withdrawn = row.get('Target Prefixes Withdrawn', 0)
    if total_prefixes_withdrawn > 0:
        prefixes_withdrawn_str = row.get('Target Prefixes Withdrawn', '[]')
        prefixes_withdrawn_list = parse_prefix_list(prefixes_withdrawn_str, idx, 'Target Prefixes Withdrawn')
        logger.debug(f"Row {idx}: Prefixes Withdrawn: {prefixes_withdrawn_list}")
        for prefix in prefixes_withdrawn_list:
            withdrawal = (
                f"At {timestamp}, AS{as_number} withdrew the prefix: {prefix}. "
                f"Total prefixes withdrawn: {int(total_prefixes_withdrawn)}."
            )
            prefix_withdrawals.append(withdrawal)
            total_updates_per_prefix[prefix] += 1  # Increment updates per prefix
            logger.debug(f"Incremented updates for prefix {prefix}: {total_updates_per_prefix[prefix]}")

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

def generate_updates_per_peer_info(timestamp, peer_asns, total_updates_per_peer):
    if peer_asns:
        updates_info = f"Peer Updates Information at {timestamp}:\n"
        for asn in peer_asns:
            updates = total_updates_per_peer.get(asn, 0)
            updates_info += f"  - AS{asn}: {int(updates)} updates\n"
    else:
        updates_info = f"Peer Updates Information at {timestamp}: No peer updates information available.\n"
    
    return updates_info

def generate_updates_per_prefix_info(timestamp, top_prefixes, prefix_updates):
    updates_info = f"At {timestamp}, updates per top prefix were as follows:\n"
    
    if top_prefixes and prefix_updates:
        for prefix, updates in zip(top_prefixes, prefix_updates):
            if prefix:  # Ensure prefix is not empty
                updates_info += f"  - Prefix {prefix}: {int(updates)} updates\n"
            else:
                updates_info += "  - Prefix: None\n"
    else:
        updates_info += "  - No prefixes available.\n"
    
    return updates_info


def detect_anomalies(row, anomaly_rules, require_all_conditions=True):
    operators_map = {
        '>': operator.gt,
        '<': operator.lt,
        '>=': operator.ge,
        '<=': operator.le,
        '==': operator.eq,
        '!=': operator.ne
    }

    anomalies_detected = []

    for anomaly_rule in anomaly_rules:
        anomaly_type = anomaly_rule['type']
        description = anomaly_rule['description']
        conditions = anomaly_rule['conditions']

        # Flags to track condition satisfaction
        if require_all_conditions:
            conditions_met = True
        else:
            conditions_met = False

        triggered_features = []
        details = {}

        for condition in conditions:
            feature = condition['feature']
            operator_str = condition['operator']
            threshold = condition['threshold']

            # Get the operator function
            op_func = operators_map.get(operator_str)
            if not op_func:
                logging.error(f"Unsupported operator '{operator_str}' in anomaly rule for feature '{feature}'.")
                continue

            # Get the feature value from the row
            feature_value = row.get(feature, None)
            if feature_value is None:
                # If feature is missing, skip this condition
                logging.warning(f"Feature '{feature}' not found in the data row. Skipping condition.")
                continue

            logging.debug(f"Evaluating anomaly condition for feature '{feature}': value={feature_value}, threshold={threshold}")

            # Apply the operator to compare feature value and threshold
            if op_func(feature_value, threshold):
                triggered_features.append(feature)
                details[feature] = feature_value
                if not require_all_conditions:
                    conditions_met = True
                    break
            else:
                if require_all_conditions:
                    conditions_met = False
                    break

        if conditions_met and triggered_features:
            anomalies_detected.append({
                'type': anomaly_type,
                'description': description,
                'features_triggered': triggered_features,
                'details': details
            })

    return anomalies_detected

def generate_anomaly_log(anomaly, timestamp, as_number):
    anomaly_log = (
        f"On {timestamp}, Autonomous System {as_number} experienced an anomaly:\n"
        f"  - {anomaly['type']} detected ({anomaly['description']})\n"
    )

    # List triggered features and their details
    for feature, value in anomaly['details'].items():
        if feature == 'unexpected_asns_in_paths':
            # Format the list of ASNs
            asns_str = ', '.join([f"AS{asn}" for asn in value])
            anomaly_log += f"    Unexpected ASNs in Paths: {asns_str}\n"
        else:
            anomaly_log += f"    Feature: {feature}\n"
            anomaly_log += f"    Value: {value}\n"

    anomaly_log += "\n"
    return anomaly_log

def generate_and_write_anomaly_summary(anomalies, output_dir, filename):
    anomaly_summary = "Anomaly Summary:\n\n"
    total_anomalies = len(anomalies)
    if total_anomalies > 0:
        anomaly_summary += f"A total of {total_anomalies} anomalies were detected during the observation period.\n\n"

        # Group anomalies by type
        from collections import defaultdict

        anomaly_counts = defaultdict(int)
        anomaly_details = defaultdict(list)

        for anomaly in anomalies:
            anomaly_type = anomaly['type']
            anomaly_counts[anomaly_type] += 1
            detail = (
                f"- At {anomaly['timestamp']}, AS{anomaly['as_number']} had triggered "
                f"{', '.join(anomaly['features_triggered'])}."
            )
            anomaly_details[anomaly_type].append(detail)

        # Create summary for each anomaly type
        for anomaly_type, count in anomaly_counts.items():
            anomaly_summary += f"{anomaly_type}: {count} occurrences\n"
            # Include brief details (e.g., first few occurrences)
            details = anomaly_details[anomaly_type][:5]  # Show up to first 5 details
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
            anomaly_log = generate_anomaly_log(anomaly, anomaly['timestamp'], anomaly['as_number'])
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