import os
import pandas as pd
import json
from transformers import pipeline
import csv
from collections import defaultdict, Counter
import ast
import inflect
import matplotlib.pyplot as plt

def combine_csv_files(directory):
    """
    Load all CSV files from a directory and combine them into a single DataFrame.

    Parameters:
    directory (str): The directory containing the CSV files.

    Returns:
    pd.DataFrame: A DataFrame containing all the data from the CSV files.
    """
    combined_df = pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df

def split_dataframe(input_data, split_size=10):
    """
    Split a DataFrame or a CSV file into smaller DataFrames with a specified number of rows.

    Parameters:
    input_data (str or pd.DataFrame): The path to the CSV file or the DataFrame to split.
    split_size (int): The number of rows in each chunk. Default is 10.

    Returns:
    list of pd.DataFrame: A list containing the smaller DataFrames.
    """
    if isinstance(input_data, str):
        # If input_data is a string, assume it's a path to a CSV file
        df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        # If input_data is already a DataFrame, use it directly
        df = input_data
    else:
        raise ValueError("input_data should be a path to a CSV file or a pandas DataFrame")

    num_chunks = len(df) // split_size + (1 if len(df) % split_size > 0 else 0)
    data_list = [df.iloc[i*split_size:(i+1)*split_size].reset_index(drop=True) for i in range(num_chunks)]
    
    return data_list

def shuffle_and_split_dataframe(input_data, split_size=10):
    """
    Shuffle the input data and split it into smaller DataFrames with a specified number of rows.

    Parameters:
    input_data (str or pd.DataFrame): The path to the CSV file or the DataFrame to split.
    split_size (int): The number of rows in each chunk. Default is 10.

    Returns:
    list of pd.DataFrame: A list containing the smaller DataFrames.
    """
    if isinstance(input_data, str):
        # If input_data is a string, assume it's a path to a CSV file
        df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        # If input_data is already a DataFrame, use it directly
        df = input_data
    else:
        raise ValueError("input_data should be a path to a CSV file or a pandas DataFrame")

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split the DataFrame into smaller chunks
    num_chunks = len(df) // split_size + (1 if len(df) % split_size > 0 else 0)
    data_list = [df.iloc[i*split_size:(i+1)*split_size].reset_index(drop=True) for i in range(num_chunks)]

    return data_list


def preprocess_data(chunk):
    """
    Convert a DataFrame chunk into the specified JSON format for LLM training.

    Parameters:
    chunk (pd.DataFrame): The DataFrame chunk to convert.

    Returns:
    dict: A dictionary in the specified JSON format.
    """
    instruction = "The goal for this task is to determine if the data indicates an anomaly. The context, section, and table columns provide important information for identifying the correct anomaly type."
    input_seg = "[TLE] The section is related to a specific time period of BGP monitoring. [TAB] col: | " + " | ".join(chunk.columns) + " |"
    
    for idx, row in chunk.iterrows():
        input_seg += " row {}: | ".format(idx+1) + " | ".join([str(x) for x in row.values]) + " | [SEP]"
    
    question = "Based on the data provided, does the data indicate an anomaly? If an anomaly is detected, include the timestamp of the anomaly data and provide a reason explaining which values are anomalous. For example, 'anomaly detected at 2024-06-10 12:00:00 due to high value of num_routes=77'"

    result = {
        "instruction": instruction,
        "input_seg": input_seg,
        "question": question,
    }
    
    return result

def preprocess_data_instruct(data):
    """
    Convert a DataFrame chunk into the specified JSON format for LLM training.

    Parameters:
    data (pd.DataFrame): The DataFrame chunk to convert.

    Returns:
    dict: A dictionary in the specified JSON format.
    """
    instruction = "The goal for this task is to determine if the data indicates an anomaly. Based on the context, section, and table columns provide important information for identifying the correct anomaly type. If an anomaly is detected, include the timestamp of the anomaly data and provide a reason explaining which values are anomalous and why. For example, 'anomaly detected at 2024-06-10 12:00:00 due to high value of num_routes=77'"
    input_seg = "[TLE] The context is about BGP data analysis for detecting anomalies. The section is related to a specific time period of BGP monitoring. [TAB] col: | " + " | ".join(data.columns) + " |"
    
    for idx, row in data.iterrows():
        input_seg += " row {}: | ".format(idx+1) + " | ".join([str(x) for x in row.values]) + " | [SEP]"
    
    
    # Determine the output based on the anomaly_status column
    anomaly_statuses = data.loc[data['anomaly_status'] != 'no anomalies detected', 'anomaly_status'].unique()
    if len(anomaly_statuses) == 0:
        output = "no anomalies detected"
    else:
        output = "; ".join(anomaly_statuses)
    
    result = {
        "instruction": instruction,
        "input": input_seg,
        "output": output,
    }
    
    return result

def preprocess_data_to_txt(data, output_file='preprocessed_data.txt'):
    """
    Convert a DataFrame into a text format for LLM training and save to a file.

    Parameters:
    data (pd.DataFrame): The DataFrame to convert.
    output_file (str): The output file to save the text. Default is 'preprocessed_data.txt'.
    """
    # Prepare the header with column names
    input_seg = "[TAB] col: | " + " | ".join(data.columns) + " |"
    
    # Prepare the rows without splitting
    for idx, row in data.iterrows():
        input_seg += " row {}: | ".format(idx + 1) + " | ".join([str(x) for x in row.values]) + " |"
    
    # Save the processed data to a text file
    with open(output_file, 'w') as f:
        f.write(input_seg)
    
    print(f"Data successfully saved to {output_file}")

def run_llm_inference(formatted_data, model, tokenizer, max_length=3050, output_results_file='llm_output.json'):
    """
    Run LLM inference on the formatted data and save the results to a JSON file.

    Parameters:
    formatted_data (list): The formatted data to run inference on.
    model: The model to use for inference.
    tokenizer: The tokenizer to use for inference.
    max_length (int): The maximum length of the generated text.
    output_results_file (str): The file to save the results to.
    """
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=max_length)
    results = []

    if not isinstance(formatted_data, list):
        formatted_data = [formatted_data]

    for i, data in enumerate(formatted_data):
        try:
            prompt = f"{data['instruction']}:\n{data['input_seg']}\n{data['question']}"
            
            result = pipe(f"<s>[INST] {prompt} [/INST]")
            generated_text = result[0]['generated_text']
            
            # Extract the part of generated_text after [/INST]
            generated_output = generated_text.split("[/INST]")[-1].strip()
            
            results.append({
                "instruction": data['instruction'],
                "input_seg": data['input_seg'],
                "question": data['question'],
                "output": generated_output
            })
            
            print(f"Processed {i + 1}/{len(formatted_data)}")
        except KeyError as e:
            print(f"KeyError: {e} in data entry {i}")
        except Exception as e:
            print(f"An error occurred: {e}")

    # Save results to a JSON file
    print(results)
    with open(output_results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_results_file}")
    
    
def evaluate_llm_results(true_anomalies, llm_results):
    """
    Evaluate the performance of LLM anomaly detection.

    Parameters:
    true_anomalies (list): A list of true anomaly timestamps.
    llm_results (list): A list of LLM-detected anomaly timestamps.

    Returns:
    dict: A dictionary containing precision, recall, and F1-score.
    """
    true_anomaly_set = set(true_anomalies)
    llm_result_set = set(llm_results)
    
    tp = len(true_anomaly_set & llm_result_set)
    fp = len(llm_result_set - true_anomaly_set)
    fn = len(true_anomaly_set - llm_result_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn
    }


def process_bgp_csv(
    csv_file_path,
    output_dir='processed_output',
    overall_summary_filename='overall_summary.txt',
    data_point_logs_filename='data_point_logs.txt',
    prefix_announcements_filename='prefix_announcements.txt',
    prefix_withdrawals_filename='prefix_withdrawals.txt',
    updates_per_peer_filename='updates_per_peer.txt',
    hijacking_filename='hijacking_anomalies.txt',
    leaks_filename='leaks_anomalies.txt',
    outages_filename='outages_anomalies.txt',
    anomaly_summary_filename='anomaly_summary.txt',
    top_n_peers=10       # Number of top peers to include in the overall summary
):
    """
    Processes a BGP CSV file and generates enriched text files in narrative format for:
    1. Overall summary with key statistics and top peers.
    2. Textual summaries of each data point.
    3. Prefix announcement information.
    4. Prefix withdrawal information.
    5. Updates per peer information.
    6. Detailed anomaly detection logs for hijacking, leaks, and outages.
    7. Anomaly summary indicating if anomalies were detected.
    
    Args:
        csv_file_path (str): Path to the input CSV file.
        output_dir (str): Directory where output files will be saved.
        overall_summary_filename (str): Filename for overall summary.
        data_point_logs_filename (str): Filename for data point logs.
        prefix_announcements_filename (str): Filename for prefix announcements.
        prefix_withdrawals_filename (str): Filename for prefix withdrawals.
        updates_per_peer_filename (str): Filename for updates per peer.
        hijacking_filename (str): Filename for hijacking anomalies.
        leaks_filename (str): Filename for leaks anomalies.
        outages_filename (str): Filename for outages anomalies.
        anomaly_summary_filename (str): Filename for anomaly summary.
        top_n_peers (int): Number of top peers to include in the overall summary.
    """
    # Define explanations for anomalies
    anomaly_definitions = {
        "Hijacking": (
            "BGP Hijacking occurs when an Autonomous System (AS) falsely advertises IP prefixes that it does not own, "
            "potentially diverting traffic to malicious destinations or causing network disruptions."
        ),
        "Leaks": (
            "BGP Leaks refer to the unintended or malicious disclosure of internal routing information to external networks, "
            "which can lead to traffic misrouting and reduced network performance."
        ),
        "Outages": (
            "BGP Outages happen when there is a loss of connectivity or a significant reduction in the number of active routes, "
            "resulting in disrupted network services."
        )
    }

    # Define detection methodologies for anomalies
    anomaly_detection_methods = {
        "Hijacking": (
            "Anomalies related to hijacking were detected using two key indicators based on statistical thresholds:\n"
            "1. High Announcements: If the number of announcements exceeded the mean by three standard deviations (Threshold: {threshold_announcements}), "
            "it was flagged as a potential hijack.\n"
            "   - Observed Announcements: {observed_announcements}\n"
            "   - Mean Announcements: {mean_announcements}\n"
            "   - Standard Deviation (Announcements): {std_announcements}\n\n"
            "2. Unexpected ASNs in Paths: If the count of unexpected ASNs in BGP paths exceeded the mean by two standard deviations (Threshold: {threshold_unexpected_asns}), "
            "it indicated a possible hijack.\n"
            "   - Observed Unexpected ASNs: {observed_unexpected_asns}\n"
            "   - Mean Unexpected ASNs: {mean_unexpected_asns}\n"
            "   - Standard Deviation (Unexpected ASNs): {std_unexpected_asns}\n"
        ),
        "Leaks": (
            "Anomalies related to leaks were detected by monitoring withdrawal activities using statistical thresholds:\n"
            "1. High Withdrawals: If the number of withdrawals exceeded the mean by three standard deviations (Threshold: {threshold_withdrawals}), "
            "it was flagged as a potential leak or outage.\n"
            "   - Observed Withdrawals: {observed_withdrawals}\n"
            "   - Mean Withdrawals: {mean_withdrawals}\n"
            "   - Standard Deviation (Withdrawals): {std_withdrawals}\n"
        ),
        "Outages": (
            "Anomalies related to outages were identified by observing the total number of active routes using statistical thresholds:\n"
            "1. Low Total Routes: If the total number of active routes fell below the mean by three standard deviations (Threshold: {threshold_total_routes}), "
            "it was indicative of a potential outage.\n"
            "   - Observed Total Routes: {observed_total_routes}\n"
            "   - Mean Total Routes: {mean_total_routes}\n"
            "   - Standard Deviation (Total Routes): {std_total_routes}\n"
        )
    }

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file into a pandas DataFrame for easier manipulation
    try:
        df = pd.read_csv(csv_file_path)
        print("CSV file successfully read. Here's a preview:")
        print(df.head())
        print("\nColumn Names:")
        print(df.columns.tolist())
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{csv_file_path}' is empty.")
        return
    except pd.errors.ParserError as e:
        print(f"Error parsing '{csv_file_path}': {e}")
        return

    # Define numeric columns (ensure these are the actual numeric columns in your CSV)
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
        'Average Prefix Length', 'Max Prefix Length', 'Min Prefix Length'
    ]

    # Identify which numeric columns are present in the DataFrame
    numeric_columns_present = [col for col in numeric_columns if col in df.columns]

    # Convert numeric columns to float, handle errors by setting invalid entries to 0
    for col in numeric_columns_present:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Warn about missing numeric columns and fill them with 0.0
    missing_numeric_columns = set(numeric_columns) - set(numeric_columns_present)
    for col in missing_numeric_columns:
        print(f"Warning: Column '{col}' not found in the CSV. Filling with 0.0.")
        df[col] = 0.0
        numeric_columns_present.append(col)  # Add to present list to include in computations

    # Handle list columns (Top Peers and Top Prefixes) appropriately
    peer_columns = ['Top Peer 1 ASN', 'Top Peer 2 ASN', 'Top Peer 3 ASN', 'Top Peer 4 ASN', 'Top Peer 5 ASN']
    prefix_columns = ['Top Prefix 1', 'Top Prefix 2', 'Top Prefix 3', 'Top Prefix 4', 'Top Prefix 5']

    # Convert peer columns to lists
    for col in peer_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(
                lambda x: [item.strip() for item in x.strip("[]").replace("'", "").split(',') if item.strip()]
            )
        else:
            print(f"Warning: Column '{col}' not found in the CSV. Filling with empty lists.")
            df[col] = [[] for _ in range(len(df))]
    
    # Convert prefix columns to strings and handle '0' as no prefix
    for col in prefix_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(
                lambda x: '' if x.strip() == '0' else x.strip("[]").replace("'", "").strip()
            )
        else:
            print(f"Warning: Column '{col}' not found in the CSV. Filling with empty strings.")
            df[col] = ''  # Empty string instead of list

    # Initialize min and max dictionaries for overall summary
    summary_metrics = {}
    for col in numeric_columns_present:
        summary_metrics[col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'average': df[col].mean(),
            'std_dev': df[col].std()
        }

    # Pre-calculate means and std_dev for detection methodologies
    means = df[numeric_columns_present].mean()
    std_devs = df[numeric_columns_present].std()

    # Calculate total updates per peer
    total_updates_per_peer = defaultdict(float)
    for idx, row in df.iterrows():
        peers_nested = row[peer_columns]  # List of lists
        # Flatten the list of peers
        peers = [asn for peer_list in peers_nested for asn in peer_list]
        for peer_asn in peers:
            # Ensure peer_asn is a string before calling isdigit()
            if isinstance(peer_asn, str) and peer_asn.isdigit():
                # Accumulate the updates per peer ASN
                total_updates_per_peer[peer_asn] += row['Average Updates per Peer']

    # Calculate prefix announcement counts
    prefix_announcement_counter = Counter()
    for idx, row in df.iterrows():
        for col in prefix_columns:
            prefix = row[col]
            if prefix and prefix != '0':
                # If multiple prefixes are present in a single string separated by commas, split them
                split_prefixes = [p.strip() for p in prefix.split(',') if p.strip() and p.strip() != '0']
                for p in split_prefixes:
                    prefix_announcement_counter[p] += 1

    # Calculate prefix withdrawal counts
    prefix_withdrawal_counter = Counter()
    for idx, row in df.iterrows():
        withdrawn_prefixes_str = row.get('Target Prefixes Withdrawn', '0')
        if isinstance(withdrawn_prefixes_str, str) and withdrawn_prefixes_str.strip() and withdrawn_prefixes_str.strip() != '0':
            try:
                withdrawn_prefixes = ast.literal_eval(withdrawn_prefixes_str)
                if isinstance(withdrawn_prefixes, list):
                    for prefix in withdrawn_prefixes:
                        if isinstance(prefix, list):
                            # Handle nested lists
                            if len(prefix) == 1 and isinstance(prefix[0], str):
                                prefix = prefix[0]
                            else:
                                print(f"Warning: Found a nested list in withdrawn prefix: {prefix}. Skipping.")
                                continue
                        if isinstance(prefix, str) and prefix.strip() != '0':
                            prefix_withdrawal_counter[prefix.strip()] += 1
            except (SyntaxError, ValueError):
                print(f"Warning: Could not parse 'Target Prefixes Withdrawn' at index {idx}. Skipping.")

    # Define anomaly thresholds based on mean and standard deviation
    anomaly_thresholds = {}
    if 'Withdrawals' in df.columns:
        anomaly_thresholds['Withdrawals'] = means['Withdrawals'] + 3 * std_devs['Withdrawals']
    if 'Announcements' in df.columns:
        anomaly_thresholds['Announcements'] = means['Announcements'] + 3 * std_devs['Announcements']
    if 'Total Routes' in df.columns:
        anomaly_thresholds['Total Routes'] = means['Total Routes'] - 3 * std_devs['Total Routes']
    if 'Count of Unexpected ASNs in Paths' in df.columns:
        anomaly_thresholds['Unexpected ASNs'] = means['Count of Unexpected ASNs in Paths'] + 2 * std_devs['Count of Unexpected ASNs in Paths']

    # Generate Overall Summary in Narrative Format
    overall_summary_text = "Overall Summary:\n\n"

    # Create a narrative paragraph summarizing key statistics
    overall_summary_text += (
        f"During the observation period, Autonomous Systems reported various BGP metrics. "
        f"The total number of routes ranged from {int(summary_metrics['Total Routes']['min'])} to {int(summary_metrics['Total Routes']['max'])}, "
        f"with an average of {summary_metrics['Total Routes']['average']:.2f} routes. "
        f"Announcements varied between {int(summary_metrics['Announcements']['min'])} and {int(summary_metrics['Announcements']['max'])}, "
        f"averaging {summary_metrics['Announcements']['average']:.2f} announcements. "
        f"Withdrawals were consistently {int(summary_metrics['Withdrawals']['min'])} throughout the period. "
        f"The maximum path length observed was {int(summary_metrics['Maximum Path Length']['max'])} hops, "
        f"with an average path length of {summary_metrics['Average Path Length']['average']:.2f} hops. "
        f"Communities ranged from {int(summary_metrics['Total Communities']['min'])} to {int(summary_metrics['Total Communities']['max'])}, "
        f"with an average of {summary_metrics['Total Communities']['average']:.2f}. "
        f"The system observed an average of {summary_metrics['Average Prefix Length']['average']:.1f} prefix length, "
        f"with a maximum of {int(summary_metrics['Max Prefix Length']['max'])} and a minimum of {int(summary_metrics['Min Prefix Length']['min'])}."
    )
    overall_summary_text += "\n\n"

    # Add top peers by total updates with descriptive language
    overall_summary_text += f"Top {top_n_peers} Peers with the Highest Number of Updates:\n"
    sorted_peers = sorted(total_updates_per_peer.items(), key=lambda x: x[1], reverse=True)[:top_n_peers]
    if sorted_peers:
        peer_details = ', '.join([f"AS{asn} ({updates:.2f} updates)" for asn, updates in sorted_peers])
        overall_summary_text += (
            f"The top {top_n_peers} peers contributing the most updates are {peer_details}.\n"
        )
    else:
        overall_summary_text += "No peer updates data available.\n"
    overall_summary_text += "\n"

    # Write Overall Summary to File
    with open(os.path.join(output_dir, overall_summary_filename), 'w', encoding='utf-8') as f:
        f.write(overall_summary_text)

    # Initialize anomaly logs and other lists
    data_point_logs = []
    hijacking_anomalies = []
    leaks_anomalies = []
    outages_anomalies = []
    prefix_announcements = []
    prefix_withdrawals = []
    updates_per_peer_info = []

    # Process each data point for logs and anomalies
    for idx, row in df.iterrows():
        timestamp = row.get('Timestamp', 'N/A')
        as_number = row.get('Autonomous System Number', 'N/A')

        # Collect all prefixes as a single string, excluding '0's
        prefixes = []
        for col in prefix_columns:
            prefix = row[col]
            if prefix and prefix != '0':
                split_prefixes = [p.strip() for p in prefix.split(',') if p.strip() and p.strip() != '0']
                prefixes.extend(split_prefixes)
        prefixes_str = ', '.join(prefixes) if prefixes else 'None'

        # Collect all peers as a single string
        peers = []
        for col in peer_columns:
            peer_list = row[col]  # List of ASNs
            peers.extend(peer_list)
        peers_str = ', '.join(peers) if peers else 'None'

        # Create the narrative log entry
        log_entry = (
            f"On {timestamp}, Autonomous System {as_number} observed {int(row['Announcements'])} announcements. "
            f"There were {int(row['New Routes'])} new routes added. The total number of active routes was {int(row['Total Routes'])}. "
            f"The maximum path length observed was {int(row['Maximum Path Length'])} hops, with an average path length of {row['Average Path Length']:.2f} hops. "
            f"The maximum edit distance was {int(row['Maximum Edit Distance'])}, with an average of {row['Average Edit Distance']:.1f}. "
            f"The average MED was {row['Average MED']:.2f}. The average local preference was {row['Average Local Preference']:.2f}. "
            f"There were {int(row['Total Communities'])} total communities observed, with {int(row['Unique Communities'])} unique communities. "
            f"The number of unique peers was {int(row['Average Updates per Peer'])}. "
            f"The average prefix length was {row['Average Prefix Length']:.1f}, with a maximum of {int(row['Max Prefix Length'])} and a minimum of {int(row['Min Prefix Length'])}. "
            f"Number of prefixes announced: {int(row['Total Prefixes Announced'])}. Number of prefixes withdrawn: {int(row['Target Prefixes Withdrawn'])}."
        )
        data_point_logs.append(log_entry + "\n")

        # Collect prefix announcements
        if row['Total Prefixes Announced'] > 0:
            prefixes_announced_str = row.get('Target Prefixes Announced', '0')
            if isinstance(prefixes_announced_str, str) and prefixes_announced_str.strip() != '0':
                try:
                    prefixes_announced_list = ast.literal_eval(prefixes_announced_str)
                    if isinstance(prefixes_announced_list, list):
                        for prefix in prefixes_announced_list:
                            if isinstance(prefix, list):
                                if len(prefix) == 1 and isinstance(prefix[0], str):
                                    prefix = prefix[0]
                                else:
                                    print(f"Warning: Found a nested list in 'Target Prefixes Announced' at index {idx}. Skipping.")
                                    continue
                            if isinstance(prefix, str) and prefix.strip() != '0':
                                prefix = prefix.strip()
                                announcement = (
                                    f"At {timestamp}, AS{as_number} announced the prefix: {prefix}. "
                                    f"Total prefixes announced: {int(row['Total Prefixes Announced'])}."
                                )
                                prefix_announcements.append(announcement)
                except (SyntaxError, ValueError):
                    print(f"Warning: Could not parse 'Target Prefixes Announced' at index {idx}. Skipping.")

        # Collect prefix withdrawals
        if row['Target Prefixes Withdrawn'] > 0:
            prefixes_withdrawn_str = row.get('Target Prefixes Withdrawn', '0')
            if isinstance(prefixes_withdrawn_str, str) and prefixes_withdrawn_str.strip() != '0':
                try:
                    prefixes_withdrawn_list = ast.literal_eval(prefixes_withdrawn_str)
                    if isinstance(prefixes_withdrawn_list, list):
                        for prefix in prefixes_withdrawn_list:
                            if isinstance(prefix, list):
                                if len(prefix) == 1 and isinstance(prefix[0], str):
                                    prefix = prefix[0]
                                else:
                                    print(f"Warning: Found a nested list in 'Target Prefixes Withdrawn' at index {idx}. Skipping.")
                                    continue
                            if isinstance(prefix, str) and prefix.strip() != '0':
                                prefix = prefix.strip()
                                withdrawal = (
                                    f"At {timestamp}, AS{as_number} withdrew the prefix: {prefix}. "
                                    f"Total prefixes withdrawn: {int(row['Target Prefixes Withdrawn'])}."
                                )
                                prefix_withdrawals.append(withdrawal)
                except (SyntaxError, ValueError):
                    print(f"Warning: Could not parse 'Target Prefixes Withdrawn' at index {idx}. Skipping.")

        # Collect updates per peer information in narrative format
        if peers:
            updates_info = f"At {timestamp}, updates per peer were as follows:"
            for peer_asn in peers:
                if isinstance(peer_asn, str) and peer_asn.isdigit():
                    updates = row['Average Updates per Peer']
                    updates_info += f" Peer AS{peer_asn} received {updates:.2f} updates."
            updates_per_peer_info.append(updates_info + "\n")
        else:
            updates_per_peer_info.append(f"At {timestamp}, no peer updates were observed.\n")

        # Detect Anomalies based on dynamic thresholds
        anomalies_detected = []
        unexpected_asns = []

        # 1. High Withdrawals (Possible Leak or Outage)
        if 'Withdrawals' in anomaly_thresholds and row['Withdrawals'] > anomaly_thresholds['Withdrawals']:
            anomalies_detected.append("High Withdrawals detected (Possible Leak or Outage).")

        # 2. High Announcements (Possible Hijack)
        if 'Announcements' in anomaly_thresholds and row['Announcements'] > anomaly_thresholds['Announcements']:
            anomalies_detected.append("High Announcements detected (Possible Hijack).")

        # 3. Low Total Routes (Possible Outage)
        if 'Total Routes' in anomaly_thresholds and row['Total Routes'] < anomaly_thresholds['Total Routes']:
            anomalies_detected.append("Low Total Routes detected (Possible Outage).")

        # 4. High Unexpected ASNs (Possible Hijack)
        if 'Unexpected ASNs' in anomaly_thresholds and row['Count of Unexpected ASNs in Paths'] > anomaly_thresholds['Unexpected ASNs']:
            anomalies_detected.append("High count of Unexpected ASNs in Paths detected (Possible Hijack).")
            # Collect the unexpected ASNs
            unexpected_asns = [
                row.get('Unexpected ASN 1', 'N/A'),
                row.get('Unexpected ASN 2', 'N/A'),
                row.get('Unexpected ASN 3', 'N/A')
            ]
            # Convert to strings and strip whitespace
            unexpected_asns = [str(asn).strip() for asn in unexpected_asns]
            # Filter out 'N/A' and non-digit ASNs
            unexpected_asns = [asn for asn in unexpected_asns if asn.isdigit()]

        # Log Anomalies in narrative format with details
        if anomalies_detected:
            anomaly_log = (
                f"On {timestamp}, Autonomous System {as_number} experienced the following anomalies:\n"
            )
            for anomaly in anomalies_detected:
                if "Unexpected ASNs" in anomaly:
                    asns_str = ', '.join([f"AS{asn}" for asn in unexpected_asns]) if unexpected_asns else 'None'
                    anomaly_log += f"  - {anomaly} (Count: {int(row['Count of Unexpected ASNs in Paths'])}, ASNs: {asns_str}).\n"
                else:
                    anomaly_log += f"  - {anomaly}\n"
            anomaly_log += "\n"

            # Append explanation and detection methodology with actual values
            for anomaly in anomalies_detected:
                if "Hijack" in anomaly:
                    explanation = anomaly_definitions['Hijacking']
                    detection_method = anomaly_detection_methods['Hijacking'].format(
                        threshold_announcements=anomaly_thresholds['Announcements'],
                        observed_announcements=row['Announcements'],
                        mean_announcements=means['Announcements'],
                        std_announcements=std_devs['Announcements'],
                        threshold_unexpected_asns=anomaly_thresholds['Unexpected ASNs'],
                        observed_unexpected_asns=row['Count of Unexpected ASNs in Paths'],
                        mean_unexpected_asns=means['Count of Unexpected ASNs in Paths'],
                        std_unexpected_asns=std_devs['Count of Unexpected ASNs in Paths']
                    )
                    hijacking_anomalies.append(anomaly_log + f"Explanation: {explanation}\n\nDetection Methodology: {detection_method}\n\n")
                elif "Leak" in anomaly:
                    explanation = anomaly_definitions['Leaks']
                    detection_method = anomaly_detection_methods['Leaks'].format(
                        threshold_withdrawals=anomaly_thresholds['Withdrawals'],
                        observed_withdrawals=row['Withdrawals'],
                        mean_withdrawals=means['Withdrawals'],
                        std_withdrawals=std_devs['Withdrawals']
                    )
                    leaks_anomalies.append(anomaly_log + f"Explanation: {explanation}\n\nDetection Methodology: {detection_method}\n\n")
                elif "Outage" in anomaly:
                    explanation = anomaly_definitions['Outages']
                    detection_method = anomaly_detection_methods['Outages'].format(
                        threshold_total_routes=anomaly_thresholds['Total Routes'],
                        observed_total_routes=row['Total Routes'],
                        mean_total_routes=means['Total Routes'],
                        std_total_routes=std_devs['Total Routes']
                    )
                    outages_anomalies.append(anomaly_log + f"Explanation: {explanation}\n\nDetection Methodology: {detection_method}\n\n")
                else:
                    # If the anomaly doesn't fit specific categories, you can create a general log
                    pass

    # Write Data Point Logs to File
    with open(os.path.join(output_dir, data_point_logs_filename), 'w', encoding='utf-8') as f:
        for log in data_point_logs:
            f.write(log)

    # Write Prefix Announcements to File in Narrative Format
    with open(os.path.join(output_dir, prefix_announcements_filename), 'w', encoding='utf-8') as f:
        for announcement in prefix_announcements:
            f.write(announcement + "\n\n")

    # Write Prefix Withdrawals to File in Narrative Format
    with open(os.path.join(output_dir, prefix_withdrawals_filename), 'w', encoding='utf-8') as f:
        for withdrawal in prefix_withdrawals:
            f.write(withdrawal + "\n\n")

    # Write Updates per Peer to File in Narrative Format
    with open(os.path.join(output_dir, updates_per_peer_filename), 'w', encoding='utf-8') as f:
        for updates in updates_per_peer_info:
            f.write(updates + "\n")

    # Write Anomaly Logs to Files in Narrative Format with Explanations and Detection Methodologies
    with open(os.path.join(output_dir, hijacking_filename), 'w', encoding='utf-8') as f:
        for log in hijacking_anomalies:
            f.write(log)

    with open(os.path.join(output_dir, leaks_filename), 'w', encoding='utf-8') as f:
        for log in leaks_anomalies:
            f.write(log)

    with open(os.path.join(output_dir, outages_filename), 'w', encoding='utf-8') as f:
        for log in outages_anomalies:
            f.write(log)

    # Generate Anomaly Summary in Narrative Format
    anomaly_summary = "Anomaly Summary:\n\n"
    total_anomalies = len(hijacking_anomalies) + len(leaks_anomalies) + len(outages_anomalies)
    if total_anomalies > 0:
        anomaly_summary += f"During the observation period, a total of {total_anomalies} anomalies were detected across various Autonomous Systems.\n\n"
        anomaly_summary += f"- Hijacking Anomalies: {len(hijacking_anomalies)} detected.\n"
        anomaly_summary += f"- Leaks Anomalies: {len(leaks_anomalies)} detected.\n"
        anomaly_summary += f"- Outages Anomalies: {len(outages_anomalies)} detected.\n\n"
        anomaly_summary += "Each anomaly has been documented with detailed explanations and detection methodologies, including the actual observed values, mean, standard deviations, and thresholds used to identify the anomalies."
    else:
        anomaly_summary += "No anomalies were detected during the observation period.\n"

    with open(os.path.join(output_dir, anomaly_summary_filename), 'w', encoding='utf-8') as f:
        f.write(anomaly_summary)
        
    print(f"Data processing complete. Output files are saved in the '{output_dir}' directory.")


def new_df_to_narrative_2(df, output=None):
    """
    Convert a DataFrame into a narrative text description, embedding maximum values into each data point.

    Parameters:
    df (pd.DataFrame): The DataFrame to convert.
    output (str, optional): Path to save the narrative text to a file.

    Returns:
    str: A narrative text description of the DataFrame content.
    """
    description = ""

    # Compute overall maximum values
    metrics = {
        'Announcements': 'announcements',
        'Withdrawals': 'withdrawals',
        'New Routes': 'new routes',
        'Origin Changes': 'origin changes',
        'Route Changes': 'route changes',
        'Total Routes': 'active routes',
        'Maximum Path Length': 'maximum path length',
        'Average Path Length': 'average path length',
        'Maximum Edit Distance': 'maximum edit distance',
        'Average Edit Distance': 'average edit distance',
        'Prefixes with AS Path Prepending': 'prefixes with AS path prepending',
        'Bogon Prefixes Detected': 'bogon prefixes detected',
        'Average Prefix Length': 'average prefix length',
        'Max Prefix Length': 'maximum prefix length',
        'Min Prefix Length': 'minimum prefix length',
        'Number of Unique Peers': 'unique peers'
    }

    max_values = {}
    for metric in metrics.keys():
        if metric in df.columns:
            max_values[metric] = df[metric].max()
        else:
            max_values[metric] = 'N/A'

    # Aggregate peer updates across all data points for maximum values
    total_peer_updates = {}
    peer_update_cols = [col for col in df.columns if col.startswith('Updates per Peer_')]

    if peer_update_cols:
        # Sum updates per peer over all data points
        for col in peer_update_cols:
            asn = col.replace('Updates per Peer_', '')
            total_updates = df[col].sum()
            total_peer_updates[asn] = total_updates

        # Get top peers overall
        sorted_total_peers = sorted(total_peer_updates.items(), key=lambda x: x[1], reverse=True)
        top_total_peers = sorted_total_peers[:5]  # Adjust as needed

    # Now proceed to generate the per-row narratives with embedded maximum values
    for index, row in df.iterrows():
        row_description = f"On {row['Timestamp']}, Autonomous System {row['Autonomous System Number']} observed "

        # Announcements and Withdrawals
        announcements = row['Announcements']
        withdrawals = row['Withdrawals'] if 'Withdrawals' in df.columns else 0
        row_description += f"{announcements} announcements"
        if withdrawals > 0:
            row_description += f" and {withdrawals} withdrawals"
        row_description += ". "

        # Embed maximum announcements and withdrawals
        row_description += f"The maximum number of announcements observed was {max_values.get('Announcements', 'N/A')}, "
        row_description += f"and the maximum number of withdrawals was {max_values.get('Withdrawals', 'N/A')}. "

        # New Routes
        if 'New Routes' in df.columns and row['New Routes'] > 0:
            row_description += f"There were {row['New Routes']} new routes added. "
        # Embed maximum new routes
        row_description += f"The maximum number of new routes added was {max_values.get('New Routes', 'N/A')}. "

        # Origin Changes
        if 'Origin Changes' in df.columns and row['Origin Changes'] > 0:
            row_description += f"{row['Origin Changes']} origin changes occurred. "
        # Embed maximum origin changes
        row_description += f"The maximum number of origin changes was {max_values.get('Origin Changes', 'N/A')}. "

        # Route Changes
        if 'Route Changes' in df.columns and row['Route Changes'] > 0:
            row_description += f"{row['Route Changes']} route changes were detected. "
        # Embed maximum route changes
        row_description += f"The maximum number of route changes detected was {max_values.get('Route Changes', 'N/A')}. "

        # Total Routes
        if 'Total Routes' in df.columns:
            row_description += f"The total number of active routes was {row['Total Routes']}. "
        # Embed maximum total routes
        row_description += f"The maximum number of active routes was {max_values.get('Total Routes', 'N/A')}. "

        # Path Lengths
        if 'Maximum Path Length' in df.columns and 'Average Path Length' in df.columns:
            row_description += (
                f"The maximum path length observed was {row['Maximum Path Length']} hops "
                f"(maximum overall: {max_values.get('Maximum Path Length', 'N/A')} hops), "
                f"with an average path length of {row['Average Path Length']:.2f} hops "
                f"(maximum overall average: {max_values.get('Average Path Length', 'N/A'):.2f} hops). "
            )

        # Edit Distances
        if 'Maximum Edit Distance' in df.columns and 'Average Edit Distance' in df.columns:
            row_description += (
                f"The maximum edit distance was {row['Maximum Edit Distance']} "
                f"(maximum overall: {max_values.get('Maximum Edit Distance', 'N/A')}), "
                f"with an average of {row['Average Edit Distance']:.2f} "
                f"(maximum overall average: {max_values.get('Average Edit Distance', 'N/A'):.2f}). "
            )

        # Prefixes with AS Path Prepending
        if 'Prefixes with AS Path Prepending' in df.columns:
            row_description += f"There were {row['Prefixes with AS Path Prepending']} prefixes with AS path prepending "
            row_description += f"(maximum overall: {max_values.get('Prefixes with AS Path Prepending', 'N/A')}). "

        # Bogon Prefixes Detected
        if 'Bogon Prefixes Detected' in df.columns:
            row_description += f"{row['Bogon Prefixes Detected']} bogon prefixes were detected "
            row_description += f"(maximum overall: {max_values.get('Bogon Prefixes Detected', 'N/A')}). "

        # Prefix Lengths
        if (
            'Average Prefix Length' in df.columns and
            'Max Prefix Length' in df.columns and
            'Min Prefix Length' in df.columns
        ):
            row_description += (
                f"The average prefix length was {row['Average Prefix Length']:.2f} "
                f"(maximum overall average: {max_values.get('Average Prefix Length', 'N/A'):.2f}), "
                f"with a maximum of {row['Max Prefix Length']} "
                f"(maximum overall: {max_values.get('Max Prefix Length', 'N/A')}) "
                f"and a minimum of {row['Min Prefix Length']} "
                f"(minimum overall: {max_values.get('Min Prefix Length', 'N/A')}). "
            )

        # Number of Unique Peers
        if 'Number of Unique Peers' in df.columns:
            row_description += f"The number of unique peers was {row['Number of Unique Peers']} "
            row_description += f"(maximum overall: {max_values.get('Number of Unique Peers', 'N/A')}). "

        # Updates per Peer
        if peer_update_cols:
            # Extract per-row peer updates
            peer_updates = {}
            for col in peer_update_cols:
                asn = col.replace('Updates per Peer_', '')
                count = row[col]
                if count > 0:
                    peer_updates[asn] = count

            if peer_updates:
                sorted_peers = sorted(peer_updates.items(), key=lambda x: x[1], reverse=True)
                top_peers = sorted_peers[:5]  # Adjust as needed
                peer_descriptions = [f"AS{asn} with {count} updates" for asn, count in top_peers]
                row_description += "The top peers by updates were: " + ", ".join(peer_descriptions) + ". "

                # Include overall top peers
                overall_peer_descriptions = [f"AS{asn} with {count} updates" for asn, count in top_total_peers]
                row_description += "The top peers by total updates overall were: " + ", ".join(overall_peer_descriptions) + ". "

        description += row_description.strip() + "\n\n"

    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as file:
            file.write(description.strip())

    return description.strip()


def new_df_to_narrative(df, output=None):
    """
    Convert a DataFrame into a narrative text description, starting with a narrative summary of maximum values.

    Parameters:
    df (pd.DataFrame): The DataFrame to convert.
    output (str, optional): Path to save the narrative text to a file.

    Returns:
    str: A narrative text description of the DataFrame content.
    """
    description = ""

    # Compute overall summary statistics
    metrics = {
        'Announcements': 'announcements',
        'Withdrawals': 'withdrawals',
        'New Routes': 'new routes',
        'Origin Changes': 'origin changes',
        'Route Changes': 'route changes',
        'Total Routes': 'active routes',
        'Maximum Path Length': 'maximum path length',
        'Average Path Length': 'average path length',
        'Maximum Edit Distance': 'maximum edit distance',
        'Average Edit Distance': 'average edit distance',
        'Prefixes with AS Path Prepending': 'prefixes with AS path prepending',
        'Bogon Prefixes Detected': 'bogon prefixes detected',
        'Average Prefix Length': 'average prefix length',
        'Max Prefix Length': 'maximum prefix length',
        'Min Prefix Length': 'minimum prefix length',
        'Number of Unique Peers': 'unique peers'
    }

    summary_values = {}
    for metric in metrics.keys():
        if metric in df.columns:
            summary_values[metric] = df[metric].max()

    # Aggregate peer updates across all data points
    total_peer_updates = {}
    # Collect all 'Updates per Peer_*' columns
    peer_update_cols = [col for col in df.columns if col.startswith('Updates per Peer_')]

    if peer_update_cols:
        # Sum updates per peer over all data points
        for col in peer_update_cols:
            asn = col.replace('Updates per Peer_', '')
            total_updates = df[col].sum()
            total_peer_updates[asn] = total_updates

        # Get top peers
        sorted_total_peers = sorted(total_peer_updates.items(), key=lambda x: x[1], reverse=True)
        top_total_peers = sorted_total_peers[:5]  # Adjust as needed

    # Build the overall narrative summary
    description += "Overall Summary:\n\n"
    description += f"During the observation period, Autonomous System {df['Autonomous System Number'].iloc[0]} experienced significant BGP activity. The maximum number of announcements observed was {summary_values.get('Announcements', 'N/A')}, and the maximum number of withdrawals was {summary_values.get('Withdrawals', 'N/A')}. There were up to {summary_values.get('New Routes', 'N/A')} new routes added, with {summary_values.get('Origin Changes', 'N/A')} origin changes and {summary_values.get('Route Changes', 'N/A')} route changes detected.\n\n"

    description += f"The AS maintained up to {summary_values.get('Total Routes', 'N/A')} active routes. The maximum path length observed was {summary_values.get('Maximum Path Length', 'N/A')} hops, with an average path length reaching {summary_values.get('Average Path Length', 'N/A'):.2f} hops. The maximum edit distance was {summary_values.get('Maximum Edit Distance', 'N/A')}, averaging {summary_values.get('Average Edit Distance', 'N/A'):.2f}.\n\n"

    description += f"There were {summary_values.get('Prefixes with AS Path Prepending', 'N/A')} prefixes with AS path prepending and {summary_values.get('Bogon Prefixes Detected', 'N/A')} bogon prefixes detected. The average prefix length peaked at {summary_values.get('Average Prefix Length', 'N/A'):.2f}, with prefix lengths ranging from {summary_values.get('Min Prefix Length', 'N/A')} to {summary_values.get('Max Prefix Length', 'N/A')}.\n\n"

    if total_peer_updates:
        description += f"The AS connected with up to {summary_values.get('Number of Unique Peers', 'N/A')} unique peers. The top peers by total updates were: "
        peer_descriptions = [f"AS{asn} with {count} updates" for asn, count in top_total_peers]
        description += ", ".join(peer_descriptions) + ".\n\n"
    else:
        description += "No peer updates were recorded.\n\n"

    description += "Detailed Narrative per Data Point:\n\n"

    # Now proceed to generate the per-row narratives
    for index, row in df.iterrows():
        row_description = f"On {row['Timestamp']}, Autonomous System {row['Autonomous System Number']} observed "

        # Announcements and Withdrawals
        announcements = row['Announcements']
        withdrawals = row['Withdrawals'] if 'Withdrawals' in df.columns else 0
        row_description += f"{announcements} announcements"
        if withdrawals > 0:
            row_description += f" and {withdrawals} withdrawals"
        row_description += ". "

        # New Routes
        if 'New Routes' in df.columns and row['New Routes'] > 0:
            row_description += f"There were {row['New Routes']} new routes added. "

        # Origin Changes
        if 'Origin Changes' in df.columns and row['Origin Changes'] > 0:
            row_description += f"{row['Origin Changes']} origin changes occurred. "

        # Route Changes
        if 'Route Changes' in df.columns and row['Route Changes'] > 0:
            row_description += f"{row['Route Changes']} route changes were detected. "

        # Total Routes
        if 'Total Routes' in df.columns:
            row_description += f"The total number of active routes was {row['Total Routes']}. "

        # Path Lengths
        if 'Maximum Path Length' in df.columns and 'Average Path Length' in df.columns:
            row_description += (
                f"The maximum path length observed was {row['Maximum Path Length']} hops, "
                f"with an average path length of {row['Average Path Length']:.2f} hops. "
            )

        # Edit Distances
        if 'Maximum Edit Distance' in df.columns and 'Average Edit Distance' in df.columns:
            row_description += (
                f"The maximum edit distance was {row['Maximum Edit Distance']}, "
                f"with an average of {row['Average Edit Distance']:.2f}. "
            )

        # Graph Metrics
        if 'Graph Average Degree' in df.columns:
            row_description += f"The graph average degree was {row['Graph Average Degree']:.2f}. "
        if 'Graph Betweenness Centrality' in df.columns:
            row_description += f"The graph betweenness centrality was {row['Graph Betweenness Centrality']:.4f}. "
        if 'Graph Closeness Centrality' in df.columns:
            row_description += f"The graph closeness centrality was {row['Graph Closeness Centrality']:.4f}. "
        if 'Graph Eigenvector Centrality' in df.columns:
            row_description += f"The graph eigenvector centrality was {row['Graph Eigenvector Centrality']:.4f}. "

        # MED and Local Preference
        if 'Average MED' in df.columns:
            row_description += f"The average MED was {row['Average MED']:.2f}. "
        if 'Average Local Preference' in df.columns:
            row_description += f"The average local preference was {row['Average Local Preference']:.2f}. "

        # Communities
        if 'Total Communities' in df.columns and 'Unique Communities' in df.columns:
            row_description += (
                f"There were {row['Total Communities']} total communities observed, "
                f"with {row['Unique Communities']} unique communities. "
            )

        # Number of Unique Peers
        if 'Number of Unique Peers' in df.columns:
            row_description += f"The number of unique peers was {row['Number of Unique Peers']}. "

        # Prefixes with AS Path Prepending
        if 'Prefixes with AS Path Prepending' in df.columns and row['Prefixes with AS Path Prepending'] > 0:
            row_description += f"{row['Prefixes with AS Path Prepending']} prefixes showed AS path prepending. "

        # Bogon Prefixes Detected
        if 'Bogon Prefixes Detected' in df.columns and row['Bogon Prefixes Detected'] > 0:
            row_description += f"{row['Bogon Prefixes Detected']} bogon prefixes were detected. "

        # Prefix Lengths
        if (
            'Average Prefix Length' in df.columns and
            'Max Prefix Length' in df.columns and
            'Min Prefix Length' in df.columns
        ):
            row_description += (
                f"The average prefix length was {row['Average Prefix Length']:.2f}, "
                f"with a maximum of {row['Max Prefix Length']} and a minimum of {row['Min Prefix Length']}. "
            )

        # Updates per Peer
        peer_update_cols = [col for col in df.columns if col.startswith('Updates per Peer_')]
        if peer_update_cols:
            # Extract per-row peer updates
            peer_updates = {}
            for col in peer_update_cols:
                asn = col.replace('Updates per Peer_', '')
                count = row[col]
                if count > 0:
                    peer_updates[asn] = count

            if peer_updates:
                sorted_peers = sorted(peer_updates.items(), key=lambda x: x[1], reverse=True)
                top_peers = sorted_peers[:5]  # Adjust as needed
                peer_descriptions = [f"AS{asn} with {count} updates" for asn, count in top_peers]
                row_description += "The top peers by updates were: " + ", ".join(peer_descriptions) + ". "

        description += row_description.strip() + "\n\n"

    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as file:
            file.write(description.strip())

    return description.strip()


def df_to_narrative(df, output=None):
    """
    Convert a DataFrame into a narrative text description.

    Parameters:
    df (pd.DataFrame): The DataFrame to convert.

    Returns:
    str: A narrative text description of the DataFrame content.
    """
    description = ""

    for index, row in df.iterrows():
        row_description = (
            f"On {row['Timestamp']}, Autonomous System {row['Autonomous System Number']} observed "
            f"{row['Announcements']} announcements"
        )
        if 'Withdrawals' in df.columns and row['Withdrawals'] > 0:
            row_description += f" and {row['Withdrawals']} withdrawals"
        row_description += ". "

        if 'New Routes' in df.columns and row['New Routes'] > 0:
            row_description += f"There were {row['New Routes']} new routes added. "

        if 'Origin Changes' in df.columns and row['Origin Changes'] > 0:
            row_description += f"{row['Origin Changes']} origin changes occurred. "

        if 'Route Changes' in df.columns and row['Route Changes'] > 0:
            row_description += f"{row['Route Changes']} route changes were detected. "

        if 'Total Routes' in df.columns:
            row_description += f"The total number of active routes was {row['Total Routes']}. "

        if 'Maximum Path Length' in df.columns:
            row_description += f"The maximum path length observed was {row['Maximum Path Length']} hops, "

        if 'Average Path Length' in df.columns:
            row_description += f"with an average path length of {row['Average Path Length']} hops. "

        description += row_description + "\n"

    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as file:
            file.write(description.strip())

    return description.strip()


def df_to_json_format(df, output):
    """
    Convert a DataFrame into a JSON format suitable for embedding models.

    Parameters:
    df (pd.DataFrame): The DataFrame to convert.
    output (str): The path to save the output JSON file.

    Returns:
    str: A JSON string representation of the DataFrame content.
    """
    records = []
    for index, row in df.iterrows():
        record = {
            "Record ID": index + 1,
            "Timestamp": str(row['Timestamp']),
            "Autonomous System Number": f"AS{row['Autonomous System Number']}",
            "Announcements": f"{row['Announcements']} messages",
        }

        if 'Withdrawals' in df.columns:
            record["Withdrawals"] = f"{row.get('Withdrawals', 0)} messages"
        if 'New Routes' in df.columns:
            record["New Routes"] = f"{row.get('New Routes', 0)} routes"
        if 'Origin Changes' in df.columns:
            record["Origin Changes"] = f"{row.get('Origin Changes', 0)} changes"
        if 'Route Changes' in df.columns:
            record["Route Changes"] = f"{row.get('Route Changes', 0)} changes"
        if 'Total Routes' in df.columns:
            record["Total Routes"] = f"{row.get('Total Routes', 0)} routes"
        if 'Maximum Path Length' in df.columns:
            record["Maximum Path Length"] = f"{row.get('Maximum Path Length', 0)} hops"
        if 'Average Path Length' in df.columns:
            record["Average Path Length"] = f"{row.get('Average Path Length', 0)} hops"
        if 'Maximum Edit Distance' in df.columns:
            record["Maximum Edit Distance"] = row.get('Maximum Edit Distance', 0)
        if 'Average Edit Distance' in df.columns:
            record["Average Edit Distance"] = row.get('Average Edit Distance', 0)
        if 'Unique Prefixes Announced' in df.columns:
            record["Unique Prefixes Announced"] = f"{row.get('Unique Prefixes Announced', 0)} prefixes"
        if 'Graph Average Degree' in df.columns:
            record["Graph Average Degree"] = row.get('Graph Average Degree', 0)
        if 'Graph Betweenness Centrality' in df.columns:
            record["Graph Betweenness Centrality"] = row.get('Graph Betweenness Centrality', 0)
        if 'Graph Closeness Centrality' in df.columns:
            record["Graph Closeness Centrality"] = row.get('Graph Closeness Centrality', 0)
        if 'Graph Eigenvector Centrality' in df.columns:
            record["Graph Eigenvector Centrality"] = row.get('Graph Eigenvector Centrality', 0)

        records.append(record)

    # Convert the list of records to JSON string
    json_data = json.dumps(records, indent=2)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output), exist_ok=True)

    # Save the JSON string to the output file
    with open(output, "w", encoding="utf-8") as file:
        file.write(json_data)

    return json_data
