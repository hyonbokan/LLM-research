import os
import pandas as pd
import json
from transformers import pipeline
import csv
from collections import defaultdict, Counter
import ast

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

import pandas as pd
import numpy as np

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
    data_point_summaries_filename='data_point_summaries.txt',
    prefix_announcements_filename='prefix_announcements.txt',
    prefix_withdrawals_filename='prefix_withdrawals.txt',
    updates_per_peer_filename='updates_per_peer.txt',
    top_n_prefixes=10,  # Number of top prefixes to include in the overall summary
    top_n_peers=10      # Number of top peers to include in the overall summary
):
    """
    Processes a BGP CSV file and generates enriched text files for:
    1. Overall summary with min and max values, top peers, and frequent prefixes.
    2. Textual summaries of each data point.
    3. Prefix announcement information.
    4. Prefix withdrawal information.
    5. Updates per peer information.

    Args:
        csv_file_path (str): Path to the input CSV file.
        output_dir (str): Directory where output files will be saved.
        overall_summary_filename (str): Filename for overall summary.
        data_point_summaries_filename (str): Filename for data point summaries.
        prefix_announcements_filename (str): Filename for prefix announcements.
        prefix_withdrawals_filename (str): Filename for prefix withdrawals.
        updates_per_peer_filename (str): Filename for updates per peer.
        top_n_prefixes (int): Number of top prefixes to include in the overall summary.
        top_n_peers (int): Number of top peers to include in the overall summary.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize variables for overall summary
    min_values = {}
    max_values = {}

    # Dictionaries to store overall peer updates and prefix counts
    total_updates_per_peer = defaultdict(float)
    prefix_announcement_counter = Counter()
    prefix_withdrawal_counter = Counter()

    # Lists to store textual summaries and other information
    data_point_summaries = []
    prefix_announcements = []
    prefix_withdrawals = []
    updates_per_peer_info = []

    # Read the CSV file
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

        # Get list of peers from column names
        peer_columns = [col for col in reader.fieldnames if col.startswith('Updates per Peer_')]

        # Initialize min and max dictionaries
        numeric_columns = [
            'Total Routes', 'New Routes', 'Withdrawals', 'Origin Changes', 'Route Changes',
            'Maximum Path Length', 'Average Path Length', 'Maximum Edit Distance', 'Average Edit Distance',
            'Announcements', 'Unique Prefixes Announced', 'Graph Average Degree',
            'Graph Betweenness Centrality', 'Graph Closeness Centrality', 'Graph Eigenvector Centrality',
            'Average MED', 'Average Local Preference', 'Total Communities', 'Unique Communities',
            'Number of Unique Peers',
            'Prefixes with AS Path Prepending', 'Bogon Prefixes Detected', 'Average Prefix Length',
            'Max Prefix Length', 'Min Prefix Length'
        ]

        for col in numeric_columns:
            min_values[col] = float('inf')
            max_values[col] = float('-inf')

        # Process each row
        for row in rows:
            timestamp = row['Timestamp']
            as_number = row['Autonomous System Number']

            # Update min and max values
            for col in numeric_columns:
                try:
                    value = float(row[col]) if row[col] != '' else 0.0
                except ValueError:
                    value = 0.0  # Handle non-numeric values
                if value < min_values[col]:
                    min_values[col] = value
                if value > max_values[col]:
                    max_values[col] = value

            # Handle 'Prefixes Announced' and 'Prefixes Withdrawn' as counts
            prefixes_announced_str = row.get('Prefixes Announced', '[]')
            prefixes_withdrawn_str = row.get('Prefixes Withdrawn', '[]')
            try:
                prefixes_announced_list = ast.literal_eval(prefixes_announced_str)
            except (SyntaxError, ValueError):
                prefixes_announced_list = []
            try:
                prefixes_withdrawn_list = ast.literal_eval(prefixes_withdrawn_str)
            except (SyntaxError, ValueError):
                prefixes_withdrawn_list = []
            num_prefixes_announced = len(prefixes_announced_list)
            num_prefixes_withdrawn = len(prefixes_withdrawn_list)

            # Update min and max for prefixes announced and withdrawn
            for col, value in [('Prefixes Announced', num_prefixes_announced), ('Prefixes Withdrawn', num_prefixes_withdrawn)]:
                if value < min_values.get(col, float('inf')):
                    min_values[col] = value
                if value > max_values.get(col, float('-inf')):
                    max_values[col] = value

            # Update prefix counters for frequent prefixes
            prefix_announcement_counter.update(prefixes_announced_list)
            prefix_withdrawal_counter.update(prefixes_withdrawn_list)

            # Generate textual summary for each data point
            data_point_summary = (
                f"On {timestamp}, Autonomous System {as_number} observed {row['Announcements']} announcements. "
                f"There were {row['New Routes']} new routes added. "
                f"The total number of active routes was {row['Total Routes']}. "
                f"The maximum path length observed was {row['Maximum Path Length']} hops, "
                f"with an average path length of {row['Average Path Length']} hops. "
                f"The maximum edit distance was {row['Maximum Edit Distance']}, "
                f"with an average of {row['Average Edit Distance']}. "
                f"The graph average degree was {row['Graph Average Degree']}. "
                f"The graph betweenness centrality was {row['Graph Betweenness Centrality']}. "
                f"The graph closeness centrality was {row['Graph Closeness Centrality']}. "
                f"The graph eigenvector centrality was {row['Graph Eigenvector Centrality']}. "
                f"The average MED was {row['Average MED']}. "
                f"The average local preference was {row['Average Local Preference']}. "
                f"There were {row['Total Communities']} total communities observed, "
                f"with {row['Unique Communities']} unique communities. "
                f"The number of unique peers was {row['Number of Unique Peers']}. "
                f"The average prefix length was {row['Average Prefix Length']}, "
                f"with a maximum of {row['Max Prefix Length']} and a minimum of {row['Min Prefix Length']}. "
                f"Number of prefixes announced: {num_prefixes_announced}. "
                f"Number of prefixes withdrawn: {num_prefixes_withdrawn}."
            )
            data_point_summaries.append(data_point_summary)

            # Collect enriched prefix announcement information
            if prefixes_announced_list:
                prefixes_str = ', '.join(prefixes_announced_list)
                announcement = (
                    f"At {timestamp}, the following prefixes were announced by AS{as_number}: {prefixes_str}. "
                    f"These prefixes contributed to a total of {num_prefixes_announced} announcements during this period."
                )
                prefix_announcements.append(announcement)

            # Collect enriched prefix withdrawal information
            if prefixes_withdrawn_list:
                prefixes_str = ', '.join(prefixes_withdrawn_list)
                withdrawal = (
                    f"At {timestamp}, the following prefixes were withdrawn by AS{as_number}: {prefixes_str}. "
                    f"These prefixes accounted for {num_prefixes_withdrawn} withdrawals during this period."
                )
                prefix_withdrawals.append(withdrawal)

            # Collect updates per peer information and sum total updates per peer
            updates_info = f"At {timestamp}, updates per peer were as follows:\n"
            for peer_col in peer_columns:
                peer_asn = peer_col.split('_')[1]
                updates_str = row[peer_col]
                try:
                    updates = float(updates_str) if updates_str != '' else 0.0
                except ValueError:
                    updates = 0.0
                total_updates_per_peer[peer_asn] += updates
                updates_info += f"  Peer AS{peer_asn} had {updates} updates.\n"
            updates_per_peer_info.append(updates_info)

        # Generate enriched overall summary
        overall_summary_text = "Overall Summary:\n\n"

        # Include min and max values with descriptions
        overall_summary_text += "The following are the minimum and maximum values observed across various metrics:\n"
        for col in numeric_columns + ['Prefixes Announced', 'Prefixes Withdrawn']:
            min_val = min_values.get(col, 0.0)
            max_val = max_values.get(col, 0.0)
            overall_summary_text += (
                f"- {col}: Minimum observed was {min_val}, and the maximum observed was {max_val}.\n"
            )

        # Add top peers by total updates with descriptive language
        overall_summary_text += f"\nTop {top_n_peers} Peers with the Highest Number of Updates:\n"
        # Sort the peers by total updates in descending order
        sorted_peers = sorted(total_updates_per_peer.items(), key=lambda x: x[1], reverse=True)[:top_n_peers]
        for rank, (peer_asn, total_updates) in enumerate(sorted_peers, start=1):
            overall_summary_text += (
                f"{rank}. Peer AS{peer_asn} had a total of {total_updates} updates, "
                f"making it one of the most active peers.\n"
            )

        # Add most frequent prefixes announced with descriptive language
        overall_summary_text += f"\nTop {top_n_prefixes} Most Frequently Announced Prefixes:\n"
        for rank, (prefix, count) in enumerate(prefix_announcement_counter.most_common(top_n_prefixes), start=1):
            overall_summary_text += (
                f"{rank}. Prefix {prefix} was announced {count} times, "
                f"making it one of the most frequently announced prefixes.\n"
            )

        # Add most frequent prefixes withdrawn with descriptive language
        overall_summary_text += f"\nTop {top_n_prefixes} Most Frequently Withdrawn Prefixes:\n"
        for rank, (prefix, count) in enumerate(prefix_withdrawal_counter.most_common(top_n_prefixes), start=1):
            overall_summary_text += (
                f"{rank}. Prefix {prefix} was withdrawn {count} times, "
                f"indicating significant routing changes.\n"
            )

        # Write the enriched overall summary to a text file
        with open(os.path.join(output_dir, overall_summary_filename), 'w', encoding='utf-8') as f:
            f.write(overall_summary_text)

        # Write textual summaries of each data point to a text file
        with open(os.path.join(output_dir, data_point_summaries_filename), 'w', encoding='utf-8') as f:
            for summary in data_point_summaries:
                f.write(summary + '\n\n')

        # Write enriched prefix announcement information to a text file
        with open(os.path.join(output_dir, prefix_announcements_filename), 'w', encoding='utf-8') as f:
            for announcement in prefix_announcements:
                f.write(announcement + '\n\n')

        # Write enriched prefix withdrawal information to a text file
        with open(os.path.join(output_dir, prefix_withdrawals_filename), 'w', encoding='utf-8') as f:
            for withdrawal in prefix_withdrawals:
                f.write(withdrawal + '\n\n')

        # Write enriched updates per peer information to a text file
        with open(os.path.join(output_dir, updates_per_peer_filename), 'w', encoding='utf-8') as f:
            for updates in updates_per_peer_info:
                f.write(updates + '\n')

        print(f"Data processing complete. Output files are saved in the '{output_dir}' directory.")


def df_to_document_list(df, output=None):
    """
    Convert a DataFrame into a list of plain text descriptions suitable for embedding models.

    Parameters:
    df (pd.DataFrame): The DataFrame to convert.
    output (str, optional): The path to save the output text file containing all documents.

    Returns:
    List[str]: A list of plain text descriptions of the DataFrame content.
    """
    documents = []

    for index, row in df.iterrows():
        row_description = f"At {row['Timestamp']}, AS{row['Autonomous System Number']} observed {row['Announcements']} announcements"
        
        if 'Withdrawals' in df.columns and row['Withdrawals'] > 0:
            row_description += f" and {row['Withdrawals']} withdrawals"
        
        if 'New Routes' in df.columns and row['New Routes'] > 0:
            row_description += f". There were {row['New Routes']} new routes added"
        
        if 'Origin Changes' in df.columns and row['Origin Changes'] > 0:
            row_description += f", with {row['Origin Changes']} origin changes"
        
        if 'Route Changes' in df.columns and row['Route Changes'] > 0:
            row_description += f" and {row['Route Changes']} route changes"
        
        if 'Total Routes' in df.columns:
            row_description += f". A total of {row['Total Routes']} routes were active"

        if 'Maximum Path Length' in df.columns:
            row_description += f", with a maximum path length of {row['Maximum Path Length']}"
        
        if 'Average Path Length' in df.columns:
            row_description += f" and an average path length of {row['Average Path Length']}"
        
        if 'Maximum Edit Distance' in df.columns:
            row_description += f". The maximum edit distance observed was {row['Maximum Edit Distance']}"
        
        if 'Average Edit Distance' in df.columns:
            row_description += f" with an average edit distance of {row['Average Edit Distance']}"
        
        if 'Unique Prefixes Announced' in df.columns:
            row_description += f". There were {row['Unique Prefixes Announced']} unique prefixes announced"

        # Add graph-related features if available
        if 'Graph Average Degree' in df.columns:
            row_description += f". The graph's average degree was {row['Graph Average Degree']}"
        
        if 'Graph Betweenness Centrality' in df.columns:
            row_description += f", betweenness centrality was {row['Graph Betweenness Centrality']}"
        
        if 'Graph Closeness Centrality' in df.columns:
            row_description += f", closeness centrality was {row['Graph Closeness Centrality']}"
        
        if 'Graph Eigenvector Centrality' in df.columns:
            row_description += f", and eigenvector centrality was {row['Graph Eigenvector Centrality']}"
        
        row_description += "."
        documents.append(row_description)
    
    # Optionally save all documents to a single text file
    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as file:
            for doc in documents:
                file.write(doc + "\n\n")
                    
    return documents


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


def df_to_narrative_with_delimiters(df, output=None):
    """
    Convert a DataFrame into a narrative text description with clear delimiters.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to convert.
    
    Returns:
    str: A narrative text description of the DataFrame content.
    """
    description = "<BEGIN_CONTEXT>\n"
    
    for index, row in df.iterrows():
        row_description = (
            f"Record {index+1}:\n"
            f"Timestamp: {row['Timestamp']}\n"
            f"Autonomous System Number: {row['Autonomous System Number']}\n"
            f"Announcements: {row['Announcements']}\n"
        )
        if 'Withdrawals' in df.columns:
            row_description += f"Withdrawals: {row['Withdrawals']}\n"
        if 'New Routes' in df.columns:
            row_description += f"New Routes: {row['New Routes']}\n"
        if 'Origin Changes' in df.columns:
            row_description += f"Origin Changes: {row['Origin Changes']}\n"
        if 'Route Changes' in df.columns:
            row_description += f"Route Changes: {row['Route Changes']}\n"
        if 'Total Routes' in df.columns:
            row_description += f"Total Routes: {row['Total Routes']}\n"
        if 'Maximum Path Length' in df.columns:
            row_description += f"Maximum Path Length: {row['Maximum Path Length']} hops\n"
        if 'Average Path Length' in df.columns:
            row_description += f"Average Path Length: {row['Average Path Length']} hops\n"
        if 'Unique Prefixes Announced' in df.columns:
            row_description += f"Unique Prefixes Announced: {row['Unique Prefixes Announced']}\n"
        
        description += row_description + "\n"
    
    description += "<END_CONTEXT>"
    
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
