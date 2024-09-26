import os
import pandas as pd
import json
from transformers import pipeline

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
    

def df_to_plain_text_description(df, output=None):
    """
    Convert a DataFrame into a plain text description.

    Parameters:
    df (pd.DataFrame): The DataFrame to convert.

    Returns:
    str: A plain text description of the DataFrame content.
    """
    description = ""

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
        description += row_description + "\n"
        
        if output:
            with open(output, "w", encoding="utf-8") as file:
                file.write(description.strip())
                
    return description.strip()