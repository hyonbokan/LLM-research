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
            os.makedirs(os.path.dirname(output), exist_ok=True)
            with open(output, "w", encoding="utf-8") as file:
                file.write(description.strip())
                
    return description.strip()

import os

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


def df_to_distinctive_text(df, output=None):
    """
    Convert a DataFrame into text with enhanced distinctiveness for embeddings.

    Parameters:
    df (pd.DataFrame): The DataFrame to convert.

    Returns:
    str: A text representation designed for better embeddings.
    """
    texts = []
    for index, row in df.iterrows():
        text = (
            f"Record ID: {index}\n"
            f"Timestamp: {row['Timestamp']}\n"
            f"AS Number: {row['Autonomous System Number']}\n"
            f"Announcements: {row['Announcements']}\n"
            f"Withdrawals: {row.get('Withdrawals', 0)}\n"
            f"New Routes: {row.get('New Routes', 0)}\n"
            f"Origin Changes: {row.get('Origin Changes', 0)}\n"
            f"Route Changes: {row.get('Route Changes', 0)}\n"
            f"Total Routes: {row.get('Total Routes', 0)}\n"
            f"Max Path Length: {row.get('Maximum Path Length', 0)}\n"
            f"Avg Path Length: {row.get('Average Path Length', 0)}\n"
        )
        texts.append(text)
    
    full_text = "\n".join(texts)
    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as file:
            file.write(full_text)
    return full_text

import os

def df_to_structured_text(df, output):
    """
    Convert a DataFrame into a structured text format suitable for embedding models.

    Parameters:
    df (pd.DataFrame): The DataFrame to convert.
    output (str): The path to save the output text file.

    Returns:
    str: A structured text representation of the DataFrame content.
    """
    texts = []
    for index, row in df.iterrows():
        record_id = index + 1  # Unique identifier for each record
        row_description = f"## RECORD {record_id} ##\n"
        row_description += f"Timestamp: {row['Timestamp']}\n"
        row_description += f"Autonomous System Number: AS{row['Autonomous System Number']}\n"
        row_description += f"Announcements: {row['Announcements']} messages\n"
        
        if 'Withdrawals' in df.columns:
            row_description += f"Withdrawals: {row.get('Withdrawals', 0)} messages\n"
        if 'New Routes' in df.columns:
            row_description += f"New Routes: {row.get('New Routes', 0)} routes\n"
        if 'Origin Changes' in df.columns:
            row_description += f"Origin Changes: {row.get('Origin Changes', 0)} changes\n"
        if 'Route Changes' in df.columns:
            row_description += f"Route Changes: {row.get('Route Changes', 0)} changes\n"
        if 'Total Routes' in df.columns:
            row_description += f"Total Routes: {row.get('Total Routes', 0)} routes\n"
        if 'Maximum Path Length' in df.columns:
            row_description += f"Maximum Path Length: {row.get('Maximum Path Length', 0)} hops\n"
        if 'Average Path Length' in df.columns:
            row_description += f"Average Path Length: {row.get('Average Path Length', 0)} hops\n"
        if 'Maximum Edit Distance' in df.columns:
            row_description += f"Maximum Edit Distance: {row.get('Maximum Edit Distance', 0)}\n"
        if 'Average Edit Distance' in df.columns:
            row_description += f"Average Edit Distance: {row.get('Average Edit Distance', 0)}\n"
        if 'Unique Prefixes Announced' in df.columns:
            row_description += f"Unique Prefixes Announced: {row.get('Unique Prefixes Announced', 0)} prefixes\n"
        if 'Graph Average Degree' in df.columns:
            row_description += f"Graph Average Degree: {row.get('Graph Average Degree', 0)}\n"
        if 'Graph Betweenness Centrality' in df.columns:
            row_description += f"Graph Betweenness Centrality: {row.get('Graph Betweenness Centrality', 0)}\n"
        if 'Graph Closeness Centrality' in df.columns:
            row_description += f"Graph Closeness Centrality: {row.get('Graph Closeness Centrality', 0)}\n"
        if 'Graph Eigenvector Centrality' in df.columns:
            row_description += f"Graph Eigenvector Centrality: {row.get('Graph Eigenvector Centrality', 0)}\n"
        
        texts.append(row_description)

    # Combine all records into one text
    full_text = "\n".join(texts)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    # Save the full text to the output file
    with open(output, "w", encoding="utf-8") as file:
        file.write(full_text.strip())
    
    return full_text.strip()

import os
import json

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
