import json, os, sys
import pandas as pd
from typing import List, Dict, Any, Optional
from types import SimpleNamespace

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
from src.config_manager import load_config

SAMPLE_SIZE = 50000 

def convert_dataframe_to_messages(df):
    """
    Convert DataFrame with messages column to messages format
    
    Args:
        df: DataFrame with messages column
        
    Returns:
        List of dictionaries with "messages" key
    """
    results = []

    for idx, row in df.iterrows():
        messages = []
        for message in row["messages"]:
            messages.append(message)
        results.append({"messages": messages})
    return results

def process_raw_conversations(dataset_path: str) -> pd.DataFrame:
    """
    Loads conversation data from multiple Parquet files and processes it into a pandas DataFrame.

    Args:
        dataset_path (str): The path to the directory containing the Parquet files.

    Returns:
        pd.DataFrame: A DataFrame with columns corresponding to speakers.
    
    Raises:
        FileNotFoundError: If the dataset_path does not exist.
        ValueError: If the dataset is not in the expected format.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Error: The directory was not found at {dataset_path}")

    dfs = []
    for seq in range(6):
        file_path = f'{dataset_path}/train-0000{seq}-of-00006.parquet'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: The file was not found at {file_path}")
        df_part = pd.read_parquet(file_path)
        dfs.append(df_part)
    
    df = pd.concat(dfs, ignore_index=True)

    return df

def process_and_save_conversations(
    dataset_path: str,
    output_path: str
) -> Optional[pd.DataFrame]:
    """
    Main pipeline function to load, process, format, and save conversation data.

    Args:
        dataset_path (str): Path to the raw JSON conversation data.
        output_path (str): Path to save the processed file (supports .csv and .parquet).

    Returns:
        Optional[pd.DataFrame]: The processed DataFrame, or None if saving fails.
    """
    print("Starting data processing...")
    
    # Step 1: Load and process the raw data
    df = process_raw_conversations(dataset_path).iloc[:SAMPLE_SIZE]
    print(f"Successfully loaded and processed {len(df)} conversations.")

    # Step 2: Save the processed data based on the output file extension
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    file_extension = os.path.splitext(output_path)[1].lower()
    json_output = convert_dataframe_to_messages(df)
    
    print(f"Attempting to save to {output_path}...")
    if file_extension == '.parquet':
        try:
            pd.DataFrame(json_output).to_parquet(output_path, index=False)
            print("Successfully saved as Parquet file.")
        except ImportError:
            print("Error: 'pyarrow' is required to save to Parquet format. Please run 'pip install pyarrow'.")
            return None
    elif file_extension == '.csv':
        pd.DataFrame(json_output).to_csv(output_path, index=False)
        print("Successfully saved as CSV file.")
    else:
        print(f"Warning: Unknown file extension '{file_extension}'. File not saved.")
        return None

    return json_output

def main():
    """
    Main execution function to load configuration and run the data processing pipeline.
    This function is executed when the script is run directly.
    """
    print("--- Starting Data Processing Pipeline for allenai/tulu-3-sft-mixture ---")
    
    configs = load_config()
    
    DATASET = configs.source.data.input_path + "/allenai_tulu-3-sft-mixture/data"
    OUTPUT = configs.source.data.output_path + "/allenai/tulu-3-sft-mixture.parquet"

    print(f"\nConfiguration:")
    print(f"  Input Dataset: {DATASET}")
    print(f"  Output File:   {OUTPUT}")

    try:
        json_output = process_and_save_conversations(
            dataset_path=DATASET,
            output_path=OUTPUT
        )

        if json_output is not None:
            print("--- Processed Data Head ---")
            print(json_output[0])

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"--- A processing error occurred ---")
        print(f"Error: {e}")
    except Exception as e:
        print(f"--- An unexpected error occurred ---")
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
