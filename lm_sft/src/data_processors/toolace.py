import json, os, sys
import pandas as pd
from typing import List, Dict, Any, Optional
from types import SimpleNamespace

# Add project root to sys.path to allow imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config_manager import load_config

SAMPLE_SIZE = 1000

def extract_functions_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract function definitions from a text containing JSON-formatted function descriptions.
    
    Args:
        text (str): The input text containing function definitions.
    
    Returns:
        list: A list of dictionaries containing function information in original format.
    """
    functions = []
    brace_count = 0
    current_block = ""
    in_function = False
    i = 0
    while i < len(text):
        char = text[i]
        if char == '{' and not in_function:
            remaining = text[i:i+20]
            if '"name"' in remaining:
                in_function = True
                brace_count = 1
                current_block = char
            else:
                current_block += char
        elif in_function:
            current_block += char
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    try:
                        cleaned_block = current_block.strip().rstrip(',')
                        function_data = json.loads(cleaned_block)
                        if 'name' in function_data and 'description' in function_data:
                            functions.append(function_data)
                    except json.JSONDecodeError:
                        pass  # Skip invalid JSON
                    in_function = False
                    current_block = ""
        else:
            current_block += char
        i += 1
    return functions

def process_toolace_data(file_path: str) -> pd.DataFrame:
    """
    Loads and processes the ToolACE dataset from a JSON file.

    Args:
        file_path (str): The path to the data.json file.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    dataset = json.load(open(file_path, 'r'))
    processed_data = [row for row in dataset[:SAMPLE_SIZE]]
    df = pd.DataFrame(processed_data)
    df['system'] = df['system'].apply(extract_functions_from_text)
    return df

def convert_to_final_format(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Converts the processed DataFrame to the final list format.

    Args:
        df (pd.DataFrame): The processed DataFrame.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries with 'tool' and 'messages' keys.
    """
    alist = []
    for _, row in df.iterrows():
        # Convert the list of functions to a JSON string to avoid Parquet conversion issues
        tool_data = json.dumps(row["system"]) if row["system"] else "[]"
        alist.append({
            "tool": tool_data,
            "messages": [{'role': turn['from'], 'content': turn['value']} for turn in row['conversations']]
        })
    return alist

def process_and_save_toolace(
    dataset_path: str,
    output_path: str
) -> Optional[List[Dict[str, Any]]]:
    """
    Main pipeline to load, process, and save the ToolACE dataset.

    Args:
        dataset_path (str): Path to the raw JSON data file.
        output_path (str): Path to save the processed Parquet file.

    Returns:
        Optional[List[Dict[str, Any]]: The processed data, or None if saving fails.
    """
    print("Starting ToolACE data processing...")
    
    df = process_toolace_data(dataset_path)
    print(f"Successfully loaded and processed {len(df)} conversations.")

    final_data = convert_to_final_format(df)
    
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    print(f"Attempting to save to {output_path}...")
    try:
        pd.DataFrame(final_data).to_parquet(output_path, index=False)
        print("Successfully saved as Parquet file.")
    except ImportError:
        print("Error: 'pyarrow' is required to save to Parquet format. Please run 'pip install pyarrow'.")
        return None

    return final_data

def main():
    """
    Main execution function to load configuration and run the data processing pipeline.
    """
    print("--- Starting Data Processing Pipeline for Team-ACE/ToolACE ---")
    
    configs = load_config()
    
    # Update these paths in your config file or here directly
    DATASET = configs.source.data.input_path +"/Team-ACE_ToolACE/data.json"
    OUTPUT = configs.source.data.output_path + "/Team-ACE/ToolACE.parquet"

    print(f"\nConfiguration:")
    print(f"  Input Dataset: {DATASET}")
    print(f"  Output File:   {OUTPUT}")

    try:
        processed_data = process_and_save_toolace(
            dataset_path=DATASET,
            output_path=OUTPUT
        )

        if processed_data is not None:
            print("--- Processed Data Head ---")
            print(processed_data[0])

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"--- A processing error occurred ---")
        print(f"Error: {e}")
    except Exception as e:
        print(f"--- An unexpected error occurred ---")
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

