import json, os, sys
import pandas as pd
from typing import List, Dict, Any, Optional
from types import SimpleNamespace

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
from src.config_manager import load_config
from src.data_processors.data_utils import convert_dataframe_to_messages

SAMPLE_SIZE = 1000  


def process_raw_conversations(dataset_path: str) -> pd.DataFrame:
    """
    Loads conversation data from a JSON file and processes it into a pandas DataFrame.

    Args:
        dataset_path (str): The full path to the input JSON dataset file.

    Returns:
        pd.DataFrame: A DataFrame with columns corresponding to speakers.
    
    Raises:
        FileNotFoundError: If the dataset_path does not exist.
        ValueError: If the dataset is not in the expected format.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Error: The file was not found at {dataset_path}")

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"Error: Could not decode JSON from the file at {dataset_path}")

    processed_data: List[Dict[str, Any]] = []
    
    if not isinstance(raw_data, list):
         raise ValueError("Expected the root of the JSON file to be a list of conversations.")

    for item in raw_data[:SAMPLE_SIZE]:
        if 'conversations' not in item or not isinstance(item['conversations'], list):
            continue
            
        row: Dict[str, Any] = {}
        for message in item['conversations']:
            column_name = message.get('from')
            cell_value = message.get('value')
            if column_name:
                row[column_name] = cell_value
        if row:
            processed_data.append(row)

    if not processed_data:
        raise ValueError("No valid conversation data was processed. Check the input file format.")

    return pd.DataFrame(processed_data)

def format_chat_data(df: pd.DataFrame, template_path: str) -> pd.Series:
    """
    Formats chat data in a DataFrame according to a specified JSON template.

    Args:
        df (pd.DataFrame): The DataFrame containing conversation turns.
        template_path (str): The path to the JSON file with the formatting template.

    Returns:
        pd.Series: A pandas Series containing the formatted text for each row.
    """
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Error: The template file was not found at {template_path}")

    with open(template_path, "r") as f:
        chat_template = json.load(f)

    required_keys = ["system_prompt", "user_turn", "assistant_turn"]
    if not all(key in chat_template for key in required_keys):
        raise KeyError("Template file is missing required keys: 'system_prompt', 'user_turn', 'assistant_turn'")

    def format_single_row(row: pd.Series) -> str:
        system_prompt = chat_template["system_prompt"]
        user_turn = chat_template["user_turn"]
        assistant_turn = chat_template["assistant_turn"]
     
        # Conditionally add the system prompt only if it's a valid, non-empty string.
        formatted_text_system = f"{system_prompt['prefix']}{system_prompt['suffix']}"
        if pd.notna(row["system"]) and row["system"]:
            formatted_text_system = f"{system_prompt['prefix']}{row['system']}{system_prompt['suffix']}"
    
        formatted_text_user = f"{user_turn['prefix']}{row['human']}{user_turn['suffix']}"
        formatted_text_assistant = f"{assistant_turn['prefix']}{row['gpt']}{assistant_turn['suffix']}"
    
        return formatted_text_system, formatted_text_user, formatted_text_assistant

    return df.apply(format_single_row, axis=1)

def process_and_save_conversations(
    dataset_path: str,
    template_path: str,
    output_path: str
) -> Optional[pd.DataFrame]:
    """
    Main pipeline function to load, process, format, and save conversation data.

    Args:
        dataset_path (str): Path to the raw JSON conversation data.
        template_path (str): Path to the JSON formatting template.
        output_path (str): Path to save the processed file (supports .csv and .parquet).

    Returns:
        Optional[pd.DataFrame]: The processed DataFrame, or None if saving fails.
    """
    print("Starting data processing...")
    
    # Step 1: Load and process the raw data
    df = process_raw_conversations(dataset_path)
    print(f"Successfully loaded and processed {len(df)} conversations.")
    
    # Step 2: Format the data using the template
    # formatted_data = format_chat_data(df, template_path)
    # df["system"], df["user"], df["assistant"] = zip(*formatted_data)
    # if "system" not in df.columns:
    #     df["system"] = ""
    df.rename(columns={"human": "user", "gpt": "assistant"}, inplace=True)
    json_output = convert_dataframe_to_messages(df[["system", "user", "assistant"]])
    print("Formatting has been commented out.")

    # Step 3: Save the processed data based on the output file extension
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    file_extension = os.path.splitext(output_path)[1].lower()
    
    print(f"Attempting to save to {output_path}...")
    if file_extension == '.parquet':
        try:
            pd.DataFrame(json_output).to_parquet(output_path, index=False)
            print("Successfully saved as Parquet file.")
        except ImportError:
            print("Error: 'pyarrow' is required to save to Parquet format. Please run 'pip install pyarrow'.")
            return None
    elif file_extension == '.csv':
        df.to_csv(output_path, index=False)
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
    print("--- Starting Data Processing Pipeline for OpenHermes-2.5 ---")
    
    configs = load_config()
    
    DATASET = configs.source.data.input_path + "/teknium_OpenHermes-2.5/openhermes2_5.json"
    TEMPLATE = configs.source.template.path
    OUTPUT = configs.source.data.output_path + "/openhermes2_5.parquet"

    print(f"\nConfiguration:")
    print(f"  Input Dataset: {DATASET}")
    print(f"  Template File: {TEMPLATE}")
    print(f"  Output File:   {OUTPUT}\n")

    try:
        json_output = process_and_save_conversations(
            dataset_path=DATASET,
            template_path=TEMPLATE,
            output_path=OUTPUT
        )

        if json_output is not None:
            print("\n--- Processed Data Head ---")
            print(json_output[0])

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\n--- A processing error occurred ---")
        print(f"Error: {e}")
    except Exception as e:
        print(f"\n--- An unexpected error occurred ---")
        print(f"Error: {e}")

if __name__ == '__main__':
    # This block allows the script to be run directly from the command line.
    # It will process the data, save it, and print the first 5 rows.
    main()