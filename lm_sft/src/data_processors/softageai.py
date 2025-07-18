import json, os, sys
import pandas as pd
from typing import List, Dict, Any, Optional
from types import SimpleNamespace

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
from src.config_manager import load_config

SAMPLE_SIZE = 400

def convert_dataframe_to_messages(df):
    """
    Convert DataFrame with user_1, assistant_1, user_2, assistant_2, ..., system columns
    to messages format for MultiTurnSFTDataset
    
    Args:
        df: DataFrame with columns like user_1, assistant_1, user_2, assistant_2, system
        
    Returns:
        List of dictionaries with "messages" key
    """
    results = []
    
    for idx, row in df.iterrows():
        messages = []
        
        # Add system message first if present
        if 'system' in row and pd.notna(row['system']) and str(row['system']).strip():
            messages.append({
                "role": "system",
                "content": str(row['system']).strip()
            })
        
        # Find all user and assistant columns
        user_cols = [col for col in df.columns if col.startswith('user_')]
        assistant_cols = [col for col in df.columns if col.startswith('assistant_')]
        
        # Sort by number to ensure correct order
        user_cols.sort(key=lambda x: int(x.split('_')[1]))
        assistant_cols.sort(key=lambda x: int(x.split('_')[1]))
        
        # Get the maximum turn number
        max_user_turn = len(user_cols)
        max_assistant_turn = len(assistant_cols)
        max_turns = max(max_user_turn, max_assistant_turn)
        
        # Interleave user and assistant messages
        for turn in range(1, max_turns + 1):
            user_col = f"user_{turn}"
            assistant_col = f"assistant_{turn}"
            
            # Add user message if present
            if user_col in row and pd.notna(row[user_col]) and str(row[user_col]).strip():
                messages.append({
                    "role": "user",
                    "content": str(row[user_col]).strip()
                })
            
            # Add assistant message if present  
            if assistant_col in row and pd.notna(row[assistant_col]) and str(row[assistant_col]).strip():
                messages.append({
                    "role": "assistant", 
                    "content": str(row[assistant_col]).strip()
                })
        
        # Only add if we have at least one message
        if messages:
            results.append({"messages": messages})
    
    return results

def process_raw_conversations(dataset_path: str) -> pd.DataFrame:
    """
    Loads conversation data from a Parquet file and processes it into a pandas DataFrame.

    Args:
        dataset_path (str): The full path to the input Parquet dataset file.

    Returns:
        pd.DataFrame: A DataFrame with the processed data.
    
    Raises:
        FileNotFoundError: If the dataset_path does not exist.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Error: The file was not found at {dataset_path}")

    df = pd.read_parquet(dataset_path)
    df = df.drop(columns=["Type","]","Category"])
    df["Use case"] = df["Use case"].apply(lambda x: "You are a helpful " + x + ".")
    
    # Rename columns to match expected format
    column_mapping = {}
    for i in range(1, 6):  # 5 turns
        column_mapping[f"P{i}"] = f"user_{i}"
        column_mapping[f"R{i}"] = f"assistant_{i}"
    column_mapping["Use case"] = "system"
    
    df = df.rename(columns=column_mapping)
    return df

# def format_chat_data(df: pd.DataFrame, template_path: str) -> pd.DataFrame:
#     """
#     Formats chat data in a DataFrame according to a specified JSON template.
#     Reads through all columns and applies appropriate templates.

#     Args:
#         df (pd.DataFrame): The DataFrame containing conversation turns.
#         template_path (str): The path to the JSON file with the formatting template.

#     Returns:
#         pd.DataFrame: A DataFrame with formatted columns.
#     """
#     # if not os.path.exists(template_path):
#     #     raise FileNotFoundError(f"Error: The template file was not found at {template_path}")

#     # with open(template_path, "r") as f:
#     #     chat_template = json.load(f)

#     # required_keys = ["system_prompt", "user_turn", "assistant_turn"]
#     # if not all(key in chat_template for key in required_keys):
#     #     raise KeyError("Template file is missing required keys: 'system_prompt', 'user_turn', 'assistant_turn'")

#     def format_multiturn_chat(df):
#         """
#         Reads through all columns of DataFrame and applies appropriate templates.
#         Handles column order: user_1, assistant_1, user_2, assistant_2, ..., system
#         Returns a DataFrame with formatted columns.
#         """
#         # system_prompt = chat_template["system_prompt"]
#         # user_turn = chat_template["user_turn"]
#         # assistant_turn = chat_template["assistant_turn"]
        
#         # # Create a copy of the DataFrame to store formatted results
#         # formatted_df = df.copy()
        
#         # # Loop through all columns in the DataFrame
#         # for column in df.columns:
#         #     if column == "system":
#         #         # Apply system template
#         #         formatted_df[column] = df[column].apply(
#         #             lambda x: f"{system_prompt['prefix']}{x}{system_prompt['suffix']}" 
#         #             if pd.notna(x) and x else f"{system_prompt['prefix']}{system_prompt['suffix']}"
#         #         )
#         #     elif column.startswith("user_"):
#         #         # Apply user template
#         #         formatted_df[column] = df[column].apply(
#         #             lambda x: f"{user_turn['prefix']}{x}{user_turn['suffix']}" 
#         #             if pd.notna(x) and x else ""
#         #         )
#         #     elif column.startswith("assistant_"):
#         #         # Apply assistant template
#         #         formatted_df[column] = df[column].apply(
#         #             lambda x: f"{assistant_turn['prefix']}{x}{assistant_turn['suffix']}" 
#         #             if pd.notna(x) and x else ""
#         #         )
#         #     # For any other columns, keep as is
#         #     else:
#         #         formatted_df[column] = df[column]
        
#         return df

#     return format_multiturn_chat(df)

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
    formatted_df = process_raw_conversations(dataset_path).iloc[:SAMPLE_SIZE]
    print(f"Successfully loaded and processed {len(formatted_df)} conversations.")
    
    # Step 2: Format the data using the template (column-wise)
    #formatted_df = format_multiturn_chat(df)
    json_output = convert_dataframe_to_messages(formatted_df)
    print("Formatting applied successfully.")

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
    print("--- Starting Data Processing Pipeline for SoftageAI ---")
    
    configs = load_config()
    
    DATASET = configs.source.data.input_path +"/SoftAge-AI_multi-turn_dataset/Multi-turn prompts.parquet"
    OUTPUT = configs.source.data.output_path + "/SoftAge-AI/multi-turn_dataset.parquet"

    print(f"\nConfiguration:")
    print(f"  Input Dataset: {DATASET}")
    print(f"  Output File:   {OUTPUT}\n")

    try:
        json_output = process_and_save_conversations(
            dataset_path=DATASET,
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
    main()
