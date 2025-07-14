import os
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.config_manager import load_config

def aggregate_and_split_datasets(configs):
    """
    Aggregates datasets specified in the config, splits them into training and testing sets,
    and saves them to the specified output paths.

    Args:
        configs: The configuration object loaded from source_configs.yaml.
    """
    
    # --- Aggregation ---
    aggregated_df = pd.DataFrame()
    
    datasets_to_aggregate = configs.source.data.dataset_name
    
    print("--- Starting Dataset Aggregation ---")
    for dataset_name in datasets_to_aggregate:
        dataset_path = os.path.join(configs.source.data.output_path, f"{dataset_name}.parquet")
        
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset not found at {dataset_path}. Skipping.")
            continue
            
        try:
            df = pd.read_parquet(dataset_path)
            aggregated_df = pd.concat([aggregated_df, df], ignore_index=True)
            print(f"Successfully aggregated {dataset_name} ({len(df)} records).")
        except Exception as e:
            print(f"Error reading {dataset_path}: {e}")
            
    if aggregated_df.empty:
        print("No data was aggregated. Exiting.")
        return

    print(f"--- Total records aggregated: {len(aggregated_df)} ---")

    # --- Splitting ---
    train_split = configs.source.data.train_split
    test_size = 1 - train_split
    
    print(f"--- Splitting data into training and testing sets (test_size={test_size}) ---")
    
    train_df, test_df = train_test_split(
        aggregated_df,
        test_size=test_size,
        random_state=42  # Keep a fixed random state for reproducibility
    )
    
    print(f"Training set size: {len(train_df)}")
    print(f"Testing set size: {len(test_df)}")

    # --- Saving ---
    train_path = os.path.join(configs.source.data.output_path, "train.parquet")
    test_path = os.path.join(configs.source.data.output_path, "test.parquet")
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    try:
        train_df.to_parquet(train_path, index=False)
        print(f"Training data saved to {train_path}")
        
        test_df.to_parquet(test_path, index=False)
        print(f"Testing data saved to {test_path}")
        
    except Exception as e:
        print(f"Error saving split datasets: {e}")

def main():
    """
    Main execution function.
    """
    print("--- Starting Data Aggregation and Splitting ---")
    
    try:
        configs = load_config()
        aggregate_and_split_datasets(configs)
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()