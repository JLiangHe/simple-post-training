#!/usr/bin/env python3
"""
Hugging Face Model and Dataset Downloader

SIMPLE USAGE:
1. Install required package: pip install huggingface_hub
2. Download a model: python hf_downloader.py bert-base-uncased --folder ./downloads
3. Download a dataset: python hf_downloader.py squad --type dataset --folder ./downloads

BASE PATH SPECIFICATION:
- Use --folder to specify where to download (default: ./hf_downloads)
- Examples: --folder /home/user/models or --folder C:\Downloads
- The script will download the content into a subfolder named after the repository inside your specified path.

Example:
python hf_downloader.py --folder /home/user/models --type model bert-base-uncased
python hf_downloader.py --folder /home/user/models --type dataset squad

For more options, run: python hf_downloader.py --help
"""

#python hf_downloader.py --folder /gpfs/radev/project/zhuoran_yang/shared --type model meta-llama/Llama-3.1-8B 
#python hf_downloader.py --folder /home/jh3439/project_pi_zy279/jh3439 --type dataset teknium/OpenHermes-2.5

import os
import argparse
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

def check_repo_type(repo_id):
    """
    Try to determine if a repository is a model or dataset
    Returns 'model' or 'dataset'
    """
    try:
        # Try to access as a model first
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Get repository info
        repo_info = api.repo_info(repo_id=repo_id)
        
        # Check if it's a dataset based on the repo type or common dataset patterns
        if hasattr(repo_info, 'cardData') and repo_info.cardData:
            # Check for dataset-specific metadata
            card_data = repo_info.cardData
            if 'dataset_info' in str(card_data) or 'datasets' in repo_id.lower():
                return 'dataset'
        
        # Default to model if we can't determine
        return 'model'
        
    except Exception as e:
        print(f"Warning: Could not determine repo type for {repo_id}. Defaulting to model.")
        return 'model'

def download_model(repo_id, local_dir):
    """Download a model from Hugging Face"""
    try:
        print(f"Downloading model: {repo_id}")
        print(f"Saving to: {local_dir}")
        
        # Download the entire model repository
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"✓ Model '{repo_id}' downloaded successfully!")
        return True
        
    except HfHubHTTPError as e:
        print(f"✗ Error downloading model '{repo_id}': {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error downloading model '{repo_id}': {e}")
        return False

def download_dataset(repo_id, local_dir):
    """Download a dataset from Hugging Face"""
    try:
        print(f"Downloading dataset: {repo_id}")
        print(f"Saving to: {local_dir}")
        
        # Download the entire dataset repository
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"✓ Dataset '{repo_id}' downloaded successfully!")
        return True
        
    except HfHubHTTPError as e:
        print(f"✗ Error downloading dataset '{repo_id}': {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error downloading dataset '{repo_id}': {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Download models and datasets from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download a model
  python hf_downloader.py bert-base-uncased --type model --folder ./downloads

  # Download a dataset
  python hf_downloader.py squad --type dataset --folder ./downloads

  # Auto-detect type (experimental)
  python hf_downloader.py microsoft/DialoGPT-medium --folder ./downloads
        """
    )
    
    parser.add_argument(
        "repo_id",
        help="Repository ID from Hugging Face (e.g., 'bert-base-uncased' or 'squad')"
    )
    
    parser.add_argument(
        "--type",
        choices=["model", "dataset", "auto"],
        default="auto",
        help="Type of repository to download (default: auto-detect)"
    )
    
    parser.add_argument(
        "--folder",
        default="./hf_downloads",
        help="Base folder for downloads (default: ./hf_downloads)"
    )
    
    args = parser.parse_args()
    
    # Create download directory if it doesn't exist
    try:
        download_dir = Path(args.folder)
        download_dir.mkdir(parents=True, exist_ok=True)
        print(f"Download directory: {download_dir.resolve()}")
        print("-" * 50)
    except Exception as e:
        print(f"Error creating directory: {e}")
        sys.exit(1)
    
    # Determine repository type
    if args.type == "auto":
        repo_type = check_repo_type(args.repo_id)
        print(f"Auto-detected type: {repo_type}")
    else:
        repo_type = args.type
    
    # Set up local directory path
    repo_name = args.repo_id.replace("/", "_")  # Replace slashes for folder names
    
    local_dir = download_dir / repo_name

    if repo_type == "model":
        success = download_model(args.repo_id, local_dir)
    else:  # dataset
        success = download_dataset(args.repo_id, local_dir)
    
    if success:
        print(f"\n✓ Download completed successfully!")
        print(f"Files saved to: {local_dir.resolve()}")
    else:
        print(f"\n✗ Download failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
