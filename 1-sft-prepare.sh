source ~/.bashrc
conda activate llm_base

# Change to the script's directory
cd "$(dirname "$0")/lm_sft"

# Load configuration
DATA_INPUT_PATH=$(python -m src.config_manager source.data.input_path)
DATASET_NAMES_STR=$(python -m src.config_manager source.data.dataset_name)

# Download datasets
# IFS=' ' read -r -a DATASET_NAMES_ARR <<< "$DATASET_NAMES_STR"
# for dataset in "${DATASET_NAMES_ARR[@]}"; do
#     python src/hf_downloader.py --folder "$DATA_INPUT_PATH" --type dataset "$dataset"
# done

# Process data
python -m src.data_processors.openhermes2_5
python -m src.data_processors.softageai
python -m src.data_processors.aggregate_and_split


# Process model
python -m src.model_processor