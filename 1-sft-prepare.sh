# =============================
# LLM SFT Data Preparation Script
# =============================
#
# This script downloads datasets, processes them using custom processors,
# aggregates and splits the data, and prepares it for supervised fine-tuning (SFT).
#
# Usage:
#   - Configure dataset names and input path in your config manager.
#   - Add dataset-to-module mappings in lm_sft/scripts/dataset_module_map.json.
#   - Run this script from the project root.
#
# =============================

module load miniconda  

source ~/.bashrc
conda activate llm_base  

# Change to the script's directory (lm_sft)
cd "$(dirname "$0")/lm_sft"

# Load configuration values using the config manager
DATA_INPUT_PATH=$(python -m src.config_manager source.data.input_path)
DATASET_NAMES_STR=$(python -m src.config_manager source.data.dataset_name)

# =============================
# Download and process datasets
# =============================

IFS=' ' read -r -a DATASET_NAMES_ARR <<< "$DATASET_NAMES_STR"
for dataset in "${DATASET_NAMES_ARR[@]}"; do
    # Download the dataset using the HuggingFace downloader script
    python src/hf_downloader.py --folder "$DATA_INPUT_PATH" --type dataset "$dataset"
    # Map the dataset name to its processor module using the mapping JSON
    module_name=$(jq -r --arg ds "$dataset" '.[$ds]' scripts/dataset_module_map.json)
    if [ "$module_name" != "null" ]; then
        # Run the corresponding data processor module
        python -m src.data_processors.$module_name
    else
        echo "[WARN] No module mapping found for $dataset, skipping processing."
    fi
done

# =============================
# Aggregate and split processed data
# =============================

python -m src.data_processors.aggregate_and_split

# =============================
# (Optional) Process model
# =============================
# Add chat template to the base model based on lm_sft/scripts/template_format.json
python -m src.model_processor