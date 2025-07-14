#!/bin/bash
#SBATCH --job-name=lm-sft     # Name of your job (will appear in squeue)
#SBATCH --partition=gpu_h200            # Which partition/queue to submit to (gpu partition for GPU jobs)
#SBATCH --gres=gpu:h200:2               # Generic RESource - request 1 H100 GPU specifically
#SBATCH --ntasks=1                      # Number of tasks (usually 1 for single-node jobs)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task (adjust based on your needs)
#SBATCH --mem=64G                       # Memory per task
#SBATCH --time=24:00:00                 # Maximum runtime (hrs:min:sec) - job will be killed after this
#SBATCH --output=slurm_output/%j.out    # Standard output file (%j gets replaced with job ID)
#SBATCH --error=slurm_output/%j.err     # Standard error file (%j gets replaced with job ID)
#SBATCH --requeue                       # Automatically requeue job if preempted or failed

module load miniconda

source ~/.bashrc
conda activate llm_base

# Change to the script's directory and then to lm_sft
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKING_DIR="$SCRIPT_DIR/lm_sft"
cd "$WORKING_DIR"

# Add the current directory to Python path so src module can be found
export PYTHONPATH="$PWD:$PYTHONPATH"

DATA_PATH=$(python -m src.config_manager source.data.output_path)
MODEL_PATH=$(python -m src.config_manager source.model.extended_model_path)
MODEL_NAME=$(python -m src.config_manager source.model.model_name)

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

set -x

NUM_GPUS=2
NPROC_PER_NODE=1

# Check if required directories exist
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Data path $DATA_PATH does not exist"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path $MODEL_PATH does not exist"
    exit 1
fi

# Check if data files exist
if [ ! -f "$DATA_PATH/train.parquet" ]; then
    echo "Error: Training data file $DATA_PATH/train.parquet does not exist"
    exit 1
fi

if [ ! -f "$DATA_PATH/test.parquet" ]; then
    echo "Error: Test data file $DATA_PATH/test.parquet does not exist"
    exit 1
fi


torchrun --standalone --nnodes=$NUM_GPUS --nproc_per_node=$NPROC_PER_NODE \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/test.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.micro_batch_size=1 \
    data.train_batch_size=128 \
    data.max_length=4096 \
    data.truncation=right \
    model.partial_pretrain=$MODEL_PATH/$MODEL_NAME\
    model.enable_gradient_checkpointing=true \
    model.fsdp_config.cpu_offload=false \
    model.fsdp_config.offload_params=false \
    model.fsdp_config.model_dtype=bfloat16 \
    trainer.default_local_dir=$WORKING_DIR/results/${MODEL_NAME}\
    trainer.project_name=verl-post-training-pipeline-sft \
    trainer.experiment_name=sft-${MODEL_NAME}-${TIMESTAMP}  \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true