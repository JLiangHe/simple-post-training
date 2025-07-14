#!/bin/bash
#SBATCH --job-name=lm-sft     # Name of your job (will appear in squeue)
#SBATCH --partition=gpu                 # Which partition/queue to submit to (gpu partition for GPU jobs)
#SBATCH --qos=qos_zhuoran_yang          # Quality of Service - our group's priority access tag
#SBATCH --gres=gpu:h200:2               # Generic RESource - request 1 H100 GPU specifically
#SBATCH --ntasks=1                      # Number of tasks (usually 1 for single-node jobs)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task (adjust based on your needs)
#SBATCH --mem=64G                       # Memory per task
#SBATCH --time=24:00:00                 # Maximum runtime (hrs:min:sec) - job will be killed after this
#SBATCH --output=slurm_output/%j.out    # Standard output file (%j gets replaced with job ID)
#SBATCH --error=slurm_output/%j.err     # Standard error file (%j gets replaced with job ID)
#SBATCH --requeue                       # Automatically requeue job if preempted or failed

source ~/.bashrc
conda activate llm_base

cd "$(dirname "$0")/lm_sft"

DATA_PATH=$(python -m src.config_manager source.data.output_path)
MODEL_PATH=$(python -m src.config_manager source.model.extended_model_path)
MODEL_NAME=$(python -m src.config_manager source.model.model_name)
WORKING_DIR=$(python -m src.config_manager source.working_dir)

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

set -x

NUM_GPUS=1
NPROC_PER_NODE=1


torchrun --standalone --nnodes=$NUM_GPUS --nproc_per_node=$NPROC_PER_NODE \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/test.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.micro_batch_size=1 \
    data.train_batch_size=64 \
    data.max_length=1024 \
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