#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --array=1-8
#SBATCH --gres=gpu,vmem:8g
#SBATCH --output=output-%A_%a.txt

# Define the args list directly in this file
readarray -t args_list <<EOL
lr xl 1 2e-4 50 512
lr xl 1 2e-4 50 256
lr m 1 2e-4 50 512
lr m 1 2e-4 50 256
lr xl 1 2e-4 10 512
lr xl 1 2e-4 10 256
lr m 1 2e-4 10 512
lr m 1 2e-4 10 256
EOL

ARGS="${args_list[$SLURM_ARRAY_TASK_ID-1]}"
read ds model_size batch_size lr max_sample_size  max_length <<< "$ARGS"
python cluster_to_phonemes_bart.py --ds "$ds" --model_size "$model_size" --batch_size "$batch_size" --lr "$lr" --max_sample_size "$max_sample_size" --max_length "$max_length"



