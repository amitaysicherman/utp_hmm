#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --array=1-12
#SBATCH --gres=gpu,vmem:8g
#SBATCH --output=output-%A_%a.txt

# Define the args list directly in this file
readarray -t args_list <<EOL
xl 50 0
m 50 0
xl 50 1
m 50 1
xl 50 2
m 50 2
xl 10 0
m 10 0
xl 10 1
m 10 1
xl 10 2
m 10 2
EOL

ARGS="${args_list[$SLURM_ARRAY_TASK_ID-1]}"
read model_size max_sample_size  start_mode <<< "$ARGS"
python cluster_to_phonemes_bart.py --model_size "$model_size" --max_sample_size "$max_sample_size" --start_mode "$start_mode"



