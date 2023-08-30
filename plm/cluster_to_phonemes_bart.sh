#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --array=1-4
#SBATCH --gres=gpu,vmem:8g
#SBATCH --output=output-%A_%a.txt

# Define the args list directly in this file
readarray -t args_list <<EOL
l 50 2
s 50 2
l 10 2
s 10 2
EOL

ARGS="${args_list[$SLURM_ARRAY_TASK_ID-1]}"
read model_size max_sample_size  start_mode <<< "$ARGS"
python cluster_to_phonemes_bart.py --model_size "$model_size" --max_sample_size "$max_sample_size" --start_mode "$start_mode"



