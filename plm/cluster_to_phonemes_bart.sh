#!/bin/bash
#SBATCH --time=0-12
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --array=1-18
#SBATCH --gres=gpu,vmem:8g
#SBATCH --killable
#SBATCH --requeue
#SBATCH --output=output-%A_%a.txt

# Define the args list directly in this file
readarray -t args_list <<EOL
tm xl 1 1e-4 10 512
tm m 1 1e-4 10 512
tm l 1 1e-4 10 512
tm xl 1 1e-4 50 512
tm m 1 1e-4 50 512
tm l 1 1e-4 50 512
tm xl 1 1e-4 10 256
tm m 1 1e-4 10 256
tm l 1 1e-4 10 256
tm xl 1 1e-4 50 256
tm m 1 1e-4 50 256
tm l 1 1e-4 50 256
tm s 1 1e-4 10 1024
tm xl 1 1e-4 10 1024
tm l 1 1e-4 10 1024
tm s 1 1e-4 50 1024
tm xl 1 1e-4 50 1024
tm l 1 1e-4 50 1024
EOL

ARGS="${args_list[$SLURM_ARRAY_TASK_ID-1]}"
read ds model_size batch_size lr max_sample_size  max_length <<< "$ARGS"
python cluster_to_phonemes_bart.py --ds "$ds" --model_size "$model_size" --batch_size "$batch_size" --lr "$lr" --max_sample_size "$max_sample_size" --max_length "$max_length"


