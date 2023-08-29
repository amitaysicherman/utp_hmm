#!/bin/bash
#SBATCH --time=0-12
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --array=1-8
#SBATCH --gres=gpu,vmem:8g
#SBATCH --killable
#SBATCH --requeue
#SBATCH --output=output-%A_%a.txt

# Define the args list directly in this file
readarray -t args_list <<EOL
tm s 1 1e-4 10 512
tm m 1 1e-4 10 512
tm l 1 1e-4 10 512
tm s 8 1e-4 10 512
tm m 8 1e-4 10 512
tm l 8 1e-4 10 512
tm s 1 1e-4 100 512
tm m 1 1e-4 100 512
tm l 1 1e-4 100 512
tm s 8 1e-4 100 512
tm m 8 1e-4 100 512
tm l 8 1e-4 100 512
tm s 1 1e-4 10 256
tm m 1 1e-4 10 256
tm l 1 1e-4 10 256
tm s 8 1e-4 10 256
tm m 8 1e-4 10 256
tm l 8 1e-4 10 256
tm s 1 1e-4 100 256
tm m 1 1e-4 100 256
tm l 1 1e-4 100 256
tm s 8 1e-4 100 256
tm m 8 1e-4 100 256
tm l 8 1e-4 100 256
EOL

ARGS="${args_list[$SLURM_ARRAY_TASK_ID-1]}"
read ds model_size batch_size lr max_sample_size  max_length <<< "$ARGS"
python cluster_to_phonemes_bart.py --ds "$ds" --model_size "$model_size" --batch_size "$batch_size" --lr "$lr" --max_sample_size "$max_sample_size" --max_length "$max_length"



