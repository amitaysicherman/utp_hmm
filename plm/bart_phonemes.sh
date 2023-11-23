#!/bin/bash
#SBATCH --time=1-00
#SBATCH --killable
#SBATCH --requeue
#SBATCH --mem=16g
#SBATCH --cpus-per-task=4
#SBATCH --array=1-70
#SBATCH --gres=gpu,vmem:8g

LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p bart_phonemes.txt)
model_size=$(echo "$LINE" | cut -f1 -d " ")
p_eq=$(echo "$LINE" | cut -f2 -d " ")
p_add=$(echo "$LINE" | cut -f3 -d " ")
p_del=$(echo "$LINE" | cut -f4 -d " ")
p_rep=$(echo "$LINE" | cut -f5 -d " ")


python bart_phonemes.py  --model_size "$model_size" --p_eq "$p_eq" --p_add "$p_add" --p_del "$p_del" --p_rep "$p_rep"