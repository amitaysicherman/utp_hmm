#!/bin/sh
#SBATCH --time=1-0
#SBATCH --array=1-3360
#SBATCH --killable
#SBATCH --requeue
#SBATCH --mem=8G

LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p find_hparam.txt)
p=$(echo "$LINE" | cut -f1 -d " ")
h=$(echo "$LINE" | cut -f2 -d " ")
t=$(echo "$LINE" | cut -f3 -d " ")
d=$(echo "$LINE" | cut -f4 -d " ")
w=$(echo "$LINE" | cut -f5 -d " ")

python find_hparam.py --prominence "${p}" --distance "${d}" --threshold "${t}" --height "${h}" --width "${w}"
