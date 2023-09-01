#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --array=1-4
#SBATCH --gres=gpu,vmem:8g
#SBATCH --output=output-%A_%a.txt

case $SLURM_ARRAY_TASK_ID in
    1)
        model_size='s'
        ;;
    2)
        model_size='m'
        ;;
    3)
        model_size='l'
        ;;
    4)
        model_size='xl'
        ;;
    *)
        echo "Invalid array index"
        exit 1
        ;;
esac

python cluster_to_phonemes_bart.py --model_size "$model_size"




