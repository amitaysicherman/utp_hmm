#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=24g
#SBATCH --cpus-per-task=4
#SBATCH --array=1-8
#SBATCH --gres=gpu,vmem:24g


case $SLURM_ARRAY_TASK_ID in
    1)
        model_size='s'
        lr=0.0001
        ;;
    2)
        model_size='s'
        lr=0.00001
        ;;
    3)
        model_size='s'
        lr=0.001
        ;;
    4)
        model_size='m'
        lr=0.0001
        ;;
    5)
        model_size='m'
        lr=0.00001
        ;;
    6)
        model_size='m'
        lr=0.001
        ;;
    7)
        model_size='l'
        lr=0.0001
        ;;
    8)
        model_size='l'
        lr=0.00001
        ;;
    *)
        echo "Invalid array index"
        exit 1
        ;;
esac

python superv_clusters_letters.py --model_size "$model_size" --lr "$lr"