#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=24g
#SBATCH --cpus-per-task=4
#SBATCH --array=1-12
#SBATCH --gres=gpu,vmem:24g


case $SLURM_ARRAY_TASK_ID in
    1)
      noise=0.1
      model_size='s'
      ;;
    2)
      noise=0.25
      model_size='s'
      ;;
    3)
      noise=0.5
      model_size='s'
      ;;
    4)
      noise=0.75
      model_size='s'
      ;;
    5)
      noise=0.1
      model_size='m'
      ;;
    6)
      noise=0.25
      model_size='m'
      ;;
    7)
      noise=0.5
      model_size='m'
      ;;
    8)
      noise=0.75
      model_size='m'
      ;;
    9)
      noise=0.1
      model_size='l'
      ;;
    10)
      noise=0.25
      model_size='l'
      ;;
    11)
      noise=0.5
      model_size='l'
      ;;
    12)
      noise=0.75
      model_size='l'
      ;;
    *)
        echo "Invalid array index"
        exit 1
        ;;
esac

python bart_phonemes_letters.py --noise "$noise" --model_size "$model_size"