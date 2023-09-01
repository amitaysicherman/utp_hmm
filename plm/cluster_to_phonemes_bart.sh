#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --array=1-4
#SBATCH --gres=gpu,vmem:8g
#SBATCH --output=output-%A_%a.txt

python cluster_to_phonemes_bart.py --model_size "s"
python cluster_to_phonemes_bart.py --model_size "m"
python cluster_to_phonemes_bart.py --model_size "l"
python cluster_to_phonemes_bart.py --model_size "xl"



