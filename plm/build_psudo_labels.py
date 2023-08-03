# sbatch --killable --gres=gpu:1,vmem:8g --mem=16G --time=0-3 --wrap "python eval_model_with_superv_mapping.py --cp=models/long_marix_1_20130000.cp"
import numpy as np
import torch
from tqdm import tqdm
from utils import get_model, PADDING_VALUE, N_TOKENS
from train_long_reaplce_matrix import PhonemesDataset, PROB, ONE
from mapping import phonemes_to_index
import argparse
from jiwer import wer

sep = PADDING_VALUE
noise_sep = 100
max_len = 1024
superv_seg = False
sil=39


def build_dataset(base_path="./pseg/data/sup_vad_km"):
    with open(f"{base_path}/features.clusters") as f:
        clusters = f.read().splitlines()
    clusters = [np.array([int(y) for y in x.split()]) for x in clusters]
    with open(f"{base_path}/features.length") as f:
        lengths = f.read().split("\n")
    lengths = [int(l) for l in lengths]
    features = np.load(f"{base_path}/features.npy")
    assert sum(lengths) == len(features)
    features = np.split(features, np.cumsum(lengths)[:-1])
    assert len(features) == len(clusters)
    return features, clusters

    clean_clusters = []
    for line in clusters:
        indexes_masking=[True]+[(line[i]!=line[i-1]) and line[i] not in SIL_CLUSTERS  for i in range(1,len(line))]

        code100.append([line[0]] + [line[i] for i in range(1, len(line)) if line[i] != line[i - 1]])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cp', type=str, default="./models/long_marix_1True_27130000.cp")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("transformer", "medium", 1024, 0.0, vocab=101, output_dim=N_TOKENS + 1)
    model.load_state_dict(torch.load(args.cp, map_location=torch.device('cpu')))
    model = model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    features, clusters = build_dataset()
    print(features.shape)


