# sbatch --killable --gres=gpu:1,vmem:8g --mem=16G --time=0-3 --wrap "python build_psudo_labels.py --cp=models/long_marix_2True_29210000.cp"
import numpy as np
import pandas as pd
import torch
from utils import get_model, PADDING_VALUE, N_TOKENS
from mapping import phonemes_to_index
import argparse
from jiwer import wer
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch
from heapq import heappush, heappop
import numpy as np
from train_lstm_lm import RNNModel, vocab_size, embed_size, hidden_size, num_layers

sep = PADDING_VALUE
noise_sep = 100
max_len = 1024
superv_seg = False
SIL_CLUSTERS = np.array([1, 2, 4, 10, 12, 20, 21, 22, 27, 31, 34, 37, 39, 40, 41, 47, 54,
                         55, 56, 57, 60, 63, 66, 67, 71, 74, 78, 81, 83, 84, 86, 89, 92, 93,
                         96])
INPUT_DIM = 768
OUTPUT_DIM = N_TOKENS + 1

BATCH_SIZE = 512
alpha = 0.5
beam_size = 5
same_prob = 0.25

# Initialize beams with empty sequence and 0 score


def build_dataset(base_path="./pseg/data/sup_vad_km"):
    with open(f"{base_path}/features.clusters") as f:
        clusters = f.read().splitlines()
    clusters = [np.array([int(y) for y in x.split()]) for x in clusters]
    clean_clusters = []
    for index, line in enumerate(clusters):
        indexes_masking = [line[0] not in SIL_CLUSTERS] + [(line[i] != line[i - 1]) and line[i] not in SIL_CLUSTERS for
                                                           i in
                                                           range(1, len(line))]
        clean_clusters.append(line[indexes_masking])

    with open(f"{base_path}/features.phonemes") as f:
        phonemes = f.read().splitlines()
    phonemes = [[phonemes_to_index[y.upper()] if y != "dx" else phonemes_to_index['T'] for y in x.split()] for x in
                phonemes]
    phonemes = [np.array(x) for x in phonemes]
    return clean_clusters, phonemes


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("transformer", "medium", 1024, 0.0, vocab=101, output_dim=N_TOKENS + 1)
    model.load_state_dict(torch.load("models/long_marix_2True_30260000.cp", map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    lm = RNNModel(vocab_size, embed_size, hidden_size, num_layers)
    lm.load_state_dict(torch.load("models/lm_best.cp", map_location=torch.device('cpu')))
    lm = lm.to(device)
    lm.eval()
    for parameter in lm.parameters():
        parameter.requires_grad_(False)

    clusters, phonemes = build_dataset()
    for i in range(0, len(clusters)):
        x = torch.LongTensor(clusters[i]).unsqueeze(0).to(device)
        model_output = model(x)[0]
        phonemes_probs = model_output.softmax(dim=-1)
        phonemes_argmax = phonemes_probs.argmax(dim=-1)


        seq_until = list(torch.topk(phonemes_probs[0],beam_size)[1].detach().numpy())

        beams = [(0.0,seq_until)]
        y = torch.LongTensor(phonemes[i]).unsqueeze(0).to(device)

        print(y)
        #TODO deel with same phoneme
        for prob_dist in tqdm(phonemes_probs):
            new_beams = []
            for score,beam in beams:
                lm_scores = lm(torch.LongTensor(beam).unsqueeze(0).to(device))[0][-1].softmax(dim=-1)
                lm_scores[beam[-1]] = same_prob
                for phoneme, prob in enumerate(prob_dist[:-1]):
                    if phoneme == beam[-1]:
                        new_seq= beam
                    else:
                        new_seq = beam + [phoneme]
                    new_score = score -((1 - alpha) * prob+ alpha * lm_scores[phoneme])
                    heappush(new_beams, (new_score, new_seq))
            beams = [heappop(new_beams) for _ in range(beam_size)]
        for beam in beams:
            print(beam)
