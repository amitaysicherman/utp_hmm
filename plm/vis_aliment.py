# sbatch --gres=gpu:1,vmem:8g --mem=16G  --time=1-0 --wrap "python vis_aliment.py"

from typing import Any

from torch.utils.data import Dataset

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from transformers import BartConfig, BartForConditionalGeneration
from mapping import index_to_phonemes, index_to_letter
from jiwer import wer

MAX_LENGTH = 256
test_file = "data/LIBRISPEECH_TEST_idx.txt"

letters_file = "data/LIBRISPEECH_TEST_letters.txt"
clusters_test_file = "data/LIBRISPEECH_TEST_clusters_100.txt"

N_PHONEMES = 39
SUPERV_BLANK = N_PHONEMES + 1
PAD_TOKEN = SUPERV_BLANK + 1
START_TOKEN = PAD_TOKEN + 1
END_TOKEN = START_TOKEN + 1
N_TOKENS = END_TOKEN + 1
with open("models/clusters_phonemes_map_100.txt", "r") as f:
    clusters_to_phonemes = f.read().splitlines()
clusters_to_phonemes = [int(x) for x in clusters_to_phonemes]
clusters_to_phonemes = np.array(clusters_to_phonemes)

d_model = 512
nhead = 8
num_layers = 6
BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PhonemesLettersDataset(Dataset):
    def __init__(self, phonemes_file, cluster_file, latters_file=letters_file):
        with open(phonemes_file, 'r') as f:
            clean_phonemes = f.read().splitlines()
        clean_phonemes = [[int(x) for x in line.strip().split()] for line in clean_phonemes]

        with open(latters_file, 'r') as f:
            latters = f.read().splitlines()
        letters = [[int(x) for x in line.strip().split()] for line in latters]
        letters = ["".join([index_to_letter[x] for x in line]) for line in letters]

        clusters_phonemes, clusters = self.read_clusters(cluster_file)

        self.clean = []
        self.noise = []
        self.clusters = []
        self.letters = []
        for c, n, cc, l in zip(clean_phonemes, clusters_phonemes, clusters, letters):
            if len(c) > MAX_LENGTH or len(n) > MAX_LENGTH:
                continue
            self.clean.append([START_TOKEN] + c + [END_TOKEN] + [PAD_TOKEN] * (MAX_LENGTH - len(c)))
            self.noise.append([START_TOKEN] + n + [END_TOKEN] + [PAD_TOKEN] * (MAX_LENGTH - len(n)))
            self.clusters.append(cc)
            self.letters.append(l)

    def read_clusters(self, cluster_file):
        with open(cluster_file, 'r') as f:
            clusters_ = f.read().splitlines()
        clusters_phonemes: list[Any] = []
        clusters = []
        for line in clusters_:
            line = line.strip().split()
            line = [int(x) for x in line]
            clusters.append([x for i, x in enumerate(line) if i == 0 or x != line[i - 1]])
            line = [clusters_to_phonemes[x] for x in line]
            line = [x for x in line if x != SUPERV_BLANK]
            line = [x for i, x in enumerate(line) if i == 0 or x != line[i - 1]]
            clusters_phonemes.append(line)
        return clusters_phonemes, clusters

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        return torch.LongTensor(self.noise[idx]), torch.LongTensor(self.clean[idx]), torch.LongTensor(
            self.clusters[idx]), self.letters[idx]


def get_model() -> BartForConditionalGeneration:
    config = BartConfig(vocab_size=N_TOKENS + 1, max_position_embeddings=MAX_LENGTH + 2, encoder_layers=num_layers,
                        encoder_ffn_dim=d_model, encoder_attention_heads=nhead,
                        decoder_layers=num_layers, decoder_ffn_dim=d_model, decoder_attention_heads=nhead,
                        d_model=d_model, pad_token_id=PAD_TOKEN, bos_token_id=START_TOKEN, eos_token_id=END_TOKEN,
                        decoder_start_token_id=START_TOKEN, forced_eos_token_id=END_TOKEN)  # Set vocab size
    model = BartForConditionalGeneration(config)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'{params:,} trainable parameters')
    return model


config_name = "m_0.0001_0.25_0.25_0.5_1.0"


def tensor_to_text(x):
    x = x.detach().cpu().numpy().tolist()
    return " ".join([index_to_phonemes[int(i)] for i in x if i not in [PAD_TOKEN, START_TOKEN, END_TOKEN]])


# main:
if __name__ == '__main__':
    model = get_model()
    model = model.to(device)
    checkpoint = torch.load(f"models/{config_name}_last.cp", map_location=device)
    model.load_state_dict(checkpoint['model'])

    model.eval()
    test_clusters_dataset = PhonemesLettersDataset(test_file, clusters_test_file)
    results = []
    with torch.no_grad():
        for x, y, c, l in tqdm(test_clusters_dataset):
            x_test = x.to(device)
            y_test = y.to(device)
            outputs = model(input_ids=x_test.unsqueeze(0), labels=y_test.unsqueeze(0), output_hidden_states=True)
            pred = outputs.logits.argmax(dim=-1)[0]
            pred = tensor_to_text(pred)
            true = tensor_to_text(y_test)
            super_m = tensor_to_text(x_test)
            wer_score = wer(true, pred)
            wer_super = wer(true, super_m)
            c = c.detach().cpu().numpy().tolist()
            c = " ".join([str(x) for x in c])
            results.append((l, c, true, pred, super_m, wer_score, wer_super))
            break
    pd.DataFrame(results, columns=["Text", "clusters", "Phonemes", "Prediction", "Supers Mapping", "Prediction WER",
                                   "Superv_WER"]).to_csv(f'results/{config_name}_results_full.csv')
