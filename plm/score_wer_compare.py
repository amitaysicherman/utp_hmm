# sbatch --killable --gres=gpu:1,vmem:8g --mem=16G --time=0-3 --wrap "python score_wer_compare.py"
import numpy as np
import torch
from utils import get_model, PADDING_VALUE, N_TOKENS
from mapping import phonemes_to_index
import argparse
from jiwer import wer
import torch.nn as nn
from bart_denoiser import get_model as get_denoiser_model, PAD_TOKEN, END_TOKEN, START_TOKEN
from tqdm import tqdm

sep = PADDING_VALUE
noise_sep = 100
max_len = 1024
SIL_CLUSTERS = np.array([1, 2, 4, 10, 12, 20, 21, 22, 27, 31, 34, 37, 39, 40, 41, 47, 54,
                         55, 56, 57, 60, 63, 66, 67, 71, 74, 78, 81, 83, 84, 86, 89, 92, 93,
                         96])
INPUT_DIM = 768
OUTPUT_DIM = N_TOKENS + 2
BATCH_SIZE = 512


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

    with open("pseg/data/p_superv/features.phonemes") as f:
        phonemes = f.read().splitlines()
    phonemes = [[phonemes_to_index[p.upper()] for p in x.split()] for x in phonemes]
    phonemes = [[p[0]] + [p[i] for i in range(1, len(p)) if p[i] != p[i - 1]] for p in phonemes]

    # with open("data/TIMIT_TRAIN_PH_IDX.txt") as f:
    #     phonemes = f.read().splitlines()
    # phonemes = [[int(y) for y in x.split()] for x in phonemes]
    # phonemes = [phonemes[i_to_j[index]] for index in range(len(phonemes))]

    phonemes = [np.array(x) for x in phonemes]
    return clean_clusters, phonemes


def get_sample(clusters, phonemes):
    sample_clusters = []
    sample_phonemes = []
    count = 0
    while count < max_len:
        index = np.random.randint(len(clusters))
        sample_clusters.append(list(clusters[index]))
        sample_phonemes.append(phonemes[index])
        count += len(clusters[index]) + 1
    return sample_clusters, sample_phonemes


class LinearModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim).to(dtype=torch.float32, device=device)

    def forward(self, x):
        return self.linear(x)


def load_models():
    model = get_model("transformer", "medium", 1024, 0.0, vocab=101, output_dim=N_TOKENS + 1)
    model.load_state_dict(torch.load("models/long_marix_2True_30260000.cp", map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(model)
    linear_model = LinearModel(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
    # linear_model.load_state_dict(torch.load("models/linear_model_d.cp", map_location=torch.device('cpu')))
    linear_model = linear_model.to(device)
    linear_model.train()
    print(linear_model)

    denoiser = get_denoiser_model().to(device)
    denoiser.load_state_dict(torch.load("models/bart_denoiser_best_train_loss.cp", map_location=torch.device('cpu')))
    denoiser = denoiser.to(device)
    denoiser.eval()
    print(denoiser)
    return model, linear_model, denoiser


def model_output_denoiser(y, list_values, denoiser):
    lengthes = [len(x) for x in list_values]
    pred = y.argmax(dim=-1)
    pred_list = []
    cur_len = 0
    for l in lengthes:
        pred_list.append(pred[cur_len:cur_len + l])
        cur_len += l
        cur_len += 1  # seq
    denoiser_output_list = []
    scores = []
    for pred in pred_list:
        denoiser_input = torch.unique_consecutive(pred)

        denoiser_input = torch.cat(
            [torch.LongTensor([START_TOKEN]).to(device), denoiser_input, torch.LongTensor([END_TOKEN]).to(device)])
        min_new_tokens = max(10, int(0.5 * len(denoiser_input)))
        max_new_tokens = min(100, int(1.5 * len(denoiser_input)))
        denoiser_input = denoiser_input.unsqueeze(0)
        denoiser_output_dict = denoiser.generate(denoiser_input, max_new_tokens=max_new_tokens,
                                                 min_new_tokens=min_new_tokens, top_k=4, num_beams=100,
                                                 output_scores=True, return_dict_in_generate=True)

        denoiser_output = denoiser_output_dict["sequences"][0]
        scores.append(denoiser_output.sequences_scores[0].item())
        denoiser_output = torch.unique_consecutive(denoiser_output)[1:-1]
        denoiser_output_list.append(denoiser_output.detach().cpu().numpy())

    return denoiser_output_list, scores


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, linear_model, denoiser = load_models()
    optimizer = torch.optim.Adam(linear_model.parameters(), lr=args.lr)
    clusters, phonemes = build_dataset()

    print("dataset loaded")
    loss_function = nn.CTCLoss(blank=sep, zero_infinity=True)
    linear_features_input = []
    linear_labels = []
    for round in tqdm(range(100_000)):

        sample_clusters, sample_phonemes = get_sample(clusters, phonemes)
        long_clusters = []
        for c in sample_clusters:
            long_clusters += c
            long_clusters.append(noise_sep)
        long_clusters = np.array(long_clusters)[:max_len]
        long_clusters = torch.LongTensor(long_clusters).to(device).unsqueeze(0)
        model_output = model(long_clusters)[0]
        denoiser_phonemes, scores = model_output_denoiser(model_output, sample_clusters, denoiser)
        for dp, p, s in enumerate(denoiser_phonemes, sample_phonemes, scores):
            y_hat = [str(x) for x in dp if x != sep]
            y_hat = [y_hat[0]] + [y_hat[i] for i in range(1, len(y_hat)) if y_hat[i] != y_hat[i - 1]]
            y_hat = " ".join(y_hat)
            y = " ".join([str(x) for x in p])
            wer_score = wer(y, y_hat)
            with open("results/wer_scores.txt", "a") as f:
                f.write(f"{round},{wer_score},{s}\n")
