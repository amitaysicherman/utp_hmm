# sbatch --killable --gres=gpu:1,vmem:8g --mem=16G --time=0-3 --wrap "python self_training.py"
import numpy as np
import torch
from utils import get_model, PADDING_VALUE, N_TOKENS
from mapping import phonemes_to_index, i_to_j
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
    with open(f"{base_path}/features.length") as f:
        lengths = f.read().split("\n")
    lengths = [int(l) for l in lengths]
    features = np.load(f"{base_path}/features.npy")
    assert sum(lengths) == len(features)
    features = np.split(features, np.cumsum(lengths)[:-1])
    assert len(features) == len(clusters)

    clean_clusters = []
    for index, line in enumerate(clusters):
        indexes_masking = [line[0] not in SIL_CLUSTERS] + [(line[i] != line[i - 1]) and line[i] not in SIL_CLUSTERS for
                                                           i in
                                                           range(1, len(line))]
        clean_clusters.append(line[indexes_masking])
        features[index] = features[index][indexes_masking]

    with open("pseg/data/p_superv/features.phonemes") as f:
        phonemes = f.read().splitlines()
    phonemes = [[phonemes_to_index[p.upper()] for p in x.split()] for x in phonemes]
    phonemes = [[p[0]] + [p[i] for i in range(1, len(p)) if p[i] != p[i - 1]] for p in phonemes]

    # with open("data/TIMIT_TRAIN_PH_IDX.txt") as f:
    #     phonemes = f.read().splitlines()
    # phonemes = [[int(y) for y in x.split()] for x in phonemes]
    # phonemes = [phonemes[i_to_j[index]] for index in range(len(phonemes))]

    phonemes = [np.array(x) for x in phonemes]
    return features, clean_clusters, phonemes


def get_sample(features, clusters, phonemes):
    sample_clusters = []
    sample_features = []
    sample_phonemes = []
    count = 0
    while count < max_len:
        index = np.random.randint(len(features))
        if len(clusters[index]) + count > max_len:
            break
        sample_clusters.append(list(clusters[index]))
        sample_features.append(features[index])
        sample_phonemes.append(phonemes[index])
        count += len(clusters[index]) + 1

    return sample_features, sample_clusters, sample_phonemes


def eval_with_phonemes(model, features, phonemes):
    scores = []
    for i, feat in enumerate(features):
        feat = torch.from_numpy(feat).float().to(device).unsqueeze(0)
        y_hat = model(feat)[0].argmax(dim=-1).detach().cpu().numpy()
        y_hat = [str(x) for x in y_hat if x != sep]
        if len(y_hat) == 0:
            continue
        y_hat = [y_hat[0]] + [y_hat[i] for i in range(1, len(y_hat)) if y_hat[i] != y_hat[i - 1]]
        y_hat = " ".join(y_hat)
        y = " ".join([str(x) for x in phonemes[i]])
        scores.append(wer(y, y_hat))
    print("wer", np.mean(scores) * 100, np.std(scores))
    return np.mean(scores) * 100


class LinearModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim).to(dtype=torch.float32, device=device)

    def forward(self, x):
        return self.linear(x)


def load_models():
    model = get_model("transformer", "medium", 1024, 0.0, vocab=101, output_dim=N_TOKENS + 1)
    model.load_state_dict(torch.load(args.cp, map_location=torch.device('cpu')))
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

    for pred in pred_list:
        denoiser_input = torch.unique_consecutive(pred)

        denoiser_input = torch.cat(
            [torch.LongTensor([START_TOKEN]).to(device), denoiser_input, torch.LongTensor([END_TOKEN]).to(device)])
        min_new_tokens = max(10, int(0.5 * len(denoiser_input)))
        max_new_tokens = min(100, int(1.5 * len(denoiser_input)))
        denoiser_input = denoiser_input.unsqueeze(0)
        denoiser_output = denoiser.generate(denoiser_input, max_new_tokens=max_new_tokens,
                                            min_new_tokens=min_new_tokens, top_k=4, num_beams=100)
        denoiser_output = torch.unique_consecutive(denoiser_output)[1:-1]
        denoiser_output_list.append(denoiser_output)

    return denoiser_output_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cp', type=str, default="./models/long_marix_2True_30260000.cp")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--top', type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, linear_model, denoiser = load_models()
    optimizer = torch.optim.Adam(linear_model.parameters(), lr=args.lr)
    # criterion = nn.CrossEntropyLoss(ignore_index=sep).to(device)
    features, clusters, phonemes = build_dataset()

    print("dataset loaded")
    loss_function = nn.CTCLoss(blank=sep, zero_infinity=True)
    linear_features_input = []
    linear_labels = []
    for round in tqdm(range(100_000)):

        sample_features, sample_clusters, sample_phonemes = get_sample(features, clusters, phonemes)
        long_clusters = []
        for c in sample_clusters:
            long_clusters += c
            long_clusters.append(noise_sep)
        long_clusters = np.array(long_clusters)[:max_len]
        long_clusters = torch.LongTensor(long_clusters).to(device).unsqueeze(0)
        model_output = model(long_clusters)[0]
        linear_labels.extend(model_output_denoiser(model_output, sample_clusters, denoiser))
        linear_features_input.extend(sample_features)

        if len(linear_features_input) > BATCH_SIZE:
            inputs_padded = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(seq, dtype=torch.float32) for seq in linear_features_input], batch_first=True).to(device)
            input_lengths = torch.LongTensor([len(x) for x in linear_features_input]).to(device)
            logits = linear_model(inputs_padded)

            target_lengths = torch.tensor([len(t) for t in linear_labels], dtype=torch.long).to(device)
            target_padded = torch.nn.utils.rnn.pad_sequence(linear_labels, batch_first=True, padding_value=sep).to(
                device)

            optimizer.zero_grad()

            loss = loss_function(logits.log_softmax(dim=-1).transpose(0, 1), target_padded, input_lengths,
                                 target_lengths)
            print(round, "loss", loss.item())

            loss.backward()
            optimizer.step()
            wer_score = eval_with_phonemes(linear_model, features, phonemes)
            with open("results/linear_model.txt", "a") as f:
                f.write(f"{round} {loss.item()} {wer_score}\n")
            linear_features_input = []
            linear_labels = []
