# sbatch --killable --gres=gpu:1,vmem:8g --mem=16G --time=0-3 --wrap "python build_psudo_labels.py --cp=models/long_marix_2True_29210000.cp"
import numpy as np
import torch
from utils import get_model, PADDING_VALUE, N_TOKENS
from mapping import phonemes_to_index
import argparse
from jiwer import wer
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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

    with open(f"{base_path}/features.phonemes") as f:
        phonemes = f.read().splitlines()
    phonemes = [[phonemes_to_index[y.upper()] if y != "dx" else phonemes_to_index['T'] for y in x.split()] for x in
                phonemes]
    phonemes = [np.array(x) for x in phonemes]
    return features, clean_clusters, phonemes


def get_batch(features, clusters, phonemes, size=32):
    clusters_data = []
    features_data = []
    phonemes_data = []
    for _ in range(size):
        sample_clusters = []
        sample_features = []
        sample_phonemes = []
        while len(sample_clusters) < max_len:
            index = np.random.randint(len(features))
            sample_clusters.extend(list(clusters[index]))
            sample_clusters.append(noise_sep)

            sample_features.append(features[index])
            sample_features.append(np.zeros((1, features[index].shape[1])))

            sample_phonemes.extend(phonemes[index])
            sample_phonemes.append(sep)
        sample_clusters = sample_clusters[:max_len]
        if len(sample_phonemes) < max_len:
            sample_phonemes.extend([sep] * (max_len - len(sample_phonemes)))
        else:
            sample_phonemes = sample_phonemes[:max_len]
        sample_features = np.vstack(sample_features)[:max_len]

        clusters_data.append(sample_clusters)
        phonemes_data.append(sample_phonemes)
        features_data.append(sample_features)

    # convert to tensor
    clusters_data = torch.LongTensor(clusters_data)
    phonemes_data = torch.LongTensor(phonemes_data)
    features_data = torch.from_numpy(np.array(features_data))

    return features_data, clusters_data, phonemes_data


def eval_with_phonemes(model, features, phonemes, print_examples=0):
    scores = []
    for i, feat in enumerate(features):
        feat = torch.from_numpy(feat).float().to(device).unsqueeze(0)
        y_hat = model(feat)[0]
        y_hat = y_hat.argmax(dim=-1).detach().cpu().numpy()
        y_hat = [str(x) for x in y_hat if x != sep]
        y_hat = [y_hat[0]] + [y_hat[i] for i in range(1, len(y_hat)) if y_hat[i] != y_hat[i - 1]]
        y_hat = " ".join(y_hat)
        y = " ".join([str(x) for x in phonemes[i]])
        if i < print_examples:
            print("example ", i)
            print(y_hat)
            print(y)

        scores.append(wer(y, y_hat))
    print("clusters wer")
    print(np.histogram(scores, bins=20))
    print(np.mean(scores) * 100)
    print(np.std(scores) * 100)
    return np.mean(scores) * 100


class LinearModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim).to(dtype=torch.float32, device=device)

    def forward(self, x):
        return self.linear(x)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cp', type=str, default="./models/long_marix_2True_29210000.cp")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--top', type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("transformer", "medium", 1024, 0.0, vocab=101, output_dim=N_TOKENS + 1)
    model.load_state_dict(torch.load(args.cp, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    linear_model = LinearModel(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
    linear_model.load_state_dict(torch.load("models/linear_model_d.cp", map_location=torch.device('cpu')))
    linear_model = linear_model.to(device)

    linear_model.train()
    optimizer = torch.optim.Adam(linear_model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss(ignore_index=sep).to(device)
    features, clusters, phonemes = build_dataset()

    for round in range(1_000):
        features_batch, clusters_batch, phonemes_batch = get_batch(features, clusters, phonemes, size=BATCH_SIZE)
        features_batch = features_batch.float().to(device)
        labels = []
        model_wer = []

        mapping = np.zeros((100, N_TOKENS))

        for i, x in tqdm(enumerate(clusters_batch)):
            x = x.to(device)
            x = x.unsqueeze(0)
            y = model(x)[0]  # .argmax(dim=-1)

            pred = y.detach().cpu().argmax(dim=-1)
            pred[x.flatten() == noise_sep] = sep
            pred = pred.numpy()
            if args.top == 1:
                y[y < y.max()] = 0
            elif args.top > 1:
                top_n_values, _ = torch.topk(y, args.top)
                y[y < top_n_values[-1]] = 0

            for x_, y_ in zip(x.flatten(), pred.flatten()):
                if x_ != noise_sep:
                    mapping[x_, y_] += 1

            pred = [pred[0]] + [pred[i] for i in range(1, len(pred)) if pred[i] != pred[i - 1]]
            pred = " ".join([str(x) for x in pred]).split(str(sep))[:-1]

            ph = phonemes_batch[i].detach().cpu().numpy()
            ph = " ".join([str(x) for x in ph]).split(str(sep))[:-1]
            for p, phat in zip(ph, pred):
                if len(p) == 0 or len(phat) == 0:
                    continue
                model_wer.append(wer(p, phat))

            labels.append(y)
        np.save("models/mapping.npy", mapping)
        print("model wer")
        print(np.histogram(model_wer, bins=20))
        print(np.mean(model_wer) * 100)
        print(np.std(model_wer) * 100)
        labels = torch.stack(labels).to(device)

        logits = linear_model(features_batch)

        mask = clusters_batch != noise_sep
        logits = logits[mask]
        labels = labels[mask]

        loss = F.cross_entropy(
            logits,  # .transpose(1, 2),
            labels.softmax(dim=-1),
            # ignore_index=sep
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("loss", loss.item(), "PER", eval_with_phonemes(linear_model, features, phonemes))

        if round > 0 and round % 10 == 0:
            eval_with_phonemes(linear_model, features, phonemes, print_examples=10)
            torch.save(linear_model.state_dict(), f"models/linear_model_d.cp")
