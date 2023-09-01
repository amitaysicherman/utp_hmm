# sbatch --gres=gpu:1,vmem:24g --mem=75G -c4 --time=7-0 --wrap "python cluster_to_phonemes_bart.py"
# https://aclanthology.org/P19-2049.pdf

import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from tqdm import tqdm
import os
from scipy.special import softmax
from scipy.spatial.distance import cdist
from transformers import BartConfig, BartForConditionalGeneration
from dataclasses import dataclass
import argparse

save_update_steps = 1_000
warmup_steps = 50
EPOCHS = 1_000
last_config = False

parser = argparse.ArgumentParser()
parser.add_argument('--ds', type=str, default="lr")
parser.add_argument('--model_size', type=str, default="s")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--max_length", type=int, default=256)
args = parser.parse_args()
BATCH_SIZE = args.batch_size
LR = args.lr
ds = args.ds
model_size = args.model_size
MAX_LENGTH = args.max_length
MAX_DS_SIZE = 2048

config_name = f"bart_{ds}_{model_size}_{BATCH_SIZE}_{LR}_{MAX_LENGTH}"
train_dataset_size = 10_000
test_size = 500

if model_size == "s":
    d_model = 256
    nhead = 4
    num_layers = 4
elif model_size == "m":
    d_model = 512
    nhead = 8
    num_layers = 8
elif model_size == "l":
    d_model = 768
    nhead = 12
    num_layers = 12
elif model_size == "xl":
    d_model = 1024
    nhead = 16
    num_layers = 16
else:
    raise ValueError(f"Unknown size {model_size}")

if ds == "lr":
    phonemes_file = "data/LIBRISPEECH_TRAIN_idx.txt"
    phonemes_file_test = "data/LIBRISPEECH_TEST_idx.txt"
    clusters_file = "data/LIBRISPEECH_TRAIN_clusters.txt"



else:
    phonemes_file = "data/TIMIT_NS_TRAIN_PH_IDX.txt"
    phonemes_file_test = "data/TIMIT_NS_TEST_PH_IDX.txt"
    clusters_file = "data/TIMIT_NS_TRAIN_CLUSTERS.txt"

output_file = f"results/{config_name}.txt"

ONE = 0
SPHERE = 2
PHONEMES_LAST_TOKEN = 38
CLUSTERS_FIRST_TOKEN = PHONEMES_LAST_TOKEN + 1
N_CLUSTERS = 100
CLUSTERS_LAST_TOKEN = CLUSTERS_FIRST_TOKEN + N_CLUSTERS
PAD_TOKEN = CLUSTERS_LAST_TOKEN + 1
SEP = PAD_TOKEN + 1
DECODER_MASK_TOKEN = SEP + 1
START_TOKEN = DECODER_MASK_TOKEN + 1
END_TOKEN = START_TOKEN + 1
N_TOKENS = END_TOKEN + 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Scores:
    train_loss = 0.0
    train_acc = 0.0
    train_count = 0
    test_loss = 0.0
    test_acc = 0.0
    test_count = 0
    cluster_loss = 0.0
    cluster_acc = 0.0
    cluster_count = 0

    def resset_train(self):
        self.train_loss = 0.0
        self.train_acc = 0.0
        self.train_count = 0.0

    def resset_test(self):
        self.test_loss = 0.0
        self.test_acc = 0.0
        self.test_count = 0

    def reset_cluster(self):
        self.cluster_loss = 0.0
        self.cluster_acc = 0.0
        self.cluster_count = 0

    def reset(self):
        self.resset_train()
        self.resset_test()
        self.reset_cluster()

    def update_value(self, loss, acc, train_test):

        if train_test == "train":
            self.train_loss += loss
            self.train_acc += acc
            self.train_count += 1
        elif train_test == "test":
            self.test_loss += loss
            self.test_acc += acc
            self.test_count += 1
        elif train_test == "cluster":
            self.cluster_loss += loss
            self.cluster_acc += acc
            self.cluster_count += 1
        else:
            raise ValueError("Unknown train_test")

    def update_values_from_output(self, outputs, y, train_test):
        loss = outputs.loss.item()
        preds = outputs.logits.argmax(dim=-1)
        mask = y != PAD_TOKEN
        acc = ((preds[mask] == y[mask]).sum() / y[mask].numel()).item()
        self.update_value(loss, acc, train_test)

    def train_to_str(self):
        train_loss = self.train_loss / self.train_count if self.train_count > 0 else 0
        train_acc = self.train_acc / self.train_count if self.train_count > 0 else 0
        return f"TRAIN:{train_loss},{train_acc}"

    def test_to_str(self):
        test_loss = self.test_loss / self.test_count if self.test_count > 0 else 0
        test_acc = self.test_acc / self.test_count if self.test_count > 0 else 0
        return f"TEST:{test_loss},{test_acc}"

    def cluster_to_str(self):
        cluster_loss = self.cluster_loss / self.cluster_count if self.cluster_count > 0 else 0
        cluster_acc = self.cluster_acc / self.cluster_count if self.cluster_count > 0 else 0
        return f"CLUSTER:{cluster_loss},{cluster_acc}"

    def to_file(self, train_test):
        if train_test == "train":
            with open(output_file, "a") as f:
                f.write(self.train_to_str() + "\n")
        elif train_test == "test":
            with open(output_file, "a") as f:
                f.write(self.test_to_str() + "\n")
        elif train_test == "cluster":
            with open(output_file, "a") as f:
                f.write(self.cluster_to_str() + "\n")
        else:
            raise ValueError("Unknown train_test")


SIL_CLUSTERS = np.array([1, 2, 4, 10, 12, 20, 21, 22, 27, 31, 34, 37, 39, 40, 41, 47, 54,
                         55, 56, 57, 60, 63, 66, 67, 71, 74, 78, 81, 83, 84, 86, 89, 92, 93,
                         96])


class ClustersPhonemesDataset(Dataset):
    def __init__(self, phonemes_file, clusters_file, max_len=MAX_LENGTH, samples_count=test_size):
        with open(phonemes_file, 'r') as f:
            phonemes_data = f.readlines()
        self.phonemes_data = [[int(x) for x in line.strip().split()] for line in phonemes_data]
        with open(clusters_file, 'r') as f:
            clusters_data = f.readlines()
        self.clusters_data = [[int(x) for x in line.strip().split()] for line in clusters_data]
        self.clusters_data = [[x for x in line if x not in SIL_CLUSTERS] for line in self.clusters_data]
        self.clusters_data = [[CLUSTERS_FIRST_TOKEN + x for x in line] for line in self.clusters_data]
        self.max_len = max_len
        self.data = []
        self.clusters = []
        max_line_index = len(self.phonemes_data)
        max_line_index = min(max_line_index, len(self.phonemes_data))
        for _ in range(samples_count):
            sample = [START_TOKEN]
            cluster_sample = [START_TOKEN]

            while True:
                new_index = np.random.randint(0, max_line_index)
                new_sample = self.phonemes_data[new_index][:] + [SEP]
                new_cluster_sample = self.clusters_data[new_index][:] + [SEP]

                if (len(sample) + len(new_sample) >= self.max_len - 1) or (
                        len(cluster_sample) + len(new_cluster_sample) >= self.max_len):
                    break

                sample += new_sample
                cluster_sample += new_cluster_sample

            sample += [END_TOKEN]
            cluster_sample += [END_TOKEN]
            sample += [PAD_TOKEN] * (self.max_len - len(sample))
            cluster_sample += [PAD_TOKEN] * (self.max_len - len(cluster_sample))
            self.data.append(sample)
            self.clusters.append(cluster_sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.clusters[idx]), torch.LongTensor(self.data[idx])


class NoiseAdder:
    def __init__(self, size: int):
        self.range_units = np.arange(N_CLUSTERS)
        self.maps = []
        for i in range(size):
            random.seed(i)
            np.random.seed(i)
            self.maps.append(self.build_mapping())

    def random_gaussian(self, n, dim=2):
        point = np.random.normal(size=(n, dim))
        point /= np.linalg.norm(point, axis=1, keepdims=True)
        return point

    def build_mapping_sphere(self):
        phonemes = self.random_gaussian(PHONEMES_LAST_TOKEN + 1)
        clusters = self.random_gaussian(N_CLUSTERS)
        cosine_distances = 100 * (1 - cdist(phonemes, clusters, metric='cosine'))
        probabilities = softmax(cosine_distances, axis=0)
        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
        np.random.shuffle(probabilities)
        return probabilities

    def build_mapping(self):
        return self.build_mapping_sphere()

    def get_length(self, count, n_max=5):
        weights = np.diff(np.sort(np.concatenate(([0, 1], np.random.random(n_max - 1)))))
        np.random.shuffle(weights)
        length = random.choices(np.arange(n_max), weights=weights, k=count)
        return length

    def get_new_token(self, inv_mapping, c):
        new_token = np.random.choice(self.range_units, p=inv_mapping[c])
        return new_token + CLUSTERS_FIRST_TOKEN

    def add_noise(self, clean_sample):

        inv_mapping = random.choice(self.maps)
        length = self.get_length(len(clean_sample))
        noise_sample = []
        for c in clean_sample:
            for _ in range(length.pop()):
                new_token = self.get_new_token(inv_mapping, c)
                noise_sample.append(new_token)
        return noise_sample


class PhonemesDataset(Dataset):
    def __init__(self, phonemes_file, max_len=MAX_LENGTH, samples_count=train_dataset_size, size=1):
        with open(phonemes_file, 'r') as f:
            phonemes_data = f.readlines()
        self.phonemes_data = [[int(x) for x in line.strip().split()] for line in phonemes_data]
        self.max_len = max_len
        self.samples_count = samples_count
        self.clean = []
        self.noise = []
        self.noise_adder = NoiseAdder(size=size)
        self.build_data()

    def build_data(self):
        max_line_index = len(self.phonemes_data)
        for _ in tqdm(range(self.samples_count), desc="Build Dataset"):
            concat_clean_samples = [START_TOKEN]
            concat_noise_samples = [START_TOKEN]
            while True:
                clean_sample = self.phonemes_data[np.random.randint(0, max_line_index)][:]
                noise_sample = self.noise_adder.add_noise(clean_sample)
                clean_sample += [SEP]
                noise_sample += [SEP]

                if (len(concat_clean_samples) + len(clean_sample) >= self.max_len - 1) or (
                        len(concat_noise_samples) + len(noise_sample) >= self.max_len - 1):
                    break
                concat_clean_samples += clean_sample
                concat_noise_samples += noise_sample

            concat_clean_samples += [END_TOKEN]
            concat_noise_samples += [END_TOKEN]
            concat_clean_samples += [PAD_TOKEN] * (self.max_len - len(concat_clean_samples))
            concat_noise_samples += [PAD_TOKEN] * (self.max_len - len(concat_noise_samples))
            self.clean.append(concat_clean_samples)
            self.noise.append(concat_noise_samples)

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):

        clean = torch.LongTensor(self.clean[idx])
        noise = torch.LongTensor(self.noise[idx])
        return noise, clean


def get_model() -> BartForConditionalGeneration:
    config = BartConfig(vocab_size=N_TOKENS + 1, max_position_embeddings=MAX_LENGTH, encoder_layers=num_layers,
                        encoder_ffn_dim=d_model, encoder_attention_heads=nhead,
                        decoder_layers=num_layers, decoder_ffn_dim=d_model, decoder_attention_heads=nhead,
                        d_model=d_model, pad_token_id=PAD_TOKEN, bos_token_id=START_TOKEN, eos_token_id=END_TOKEN,
                        decoder_start_token_id=START_TOKEN, forced_eos_token_id=END_TOKEN)  # Set vocab size
    model = BartForConditionalGeneration(config)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'{params:,} trainable parameters')
    return model


def get_datasets(curr_size, train_sample_count):
    train_dataset = PhonemesDataset(phonemes_file, size=curr_size, samples_count=train_sample_count)
    train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataset = PhonemesDataset(phonemes_file_test, size=curr_size, samples_count=test_size)
    test_data = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    clusters_dataset = ClustersPhonemesDataset(phonemes_file, clusters_file, samples_count=test_size)
    clusters_data = DataLoader(clusters_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return train_data, test_data, clusters_data


def save(model, optimizer, i, best_score, update_best, conf_size):
    data_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': i,
        'best_score': best_score,
        'conf_size': conf_size
    }
    torch.save(data_save, f"models/{config_name}_last.cp")
    if update_best:
        torch.save(data_save, f"models/{config_name}_best.cp")


def load_last(model, optimizer):
    if not os.path.exists(f"models/{config_name}_last.cp"):
        return 0, 0, 1
    checkpoint = torch.load(f"models/{config_name}_last.cp", map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    load_step = checkpoint['step']
    best_score = checkpoint['best_score']
    conf_size = checkpoint['conf_size']
    return load_step, best_score, conf_size


# main:
if __name__ == '__main__':

    model = get_model()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    i, best_test_acc, curr_size = load_last(model, optimizer)

    print(
        f"load cp-  i:{i}, best_test_acc:{best_test_acc},  curr_size:{curr_size}")
    model = model.train()
    scores = Scores()
    train_data, test_data, clusters_data = get_datasets(curr_size, train_dataset_size)

    for epoch in range(EPOCHS):
        pbar = tqdm(train_data, total=len(train_data))
        for x_train, y_train in pbar:
            i += 1
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            outputs = model(input_ids=x_train, labels=y_train, output_hidden_states=False)

            scores.update_values_from_output(outputs, y_train, "train")
            optimizer.zero_grad()
            outputs.loss.backward()
            optimizer.step()

            if i % save_update_steps == 0:
                model.eval()
                with torch.no_grad():
                    for x_test, y_test in test_data:
                        x_test = x_test.to(device)
                        y_test = y_test.to(device)
                        outputs = model(input_ids=x_test, labels=y_test, output_hidden_states=False)
                        scores.update_values_from_output(outputs, y_test, "test")
                    scores.to_file("test")

                    for x_test, y_test in clusters_data:
                        x_test = x_test.to(device)
                        y_test = y_test.to(device)
                        outputs = model(input_ids=x_test, labels=y_test, output_hidden_states=False)
                        scores.update_values_from_output(outputs, y_test, "cluster")
                    scores.to_file("cluster")
                model.train()
                update_best = False
                if scores.test_acc > best_test_acc:
                    best_test_acc = scores.test_acc / scores.test_count
                    update_best = True

                scores.resset_test()
                scores.reset_cluster()

                save(model, optimizer, i, best_test_acc, update_best, curr_size)

            if i % warmup_steps == 0:

                scores.to_file("train")
                cur_acc = scores.train_acc / scores.train_count
                scores.resset_train()

                if curr_size < MAX_DS_SIZE and cur_acc > 0.8:
                    curr_size *= 2
                    with open(output_file, "a") as f:
                        f.write(f"step {i}, update config to {curr_size}" + "\n")
                    new_sample_counts = int(train_dataset_size * np.sqrt(curr_size))
                    train_data, test_data, clusters_data = get_datasets(curr_size, new_sample_counts)
                    break
