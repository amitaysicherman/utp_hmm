# sbatch --gres=gpu:4,vmem:24g --mem=75G -c4 --time=7-0 --wrap "python cluster_to_phonemes_encoder.py"
# https://aclanthology.org/P19-2049.pdf

import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from x_transformers import TransformerWrapper, Encoder
import torch.nn.functional as F

import numpy as np
import torch
from tqdm import tqdm
import os
from scipy.special import softmax
from scipy.spatial.distance import cdist
from dataclasses import dataclass
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch import nn

save_update_steps = 1_000
warmup_steps = 50
EPOCHS = 1_000
last_config = False

parser = argparse.ArgumentParser()
parser.add_argument('--ds', type=str, default="lr")
parser.add_argument('--model_size', type=str, default="m")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--max_length", type=int, default=256)
args = parser.parse_args()
BATCH_SIZE = args.batch_size
LR = args.lr
ds = args.ds
model_size = args.model_size
MAX_LENGTH = args.max_length
MAX_DS_SIZE = 1_048_576
config_name = f"encoder_all_{ds}_{model_size}_{BATCH_SIZE}_{LR}_{MAX_LENGTH}"
writer = SummaryWriter(f"results/{config_name}")

train_dataset_size = 1_000_000
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
    clusters_train_file = "data/LIBRISPEECH_TRAIN_clusters.txt"
    clusters_test_file = "data/LIBRISPEECH_TEST_clusters.txt"

else:
    phonemes_file = "data/TIMIT_NS_TRAIN_PH_IDX.txt"
    phonemes_file_test = "data/TIMIT_NS_TEST_PH_IDX.txt"
    clusters_train_file = "data/TIMIT_NS_TRAIN_CLUSTERS.txt"
    clusters_test_file = "data/TIMIT_NS_TEST_CLUSTERS.txt"

output_file = f"results/{config_name}.txt"

ONE = 0
SPHERE = 2
PHONEMES_LAST_TOKEN = 38
CLUSTERS_FIRST_TOKEN = PHONEMES_LAST_TOKEN + 1
N_CLUSTERS = 100
CLUSTERS_LAST_TOKEN = CLUSTERS_FIRST_TOKEN + N_CLUSTERS
PAD_TOKEN = CLUSTERS_LAST_TOKEN + 1
SEP = PAD_TOKEN + 1
N_TOKENS = SEP + 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Scores:
    name: str
    loss = 0.0
    acc = 0.0
    count = 0

    def reset(self):
        self.loss = 0.0
        self.acc = 0.0
        self.count = 0.0

    def update_value(self, loss, acc):
        self.loss += loss
        self.acc += acc
        self.count += 1

    def update_values_from_output(self, logit, loss, y):
        preds = logit.argmax(dim=-1)
        mask = y != PAD_TOKEN
        acc = ((preds[mask] == y[mask]).sum() / y[mask].numel()).item()
        self.update_value(loss, acc)

    def get_scores(self):
        loss = self.loss / self.count if self.count > 0 else 0
        acc = self.acc / self.count if self.count > 0 else 0
        return loss, acc

    def to_file(self, i):
        loss, acc = self.get_scores()
        writer.add_scalar(f'{self.name}_loss', loss, i)
        writer.add_scalar(f'{self.name}_acc', acc, i)
        self.reset()


SIL_CLUSTERS = np.array([1, 2, 4, 10, 12, 20, 21, 22, 27, 31, 34, 37, 39, 40, 41, 47, 54,
                         55, 56, 57, 60, 63, 66, 67, 71, 74, 78, 81, 83, 84, 86, 89, 92, 93,
                         96])


class NoiseAdder:
    def __init__(self, size: int, single_seed=-1):
        self.range_units = np.arange(N_CLUSTERS)
        self.maps = []
        if single_seed != -1:
            random.seed(single_seed)
            np.random.seed(single_seed)
            self.maps.append(self.build_mapping())
            return
        else:
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

    def add_noise(self, clean_sample, max_noise_len):

        inv_mapping = random.choice(self.maps)
        length = self.get_length(len(clean_sample))
        noise_sample = []
        new_clean_sample = []
        clean_count = 0
        for c in clean_sample:
            for _ in range(length.pop()):

                if clean_count >= max_noise_len:
                    break
                clean_count += 1
                new_token = self.get_new_token(inv_mapping, c)
                if len(noise_sample) and new_token == noise_sample[-1]:
                    continue
                noise_sample.append(new_token)
                new_clean_sample.append(c)
        return new_clean_sample, noise_sample, clean_count


class PhonemesDataset(Dataset):
    def __init__(self, phonemes_file, max_len=MAX_LENGTH, samples_count=train_dataset_size, size=1, single_seed=-1):
        with open(phonemes_file, 'r') as f:
            phonemes_data = f.readlines()
        self.phonemes_data = [[int(x) for x in line.strip().split()] for line in phonemes_data]
        self.phonemes_data = [x for x in self.phonemes_data if len(x) < MAX_LENGTH - 3]
        self.max_len = max_len
        self.samples_count = samples_count
        self.noise_adder = NoiseAdder(size=size, single_seed=single_seed)
        self.max_line_index = len(self.phonemes_data)

    def get_clean_noise_sample(self):
        concat_clean_samples = []
        concat_noise_samples = []
        while True:
            clean_sample = self.phonemes_data[np.random.randint(0, self.max_line_index)][:]
            max_noise_len = self.max_len - len(concat_noise_samples) - 2
            if max_noise_len <= 0:
                break
            clean_sample, noise_sample, max_clean_len = self.noise_adder.add_noise(clean_sample, max_noise_len)
            clean_sample = clean_sample[:max_clean_len]
            clean_sample += [SEP]
            noise_sample += [SEP]

            if len(concat_clean_samples) + len(clean_sample) >= self.max_len - 1:
                break
            concat_clean_samples += clean_sample
            concat_noise_samples += noise_sample

        concat_clean_samples += [PAD_TOKEN] * (self.max_len - len(concat_clean_samples))
        concat_noise_samples += [PAD_TOKEN] * (self.max_len - len(concat_noise_samples))
        return concat_clean_samples, concat_noise_samples

    def __len__(self):
        return self.samples_count

    def __getitem__(self, idx):
        clean, noise = self.get_clean_noise_sample()
        clean = torch.LongTensor(clean)
        noise = torch.LongTensor(noise)
        return noise, clean


def get_model() -> TransformerWrapper:
    model = TransformerWrapper(
        num_tokens=N_TOKENS + 1,
        max_seq_len=MAX_LENGTH,
        logits_dim=N_TOKENS + 1,
        attn_layers=Encoder(
            dim=d_model,
            depth=num_layers,
            heads=nhead,
        )
    )
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'{params:,} trainable parameters')
    return model


def get_datasets(curr_size, train_sample_count):

    train_dataset = PhonemesDataset(phonemes_file, size=curr_size, samples_count=train_sample_count)
    train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    test_dataset = PhonemesDataset(phonemes_file_test, size=curr_size, samples_count=test_size)
    test_data = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    test_map_dataset = PhonemesDataset(phonemes_file, size=curr_size, samples_count=test_size,
                                       single_seed=curr_size * 2)
    test_map_data = DataLoader(test_map_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    return train_data, test_data, test_map_data


def save(model, optimizer, i):
    data_save = {
        'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': i,
    }
    torch.save(data_save, f"models/{config_name}_last.cp")


def load_last(model, optimizer):
    if not os.path.exists(f"models/{config_name}_last.cp"):
        return 0, 1
    checkpoint = torch.load(f"models/{config_name}_last.cp", map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    load_step = checkpoint['step']
    return load_step


def eval_test_dataset(model, dataset, score):
    for x_test, y_test in dataset:
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        logits = model(x_test)
        loss = F.cross_entropy(
            logits.transpose(1, 2),
            y_test,
            ignore_index=PAD_TOKEN
        )
        score.update_values_from_output(logits, loss.item(), y_test)

    score.to_file(i)


# main:
if __name__ == '__main__':

    model = get_model()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    i = load_last(model, optimizer)
    curr_size = MAX_DS_SIZE
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    print(
        f"load cp-  i:{i},   curr_size:{curr_size}")
    model = model.train()
    train_data, test_data, test_map_data = get_datasets(curr_size, train_dataset_size)
    train_scores = Scores("train")
    test_scores = Scores("test")
    test_map_scores = Scores("test_map")

    for epoch in range(EPOCHS):
        pbar = tqdm(train_data, total=len(train_data))
        for x_train, y_train in pbar:
            i += 1
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            logits = model(x_train)
            loss = F.cross_entropy(
                logits.transpose(1, 2),
                y_train,
                ignore_index=PAD_TOKEN
            )
            train_scores.update_values_from_output(logits, loss.item(), y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % save_update_steps == 0:
                model.eval()
                with torch.no_grad():
                    eval_test_dataset(model, test_data, test_scores)
                    eval_test_dataset(model, test_map_data, test_map_scores)
                model.train()
                save(model, optimizer, i)

            if i % warmup_steps == 0:
                _, cur_acc = train_scores.get_scores()
                train_scores.to_file(i)

                if curr_size < MAX_DS_SIZE and cur_acc > 0.75:
                    curr_size *= 2
                    writer.add_scalar('ds_size', curr_size, i)
                    new_sample_counts = int(train_dataset_size * np.sqrt(curr_size))
                    train_data, test_data, test_map_data = get_datasets(curr_size, new_sample_counts)
                    break
