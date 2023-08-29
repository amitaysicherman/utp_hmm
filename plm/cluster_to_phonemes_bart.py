# sbatch --gres=gpu:1,vmem:24g --mem=75G -c4 --time=7-0 --wrap "python cluster_to_phonemes_bart.py"
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
from jiwer import wer
import argparse

save_update_steps = 1_000
warmup_steps = 50
EPOCHS = 1_000
last_config = False

parser = argparse.ArgumentParser()
parser.add_argument('--ds', type=str, default="tm")
parser.add_argument('--model_size', type=str, default="s")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--max_sample_size", type=int, default=10)
parser.add_argument("--max_length", type=int, default=512)
args = parser.parse_args()

BATCH_SIZE = args.batch_size
LR = args.lr
ds = args.ds
max_sample_size = args.max_sample_size
model_size = args.model_size
MAX_LENGTH = args.max_length

config_name = f"learn_mapping_bart_{ds}_{model_size}_{BATCH_SIZE}_{LR}_{max_sample_size}_{MAX_LENGTH}"

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
else:
    raise ValueError(f"Unknown size {model_size}")

if ds == "lr":
    phonemes_file = "data/lr_train.txt"
    phonemes_file_test = "data/lr_test.txt"
    MAX_DS_SIZE = 2 ** 17
    train_dataset_size = 100_000
    test_size = 1_000


else:
    phonemes_file = "data/TIMIT_NS_TRAIN_PH_IDX.txt"
    phonemes_file_test = "data/TIMIT_NS_TEST_PH_IDX.txt"
    MAX_DS_SIZE = 4000
    train_dataset_size = 1_000
    test_size = 100

    num_layers = 4

# gen_file = f"results/{config_name}_gen.txt"
output_file = f"results/{config_name}.txt"

ONE = 0
SPHERE = 2
PHONEMES_LAST_TOKEN = 38
CLUSTERS_FIRST_TOKEN = PHONEMES_LAST_TOKEN + 1
N_CLUSTERS = 100
CLUSTERS_LAST_TOKEN = CLUSTERS_FIRST_TOKEN + N_CLUSTERS
PAD_TOKEN = CLUSTERS_LAST_TOKEN + 1
SEP = PAD_TOKEN + 1
START_TOKEN = SEP + 1
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


def random_gaussian(n, dim=2):
    point = np.random.normal(size=(n, dim))
    point /= np.linalg.norm(point, axis=1, keepdims=True)
    return point


class ClustersPhonemesDataset(Dataset):
    def __init__(self, phonemes_file, clusters_file, max_len=MAX_LENGTH, samples_count=train_dataset_size):
        with open(phonemes_file, 'r') as f:
            phonemes_data = f.readlines()
        self.phonemes_data = [[int(x) for x in line.strip().split()] for line in phonemes_data]
        with open(clusters_file, 'r') as f:
            clusters_data = f.readlines()
        self.clusters_data = [[int(x) for x in line.strip().split()] for line in clusters_data]

        SIL_CLUSTERS = np.array([1, 2, 4, 10, 12, 20, 21, 22, 27, 31, 34, 37, 39, 40, 41, 47, 54,
                                 55, 56, 57, 60, 63, 66, 67, 71, 74, 78, 81, 83, 84, 86, 89, 92, 93,
                                 96])
        self.clusters_data = [[x for x in line if x not in SIL_CLUSTERS] for line in self.clusters_data]

        self.clusters_data = [[CLUSTERS_FIRST_TOKEN + x for x in line] for line in self.clusters_data]
        self.max_len = max_len
        self.data = []
        self.clusters = []
        for _ in range(samples_count):
            sample = [START_TOKEN]
            cluster_sample = [START_TOKEN]
            while len(sample) < self.max_len and len(cluster_sample) < self.max_len:
                new_index = np.random.randint(0, len(self.phonemes_data))
                sample += self.phonemes_data[new_index] + [SEP]
                cluster_sample = self.clusters_data[new_index] + [SEP]
            sample = sample[:self.max_len - 1] + [END_TOKEN]
            cluster_sample = cluster_sample[:self.max_len - 1] + [END_TOKEN]
            self.clusters.append(cluster_sample)
            self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.clusters[idx]), torch.LongTensor(self.data[idx])


class PhonemesDataset(Dataset):
    def __init__(self, phonemes_file, type_, dup, max_len=MAX_LENGTH, samples_count=train_dataset_size, size=-1):
        with open(phonemes_file, 'r') as f:
            phonemes_data = f.readlines()
        self.phonemes_data = [[int(x) for x in line.strip().split()] for line in phonemes_data]
        self.size = size
        self.max_len = max_len
        self.type = type_
        self.dup = dup
        self.samples_count = samples_count
        self.data = []

        self.build_data()

    def build_data(self):
        max_line_index = len(self.phonemes_data) if self.size == -1 else self.size
        max_line_index = min(max_line_index, len(self.phonemes_data))
        self.data = []
        for _ in range(self.samples_count):
            sample = [START_TOKEN]
            while len(sample) < self.max_len:

                new_sample = self.phonemes_data[np.random.randint(0, max_line_index)]
                if len(new_sample) > max_sample_size:
                    random_start = np.random.randint(0, len(new_sample) - max_sample_size)
                    new_sample = new_sample[random_start:random_start + max_sample_size]
                sample += new_sample
                sample += [SEP]
            sample = sample[:self.max_len - 1] + [END_TOKEN]
            self.data.append(sample)

    def update_config(self, type_, dup, size):
        self.type = type_
        self.dup = dup
        self.size = size
        self.build_data()

    def __len__(self):
        return len(self.data)

    def build_mapping_one(self):
        units_mapping = list(range(PHONEMES_LAST_TOKEN + 1))
        units_mapping += [random.randint(0, PHONEMES_LAST_TOKEN) for _ in
                          range(N_CLUSTERS - (PHONEMES_LAST_TOKEN + 1))]
        random.shuffle(units_mapping)
        units_mapping = np.array(units_mapping)
        inv_mapping = {i: [] for i in range(PHONEMES_LAST_TOKEN + 1)}
        for i, u in enumerate(units_mapping):
            inv_mapping[u].append(i)
        return inv_mapping

    def build_mapping_sphere(self):
        phonemes = random_gaussian(PHONEMES_LAST_TOKEN + 1)
        clusters = random_gaussian(N_CLUSTERS)
        cosine_distances = 100 * (1 - cdist(phonemes, clusters, metric='cosine'))
        probabilities = softmax(cosine_distances, axis=0)
        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
        np.random.shuffle(probabilities)
        return probabilities

    def build_mapping(self):
        if self.type == ONE:
            return self.build_mapping_one()
        elif self.type == SPHERE:
            return self.build_mapping_sphere()
        else:
            raise ValueError("Unknown type")

    def add_noise(self, clean):
        inv_mapping = self.build_mapping()

        values = np.arange(5)
        random_numbers = np.random.random(4)
        sorted_numbers = np.sort(np.concatenate(([0, 1], random_numbers)))
        weights = np.diff(sorted_numbers)
        np.random.shuffle(weights)
        if self.dup:
            length = random.choices(values, weights=weights, k=len(clean))

        else:
            length = [1] * len(clean)

        final_clean = [clean[0]]  # start token
        final_noise = [clean[0]]  # start token
        range_units = np.arange(N_CLUSTERS)
        finish = False
        for c in clean[1:]:
            if finish:
                break
            if c == SEP:
                final_clean.append(SEP)
                final_noise.append(SEP)
                if len(final_noise) == self.max_len - 1:
                    final_clean.append(END_TOKEN)
                    final_noise.append(END_TOKEN)
                    break

            elif c == END_TOKEN:
                final_clean.append(END_TOKEN)
                final_noise.append(END_TOKEN)
                break
            else:
                final_clean.append(c)
                for _ in range(length.pop()):

                    if self.type == ONE:
                        new_token = random.choice(inv_mapping[c])
                    else:  # self.type == SPHERE:
                        new_token = np.random.choice(range_units, p=inv_mapping[c])

                    new_token = CLUSTERS_FIRST_TOKEN + new_token
                    final_noise.append(new_token)
                    if len(final_noise) == self.max_len - 1:
                        final_clean.append(END_TOKEN)
                        final_noise.append(END_TOKEN)
                        finish = True
                        break

        final_clean += [PAD_TOKEN] * (MAX_LENGTH - len(final_clean))
        final_noise += [PAD_TOKEN] * (MAX_LENGTH - len(final_noise))

        return final_clean, final_noise

    def __getitem__(self, idx):
        clean, noise = self.add_noise(self.data[idx])
        clean = torch.LongTensor(clean)
        noise = torch.LongTensor(noise)
        return noise, clean


def step_config(cur_type, cur_dup, curr_size, score):
    is_update = False
    if score > 0.6:
        if curr_size < MAX_DS_SIZE:
            curr_size *= 2
            is_update = True
        elif cur_type == ONE:
            cur_type = SPHERE
            is_update = True
        elif cur_type == SPHERE and not cur_dup:
            cur_dup = True
            is_update = True
    return cur_type, cur_dup, curr_size, is_update


def get_model() -> BartForConditionalGeneration:
    config = BartConfig(vocab_size=N_TOKENS + 1, max_position_embeddings=MAX_LENGTH, encoder_layers=num_layers,
                        encoder_ffn_dim=d_model,
                        encoder_attention_heads=nhead, decoder_layers=num_layers, decoder_ffn_dim=d_model,
                        decoder_attention_heads=nhead,
                        d_model=d_model, pad_token_id=PAD_TOKEN, bos_token_id=START_TOKEN, eos_token_id=END_TOKEN,
                        decoder_start_token_id=START_TOKEN, forced_eos_token_id=END_TOKEN)  # Set vocab size
    model = BartForConditionalGeneration(config)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'{params:,} trainable parameters')
    return model


def get_datasets():
    train_dataset = PhonemesDataset(phonemes_file, type_=curr_type, dup=curr_dup,
                                    size=curr_size)
    train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataset = PhonemesDataset(phonemes_file_test, type_=curr_type, dup=curr_dup,
                                   size=-1, samples_count=test_size)
    test_data = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    clusters_dataset = ClustersPhonemesDataset(phonemes_file, phonemes_file.replace("PH_IDX", "CLUSTERS"),
                                               samples_count=curr_size)
    clusters_data = DataLoader(clusters_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return train_dataset, train_data, test_dataset, test_data, clusters_dataset, clusters_data


def save(model, optimizer, i, best_score, update_best, conf_type, conf_dup, conf_size):
    data_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': i,
        'best_score': best_score,
        'conf_type': conf_type,
        'conf_dup': conf_dup,
        'conf_size': conf_size
    }
    torch.save(data_save, f"models/{config_name}_last.cp")
    if update_best:
        torch.save(data_save, f"models/{config_name}_best.cp")


def load_last(model, optimizer):
    if not os.path.exists(f"models/{config_name}_last.cp"):
        return 0, 0, ONE, False, 2
    checkpoint = torch.load(f"models/{config_name}_last.cp", map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    load_step = checkpoint['step']
    best_score = checkpoint['best_score']
    conf_type = checkpoint['conf_type']
    conf_dup = checkpoint['conf_dup']
    conf_size = checkpoint['conf_size']
    return load_step, best_score, conf_type, conf_dup, conf_size


# main:
if __name__ == '__main__':

    model = get_model()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    i, best_test_acc, curr_type, curr_dup, curr_size = load_last(model, optimizer)
    print(
        f"load cp-  i:{i}, best_test_acc:{best_test_acc}, curr_type:{curr_type}, curr_dup:{curr_dup}, curr_size:{curr_size}")
    model = model.train()
    scores = Scores()
    for epoch in range(EPOCHS):
        train_dataset, train_data, test_dataset, test_data, clusters_dataset, clusters_data = get_datasets()
        pbar = tqdm(train_data, total=len(train_data))

        for x_train, y_train in pbar:
            i += 1
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            outputs = model(input_ids=x_train, labels=y_train, output_hidden_states=True)
            scores.update_values_from_output(outputs, y_train, "train")
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % save_update_steps == 0:
                model.eval()
                with torch.no_grad():
                    for x_test, y_test in test_data:
                        x_test = x_test.to(device)
                        y_test = y_test.to(device)
                        outputs = model(input_ids=x_test, labels=y_test, output_hidden_states=True)
                        scores.update_values_from_output(outputs, y_test, "test")
                    scores.to_file("test")

                    for x_test, y_test in clusters_data:
                        x_test = x_test.to(device)
                        y_test = y_test.to(device)
                        outputs = model(input_ids=x_test, labels=y_test, output_hidden_states=True)
                        scores.update_values_from_output(outputs, y_test, "cluster")
                    scores.to_file("cluster")
                model.train()
                update_best = False
                if scores.test_acc > best_test_acc:
                    best_test_acc = scores.test_acc / scores.test_count
                    update_best = True

                scores.resset_test()
                scores.reset_cluster()

                save(model, optimizer, i, best_test_acc, update_best, curr_type, curr_dup, curr_size)

            if i % warmup_steps == 0:
                scores.to_file("train")

                if not last_config:
                    curr_type, curr_dup, curr_size, is_update = step_config(curr_type, curr_dup, curr_size,
                                                                            scores.train_acc / scores.train_count)
                    if curr_type == SPHERE and curr_dup and curr_size == MAX_DS_SIZE:
                        last_config = True
                    if is_update:
                        with open(output_file, "a") as f:
                            f.write(f"step {i}, update config to {curr_type}, {curr_dup}, {curr_size}" + "\n")
                        config_update_counter = 0
                        break
                scores.resset_train()
