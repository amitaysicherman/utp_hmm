# sbatch --gres=gpu:1,vmem:24g --mem=75G -c4 --time=7-0 --wrap "python superv_clusters_letters.py.py"
# https://aclanthology.org/P19-2049.pdf

import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel

import numpy as np
import torch
from tqdm import tqdm
import os
from transformers import BartConfig, BartForConditionalGeneration
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE = 64
LR = 0.001
MAX_LENGTH = 256
save_update_steps = 1_000
warmup_steps = 50
EPOCHS = 1_000
config_name = "super_clustering"
writer = SummaryWriter(f"results/super_clustering")
letters_train_file = "data/LIBRISPEECH_TRAIN_letters.txt"
letters_test_file = "data/LIBRISPEECH_TEST_letters.txt"
clusters_train_file = "data/LIBRISPEECH_TRAIN_clusters.txt"
clusters_test_file = "data/LIBRISPEECH_TEST_clusters.txt"

LETTERS_LAST_TOKEN = 29
CLUSTERS_FIRST_TOKEN = LETTERS_LAST_TOKEN + 1
N_CLUSTERS = 100
CLUSTERS_LAST_TOKEN = CLUSTERS_FIRST_TOKEN + N_CLUSTERS
PAD_TOKEN = CLUSTERS_LAST_TOKEN + 1
START_TOKEN = PAD_TOKEN + 1
END_TOKEN = START_TOKEN + 1
N_TOKENS = END_TOKEN + 1

d_model = 512
nhead = 8
num_layers = 8

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

    def update_values_from_output(self, outputs, y):
        loss = outputs.loss.mean().item()
        preds = outputs.logits.argmax(dim=-1)
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


class ClustersLettersDataset(Dataset):
    def __init__(self, clusters_file, letters_file):
        with open(clusters_file, 'r') as f:
            clusters = f.readlines()
        clusters = [[CLUSTERS_FIRST_TOKEN + int(x) for x in line.strip().split()] for line in clusters]
        with open(letters_file, 'r') as f:
            letters = f.readlines()
        letters = [[int(x) for x in line.strip().split()] for line in letters]

        self.clusters = []
        self.letters = []
        for clus, let in zip(clusters, letters):
            if len(clus) > MAX_LENGTH or len(let) > MAX_LENGTH:
                continue
            self.clusters.append(clus + [PAD_TOKEN] * (MAX_LENGTH - len(clus)))
            self.letters.append(let + [PAD_TOKEN] * (MAX_LENGTH - len(let)))

    def __len__(self):
        return len(self.letters)

    def __getitem__(self, idx):
        return torch.LongTensor(self.clusters[idx]), torch.LongTensor(self.letters[idx])


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
    conf_size = checkpoint['conf_size']
    return load_step, conf_size


def eval_test_dataset(model, dataset, score):
    for x_test, y_test in dataset:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        outputs = model(input_ids=x_test, labels=y_test, output_hidden_states=True)
        score.update_values_from_output(outputs, y_test)
    score.to_file(i)


# main:
if __name__ == '__main__':

    model = get_model()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    i, curr_size = load_last(model, optimizer)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    print(
        f"load cp-  i:{i},   curr_size:{curr_size}")
    model = model.train()
    train_dataset = DataLoader(ClustersLettersDataset(clusters_train_file, letters_train_file), batch_size=BATCH_SIZE,
                               shuffle=True, drop_last=True)
    test_dataset = DataLoader(ClustersLettersDataset(clusters_test_file, letters_test_file), batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True)
    train_scores = Scores("train")
    test_scores = Scores("test")
    test_map_scores = Scores("test_map")
    cluster_train_scores = Scores("cluster_train")
    cluster_test_scores = Scores("cluster_test")

    for epoch in range(EPOCHS):
        pbar = tqdm(train_dataset, total=len(train_dataset))
        for x_train, y_train in pbar:
            i += 1
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            outputs = model(input_ids=x_train, labels=y_train, output_hidden_states=True)
            loss = outputs.loss.mean()
            train_scores.update_values_from_output(outputs, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % save_update_steps == 0:
                model.eval()
                with torch.no_grad():
                    eval_test_dataset(model, test_dataset, test_scores)
                model.train()
                save(model, optimizer, i)

            if i % warmup_steps == 0:
                _, cur_acc = train_scores.get_scores()
                train_scores.to_file(i)
