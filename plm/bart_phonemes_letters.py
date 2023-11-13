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

import argparse
from jiwer import wer

parser = argparse.ArgumentParser()
parser.add_argument('--model_size', type=str, default="m")
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--noise", type=float, default=0.5)
args = parser.parse_args()

LR = args.lr
MAX_LENGTH = 256
save_update_steps = 1_000
warmup_steps = 50
EPOCHS = 1_000
letters_train_file = "data/LIBRISPEECH_TRAIN_letters.txt"
letters_test_file = "data/LIBRISPEECH_TEST_letters.txt"
phonemes_train_file = "data/LIBRISPEECH_TRAIN_idx.txt"
phonemes_test_file = "data/LIBRISPEECH_TEST_idx.txt"
clusters_test_file = "data/LIBRISPEECH_TEST_clusters.txt"
noise = args.noise
LETTERS_LAST_TOKEN = 29
CLUSTERS_FIRST_TOKEN = LETTERS_LAST_TOKEN + 1
N_CLUSTERS = 100
CLUSTERS_LAST_TOKEN = CLUSTERS_FIRST_TOKEN + N_CLUSTERS
PAD_TOKEN = CLUSTERS_LAST_TOKEN + 1
START_TOKEN = PAD_TOKEN + 1
END_TOKEN = START_TOKEN + 1
N_TOKENS = END_TOKEN + 1
clusters_to_phonemes = np.array(
    [10, 39, 39, 13, 39, 11, 28, 28, 0, 0, 39, 17, 39, 5, 35, 28, 21, 20, 22, 20, 39, 39, 39, 16, 0, 27, 22, 39, 37, 37,
     27, 39, 5, 19, 39, 20, 28, 39, 16, 39, 39, 39, 16, 0, 22, 17, 0, 39, 0, 28, 21, 0, 27, 16, 39, 39, 39, 39, 5, 19,
     39, 0, 16, 39, 17, 29, 39, 39, 2, 33, 35, 39, 0, 2, 39, 15, 35, 34, 39, 11, 22, 39, 9, 39, 39, 30, 39, 1, 23, 39,
     20, 1, 39, 39, 12, 29, 39, 24, 36, 9])

if args.model_size == "s":
    d_model = 256
    nhead = 4
    num_layers = 3

    BATCH_SIZE = 256

elif args.model_size == "m":
    d_model = 512
    nhead = 8
    num_layers = 6
    BATCH_SIZE = 64

elif args.model_size == "l":
    d_model = 1024
    nhead = 16
    num_layers = 12
    BATCH_SIZE = 8

config_name = f"bart_phonemes_letters/{args.model_size}_{LR}_{noise}"
os.makedirs(f"results/{config_name}", exist_ok=True)
os.makedirs(f"models/{config_name}", exist_ok=True)
writer = SummaryWriter(f"results/{config_name}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Scores:
    name: str
    loss = 0.0
    acc = 0.0
    count = 0
    wer_score = 0.0

    def reset(self):
        self.loss = 0.0
        self.acc = 0.0
        self.wer_score = 0.0
        self.count = 0.0

    def update_value(self, loss, acc, wer_score):
        self.loss += loss
        self.acc += acc
        self.wer_score += wer_score
        self.count += 1

    def update_values_from_output(self, outputs, y):
        loss = outputs.loss.mean().item()
        preds = outputs.logits.argmax(dim=-1)
        mask = y != PAD_TOKEN
        acc = ((preds[mask] == y[mask]).sum() / y[mask].numel()).item()

        new_wer = []
        for i in range(len(preds)):
            pred = preds[i].detach().cpu().numpy().tolist()
            pred = " ".join([str(x) for x in pred if x not in [PAD_TOKEN, START_TOKEN, END_TOKEN]])
            true = y[i].detach().cpu().numpy().tolist()
            true = " ".join([str(x) for x in true if x not in [PAD_TOKEN, START_TOKEN, END_TOKEN]])
            new_wer.append(wer(true, pred))
        new_wer = np.mean(new_wer)

        self.update_value(loss, acc, new_wer)

    def get_scores(self):
        loss = self.loss / self.count if self.count > 0 else 0
        acc = self.acc / self.count if self.count > 0 else 0
        wer_score = self.wer_score / self.count if self.count > 0 else 0

        return loss, acc, wer_score

    def to_file(self, i):
        loss, acc, wer_score = self.get_scores()
        writer.add_scalar(f'{self.name}_loss', loss, i)
        writer.add_scalar(f'{self.name}_acc', acc, i)
        writer.add_scalar(f'{self.name}_wer', wer_score, i)
        self.reset()


class PhonemesLettersDataset(Dataset):
    def __init__(self, phonemes_file, letters_file, superv_clusters=False):
        with open(phonemes_file, 'r') as f:
            clusters = f.readlines()
        if superv_clusters:
            clusters = [[CLUSTERS_FIRST_TOKEN + clusters_to_phonemes[int(x)] for x in line.strip().split()] for line in
                        clusters]

        else:
            clusters = [[CLUSTERS_FIRST_TOKEN + int(x) for x in line.strip().split()] for line in clusters]
            if noise > 0:
                for i in range(len(clusters)):
                    for j in range(len(clusters[i])):
                        if random.random() < noise:
                            clusters[i][j] = random.randint(CLUSTERS_FIRST_TOKEN, CLUSTERS_LAST_TOKEN)

        with open(letters_file, 'r') as f:
            letters = f.readlines()
        letters = [[int(x) for x in line.strip().split()] for line in letters]

        self.clusters = []
        self.letters = []
        for clus, let in zip(clusters, letters):
            if len(clus) > MAX_LENGTH or len(let) > MAX_LENGTH:
                continue
            self.clusters.append([START_TOKEN] + clus + [END_TOKEN] + [PAD_TOKEN] * (MAX_LENGTH - len(clus)))
            self.letters.append([START_TOKEN] + let + [END_TOKEN] + [PAD_TOKEN] * (MAX_LENGTH - len(let)))

    def __len__(self):
        return len(self.letters)

    def __getitem__(self, idx):
        return torch.LongTensor(self.clusters[idx]), torch.LongTensor(self.letters[idx])


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


def save(model, optimizer, i):
    data_save = {
        'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': i,
    }
    torch.save(data_save, f"models/{config_name}_last.cp")


def load_last(model, optimizer):
    if not os.path.exists(f"models/{config_name}_last.cp"):
        return 0
    checkpoint = torch.load(f"models/{config_name}_last.cp", map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    load_step = checkpoint['step']
    return load_step


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

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    i = load_last(model, optimizer)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    print(f"load cp-  i:{i}")
    model = model.train()
    train_dataset = DataLoader(PhonemesLettersDataset(phonemes_train_file, letters_train_file), batch_size=BATCH_SIZE,
                               shuffle=True, drop_last=True)
    test_dataset = DataLoader(PhonemesLettersDataset(phonemes_test_file, letters_test_file), batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True)

    test_clusters_dataset = DataLoader(PhonemesLettersDataset(clusters_test_file, letters_test_file, True),
                                       batch_size=BATCH_SIZE,
                                       shuffle=True, drop_last=True)

    train_scores = Scores("train")
    test_scores = Scores("test")
    cluster_test_scores = Scores("cluster_test")

    for epoch in range(EPOCHS):
        pbar = tqdm(train_dataset, total=len(train_dataset))
        for x_train, y_train in pbar:
            i += 1
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            outputs = model(input_ids=x_train, labels=y_train, output_hidden_states=True)
            loss = outputs.loss
            train_scores.update_values_from_output(outputs, y_train)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            if i % save_update_steps == 0:
                model.eval()
                with torch.no_grad():
                    eval_test_dataset(model, test_dataset, test_scores)
                    eval_test_dataset(model, test_clusters_dataset, cluster_test_scores)
                model.train()
                save(model, optimizer, i)

            if i % warmup_steps == 0:
                _ = train_scores.get_scores()

                train_scores.to_file(i)
