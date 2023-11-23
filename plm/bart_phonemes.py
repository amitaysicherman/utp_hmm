# running exmplae:
# python bart_phonemes.py --model_size s --lr 0.0001 --p_eq 0.5 --p_add 0.05 --p_del 0.25 --p_rep 0.2
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


@dataclass
class Noiser:
    p_eq: float = 0.5
    p_add: float = 0.05
    p_del: float = 0.25
    p_rep: float = 0.2

    def normalize(self):
        s = self.p_eq + self.p_add + self.p_del + self.p_rep
        self.p_eq /= s
        self.p_add /= s
        self.p_add += self.p_eq
        self.p_del /= s
        self.p_del += self.p_add
        self.p_rep /= s
        self.p_rep += self.p_del

    def from_args(self, args):
        self.p_eq = args.p_eq
        self.p_add = args.p_add
        self.p_del = args.p_del
        self.p_rep = args.p_rep
        self.normalize()

    def add_noise_to_seq(self, values):
        noise = []
        for x in values:
            new_x = self.noise_single(x)
            if new_x:
                noise.extend(new_x)
        return noise

    def noise_single(self, x):
        rnd = random.random()
        if rnd < self.p_eq:
            return [x]
        if random.random() < self.p_add:
            return [x, random.randint(0, N_PHONEMES - 1)]
        if random.random() < self.p_del:
            return None
        else:  # replace
            return [random.randint(0, N_PHONEMES - 1)]

    def __repr__(self):
        return f"{self.p_eq}_{self.p_add}_{self.p_del}_{self.p_rep}"


parser = argparse.ArgumentParser()
parser.add_argument('--model_size', type=str, default="s")
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--p_eq", type=float, default=0.5)
parser.add_argument("--p_add", type=float, default=0.05)
parser.add_argument("--p_del", type=float, default=0.25)
parser.add_argument("--p_rep", type=float, default=0.2)
args = parser.parse_args()
LR = args.lr
MAX_LENGTH = 256
save_update_steps = 1000
warmup_steps = 500
steps = 250_000
train_file = "data/LIBRISPEECH_TRAIN_idx.txt"
test_file = "data/LIBRISPEECH_TEST_idx.txt"
clusters_test_file = "data/LIBRISPEECH_TEST_clusters_100.txt"

noiser = Noiser()
noiser.from_args(args)
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

if args.model_size == "s":
    d_model = 256
    nhead = 4
    num_layers = 3
    BATCH_SIZE = 64

elif args.model_size == "m":
    d_model = 512
    nhead = 8
    num_layers = 6
    BATCH_SIZE = 8

config_name = f"bart_phonemes/{args.model_size}_{LR}_{noiser}"
os.makedirs(f"results/{config_name}", exist_ok=True)
os.makedirs(f"models/{config_name}", exist_ok=True)
writer = SummaryWriter(f"results/{config_name}")
results_file = f"results/{config_name}/results.txt"
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
        with open(results_file, "a") as f:
            f.write(f"{i},{self.name},{loss},{acc},{wer_score}\n")
        self.reset()


class PhonemesLettersDataset(Dataset):
    def __init__(self, phonemes_file, cluster_file=""):
        with open(phonemes_file, 'r') as f:
            clean_phonemes = f.read().splitlines()
        clean_phonemes = [[int(x) for x in line.strip().split()] for line in clean_phonemes]

        if cluster_file:
            noise_phonemes = self.read_clusters(cluster_file)
        else:
            clean_phonemes, noise_phonemes = self.add_noise(clean_phonemes)

        self.clean = []
        self.noise = []
        for c, n in zip(clean_phonemes, noise_phonemes):
            if len(c) > MAX_LENGTH or len(n) > MAX_LENGTH:
                continue
            self.clean.append([START_TOKEN] + c + [END_TOKEN] + [PAD_TOKEN] * (MAX_LENGTH - len(c)))
            self.noise.append([START_TOKEN] + n + [END_TOKEN] + [PAD_TOKEN] * (MAX_LENGTH - len(n)))

    def read_clusters(self, cluster_file):
        with open(cluster_file, 'r') as f:
            clusters_ = f.read().splitlines()
        clusters = []
        for line in clusters_:
            line = line.strip().split()
            line = [clusters_to_phonemes[int(x)] for x in line]
            line = [x for x in line if x != SUPERV_BLANK]
            line = [x for i, x in enumerate(line) if i == 0 or x != line[i - 1]]
            clusters.append(line)
        return clusters

    def add_noise(self, clean):
        noise = []
        for line in clean:
            line = noiser.add_noise_to_seq(line[:])
            noise.append(line)
        return clean, noise

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        return torch.LongTensor(self.noise[idx]), torch.LongTensor(self.clean[idx])


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
    train_dataset = DataLoader(PhonemesLettersDataset(train_file), batch_size=BATCH_SIZE,
                               shuffle=True, drop_last=True)
    test_dataset = DataLoader(PhonemesLettersDataset(test_file), batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True)

    test_clusters_dataset = DataLoader(PhonemesLettersDataset(test_file, clusters_test_file),
                                       batch_size=BATCH_SIZE,
                                       shuffle=True, drop_last=True)

    train_scores = Scores("train")
    test_scores = Scores("test")
    cluster_test_scores = Scores("cluster_test")

    while i < steps:
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
