# sbatch --gres=gpu:4,vmem:24g --mem=75G --time=3-0 --wrap "python train.py"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from torch.nn.parallel import DataParallel
from x_transformers import TransformerWrapper, Encoder
import torch.nn.functional as F
from mapping import phonemes_to_index
import argparse
from conv_encoder import get_conv_encoder_model

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=["conv", "transformer"], default="conv")
parser.add_argument('--max_len', type=int, default=100)
parser.add_argument('--size', type=str, default="small", choices=["small", "medium", "large"])
parser.add_argument('--data_train', type=str, default="TIMIT_TRAIN_PH_dup")
parser.add_argument('--data_val', type=str, default="TIMIT_TRAIN_VAL_PH_dup")
parser.add_argument('--data_test', type=str, default="TIMIT_TEST_PH_dup")
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--drop_out', type=float, default=0.0)
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()
config_name = f"{args.model}_{args.size}_{args.data_train.replace('TRAIN', '')}"

INPUT_SIZE = len(phonemes_to_index) + 1
PADDING_VALUE = INPUT_SIZE


class Scores:
    def __init__(self, name):
        self.name = name
        self.loss = []
        self.acc = []

    def update(self, loss, acc):
        self.loss.append(loss)
        self.acc.append(acc)

    def __repr__(self):
        return f"\n{self.name} loss: {np.mean(self.loss)} \n{self.name} acc: {np.mean(self.acc)}"

    def reset(self):
        self.loss = []
        self.acc = []


class PhonemesDataset(Dataset):
    def __init__(self, prefix, max_len, padding_value=PADDING_VALUE):

        with open(f'{prefix}_clean.txt', 'r') as f:
            clean_data = f.read().splitlines()
        clean_data = [list(map(int, line.split())) for line in clean_data]
        with open(f'{prefix}_noise.txt', 'r') as f:
            noise_data = f.read().splitlines()
        noise_data = [list(map(int, line.split())) for line in noise_data]
        self.step = 0
        self.x = []
        self.y = []
        for i, (clean, noise) in tqdm(enumerate(zip(clean_data, noise_data)), total=len(clean_data)):
            assert len(clean) == len(noise)
            seq_len = len(clean)
            if seq_len > max_len:
                start_index = np.random.randint(0, seq_len - max_len)
                clean = clean[start_index:start_index + max_len]
                noise = noise[start_index:start_index + max_len]
                clean = np.array(list(clean), dtype=np.int8)
                noise = np.array(list(noise), dtype=np.int8)
            else:
                clean = np.array(list(clean) + [padding_value] * (max_len - len(clean)), dtype=np.int8)
                noise = np.array(list(noise) + [padding_value] * (max_len - len(noise)), dtype=np.int8)
            self.x.append(noise)
            self.y.append(clean)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.LongTensor(self.x[idx]), torch.LongTensor(self.y[idx])


def single_round(model, x, y, is_train, scorer):
    x = x.to(device)
    y = y.to(device)
    logits = model(x)
    loss = F.cross_entropy(
        logits.transpose(1, 2),
        y,
        ignore_index=PADDING_VALUE
    )
    if is_train:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    predicted_labels = torch.argmax(logits, dim=-1)
    predicted_labels = predicted_labels[y != PADDING_VALUE]
    y = y[y != PADDING_VALUE]
    acc = (predicted_labels == y).sum().item() / y.numel()
    scorer.update(loss.item(), acc)


def get_model(arc, size, max_len, dropout, vocab=INPUT_SIZE):
    if arc == "transformer":
        if size == "small":
            d_model = 256
            nhead = 4
            num_layers = 6
        else:
            d_model = 768
            nhead = 12
            num_layers = 12
        return TransformerWrapper(
            num_tokens=vocab + 1,
            max_seq_len=max_len,
            emb_dropout=dropout,
            attn_layers=Encoder(
                dim=d_model,
                depth=num_layers,
                heads=nhead,
                layer_dropout=dropout,
                attn_dropout=dropout,
                ff_dropout=dropout
            )
        )
    else:
        return get_conv_encoder_model(size)


if __name__ == '__main__':
    model = get_model(args.model, args.size, args.max_len, args.drop_out)

    print(model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'{params:,} trainable parameters')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    criterion = nn.CrossEntropyLoss().to(device)

    train_dataset = PhonemesDataset(args.data_train, args.max_len)
    val_dataset = PhonemesDataset(args.data_val, args.max_len)
    test_dataset = PhonemesDataset(args.data_test, args.max_len)

    train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    val_data = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_data = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        train_scores = Scores("train")
        val_scores = Scores("val")
        test_scores = Scores("test")
        model.train()
        for (x, y) in tqdm(train_data):
            single_round(model, x, y, True, train_scores)

        model.eval()
        with torch.no_grad():
            for (x, y) in tqdm(val_data):
                single_round(model, x, y, False, val_scores)
            for (x, y) in tqdm(test_data):
                single_round(model, x, y, False, test_scores)

        cp_name = f"models/{config_name}_{epoch}.cp"
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), cp_name)
        else:
            torch.save(model.state_dict(), cp_name)
        torch.save(optimizer.state_dict(), cp_name.replace(".cp", "_opt.cp"))

        print("Epoch", epoch)
        print(train_scores)
        print(val_scores)
        print(test_scores)

        with open(f"results_{config_name}.txt", "a") as f:
            f.write(f"Epoch {epoch}\n")
            f.write(f"{train_scores}\n")
            f.write(f"{val_scores}\n")
            f.write(f"{test_scores}\n")
            f.write("\n")

        train_scores.reset()
        val_scores.reset()
        test_scores.reset()
