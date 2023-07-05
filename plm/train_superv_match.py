# sbatch --gres=gpu:1,vmem:24g --mem=75G --time=0:30:0 --wrap "python train_superv_match.py"
import torch
import torch.nn as nn
import torch.optim as optim
from train import input_size, padding_value
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from mapping import phonemes_to_index
import torch.nn.functional as F
import argparse
from jiwer import wer

input_dim = 768
phonemes_count = input_size - 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

max_len = 100
batch_size = 2048
ephocs = 50
lr = 0.01
prefix = "./pseg/data/sup_vad/"
features_path = f"{prefix}features.npy"
len_path = f"{prefix}features.length"
phonemes_path = f"{prefix}features.phonemes"


def eval_mapping(x, y):
    wer_score = []
    for c, p in zip(x, y):
        c = list(c.detach().cpu().numpy())
        p = list(p.detach().cpu().numpy())

        c = [c[0]] + [c[i] for i in range(1, len(c)) if c[i] != c[i - 1]]
        p = [p[0]] + [p[i] for i in range(1, len(p)) if p[i] != p[i - 1]]

        c = " ".join([str(c_) for c_ in c if c_ != padding_value])
        p = " ".join([str(p_) for p_ in p if p_ != padding_value])
        p = " ".join([str(xx) for xx in p])
        c = " ".join([str(xx) for xx in c])
        wer_score.append(wer(p, c))
    return 100 * (1 - np.mean(wer_score))


def pad_array(data, max_len=max_len):
    n = len(data)
    if len(data) > max_len:
        return data[:max_len]
    else:
        desired_shape = (max_len, input_dim)
        pad_width = ((0, max(0, desired_shape[0] - n)), (0, 0))
        return np.pad(data, pad_width, mode='constant')


def pad_seq(data, max_len, padding_value):
    x = []
    for d in tqdm(data):
        sequence = d
        if len(sequence) > max_len:
            sequence = np.array(list(sequence[:max_len]))
        else:
            sequence = np.array(list(sequence) + [padding_value] * (max_len - len(sequence)))
        x.append(sequence)
    return x


class UnitsDataset(Dataset):
    def __init__(self, features_path=features_path, len_path=len_path,
                 phonemes_path=phonemes_path, max_len=max_len,
                 padding_value=padding_value):
        self.features = np.load(features_path)
        with open(len_path, 'r') as f:
            length = f.read().splitlines()

        length = [int(x) for x in length]
        cur = 0
        self.x = []
        self.len_x = []
        for l in length:
            self.len_x.append(l)
            self.x.append(pad_array(self.features[cur:cur + l]))
            cur += l
        with open(phonemes_path, 'r') as f:
            lines = f.read().splitlines()
        self.len_y = []
        data = []
        for line in lines:
            self.len_y.append(len(line.split()))
            data.append(
                [phonemes_to_index[x.upper()] if x.upper() != "DX" else phonemes_to_index["T"] for x in line.split()])

        self.y = pad_seq(data, max_len, padding_value)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.x[idx]), torch.LongTensor(self.y[idx]), torch.LongTensor(
            [self.len_x[idx]]), torch.LongTensor([self.len_y[idx]])


# Define the linear model
class LinearModel(nn.Module):

    def __init__(self, input_dim=input_dim, output_dim=padding_value + 1):
        super(LinearModel, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

        x = np.load("models/linear_14.npz")
        w = torch.FloatTensor(x['w'])
        b = torch.FloatTensor(x['b'])
        self.lin.weight.data.copy_(w)
        self.lin.bias.data.copy_(b)

    def forward(self, x):
        x = self.lin(x)
        return x


loss_fn = nn.CTCLoss(blank=padding_value).to(device)
train_data = DataLoader(UnitsDataset(), batch_size=batch_size, shuffle=False, drop_last=True)

linear_model = LinearModel()
linear_model.to(device)
optimizer = optim.Adam(linear_model.parameters(), lr=lr)
loss_all = []
acc_all = []
for ephoc in tqdm(range(ephocs)):
    e_loss = []
    e_acc = []
    e_acc_m = []
    for j, (x, y, len_x, len_y) in tqdm(enumerate(train_data)):
        x = x.to(device)
        y = y.to(device)
        len_x = len_x.to(device).flatten()
        len_y = len_y.to(device).flatten()
        linear_output = linear_model(x)

        loss = F.ctc_loss(
            linear_output.log_softmax(dim=-1).transpose(0, 1),
            y,
            len_x,
            len_y,
            blank=padding_value,
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        argmax_output = linear_output.argmax(dim=-1)
        e_acc.append(eval_mapping(argmax_output, y))
        e_loss.append(loss.item())

    print(f"loss: {e_loss[-1]}, acc: {e_acc[-1]}")

    torch.save(linear_model.state_dict(), f"./models/linear_{ephoc}.cp")
