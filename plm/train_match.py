import torch
import torch.nn as nn
import torch.optim as optim
from train import input_size, max_len, padding_value
from plm.utils import get_model
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from mapping import phonemes_to_index, mis_index
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

unit_count = 100
phonemes_count = input_size - 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cp_file = "./models/prep_random_small_timit_99.cp"#timit_small_20.cp"

units_padding_value = unit_count
batch_size = 2048
ephocs = 50
lr = 0.1


def get_random_mapping():
    units_to_phonemes = np.zeros((unit_count + 1, padding_value + 1))
    for i in range(len(units_to_phonemes)):
        units_to_phonemes[i] = np.random.dirichlet(np.ones(padding_value + 1), size=1)
        units_to_phonemes[i] /= units_to_phonemes[i].sum()
    return units_to_phonemes


def get_superv_mapping(random_count=100):
    if random_count == 100:
        return get_random_mapping()

    with open("../data/code100.txt") as f:
        code100 = f.read().splitlines()
    code100 = [[int(y) for y in x.split()] for x in code100]

    with open("../data/phonemes.txt") as f:
        phonemes = f.read().splitlines()
    phonemes = [[phonemes_to_index[y.upper()] if y != "dx" else phonemes_to_index['T'] for y in x.split()] for x in
                phonemes]

    units_to_phonemes = np.zeros((unit_count + 1, padding_value + 1))
    for i, (u, p) in enumerate(tqdm(zip(sum(code100, []), sum(phonemes, [])))):
        units_to_phonemes[u, p] += 1
    for i in range(unit_count + 1):
        units_to_phonemes[i] /= units_to_phonemes[i].sum()
    units_to_phonemes[unit_count, padding_value] = 1
    random_units_choise = random.sample(range(unit_count), k=random_count)
    for i in random_units_choise:
        units_to_phonemes[i] = np.random.dirichlet(np.ones(padding_value + 1), size=1)
        units_to_phonemes[i] /= units_to_phonemes[i].sum()
    print(units_to_phonemes.shape)
    return units_to_phonemes


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
    def __init__(self, data_path='../data/code100.txt', phonemes_path="../data/phonemes.txt", max_len=max_len,
                 padding_value=padding_value):
        with open(data_path, 'r') as f:
            lines = f.read().splitlines()
        data = [[int(x) for x in line.split()] for line in lines]
        self.x = pad_seq(data, max_len, units_padding_value)

        with open(phonemes_path, 'r') as f:
            lines = f.read().splitlines()
        data = [[phonemes_to_index[x.upper()] if x.upper() != "DX" else phonemes_to_index["T"] for x in line.split()]
                for line in
                lines]
        self.y = pad_seq(data, max_len, padding_value)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.LongTensor(self.x[idx]), torch.LongTensor(self.y[idx])


# Define the linear model
class LinearModel(nn.Module):

    def __init__(self, input_dim=unit_count + 1, output_dim=padding_value + 1, random_count=100):
        super(LinearModel, self).__init__()
        self.emb = nn.Embedding(input_dim, output_dim, )  # max_norm=1, norm_type=1
        print(self.emb.weight.data.shape)
        self.superv_map = get_superv_mapping(random_count=0)
        super_map_noise = torch.from_numpy(get_superv_mapping(random_count=random_count))
        # row_sums = super_map.sum(dim=1, keepdim=True)
        # normalized_tensor = super_map / row_sums
        self.emb.weight.data.copy_(super_map_noise)

        # self.linear = nn.Linear(emd_dim, output_dim)

    def check_map_acc(self):
        real_map = self.superv_map.argmax(axis=1).flatten()
        emb_map = self.emb.weight.data.argmax(axis=1).cpu().detach().numpy().flatten()
        return np.mean(real_map == emb_map)

    def forward(self, x):
        x = self.emb(x)
        # return self.linear(x)
        return x


pretrained_model = get_model()
pretrained_model.load_state_dict(torch.load(cp_file, map_location=torch.device('cpu')))
pretrained_model.to(device)
pretrained_model.eval()

for param in pretrained_model.parameters():
    param.requires_grad = False

loss_fn = nn.CrossEntropyLoss().to(device)
train_data = DataLoader(UnitsDataset(), batch_size=batch_size, shuffle=False, drop_last=True)

for iii, random_count in enumerate([100] * 10):
    linear_model = LinearModel(random_count=random_count)
    linear_model.to(device)
    optimizer = optim.Adam(linear_model.parameters(), lr=lr)
    loss_all = []
    acc_all = []
    acc_m_all = []
    mapp_all = []
    mapping = linear_model.emb.weight.data.argmax(axis=1).cpu().detach().numpy().flatten()
    print(mapping)
    for ephoc in tqdm(range(ephocs)):
        e_loss = []
        e_acc = []
        e_acc_m = []
        map_acc = []
        for j, (x, y) in enumerate(train_data):
            map_acc.append(linear_model.check_map_acc())
            x = x.to(device)
            y = y.to(device)
            # apply the models:
            linear_output = linear_model(x)
            argmax_output = torch.argmax(linear_output.detach(), dim=-1)
            argmax_output[y == padding_value] = padding_value

            pretrained_output = pretrained_model(argmax_output)
            model_predicted_labels = torch.argmax(pretrained_output, dim=-1)
            model_predicted_labels[y == padding_value] = padding_value

            loss = F.cross_entropy(
                linear_output.transpose(1, 2),
                model_predicted_labels,
                ignore_index=padding_value
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            model_predicted_labels = model_predicted_labels[y != padding_value]
            predicted_labels = argmax_output[y != padding_value]
            y = y[y != padding_value]
            conf_indx = predicted_labels != model_predicted_labels

            e_acc.append((predicted_labels == y).sum().item() / y.numel())
            e_acc_m.append((model_predicted_labels == y).sum().item() / y.numel())
            e_loss.append(loss.item())

        loss_all.append(np.mean(e_loss))
        acc_all.append(np.mean(e_acc))
        acc_m_all.append(np.mean(e_acc_m))
        mapp_all.append(np.mean(map_acc))
        # torch.save(linear_model.state_dict(), f"./models/linear_{ephoc}.cp")
    print(f"acc: {e_acc[-1]}")
    print(f"acc_m: {e_acc_m[-1]}")
    print(f"map_acc: {map_acc[-1]}")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    ax1.plot(loss_all, label="loss")
    ax1.set_title(random_count)
    ax1.legend()

    ax2.plot(acc_all, label="acc")
    ax2.plot(acc_m_all, label="acc_m")
    ax2.legend()

    ax3.plot(mapp_all, label="acc_map")
    ax3.legend()
    fig.tight_layout()
    fig.savefig(f"./tmp{iii}.png")
    plt.show()
