import torch
import torch.nn as nn
import torch.optim as optim
from train import get_model, input_size, max_len, padding_value
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from text_to_phonemes_index import phonemes_to_index

unit_count = 100
phonemes_count = input_size + 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cp_file = "./models/best.cp"

batch_size = 4
ephocs = 10


def pad_seq(data, max_len, padding_value):
    x = []
    for d in tqdm(data):
        sequence = d
        if len(sequence) > max_len:
            sequence = np.array(list(sequence[:max_len]), dtype=np.int8)
        else:
            sequence = np.array(list(sequence) + [padding_value] * (max_len - len(sequence)), dtype=np.int8)
        x.append(sequence)
    return x


class PhonemesDataset(Dataset):
    def __init__(self, data_path='../data/code100.txt', phonemes_path="./data/phonemes.txt", max_len=max_len,
                 padding_value=padding_value):
        with open(data_path, 'r') as f:
            lines = f.read().splitlines()
        data = [[int(x) for x in line.split()] for line in lines]
        self.x = pad_seq(data, max_len, padding_value)

        with open(phonemes_path, 'r') as f:
            lines = f.read().splitlines()
        data = [[phonemes_to_index(x.upper()) for x in line.split() if x.upper() in phonemes_to_index] for line in
                lines]
        self.y = pad_seq(data, max_len, padding_value)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.LongTensor(self.x[idx]), torch.LongTensor(self.y[idx])


# Define the linear model
class LinearModel(nn.Module):
    def __init__(self, input_dim=unit_count, emd_dim=256, output_dim=phonemes_count):
        super(LinearModel, self).__init__()
        self.emb = nn.Embedding(input_dim, emd_dim)
        self.linear = nn.Linear(emd_dim, output_dim)

    def forward(self, x):
        x = self.emb(x)
        return nn.functional.softmax(self.linear(x), dim=-1)


pretrained_model = get_model()
pretrained_model.load_state_dict(torch.load(cp_file, map_location=torch.device('cpu')))
pretrained_model.to(device)
pretrained_model.eval()

linear_model = LinearModel()
linear_model.to(device)

for param in pretrained_model.parameters():
    param.requires_grad = False

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(linear_model.parameters(), lr=0.001)

train_data = DataLoader(PhonemesDataset(), batch_size=batch_size, shuffle=False)
for ephoc in range(ephocs):
    for x, y in train_data:
        linear_output = linear_model(x)
        argmax_output = torch.argmax(linear_output, dim=-1)
        pretrained_output = pretrained_model(argmax_output)
        loss = loss_fn(linear_output, pretrained_output)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for single_x, single_x in zip(x, y):
            print(single_x.numpy(), single_x.numpy())
            single_x = single_x[single_x != padding_value]
            single_y = single_y[single_y != padding_value]
            print(single_x.shape, single_x.shape)

    torch.save(linear_model.state_dict(), f"./models/linear_{ephoc}.cp")
