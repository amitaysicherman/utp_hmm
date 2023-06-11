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
ephocs = 300


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


class UnitsDataset(Dataset):
    def __init__(self, data_path='../data/code100.txt', phonemes_path="../data/phonemes.txt", max_len=max_len,
                 padding_value=padding_value):
        with open(data_path, 'r') as f:
            lines = f.read().splitlines()
        data = [[int(x) for x in line.split()] for line in lines]
        self.x = pad_seq(data, max_len, padding_value)

        with open(phonemes_path, 'r') as f:
            lines = f.read().splitlines()
        data = [[phonemes_to_index[x.upper()] if x.upper() != "DX" else phonemes_to_index["D"] for x in line.split()]
                for line in
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
        self.emb = nn.Embedding(input_dim, output_dim)
        # self.linear = nn.Linear(emd_dim, output_dim)

    def forward(self, x):
        x = self.emb(x)
        return x
        # return self.linear(x)



pretrained_model = get_model()
pretrained_model.load_state_dict(torch.load(cp_file, map_location=torch.device('cpu')))
pretrained_model.to(device)
pretrained_model.eval()

linear_model = LinearModel()
linear_model.to(device)

for param in pretrained_model.parameters():
    param.requires_grad = False

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(linear_model.parameters(), lr=0.01)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.1)

train_data = DataLoader(UnitsDataset(), batch_size=batch_size, shuffle=False, drop_last=True)
for ephoc in range(ephocs):
    e_loss=[]
    e_acc=[]
    for j,(x, y) in enumerate(train_data):
        x=x.to(device)
        mask=torch.zeros_like(x)
        mask[x!=padding_value]=1
        linear_output = linear_model(x)
        argmax_output = torch.argmax(linear_output.detach(), dim=-1)
        pretrained_output = pretrained_model(argmax_output)
        pretrained_output=pretrained_output.softmax(dim=-1)
        linear_output = linear_output.view(-1, linear_output.shape[-1])
        pretrained_output = pretrained_output.view(-1,pretrained_output.shape[-1])
        masked_inputs = linear_output[mask.view(-1)]
        masked_targets = pretrained_output[mask.view(-1)]

        loss = loss_fn(masked_inputs, masked_targets)
        e_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = 0
        count = 0
        for i,(single_x, single_y) in enumerate(zip(argmax_output, y)):

            single_x = single_x.numpy()
            single_y = single_y.numpy()
            single_x = single_x[single_y != padding_value]
            single_y = single_y[single_y != padding_value]
            if i==0 and j==0:
                print(single_x)
            if len(single_x) != len(single_y):
                continue
            acc += (single_x == single_y).mean()
            count += 1
        if count:
            e_acc.append(acc / count)
    # scheduler.step()
    print(f"ephoc {ephoc} loss {np.mean(e_loss)} acc {np.mean(e_acc)}")
    # torch.save(linear_model.state_dict(), f"./models/linear_{ephoc}.cp")