#sbatch --gres=gpu:1,vmem:24g --mem=75G --time=2:0:0 --wrap "python train_match_dense.py --model_name=timit_dupsmall_99.cp"
import torch
import torch.nn as nn
import torch.optim as optim
from train import get_model, input_size, max_len, padding_value
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from mapping import phonemes_to_index, mis_index
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse

input_dim = 768
phonemes_count = input_size - 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str,
                    default="prep_random_small_timit_99")
args = parser.parse_args()

cp_file = f"./models/{args.model_name}"  # timit_dupsmall_13.cp"
if args.model_name == "prep_random_small_timit_99":
    max_len = 50
else:
    max_len = 100

batch_size = 512
ephocs = 50
lr = 0.1
prefix = "/cs/labs/adiyoss/amitay.sich/utp_hmm/plm/pseg/data/p_superv/"
features_path = f"{prefix}features.npy"
len_path = f"{prefix}features.length"
phonemes_path = f"{prefix}features.phonemes"


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
        for l in length:
            self.x.append(pad_array(self.features[cur:cur + l]))
            cur += l
        with open(phonemes_path, 'r') as f:
            lines = f.read().splitlines()
        data = [[phonemes_to_index[x.upper()] if x.upper() != "DX" else phonemes_to_index["T"] for x in line.split()]
                for line in
                lines]
        self.y = pad_seq(data, max_len, padding_value)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.x[idx]), torch.LongTensor(self.y[idx])


# Define the linear model
class LinearModel(nn.Module):

    def __init__(self, input_dim=input_dim, output_dim=padding_value + 1):
        super(LinearModel, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.lin(x)
        return x


pretrained_model = get_model(max_len=max_len)
pretrained_model.load_state_dict(torch.load(cp_file, map_location=torch.device('cpu')))
pretrained_model.to(device)
pretrained_model.eval()

for param in pretrained_model.parameters():
    param.requires_grad = False

loss_fn = nn.CrossEntropyLoss().to(device)
train_data = DataLoader(UnitsDataset(), batch_size=batch_size, shuffle=False, drop_last=True)

linear_model = LinearModel()
linear_model.to(device)
optimizer = optim.Adam(linear_model.parameters(), lr=lr)
loss_all = []
acc_all = []
acc_m_all = []
mapp_all = []
for ephoc in tqdm(range(ephocs)):
    e_loss = []
    e_acc = []
    e_acc_m = []
    for j, (x, y) in tqdm(enumerate(train_data)):
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
        if j==0:
            print(argmax_output[0])
            print(model_predicted_labels[0])
    print(f"loss: {loss.item()}, acc: {e_acc[-1]}, acc_m: {e_acc_m[-1]}")
    loss_all.append(np.mean(e_loss))
    acc_all.append(np.mean(e_acc))
    acc_m_all.append(np.mean(e_acc_m))
    # torch.save(linear_model.state_dict(), f"./models/linear_{ephoc}.cp")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
ax1.plot(loss_all, label="loss")
ax1.legend()

ax2.plot(acc_all, label="acc")
ax2.plot(acc_m_all, label="acc_m")
ax2.legend()

ax3.plot(mapp_all, label="acc_map")
ax3.legend()
fig.tight_layout()
fig.savefig(f"./tmp.png")
plt.show()
