# sbatch --gres=gpu:1,vmem:24g --mem=75G --time=0:10:0 --wrap "python train_match_dense.py --lr=0.01 --epochs=50 --model=transformer --match_data=./pseg/data/p_superv --match_cp=./models/transformer_small_lr_100_0.0005_0.0_90_0.8.cp"
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_model, args_parser, PADDING_VALUE, INPUT_SIZE, MASK_VALUE, phonemes_to_index, N_TOKENS
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from jiwer import wer
from mlm import MLM

N_CLUSTERS = 100
CLUSTERS_PADDING_VALUE = N_CLUSTERS
args = args_parser()
padding_value = PADDING_VALUE
input_dim = 768
input_size = INPUT_SIZE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_len = args.max_len
batch_size = args.batch_size
ephocs = args.epochs
lr = args.lr
features_path = f"{args.match_data}/features.npy"
len_path = f"{args.match_data}/features.length"
phonemes_path = f"{args.match_data}/features.phonemes"
clusters_path = f"{args.match_data}/features.clusters"


def eval_mapping(x, y):
    wer_score = []
    for c, p in zip(x, y):
        c = list(c.detach().cpu().numpy())
        p = list(p.detach().cpu().numpy())
        c = [c_ for c_ in c if c_ != padding_value]
        p = [p_ for p_ in p if p_ != padding_value]
        if len(c) == len(p):
            wer_score.append((np.array(c) != np.array(p)).mean())
            continue
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
                 phonemes_path=phonemes_path, clusters_path=clusters_path, max_len=max_len,
                 padding_value=padding_value, clusters_padding_value=CLUSTERS_PADDING_VALUE):
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
            phonemes_lines = f.read().splitlines()

        with open(clusters_path, 'r') as f:
            clusters_lines = f.read().splitlines()
        clusters = [[int(x) for x in line.split()] for line in clusters_lines]

        data = []
        for line in phonemes_lines:
            line = line.split()
            data.append([phonemes_to_index[x.upper()] if x.upper() != "DX" else phonemes_to_index["T"] for x in line])
        self.y = pad_seq(data, max_len, padding_value)
        self.clusters = pad_seq(clusters, max_len, clusters_padding_value)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.x[idx]), torch.LongTensor(self.y[idx]), torch.LongTensor(self.clusters[idx])


# Define the linear model
class LinearModel(nn.Module):

    def __init__(self, mlm_trainer, input_dim=input_dim, output_dim=padding_value + 2, n_clusters=N_CLUSTERS,
                 alpha=0.1):
        super(LinearModel, self).__init__()
        self.mlm_trainer = mlm_trainer
        self.lin = nn.Linear(input_dim, output_dim)
        self.lin_cyclic = nn.Linear(output_dim, n_clusters)
        self.alpha = alpha

    def forward(self, x):
        x = self.lin(x)
        x_cyc = self.lin_cyclic(x)
        return x, x_cyc

    def get_loss(self, x, y, clusters):
        linear_output, cyc_output = self.forward(x)
        argmax_output = torch.argmax(linear_output.detach(), dim=-1)
        argmax_cyc = torch.argmax(cyc_output.detach(), dim=-1)
        x_padding_mask = x.sum(axis=-1) == 0
        argmax_output[x_padding_mask] = padding_value

        mask_in = ~x_padding_mask

        # print(argmax_output.shape)

        mlm_loss, logits, labels = self.mlm_trainer(argmax_output)
        model_predicted_labels = torch.argmax(logits, dim=-1)
        mask_in &= (labels != padding_value)

        model_predicted_labels[~mask_in] = padding_value
        # clusters[~mask_in] = padding_value

        # print("----------")
        # print(argmax_output[0])
        # print(model_predicted_labels[0])
        # print(y[0])

        loss_forward = F.cross_entropy(
            linear_output.transpose(1, 2),
            model_predicted_labels,
            ignore_index=padding_value
        )
        clusters_mask = clusters.clone()
        clusters_mask[~mask_in] = CLUSTERS_PADDING_VALUE
        loss_cyclic = F.cross_entropy(
            cyc_output.transpose(1, 2),
            clusters_mask,
            ignore_index=CLUSTERS_PADDING_VALUE
        )
        loss = loss_forward + self.alpha * loss_cyclic
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        acc_lin = eval_mapping(argmax_output, y)
        # acc_model = eval_mapping(model_predicted_labels, y)

        argmax_cyc[x_padding_mask] = padding_value
        # print(argmax_cyc[0])
        # print(clusters[0])
        acc_cycl = 100 * (argmax_cyc == clusters).sum().item() / argmax_cyc.numel()

        return {"loss": loss.item(),
                "loss_f": loss_forward.item(),
                "loss_c": loss_cyclic.item(),
                "acc_lin": acc_lin,
                "acc_cyc": acc_cycl
                }


pretrained_model = get_model(args.model, args.size, args.max_len, args.drop_out, MASK_VALUE + 1)
pretrained_model.load_state_dict(torch.load(args.match_cp, map_location=torch.device('cpu')))
pretrained_model.to(device)
pretrained_model.eval()

for param in pretrained_model.parameters():
    param.requires_grad = False

mlm_trainer = MLM(pretrained_model, mask_token_id=MASK_VALUE, pad_token_id=PADDING_VALUE, mask_prob=0.15,
                  random_token_prob=0, num_tokens=N_TOKENS, replace_prob=0.90).to(device)

loss_fn = nn.CrossEntropyLoss().to(device)
train_data = DataLoader(UnitsDataset(), batch_size=batch_size, shuffle=False, drop_last=True)

linear_model = LinearModel(mlm_trainer, alpha=0.001)
linear_model.to(device)

optimizer = optim.Adam(linear_model.parameters(), lr=lr)

# loss_all = []
# acc_all = []
# acc_m_all = []

for ephoc in tqdm(range(ephocs)):
    res = []
    for j, (x, y, c) in tqdm(enumerate(train_data), total=len(train_data)):
        x = x.to(device)
        y = y.to(device)
        c = c.to(device)
        res.append(linear_model.get_loss(x, y, c))
    print("Ephoc: ", ephoc)
    print(pd.DataFrame(res).mean())
