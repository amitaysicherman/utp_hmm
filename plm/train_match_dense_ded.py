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
from jiwer import wer

input_dim = 768
phonemes_count = input_size - 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cp_file = "./models/timit_small_drop_out_99.cp"  # ./models/timit_small_20.cp"

batch_size = 512
ephocs = 15
lr = 0.1

prefix = "/cs/labs/adiyoss/amitay.sich/utp_hmm/plm/pseg/data/p_superv_test/"  # plm/pseg/
features_path = f"{prefix}features.npy"
len_path = f"{prefix}features.length"
phonemes_path = f"{prefix}features.phonemes"
print(padding_value, mis_index)


def eval_mapping(x, y):
    wer_score = []
    for c, p in zip(x, y):
        c = " ".join([str(c_) for c_ in c.detach().cpu().numpy() if c_ != padding_value and c_ != mis_index])
        p = " ".join([str(p_) for p_ in p.detach().cpu().numpy() if p_ != padding_value and p_ != mis_index])
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
        for l in length:
            self.x.append(pad_array(self.features[cur:cur + l]))
            cur += l
        with open(phonemes_path, 'r') as f:
            lines = f.read().splitlines()
        data = []
        for l in lines:
            l = l.split()
            l = [x for x in l if x != "sil"]
            l = [phonemes_to_index[x.upper()] if x.upper() != "DX" else phonemes_to_index["T"] for x in l]
            data.append(l)

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
        # w_b = np.load(f"./models/linear_14.npz")
        # self.lin.weight.data.copy_(torch.from_numpy(w_b["w"]))
        # self.lin.bias.data.copy_(torch.from_numpy(w_b["b"]))

    def forward(self, x):
        x = self.lin(x)
        return x


pretrained_model = get_model()
pretrained_model.load_state_dict(torch.load(cp_file, map_location=torch.device('cpu')))
pretrained_model.to(device)
pretrained_model.eval()

for param in pretrained_model.parameters():
    param.requires_grad = False

loss_fn = nn.CrossEntropyLoss().to(device)
dataset = UnitsDataset()
train_data = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

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

        x_padding_mask = x.sum(axis=-1) == 0
        y = y.to(device)

        # apply the models:
        linear_output = linear_model(x)
        argmax_output = torch.argmax(linear_output.detach(), dim=-1)

        argmax_output[x_padding_mask] = padding_value


        def ded_and_pad(x):
            x = torch.unique_consecutive(x)
            return F.pad(x, (0, max_len - len(x)), value=padding_value)


        # def ded_mask(x):
        #     x=torch.unique_consecutive(x, return_inverse=True)[1]
        #     diff=x[1:]-x[:-1]
        #     diff=diff==1
        #     return torch.cat([torch.BoolTensor([True]).to(diff.device), diff])

        # argmax_output_ded = [ded_and_pad(x) for x in argmax_output]
        # argmax_output_ded = torch.stack(argmax_output_ded, dim=0)
        # argmax_output_ded_inverse = torch.stack(
        #     [torch.unique_consecutive(x, return_inverse=True)[1] for x in argmax_output], dim=0)
        # argmax_output_ded_mask=[ded_mask(x) for x in argmax_output]
        # argmax_output_ded_mask = torch.stack(argmax_output_ded_mask, dim=0)

        pretrained_output = pretrained_model(argmax_output)
        model_predicted_labels = torch.argmax(pretrained_output, dim=-1)
        model_predicted_labels[x_padding_mask] = padding_value
        # model_predicted_labels[model_predicted_labels==mis_index]=padding_value

        # model_predicted_labels_inv = torch.zeros_like(model_predicted_labels)
        # for i in range(len(model_predicted_labels)):
        #     model_predicted_labels_inv[i, :] = model_predicted_labels[i, argmax_output_ded_inverse[i]]
        # model_predicted_labels_inv[x_padding_mask] = padding_value


        # linear_output_ded = linear_output[argmax_output_ded_mask]
        # model_predicted_labels_ded = model_predicted_labels[argmax_output_ded_mask]

        # def replace_with_reciprocal_frequency(input_tensor):
        #     unique_values, counts = torch.unique(input_tensor, return_counts=True)
        #     reciprocal_frequency = 1.0 / counts.float()
        #     value_to_frequency = dict(zip(unique_values.tolist(), reciprocal_frequency.tolist()))
        #     output_tensor = torch.zeros_like(input_tensor, dtype=torch.float)
        #     for i in range(len(input_tensor)):
        #         output_tensor[i] = value_to_frequency[input_tensor[i].item()]
        #     return output_tensor
        #
        #
        # loss_w = torch.stack([replace_with_reciprocal_frequency(argmax_output_ded_inverse[i]) for i in
        #                       range(len(argmax_output_ded_inverse))], dim=0)
        # loss_w.to(device)
        # loss_w[x_padding_mask] = 0.0

        loss = F.cross_entropy(
            linear_output.transpose(1, 2),
            model_predicted_labels,
            ignore_index=padding_value,
            # reduction='none'
        )
        # loss = (loss_w * loss).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        e_acc.append(eval_mapping(argmax_output, y))
        e_acc_m.append(eval_mapping(model_predicted_labels, y))
        e_loss.append(loss.item())
        print(f"loss: {loss.item()}, acc: {e_acc[-1]}, acc_m: {e_acc_m[-1]}")

    loss_all.append(np.mean(e_loss))
    acc_all.append(np.mean(e_acc))
    acc_m_all.append(np.mean(e_acc_m))

# w=linear_model.lin.weight.detach().cpu().numpy()
# b=linear_model.lin.bias.detach().cpu().numpy()
#
# np.savez(f"./models/linear_{ephoc}.npz", w=w, b=b)
# torch.save(linear_model.state_dict(), f"./models/linear_{ephoc}.cp")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
ax1.plot(loss_all, label="loss")
ax1.legend()

ax2.plot(acc_all, label="acc")
ax2.plot(acc_m_all, label="acc_m")
ax2.legend()
fig.tight_layout()
fig.savefig(f"./tmp.png")
plt.show()
