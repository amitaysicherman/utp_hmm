import torch
import torch.nn as nn
import torch.optim as optim
from train import get_model, input_size, max_len, padding_value
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from torch.nn import CTCLoss
from jiwer import wer

unit_count = 100
phonemes_count = input_size - 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cp_file = "./models/prep_random_small_timit_15.cp"
units_padding_value = unit_count
batch_size = 1  # 2048
blank_value = padding_value + 1
ephocs = 50
lr = 0.01

sil_units = [71, 93, 83, 22, 63, 4, 56, 20, 10, 81, 12, 40, 54, 39, 67, 84, 96, 55, 8, 34, 57, 72]


class Evaluator:
    def __init__(self, data_file="../data/TIMIT_UPDATE_PH.txt", code_file="../data/TIMIT_UPDATE_CODES.txt"):
        with open(code_file) as f:
            lines = f.read().splitlines()
        lines = [[int(y) for y in x.split()] for x in lines]
        self.code100 = []
        for line in lines:
            self.code100.append([line[0]] + [line[i] for i in range(1, len(line)) if
                                             line[i] != line[i - 1] and line[i] not in sil_units])

        with open(data_file) as f:
            phonemes = f.read().splitlines()
        self.phonemes = [[int(y) for y in x.split()] for x in phonemes]

    def eval_mapping(self, mapping_):
        mapping = [m if m != blank_value else "" for m in mapping_]
        wer_score = []
        for c, p in zip(self.code100, self.phonemes):
            p_c = [mapping[c_] for c_ in c]
            p = " ".join([str(xx) for xx in p])
            p_c = " ".join([str(xx) for xx in p_c])
            wer_score.append(wer(p, p_c))
        return np.mean(wer_score)


def get_superv_mapping():
    units_to_phonemes = np.zeros((unit_count + 1, blank_value + 1))
    for i in range(len(units_to_phonemes)):
        units_to_phonemes[i] = np.random.dirichlet(np.ones(blank_value + 1), size=1)
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
    def __init__(self, data_path='../data/TIMIT_UPDATE_CODES.txt', max_len=max_len):
        with open(data_path, 'r') as f:
            lines = f.read().splitlines()
        data = []
        for line in lines:
            line = [int(x) for x in line.split()]
            line = [line[0]] + [line[i] for i in range(1, len(line)) if line[i] != line[i - 1]]
            line = [x for x in line if x not in sil_units]
            data.append(line)
            print(len(line),line)
        self.x = pad_seq(data, max_len, units_padding_value)

    def __len__(self):
        return 1  # len(self.x)

    def __getitem__(self, idx):
        return torch.LongTensor(self.x[idx])


# Define the linear model
class LinearModel(nn.Module):

    def __init__(self, input_dim=unit_count + 1, output_dim=blank_value + 1):
        super(LinearModel, self).__init__()
        self.emb = nn.Embedding(input_dim, output_dim, max_norm=1)
        print(self.emb.weight.data.shape)
        self.superv_map = get_superv_mapping()
        super_map_noise = torch.from_numpy(get_superv_mapping())

        self.emb.weight.data.copy_(super_map_noise)

    def check_map_acc(self):
        real_map = self.superv_map.argmax(axis=1).flatten()
        emb_map = self.emb.weight.data.argmax(axis=1).cpu().detach().numpy().flatten()
        return np.mean(real_map == emb_map)

    def forward(self, x):
        x = self.emb(x)
        return x


pretrained_model = get_model()
pretrained_model.load_state_dict(torch.load(cp_file, map_location=torch.device('cpu')))
pretrained_model.to(device)
pretrained_model.eval()

for param in pretrained_model.parameters():
    param.requires_grad = False

ctc_loss = CTCLoss(blank=blank_value).to(device)

train_data = DataLoader(UnitsDataset(), batch_size=batch_size, shuffle=False, drop_last=True)
Evaluator = Evaluator()
for random_count in [100]:
    linear_model = LinearModel()
    linear_model.to(device)
    optimizer = optim.Adam(linear_model.parameters(), lr=lr)
    loss_all = []
    mapp_all = []
    for ephoc in tqdm(range(ephocs)):
        e_loss = []
        mapping = linear_model.emb.weight.data.argmax(axis=1).cpu().detach().numpy().flatten()
        print(mapping)
        map_acc = Evaluator.eval_mapping(mapping)
        for j, x in enumerate(train_data):
            print("map: ", linear_model.emb.weight.data.argmax(axis=1).cpu().detach().numpy().flatten())
            x = x.to(device)
            # apply the models:

            linear_output = linear_model(x)
            linear_output[:, :, blank_value] = 0
            argmax_output = torch.argmax(linear_output.detach(), dim=-1)
            argmax_output[x == units_padding_value] = padding_value
            argmax_output_ded = []
            argmax_output_ded_len = []
            for i in argmax_output:
                new_seq = [i[0]] + [i[j] for j in range(1, len(i)) if i[j] != i[j - 1]]
                new_seq = [x for x in new_seq if x != padding_value]
                argmax_output_ded_len.append(len(new_seq))
                argmax_output_ded.append(new_seq + [padding_value] * (max_len - len(new_seq)))
            argmax_output_ded = torch.LongTensor(argmax_output_ded).to(device)
            argmax_output_ded_len = torch.LongTensor(argmax_output_ded_len).to(device)
            input_seq_lengths = []
            for i in range(x.size(0)):
                seq_length = 0
                for j in range(len(x[i])):
                    if x[i][j] != units_padding_value:
                        seq_length += 1
                    else:
                        break
                input_seq_lengths.append(seq_length)

            input_seq_lengths = torch.tensor(input_seq_lengths, dtype=torch.long)

            # print(argmax_output_ded)

            pretrained_output = pretrained_model(argmax_output_ded)
            model_predicted_labels = torch.argmax(pretrained_output, dim=-1)
            model_predicted_labels[argmax_output_ded == padding_value] = padding_value
            model_predicted_labels_ded = []
            model_predicted_labels_ded_len = []
            for i in model_predicted_labels:
                new_seq = [i[0]] + [i[j] for j in range(1, len(i)) if i[j] != i[j - 1]]
                model_predicted_labels_ded_len.append(len(new_seq) - (1 if new_seq[-1] == padding_value else 0))
                model_predicted_labels_ded.append(new_seq + [padding_value] * (max_len - len(new_seq)))
            model_predicted_labels_ded = torch.LongTensor(model_predicted_labels_ded).to(device)
            model_predicted_labels_ded_len = torch.LongTensor(model_predicted_labels_ded_len).to(device)
            print(input_seq_lengths, model_predicted_labels_ded_len)
            loss = ctc_loss(linear_output.transpose(0, 1).log_softmax(2), model_predicted_labels_ded, input_seq_lengths,
                            model_predicted_labels_ded_len)

            # print(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            e_loss.append(loss.item())
            print(loss.item())

        loss_all.append(np.mean(e_loss))
        mapp_all.append(map_acc)
        print(f"ephoc: {ephoc}, loss: {loss_all[-1]}, map_acc: {mapp_all[-1]}")
        # torch.save(linear_model.state_dict(), f"./models/linear_{ephoc}.cp")

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.plot(loss_all, label="loss")
    ax1.set_title(random_count)
    ax1.legend()

    ax3.plot(mapp_all, label="acc_map")
    ax3.legend()
    fig.tight_layout()
    plt.show()
