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
batch_size = 1#2048
ephocs = 50
lr = 0.01

random_count = 85
def eval_mapping(mapping):
    with open("../data/TIMIT_UPDATE_CODES.txt") as f:
        lines = f.read().splitlines()
    lines = [[int(y) for y in x.split()] for x in lines]
    code100 = []
    for line in lines:
        code100.append([line[0]]+[line[i] for i in range(1, len(line)) if line[i] != line[i-1]])
    with open("../data/TIMIT_UPDATE_PH.txt") as f:
        phonemes = f.read().splitlines()
    phonemes = [[int(y) for y in x.split()] for x in phonemes]
    wer_score=[]
    for c,p in zip(code100, phonemes):
        p_c=[mapping[c_] for c_ in c]
        p=" ".join([str(xx) for xx in p])
        p_c=" ".join([str(xx) for xx in p_c])
        wer_score.append(wer(p, p_c))
    return np.mean(wer_score)

def get_superv_mapping(random_count=0):
    units_to_phonemes = np.zeros((unit_count + 1, padding_value + 1))

    if random_count == 100:
        for i in range(len(units_to_phonemes)):
            units_to_phonemes[i] = np.random.dirichlet(np.ones(padding_value + 1), size=1)
            units_to_phonemes[i] /= units_to_phonemes[i].sum()
        units_to_phonemes[:, padding_value] = 0
        return units_to_phonemes

    with open("../data/TIMIT_UPDATE_CODES.txt") as f:
        lines = f.read().splitlines()
    lines = [[int(y) for y in x.split()] for x in lines]
    code100 = []
    for line in lines:
        code100.append([line[0]]+[line[i] for i in range(1, len(line)) if line[i] != line[i-1]])


    with open("../data/TIMIT_UPDATE_PH.txt") as f:
        phonemes = f.read().splitlines()
    phonemes = [[int(y) for y in x.split()] for x in phonemes]


    units_to_phonemes = np.zeros((unit_count + 1, padding_value + 1))+1e-4
    for i, (u, p) in enumerate(tqdm(zip(sum(code100, []), sum(phonemes, [])))):
        units_to_phonemes[u, p] += 1
    for i in range(unit_count + 1):
        units_to_phonemes[i] /= units_to_phonemes[i].sum()
    units_to_phonemes[unit_count, padding_value] = 1
    random_units_choise = random.sample(range(unit_count), k=random_count)
    for i in random_units_choise:
        units_to_phonemes[i] = np.random.dirichlet(np.ones(padding_value + 1), size=1)
        units_to_phonemes[i] /= units_to_phonemes[i].sum()
    units_to_phonemes[:, padding_value] = 0
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
    def __init__(self, data_path='../data/TIMIT_UPDATE_CODES.txt', max_len=max_len):
        with open(data_path, 'r') as f:
            lines = f.read().splitlines()
        data = []
        for line in lines:
            line = [int(x) for x in line.split()]
            line = [line[0]] + [line[i] for i in range(1, len(line)) if line[i] != line[i - 1]]
            data.append(line)
        self.x = pad_seq(data, max_len, units_padding_value)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.LongTensor(self.x[idx])


# Define the linear model
class LinearModel(nn.Module):

    def __init__(self, input_dim=unit_count + 1, output_dim=padding_value + 1, random_count=0):
        super(LinearModel, self).__init__()
        self.emb = nn.Embedding(input_dim, output_dim, )  # max_norm=1, norm_type=1
        print(self.emb.weight.data.shape)
        self.superv_map = get_superv_mapping(random_count=0)
        super_map_noise = torch.from_numpy(get_superv_mapping(random_count=random_count))

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

ctc_loss = CTCLoss(blank=padding_value).to(device)

train_data = DataLoader(UnitsDataset(), batch_size=batch_size, shuffle=False, drop_last=True)

for random_count in [100]:
    linear_model = LinearModel(random_count=random_count)
    linear_model.to(device)
    optimizer = optim.Adam(linear_model.parameters(), lr=lr)
    loss_all = []
    mapp_all = []
    for ephoc in tqdm(range(ephocs)):
        e_loss = []
        mapping = linear_model.emb.weight.data.argmax(axis=1).cpu().detach().numpy().flatten()
        print(mapping)
        map_acc = eval_mapping(mapping)
        for j, x in enumerate(train_data):

            x = x.to(device)
            # apply the models:
            linear_output = linear_model(x)
            argmax_output = torch.argmax(linear_output.detach(), dim=-1)
            argmax_output[x == units_padding_value] = padding_value

            argmax_output_ded = []
            argmax_output_ded_len = []
            for i in argmax_output:
                new_seq = [i[0]] + [i[j] for j in range(1, len(i)) if i[j] != i[j - 1]]
                argmax_output_ded_len.append(len(new_seq)-(1 if new_seq[-1]==padding_value else 0))
                argmax_output_ded.append(new_seq + [padding_value] * (max_len - len(new_seq)))
            argmax_output_ded = torch.LongTensor(argmax_output_ded).to(device)
            argmax_output_ded_len = torch.LongTensor(argmax_output_ded_len).to(device)
            input_seq_lengths = []
            for i in range(x.size(0)):  # Iterate over each sequence in the batch
                seq_length=0
                for j in range(len(x[i])):
                    if x[i][j] != units_padding_value:
                        seq_length+=1
                    else:
                        break
                input_seq_lengths.append(seq_length)
            input_seq_lengths = torch.tensor(input_seq_lengths, dtype=torch.long)

            pretrained_output = pretrained_model(argmax_output_ded)
            model_predicted_labels = torch.argmax(pretrained_output, dim=-1)
            model_predicted_labels[argmax_output_ded == padding_value] = padding_value
            model_predicted_labels_ded=[]
            model_predicted_labels_ded_len=[]
            for i in model_predicted_labels:
                new_seq = [i[0]] + [i[j] for j in range(1, len(i)) if i[j] != i[j - 1]]
                model_predicted_labels_ded_len.append(len(new_seq)-(1 if new_seq[-1]==padding_value else 0))
                model_predicted_labels_ded.append(new_seq + [padding_value] * (max_len - len(new_seq)))
            model_predicted_labels_ded = torch.LongTensor(model_predicted_labels_ded).to(device)
            model_predicted_labels_ded_len = torch.LongTensor(model_predicted_labels_ded_len).to(device)


            loss = ctc_loss(linear_output.transpose(0, 1).log_softmax(2), model_predicted_labels_ded, input_seq_lengths,
                            model_predicted_labels_ded_len)
            # print(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            optimizer.zero_grad()
            e_loss.append(loss.item())

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
