import torch
import torch.nn as nn
import torch.optim as optim
from train import get_model, input_size, max_len, padding_value
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from text_to_phonemes_index import phonemes_to_index
import torch.nn.functional as F

unit_count = 100
phonemes_count = input_size + 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cp_file = "./models/prep_random_small_timit_15.cp"

batch_size = 512
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

    def reset_parameters(self):
        output_dim = self.emb.embedding_dim
        input_dim = self.emb.num_embeddings
        for i in range(input_dim):
            values = torch.randn(output_dim)
            random_index = torch.randint(output_dim, size=(1,))
            values[random_index] = 0.5
            values /= values.sum()  # Normalize values to sum to 0.5
            self.emb.weight.data[i].copy_(values)

    def __init__(self, input_dim=unit_count, emd_dim=256, output_dim=phonemes_count):
        super(LinearModel, self).__init__()
        self.emb = nn.Embedding(input_dim, output_dim, max_norm=1, norm_type=1)

        # Initialize each embedding vector
        for i in range(input_dim):
            values = torch.randn(output_dim)
            random_index = torch.randint(output_dim, size=(1,))
            values[random_index] = 0.5
            values /= values.sum()  # Normalize values to sum to 0.5
            self.emb.weight.data[i].copy_(values)

        # self.linear = nn.Linear(emd_dim, output_dim)

    def forward(self, x):
        x = self.emb(x)
        # return self.linear(x)
        return x


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

best_loss = float('inf')
no_improvement_epochs = 0

for ephoc in range(ephocs):
    e_loss = []
    e_acc = []
    e_acc_m = []
    for j, (x, y) in enumerate(train_data):
        x = x.to(device)
        y = y.to(device)
        # apply the models:
        linear_output = linear_model(x)
        argmax_output = torch.argmax(linear_output.detach(), dim=-1)
        # argmax_output = torch.multinomial(linear_output.softmax(dim=-1).view(-1, phonemes_count), 1).view(
        #     linear_output.size()[:-1])

        pretrained_output = pretrained_model(argmax_output)
        model_predicted_labels = torch.argmax(pretrained_output, dim=-1)
        loss = F.cross_entropy(
            linear_output.transpose(1, 2),
            model_predicted_labels,
            ignore_index=padding_value
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        optimizer.zero_grad()

        # calculate the accuracy:
        model_predicted_labels = model_predicted_labels[y != padding_value]
        predicted_labels = argmax_output[y != padding_value]
        y = y[y != padding_value]
        e_acc.append((predicted_labels == y).sum().item() / y.numel())
        e_acc_m.append((model_predicted_labels == y).sum().item() / y.numel())

        # linear_output = linear_output.view(-1, linear_output.shape[-1])
        # pretrained_output = pretrained_output.view(-1, pretrained_output.shape[-1])
        # mask = torch.zeros_like(x)
        # mask[x != padding_value] = 1
        # masked_inputs = linear_output[mask.view(-1)]
        # masked_targets = pretrained_output[mask.view(-1)]
        # loss = loss_fn(masked_inputs, masked_targets)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        e_loss.append(loss.item())
        if loss < best_loss:
            best_loss = loss
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        # Reset embedding values if no improvement for specified epochs
        if no_improvement_epochs >= 5:
            linear_model.reset_parameters()
            no_improvement_epochs = 0
    # scheduler.step()
    print(f"ephoc {ephoc} loss {np.mean(e_loss)} acc {np.mean(e_acc)} acc_m {np.mean(e_acc_m)}")
    torch.save(linear_model.state_dict(), f"./models/linear_{ephoc}.cp")
