import random
from collections import defaultdict
from train import PhonemesDataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from train import input_size, max_len, padding_value
from plm.utils import get_model
import numpy as np
from tqdm import tqdm

batch_size = 1024
log_dir = f"./logs/{time.time()}"  # Set the directory for saving TensorBoard logs
n_phonemes = input_size - 1
n_units = 100
p_noise = 0.15
writer = SummaryWriter(log_dir=log_dir)


class ReplacePhonemesDataset(PhonemesDataset):
    def __init__(self, data_path='data/LR960_PH.npz', data_len_path="data/LR960_PH_LEN.txt", max_len=max_len,
                 padding_value=padding_value, n_units=n_units):
        super().__init__(data_path, data_len_path, max_len, padding_value)

        self.n_units = n_units
        assert n_units >= n_phonemes

        mapping = list(range(n_phonemes))

        mapping += [random.randint(0, n_phonemes - 1) for _ in range(n_units - n_phonemes)]
        random.shuffle(mapping)

        print(mapping)
        self.mapping = mapping
        self.inv_mapping = defaultdict(list)
        for i, v in enumerate(self.mapping):
            if i == padding_value:
                self.inv_mapping[v].append(32)
            else:
                self.inv_mapping[v].append(i)
        print(self.inv_mapping)
        self.y = []

        for x_ in tqdm(self.x):
            y_ = []
            for v in x_:
                if v == padding_value:
                    y_.append(padding_value)
                else:
                    if random.random() >= p_noise:
                        y_.append(random.choice(self.inv_mapping[v]))
                    else:
                        element = random.randint(0, n_units-1)
                        while element == padding_value:
                            element = random.randint(0, n_units-1)
                        y_.append(element)

            self.y.append(y_)

    def __getitem__(self, idx):
        return torch.LongTensor(self.x[idx]), torch.LongTensor(self.y[idx])


class LinearModel(nn.Module):
    def __init__(self, input_dim=n_units, output_dim=input_size + 1):
        super(LinearModel, self).__init__()
        self.emb = nn.Embedding(input_dim, output_dim, max_norm=1, norm_type=1)
        # identity_matrix = torch.eye(input_size + 1)
        # self.emb.weight.data.copy_(identity_matrix)

    def forward(self, x):
        return self.emb(x)


if __name__ == '__main__':
    cp_file = "./models/best.cp"
    data_path = 'data/LR960_PH.npz'
    data_len_path = 'data/LR960_PH_LEN.txt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    model.load_state_dict(torch.load(cp_file, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    linear_model = LinearModel()
    linear_model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(linear_model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.1)
    dataset = ReplacePhonemesDataset(data_path=data_path, data_len_path=data_len_path)

    mapping = linear_model.emb.weight.data.cpu().argmax(dim=-1).numpy()
    print(mapping)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(100):
        e_loss = []
        e_acc = []
        e_acc_z = []
        e_updates = 0

        for batch_idx, (y, x) in enumerate(data_loader):
            x = x.to(device)
            mask = torch.zeros_like(x)
            mask[x != padding_value] = 1

            linear_output = linear_model(x)
            argmax_output = torch.argmax(linear_output.detach(), dim=-1)
            with torch.no_grad():
                pretrained_output = model(argmax_output)
            pretrained_output = pretrained_output.softmax(dim=-1)
            linear_output = linear_output.view(-1, linear_output.shape[-1])
            pretrained_output = pretrained_output.view(-1, pretrained_output.shape[-1])
            masked_inputs = linear_output[mask.view(-1)]
            masked_targets = pretrained_output[mask.view(-1)]

            loss = loss_fn(masked_inputs, masked_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            single_x = argmax_output.cpu().numpy().flatten()
            single_y = y.numpy().flatten()
            single_z = pretrained_output.argmax(dim=-1).cpu().numpy().flatten()

            single_x = single_x[single_y != padding_value]
            single_z = single_z[single_y != padding_value]

            single_y = single_y[single_y != padding_value]

            if len(single_x) != len(single_y):
                print("skip", len(single_y), len(single_x), flush=True)
                continue

            e_loss.append(loss.item())
            e_acc.append((single_x == single_y).sum() / len(single_y))
            e_acc_z.append((single_z == single_y).sum() / len(single_y))
            new_mapping = linear_model.emb.weight.data.cpu().argmax(dim=-1).numpy()
            e_updates += (new_mapping != mapping).sum()
            mapping = new_mapping

        writer.add_scalar("Loss", np.mean(e_loss), epoch)
        writer.add_scalar("Accuracy", np.mean(e_acc), epoch)
        writer.add_scalar("Accuracy_z", np.mean(e_acc_z), epoch)
        writer.add_scalar("Updates", e_updates, epoch)

        print("Epoch", epoch, "loss", np.mean(e_loss), "acc", np.mean(e_acc), "acc_z", np.mean(e_acc_z), "updates",
              e_updates, flush=True)
