import random
from collections import defaultdict
from train import PhonemesDataset
import torch
import torch.nn as nn
import torch.optim as optim
from train import get_model, input_size, max_len, padding_value
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

batch_size = 256


class ReplacePhonemesDataset(PhonemesDataset):
    def __init__(self, data_path='LR960_PH.npz', data_len_path="LR960_PH_LEN.txt", max_len=max_len,
                 padding_value=padding_value, n_units=input_size - 1):
        super().__init__(data_path, data_len_path, max_len, padding_value)
        self.n_units = n_units
        assert n_units >= input_size-1

        n_phonemes = input_size - 1
        mapping = list(range(n_phonemes))

        mapping += [random.randint(0, n_phonemes - 1) for _ in range(n_units - n_phonemes)]
        random.shuffle(mapping)

        print(mapping)
        self.mapping = mapping
        self.inv_mapping = defaultdict(list)
        for i, v in enumerate(self.mapping):
            self.inv_mapping[v].append(i)

        self.y = []

        for x_ in tqdm(self.x):
            y_ = [random.choice(self.inv_mapping[v]) if v != padding_value else padding_value for v in x_]
            self.y.append(y_)

    def __getitem__(self, idx):
        return torch.LongTensor(self.x[idx]), torch.LongTensor(self.y[idx])


class LinearModel(nn.Module):
    def __init__(self, input_dim=input_size + 1, output_dim=input_size + 1):
        super(LinearModel, self).__init__()
        self.emb = nn.Embedding(input_dim, output_dim, max_norm=1, norm_type=1)
        # identity_matrix = torch.eye(input_size + 1)
        # self.emb.weight.data.copy_(identity_matrix)

    def forward(self, x):
        return self.emb(x)


if __name__ == '__main__':
    cp_file = "./models/best.cp"
    data_path = 'LR960_PH.npz'
    data_len_path = 'LR960_PH_LEN.txt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    model.load_state_dict(torch.load(cp_file, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()

    linear_model = LinearModel()
    linear_model.to(device)


    for param in model.parameters():
        param.requires_grad = False

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(linear_model.parameters(), lr=0.01)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.1)
    dataset = ReplacePhonemesDataset(data_path=data_path, data_len_path=data_len_path)

    mapping = linear_model.emb.weight.data.cpu().argmax(dim=-1).numpy()
    print(mapping)

    for ephoc in range(100):
        loss = 0
        loss_count = 0
        acc = 0
        count = 0

        for j, (y, x) in enumerate(dataset):
            x = x.to(device)
            mask = torch.zeros_like(x)
            mask[x != padding_value] = 1
            x = x.unsqueeze(0)

            linear_output = linear_model(x)
            argmax_output = torch.argmax(linear_output.detach(), dim=-1)
            with torch.no_grad():
                pretrained_output = model(argmax_output)
            pretrained_output = pretrained_output.softmax(dim=-1)
            linear_output = linear_output.view(-1, linear_output.shape[-1])
            pretrained_output = pretrained_output.view(-1, pretrained_output.shape[-1])
            masked_inputs = linear_output[mask.view(-1)]
            masked_targets = pretrained_output[mask.view(-1)]
            if len(masked_inputs.shape) < 20:
                continue
            loss += loss_fn(masked_inputs, masked_targets)
            loss_count += 1
            single_x = argmax_output.cpu().numpy().flatten()
            single_y = y.numpy().flatten()
            single_x = single_x[single_y != padding_value]
            single_y = single_y[single_y != padding_value]
            if len(single_x) != len(single_y):
                continue
            acc += (single_x == single_y).sum()
            count += single_y.shape[0]

            if loss_count > 0 and loss_count % batch_size == 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(acc / count, acc, count)
                print(loss.item())
                loss = 0
                acc = 0
                count = 0

                new_mapping = linear_model.emb.weight.data.cpu().argmax(dim=-1).numpy()
                print(new_mapping)
                print((new_mapping != mapping).sum(), "update mapping", flush=True)
                mapping = new_mapping
                # plt.imshow(linear_model.emb.weight.data.cpu().numpy(), cmap='gray')
                # plt.show()

            # scheduler.step()
            # print(f"ephoc {ephoc} loss {np.mean(e_loss)} acc {np.mean(e_acc)}")

    #
