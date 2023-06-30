# sbatch --gres=gpu:4,vmem:24g --mem=75G --time=3-0 --wrap "python train.py"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from torch.nn.parallel import DataParallel
from x_transformers import TransformerWrapper, Encoder
import torch.nn.functional as F
from collections import defaultdict
from mapping import phonemes_to_index, mis_index

input_size = 40  # Number of tokens (0-38 + padding token)
d_model = 768  # 256
nhead = 12  # 4
num_layers = 12  # 6
batch_size = 512  # 2048

num_epochs = 1000
max_len = 100
mask_value = input_size - 1
padding_value = input_size
do_dropout = False

train_file = f"lr_train"  # "
val_file = f"lr_train_val.txt"
test_file = f"lr_test"
config_name = "lr_large_005"  #

lr = 5e-3


class Scores:
    def __init__(self, name):
        self.name = name
        self.loss = []
        self.acc = []

    def update(self, loss, acc):
        self.loss.append(loss)
        self.acc.append(acc)

    def __repr__(self):
        return f"\n{self.name} loss: {np.mean(self.loss)} \n{self.name} acc: {np.mean(self.acc)}"

    def reset(self):
        self.loss = []
        self.acc = []


class PhonemesDataset(Dataset):
    def __init__(self, prefix, max_len=max_len,
                 padding_value=padding_value):

        with open(f'{prefix}_clean.txt', 'r') as f:
            clean_data = f.read().splitlines()
        clean_data = [list(map(int, line.split())) for line in clean_data]
        with open(f'{prefix}_noise.txt', 'r') as f:
            noise_data = f.read().splitlines()
        noise_data = [list(map(int, line.split())) for line in noise_data]
        self.step = 0
        self.x = []
        self.y = []
        # self.noise_levels = defaultdict(list)
        for i, (clean, noise) in tqdm(enumerate(zip(clean_data, noise_data)), total=len(clean_data)):
            n = (100 * (np.array(noise) != np.array(clean)).mean()) // 5
            # self.noise_levels[n].append(i)
            assert len(clean) == len(noise)
            seq_len = len(clean)
            if seq_len > max_len:
                clean = np.array(list(clean[:max_len]), dtype=np.int8)
                noise = np.array(list(noise[:max_len]), dtype=np.int8)
            else:
                clean = np.array(list(clean) + [padding_value] * (max_len - len(clean)), dtype=np.int8)
                noise = np.array(list(noise) + [padding_value] * (max_len - len(noise)), dtype=np.int8)

            self.x.append(noise)
            self.y.append(clean)

    def __len__(self):
        return len(self.x)# len(self.noise_levels[self.step])

    def __getitem__(self, idx):
        # idx = self.noise_levels[self.step][idx]
        return torch.LongTensor(self.x[idx]), torch.LongTensor(self.y[idx])


def get_model(input_size=input_size, d_model=d_model, nhead=nhead, num_layers=num_layers, max_len=max_len):
    if do_dropout:
        dropout = 0.1
    else:
        dropout = 0.0
    model = TransformerWrapper(
        num_tokens=input_size + 1,
        max_seq_len=max_len,
        emb_dropout=dropout,
        attn_layers=Encoder(
            dim=d_model,
            depth=num_layers,
            heads=nhead,
            layer_dropout=dropout,  # stochastic depth - dropout entire layer
            attn_dropout=dropout,  # dropout post-attention
            ff_dropout=dropout  # feedforward dropout
        )
    )
    return model


def single_round(model, x, y, is_train, scorer):
    x = x.to(device)
    y = y.to(device)
    logits = model(x)
    loss = F.cross_entropy(
        logits.transpose(1, 2),
        y,
        ignore_index=padding_value
    )
    if is_train:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    predicted_labels = torch.argmax(logits, dim=-1)
    predicted_labels = predicted_labels[y != padding_value]
    y = y[y != padding_value]
    acc = (predicted_labels == y).sum().item() / y.numel()
    scorer.update(loss.item(), acc)


if __name__ == '__main__':
    model = get_model()

    print(model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'{params:,} trainable parameters')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    criterion = nn.CrossEntropyLoss().to(device)

    train_dataset = PhonemesDataset(train_file)
    val_dataset = PhonemesDataset(val_file)
    test_dataset = PhonemesDataset(test_file)
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    val_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # step_count = 0
    for epoch in range(0, num_epochs):
        # step_count += 1

        train_scores = Scores("train")
        val_scores = Scores("val")
        test_scores = Scores("test")
        model.train()
        for (x, y) in tqdm(train_data):
            single_round(model, x, y, True, train_scores)

        model.eval()
        with torch.no_grad():
            for (x, y) in tqdm(val_data):
                single_round(model, x, y, False, val_scores)
            for (x, y) in tqdm(test_data):
                single_round(model, x, y, False, test_scores)

        cp_name = f"models/{config_name}_{epoch}.cp"
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), cp_name)
        else:
            torch.save(model.state_dict(), cp_name)
        torch.save(optimizer.state_dict(), cp_name.replace(".cp", "_opt.cp"))

        print("Epoch", epoch)
        print(train_scores)
        print(val_scores)
        print(test_scores)

        with open(f"results_{config_name}.txt", "a") as f:
            f.write(f"Epoch {epoch}\n")
            f.write(f"{train_scores}\n")
            f.write(f"{val_scores}\n")
            f.write(f"{test_scores}\n")
            f.write("\n")

        # acc = 100 * np.mean(train_scores.acc)
        # noise_level = (0.5 + train_dataset.step) * 5
        # acc_indentity = 100 - noise_level + np.sqrt(noise_level)
        #
        # if acc > acc_indentity or step_count > 20:
        #     print("Noise level increased")
        #     print(f"Current noise level: {train_dataset.step * 5}%")
        #     print("epoch", epoch)
        #     train_dataset.step += 1
        #     val_dataset.step += 1
        #     test_dataset.step += 1
        #     step_count = 0
        #     if train_dataset.step == 20:
        #         break

        train_scores.reset()
        val_scores.reset()
        test_scores.reset()
