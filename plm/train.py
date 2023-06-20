# sbatch --gres=gpu:2,vmem:24g --mem=75G --time=7-0 --wrap "python train.py"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from torch.nn.parallel import DataParallel
from x_transformers import TransformerWrapper, Encoder
import torch.nn.functional as F

input_size = 40  # Number of tokens (0-38 + padding token)
d_model = 256
nhead = 4
num_layers = 6
batch_size = 2048
num_epochs = 100
max_len = 50
mask_value = input_size - 1
padding_value = input_size

config_name = "prep_random_small_timit"

noise_p = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

lr = 5e-4


class Scores:
    def __init__(self, name):
        self.name = name
        self.loss = [0] * len(noise_p)
        self.acc = [0] * len(noise_p)
        self.counts = [0] * len(noise_p)

    def update(self, loss, acc, p):
        i = noise_p.index(p)
        self.loss[i] += loss
        self.acc[i] += acc
        self.counts[i] += 1

    def __repr__(self):
        loss = [self.loss[i] / self.counts[i] if self.counts[i] > 0 else 0 for i in range(len(noise_p))]
        acc = [self.acc[i] / self.counts[i] if self.counts[i] > 0 else 0 for i in range(len(noise_p))]
        tot_loss = sum([l * c for l, c in zip(loss, self.counts)]) / sum(self.counts)
        tot_acc = sum([a * c for a, c in zip(acc, self.counts)]) / sum(self.counts)
        return f"{self.name} loss: {loss} \n{self.name} acc: {acc} \n{self.name} counts: {self.counts}" \
               f"\n{self.name} total loss: {tot_loss} \n{self.name} total acc: {tot_acc}"

    def reset(self):
        self.loss = [0] * len(noise_p)
        self.acc = [0] * len(noise_p)
        self.counts = [0] * len(noise_p)


class PhonemesDataset(Dataset):
    def __init__(self, prefix, max_len=max_len,
                 padding_value=padding_value):

        with open(f'{prefix}_clean.txt', 'r') as f:
            clean_data = f.read().splitlines()
        clean_data = [list(map(int, line.split())) for line in clean_data]
        with open(f'{prefix}_noise.txt', 'r') as f:
            noise_data = f.read().splitlines()
        noise_data = [list(map(int, line.split())) for line in noise_data]

        self.x = []
        self.y = []
        for clean, noise in tqdm(zip(clean_data, noise_data), total=len(clean_data)):
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

        return len(self.x)

    def __getitem__(self, idx):
        return torch.LongTensor(self.x[idx]), torch.LongTensor(self.y[idx])


def get_model(input_size=input_size, d_model=d_model, nhead=nhead, num_layers=num_layers, max_len=max_len):
    model = TransformerWrapper(
        num_tokens=input_size + 1,
        max_seq_len=max_len,
        attn_layers=Encoder(
            dim=d_model,
            depth=num_layers,
            heads=nhead
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
    x = x[x != padding_value]
    acc = (predicted_labels == y).sum().item() / y.numel()
    p = (x != y).cpu().numpy().astype(int).mean()
    p = round(p, 1)
    scorer.update(loss.item(), acc, p)


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

    train_data = DataLoader(PhonemesDataset("TIMIT_TRAIN_PH"), batch_size=batch_size, shuffle=False, drop_last=True)
    test_data = DataLoader(PhonemesDataset("TIMIT_TEST_PH"), batch_size=batch_size, shuffle=False,
                           drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(0, num_epochs):
        train_scores = Scores("train")
        test_scores = Scores("test")
        model.train()
        for (x, y) in tqdm(train_data):
            single_round(model, x, y, True, train_scores)

        model.eval()
        with torch.no_grad():
            for (x, y) in tqdm(test_data):
                single_round(model, x, y, False, test_scores)

        cp_name = f"models/{config_name}_{epoch}.cp"
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), cp_name)
        else:
            torch.save(model.state_dict(), cp_name)

        print("Epoch", epoch)
        print(train_scores)
        print(test_scores)

        with open(f"results_{config_name}.txt", "a") as f:
            f.write(f"Epoch {epoch}\n")
            f.write(f"{train_scores}\n")
            f.write(f"{test_scores}\n")
            f.write("\n")

        train_scores.reset()
        test_scores.reset()
