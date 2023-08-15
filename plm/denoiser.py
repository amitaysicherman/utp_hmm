import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
from x_transformers import XTransformer

MAX_TOKEN = 38
START_TOKEN = 39
END_TOKEN = 40
PAD_TOKEN = 41
MAX_LENGTH = 150


class NoiseDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            self.data = [list(map(int, line.strip().split())) for line in file.readlines()]

    def add_noise(self, sample):
        length = len(sample)
        # Replace characters
        for _ in range(random.randint(int(length * 0.05), int(length * 0.15))):
            idx = random.randint(0, length - 1)
            sample[idx] = random.randint(1, MAX_TOKEN)

        # Add random characters
        for _ in range(random.randint(int(length * 0.05), int(length * 0.15))):
            idx = random.randint(0, length - 1)
            sample.insert(idx, random.randint(1, MAX_TOKEN))

        # Remove characters
        for _ in range(random.randint(int(length * 0.05), int(length * 0.15))):
            if len(sample) > 0:
                idx = random.randint(0, len(sample) - 1)
                sample.pop(idx)

        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        noisy_sample = self.add_noise(self.data[index][:])
        sample = self.data[index][:MAX_LENGTH]
        noisy_sample = noisy_sample[:MAX_LENGTH]

        sample = [START_TOKEN] + sample + [END_TOKEN]
        sample = sample + [PAD_TOKEN] * (MAX_LENGTH - len(sample))

        noisy_sample = [START_TOKEN] + noisy_sample + [END_TOKEN]
        noisy_sample = noisy_sample + [PAD_TOKEN] * (MAX_LENGTH - len(noisy_sample))

        return torch.tensor(noisy_sample), torch.tensor(sample)


def loss_fn(output, target):
    loss = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    return loss(output.view(-1, 40), target.view(-1))


def train(model, train_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for noisy_data, clean_data in train_loader:
            optimizer.zero_grad()
            output = model(noisy_data.permute(1, 0), clean_data.permute(1, 0))
            loss = loss_fn(output, clean_data.permute(1, 0))
            loss.backward()
            optimizer.step()


file_path = 'path/to/your/file.txt'
dataset = NoiseDataset(file_path)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
model = XTransformer(
    dim=512,
    enc_num_tokens=PAD_TOKEN + 1,
    enc_depth=6,
    enc_heads=8,
    enc_max_seq_len=MAX_LENGTH,
    dec_num_tokens=PAD_TOKEN + 1,
    dec_depth=6,
    dec_heads=8,
    dec_max_seq_len=MAX_LENGTH,
    tie_token_emb=True  # tie embeddings of encoder and decoder
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train(model, train_loader, optimizer, epochs=10)
