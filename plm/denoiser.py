# sbatch --gres=gpu:2,vmem:24g --mem=75G -c4 --time=7-0 --wrap "python denoiser.py"

import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
from x_transformers import XTransformer
import numpy as np

MAX_TOKEN = 38
PAD_TOKEN = 39
MIN_P = 0.0
MAX_P = 0.3
START_TOKEN = 40
END_TOKEN = 41
N_TOKENS = 42
MAX_LENGTH = 100
EPOCHS = 200
BATCH_SIZE = 32
LR = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoiseDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            self.data = [list(map(int, line.strip().split())) for line in file.readlines()]

    def add_noise(self, sample):
        length = len(sample)
        # Replace characters
        for _ in range(random.randint(int(length * MIN_P), int(length * MAX_P))):
            idx = random.randint(0, length - 1)
            sample[idx] = random.randint(1, MAX_TOKEN)

        # Add random characters
        for _ in range(random.randint(int(length * MIN_P), int(length * MAX_P))):
            idx = random.randint(0, length - 1)
            sample.insert(idx, random.randint(1, MAX_TOKEN))

        # Remove characters
        for _ in range(random.randint(int(length * MIN_P), int(length * MAX_P))):
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


def get_denoiser_model():
    return XTransformer(
        pad_value=PAD_TOKEN,
        ignore_index=PAD_TOKEN,
        dim=256,
        enc_num_tokens=N_TOKENS,
        enc_depth=3,
        enc_heads=4,
        enc_max_seq_len=MAX_LENGTH,
        dec_num_tokens=N_TOKENS,
        dec_depth=3,
        dec_heads=4,
        dec_max_seq_len=MAX_LENGTH,
        tie_token_emb=False
    )


if __name__ == '__main__':

    train_loader = DataLoader(NoiseDataset('data/TIMIT_TRAIN_PH_IDX.txt'), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(NoiseDataset('data/TIMIT_TEST_PH_IDX.txt'), batch_size=BATCH_SIZE, shuffle=True)

    model = get_denoiser_model().to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'{params:,} trainable parameters')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    model.train()
    best_train_loss = 1000
    best_test_loss = 1000
    for epoch in range(EPOCHS):
        train_loss = []
        test_loss = []
        for noisy_data, clean_data in train_loader:
            noisy_data = noisy_data.to(device)
            clean_data = clean_data.to(device)
            optimizer.zero_grad()
            loss = model(noisy_data, clean_data)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        for noisy_data, clean_data in test_loader:
            noisy_data = noisy_data.to(device)
            clean_data = clean_data.to(device)
            loss = model(noisy_data, clean_data)
            test_loss.append(loss.item())
        print(f'Epoch {epoch} train loss: {np.mean(train_loss)} test loss: {np.mean(test_loss)}', flush=True)
        with open("results/denoiser.txt", 'a') as f:
            f.write(f'Epoch {epoch} train loss: {np.mean(train_loss)} test loss: {np.mean(test_loss)}\n')
        if best_test_loss > np.mean(test_loss):
            best_test_loss = np.mean(test_loss)
            torch.save(model.state_dict(), f'models/denoiser_best_test_loss.cp')
            torch.save(optimizer.state_dict(), f'models/denoiser_opt_best_test_loss.cp')
        if best_train_loss > np.mean(train_loss):
            best_train_loss = np.mean(train_loss)
            torch.save(model.state_dict(), f'models/denoiser_best_train_loss.cp')
            torch.save(optimizer.state_dict(), f'models/denoiser_opt_best_train_loss.cp')
