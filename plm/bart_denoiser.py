# sbatch --gres=gpu:1,vmem:24g --mem=75G -c4 --time=1-0 --wrap "python bart_denoiser.py"

import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BartConfig, BartForConditionalGeneration
from jiwer import wer

MAX_TOKEN = 38
PAD_TOKEN = MAX_TOKEN + 1
START_TOKEN = PAD_TOKEN + 1
END_TOKEN = START_TOKEN + 1
N_TOKENS = END_TOKEN + 1

MIN_P = 0.0
MAX_P = 0.5

MAX_LENGTH = 256
EPOCHS = 200
BATCH_SIZE = 32
LR = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoiseDataset(Dataset):
    def __init__(self, file_path):
        self.max_len_no_start_end = MAX_LENGTH - 2
        with open(file_path, 'r') as file:
            self.data = [list(map(int, line.strip().split())) for line in file.readlines()]

    def add_noise(self, sample):
        length = len(sample)
        # Replace characters
        for _ in range(random.randint(int(length * MIN_P), int(length * MAX_P))):
            idx = random.randint(0, length - 1)
            sample[idx] = random.randint(0, MAX_TOKEN)

        # Add random characters
        for _ in range(random.randint(int(length * MIN_P), int(length * MAX_P))):
            idx = random.randint(0, len(sample) - 1)
            sample.insert(idx, random.randint(0, MAX_TOKEN))

        # Remove characters
        for _ in range(random.randint(int(length * MIN_P), int(length * MAX_P))):
            if len(sample) > 0:
                idx = random.randint(0, len(sample) - 1)
                sample.pop(idx)

        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            noisy_sample = self.add_noise(self.data[index][:])
            sample = self.data[index][:]
            if len(noisy_sample) <= self.max_len_no_start_end and len(sample) <= self.max_len_no_start_end:
                sample = [START_TOKEN] + sample + [END_TOKEN]
                noisy_sample = [START_TOKEN] + noisy_sample + [END_TOKEN]
                break
            index = random.randint(0, len(self.data) - 1)

        sample = sample + [PAD_TOKEN] * (MAX_LENGTH - len(sample))
        noisy_sample = noisy_sample + [PAD_TOKEN] * (MAX_LENGTH - len(noisy_sample))

        return torch.tensor(noisy_sample), torch.tensor(sample)


def eval_wer_ds(dataset, model):
    model.eval()

    with torch.no_grad():
        wer_score = []
        for i, (noisy_data, clean_data) in enumerate(dataset):
            noisy_data = noisy_data.to(device)
            clean_data = clean_data.to(device)
            outputs = model.generate(noisy_data.unsqueeze(0), max_length=MAX_LENGTH)[0]
            clean_data = [x for x in clean_data.cpu().numpy().tolist() if x not in [PAD_TOKEN, START_TOKEN, END_TOKEN]]
            clean_data = " ".join([str(x) for x in clean_data])
            outputs = [x for x in outputs.cpu().numpy().tolist() if x not in [PAD_TOKEN, START_TOKEN, END_TOKEN]]
            outputs = " ".join([str(x) for x in outputs])
            print(i)
            print(f'clean: {clean_data}')
            print(f'noisy: {outputs}')
            if len(outputs) and len(clean_data):
                wer_score.append(wer(outputs, clean_data))
            if i > 20:
                break

    return np.mean(wer_score)


def get_model() -> BartForConditionalGeneration:
    config = BartConfig(vocab_size=N_TOKENS, max_position_embeddings=MAX_LENGTH, encoder_layers=3, encoder_ffn_dim=256,
                        encoder_attention_heads=4, decoder_layers=3, decoder_ffn_dim=256, decoder_attention_heads=4,
                        d_model=256, pad_token_id=PAD_TOKEN, bos_token_id=START_TOKEN, eos_token_id=END_TOKEN,
                        decoder_start_token_id=START_TOKEN, forced_eos_token_id=END_TOKEN)  # Set vocab size
    model = BartForConditionalGeneration(config)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'{params:,} trainable parameters')
    return model


if __name__ == '__main__':

    train_dataset = NoiseDataset('data/LIBRISPEECH_TRAIN_idx.txt')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = NoiseDataset('data/LIBRISPEECH_TEST_idx.txt')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = get_model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    model.train()
    best_train_loss = 1000
    best_test_loss = 1000
    for epoch in range(EPOCHS):
        train_loss = []
        test_loss = []
        model.train()
        for noisy_data, clean_data in train_loader:
            noisy_data = noisy_data.to(device)
            clean_data = clean_data.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=noisy_data, labels=clean_data)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        with torch.no_grad():
            model.eval()
            for noisy_data, clean_data in test_loader:
                noisy_data = noisy_data.to(device)
                clean_data = clean_data.to(device)
                outputs = model(input_ids=noisy_data, labels=clean_data)
                loss = outputs.loss
                test_loss.append(loss.item())

        train_wer = eval_wer_ds(train_dataset, model)
        test_wer = eval_wer_ds(test_dataset, model)
        print(
            f'Epoch {epoch} train loss: {np.mean(train_loss)} test loss: {np.mean(test_loss)} train wer: {train_wer} test wer: {test_wer}',
            flush=True)

        with open("results/bart_denoiser.txt", 'a') as f:
            f.write(f'Epoch {epoch} train loss: {np.mean(train_loss)} test loss: {np.mean(test_loss)}\n')
            f.write(f'Epoch {epoch} train wer: {train_wer} test wer: {test_wer}\n')
        if best_test_loss > np.mean(test_loss):
            best_test_loss = np.mean(test_loss)
            torch.save(model.state_dict(), f'models/bart_denoiser_lr_test.cp')
            torch.save(optimizer.state_dict(), f'models/bart_denoiser_lr_test_opt.cp')
        if best_train_loss > np.mean(train_loss):
            best_train_loss = np.mean(train_loss)
            torch.save(model.state_dict(), f'models/bart_denoiser_lr_train.cp')
            torch.save(optimizer.state_dict(), f'models/bart_denoiser_lr_train_opt.cp')
