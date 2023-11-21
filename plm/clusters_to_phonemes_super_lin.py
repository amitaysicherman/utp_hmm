# sbatch --gres=gpu:1,vmem:8g --mem=16G -c4 --time=1-0 --wrap "python clusters_to_phonemes_super_lin.py"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from jiwer import wer

input_size = 200
input_pad = input_size
output_size = 39  # Range 0-38
output_pad = output_size
blank = output_size + 1
ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomDataset(Dataset):
    def __init__(self, input_file, output_file, max_seq_length=512):
        with open(input_file) as f:
            input_data = [[int(x) for x in l.split()] for l in f.read().splitlines()]
        with open(output_file) as f:
            output_data = [[int(x) for x in l.split()] for l in f.read().splitlines()]
        self.input_data = []
        self.output_data = []
        for i, o in zip(input_data, output_data):
            if len(i) < max_seq_length and len(o) < max_seq_length:
                self.input_data.append(i)
                self.output_data.append(o)

        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_seq = self.input_data[idx]
        output_seq = self.output_data[idx]
        input_len = len(input_seq)
        input_seq = np.pad(input_seq, (0, self.max_seq_length - len(input_seq)), mode='constant',
                           constant_values=input_pad)
        output_len = len(output_seq)
        output_seq = np.pad(output_seq, (0, self.max_seq_length - len(output_seq)), mode='constant',
                            constant_values=output_pad)

        return input_seq, output_seq, input_len, output_len


class SimpleSeq2SeqModel(nn.Module):
    def __init__(self, hidden_size=256):
        super(SimpleSeq2SeqModel, self).__init__()
        self.embedding = nn.Embedding(input_size + 1, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size + 2)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        return x


def dataset_loop(data, test=False):
    losses = []
    accs = []
    if test:
        model.eval()
    else:
        model.train()

    for batch in tqdm(data):
        if batch is None:
            continue
        inputs, target, input_lengths, target_lengths = batch
        if not test:
            optimizer.zero_grad()

        inputs = torch.tensor(inputs, dtype=torch.long).to(device)
        input_lengths = torch.tensor(input_lengths, dtype=torch.long).to(device)

        target = torch.tensor(target, dtype=torch.long).to(device)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long).to(device)

        # Forward pass
        outputs = model(inputs)

        outputs = torch.nn.functional.log_softmax(outputs, dim=2)
        outputs = outputs.transpose(0, 1)
        loss = ctc_loss(outputs, target, input_lengths, target_lengths)

        if not test:
            loss.backward()
            optimizer.step()

        _, predicted = outputs.max(2)
        predicted = predicted.T
        acc = []
        for p, l, t, tl in zip(predicted, input_lengths, target, target_lengths):
            p = p[:l]
            p = p[p != blank]
            t = t[:tl]
            p = p.detach().cpu().numpy().tolist()
            p = " ".join([str(x) for x in p])
            t = t.detach().cpu().numpy().tolist()
            t = " ".join([str(x) for x in t])
            acc.append(wer(t, p))

        accs.append(np.mean(acc))
        losses.append(loss.item())

    return np.mean(losses), np.mean(accs)


if __name__ == '__main__':

    model = SimpleSeq2SeqModel().to(device)

    train_data = DataLoader(CustomDataset("data/LIBRISPEECH_TRAIN_clusters_200.txt", "data/LIBRISPEECH_TRAIN_idx.txt"),
                            batch_size=64, shuffle=True, drop_last=True)
    test_data = DataLoader(CustomDataset("data/LIBRISPEECH_TEST_clusters_200.txt", "data/LIBRISPEECH_TEST_idx.txt"),
                           batch_size=64, shuffle=True, drop_last=True)

    # Use AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 100
    best_test_acc = 0
    for epoch in range(num_epochs):

        train_loss, train_acc = dataset_loop(train_data)
        test_loss, test_acc = dataset_loop(test_data, True)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss}, Train ACC :{train_acc}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss}, Test ACC :{test_acc}")
        if test_acc > best_test_acc:
            best_test_acc = test_acc

            torch.save(model.state_dict(), "models/simple_seq2seq_model_best_200.pth")
            mapping = torch.arange(100).to(device).unsqueeze(0)
            mapping = model(mapping)[0].cpu().detach().numpy().argmax(axis=-1)
            with open("models/clusters_phonemes_map_200.txt", "w") as f:
                f.write("\n".join([str(x) for x in mapping]))

    # Save the trained model
    torch.save(model.state_dict(), "models/simple_seq2seq_model_200.pth")
