#sbatch --gres=gpu:1,vmem:24g --mem=75G --time=0-3 --wrap "python train_superv_ctc.py"

from mapping import phonemes_to_index

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from jiwer import wer

# Load features and phonemes
data_name = "p_superv"
input_file = f"pseg/data/{data_name}/features.npy"
len_file = f"pseg/data/{data_name}/features.length"
phonemes_file = f"pseg/data/{data_name}/features.phonemes"

features = np.load(input_file)
with open(len_file, 'r') as f:
    lengths = f.read().split("\n")
lengths = [int(l) for l in lengths]
assert sum(lengths) == len(features)
features = np.split(features, np.cumsum(lengths)[:-1])
with open(phonemes_file) as f:
    phonemes = f.read().splitlines()
phonemes = [[phonemes_to_index[y.upper()] if y != "dx" else phonemes_to_index['T'] for y in x.split()] for x in
            phonemes]
phonemes = [np.array(x) for x in phonemes]

blank_label = max(phonemes_to_index.values()) + 1


# Custom Dataset with Padding
class FeaturesPhonemesDataset(Dataset):
    def __init__(self, features, phonemes):
        self.features = [torch.tensor(f) for f in features]
        self.phonemes = [torch.tensor(p) for p in phonemes]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.phonemes[idx]

    def collate_fn(self, batch):
        features, phonemes = zip(*batch)
        features_lens = torch.tensor([len(f) for f in features])
        phonemes_lens = torch.tensor([len(p) for p in phonemes])
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        phonemes = torch.nn.utils.rnn.pad_sequence(phonemes, batch_first=True, padding_value=blank_label)
        return features, phonemes, features_lens, phonemes_lens


# Linear Model
class FeaturesPhonemesLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeaturesPhonemesLinear, self).__init__()
        self.linear = nn.Linear(input_size, output_size + 1)  # Include additional class for CTC blank label

    def forward(self, x):
        return self.linear(x)


# Create Dataset and DataLoader
dataset = FeaturesPhonemesDataset(features, phonemes)
train_loader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=dataset.collate_fn)

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model, Loss, Optimizer
model = FeaturesPhonemesLinear(768, blank_label).to(device)
criterion = nn.CTCLoss(blank=blank_label, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training Loop
epochs = 10000
for epoch in range(epochs):
    scores = []  # To store CER scores for each batch

    for batch_idx, (data, target, data_lens, target_lens) in enumerate(train_loader):
        data = data.to(device).float()
        target = target.to(device)

        # Forward pass
        outputs = model(data)
        outputs = torch.nn.functional.log_softmax(outputs, dim=2)
        outputs = outputs.transpose(0, 1)  # CTC requires [T, N, C] shape

        # Compute loss
        loss = criterion(outputs, target, data_lens, target_lens)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, dim=2)
        for i in range(data.size(0)):
            y_hat = predicted[:, i].cpu().numpy()
            y_hat = [str(x) for x in y_hat if x != blank_label]  # Exclude blank labels
            y_hat = [y_hat[0]] + [y_hat[i] for i in range(1, len(y_hat)) if y_hat[i] != y_hat[i - 1]]
            y_hat = " ".join(y_hat)
            y = " ".join([str(x) for x in target[i].cpu().numpy() if x != blank_label])  # Exclude padding
            scores.append(wer(y, y_hat))

        # Print progress
        if batch_idx % 5 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}')
    avg_cer = sum(scores) / len(scores)
    print(f'Epoch [{epoch + 1}/{epochs}], Average CER: {avg_cer * 100:.2f}%')

model_path = "models/linear_superv.cp"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

print("Training completed!")
