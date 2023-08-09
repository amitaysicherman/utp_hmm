#sbatch --gres=gpu:1,vmem:24g --mem=75G --time=3-0 --wrap "python train_lstm_lm.py"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from mapping import phonemes_to_index

# Define your phoneme-to-index mapping
data_name = "p_superv"
phonemes_file = f"pseg/data/{data_name}/features.phonemes"
padding_values = len(phonemes_to_index)
# Load phonemes from file and convert to integer sequences
with open(phonemes_file) as f:
    data = f.read().splitlines()
data = [[phonemes_to_index[y.upper()] if y != "dx" else phonemes_to_index['T'] for y in x.split()] for x in
        data]

print("read data")


class PhonemeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx][:-1]), torch.LongTensor(self.data[idx][1:])


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=padding_values)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=padding_values)
    return inputs_padded, targets_padded


dataset = PhonemeDataset(data)
print("created dataset", len(dataset))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
print("created dataloader")


# Define the RNN model
# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_size, padding_idx=padding_values)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        output = self.fc(output)
        return output


vocab_size = len(phonemes_to_index)
embed_size = 128
hidden_size = 256
num_layers = 1

model = RNNModel(vocab_size, embed_size, hidden_size, num_layers)

# Training settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

loss_function = nn.CrossEntropyLoss(ignore_index=padding_values)
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# Function to calculate accuracy
def accuracy(predictions, targets, pad_index=padding_values):
    predictions = torch.argmax(predictions, dim=2)
    non_pads = targets != pad_index
    correct_predictions = (predictions == targets) * non_pads
    accuracy = correct_predictions.sum().float() / non_pads.sum().float()
    return accuracy.item()


# Training loop
best_acc = 0
epochs = 1000
for epoch in range(epochs):
    total_loss = 0
    total_accuracy = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_function(outputs.permute(0, 2, 1), targets)
        total_loss += loss.item()
        acc = accuracy(outputs, targets)
        total_accuracy += acc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(
        f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}, Accuracy: {total_accuracy / len(dataloader)}')

    acc = total_accuracy / len(dataloader)
    if acc > best_acc:
        best_acc = acc
        model_path = "models/lm_best.cp"
        torch.save(model.state_dict(), model_path)

    model_path = "models/lm_last.cp"
    torch.save(model.state_dict(), model_path)
