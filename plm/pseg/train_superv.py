import torch
import torch.nn as nn
import torch.optim as optim
import jiwer
import numpy as np
from tqdm import tqdm

# torch.autograd.set_detect_anomaly(True)

MAX_LEN = 100
PADDING_VALUE = 0
# Set hyperparameters
num_units = 101  # Number of unique units
num_phonemes = 39  # Number of unique phonemes
BLANK_VALUE = num_phonemes + 1
learning_rate = 0.1
num_epochs = 100


# Define the model
class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()

        units_to_phonemes = np.zeros((input_dim, output_dim))
        for i in range(len(units_to_phonemes)):
            units_to_phonemes[i] = np.random.dirichlet(np.ones(output_dim), size=1)
        units_to_phonemes = torch.from_numpy(units_to_phonemes)
        self.embedding = nn.Embedding(input_dim, output_dim)  # max_norm=1, norm_type=1
        self.embedding.weight.data.copy_(units_to_phonemes)
        # self.conv = nn.Conv1d(in_channels=output_dim, out_channels=output_dim, kernel_size=3)

    def forward(self, x):
        x = self.embedding(x)
        # x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x

    def print_mapping(self, index_to_phoneme):
        for i in range(len(self.embedding.weight)):
            p_index = self.embedding.weight[i].argmax().item()
            if p_index in index_to_phoneme:
                p = index_to_phoneme[p_index]
            elif p_index == BLANK_VALUE:
                p = "BLANK"
            else:
                p = "UNKNOWN"

            print(p,end=" ")


# Define a function to convert phonemes from text to indexes
def convert_phonemes_to_indexes(phonemes, phoneme_to_index):
    return [phoneme_to_index[p] + 1 for p in phonemes]  # Shift the indexes by 1


def padding(x):
    if len(x) >= MAX_LEN:
        return x[:MAX_LEN]
    else:
        return x + [PADDING_VALUE] * (MAX_LEN - len(x))


# Read units from input file
units_file = f"./data/clusters_{num_units - 1}.txt"
with open(units_file, 'r') as f:
    units_data = f.readlines()
units_data = [line.strip().split() for line in units_data]

phonemes_file ="./data/features.phonemes"
with open(phonemes_file, 'r') as f:
    phonemes_data = f.readlines()
phonemes_data = [line.strip().split() for line in phonemes_data]
phoneme_to_index = {p: i for i, p in enumerate(set(sum(phonemes_data, [])))}
index_to_phoneme = {i: p for p, i in phoneme_to_index.items()}
units = []
phonemes = []
skip_count = 0
units_lens = []
phonemes_lens = []
for u_line, p_line in zip(units_data, phonemes_data):
    # u_line = [u_line[0]] + [u_line[i] for i in range(1, len(u_line)) if u_line[i - 1] != u_line[i]]
    p_line = [p for p in p_line if p != 'sil']
    if len(u_line) < len(p_line) or len(u_line) > MAX_LEN or len(p_line) < 10:
        skip_count += 1
        continue
    units_lens.append(len(u_line))
    u_line = [int(x) + 1 for x in u_line]
    u_line = padding(u_line)
    units.append(u_line)

    phonemes_lens.append(len(p_line))
    p_line = convert_phonemes_to_indexes(p_line, phoneme_to_index)
    p_line = padding(p_line)
    phonemes.append(p_line)

print("skip_count:", skip_count, "/", len(units_data))

units_tensor = torch.tensor(units, dtype=torch.long)
units_lens = torch.tensor(units_lens, dtype=torch.long)
phonemes_tensor = torch.tensor(phonemes, dtype=torch.long)
phonemes_lens = torch.tensor(phonemes_lens, dtype=torch.long)

# Create an instance of the model
model = MyModel(num_units, BLANK_VALUE + 1)

# Define the CTC loss function
ctc_loss = nn.CTCLoss(blank=BLANK_VALUE, zero_infinity=True)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Forward pass

    model.print_mapping(index_to_phoneme)
    outputs = model(units_tensor)

    # Calculate the CTC loss

    loss = ctc_loss(outputs.transpose(0, 1).log_softmax(dim=-1), phonemes_tensor, units_lens, phonemes_lens)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    predicted_indexes = outputs.argmax(dim=2).detach().cpu().numpy()
    ued = []
    for i in range(len(phonemes)):
        l = phonemes_lens[i]
        l2 = units_lens[i]
        ref_p = " ".join([str(pp) for pp in phonemes[i][:l]])

        hyp_p = [pp for pp in predicted_indexes[i][:l2]]
        hyp_p = [hyp_p[0]] + [hyp_p[i] for i in range(1, len(hyp_p)) if hyp_p[i - 1] != hyp_p[i]]
        hyp_p = [pp for pp in hyp_p if pp != BLANK_VALUE and pp != PADDING_VALUE]
        hyp_p = " ".join([str(pp) for pp in hyp_p])
        if epoch == 30 and i < 10:
            print("----")
            print(ref_p)
            print(hyp_p)

        ued.append(jiwer.wer(ref_p, hyp_p))
    ued = sum(ued) / len(ued)
    print()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, UED: {ued}")

    # torch.save(model.state_dict(), f"./models/superv_{epoch}_{ued}.pth")
