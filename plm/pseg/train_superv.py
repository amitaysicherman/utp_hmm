import torch
import torch.nn as nn
import torch.optim as optim
import jiwer


# Define the model
class MyModel(nn.Module):
    def __init__(self, num_units, num_phonemes):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(num_units, num_phonemes)

    def forward(self, x):
        x = self.embedding(x)
        return x


# Define a function to convert phonemes from text to indexes
def convert_phonemes_to_indexes(phonemes, phoneme_to_index):
    return [phoneme_to_index[p] + 1 for p in phonemes]  # Shift the indexes by 1


# Read units from input file
units_file = "./data/clusters_100.txt"
with open(units_file, 'r') as f:
    units_data = f.readlines()
units_data = [line.strip().split() for line in units_data]
units = [[int(u) for u in line] for line in units_data]

# Read phonemes from input file
phonemes_file = "./data/features.phonemes"  # Replace with your phonemes file path
with open(phonemes_file, 'r') as f:
    phonemes_data = f.readlines()
phonemes_data = [line.strip().split() for line in phonemes_data]
phoneme_to_index = {p: i for i, p in enumerate(set(sum(phonemes_data, [])))}
phonemes = [convert_phonemes_to_indexes(line, phoneme_to_index) for line in phonemes_data]

# Convert input data to tensors
units_tensor = torch.tensor(units, dtype=torch.long)
phonemes_tensor = torch.tensor(phonemes, dtype=torch.long)

# Set hyperparameters
num_units = 100  # Number of unique units
num_phonemes = 39  # Number of unique phonemes
learning_rate = 0.001
num_epochs = 100

# Create an instance of the model
model = MyModel(num_units, num_phonemes)

# Define the CTC loss function
ctc_loss = nn.CTCLoss(blank=0)  # Set blank value to 0

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(units_tensor)

    # Calculate the CTC loss
    input_lengths = torch.full((units_tensor.size(0),), outputs.size(1), dtype=torch.long)
    target_lengths = torch.full((phonemes_tensor.size(0),), phonemes_tensor.size(1), dtype=torch.long)
    loss = ctc_loss(outputs.transpose(0, 1), phonemes_tensor, input_lengths, target_lengths)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss for monitoring
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    # Evaluate unit edit distance
    _, predicted_indexes = outputs.max(dim=2)
    predicted_indexes = predicted_indexes.transpose(0, 1).tolist()
    predicted_phonemes = [[list(phoneme_to_index.keys())[i - 1] for i in p] for p in predicted_indexes]

    # Calculate unit edit distance
    ued = []
    for i in range(len(phonemes)):
        ref_p = " ".join([str(pp) for pp in phonemes[i]])
        hyp_p = " ".join([str(pp) for pp in predicted_indexes[i]])
        ued.append(jiwer.wer(ref_p, hyp_p))
    ued = sum(ued) / len(ued)
    print(f"Unit Edit Distance: {ued}")

    torch.save(model.state_dict(), f"./models/superv_{epoch}_{ued}.pth")
