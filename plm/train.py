import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import math
import numpy as np

# Set hyperparameters
input_size = 41  # Number of tokens (0-39 + padding token)
d_model = 384
nhead = 6
num_layers = 6
batch_size = 2
num_epochs = 10
max_len=250

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers,dropout=0.1,dim_feedforward=256):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Embedding(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.transformer = Transformer(d_model, nhead, num_layers,num_decoder_layers=0, dropout=dropout,dim_feedforward=dim_feedforward)
        self.decoder = nn.Linear(d_model, input_size)

    def forward(self, src):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer(src, src)
        output = self.decoder(output)
        return output

# Define the custom dataset
class CustomDataset(Dataset):
    def __init__(self, sequences, max_length=max_len,padding_value=input_size-1):
        self.sequences = []
        for sequence in sequences:
            if len(sequence) > max_length:
                self.sequences.append(sequence[:max_length])
            else:
                self.sequences.append(torch.LongTensor(sequence+[padding_value]*(max_length-len(sequence))))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]



# Generate random sequences
sequences = [[random.randint(0, input_size-1) for _ in range(random.randint(10, 250))] for _ in range(100)]



# Create the model
model = TransformerModel(input_size, d_model, nhead, num_layers)

print(model)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f'{params:,} trainable parameters')
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create data loader
dataset = CustomDataset(sequences)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        mask = torch.randn(batch.shape)  <=0.4
        random_tokens=torch.randint_like(batch,input_size) # Generate random tokens
        batch[mask]= random_tokens[mask] # Replace tokens with random tokens
        # Forward pass
        output = model(batch)
        output=output[mask]
        batch=batch[mask]
        # Compute loss
        loss = criterion(output.view(-1, input_size), batch.view(-1))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss / len(data_loader)}")

