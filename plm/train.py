import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import random
import numpy as np
from torch.nn.parallel import DataParallel


input_size = 41  # Number of tokens (0-39 + padding token)
d_model = 768
nhead = 12
num_layers = 12
batch_size = 256
num_epochs = 100
max_len = 250
padding_value = input_size-1
dim_feedforward=2048

class PhonemesDataset(Dataset):
    def __init__(self):
        data_flat = np.load('LR960_PH.npz')['a']
        with open('LR960_PH_LEN.txt', 'r') as f:
            lengths = f.read().splitlines()
        lengths = [int(i) for i in lengths]
        curr = 0
        self.data = []
        for l in tqdm(lengths):
            sequence = data_flat[curr:curr+l]
            if len(sequence) > max_len:
                sequence = np.array(list(sequence[:max_len]), dtype=np.int8)
            else:
                sequence = np.array(list(sequence)+[padding_value]*(max_len-len(sequence)), dtype=np.int8)
            self.data.append(sequence)
            curr += l

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx])




class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
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
    def __init__(self, input_size, d_model, nhead, num_layers, dropout=0.1, dim_feedforward=dim_feedforward):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Embedding(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.transformer = Transformer(
            d_model, nhead, num_layers, num_decoder_layers=0, dropout=dropout, dim_feedforward=dim_feedforward)
        self.decoder = nn.Linear(d_model, input_size)

    def forward(self, src):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer(src, src)
        output = self.decoder(output)
        return output


# Create the model

if __name__ == '__main__' :
    model = TransformerModel(input_size, d_model, nhead, num_layers)
    print(model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'{params:,} trainable parameters')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        # If multiple GPUs are available, wrap the model with DataParallel
        model = DataParallel(model)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    dataset = PhonemesDataset()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(len(dataset))
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_accuracy = 0
        for batch in tqdm(data_loader):
            batch = batch.to(device)
            y=batch.clone()
            optimizer.zero_grad()
            mask = torch.randn(batch.shape) <= random.random()
            mask[batch == input_size-1] = False
            random_tokens = torch.randint_like(batch, input_size)
            batch[mask] = random_tokens[mask]
            output = model(batch)
            output = output[mask]
            y=y[mask]

            loss = criterion(output.view(-1, input_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            predicted_labels = torch.argmax(output, dim=1)
            correct_predictions = (predicted_labels == y).sum().item()
            total_accuracy += correct_predictions / (y.numel())
        torch.save(model.module.state_dict(), f'{epoch}.cp')
        print(f"Epoch {epoch+1} Loss: {total_loss / len(data_loader)}")
        print(f"Epoch {epoch+1} Accuracy: {total_accuracy / len(data_loader)}")
