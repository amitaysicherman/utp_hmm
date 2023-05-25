import torch
import torch.nn as nn
from torch.nn import TransformerEncoder,TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import random
import numpy as np
from torch.nn.parallel import DataParallel
from mlm_pytorch import MLM
from x_transformers import TransformerWrapper, Encoder



input_size = 41  # Number of tokens (0-39 + padding token)
d_model = 768
nhead =  12
num_layers =  12
batch_size =  256
num_epochs = 100
max_len = 250
mask_value=input_size-1
padding_value = input_size

dim_feedforward = 2048




class PhonemesDataset(Dataset):
    def __init__(self):
        data_flat = np.load('LR960_PH.npz')['a']
        with open('LR960_PH_LEN.txt', 'r') as f:
            lengths = f.read().splitlines()
        lengths = [int(i) for i in lengths]
        curr = 0
        self.x = []
        # self.y = []
        # self.m = []

        for l in tqdm(lengths):
            sequence = data_flat[curr:curr+l]
            if len(sequence) > max_len:
                sequence = np.array(list(sequence[:max_len]), dtype=np.int8)
            else:
                sequence = np.array(list(sequence)+[padding_value]*(max_len-len(sequence)), dtype=np.int8)
            self.x.append(sequence)
            

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.LongTensor(self.x[idx]).to(device)
        


if __name__ == '__main__' :
    model = TransformerWrapper(
    num_tokens = input_size+1,
    max_seq_len = max_len,
        attn_layers = Encoder(
            dim = d_model,
            depth = num_layers,
            heads = nhead
        )
    )

    print(model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'{params:,} trainable parameters')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    criterion = nn.CrossEntropyLoss().to(device)

    dataset = PhonemesDataset()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(len(dataset))
    
    
    trainer = MLM(
        model,
        mask_token_id = mask_value,          # the token id reserved for masking
        pad_token_id = padding_value,           # the token id for padding
        mask_prob = 0.15,           # masking probability for masked language modeling
        random_token_prob=0.1,      # chance of replacing a mask token with a random token from the entire vocab
        replace_prob = 0.90,        # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
        mask_ignore_token_ids = []  # other tokens to exclude from masking, include the [cls] and [sep] here
    ).to(device)
    optimizer = torch.optim.Adam(trainer.parameters(), lr=3e-4)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        total_accuracy = 0
        for (x) in tqdm(data_loader):
            loss = trainer(x)
            loss.backward()
            total_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                model.eval()
                y=x.clone()                
                mask = torch.randn(x.shape) <= random.random()
                mask[x == padding_value] = False
                random_tokens = torch.randint_like(x, input_size)
                x[mask] = random_tokens[mask]
                output = model(x)
                output = output[mask]
                y=y[mask]
                
                predicted_labels = torch.argmax(output, dim=1)
                print(predicted_labels[:10])
                correct_predictions = (predicted_labels == y).sum().item()
                total_accuracy += correct_predictions / (y.numel())
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), f'{epoch}.cp')
        else:
            torch.save(model.state_dict(), f'{epoch}.cp')
        print(f"Epoch {epoch+1} Loss: {total_loss / len(data_loader)}")
        print(f"Epoch {epoch+1} Accuracy: {total_accuracy / len(data_loader)}")
