# sbatch --gres=gpu:2,vmem:16g --mem=32G --time=7-0 --wrap "python train.py"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import numpy as np
from torch.nn.parallel import DataParallel
from mlm_pytorch import MLM
from x_transformers import TransformerWrapper, Encoder

input_size = 41  # Number of tokens (0-39 + padding token)
d_model = 768
nhead = 12
num_layers = 12
batch_size = 64
num_epochs = 100
max_len = 150
mask_value = input_size - 1
padding_value = input_size

mask_prob = 0
random_token_prob = -1


def get_cp_name(epoch,acc,loss):
    return f"models/{mask_prob}_{random_token_prob}_{epoch}_{loss}_{acc}.cp"

class PhonemesDataset(Dataset):
    def __init__(self, data_path='LR960_PH.npz', data_len_path="LR960_PH_LEN.txt", max_len=max_len,
                 padding_value=padding_value):
        data_flat = np.load(data_path)['a']
        with open(data_len_path, 'r') as f:
            lengths = f.read().splitlines()
        lengths = [int(i) for i in lengths]
        curr = 0
        self.x = []

        for l in tqdm(lengths):
            sequence = data_flat[curr:curr + l]
            if len(sequence) > max_len:
                sequence = np.array(list(sequence[:max_len]), dtype=np.int8)
            else:
                sequence = np.array(list(sequence) + [padding_value] * (max_len - len(sequence)), dtype=np.int8)
            self.x.append(sequence)
            curr += l

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.LongTensor(self.x[idx])


if __name__ == '__main__':
    model = TransformerWrapper(
        num_tokens=input_size + 1,
        max_seq_len=max_len,
        attn_layers=Encoder(
            dim=d_model,
            depth=num_layers,
            heads=nhead
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

    train_data = DataLoader(PhonemesDataset(), batch_size=batch_size, shuffle=False)
    test_data = DataLoader(PhonemesDataset('LRTEST_PH.npz', "LRTEST_PH_LEN.txt"), batch_size=batch_size, shuffle=False)

    trainer = MLM(
        model,
        mask_token_id=mask_value,
        num_tokens=input_size + 1,
        pad_token_id=padding_value,
        mask_prob=mask_prob,
        random_token_prob=random_token_prob,
        replace_prob=0.90,
        mask_ignore_token_ids=[]
    ).to(device)
    optimizer = torch.optim.Adam(trainer.parameters(), lr=3e-4)

    # Training loop
    for epoch in range(num_epochs):
        train_total_loss = 0
        train_total_accuracy = 0
        for (x) in tqdm(train_data):
            x = x.to(device)
            loss = trainer(x)
            loss.backward()
            train_total_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                model.eval()
                y = x.clone()
                mask = torch.randn(x.shape) <= random.random()
                mask[x == padding_value] = False
                random_tokens = torch.randint_like(x, input_size)
                x[mask] = random_tokens[mask]
                output = model(x)
                output = output[mask]
                y = y[mask]

                predicted_labels = torch.argmax(output, dim=1)
                correct_predictions = (predicted_labels == y).sum().item()
                train_total_accuracy += correct_predictions / (y.numel())
        model.eval()
        test_total_loss = 0
        test_total_accuracy = 0
        with torch.no_grad():
            for (x) in tqdm(test_data):
                x = x.to(device)
                loss = trainer(x)
                test_total_loss += loss.item()

                y = x.clone()
                mask = torch.randn(x.shape) <= random.random()
                mask[x == padding_value] = False
                random_tokens = torch.randint_like(x, input_size)
                x[mask] = random_tokens[mask]
                output = model(x)
                output = output[mask]
                y = y[mask]

                predicted_labels = torch.argmax(output, dim=1)
                correct_predictions = (predicted_labels == y).sum().item()
                test_total_accuracy += correct_predictions / (y.numel())
        cp_name=get_cp_name(epoch,test_total_accuracy,test_total_loss)
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), cp_name)
        else:
            torch.save(model.state_dict(), cp_name)
        print(f"Epoch {epoch + 1} Loss: {train_total_loss / len(train_data)}")
        print(f"Epoch {epoch + 1} Accuracy: {train_total_accuracy / len(train_data)}")
        print(f"Epoch {epoch + 1} Test Loss: {test_total_loss / len(test_data)}")
        print(f"Epoch {epoch + 1} Test Accuracy: {test_total_accuracy / len(test_data)}")
