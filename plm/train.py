# sbatch --gres=gpu:4,vmem:24g --mem=75G --time=3-0 --wrap "python train.py"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from torch.nn.parallel import DataParallel
import torch.nn.functional as F
from mapping import phonemes_to_index
from plm.utils import args_parser, get_model_from_args, save_model, Scores, get_config_name

args = args_parser()

INPUT_SIZE = len(phonemes_to_index) + 1
PADDING_VALUE = INPUT_SIZE


class PhonemesDataset(Dataset):
    def __init__(self, prefix, max_len, padding_value=PADDING_VALUE):

        with open(f'{prefix}_clean.txt', 'r') as f:
            clean_data = f.read().splitlines()
        clean_data = [list(map(int, line.split())) for line in clean_data]
        with open(f'{prefix}_noise.txt', 'r') as f:
            noise_data = f.read().splitlines()
        noise_data = [list(map(int, line.split())) for line in noise_data]
        self.step = 0
        self.x = []
        self.y = []
        for i, (clean, noise) in tqdm(enumerate(zip(clean_data, noise_data)), total=len(clean_data)):
            assert len(clean) == len(noise)
            seq_len = len(clean)
            if seq_len > max_len:
                start_index = np.random.randint(0, seq_len - max_len)
                clean = clean[start_index:start_index + max_len]
                noise = noise[start_index:start_index + max_len]
                clean = np.array(list(clean), dtype=np.int8)
                noise = np.array(list(noise), dtype=np.int8)
            else:
                clean = np.array(list(clean) + [padding_value] * (max_len - len(clean)), dtype=np.int8)
                noise = np.array(list(noise) + [padding_value] * (max_len - len(noise)), dtype=np.int8)
            self.x.append(noise)
            self.y.append(clean)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.LongTensor(self.x[idx]), torch.LongTensor(self.y[idx])


def single_round(model, x, y, is_train, scorer):
    x = x.to(device)
    y = y.to(device)
    logits = model(x)
    loss = F.cross_entropy(
        logits.transpose(1, 2),
        y,
        ignore_index=PADDING_VALUE
    )
    if is_train:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scorer.update(y, logits, loss)


if __name__ == '__main__':
    model = get_model_from_args(args)

    print(model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'{params:,} trainable parameters')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    criterion = nn.CrossEntropyLoss().to(device)

    train_dataset = PhonemesDataset(args.data_train, args.max_len)
    val_dataset = PhonemesDataset(args.data_val, args.max_len)
    test_dataset = PhonemesDataset(args.data_test, args.max_len)

    train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    val_data = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_data = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        config_name = get_config_name(args)
        train_scores = Scores("train", config_name)
        val_scores = Scores("val", config_name)
        test_scores = Scores("test", config_name)
        model.train()
        for (x, y) in tqdm(train_data):
            single_round(model, x, y, True, train_scores)

        model.eval()
        with torch.no_grad():
            for (x, y) in tqdm(val_data):
                single_round(model, x, y, False, val_scores)
            for (x, y) in tqdm(test_data):
                single_round(model, x, y, False, test_scores)
        save_model(model, optimizer, args, epoch)
        print("Epoch", epoch)
        print(train_scores)
        print(val_scores)
        print(test_scores)

        train_scores.save_and_reset()
        val_scores.save_and_reset()
        test_scores.save_and_reset()
