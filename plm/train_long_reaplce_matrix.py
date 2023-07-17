# sbatch --gres=gpu:4,vmem:24g --mem=75G --time=3-0 --wrap "python train_long_reaplce_matrix.py --batch_size=32 --lr=1e-5"
import random
from torch.utils.data import Dataset, DataLoader
from utils import get_model, PADDING_VALUE, N_TOKENS, args_parser, Scores, save_model_to_name, load_model
import numpy as np
import torch
from tqdm import tqdm
from mapping import phonemes_to_index
from torch.nn.parallel import DataParallel
import torch.nn as nn
import torch.nn.functional as F

ONE = 0
PROB = 1


class PhonemesDataset(Dataset):
    def __init__(self, phonemes_file="pseg/data/p_superv/features.phonemes", target_units=100, max_len=1024,
                 sep=PADDING_VALUE, size=1_000_000):
        self.target_units = target_units
        with open(phonemes_file, 'r') as f:
            phonemes_data = f.readlines()
        phonemes_data = [[phonemes_to_index[x.upper()] if x.upper() != "DX" else phonemes_to_index["T"] for x in
                          line.strip().split()] for line in phonemes_data]
        self.sep = sep
        self.noise_sep = target_units
        self.data = []
        for _ in range(size):
            sample = []
            while len(sample) < max_len:
                sample += phonemes_data[np.random.randint(0, len(phonemes_data))]
                sample += [sep]
            sample = sample[:max_len]
            self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def build_mapping_one(self):
        assert self.target_units >= len(phonemes_to_index)
        units_mapping = list(range(N_TOKENS))
        units_mapping += [random.randint(0, N_TOKENS - 1) for _ in range(self.target_units - N_TOKENS)]
        random.shuffle(units_mapping)
        units_mapping = np.array(units_mapping)
        inv_mapping = {i: [] for i in range(N_TOKENS)}
        for i, u in enumerate(units_mapping):
            inv_mapping[u].append(i)
        return inv_mapping

    def build_mapping_prob(self):
        mapping_prob = np.zeros((N_TOKENS, self.target_units))
        for i in range(N_TOKENS):
            mapping_prob[i, :] = np.random.dirichlet(np.ones(self.target_units), size=1)

    def build_mapping(self, type=ONE):
        if type == ONE:
            return self.build_mapping_one()
        # elif type==PROB:
        #     return self.build_mapping_prob()
        else:
            raise ValueError("Unknown type")

    def add_noise(self, clean):
        inv_mapping = self.build_mapping()
        noise = []
        for c in clean:
            if c == self.sep:
                noise.append(self.noise_sep)
            else:
                noise.append(random.choice(inv_mapping[c]))
        return noise

    def __getitem__(self, idx):
        clean = self.data[idx]
        noise = self.add_noise(clean)
        return torch.LongTensor(noise), torch.LongTensor(clean)


if __name__ == '__main__':

    args = args_parser()
    model = get_model("transformer", "medium", 1024, 0.0, vocab=101, output_dim=N_TOKENS + 1)
    print(model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'{params:,} trainable parameters')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    load_step = 0
    if args.load_cp:
        model, optimizer = load_model(args.load_cp, model, optimizer)
        load_step = int(args.load_cp.split("_")[-1].replace(".cp", ""))
    print("load_step", load_step, flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer=optimizer.to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    criterion = nn.CrossEntropyLoss().to(device)
    train_dataset = PhonemesDataset()
    train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    for epoch in range(args.epochs):
        config_name = "long_marix"
        train_scores = Scores("train", config_name)
        model.train()
        for i, (x, y) in tqdm(enumerate(train_data), total=len(train_data)):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.transpose(1, 2),
                y,
                ignore_index=train_dataset.sep
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_scores.update(y, logits, loss.item())
            if i % 250 == 0:
                print("Epoch", epoch)
                print(train_scores)
                train_scores.save_and_reset()

            if i % 10000 == 0:
                n = len(train_dataset) * epoch + i + load_step
                save_model_to_name(model, optimizer, f"models/{config_name}_{n}.cp")
