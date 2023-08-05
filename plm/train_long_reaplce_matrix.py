# sbatch --gres=gpu:4,vmem:24g --mem=75G --time=3-0 --wrap "python train_long_reaplce_matrix.py --batch_size=32 --lr=1e-4 --load_cp=models/long_marix_1True_28160000.cp"
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
from scipy.special import softmax
from scipy.spatial.distance import cdist

ONE = 0
PROB = 1
SPHERE = 2
type_ = SPHERE
DUP = True
# each 1|2 phones maping to 1|2 units.
# 1,1. one to one mapping
# 1,2. one to two mapping - single phone maps to two units.
# 2,1. two to one mapping - two phones map to single unit.

# each unit can be mapped to more than one phones (probebility) -  use  scipy.stats.expon with scale between 0-4
# choose in random for each unit. the acctual phonemes in suffle the order .

# how to combine this two?
# choose p1% to conbine and use this as pair (that will map to single phone) [for all this appers of this pair]
# for each phoneme mapping add %n double-units and use them in the probebily.
from scipy.stats import expon


def get_phone_to_unit_probes(n_units=100):
    scale = np.clip(np.random.normal(loc=1.3, scale=1), 0.001, 5)
    probs = expon.pdf(range(n_units), scale=scale)
    probs /= probs.sum()

    np.random.shuffle(probs)
    return np.arange(n_units), probs


def random_gaussian(n, dim=2):
    point = np.random.normal(size=(n, dim))
    point /= np.linalg.norm(point, axis=1, keepdims=True)
    return point


def convert_to_units(phonemes):
    double_prob = 0.01
    all_pairs = set([(phonemes[i], phonemes[i - 1]) for i in range(1, len(phonemes))])
    double_pairs = np.random.choice(all_pairs, int(len(all_pairs) * double_prob))
    mapping = dict()
    for i in range(len(phonemes)):
        p = phonemes[i]
        p_next = phonemes[i + 1] if i + 1 < len(phonemes) else None

        if p_next and (p, p_next) in double_pairs and (p, p_next) not in mapping:
            mapping[(p, p_next)] = get_phone_to_unit_probes()


class PhonemesDataset(Dataset):
    def __init__(self, phonemes_file="pseg/data/p_superv/features.phonemes", target_units=100, max_len=1024,
                 sep=PADDING_VALUE, size=1_000_000, type_=type_, dup=DUP):
        self.target_units = target_units
        with open(phonemes_file, 'r') as f:
            phonemes_data = f.readlines()
        phonemes_data = [[phonemes_to_index[x.upper()] if x.upper() != "DX" else phonemes_to_index["T"] for x in
                          line.strip().split()] for line in phonemes_data]
        self.sep = sep
        self.max_len = max_len
        self.noise_sep = target_units
        self.type = type_
        self.dup = dup

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
        inv_mapping = {i: [] for i in range(N_TOKENS)}

        for i in range(N_TOKENS):
            inv_mapping[i] = get_phone_to_unit_probes(self.target_units)
        return inv_mapping

    def build_mapping_sphere(self):
        phonemes = random_gaussian(N_TOKENS)
        clusters = random_gaussian(self.target_units)
        cosine_distances = 100 * (1 - cdist(phonemes, clusters, metric='cosine'))
        probabilities = softmax(cosine_distances, axis=0)
        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
        np.random.shuffle(probabilities)
        return probabilities

    def build_mapping(self):
        if self.type == ONE:
            return self.build_mapping_one()
        elif self.type == PROB:
            return self.build_mapping_prob()
        elif self.type == SPHERE:
            return self.build_mapping_sphere()
        else:
            raise ValueError("Unknown type")

    def add_noise(self, clean):
        inv_mapping = self.build_mapping()

        values = np.arange(5)
        random_numbers = np.random.random(4)
        sorted_numbers = np.sort(np.concatenate(([0, 1], random_numbers)))
        weights = np.diff(sorted_numbers)
        np.random.shuffle(weights)
        length = random.choices(values, weights=weights, k=len(clean))
        # length = random.choices([0, 1, 2, 3], weights=[0.1, 0.5, 0.3, 0.2], k=len(clean))

        final_clean = []
        final_noise = []
        range_units = np.arange(self.target_units)
        for c in clean:
            if c == self.sep:
                final_clean.append(self.sep)
                final_noise.append(self.noise_sep)
            else:
                for _ in range(length.pop()):
                    final_clean.append(c)
                    if self.type == ONE:
                        final_noise.append(random.choice(inv_mapping[c]))
                    elif self.type == PROB:
                        final_noise.append(np.random.choice(inv_mapping[c][0], p=inv_mapping[c][1]))
                    elif self.type == SPHERE:
                        final_noise.append(np.random.choice(range_units, p=inv_mapping[c]))
                    else:
                        raise ValueError("Unknown type")
        if len(final_clean) < self.max_len:
            final_clean += [self.sep] * (self.max_len - len(final_clean))
            final_noise += [self.noise_sep] * (self.max_len - len(final_noise))
        final_clean = final_clean[:self.max_len]
        final_noise = final_noise[:self.max_len]
        return final_clean, final_noise

    def __getitem__(self, idx):
        clean, noise = self.add_noise(self.data[idx])
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    if args.load_cp:
        model, optimizer = load_model(args.load_cp, model, optimizer)
        load_step = int(args.load_cp.split("_")[-1].replace(".cp", ""))
    print("load_step", load_step, flush=True)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    criterion = nn.CrossEntropyLoss().to(device)
    config_name = f"long_marix_{type_}{DUP}"

    for epoch in range(args.epochs):
        train_dataset = PhonemesDataset(type_=type_)
        train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
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
