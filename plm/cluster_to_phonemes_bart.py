# sbatch --gres=gpu:4,vmem:24g --mem=75G -c20 --time=7-0 --wrap "python cluster_to_phonemes_bart.py"
import random
from torch.utils.data import Dataset, DataLoader
from utils import args_parser, save_model_to_name, load_model
import numpy as np
import torch
from tqdm import tqdm
from mapping import phonemes_to_index
from torch.nn.parallel import DataParallel
import torch.nn as nn
from scipy.special import softmax
from scipy.spatial.distance import cdist
from transformers import BartConfig, BartForConditionalGeneration
from jiwer import wer

BATCH_SIZE =  32
LR = 1e-3
EPOCHS = 100
load_cp = ""
config_name = "learn_mapping_lr_bart"

ONE = 0
SPHERE = 2
MAX_LENGTH =  1024


PHONEMES_LAST_TOKEN = max(phonemes_to_index.values())
CLUSTERS_FIRST_TOKEN = PHONEMES_LAST_TOKEN + 1
N_CLUSTERS = 100
CLUSTERS_LAST_TOKEN = CLUSTERS_FIRST_TOKEN + N_CLUSTERS

PAD_TOKEN = CLUSTERS_LAST_TOKEN + 1
SEP = PAD_TOKEN + 1
START_TOKEN = SEP + 1
END_TOKEN = START_TOKEN + 1
N_TOKENS = END_TOKEN + 1


def random_gaussian(n, dim=2):
    point = np.random.normal(size=(n, dim))
    point /= np.linalg.norm(point, axis=1, keepdims=True)
    return point


class PhonemesDataset(Dataset):
    def __init__(self, phonemes_file, type_, dup, max_len=MAX_LENGTH, size=1_000_000):
        with open(phonemes_file, 'r') as f:
            phonemes_data = f.readlines()
        phonemes_data = [[int(x) for x in line.strip().split()] for line in phonemes_data]
        self.max_len = max_len
        self.type = type_
        self.dup = dup
        self.size = size

        self.data = []
        for _ in range(size):
            sample = [START_TOKEN]
            while len(sample) < max_len:
                sample += phonemes_data[np.random.randint(0, len(phonemes_data))]
                sample += [SEP]
            sample = sample[:max_len - 1] + [END_TOKEN]
            self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def build_mapping_one(self):
        units_mapping = list(range(PHONEMES_LAST_TOKEN + 1))
        units_mapping += [random.randint(0, PHONEMES_LAST_TOKEN) for _ in
                          range(N_CLUSTERS - (PHONEMES_LAST_TOKEN + 1))]
        random.shuffle(units_mapping)
        units_mapping = np.array(units_mapping)
        inv_mapping = {i: [] for i in range(PHONEMES_LAST_TOKEN + 1)}
        for i, u in enumerate(units_mapping):
            inv_mapping[u].append(i)
        return inv_mapping

    def build_mapping_sphere(self):
        phonemes = random_gaussian(PHONEMES_LAST_TOKEN + 1)
        clusters = random_gaussian(N_CLUSTERS)
        cosine_distances = 100 * (1 - cdist(phonemes, clusters, metric='cosine'))
        probabilities = softmax(cosine_distances, axis=0)
        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
        np.random.shuffle(probabilities)
        return probabilities

    def build_mapping(self):
        if self.type == ONE:
            return self.build_mapping_one()
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
        if self.dup:
            length = random.choices(values, weights=weights, k=len(clean))

        else:
            length = [1] * len(clean)

        final_clean = []
        final_noise = []
        range_units = np.arange(N_CLUSTERS)
        for c in clean:
            if len(final_noise) > self.max_len - START_END_COUNT:
                break

            if c in [START_TOKEN, END_TOKEN, PAD_TOKEN, SEP]:
                final_clean.append(c)
                final_noise.append(c)
            else:
                final_clean.append(c)
                for _ in range(length.pop()):
                    if self.type == ONE:
                        new_token = random.choice(inv_mapping[c])
                    else:  # self.type == SPHERE:
                        new_token = np.random.choice(range_units, p=inv_mapping[c])

                    new_token = CLUSTERS_FIRST_TOKEN + new_token
                    final_noise.append(new_token)

        if len(final_clean) < MAX_LENGTH:
            final_clean += [PAD_TOKEN] * (MAX_LENGTH-len(final_clean))
        if len(final_noise) < MAX_LENGTH:
            final_noise += [PAD_TOKEN] * (MAX_LENGTH-len(final_noise))

        if len(final_noise) > MAX_LENGTH:
            final_noise = final_noise[:MAX_LENGTH-1]+[END_TOKEN]
        if len(final_clean) > MAX_LENGTH:
            final_clean = final_clean[:MAX_LENGTH-1]+[END_TOKEN]


        return final_clean, final_noise

    def __getitem__(self, idx):
        clean, noise = self.add_noise(self.data[idx])
        clean = torch.LongTensor(clean)
        noise = torch.LongTensor(noise)
        return clean, noise


class PhonemesDatasetSubset(PhonemesDataset):
    def __len__(self):
        return 250

    def __getitem__(self, _):
        idx = np.random.randint(0, self.size - 1)
        return super().__getitem__(idx)


def step_config(cur_type, cur_dup, score):
    if score > 0.6:
        if cur_type == ONE:
            print("Change to sphere", flush=True)
            return SPHERE, cur_dup
        if cur_type == SPHERE and not cur_dup:
            print("Change to dup", flush=True)
            return SPHERE, True
    return cur_type, cur_dup


def get_model() -> BartForConditionalGeneration:
    d_model = 768
    nhead = 12
    num_layers = 12

    config = BartConfig(vocab_size=N_TOKENS + 1, max_position_embeddings=MAX_LENGTH, encoder_layers=num_layers,
                        encoder_ffn_dim=d_model,
                        encoder_attention_heads=nhead, decoder_layers=num_layers, decoder_ffn_dim=d_model,
                        decoder_attention_heads=nhead,
                        d_model=d_model, pad_token_id=PAD_TOKEN, bos_token_id=START_TOKEN, eos_token_id=END_TOKEN,
                        decoder_start_token_id=START_TOKEN, forced_eos_token_id=END_TOKEN)  # Set vocab size
    model = BartForConditionalGeneration(config)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'{params:,} trainable parameters')
    return model


if __name__ == '__main__':
    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    load_step = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    if load_cp:
        model, optimizer = load_model(load_cp, model, optimizer)
        load_step = int(load_cp.split("_")[-1].replace(".cp", ""))
    print("load_step", load_step, flush=True)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    criterion = nn.CrossEntropyLoss().to(device)

    curr_type = ONE
    curr_dup = False
    curr_acc = 0
    losses = []
    for epoch in range(EPOCHS):
        curr_type, curr_dup = step_config(curr_type, curr_dup, curr_acc)
        train_dataset = PhonemesDataset(phonemes_file="data/lr_train.txt", type_=curr_type, dup=curr_dup)
        train_subset = PhonemesDatasetSubset(phonemes_file="data/lr_train.txt", type_=curr_type, dup=curr_dup)
        train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        train_subset_data = DataLoader(train_subset, batch_size=1, shuffle=False, drop_last=True)
        model.train()
        for i, (x, y) in tqdm(enumerate(train_data), total=len(train_data)):
            x = x.to(device)
            y = y.to(device)
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

            if i % 250 == 0:
                with open(f"results/{config_name}.txt", 'a') as f:
                    f.write(f"step {i} loss {np.mean(losses)}")
                losses = []
            if i % 10000 == 0:
                model.eval()
                with torch.no_grad():
                    wer_scores = []
                    for x, y in tqdm(train_subset_data):
                        x = x.to(device)
                        y = y[0]
                        denoiser_output = model.generate(x, max_new_tokens=MAX_LENGTH,
                                                         min_new_tokens=MAX_LENGTH * 0.5, top_k=4,
                                                         num_beams=100).cpu().numpy()[0]
                        pred = " ".join([str(x) for x in pred if x != PAD_TOKEN])
                        y = " ".join([str(x) for x in y.cpu().numpy() if x != PAD_TOKEN])
                        wer_scores.append(wer(y, pred))
                with open(f"results/{config_name}.txt", 'a') as f:
                    f.write(f"step {i} wer {np.mean(wer_scores)}")

                model.train()

                n = len(train_dataset) * epoch + i + load_step
                save_model_to_name(model, optimizer, f"models/{config_name}_{n}.cp")
